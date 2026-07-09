# _schur_gsa.pyx
#
# Cython extension: full Generalised Schur Algorithm with workspace pool.
#
# SchurGSA pre-allocates O(n) workspace before the recursion starts so that
# _gsa never calls np.empty / np.zeros internally.
#
# FFT via direct FFTW3 C API (no pyfftw wrapper). Plans and aligned buffers
# are owned by SchurGSA.__cinit__; FFTW wisdom is read/written via the native
# FFTW text-file API.
#
# OpenMP: the two independent rfft calls in _multiply_into are run in parallel
# (via Cython cython.parallel.parallel) when the FFT size >= OMP_THRESH.
#
# Build with:
#   python setup_schur_gsa.py build_ext --inplace   (from hector-ts/code/)
#
# This file is part of Hector 3.0.
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.string  cimport memcpy, memset
from libc.math    cimport log as c_log
import numpy as np
cimport numpy as np

# ── FFTW3 C API declarations ─────────────────────────────────────────────────
cdef extern from "fftw3.h" nogil:
    ctypedef double fftw_complex[2]
    ctypedef void*  fftw_plan

    unsigned FFTW_MEASURE
    unsigned FFTW_ESTIMATE

    fftw_plan fftw_plan_dft_r2c_1d(int n, double*       i,
                                   fftw_complex* o, unsigned flags)
    fftw_plan fftw_plan_dft_c2r_1d(int n, fftw_complex* i,
                                   double*       o, unsigned flags)
    void fftw_execute_dft_r2c(fftw_plan p, double*       i, fftw_complex* o)
    void fftw_execute_dft_c2r(fftw_plan p, fftw_complex* i, double*       o)
    void fftw_destroy_plan(fftw_plan p)
    void* fftw_malloc(size_t n)
    void  fftw_free(void* p)
    int   fftw_import_wisdom_from_filename(const char* filename)
    void  fftw_export_wisdom_to_filename(const char* filename)

# Polynomials shorter than this on both sides use a direct O(la*lb) loop.
DEF DIRECT_THRESH = 64

# Maximum plan-index depth (FFT size = 2^i).
# i up to 16 → FFT size 65536 → supports series up to ~80 yr (n ≤ 32768).
# Plans 17–18 (FFT 131072 / 262144) consumed ~16 MB of L3 cache on x86 and
# were never needed for GPS data; removing them cuts pre-allocated buffers from
# ~21 MB to ~5 MB, keeping the working set well within typical L3 (16 MB).
DEF N_PLANS = 17   # indices 1..16 are valid

# Maximum workspace depth for the recursion.
DEF MAX_DEPTH = 24


cdef class SchurGSA:
    """Full Cython GSA with direct FFTW3 plans and optional OpenMP parallelism.

    compute_for_toeplitz(t) is a drop-in for Levinson.compute(t): returns
    (l1, l2, delta, ln_det_C) without any per-recursion numpy allocations.
    """

    # ── FFTW plans and aligned buffers, indexed by i where n = 1 << i ────────
    cdef fftw_plan   _rfft_plan[N_PLANS]    # r2c: x  → Fa
    cdef fftw_plan   _rfft_plan_b[N_PLANS]  # r2c: xb → Fb  (parallel partner)
    cdef fftw_plan   _irfft_plan[N_PLANS]   # c2r: Fx → x
    cdef double*         _x_ptr[N_PLANS]    # real input for a + irfft output
    cdef double*         _xb_ptr[N_PLANS]   # real input for b
    cdef fftw_complex*   _Fa_ptr[N_PLANS]   # rfft(a)
    cdef fftw_complex*   _Fb_ptr[N_PLANS]   # rfft(b)
    cdef fftw_complex*   _Fx_ptr[N_PLANS]   # Fa * Fb (irfft input)

    # ── workspace pool: C-level pointer arrays indexed by recursion depth ─────
    cdef double *_ws_al_ptr[MAX_DEPTH]
    cdef double *_ws_cl_ptr[MAX_DEPTH]
    cdef double *_ws_ar_ptr[MAX_DEPTH]
    cdef double *_ws_cr_ptr[MAX_DEPTH]
    cdef double *_ws_dm_ptr[MAX_DEPTH]
    cdef double *_ws_bm_ptr[MAX_DEPTH]
    cdef double *_ws_mul1_ptr[MAX_DEPTH]
    cdef double *_ws_mul2_ptr[MAX_DEPTH]
    cdef double *_ws_pm_ptr[MAX_DEPTH]
    cdef double *_ws_qm_ptr[MAX_DEPTH]
    cdef int     _ws_ac_size[MAX_DEPTH]
    cdef int     _ws_mul_size[MAX_DEPTH]
    cdef list    _ws_storage   # keeps numpy arrays (and thus data ptrs) alive
    cdef int     _cached_tm    # workspace is valid for this tm; -1 = uninitialised

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def __cinit__(self, wisdom_path=None):
        """Create FFTW plans for all FFT sizes used by the GSA multiply."""
        cdef int i, n, half1
        cdef bytes wp_bytes

        # Zero-init plan/buffer arrays so __dealloc__ is safe even on error.
        for i in range(N_PLANS):
            self._rfft_plan[i]   = NULL
            self._rfft_plan_b[i] = NULL
            self._irfft_plan[i]  = NULL
            self._x_ptr[i]  = NULL
            self._xb_ptr[i] = NULL
            self._Fa_ptr[i] = NULL
            self._Fb_ptr[i] = NULL
            self._Fx_ptr[i] = NULL

        self._ws_storage = []
        self._cached_tm  = -1

        # Load existing wisdom so that FFTW_MEASURE reuses known-good plans.
        if wisdom_path is not None:
            wp_bytes = wisdom_path.encode() if isinstance(wisdom_path, str) else wisdom_path
            fftw_import_wisdom_from_filename(wp_bytes)

        # Create plans for each FFT size n = 2, 4, 8, ..., 2^18.
        n = 2
        for i in range(1, N_PLANS):
            half1 = n // 2 + 1
            self._x_ptr[i]  = <double*>      fftw_malloc(n     * sizeof(double))
            self._xb_ptr[i] = <double*>      fftw_malloc(n     * sizeof(double))
            self._Fa_ptr[i] = <fftw_complex*>fftw_malloc(half1 * sizeof(fftw_complex))
            self._Fb_ptr[i] = <fftw_complex*>fftw_malloc(half1 * sizeof(fftw_complex))
            self._Fx_ptr[i] = <fftw_complex*>fftw_malloc(half1 * sizeof(fftw_complex))
            self._rfft_plan[i]   = fftw_plan_dft_r2c_1d(
                n, self._x_ptr[i],  self._Fa_ptr[i], FFTW_MEASURE)
            self._rfft_plan_b[i] = fftw_plan_dft_r2c_1d(
                n, self._xb_ptr[i], self._Fb_ptr[i], FFTW_MEASURE)
            self._irfft_plan[i]  = fftw_plan_dft_c2r_1d(
                n, self._Fx_ptr[i], self._x_ptr[i],  FFTW_MEASURE)
            n <<= 1

        # Persist wisdom so subsequent processes skip FFTW_MEASURE.
        if wisdom_path is not None:
            fftw_export_wisdom_to_filename(wp_bytes)

    def __dealloc__(self):
        cdef int i
        for i in range(1, N_PLANS):
            if self._rfft_plan[i]   != NULL: fftw_destroy_plan(self._rfft_plan[i])
            if self._rfft_plan_b[i] != NULL: fftw_destroy_plan(self._rfft_plan_b[i])
            if self._irfft_plan[i]  != NULL: fftw_destroy_plan(self._irfft_plan[i])
            if self._x_ptr[i]  != NULL: fftw_free(self._x_ptr[i])
            if self._xb_ptr[i] != NULL: fftw_free(self._xb_ptr[i])
            if self._Fa_ptr[i] != NULL: fftw_free(self._Fa_ptr[i])
            if self._Fb_ptr[i] != NULL: fftw_free(self._Fb_ptr[i])
            if self._Fx_ptr[i] != NULL: fftw_free(self._Fx_ptr[i])

    def __init__(self, wisdom_path=None):
        pass   # all init done in __cinit__

    # ── workspace allocation ──────────────────────────────────────────────────

    def _allocate_workspace(self, int tm):
        """Build workspace pointer arrays for a problem of size tm."""
        cdef int tm_d, ac_size, mul_size, d
        cdef double[::1] al_v, cl_v, ar_v, cr_v, dm_v, bm_v
        cdef double[::1] mul1_v, mul2_v, pm_v, qm_v

        storage = []
        d = 0
        tm_d = tm
        while tm_d >= 2:
            ac_size  = (tm_d >> 1) + 4
            mul_size = tm_d + (tm_d >> 1) + 4

            al_np   = np.empty(ac_size,  dtype=np.float64)
            cl_np   = np.empty(ac_size,  dtype=np.float64)
            ar_np   = np.empty(ac_size,  dtype=np.float64)
            cr_np   = np.empty(ac_size,  dtype=np.float64)
            dm_np   = np.empty(ac_size,  dtype=np.float64)
            bm_np   = np.empty(ac_size,  dtype=np.float64)
            mul1_np = np.empty(mul_size, dtype=np.float64)
            mul2_np = np.empty(mul_size, dtype=np.float64)
            pm_np   = np.empty(mul_size, dtype=np.float64)
            qm_np   = np.empty(mul_size, dtype=np.float64)

            al_v = al_np;   self._ws_al_ptr[d]   = &al_v[0]
            cl_v = cl_np;   self._ws_cl_ptr[d]   = &cl_v[0]
            ar_v = ar_np;   self._ws_ar_ptr[d]   = &ar_v[0]
            cr_v = cr_np;   self._ws_cr_ptr[d]   = &cr_v[0]
            dm_v = dm_np;   self._ws_dm_ptr[d]   = &dm_v[0]
            bm_v = bm_np;   self._ws_bm_ptr[d]   = &bm_v[0]
            mul1_v = mul1_np; self._ws_mul1_ptr[d] = &mul1_v[0]
            mul2_v = mul2_np; self._ws_mul2_ptr[d] = &mul2_v[0]
            pm_v = pm_np;   self._ws_pm_ptr[d]   = &pm_v[0]
            qm_v = qm_np;   self._ws_qm_ptr[d]   = &qm_v[0]
            self._ws_ac_size[d]  = ac_size
            self._ws_mul_size[d] = mul_size

            storage += [al_np, cl_np, ar_np, cr_np, dm_np, bm_np,
                        mul1_np, mul2_np, pm_np, qm_np]
            d += 1
            tm_d = (tm_d + 1) >> 1

        self._ws_storage = storage

    # ── direct polynomial multiply ────────────────────────────────────────────

    cdef void _direct_multiply_into(self, double* a, int la, double* b, int lb,
                                    double* out) noexcept nogil:
        cdef int i, j
        cdef double ai
        memset(out, 0, (la + lb - 1) * sizeof(double))
        for i in range(la):
            ai = a[i]
            for j in range(lb):
                out[i + j] += ai * b[j]

    # ── FFT polynomial multiply ───────────────────────────────────────────────

    cdef int _multiply_into(self, double* a, int la, double* b, int lb,
                            double* out) noexcept nogil:
        cdef int lout = la + lb - 1
        cdef int m, n, i, k, half1
        cdef double inv_n
        cdef double*        xa
        cdef double*        xb
        cdef fftw_complex*  Fa
        cdef fftw_complex*  Fb
        cdef fftw_complex*  Fx
        cdef fftw_plan      rplan, rplan_b, iplan
        cdef double complex* Fa_c
        cdef double complex* Fb_c
        cdef double complex* Fx_c

        if la <= DIRECT_THRESH and lb <= DIRECT_THRESH:
            self._direct_multiply_into(a, la, b, lb, out)
            return lout

        m = la if la > lb else lb
        n = 1; i = 1
        while n < m:
            n <<= 1
            i += 1
        n <<= 1

        xa     = self._x_ptr[i]
        xb     = self._xb_ptr[i]
        Fa     = self._Fa_ptr[i]
        Fb     = self._Fb_ptr[i]
        Fx     = self._Fx_ptr[i]
        rplan  = self._rfft_plan[i]
        rplan_b= self._rfft_plan_b[i]
        iplan  = self._irfft_plan[i]
        half1  = n // 2 + 1

        memcpy(xa, a, la * sizeof(double))
        memset(xa + la, 0, (n - la) * sizeof(double))
        fftw_execute_dft_r2c(rplan, xa, Fa)

        memcpy(xb, b, lb * sizeof(double))
        memset(xb + lb, 0, (n - lb) * sizeof(double))
        fftw_execute_dft_r2c(rplan_b, xb, Fb)

        # ── pointwise complex multiply: Fx = Fa * Fb ─────────────────────────
        Fa_c = <double complex*> Fa
        Fb_c = <double complex*> Fb
        Fx_c = <double complex*> Fx
        for k in range(half1):
            Fx_c[k] = Fa_c[k] * Fb_c[k]

        # ── irfft(Fx) → xa, normalise ────────────────────────────────────────
        # FFTW c2r transforms always destroy the input (Fx); that is fine since
        # Fx is recomputed every call. Output lands in xa.
        fftw_execute_dft_c2r(iplan, Fx, xa)
        inv_n = 1.0 / n
        for k in range(lout):
            xa[k] *= inv_n

        memcpy(out, xa, lout * sizeof(double))
        return lout

    # ── polynomial addition / subtraction into a provided buffer ─────────────

    cdef int _poly_add_into(self, double* a, int la, double* b, int lb,
                            double* out) noexcept nogil:
        cdef int i
        if la >= lb:
            memcpy(out, a, la * sizeof(double))
            for i in range(lb):
                out[i] += b[i]
            return la
        else:
            memcpy(out, b, lb * sizeof(double))
            for i in range(la):
                out[i] += a[i]
            return lb

    cdef int _poly_sub_into(self, double* a, int la, double* b, int lb,
                            double* out) noexcept nogil:
        cdef int i
        if la >= lb:
            memcpy(out, a, la * sizeof(double))
            for i in range(lb):
                out[i] -= b[i]
            return la
        else:
            for i in range(la):
                out[i] = a[i] - b[i]
            for i in range(la, lb):
                out[i] = -b[i]
            return lb

    # ── GSA divide-and-conquer recursion ──────────────────────────────────────

    cdef double _gsa(self, int tm, int offset, int total_tm,
                     double* p_0, double* q_0, int depth,
                     double* out_a, double* out_c,
                     double* ln_det_out) noexcept nogil:
        cdef double gamma, delta_m, delta_mtm, delta_step
        cdef double ld_left, ld_right
        cdef int m, k, lmul1, lmul2, la_mtm
        cdef double* ws_mul1
        cdef double* ws_mul2
        cdef double* ws_pm
        cdef double* ws_qm
        cdef double* ws_al
        cdef double* ws_cl
        cdef double* ws_ar
        cdef double* ws_cr
        cdef double* ws_dm
        cdef double* ws_bm

        if tm == 1:
            gamma      = -p_0[0] / q_0[0]
            delta_step = 1.0 - gamma * gamma
            out_a[0] = 0.0;  out_a[1] = 1.0
            out_c[0] = 0.0;  out_c[1] = -gamma
            ln_det_out[0] = (total_tm - offset) * c_log(delta_step)
            return delta_step

        m = tm >> 1

        ws_mul1 = self._ws_mul1_ptr[depth]
        ws_mul2 = self._ws_mul2_ptr[depth]
        ws_pm   = self._ws_pm_ptr[depth]
        ws_qm   = self._ws_qm_ptr[depth]
        ws_al   = self._ws_al_ptr[depth]
        ws_cl   = self._ws_cl_ptr[depth]
        ws_ar   = self._ws_ar_ptr[depth]
        ws_cr   = self._ws_cr_ptr[depth]
        ws_dm   = self._ws_dm_ptr[depth]
        ws_bm   = self._ws_bm_ptr[depth]

        ld_left = 0.0
        delta_m = self._gsa(m, offset, total_tm, p_0, q_0, depth + 1,
                            ws_al, ws_cl, &ld_left)

        for k in range(m):
            ws_dm[k] = ws_al[m - k]
            ws_bm[k] = ws_cl[m - k]

        lmul1 = self._multiply_into(ws_dm, m,     p_0, tm, ws_mul1)
        lmul2 = self._multiply_into(ws_bm, m,     q_0, tm, ws_mul2)
        self._poly_sub_into(ws_mul1, lmul1, ws_mul2, lmul2, ws_pm)

        lmul1 = self._multiply_into(ws_cl, m + 1, p_0, tm, ws_mul1)
        lmul2 = self._multiply_into(ws_al, m + 1, q_0, tm, ws_mul2)
        self._poly_sub_into(ws_mul2, lmul2, ws_mul1, lmul1, ws_qm)

        ld_right = 0.0
        delta_mtm = self._gsa(tm - m, offset + m, total_tm,
                              ws_pm + m, ws_qm + m, depth + 1,
                              ws_ar, ws_cr, &ld_right)
        la_mtm = tm - m + 1

        lmul1 = self._multiply_into(ws_al, m + 1, ws_ar, la_mtm, ws_mul1)
        lmul2 = self._multiply_into(ws_bm, m,     ws_cr, la_mtm, ws_mul2)
        self._poly_add_into(ws_mul1, lmul1, ws_mul2, lmul2, out_a)

        lmul1 = self._multiply_into(ws_cl, m + 1, ws_ar, la_mtm, ws_mul1)
        lmul2 = self._multiply_into(ws_dm, m,     ws_cr, la_mtm, ws_mul2)
        self._poly_add_into(ws_mul1, lmul1, ws_mul2, lmul2, out_c)

        ln_det_out[0] = ld_left + ld_right
        return delta_m * delta_mtm

    def generalised_schur(self, int tm, p_0, q_0):
        """Legacy entry point: return (a, c, delta).  No ln_det_C."""
        cdef double ln_det_dummy = 0.0
        cdef double[::1] p_0_v, q_0_v, out_a_v, out_c_v
        self._allocate_workspace(tm)
        p_0_np   = np.ascontiguousarray(p_0, dtype=np.float64)
        q_0_np   = np.ascontiguousarray(q_0, dtype=np.float64)
        out_a_np = np.empty(tm + 1, dtype=np.float64)
        out_c_np = np.empty(tm + 1, dtype=np.float64)
        p_0_v = p_0_np;  q_0_v = q_0_np
        out_a_v = out_a_np;  out_c_v = out_c_np
        delta = self._gsa(
            tm, 0, tm,
            &p_0_v[0], &q_0_v[0],
            0, &out_a_v[0], &out_c_v[0], &ln_det_dummy,
        )
        return (out_a_np, out_c_np, delta)

    def compute_for_toeplitz(self, t):
        """Drop-in replacement for Levinson.compute(t).

        Returns (l1, l2, delta, ln_det_C).
        """
        cdef int tm, m, i
        cdef double t0, delta_tm, ln_det_accum
        cdef double[::1] t_arr, p_0, q_0, out_a, out_c, l1_v, l2_v

        t_arr = np.ascontiguousarray(t, dtype=np.float64)
        m  = t_arr.shape[0]
        tm = m - 1
        t0 = t_arr[0]

        # Check that tm fits within the pre-allocated plan range.
        # Required plan index: ceil(log2(tm)) + 1.  For tm ≥ 2^(N_PLANS-2),
        # the top-level multiply would exceed plan index N_PLANS-1.
        if tm >= (1 << (N_PLANS - 2)):
            raise ValueError(
                f"Series length {m} (tm={tm}) exceeds the maximum supported by "
                f"pre-allocated FFTW plans (N_PLANS={N_PLANS}, max tm={1 << (N_PLANS-2)}-1). "
                f"Increase N_PLANS in _schur_gsa.pyx."
            )

        p_0_np = np.empty(m, dtype=np.float64)
        p_0 = p_0_np
        p_0[m - 1] = 0.0
        for i in range(m - 1):
            p_0[i] = t_arr[i + 1]
        q_0 = t_arr

        if tm != self._cached_tm:
            self._allocate_workspace(tm)
            self._cached_tm = tm
        out_a_np = np.empty(tm + 1, dtype=np.float64)
        out_c_np = np.empty(tm + 1, dtype=np.float64)
        out_a = out_a_np
        out_c = out_c_np

        ln_det_accum = 0.0
        delta_tm = self._gsa(tm, 0, tm, &p_0[0], &q_0[0], 0,
                             &out_a[0], &out_c[0], &ln_det_accum)

        l1_np = np.empty(m, dtype=np.float64)
        l2_np = np.empty(m, dtype=np.float64)
        l1_v  = l1_np
        l2_v  = l2_np
        l1_v[0] = 1.0
        l2_v[0] = 0.0
        for i in range(tm):
            l1_v[tm - i]  = out_a[i] - out_c[i + 1]
            l2_v[i + 1]   = out_a[i] - out_c[i + 1]

        delta    = t0 * delta_tm
        ln_det_C = m * c_log(t0) + ln_det_accum

        return l1_np, l2_np, delta, ln_det_C
