# _gap_matvec.pyx
#
# Cython extension: matrix-free gap-Gram matvec  M v = G1(G1^T v) - G2(G2^T v)
# using direct FFTW3 C API with batched, multi-threaded plans.
#
# The gapped least-squares correction applies M^{-1} (via CG) and estimates
# log|M| (via stochastic Lanczos quadrature).  Both are dominated by repeated
# applications of M to a block of right-hand sides.  This module performs that
# block matvec entirely in C: FFTW batched r2c/c2r plans (created once per
# batch width and reused), aligned fftw_malloc buffers, and C loops for the
# scatter / complex-multiply / gather glue.  FFTW's own threads parallelise the
# batch; plans are tuned with FFTW_MEASURE and cached via FFTW wisdom.
#
# Bit-for-bit this differs from the numpy/pocketfft path only at FFT round-off
# (~1e-14 relative), like any FFT-backend change.
#
# This file is part of Hector 3.1.
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.string cimport memcpy, memset
import numpy as np
cimport numpy as np

# ── FFTW3 C API (batched/advanced interface + threads) ───────────────────────
cdef extern from "fftw3.h" nogil:
    ctypedef double fftw_complex[2]
    ctypedef void*  fftw_plan

    unsigned FFTW_MEASURE
    unsigned FFTW_ESTIMATE

    fftw_plan fftw_plan_many_dft_r2c(int rank, const int* n, int howmany,
                                     double* i, const int* inembed,
                                     int istride, int idist,
                                     fftw_complex* o, const int* onembed,
                                     int ostride, int odist, unsigned flags)
    fftw_plan fftw_plan_many_dft_c2r(int rank, const int* n, int howmany,
                                     fftw_complex* i, const int* inembed,
                                     int istride, int idist,
                                     double* o, const int* onembed,
                                     int ostride, int odist, unsigned flags)
    void fftw_execute_dft_r2c(fftw_plan p, double* i, fftw_complex* o)
    void fftw_execute_dft_c2r(fftw_plan p, fftw_complex* i, double* o)
    void fftw_destroy_plan(fftw_plan p)
    void* fftw_malloc(size_t n)
    void  fftw_free(void* p)
    int   fftw_import_wisdom_from_filename(const char* filename)
    void  fftw_export_wisdom_to_filename(const char* filename)
    int   fftw_init_threads()
    void  fftw_plan_with_nthreads(int nthreads)

# Largest supported batch width (LOGDET_PROBES=8; n_reg = polynomial +
# 2*periodic + offsets, occasionally large).  Callers must fall back for
# nb > MAXNB.  Exposed to Python as GapMatvec.MAXNB.
DEF MAXNB = 32

# Python-visible copy of MAXNB so callers can fall back for wider batches.
GAP_MAXNB = 32

# One-time global FFTW thread initialisation.
cdef bint _threads_started = False


cdef class GapMatvec:
    """Batched, multi-threaded FFTW gap-Gram matvec.

    Create once per (series length N, gap pattern, whitening filters); reuse
    `matvec` for every CG iteration and log-det probe.  Plans for each batch
    width are built lazily on first use and cached.
    """
    cdef int N, m, k, nc, nthreads
    cdef int*          gidx        # gap indices (k)
    cdef fftw_complex* Fl1         # rfft of whitening filter 1 (nc)
    cdef fftw_complex* Fl2         # rfft of whitening filter 2 (nc)
    cdef fftw_complex* Fl1c        # conj(Fl1) (nc)
    cdef fftw_complex* Fl2c        # conj(Fl2) (nc)
    # work buffers sized (MAXNB, N) real and (MAXNB, nc) complex, row-contiguous
    cdef double*       Wr          # W / U1 / A1
    cdef double*       Wr2         # U2 / A2
    cdef fftw_complex* Fc          # Fw / FU1 / Fu1 / FA1
    cdef fftw_complex* Fc2         # FU2 / Fu2 / FA2
    # plan cache indexed by batch width nb (1..MAXNB)
    cdef fftw_plan     rplan[MAXNB + 1]
    cdef fftw_plan     cplan[MAXNB + 1]
    cdef bytes         _wp

    def __cinit__(self, int N, int m, gap_idx, Fl1, Fl2,
                  int nthreads=4, wisdom_path=None):
        global _threads_started
        cdef int i
        cdef long[::1]              g_v
        cdef double complex[::1]    f1_v, f2_v

        self.N = N
        self.m = m
        self.nc = N // 2 + 1
        self.nthreads = nthreads
        self.gidx = NULL
        self.Fl1 = self.Fl2 = self.Fl1c = self.Fl2c = NULL
        self.Wr = self.Wr2 = NULL
        self.Fc = self.Fc2 = NULL
        for i in range(MAXNB + 1):
            self.rplan[i] = NULL
            self.cplan[i] = NULL

        if not _threads_started:
            fftw_init_threads()
            _threads_started = True

        self._wp = None
        if wisdom_path is not None:
            self._wp = wisdom_path.encode() if isinstance(wisdom_path, str) else wisdom_path
            fftw_import_wisdom_from_filename(self._wp)

        # gap indices
        g_np = np.ascontiguousarray(gap_idx, dtype=np.int64)
        self.k = g_np.shape[0]
        g_v = g_np
        self.gidx = <int*> fftw_malloc(self.k * sizeof(int))
        for i in range(self.k):
            self.gidx[i] = <int> g_v[i]

        # whitening-filter FFTs (complex128 layout == fftw_complex)
        f1_np = np.ascontiguousarray(Fl1, dtype=np.complex128)
        f2_np = np.ascontiguousarray(Fl2, dtype=np.complex128)
        self.Fl1  = <fftw_complex*> fftw_malloc(self.nc * sizeof(fftw_complex))
        self.Fl2  = <fftw_complex*> fftw_malloc(self.nc * sizeof(fftw_complex))
        self.Fl1c = <fftw_complex*> fftw_malloc(self.nc * sizeof(fftw_complex))
        self.Fl2c = <fftw_complex*> fftw_malloc(self.nc * sizeof(fftw_complex))
        f1_v = f1_np; f2_v = f2_np
        memcpy(self.Fl1, &f1_v[0], self.nc * sizeof(fftw_complex))
        memcpy(self.Fl2, &f2_v[0], self.nc * sizeof(fftw_complex))
        # precompute conjugates (avoids imaginary literals in the nogil loop)
        cdef double complex* _f1  = <double complex*> self.Fl1
        cdef double complex* _f2  = <double complex*> self.Fl2
        cdef double complex* _f1c = <double complex*> self.Fl1c
        cdef double complex* _f2c = <double complex*> self.Fl2c
        for i in range(self.nc):
            _f1c[i] = _f1[i].real - 1j * _f1[i].imag
            _f2c[i] = _f2[i].real - 1j * _f2[i].imag

        # aligned work buffers for the widest batch
        self.Wr  = <double*>       fftw_malloc(MAXNB * self.N  * sizeof(double))
        self.Wr2 = <double*>       fftw_malloc(MAXNB * self.N  * sizeof(double))
        self.Fc  = <fftw_complex*> fftw_malloc(MAXNB * self.nc * sizeof(fftw_complex))
        self.Fc2 = <fftw_complex*> fftw_malloc(MAXNB * self.nc * sizeof(fftw_complex))

    cdef void _ensure_plans(self, int nb):
        """Build (and cache) batched r2c/c2r plans for batch width nb."""
        cdef int n_arr[1]
        if self.rplan[nb] != NULL:
            return
        n_arr[0] = self.N
        fftw_plan_with_nthreads(self.nthreads)
        # row-contiguous batch: howmany=nb, stride=1, real dist=N, complex dist=nc
        self.rplan[nb] = fftw_plan_many_dft_r2c(
            1, n_arr, nb, self.Wr, NULL, 1, self.N,
            self.Fc, NULL, 1, self.nc, FFTW_MEASURE)
        self.cplan[nb] = fftw_plan_many_dft_c2r(
            1, n_arr, nb, self.Fc, NULL, 1, self.nc,
            self.Wr, NULL, 1, self.N, FFTW_MEASURE)
        if self._wp is not None:
            fftw_export_wisdom_to_filename(self._wp)

    def set_filters(self, Fl1, Fl2):
        """Refresh the whitening-filter FFTs (they change every MLE eval; the
        FFTW plans depend only on N and are reused)."""
        cdef int i
        cdef double complex[::1] f1_v, f2_v
        f1_np = np.ascontiguousarray(Fl1, dtype=np.complex128)
        f2_np = np.ascontiguousarray(Fl2, dtype=np.complex128)
        f1_v = f1_np; f2_v = f2_np
        memcpy(self.Fl1, &f1_v[0], self.nc * sizeof(fftw_complex))
        memcpy(self.Fl2, &f2_v[0], self.nc * sizeof(fftw_complex))
        cdef double complex* _f1  = <double complex*> self.Fl1
        cdef double complex* _f2  = <double complex*> self.Fl2
        cdef double complex* _f1c = <double complex*> self.Fl1c
        cdef double complex* _f2c = <double complex*> self.Fl2c
        for i in range(self.nc):
            _f1c[i] = _f1[i].real - 1j * _f1[i].imag
            _f2c[i] = _f2[i].real - 1j * _f2[i].imag

    def matvec(self, V):
        """Return M @ V.  V is (k,) or (k, nb); result matches shape."""
        cdef bint oned = (V.ndim == 1)
        Vin = np.ascontiguousarray(
            (V[:, None] if oned else V), dtype=np.float64)
        cdef int nb = Vin.shape[1]
        if nb > MAXNB:
            raise ValueError(f"batch width {nb} exceeds MAXNB={MAXNB}")
        self._ensure_plans(nb)

        cdef int N = self.N, m = self.m, k = self.k, nc = self.nc
        cdef int col, j, f, base_r, base_c
        cdef double inv_n = 1.0 / N
        cdef double complex w
        cdef double* Wr  = self.Wr
        cdef double* Wr2 = self.Wr2
        cdef fftw_complex* Fc  = self.Fc
        cdef fftw_complex* Fc2 = self.Fc2
        cdef double complex* Fc_c   = <double complex*> self.Fc
        cdef double complex* Fc2_c  = <double complex*> self.Fc2
        cdef double complex* Fl1_c  = <double complex*> self.Fl1
        cdef double complex* Fl2_c  = <double complex*> self.Fl2
        cdef double complex* Fl1c_c = <double complex*> self.Fl1c
        cdef double complex* Fl2c_c = <double complex*> self.Fl2c
        cdef int* gidx = self.gidx
        cdef double[:, ::1] Vv = Vin
        out_np = np.empty((k, nb), dtype=np.float64)
        cdef double[:, ::1] Ov = out_np

        with nogil:
            # 1. scatter V into W (zero elsewhere)
            memset(Wr, 0, nb * N * sizeof(double))
            for col in range(nb):
                base_r = col * N
                for j in range(k):
                    Wr[base_r + gidx[j]] = Vv[j, col]

            # 2. Fw = rfft(W)  -> Fc
            fftw_execute_dft_r2c(self.rplan[nb], Wr, Fc)

            # 3. FU2 = Fl2 * Fw -> Fc2 ;  FU1 = Fl1 * Fw -> Fc (overwrite)
            for col in range(nb):
                base_c = col * nc
                for f in range(nc):
                    w = Fc_c[base_c + f]
                    Fc2_c[base_c + f] = Fl2_c[f] * w
                    Fc_c[base_c + f]  = Fl1_c[f] * w

            # 4. U1 = irfft(FU1)/N -> Wr ;  U2 = irfft(FU2)/N -> Wr2
            fftw_execute_dft_c2r(self.cplan[nb], Fc,  Wr)
            fftw_execute_dft_c2r(self.cplan[nb], Fc2, Wr2)
            # normalise + zero the tail [m:N] for the zero-padded forward FFT
            for col in range(nb):
                base_r = col * N
                for j in range(m):
                    Wr[base_r + j]  *= inv_n
                    Wr2[base_r + j] *= inv_n
                memset(&Wr[base_r + m],  0, (N - m) * sizeof(double))
                memset(&Wr2[base_r + m], 0, (N - m) * sizeof(double))

            # 5. Fu1 = rfft(U1) -> Fc ;  Fu2 = rfft(U2) -> Fc2
            fftw_execute_dft_r2c(self.rplan[nb], Wr,  Fc)
            fftw_execute_dft_r2c(self.rplan[nb], Wr2, Fc2)

            # 6. FA1 = conj(Fl1) * Fu1 -> Fc ;  FA2 = conj(Fl2) * Fu2 -> Fc2
            for col in range(nb):
                base_c = col * nc
                for f in range(nc):
                    Fc_c[base_c + f]  = Fl1c_c[f] * Fc_c[base_c + f]
                    Fc2_c[base_c + f] = Fl2c_c[f] * Fc2_c[base_c + f]

            # 7. A1 = irfft(FA1)/N -> Wr ;  A2 = irfft(FA2)/N -> Wr2
            fftw_execute_dft_c2r(self.cplan[nb], Fc,  Wr)
            fftw_execute_dft_c2r(self.cplan[nb], Fc2, Wr2)

            # 8. gather at gap indices:  out = (A1 - A2)/N
            for col in range(nb):
                base_r = col * N
                for j in range(k):
                    Ov[j, col] = (Wr[base_r + gidx[j]] - Wr2[base_r + gidx[j]]) * inv_n

        return out_np[:, 0] if oned else out_np

    def __dealloc__(self):
        cdef int i
        for i in range(MAXNB + 1):
            if self.rplan[i] != NULL: fftw_destroy_plan(self.rplan[i])
            if self.cplan[i] != NULL: fftw_destroy_plan(self.cplan[i])
        if self.gidx != NULL: fftw_free(self.gidx)
        if self.Fl1  != NULL: fftw_free(self.Fl1)
        if self.Fl2  != NULL: fftw_free(self.Fl2)
        if self.Fl1c != NULL: fftw_free(self.Fl1c)
        if self.Fl2c != NULL: fftw_free(self.Fl2c)
        if self.Wr   != NULL: fftw_free(self.Wr)
        if self.Wr2  != NULL: fftw_free(self.Wr2)
        if self.Fc   != NULL: fftw_free(self.Fc)
        if self.Fc2  != NULL: fftw_free(self.Fc2)
