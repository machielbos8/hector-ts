# ammargrag.py
#
# Python3 implemenation of AmmarGrag.cpp. It provides a quick subroutine to
# perform least-squares given the design matrix H, the observations y and
# the first column of the Toeplitz covariance matrix C.
#
# The Durbin-Levinson algorithm is based on Chapter 3 of "Iterative Methods
# for Toeplitz Systems", By Michael K. Ng  (page 28-29)
#
# Equations are taken from Bos et al. (2013), "Fast error analysis of
# continuous GNSS observations with missing data", Journal of Geodesy,
# DOI 10.1007/s00190-012-0605-0.
#
# Gap-correction optimisations (Approach F):
#   Gram:     spectral Toeplitz inverse  IFFT(1/ev)  — O(k²) lookup
#   Products: FFT cross-correlation  irfft(Fl1c * Fv)[gap_idx]  — O(m log m)
#   Qt:       algebraic shortcut  Qy − QA @ theta  — no extra FFTs
#
# This file is part of Hector 3.0.
#
# Hector is distributed under a source-available license.
# It may be used free of charge for academic, research, and other
# non-commercial purposes.
# Commercial use is not permitted under this license and requires a
# separate agreement with TeroMovigo - Earth Innovation Lda.
# The complete license terms are provided in the LICENSE file.
#
# 28/6/2026 Machiel Bos
#===============================================================================

import math
import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_triangular
import scipy.fft as sp_fft
from hector.control import SingletonMeta

from hector.levinson import Levinson
from hector.schur import Schur
from hector._fpe import quiet_matmul

try:
    from hector._epoch_scan_gaps import fast_epoch_scan_gaps as _epoch_scan_gaps_cython
    _USE_CYTHON_GAPS = True
except ImportError:
    _USE_CYTHON_GAPS = False

try:
    from hector._epoch_scan_nogap import fast_epoch_scan_nogap as _epoch_scan_nogap_cython
    _USE_CYTHON_NOGAP = True
except ImportError:
    _USE_CYTHON_NOGAP = False

# Optional FFTW-C batched gap matvec (direct FFTW3, threaded plans). Falls back
# to the pure-numpy/scipy _gap_matvec when the extension is unavailable.
from pathlib import Path as _Path
try:
    from hector._gap_matvec import GapMatvec as _GapMatvec, GAP_MAXNB as _GAP_MAXNB
    _HAVE_FFTW_MATVEC = True
except ImportError:
    _HAVE_FFTW_MATVEC = False
    _GAP_MAXNB = 0
_GAP_WISDOM = _Path.home() / '.cache' / 'hector' / 'fftw_wisdom_gap.dat'
GAP_MATVEC_THREADS = 4

# Series longer than this use the O(n log² n) GSA; shorter ones use O(n²) DL.
# Measured crossover on Apple Silicon: GSA faster above n ≈ 3 500.
GSA_THRESHOLD   = 1000
GRAM_THRESHOLD  =  500   # below this, build G1/G2 explicitly and use exact Cholesky

# Exact gap log-determinant (Chan case): log|M| = log|M_chan| + log|M_chan⁻¹M|,
# the correction estimated by stochastic Lanczos quadrature harvested from the
# Chan-preconditioned CG (Golub–Meurant).  A fixed probe seed keeps the MLE
# objective deterministic; the preconditioned operator clusters near 1, so a
# few probes and CG iterations suffice.
LOGDET_PROBES   =    8
LOGDET_MAXITER  =   30
LOGDET_SEED     =    0


def _chol_downdate_lower(L, x, start=0, check=True):
    """In-place rank-1 Cholesky downdate: L L.T  →  L L.T − x xᵀ.

    L    : (n, n) lower-triangular Cholesky factor, modified in place.
    x    : (n,) downdate vector.
    start: first index to process; x[:start] must be zero.
    check: raise LinAlgError if the downdated matrix ceases to be PD.
    """
    x = x.copy()
    n = L.shape[0]
    for k in range(start, n):
        r2 = L[k, k] ** 2 - x[k] ** 2
        if check and r2 <= 0.0:
            raise np.linalg.LinAlgError(
                "Cholesky downdate failed: matrix is not positive definite")
        r = math.sqrt(r2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k + 1 < n:
            L[k + 1:, k] = (L[k + 1:, k] - s * x[k + 1:]) / c
            x[k + 1:]    =  c * x[k + 1:] - s * L[k + 1:, k]


def _difference_cholesky_lower(L1, L2, check=True):
    """Compute L s.t. L L.T = L1 L1.T − L2 L2.T via sequential rank-1 downdates.

    L1, L2 must be (k, k) lower-triangular Cholesky factors.
    The start=j optimisation exploits the lower-triangular structure of L2:
    column j of L2 has zeros at rows 0..j-1, so each downdate costs O((k-j)²)
    instead of O(k²) — total O(k³/3).
    """
    L = L1.copy()
    for j in range(L1.shape[0]):
        _chol_downdate_lower(L, L2[:, j], start=j, check=check)
    return L

#===============================================================================
# Class definitions
#===============================================================================

class AmmarGrag(metaclass=SingletonMeta):

    def __init(self):
        """ Define class variables
        """
        self.z
        self.y1
        self.y2
        self.Fl1
        self.Fl2
        self.Mch
        self.Qy
        self.gap_idx
        self.ln_det_C


    def _gap_matvec(self, v):
        """Apply the gap Gram  M = G1 G1ᵀ − G2 G2ᵀ, matrix-free.

        Accepts a single vector ``(k,)`` or a block of right-hand sides
        ``(k, nrhs)``.  G1/G2 have rows l1s/l2s shifted to each gap epoch, so
        both G·(·) and Gᵀ·(·) are FFT convolutions/cross-correlations —
        O(m log m), no k×k matrix ever formed.  For a block, all columns are
        transformed in one call with the FFT running along axis 0 and threaded
        over the batch (``scipy.fft`` ``workers=-1``), which turns the CG inner
        product into BLAS-3-like throughput.  Used by the CG solvers for the
        exact gap correction.

        When the FFTW-C extension is available (``self._gm`` set by
        :meth:`_setup_gap_matvec`), the whole matvec runs in C with reused
        threaded FFTW plans; otherwise this pure-numpy path is used.
        """
        gm = getattr(self, '_gm', None)
        if gm is not None and (v.ndim == 1 or v.shape[1] <= _GAP_MAXNB):
            return gm.matvec(v)
        m    = self._m
        N    = self.N_fft
        v1d  = (v.ndim == 1)
        V    = v[:, None] if v1d else v
        nb   = V.shape[1]
        W            = np.zeros((N, nb))
        W[self.gap_idx, :] = V
        Fw  = sp_fft.rfft(W, axis=0, workers=-1)
        U1  = sp_fft.irfft(self.Fl1[:, None] * Fw, n=N, axis=0)[:m]
        U2  = sp_fft.irfft(self.Fl2[:, None] * Fw, n=N, axis=0)[:m]
        pad = np.zeros((N - m, nb))
        Fu1 = sp_fft.rfft(np.concatenate([U1, pad]), axis=0, workers=-1)
        Fu2 = sp_fft.rfft(np.concatenate([U2, pad]), axis=0, workers=-1)
        A1  = sp_fft.irfft(self.Fl1.conj()[:, None] * Fu1, n=N, axis=0)[:m][self.gap_idx]
        A2  = sp_fft.irfft(self.Fl2.conj()[:, None] * Fu2, n=N, axis=0)[:m][self.gap_idx]
        R   = A1 - A2
        return R[:, 0] if v1d else R


    def _setup_gap_matvec(self, N_fft, m, gap_idx):
        """Create or refresh the optional FFTW-C batched gap matvec.

        The FFTW plans depend only on (N_fft, gap pattern), so across MLE
        evaluations the object is reused and only the whitening filters are
        refreshed (they change every eval).  Set to None for the small-m exact
        path or when the extension is unavailable, so `_gap_matvec` falls back
        to the pure-numpy implementation.
        """
        if not _HAVE_FFTW_MATVEC or self._gram_exact:
            self._gm = None
            return
        gm = getattr(self, '_gm', None)
        if (gm is None
                or getattr(self, '_gm_N', -1) != N_fft
                or not np.array_equal(getattr(self, '_gm_gid', None), gap_idx)):
            _GAP_WISDOM.parent.mkdir(parents=True, exist_ok=True)
            self._gm = _GapMatvec(N_fft, m, gap_idx, self.Fl1, self.Fl2,
                                  nthreads=GAP_MATVEC_THREADS,
                                  wisdom_path=str(_GAP_WISDOM))
            self._gm_N   = N_fft
            self._gm_gid = gap_idx.copy()
        else:
            self._gm.set_filters(self.Fl1, self.Fl2)


    def _pcg(self, b, tol=1.0e-8, maxiter=1000):
        """Preconditioned CG solve of  M x = b  (single right-hand side).

        Matrix-free matvec (_gap_matvec) preconditioned by the Chan-circulant
        Cholesky factor (self.Mch).  M is well conditioned, so this converges in
        a handful of iterations and returns the EXACT M⁻¹ b (to tol) even though
        M itself is only the Chan approximation used for the preconditioner.
        """
        Lc = self.Mch
        def prec(r):
            y = solve_triangular(Lc, r, lower=True)
            return solve_triangular(Lc.T, y, lower=False)
        bnorm = math.sqrt(float(b @ b))
        x = np.zeros_like(b)
        if bnorm == 0.0:
            return x
        r  = b.copy()
        z  = prec(r)
        p  = z.copy()
        rz = float(r @ z)
        thresh = tol * bnorm
        for _ in range(maxiter):
            Ap    = self._gap_matvec(p)
            alpha = rz / float(p @ Ap)
            x    += alpha * p
            r    -= alpha * Ap
            if math.sqrt(float(r @ r)) <= thresh:
                break
            z      = prec(r)
            rz_new = float(r @ z)
            p      = z + (rz_new / rz) * p
            rz     = rz_new
        return x


    def _block_pcg(self, B, tol=1.0e-8, maxiter=1000):
        """Preconditioned CG solving  M X = B  for all columns of B at once.

        Independent per-column scalars (α, β) with a single shared batched
        matvec (`_gap_matvec` on the whole block) and one BLAS-3 preconditioner
        solve per iteration.  Identical result to solving each column with
        `_pcg`, but the FFTs and triangular solves run over the batch, which is
        where multi-threading pays off.
        """
        Lc = self.Mch
        def prec(R):
            Y = solve_triangular(Lc, R, lower=True)
            return solve_triangular(Lc.T, Y, lower=False)
        bnorm  = np.sqrt(np.einsum('ij,ij->j', B, B))
        X      = np.zeros_like(B)
        if not np.any(bnorm > 0.0):
            return X
        R      = B.copy()
        Z      = prec(R)
        P      = Z.copy()
        rz     = np.einsum('ij,ij->j', R, Z)
        thresh = tol * bnorm
        for _ in range(maxiter):
            AP    = self._gap_matvec(P)
            pAp   = np.einsum('ij,ij->j', P, AP)
            safe  = pAp != 0.0
            alpha = np.where(safe, rz / np.where(safe, pAp, 1.0), 0.0)
            X    += alpha * P
            R    -= alpha * AP
            if np.all(np.sqrt(np.einsum('ij,ij->j', R, R)) <= thresh):
                break
            Z      = prec(R)
            rz_new = np.einsum('ij,ij->j', R, Z)
            good   = rz != 0.0
            beta   = np.where(good, rz_new / np.where(good, rz, 1.0), 0.0)
            P      = Z + beta * P
            rz     = rz_new
        return X


    def _Msolve(self, B):
        """Return the exact  M⁻¹ B  (B a vector or a (k, n_reg) matrix).

        For m < GRAM_THRESHOLD, M (hence self.Mch) is exact, so M⁻¹ is applied
        with two triangular solves.  Otherwise M is the Chan approximation and
        the exact inverse is obtained by preconditioned CG — a single block
        solve over all columns when B is a matrix.
        """
        if self._gram_exact:
            Y = solve_triangular(self.Mch, B, lower=True)
            return solve_triangular(self.Mch.T, Y, lower=False)
        if B.ndim == 1:
            return self._pcg(B)
        return self._block_pcg(B)


    def _quad_from_ab(self, al, be, zn2):
        """Gauss-quadrature estimate of  zᵀ log(P) z  from harvested CG (α, β).

        Rebuilds the Lanczos tridiagonal T (Golub–Meurant) from the coefficient
        sequences of one probe's CG run and evaluates
        zᵀ log(P) z ≈ ‖z‖² Σ_i (u_{i,0})² log(θ_i),  (θ_i, u_i) = eig(T).
        """
        n = len(al)
        if n == 0:
            return 0.0
        dgl = np.empty(n)
        off = np.empty(max(n - 1, 0))
        dgl[0] = 1.0 / al[0]
        for i in range(1, n):
            dgl[i] = 1.0 / al[i] + be[i - 1] / al[i - 1]
        for i in range(n - 1):
            off[i] = math.sqrt(be[i]) / al[i]
        T = np.diag(dgl) + np.diag(off, 1) + np.diag(off, -1)
        theta, U = np.linalg.eigh(T)
        return zn2 * float(np.sum(U[0, :] ** 2 * np.log(np.maximum(theta, 1.0e-30))))


    def _logdet_correction(self):
        """Estimate log|M_chan⁻¹ M| = log|M| − log|M_chan| (the exact correction
        to the Chan log-determinant) by stochastic Lanczos quadrature on the
        symmetric preconditioned operator  P = L⁻¹ M L⁻ᵀ  (L = self.Mch).

        All LOGDET_PROBES Rademacher probes are advanced together as one block
        CG — a single shared batched matvec and BLAS-3 triangular solves per
        iteration — while each probe keeps its own (α, β) sequence and Lanczos
        tridiagonal.  The per-column recurrences are independent, so the result
        is identical to running the probes one at a time.  P's spectrum clusters
        near 1, so a handful of iterations suffice.  Fixed seed → deterministic
        objective.
        """
        Lc = self.Mch
        k  = self.gap_idx.size
        def Pmatvec(V):                              # P = L⁻¹ M L⁻ᵀ, batched
            W = solve_triangular(Lc, V, lower=True, trans='T')
            return solve_triangular(Lc, self._gap_matvec(W), lower=True)

        # Draw one probe at a time (same RNG call order as the original
        # per-probe loop) so the block estimate is bit-identical to solving the
        # probes sequentially — SLQ with few probes is sensitive to the exact
        # Rademacher vectors, so the draw order must be preserved.
        rng    = np.random.default_rng(LOGDET_SEED)
        Z      = np.empty((k, LOGDET_PROBES))
        for j in range(LOGDET_PROBES):
            Z[:, j] = rng.integers(0, 2, k).astype(float) * 2.0 - 1.0
        zn2    = np.einsum('ij,ij->j', Z, Z)

        R      = Z.copy()
        P      = R.copy()
        rs     = np.einsum('ij,ij->j', R, R)
        thresh = 1.0e-8 * np.sqrt(zn2)
        al     = [[] for _ in range(LOGDET_PROBES)]
        be     = [[] for _ in range(LOGDET_PROBES)]
        done   = np.zeros(LOGDET_PROBES, dtype=bool)
        for _ in range(LOGDET_MAXITER):
            AP    = Pmatvec(P)
            pAp   = np.einsum('ij,ij->j', P, AP)
            act   = ~done & (pAp > 0.0)
            alpha = np.where(act, rs / np.where(act, pAp, 1.0), 0.0)
            R     = R - alpha * AP
            rs_new = np.einsum('ij,ij->j', R, R)
            for j in range(LOGDET_PROBES):
                if done[j]:
                    continue
                if pAp[j] <= 0.0:                    # breakdown: freeze, keep T so far
                    done[j] = True
                    continue
                al[j].append(float(alpha[j]))
                be[j].append(float(rs_new[j] / rs[j]))
                if math.sqrt(rs_new[j]) <= thresh[j]:
                    done[j] = True
            if np.all(done):
                break
            good = ~done
            beta = np.where(good, rs_new / np.where(good, rs, 1.0), 0.0)
            P    = R + beta * P
            rs   = rs_new

        total = sum(self._quad_from_ab(al[j], be[j], zn2[j])
                    for j in range(LOGDET_PROBES))
        return total / LOGDET_PROBES


    @quiet_matmul
    def compute_leastsquares(self, t, H, x, F, samenoise=False):
        """ Compute least-squares

        Arg:
            t (m*1 matrix)  : first column of Toeplitz covariance matrix C
            H (m*n matrix)  : design matrix
            x (m*1 matrix)  : observations
            F (m*k matrix)  : gap indicator matrix (one column per gap)
            samenoise (bool): reuse cached whitening filters and Gram from
                              the previous call (noise params unchanged)

        Returns:
            theta (n*1 matrix)    : estimated parameters
            C_theta  (n*n matrix) : covariance matrix of estimated parameters
            ln_det_C (float)      : log(det(C))
            sigma_eta (float)     : driving noise
        """

        if not samenoise:
            (m, n_reg) = H.shape
            (m, k)     = F.shape

            # Pad all convolution FFTs to next power of 2 above 2m.
            # rfft(2m) with a large prime factor is catastrophically slow on
            # some platforms (e.g. 21916 = 4×5479prime on x86-64 pocketfft).
            N_fft = 1 << int(np.ceil(np.log2(2 * m)))
            self.N_fft = N_fft

            #--- Factorise Toeplitz C via GSA (large n) or Durbin-Levinson (small n)
            if m > GSA_THRESHOLD:
                l1_vec, l2_vec, delta, self.ln_det_C = Schur().compute(t)
            else:
                l1_vec, l2_vec, delta, self.ln_det_C = Levinson().compute(t)

            #--- Zero-pad to N_fft for FFT convolution; normalise by sqrt(delta)
            scale = 1.0 / math.sqrt(delta)
            l1 = np.zeros(N_fft); l1[:m] = l1_vec * scale
            l2 = np.zeros(N_fft); l2[:m] = l2_vec * scale
            self.Fl1    = np.fft.rfft(l1)
            self.Fl2    = np.fft.rfft(l2)
            self.z      = np.zeros(N_fft - m)
            self.l1_vec = l1_vec * scale   # m-length time-domain whitening impulse
            self.l2_vec = l2_vec * scale

            #--- Whiten observations (NaN gaps → 0 before FFT)
            xm = np.where(np.isnan(x), 0.0, x)
            Fx = np.fft.rfft(np.concatenate([xm, self.z]))
            self.y1 = np.fft.irfft(self.Fl1 * Fx, n=N_fft)[:m]
            self.y2 = np.fft.irfft(self.Fl2 * Fx, n=N_fft)[:m]

            if k > 0:
                #--- Gap positions from indicator matrix F
                self.gap_idx = np.array(
                    [int(np.argmax(F[:, i])) for i in range(k)])
                gap_idx = self.gap_idx

                # ── Gram matrix M = G1 G1ᵀ − G2 G2ᵀ ─────────────────────────
                if m < GRAM_THRESHOLD:
                    # Exact: construct G_s row-by-row from the scaled whitening
                    # vectors, then Cholesky-factorise M directly.
                    l1s = l1[:m]
                    l2s = l2[:m]
                    G1 = np.zeros((k, m))
                    G2 = np.zeros((k, m))
                    for i, gi in enumerate(gap_idx):
                        G1[i, gi:] = l1s[:m - gi]
                        G2[i, gi:] = l2s[:m - gi]
                    M_exact = G1 @ G1.T - G2 @ G2.T
                    Mch = np.linalg.cholesky(M_exact)
                else:
                    # Approach F: Chan's optimal circulant (m-point)
                    # c_j = t_j for j ≤ m//2, t_{m−j} otherwise.
                    # r_inv[d] = IFFT(1/S)[d];  M[i,j] = r_inv[|g_i − g_j|]
                    half  = m // 2
                    c_chan = np.concatenate([t[:half + 1], t[1:m - half][::-1]])
                    ev    = np.fft.rfft(c_chan).real
                    r_inv = np.fft.irfft(1.0 / np.maximum(ev, 1e-30), n=m).real
                    lags  = np.abs(gap_idx[:, None] - gap_idx[None, :])
                    M_approx = r_inv[lags]
                    Mch = np.linalg.cholesky(M_approx)

                self.ln_det_C += 2.0 * np.sum(np.log(np.diag(Mch)))
                self.Mch = Mch   # store factor; solve_triangular replaces dtrtri

                # ── Qy via FFT cross-correlation ──────────────────────────────
                # (G1 @ y1)[i] = xcorr(l1s, y1)[gap_idx[i]]
                #              = irfft(Fl1c * rfft([y1, 0]))[gap_idx[i]]
                Fl1c = self.Fl1.conj()
                Fl2c = self.Fl2.conj()
                Fy1 = np.fft.rfft(np.concatenate([self.y1, self.z]))
                Fy2 = np.fft.rfft(np.concatenate([self.y2, self.z]))
                G1y1 = np.fft.irfft(Fl1c * Fy1, n=N_fft)[:m].real[gap_idx]
                G2y2 = np.fft.irfft(Fl2c * Fy2, n=N_fft)[:m].real[gap_idx]
                # Mch @ Qy = (G1y1 − G2y2)  ← O(k²) forward substitution.
                # Half-factor solve kept as-is for the offset-scan fast path.
                self.Qy = solve_triangular(self.Mch, G1y1 - G2y2, lower=True)

                #--- Exact gap correction (design X): apply the FULL M⁻¹ so that
                #    C_theta, theta and the residual sum-of-squares are exact even
                #    when M is the Chan circulant approximation (m ≥ GRAM_THRESHOLD).
                #    For m < GRAM_THRESHOLD, M is already exact; otherwise M⁻¹ is
                #    applied matrix-free by preconditioned CG (Chan factor as the
                #    preconditioner, FFT matrix-vector products).
                self._m          = m
                self._gram_exact = m < GRAM_THRESHOLD
                self._setup_gap_matvec(N_fft, m, gap_idx)
                self.rhs_y       = G1y1 - G2y2
                self.Qy_full     = self._Msolve(self.rhs_y)

                #--- Exact log-determinant (design X+): the line above added the
                #    Chan value log|M_chan|; for the Chan case add the correction
                #    log|M_chan⁻¹M| (CG-harvested SLQ) so ln_det_C is exact too.
                #    (Small-m path uses the exact M, so no correction is needed.)
                if not self._gram_exact:
                    self.ln_det_C += self._logdet_correction()

        #=== END OF NOISE-DEPENDENT SECTION

        #--- Dimensions (re-derived so samenoise=True path has them)
        (m, n_reg) = H.shape
        (m, k)     = F.shape

        #--- Whiten design matrix columns (NaN rows → 0)
        Hm = np.where(np.isnan(x)[:, None], 0.0, H)
        A1 = np.zeros((n_reg, m))
        A2 = np.zeros((n_reg, m))
        for i in range(n_reg):
            FH    = np.fft.rfft(np.concatenate([Hm[:, i], self.z]))
            A1[i] = np.fft.irfft(self.Fl1 * FH, n=self.N_fft)[:m]
            A2[i] = np.fft.irfft(self.Fl2 * FH, n=self.N_fft)[:m]

        if k > 0:
            gap_idx = self.gap_idx
            Fl1c    = self.Fl1.conj()
            Fl2c    = self.Fl2.conj()

            # ── GA12 via FFT cross-correlation ────────────────────────────────
            # GA12[i,j] = xcorr(l1s, A1[j])[g_i] − xcorr(l2s, A2[j])[g_i]
            #           = (G1 @ A1.T − G2 @ A2.T)[i, j]
            GA12 = np.zeros((k, n_reg))
            for j in range(n_reg):
                FA1j = np.fft.rfft(np.concatenate([A1[j], self.z]))
                FA2j = np.fft.rfft(np.concatenate([A2[j], self.z]))
                G1A1j = np.fft.irfft(Fl1c * FA1j, n=self.N_fft)[:m].real[gap_idx]
                G2A2j = np.fft.irfft(Fl2c * FA2j, n=self.N_fft)[:m].real[gap_idx]
                GA12[:, j] = G1A1j - G2A2j

            # Exact gap correction (design X): use the FULL M⁻¹ (via _Msolve),
            # not the Chan half-factor, so C_theta, theta and the residual
            # sum-of-squares are exact even when M is the Chan approximation.
            #   QA = M⁻¹ GA12 ,  self.Qy_full = M⁻¹ (G1@y1 − G2@y2)
            QA      = self._Msolve(GA12)
            C_theta = inv(A1 @ A1.T - A2 @ A2.T - GA12.T @ QA)
            theta   = C_theta @ (A1 @ self.y1 - A2 @ self.y2 - GA12.T @ self.Qy_full)

            t1 = self.y1 - A1.T @ theta
            t2 = self.y2 - A2.T @ theta

            # Whitened-residual gap term  bᵀ M⁻¹ b  with  b = (G1@y1−G2@y2) − GA12·theta
            # and  Qt = M⁻¹ b = Qy_full − QA·theta  (no extra FFTs).
            b  = self.rhs_y - GA12 @ theta
            Qt = self.Qy_full - QA @ theta

            rss = (np.dot(t1, t1) - np.dot(t2, t2) - np.dot(b, Qt)) / (m - k)
            sigma_eta = math.sqrt(rss) if rss > 0.0 else math.nan
        else:
            C_theta = inv(A1 @ A1.T - A2 @ A2.T)
            theta   = C_theta @ (A1 @ self.y1 - A2 @ self.y2)

            t1 = self.y1 - A1.T @ theta
            t2 = self.y2 - A2.T @ theta
            rss = (np.dot(t1, t1) - np.dot(t2, t2)) / m
            sigma_eta = math.sqrt(rss) if rss > 0.0 else math.nan

        return [theta, C_theta, self.ln_det_C, sigma_eta]


    @quiet_matmul
    def fast_epoch_scan(self, H_base, x, N, useRMLE, offset_index):
        """Efficient O(n_fixed × m²) epoch scan replacing the naive O(n_reg × m² log m) loop.

        Two key optimisations over the original test_new_offset inner loop:

          1. Fixed columns whitened once.  The n_fixed design-matrix columns
             (trend, annual, semi-annual, previously accepted offsets) never
             change between epochs.  We whiten them all before entering the
             loop, saving (n_fixed / n_reg) × m FFT pairs.

          2. Incremental Heaviside update.  The Heaviside step column shifts
             by one sample each iteration, so its whitened version satisfies
               A1h_{i+1}[i:] = A1h_i[i:] − l1_vec[:m−i]
             (O(m) subtraction) instead of a fresh O(m log m) FFT.

        The augmented LS solution at each epoch uses the Schur complement of
        the Gram matrix so only O(n_fixed²) work is needed per epoch.

        Requires no data gaps (k = 0) and a prior samenoise=False call that
        populates self.Fl1, self.Fl2, self.l1_vec, self.l2_vec, self.y1, self.y2.

        Args:
            H_base      : design matrix WITHOUT the candidate Heaviside column (m × n_fixed)
            x           : observations (m,), used only for NaN gap detection
            N           : effective observation count (= m for no-gap series)
            useRMLE     : bool
            offset_index: list of row indices already occupied by accepted offsets

        Returns:
            dln_L_new (np.ndarray, shape m): delta log-likelihood for each candidate epoch;
                      0.0 at epoch 0, at gap epochs, and at already-accepted offset epochs.
        """
        (m, n_fixed) = H_base.shape
        offset_set   = set(offset_index)

        # ── Whiten fixed columns once ─────────────────────────────────────
        Hm  = np.where(np.isnan(x)[:, None], 0.0, H_base)
        A1f = np.zeros((n_fixed, m))
        A2f = np.zeros((n_fixed, m))
        for j in range(n_fixed):
            FH     = np.fft.rfft(np.concatenate([Hm[:, j], self.z]))
            A1f[j] = np.fft.irfft(self.Fl1 * FH, n=self.N_fft)[:m]
            A2f[j] = np.fft.irfft(self.Fl2 * FH, n=self.N_fft)[:m]

        # ── Base-model Gram, inverse, and baseline RSS ────────────────────
        Gf       = A1f @ A1f.T - A2f @ A2f.T        # (n_fixed × n_fixed)
        Gf_inv   = inv(Gf)
        gf       = A1f @ self.y1 - A2f @ self.y2    # (n_fixed,)
        theta_f  = Gf_inv @ gf
        rss_y    = np.dot(self.y1, self.y1) - np.dot(self.y2, self.y2)
        rss_base = rss_y - np.dot(gf, theta_f)

        # ── RMLE pre-computation ──────────────────────────────────────────
        if useRMLE:
            HbTHb     = H_base.T @ H_base             # (n_fixed × n_fixed)
            HbTHb_inv = inv(HbTHb)
            # suffix_H[i] = H_base[i:].sum(axis=0) = H_base.T @ h_heaviside(i)
            suffix_H = np.cumsum(H_base[::-1], axis=0)[::-1]   # (m, n_fixed)

        # ── Initial whitened Heaviside at epoch 1: h = [0, 1, 1, ..., 1] ─
        h_init    = np.ones(m); h_init[0] = 0.0
        FH        = np.fft.rfft(np.concatenate([h_init, self.z]))
        A1h = np.fft.irfft(self.Fl1 * FH, n=self.N_fft)[:m]
        A2h = np.fft.irfft(self.Fl2 * FH, n=self.N_fft)[:m]

        l1m = self.l1_vec   # (m,) time-domain whitening filter
        l2m = self.l2_vec

        # ── Epoch scan ────────────────────────────────────────────────────
        skip = np.zeros(m, dtype=np.uint8)
        skip[0] = 1
        for i in range(1, m):
            if np.isnan(x[i]) or i in offset_set:
                skip[i] = 1

        if _USE_CYTHON_NOGAP:
            sfxH_arg    = suffix_H    if useRMLE else np.zeros((1, 1))
            HbHb_arg    = HbTHb_inv  if useRMLE else np.zeros((1, 1))
            dln_L_new = _epoch_scan_nogap_cython(
                A1f, A2f, A1h, A2h,
                Gf_inv, theta_f, rss_base,
                self.y1, self.y2,
                l1m, l2m,
                N, n_fixed, int(useRMLE),
                sfxH_arg, HbHb_arg,
                skip,
            )
        else:
            dln_L_new = np.zeros(m)
            for i in range(1, m):
                if not skip[i]:
                    c        = A1f @ A1h - A2f @ A2h
                    h_new    = np.dot(A1h, self.y1) - np.dot(A2h, self.y2)
                    s        = np.dot(A1h, A1h)    - np.dot(A2h, A2h)
                    u        = Gf_inv @ c
                    S_schur  = s - np.dot(c, u)
                    if S_schur > 1e-10:
                        r        = np.dot(theta_f, c)
                        rss_aug  = rss_base - (h_new - r) ** 2 / S_schur
                        if rss_aug > 0.0:
                            if useRMLE:
                                f    = suffix_H[i]
                                S_HH = float(m - i) - float(f @ HbTHb_inv @ f)
                                if S_HH > 0.0:
                                    dln_L_new[i] = (0.5 * (N - n_fixed) * math.log(rss_base / rss_aug)
                                                    + 0.5 * math.log(s / S_schur)
                                                    + 0.5 * math.log(S_HH))
                            else:
                                dln_L_new[i] = 0.5 * N * math.log(rss_base / rss_aug)
                if i < m - 1:
                    n_rem    = m - i
                    A1h[i:] -= l1m[:n_rem]
                    A2h[i:] -= l2m[:n_rem]

        return dln_L_new


    def _precompute_GA12h_table(self):
        """Precompute GA12h for all epochs via cumsum/xcorr formula (O(k·m log m)).

        Returns GA12h_table_T, shape (m, k), where
            GA12h_table_T[i, j] = xcorr(l1s_j, L1h_ext)[g_j - i] - xcorr(l2s_j, L2h_ext)[g_j - i]
        and L1h = cumsum(l1s), L1h_ext[n] = L1h[n] for n≥0, else 0.
        This is the exact gap-correction cross-correlation at each epoch,
        computed once and looked up O(k) per epoch scan step.
        """
        m   = len(self.l1_vec)
        k   = len(self.gap_idx)
        l1s = self.l1_vec
        l2s = self.l2_vec
        N_fft = 1 << int(np.ceil(np.log2(2 * m + 1)))
        L1h_pad = np.zeros(N_fft); L1h_pad[:m] = np.cumsum(l1s)
        L2h_pad = np.zeros(N_fft); L2h_pad[:m] = np.cumsum(l2s)
        FL1h = np.fft.rfft(L1h_pad)
        FL2h = np.fft.rfft(L2h_pad)
        GA12h_table_T = np.zeros((m, k))   # (m, k) — row i has k values for epoch i
        epoch_idx = np.arange(m)
        for j, g in enumerate(self.gap_idx):
            l1s_j = np.zeros(N_fft); l1s_j[:m - g] = l1s[:m - g]
            l2s_j = np.zeros(N_fft); l2s_j[:m - g] = l2s[:m - g]
            xc1 = np.fft.irfft(np.conj(np.fft.rfft(l1s_j)) * FL1h)
            xc2 = np.fft.irfft(np.conj(np.fft.rfft(l2s_j)) * FL2h)
            lags = (g - epoch_idx) % N_fft
            GA12h_table_T[:, j] = (xc1 - xc2)[lags]
        return np.ascontiguousarray(GA12h_table_T)

    @quiet_matmul
    def fast_epoch_scan_with_gaps(self, H_base, x, N, useRMLE, offset_index):
        """Efficient epoch scan for series WITH data gaps (k > 0).

        Precomputes the GA12h table (O(k·m log m)) then delegates to the
        Cython extension (if available) or the Python fallback.  Both paths
        do an O(k) table lookup per epoch instead of 4 FFTs.

        Requires a prior samenoise=False call that populates
        self.Fl1, self.Fl2, self.l1_vec, self.l2_vec,
        self.y1, self.y2, self.gap_idx, self.Mch, self.Qy.
        """
        k = len(self.gap_idx)
        GA12h_table_T = self._precompute_GA12h_table() if k > 0 else None

        if _USE_CYTHON_GAPS:
            return _epoch_scan_gaps_cython(
                self.Fl1, self.Fl2, self.l1_vec, self.l2_vec,
                self.y1, self.y2, self.gap_idx,
                self.Mch,
                self.Qy, H_base, x, N,
                int(useRMLE), offset_index,
                GA12h_table_T,
            )
        # Python fallback: same algorithm without Cython acceleration
        return self._fast_epoch_scan_with_gaps_python(
            H_base, x, N, useRMLE, offset_index)


    @quiet_matmul
    def _fast_epoch_scan_with_gaps_python(self, H_base, x, N, useRMLE, offset_index,
                                           incremental=True):
        """Pure-Python reference implementation of fast_epoch_scan_with_gaps.

        incremental=True  : update GA12h via O(k) arithmetic each epoch.
        incremental=False : recompute GA12h via 4 FFTs each epoch (old code).
        """
        m       = len(self.y1)
        k       = len(self.gap_idx)
        n_fixed = H_base.shape[1]
        offset_set = set(offset_index)
        N_fft = self.N_fft
        z = np.zeros(N_fft - m)

        # Precompute GA12h table for all epochs (incremental path only).
        # GA12h_table[j, i] = xcorr(l1s_j, L1h_ext)[g_j - i] - xcorr(l2s_j, L2h_ext)[g_j - i]
        # where L1h = cumsum(l1s), l1s_j = l1s[:m-g_j] (zero outside that range),
        # and L1h_ext[n] = L1h[n] for n≥0, else 0.
        # Negative lags (i > g_j) map to wrap-around indices in N_fft-point DFT.
        # Cost: O(k · m log m) — replaces O(m² log m) FFT-per-epoch.
        if incremental and k > 0:
            l1s = self.l1_vec           # (m,) scaled whitening filter
            l2s = self.l2_vec
            N_fft = 1 << int(np.ceil(np.log2(2 * m + 1)))
            L1h_pad = np.zeros(N_fft);  L1h_pad[:m] = np.cumsum(l1s)
            L2h_pad = np.zeros(N_fft);  L2h_pad[:m] = np.cumsum(l2s)
            FL1h = np.fft.rfft(L1h_pad)
            FL2h = np.fft.rfft(L2h_pad)
            GA12h_table = np.zeros((k, m))
            epoch_idx   = np.arange(m)
            for j, g in enumerate(self.gap_idx):
                l1s_j = np.zeros(N_fft);  l1s_j[:m - g] = l1s[:m - g]
                l2s_j = np.zeros(N_fft);  l2s_j[:m - g] = l2s[:m - g]
                xc1 = np.fft.irfft(np.conj(np.fft.rfft(l1s_j)) * FL1h)
                xc2 = np.fft.irfft(np.conj(np.fft.rfft(l2s_j)) * FL2h)
                lags = (g - epoch_idx) % N_fft    # negative lags wrap to N_fft + lag
                GA12h_table[j] = (xc1 - xc2)[lags]

        # Whiten fixed columns once
        Hm  = np.where(np.isnan(x[:, None]), 0.0, H_base)
        A1f = np.zeros((n_fixed, m))
        A2f = np.zeros((n_fixed, m))
        for j in range(n_fixed):
            FH     = np.fft.rfft(np.concatenate([Hm[:, j], z]))
            A1f[j] = np.fft.irfft(self.Fl1 * FH, n=N_fft)[:m].real
            A2f[j] = np.fft.irfft(self.Fl2 * FH, n=N_fft)[:m].real

        # Gap-correct fixed columns
        Fl1c  = self.Fl1.conj()
        Fl2c  = self.Fl2.conj()
        GA12f = np.zeros((k, n_fixed))
        for col in range(n_fixed):
            FA1 = np.fft.rfft(np.concatenate([A1f[col], z]))
            FA2 = np.fft.rfft(np.concatenate([A2f[col], z]))
            GA12f[:, col] = (np.fft.irfft(Fl1c * FA1, n=N_fft)[:m].real[self.gap_idx]
                            - np.fft.irfft(Fl2c * FA2, n=N_fft)[:m].real[self.gap_idx])
        QA_fT = solve_triangular(self.Mch, GA12f, lower=True).T  # (n_fixed, k)

        # Base-model Gram and RSS
        Gf_raw   = A1f @ A1f.T - A2f @ A2f.T - QA_fT @ QA_fT.T
        gf       = A1f @ self.y1 - A2f @ self.y2 - QA_fT @ self.Qy
        Gf_inv   = inv(Gf_raw)
        theta_f  = Gf_inv @ gf
        rss_y    = (np.dot(self.y1, self.y1) - np.dot(self.y2, self.y2)
                    - np.dot(self.Qy, self.Qy))
        rss_base = rss_y - float(gf @ theta_f)

        # RMLE pre-computation
        if useRMLE:
            HbTHb_inv = inv(H_base.T @ H_base)
            suffix_H  = np.cumsum(H_base[::-1], axis=0)[::-1]

        # Initial whitened Heaviside at epoch 1
        h_init    = np.ones(m); h_init[0] = 0.0
        FH        = np.fft.rfft(np.concatenate([h_init, z]))
        A1h = np.fft.irfft(self.Fl1 * FH, n=N_fft)[:m].real
        A2h = np.fft.irfft(self.Fl2 * FH, n=N_fft)[:m].real

        l1m = self.l1_vec
        l2m = self.l2_vec
        dln_L_new = np.zeros(m)
        buf2m = np.zeros(N_fft)

        for i in range(1, m):
            if incremental and k > 0:
                # Table lookup: exact GA12h for epoch i, O(k) per step
                GA12h = GA12h_table[:, i]
                QA_h  = solve_triangular(self.Mch, GA12h, lower=True)
            elif k > 0:
                # Recompute GA12h from current A1h/A2h via 4 FFTs
                buf2m[:m] = A1h
                FA1h  = np.fft.rfft(buf2m)
                buf2m[:m] = A2h
                FA2h  = np.fft.rfft(buf2m)
                GA12h = (np.fft.irfft(Fl1c * FA1h, n=N_fft)[:m].real[self.gap_idx]
                        - np.fft.irfft(Fl2c * FA2h, n=N_fft)[:m].real[self.gap_idx])
                QA_h  = solve_triangular(self.Mch, GA12h, lower=True)
            else:
                QA_h = np.zeros(0)

            if not np.isnan(x[i]) and i not in offset_set:
                c_eff  = A1f @ A1h - A2f @ A2h - QA_fT @ QA_h
                s_h    = (np.dot(A1h, A1h) - np.dot(A2h, A2h)
                          - np.dot(QA_h, QA_h))
                h_new_h = (np.dot(A1h, self.y1) - np.dot(A2h, self.y2)
                           - np.dot(QA_h, self.Qy))
                u       = Gf_inv @ c_eff
                S_schur = s_h - float(c_eff @ u)

                if S_schur > 1e-10:
                    r_h     = float(theta_f @ c_eff)
                    rss_aug = rss_base - (h_new_h - r_h)**2 / S_schur
                    if rss_aug > 0.0:
                        lnrat = math.log(rss_base / rss_aug)
                        if useRMLE:
                            f        = suffix_H[i]
                            S_HH     = float(m - i) - float(f @ HbTHb_inv @ f)
                            if S_HH > 0.0 and s_h > 0.0:
                                dln_L_new[i] = (0.5 * (N - n_fixed) * lnrat
                                                + 0.5 * math.log(s_h / S_schur)
                                                + 0.5 * math.log(S_HH))
                        else:
                            dln_L_new[i] = 0.5 * N * lnrat

            # Heaviside update: A1h/A2h are needed every epoch for dot products
            # with A1f, y1, etc. regardless of whether the GA12h is from the table.
            if i < m - 1:
                n_rem    = m - i
                A1h[i:] -= l1m[:n_rem]
                A2h[i:] -= l2m[:n_rem]

        return dln_L_new
