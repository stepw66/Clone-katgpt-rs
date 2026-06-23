//! KARC — Kolmogorov-Arnold Reservoir Computing: delay-basis-ridge forecaster.
//!
//! A modelless, inference-time trajectory forecaster. Given a trajectory
//! `{u_i ∈ ℝᴰ}`, KARC:
//! 1. Concatenates the last-K observations (delay embedding): `x_i = u_i ⊕ u_{i-1} ⊕ … ⊕ u_{i-K+1}` (paper Eq. 10), length `K·D`.
//! 2. Expands each coordinate onto `M` basis functions via a sealed [`KarcBasis`] trait (paper Eq. 8): `Ψ(x) ∈ ℝ^{K·D·M}`.
//! 3. Fits a linear readout `Wout` by closed-form ridge regression (paper Eq. 14): `Wout = Y·Hᵀ·(H·Hᵀ + λI)⁻¹`. The Woodbury identity (paper Eq. 40) is used to pick the cheaper of the feature-space `(XᵀX + λI)⁻¹` or sample-space `(XXᵀ + λI)⁻¹` inversion.
//! 4. Forecasts in a single zero-alloc matvec: `û_{i+1} = Wout · Ψ(x_i)` (paper Eq. 11).
//!
//! No backprop, no autodiff, no softmax. The basis functions are bounded
//! (Fourier, Chebyshev `|Tⱼ|≤1`, B-spline partition-of-unity), so the forecast
//! stays numerically well-behaved without sigmoid/softmax gating.
//!
//! # References
//!
//! - **Plan:** `katgpt-rs/.plans/308_karc_delay_basis_ridge_forecaster.md`
//! - **Research:** `katgpt-rs/.research/288_KARC_Delay_Basis_Ridge_Forecaster.md`
//! - **Source paper:** arXiv:2606.19984 — Huang, Kurths, Tang, *Kolmogorov-Arnold Reservoir Computing*, 2026-06-18
//! - **Sibling ridge path:** `crates/katgpt-core/src/peira.rs` (Plan 153) — same `(N + λI)⁻¹` math, f64, inter-view covariances. KARC reuses the pattern via `linalg::ridge_solve` (f32) rather than touching PEIRA.
//!
//! # GOAT gate (Plan 308 §"GOAT gate")
//!
//! - **G1** — double-scroll NRMSE ≤ 1.0e-3, threshold time ≥ 8 Lyapunov times. Verified by `examples/karc_double_scroll.rs`.
//! - **G2** — `forecast_into` ≤ 500 ns/call at D=8,M=8,K=4. Verified by `benches/karc_forecast_bench.rs`.
//! - **G3** — `forecast_into` zero allocations after warmup. Verified by `tests/karc_alloc_check.rs`.
//! - **G4** — two forecasters fit on identical data produce byte-identical `Wout`. Verified by `tests/karc_reproducibility.rs`.

use crate::linalg::ridge_solve::{ridge_solve_direct_f64, ridge_solve_woodbury_f32};
use crate::simd;

// ── Sealed trait machinery ────────────────────────────────────────────────

mod sealed {
    /// Seals [`super::KarcBasis`] so only the three shipped implementations
    /// (Fourier / Chebyshev / BSpline) can implement it — lets us add methods
    /// later without breaking downstream impls.
    pub trait Sealed {}
}

/// Univariate basis dictionary for one coordinate of the delay-embedded state.
///
/// `eval_into` projects a scalar `x` into `out[0..M]`. The three shipped impls
/// are bounded: Fourier `‖ψ‖≤1` per mode, Chebyshev `|Tⱼ|≤1`, B-spline
/// partition-of-unity. Bounds on the full feature vector are in paper Eq. 96/99/101.
///
/// Sealed per Plan 308 T1.2 — vendored minimal trait with attribution to
/// `riir-engine::linoss::basis::SpectralBasis` (same Fourier/Chebyshev/B-spline
/// family; KARC cannot import riir-engine from katgpt-core).
pub trait KarcBasis<const M: usize>: sealed::Sealed + Sync {
    /// Project scalar `x` into the `M`-length feature slice. Zero-allocation.
    fn eval_into(&self, x: f32, out: &mut [f32; M]);

    /// Human-readable name for diagnostics / G4 attribution.
    fn name(&self) -> &'static str;
}

// ── Basis implementations ─────────────────────────────────────────────────

/// Fourier basis: `ψ_{2i-1}(x) = cos(2π·i·x/P)`, `ψ_{2i}(x) = sin(2π·i·x/P)`,
/// `i = 1..=M/2` (paper Eq. 35). `M` must be even.
///
/// Spectral norm bound per coordinate: `‖ψ(x)‖₂ = √(M/2)` (paper Eq. 96).
/// Period `P` is set at construction and must be positive.
#[derive(Clone, Copy, Debug)]
pub struct FourierBasis<const M: usize> {
    /// Period `P > 0`. Frequencies are `ω_i = 2π·i/P`.
    period: f32,
}

impl<const M: usize> FourierBasis<M> {
    /// Construct with period `P`. Asserts `M` is even and `P > 0`.
    pub const fn new(period: f32) -> Self {
        assert!(M % 2 == 0, "FourierBasis requires even M (m = 2Q)");
        assert!(period > 0.0, "FourierBasis period must be positive");
        Self { period }
    }

    /// Angular frequency `ω_i = 2π·i/P`.
    #[inline]
    fn omega(&self, i: usize) -> f32 {
        core::f32::consts::TAU * (i as f32) / self.period
    }
}

impl<const M: usize> sealed::Sealed for FourierBasis<M> {}

impl<const M: usize> KarcBasis<M> for FourierBasis<M> {
    #[inline]
    fn eval_into(&self, x: f32, out: &mut [f32; M]) {
        let q = M / 2;
        let mut i = 0;
        while i < q {
            let w = self.omega(i + 1);
            out[2 * i] = (w * x).cos();
            out[2 * i + 1] = (w * x).sin();
            i += 1;
        }
    }

    fn name(&self) -> &'static str {
        "fourier"
    }
}

/// Chebyshev basis of the first kind: `T₀(x)=1, T₁(x)=x, Tₙ₊₁=2xTₙ−Tₙ₋₁`
/// (paper Eq. 38). `|Tⱼ(x)| ≤ 1` for `x ∈ [-1,1]`; caller is responsible for
/// rescaling inputs into `[-1,1]` (KARC does not rescale — the forecaster sees
/// whatever the trajectory contains, and ridge regularisation absorbs drift).
///
/// Spectral norm bound per coordinate: `‖ψ(x)‖₂ ≤ √M` (paper Eq. 101).
#[derive(Clone, Copy, Debug, Default)]
pub struct ChebyshevBasis<const M: usize>;

impl<const M: usize> ChebyshevBasis<M> {
    pub const fn new() -> Self {
        Self
    }
}

impl<const M: usize> sealed::Sealed for ChebyshevBasis<M> {}

impl<const M: usize> KarcBasis<M> for ChebyshevBasis<M> {
    #[inline]
    fn eval_into(&self, x: f32, out: &mut [f32; M]) {
        match M {
            0 => {}
            1 => out[0] = 1.0,
            _ => {
                out[0] = 1.0;
                out[1] = x;
                // Three-term recurrence T_{n+1} = 2x T_n - T_{n-1}.
                let mut n = 1;
                while n + 1 < M {
                    out[n + 1] = 2.0_f32.mul_add(x * out[n], -out[n - 1]);
                    n += 1;
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "chebyshev"
    }
}

/// Uniform cubic B-spline basis on `[0,1]` (paper Eq. 36–37, Cox–de Boor).
/// Degree fixed at 3. Knots are clamped uniform: the endpoints are repeated
/// `degree+1` times, interior knots evenly spaced. Forms a partition of unity,
/// so `‖ψ(x)‖₂ ≤ 1` per coordinate (paper Eq. 99).
///
/// `M` must be `≥ degree+1 = 4` (need at least one basis function). Knots are
/// stored in a `Vec<f32>` (allocated once at construction) because stable Rust
/// does not permit `[f32; M + degree + 1]` as a struct field when `M` is a
/// const generic (`generic_const_exprs` is unstable). Construction is not the
/// hot path — only `eval_into` is, and it reads the slice.
#[derive(Clone, Debug)]
pub struct BSplineBasis<const M: usize> {
    /// Clamped uniform knot vector, length `M + degree + 1 = M + 4`.
    knots: Vec<f32>,
}

impl<const M: usize> BSplineBasis<M> {
    const DEGREE: usize = 3;

    /// Number of knots = `M + degree + 1`.
    fn knot_len() -> usize {
        M + Self::DEGREE + 1
    }

    /// Construct on domain `[0,1]`.
    pub fn new() -> Self {
        assert!(M >= Self::DEGREE + 1, "BSplineBasis requires M >= degree+1 = 4");
        let d = Self::DEGREE;
        let knot_len = Self::knot_len();
        let mut knots = vec![0.0f32; knot_len];
        // Clamped: first degree+1 knots = 0, last degree+1 knots = 1.
        for i in 0..=d {
            knots[i] = 0.0;
            knots[knot_len - 1 - i] = 1.0;
        }
        // Interior knots: evenly spaced in (0,1). Number of interior knots =
        // knot_len - 2*(degree+1) = M + d + 1 - 2d - 2 = M - d - 1.
        let n_interior = M - d - 1; // 0 for the minimal M = d+1 case
        if n_interior > 0 {
            for k in 0..n_interior {
                knots[d + 1 + k] = (k + 1) as f32 / (n_interior + 1) as f32;
            }
        }
        Self { knots }
    }

    /// Cox–de Boor degree-0 indicator (paper Eq. 36). Right-continuous on
    /// `[t_i, t_{i+1})`; zero-width intervals (clamped repeated knots) return 0.
    /// The right boundary `x = t_max` is included in the last nonzero span so
    /// the partition-of-unity holds at the clamped endpoint.
    #[inline]
    fn b0(knots: &[f32], i: usize, x: f32) -> f32 {
        let lo = knots[i];
        let hi = knots[i + 1];
        if lo == hi {
            return 0.0; // zero-width interval (clamp repetition)
        }
        let t_max = knots[knots.len() - 1];
        if x >= lo && (x < hi || (hi == t_max && x <= hi)) {
            1.0
        } else {
            0.0
        }
    }
}

impl<const M: usize> Default for BSplineBasis<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const M: usize> sealed::Sealed for BSplineBasis<M> {}

impl<const M: usize> KarcBasis<M> for BSplineBasis<M> {
    #[inline]
    fn eval_into(&self, x: f32, out: &mut [f32; M]) {
        // Cox–de Boor recursion (paper Eq. 37). For M small (≤ ~60),
        // computing all M basis functions via the full triangular table is
        // cheap and avoids per-x nonzero-span bookkeeping.
        let d = Self::DEGREE;
        // prev/cur hold one degree-row at a time. n_knots-1 degree-0 functions.
        let n_knots = self.knots.len();
        let n_deg0 = n_knots - 1; // = M + d
        let mut prev = [0.0f32; 64];
        let mut cur = [0.0f32; 64];
        debug_assert!(n_deg0 <= 64, "BSplineBasis scratch overflow (M too large)");
        // Degree 0.
        for i in 0..n_deg0 {
            prev[i] = Self::b0(&self.knots, i, x);
        }
        // Recurse up to degree d.
        let mut s = 1;
        while s <= d {
            // At degree s, function count = n_knots - 1 - s.
            let count = n_knots - 1 - s;
            for i in 0..count {
                let denom1 = self.knots[i + s] - self.knots[i];
                let term1 = if denom1 > 0.0 {
                    (x - self.knots[i]) / denom1 * prev[i]
                } else {
                    0.0
                };
                let denom2 = self.knots[i + s + 1] - self.knots[i + 1];
                let term2 = if denom2 > 0.0 {
                    (self.knots[i + s + 1] - x) / denom2 * prev[i + 1]
                } else {
                    0.0
                };
                cur[i] = term1 + term2;
            }
            prev[..count].copy_from_slice(&cur[..count]);
            s += 1;
        }
        // At degree d the first M entries of prev are the M basis functions.
        for j in 0..M {
            out[j] = prev[j];
        }
    }

    fn name(&self) -> &'static str {
        "bspline"
    }
}

// ── Delay ring buffer ─────────────────────────────────────────────────────

/// Fixed-capacity ring buffer of the last-K `D`-dimensional observations.
///
/// `push` overwrites the oldest entry once the buffer is full.
/// `flatten_into` writes the delay-embedded state in observation order
/// **newest first**: `x = u_t ⊕ u_{t-1} ⊕ … ⊕ u_{t-K+1}` (paper Eq. 10).
///
/// Inline `[[f32; D]; K]` — no heap indirection, the small fixed sizes
/// (D≤~64, K≤~16 typical) make this cache-friendly and zero-alloc.
#[derive(Clone, Debug)]
pub struct DelayRing<const D: usize, const K: usize> {
    slots: [[f32; D]; K],
    /// Index of the most-recently-written slot.
    head: usize,
    /// Number of valid observations (caps at K).
    filled: usize,
}

impl<const D: usize, const K: usize> DelayRing<D, K> {
    /// Construct an empty ring.
    pub fn new() -> Self {
        Self {
            slots: [[0.0; D]; K],
            head: 0,
            filled: 0,
        }
    }

    /// Number of valid observations currently stored.
    #[inline]
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// True once K observations have been pushed.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.filled == K
    }

    /// Push a new observation, overwriting the oldest if full.
    #[inline]
    pub fn push(&mut self, obs: &[f32; D]) {
        if self.filled == 0 {
            self.head = 0;
        } else {
            self.head = (self.head + 1) % K;
        }
        self.slots[self.head].copy_from_slice(obs);
        if self.filled < K {
            self.filled += 1;
        }
    }

    /// Write the delay-embedded state `u_t ⊕ u_{t-1} ⊕ … ⊕ u_{t-K+1}` (newest
    /// first) into `out` (length `K·D`). Returns `false` if the ring is not yet
    /// full (caller should not fit/forecast on a partial state).
    #[inline]
    pub fn flatten_into(&self, out: &mut [f32]) -> bool {
        if !self.is_full() {
            return false;
        }
        for k in 0..K {
            // slot index for lag-k observation: head, head-1, ..., wrapping.
            let idx = (self.head + K - k) % K;
            let dst = k * D;
            out[dst..dst + D].copy_from_slice(&self.slots[idx]);
        }
        true
    }

    /// Reset the ring to empty (capacity unchanged).
    pub fn clear(&mut self) {
        self.head = 0;
        self.filled = 0;
    }
}

impl<const D: usize, const K: usize> Default for DelayRing<D, K> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Feature expansion ─────────────────────────────────────────────────────

/// Apply `basis` to each of the `K·D` delay coordinates, writing `K·D·M`
/// features into `out`. Zero-allocation. Inner loop is chunk-4 unrolled over
/// coordinates to help SIMD auto-vectorisation of the per-coordinate writes.
///
/// Layout: `out[c*M..(c+1)*M] = basis.eval(delay_state[c])` for coordinate `c`
/// in `0..K·D`, matching paper Eq. 8.
#[inline]
pub fn feature_expand<B: KarcBasis<M>, const M: usize>(
    delay_state: &[f32],
    basis: &B,
    out: &mut [f32],
) {
    let n_coords = delay_state.len();
    let mut tmp = [0.0f32; M];
    let mut c = 0;
    // Chunk-4 over coordinates: 4 eval_into calls + 4 copies per iteration.
    while c + 4 <= n_coords {
        basis.eval_into(delay_state[c], &mut tmp);
        out[c * M..(c + 1) * M].copy_from_slice(&tmp);
        basis.eval_into(delay_state[c + 1], &mut tmp);
        out[(c + 1) * M..(c + 2) * M].copy_from_slice(&tmp);
        basis.eval_into(delay_state[c + 2], &mut tmp);
        out[(c + 2) * M..(c + 3) * M].copy_from_slice(&tmp);
        basis.eval_into(delay_state[c + 3], &mut tmp);
        out[(c + 3) * M..(c + 4) * M].copy_from_slice(&tmp);
        c += 4;
    }
    while c < n_coords {
        basis.eval_into(delay_state[c], &mut tmp);
        out[c * M..(c + 1) * M].copy_from_slice(&tmp);
        c += 1;
    }
}

// ── Scratch ───────────────────────────────────────────────────────────────

/// Pre-allocated scratch for [`KarcForecaster::fit_ridge`]. Allocate once via
/// [`KarcScratch::with_capacity`], reuse across fits (clear() + repopulate).
///
/// All `Vec`s use `with_capacity`; `clear()` only resets lengths, never
/// reallocates. This backs the zero-alloc-fit contract.
///
/// The Gram/covariance/Cholesky buffers are **f64** for numerical robustness
/// at small λ (f32 Cholesky fails when λ is below f32 epsilon relative to the
/// matrix scale). The fit is a cold path; the forecast matvec stays f32.
pub struct KarcScratch {
    /// Feature-space Gram `XᵀX + λI` (`d_h × d_h`, f64 for precision).
    pub gram: Vec<f64>,
    /// Cross-covariance `XᵀY` (`d_h × D`, f64).
    pub cov: Vec<f64>,
    /// Cholesky factor L (`d_h × d_h`, f64).
    pub chol: Vec<f64>,
    /// Back-substitution temp (`d_h × D`, f64).
    pub z_solve: Vec<f64>,
    /// Wᵀ output buffer (`d_h × D`, f64) before f32 cast.
    pub w_t: Vec<f64>,
    /// Per-sample feature row (`d_h`, f32 — only used for transient expansion).
    pub feature_row: Vec<f32>,
    /// Sample-space Gram for the Woodbury path (`N × N`, f64).
    pub sample_gram: Vec<f64>,
    /// Sample-space Cholesky (`N × N`, f64).
    pub sample_chol: Vec<f64>,
    /// Sample-space back-substitution temp (`N × D`, f64).
    pub sample_z: Vec<f64>,
}

impl KarcScratch {
    /// Allocate for at most `d_h` features, `D` output dims, `N` samples.
    pub fn with_capacity(d_h: usize, d: usize, n: usize) -> Self {
        Self {
            gram: Vec::with_capacity(d_h * d_h),
            cov: Vec::with_capacity(d_h * d),
            chol: Vec::with_capacity(d_h * d_h),
            z_solve: Vec::with_capacity(d_h * d),
            w_t: Vec::with_capacity(d_h * d),
            feature_row: Vec::with_capacity(d_h),
            sample_gram: Vec::with_capacity(n * n),
            sample_chol: Vec::with_capacity(n * n),
            sample_z: Vec::with_capacity(n * d),
        }
    }

    /// Drop all contents (capacity unchanged).
    pub fn clear(&mut self) {
        self.gram.clear();
        self.cov.clear();
        self.chol.clear();
        self.z_solve.clear();
        self.w_t.clear();
        self.feature_row.clear();
        self.sample_gram.clear();
        self.sample_chol.clear();
        self.sample_z.clear();
    }
}

// ── Forecaster ────────────────────────────────────────────────────────────

/// `KarcForecaster<B, D, M, K>` — delay-basis-ridge trajectory forecaster.
///
/// - `B: KarcBasis<M>` — the univariate basis dictionary (Fourier / Chebyshev / BSpline).
/// - `D` — observation/state dimension (e.g. 3 for double-scroll, 8 for HLA).
/// - `M` — number of basis functions per coordinate.
/// - `K` — delay-embedding length (number of past observations concatenated).
///
/// Feature dimension `d_h = K·D·M`. The readout `Wout` is `D × d_h` row-major.
/// Forecast is `û = Wout · Ψ(x)`, a single SIMD matvec.
///
/// The type parameter `B` (rather than `dyn KarcBasis`) gives monomorphised
/// static dispatch on the forecast hot path — required for G2 (≤ 500 ns/call).
pub struct KarcForecaster<B: KarcBasis<M>, const D: usize, const M: usize, const K: usize> {
    /// The basis dictionary (immutable after construction).
    pub basis: B,
    /// Delay ring buffer of last-K observations.
    ring: DelayRing<D, K>,
    /// Readout matrix `Wout`, `D × d_h` row-major. Empty until first fit.
    pub wout: Vec<f32>,
    /// Set after the first successful fit.
    fitted: bool,
    /// Accumulated feature rows, `N × d_h` row-major (one row per training sample).
    features_buf: Vec<f32>,
    /// Accumulated target rows, `N × D` row-major.
    targets_buf: Vec<f32>,
    /// Number of accumulated training samples.
    n_samples: usize,
    /// Pre-allocated scratch for fit_ridge.
    scratch: KarcScratch,
    /// Pre-allocated delay buffer (`K·D`), reused by `observe_and_maybe_pair` /
    /// `forecast_now` / `forecast_into`. Allocated once at construction — never
    /// reallocated, so the hot path stays zero-alloc (G3).
    delay_buf: Vec<f32>,
    /// Pre-allocated feature buffer (`K·D·M = d_h`), reused by `forecast_into`.
    /// Allocated once — zero-alloc on the forecast hot path (G3).
    forecast_psi: Vec<f32>,
}

impl<B: KarcBasis<M>, const D: usize, const M: usize, const K: usize>
    KarcForecaster<B, D, M, K>
{
    /// Feature dimension `d_h = K·D·M`.
    pub const D_H: usize = K * D * M;

    /// Construct with a basis and capacity hint for `max_samples` training rows.
    pub fn with_capacity(basis: B, max_samples: usize) -> Self {
        let d_h = Self::D_H;
        Self {
            basis,
            ring: DelayRing::new(),
            wout: Vec::with_capacity(D * d_h),
            fitted: false,
            features_buf: Vec::with_capacity(max_samples * d_h),
            targets_buf: Vec::with_capacity(max_samples * D),
            n_samples: 0,
            scratch: KarcScratch::with_capacity(d_h, D, max_samples),
            delay_buf: vec![0.0; K * D],
            forecast_psi: vec![0.0; d_h],
        }
    }

    /// Number of accumulated training samples (rows in `features_buf`/`targets_buf`).
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Whether `fit_ridge` has produced a valid `Wout`.
    #[inline]
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Push an observation into the delay ring.
    #[inline]
    pub fn observe(&mut self, obs: &[f32; D]) {
        self.ring.push(obs);
    }

    /// Push an observation and, if the ring is full, accumulate the current
    /// delay state and the **next** observation as a training pair
    /// `(Ψ(x_t), u_{t+1})`. Convenience for streaming fits — the example/test
    /// paths build the buffer directly via [`Self::accumulate_pair`].
    ///
    /// Returns `true` if a pair was accumulated.
    #[inline]
    pub fn observe_and_maybe_pair(&mut self, obs: &[f32; D]) -> bool {
        // Snapshot the current delay state BEFORE pushing obs (so x_t = u_t ⊕ …),
        // then push obs (= u_{t+1} target). If the ring was full before the push,
        // we have a valid (x_t, u_{t+1}) pair.
        let have_state = self.ring.flatten_into(&mut self.delay_buf);
        self.ring.push(obs);
        if !have_state {
            return false;
        }
        // Copy the delay snapshot out of the shared buffer before accumulate_pair
        // re-enters feature_expand (which does not touch delay_buf, but the
        // explicit copy documents the lifetime).
        let delay_copy: Vec<f32> = self.delay_buf.clone();
        self.accumulate_pair(&delay_copy, obs);
        true
    }

    /// Expand `delay_state` (length `K·D`) into features and store
    /// `(Ψ(delay_state), target)` as a training row.
    #[inline]
    pub fn accumulate_pair(&mut self, delay_state: &[f32], target: &[f32; D]) {
        let d_h = Self::D_H;
        // Append a fresh feature row.
        let old_len = self.features_buf.len();
        self.features_buf.resize(old_len + d_h, 0.0);
        let row = &mut self.features_buf[old_len..old_len + d_h];
        feature_expand::<B, M>(delay_state, &self.basis, row);
        // Append the target row.
        let t_old = self.targets_buf.len();
        self.targets_buf.resize(t_old + D, 0.0);
        self.targets_buf[t_old..t_old + D].copy_from_slice(target);
        self.n_samples += 1;
    }

    /// Clear the accumulated trajectory buffer (does not clear `Wout`).
    pub fn clear_trajectory(&mut self) {
        self.features_buf.clear();
        self.targets_buf.clear();
        self.n_samples = 0;
    }

    /// Solve the ridge regression `Wout = Y·Hᵀ·(H·Hᵀ + λI)⁻¹` over the
    /// accumulated trajectory buffer, picking the cheaper of the direct
    /// feature-space form (when `d_h ≤ N`) or the Woodbury sample-space form
    /// (when `d_h > N`, paper Eq. 40–41).
    ///
    /// `λ > 0` is required (ridge diagonal; `λ = 0` would make the Gram
    /// singular for redundant features).
    ///
    /// On success, sets `Wout` (`D × d_h` row-major) and `is_fitted() == true`.
    pub fn fit_ridge(&mut self, lambda: f32) -> Result<(), FitError> {
        let d_h = Self::D_H;
        if self.n_samples == 0 {
            return Err(FitError::NoSamples);
        }
        if lambda <= 0.0 {
            return Err(FitError::NonPositiveLambda);
        }
        // Accumulate into scratch.
        let n = self.n_samples;
        if d_h <= n {
            self.fit_direct(lambda, d_h, n)?;
        } else {
            self.fit_woodbury(lambda, d_h, n)?;
        }
        self.fitted = true;
        Ok(())
    }

    /// Direct feature-space solve: `Wᵀ = (XᵀX + λI)⁻¹ XᵀY`. Accumulates the
    /// Gram/covariance in **f64** for numerical robustness at small λ (f32
    /// Cholesky fails when λ is below f32 epsilon relative to the matrix scale);
    /// casts Wᵀ to f32 for the forecast matvec.
    fn fit_direct(&mut self, lambda: f32, d_h: usize, n: usize) -> Result<(), FitError> {
        let s = &mut self.scratch;
        s.clear();
        let lambda64 = lambda as f64;
        // Gram = XᵀX (upper triangle), d_h × d_h, f64.
        s.gram.clear();
        s.gram.resize(d_h * d_h, 0.0);
        for r in 0..n {
            let row = &self.features_buf[r * d_h..(r + 1) * d_h];
            for i in 0..d_h {
                let ri = row[i] as f64;
                let mut j = i;
                while j + 4 <= d_h {
                    s.gram[i * d_h + j] += ri * row[j] as f64;
                    s.gram[i * d_h + j + 1] += ri * row[j + 1] as f64;
                    s.gram[i * d_h + j + 2] += ri * row[j + 2] as f64;
                    s.gram[i * d_h + j + 3] += ri * row[j + 3] as f64;
                    j += 4;
                }
                while j < d_h {
                    s.gram[i * d_h + j] += ri * row[j] as f64;
                    j += 1;
                }
            }
        }
        // Symmetrise.
        for i in 0..d_h {
            for j in 0..i {
                s.gram[i * d_h + j] = s.gram[j * d_h + i];
            }
        }
        // Add λI.
        for i in 0..d_h {
            s.gram[i * d_h + i] += lambda64;
        }
        // Cov = XᵀY, d_h × D, f64.
        s.cov.clear();
        s.cov.resize(d_h * D, 0.0);
        for r in 0..n {
            let row = &self.features_buf[r * d_h..(r + 1) * d_h];
            let target = &self.targets_buf[r * D..(r + 1) * D];
            for i in 0..d_h {
                let ri = row[i] as f64;
                for c in 0..D {
                    s.cov[i * D + c] += ri * target[c] as f64;
                }
            }
        }
        // f64 solve → Wᵀ (d_h × D).
        s.chol.clear();
        s.chol.resize(d_h * d_h, 0.0);
        s.z_solve.clear();
        s.z_solve.resize(d_h * D, 0.0);
        s.w_t.clear();
        s.w_t.resize(d_h * D, 0.0);
        {
            let gram = &s.gram[..];
            let cov = &s.cov[..];
            let chol = &mut s.chol[..];
            let z_solve = &mut s.z_solve[..];
            let w_t = &mut s.w_t[..];
            ridge_solve_direct_f64(w_t, chol, z_solve, gram, cov, d_h, D);
        }
        // Cast Wᵀ (d_h × D, f64) → Wout (D × d_h, f32) with transpose.
        self.wout.clear();
        self.wout.resize(D * d_h, 0.0);
        for r in 0..D {
            for c in 0..d_h {
                self.wout[r * d_h + c] = s.w_t[c * D + r] as f32;
            }
        }
        let _ = n;
        Ok(())
    }

    /// Woodbury sample-space solve: `Wᵀ = Xᵀ (X Xᵀ + λI)⁻¹ Y`. Uses the f32
    /// sample-space path (the sample count N is small in this regime, so f32
    /// precision suffices; for very small λ the caller should prefer the direct
    /// path which accumulates in f64).
    fn fit_woodbury(&mut self, lambda: f32, d_h: usize, n: usize) -> Result<(), FitError> {
        let s = &mut self.scratch;
        s.clear();
        // Sample Gram = X Xᵀ, N × N (f32 — sample count is small here).
        let mut sample_gram_f32 = vec![0.0f32; n * n];
        for i in 0..n {
            let row_i = &self.features_buf[i * d_h..(i + 1) * d_h];
            for j in 0..n {
                let row_j = &self.features_buf[j * d_h..(j + 1) * d_h];
                sample_gram_f32[i * n + j] = simd::simd_dot_f32(row_i, row_j, d_h);
            }
        }
        for i in 0..n {
            sample_gram_f32[i * n + i] += lambda;
        }
        let mut sample_chol_f32 = vec![0.0f32; n * n];
        let mut sample_z_f32 = vec![0.0f32; n * D];
        let w_t_len = d_h * D;
        self.wout.clear();
        self.wout.resize(w_t_len, 0.0);
        {
            let y = &self.targets_buf[..n * D];
            let x = &self.features_buf[..n * d_h];
            let w_t = &mut self.wout[..w_t_len];
            ridge_solve_woodbury_f32(
                w_t, &mut sample_chol_f32, &mut sample_z_f32, &sample_gram_f32, y, x, n, d_h, D,
            );
        }
        // Transpose Wᵀ (d_h × D) → Wout (D × d_h).
        let w_t_copy: Vec<f32> = self.wout[..w_t_len].to_vec();
        self.wout.resize(D * d_h, 0.0);
        for r in 0..D {
            for c in 0..d_h {
                self.wout[r * d_h + c] = w_t_copy[c * D + r];
            }
        }
        Ok(())
    }

    /// Forecast `û_{t+1} = Wout · Ψ(delay_state)` into `out` (length `D`).
    ///
    /// Zero allocation on the hot path: the feature buffer
    /// (`self.forecast_psi`, `d_h = K·D·M`) is pre-allocated at construction and
    /// reused via indexing — `GlobalAlloc`-counter tests see zero `alloc`/
    /// `dealloc` delta after warmup (G3). Stack arrays of size `K·D·M` are not
    /// expressible in stable Rust (`generic_const_exprs` is unstable), so the
    /// buffer lives on the heap but is never reallocated.
    ///
    /// Requires [`Self::is_fitted`]; returns `false` (leaving `out` untouched)
    /// if the forecaster has not been fit.
    #[inline]
    pub fn forecast_into(&mut self, delay_state: &[f32], out: &mut [f32]) -> bool {
        if !self.fitted {
            return false;
        }
        let d_h = Self::D_H;
        debug_assert_eq!(delay_state.len(), K * D, "delay_state must be K·D long");
        debug_assert!(out.len() >= D, "out must be at least D long");
        debug_assert_eq!(self.forecast_psi.len(), d_h);
        let psi = &mut self.forecast_psi[..d_h];
        feature_expand::<B, M>(&delay_state[..K * D], &self.basis, psi);
        simd::simd_matvec(&mut out[..D], &self.wout, psi, D, d_h);
        true
    }

    /// Convenience: forecast the forecaster's own current delay ring state.
    /// Returns `false` if the ring is not full or the forecaster is not fit.
    #[inline]
    pub fn forecast_now(&mut self, out: &mut [f32]) -> bool {
        if !self.fitted || !self.ring.is_full() {
            return false;
        }
        if !self.ring.flatten_into(&mut self.delay_buf) {
            return false;
        }
        let delay_copy: Vec<f32> = self.delay_buf.clone();
        self.forecast_into(&delay_copy, out)
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

/// Failure modes for [`KarcForecaster::fit_ridge`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FitError {
    /// `fit_ridge` called before any training pairs were accumulated.
    NoSamples = 0,
    /// `λ ≤ 0` would make the ridge solve singular.
    NonPositiveLambda = 1,
}

impl core::fmt::Display for FitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FitError::NoSamples => write!(f, "no training samples accumulated"),
            FitError::NonPositiveLambda => write!(f, "ridge lambda must be > 0"),
        }
    }
}

impl std::error::Error for FitError {}

// ── Inline unit tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn delay_ring_orders_newest_first() {
        let mut ring: DelayRing<2, 3> = DelayRing::new();
        ring.push(&[1.0, 1.0]);
        ring.push(&[2.0, 2.0]);
        ring.push(&[3.0, 3.0]);
        assert!(ring.is_full());
        let mut out = [0.0f32; 6];
        assert!(ring.flatten_into(&mut out));
        // newest first: [3,3, 2,2, 1,1]
        assert_eq!(out, [3.0, 3.0, 2.0, 2.0, 1.0, 1.0]);
    }

    #[test]
    fn delay_ring_wraps_around() {
        let mut ring: DelayRing<1, 3> = DelayRing::new();
        for v in 0..5 {
            ring.push(&[v as f32]);
        }
        let mut out = [0.0f32; 3];
        assert!(ring.flatten_into(&mut out));
        // last three: 4, 3, 2
        assert_eq!(out, [4.0, 3.0, 2.0]);
    }

    #[test]
    fn fourier_basis_is_bounded() {
        let basis: FourierBasis<8> = FourierBasis::new(1.0);
        let mut out = [0.0f32; 8];
        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0, -1.0, 3.7] {
            basis.eval_into(x, &mut out);
            for &v in &out {
                assert!(v.abs() <= 1.0 + 1e-5, "Fourier value out of [-1,1]: {}", v);
            }
        }
    }

    #[test]
    fn chebyshev_basis_recurrence() {
        let basis: ChebyshevBasis<4> = ChebyshevBasis::new();
        let mut out = [0.0f32; 4];
        basis.eval_into(0.5, &mut out);
        // T0=1, T1=0.5, T2=2*0.5*0.5 - 1 = -0.5, T3 = 2*0.5*(-0.5) - 0.5 = -1.0.
        assert!(approx_eq(out[0], 1.0, 1e-5));
        assert!(approx_eq(out[1], 0.5, 1e-5));
        assert!(approx_eq(out[2], -0.5, 1e-5));
        assert!(approx_eq(out[3], -1.0, 1e-5));
    }

    #[test]
    fn bspline_partition_of_unity() {
        let basis: BSplineBasis<8> = BSplineBasis::new();
        let mut out = [0.0f32; 8];
        for x in (0..=100).map(|i| i as f32 / 100.0) {
            basis.eval_into(x, &mut out);
            let sum: f32 = out.iter().sum();
            assert!(approx_eq(sum, 1.0, 1e-3), "B-spline sum at x={} = {}", x, sum);
        }
    }

    #[test]
    fn feature_expand_layout() {
        // 2 coords, M=2 basis each → 4 features.
        let basis: ChebyshevBasis<2> = ChebyshevBasis::new();
        let delay = [0.5f32, 0.0];
        let mut out = [0.0f32; 4];
        feature_expand::<ChebyshevBasis<2>, 2>(&delay, &basis, &mut out);
        // coord 0 (x=0.5): T0=1, T1=0.5
        assert!(approx_eq(out[0], 1.0, 1e-5));
        assert!(approx_eq(out[1], 0.5, 1e-5));
        // coord 1 (x=0.0): T0=1, T1=0
        assert!(approx_eq(out[2], 1.0, 1e-5));
        assert!(approx_eq(out[3], 0.0, 1e-5));
    }

    #[test]
    fn forecaster_fits_and_forecasts_linear_map() {
        // Build a forecaster where the true map is û = 2·x_coord_0 (linear).
        // Use Chebyshev with M=2 so T0=1 (bias) and T1=x are present.
        type F = KarcForecaster<ChebyshevBasis<2>, 1, 2, 1>;
        let mut f: F = KarcForecaster::with_capacity(ChebyshevBasis::new(), 20);
        // 20 samples: u_t = x, u_{t+1} = 2x.
        for i in 0..20 {
            let x = (i as f32) * 0.1 - 1.0; // x in [-1, 0.9]
            let delay = [x];
            let target = [2.0 * x];
            f.accumulate_pair(&delay, &target);
        }
        f.fit_ridge(1e-6).unwrap();
        assert!(f.is_fitted());
        // Forecast at x=0.5 → expect ≈ 1.0.
        let mut out = [0.0f32];
        assert!(f.forecast_into(&[0.5], &mut out));
        assert!(
            approx_eq(out[0], 1.0, 1e-2),
            "forecast at x=0.5: {} (expected ~1.0)",
            out[0]
        );
    }

    #[test]
    fn forecaster_rejects_zero_lambda() {
        type F = KarcForecaster<ChebyshevBasis<2>, 1, 2, 1>;
        let mut f: F = KarcForecaster::with_capacity(ChebyshevBasis::new(), 4);
        f.accumulate_pair(&[0.0], &[0.0]);
        let err = f.fit_ridge(0.0).unwrap_err();
        assert_eq!(err, FitError::NonPositiveLambda);
    }

    #[test]
    fn forecaster_rejects_no_samples() {
        type F = KarcForecaster<ChebyshevBasis<2>, 1, 2, 1>;
        let mut f: F = KarcForecaster::with_capacity(ChebyshevBasis::new(), 4);
        let err = f.fit_ridge(1e-6).unwrap_err();
        assert_eq!(err, FitError::NoSamples);
    }
}
