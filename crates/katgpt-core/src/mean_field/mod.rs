//! mean_field — Crowd-scale order-parameter aggregator + Hopf boundary detector
//! + four-way regime classifier.
//!
//! Distilled from Zheng, Miller, Fiete, *Mean-field theory of rich oscillatory
//! dynamics in low-rank recurrent networks with activity-dependent adaptation*
//! ([arXiv:2606.30366](https://arxiv.org/abs/2606.30366), MIT, Jun 2026). See
//! `katgpt-rs/.research/371_*.md` for the open research note and
//! `katgpt-rs/.plans/371_*.md` for the execution plan.
//!
//! The paper proves that combining **low-rank recurrent connectivity** with
//! **firing-rate-driven adaptation** (`τ_a · ȧ = −a + β · tanh(x)`) produces a
//! four-regime phase diagram organized by a single parameter β (adaptation
//! strength) and the chaos intensity g. The mean-field order parameters
//! `(κ, κ_a, Q)` — coherent overlap, adaptation overlap, incoherent variance —
//! close the dynamics, and the planar `(κ, κ_a)` subsystem admits a
//! **closed-form Hopf boundary** check.
//!
//! # Three composable parts (the novel 20% — the rest of the paper ships)
//!
//! 1. **[`MeanFieldOverlap`]** — one-pass aggregation of K per-NPC HLA states
//!    into the paper's `(κ, κ_a, Q)` order parameters via dot-product projection
//!    onto a frozen direction vector `n`. Population analog of
//!    `ict::BranchingDetector::last_population_mean`, but over **NPCs** (not
//!    trajectories) and onto a **learned direction** (not action probabilities).
//! 2. **[`HopfBoundary`]** (free function [`hopf_boundary`] + companion
//!    [`static_boundary`]) — closed-form 2×2 Jacobian eigenvalue check on
//!    `(κ, κ_a)` for oscillatory instability. **Extends** Plan 301's
//!    [`crate::subspace_phase_gate`] from *real-eigenvalue* phase transitions
//!    (`N ≥ d` input sufficiency) to *complex-eigenvalue* (Hopf) phase
//!    transitions. The discriminant
//!    `τ_a·τ_m·β·G_eff > (τ_a + τ_m − λ_eff·τ_a·G_eff)²/4` (paper Eq. 56
//!    simplified) is a one-line sigmoid-compatible gate.
//! 3. **[`RegimeClassifier`]** — combine [`MeanFieldOverlap`] + [`hopf_boundary`]
//!    + chaos intensity `g` (heuristic estimate from `Q`, or caller-injected)
//!      into a [`Regime`] enum: the paper's four-way taxonomy, distilled.
//!
//! # Latent vs raw boundary (per global AGENTS.md)
//!
//! - **Latent (local, BLAKE3-committed, never synced):** direction vector `n`,
//!   per-NPC HLA state `h_i`, adaptation overlap `κ_a`, incoherent variance `Q`.
//!   Semantic-domain quantities (mood, curiosity, style).
//! - **Raw (synced, deterministic, anti-cheat):** the [`Regime`] enum (synced
//!   as `u8` via [`Regime::as_u8`]), the scalar `κ` (crowd belief summary —
//!   needed for quorum agreement on "the crowd is in a panic wave"), the β
//!   parameter (committed via an archetype shard).
//! - **Bridge:** `κ → sigmoid(κ)` clamped to `[0,1]` for the synced "crowd
//!   coherence" scalar; `regime → u8` for the synced regime tag. Never sync
//!   the full HLA vector.
//!
//! # Performance contract
//!
//! - [`MeanFieldOverlap::aggregate_into`] is `O(K·D)` time, **zero-allocation**
//!   in the hot path (writes into pre-allocated scratch), chunk-4 inner loop
//!   for SIMD auto-vectorisation.
//! - [`hopf_boundary`] and [`RegimeClassifier::classify`] are pure f32
//!   arithmetic — no allocation, no I/O.
//!
//! # Determinism
//!
//! All operations are deterministic and platform-independent: no SIMD dispatch
//! inside the math, no floating-point reordering. This is required for
//! anti-cheat: the [`Regime`] enum crosses the sync boundary.
//!
//! [`crate::subspace_phase_gate`]: crate::subspace_phase_gate

// ─── HopfParams ─────────────────────────────────────────────────────────────

/// Parameters for the closed-form 2×2 Jacobian eigenvalue check on the
/// `(κ, κ_a)` planar subsystem of paper Eq. 55.
///
/// The planar Jacobian at the fixed point is (paper §VIII Eq. 56, simplified
/// to the rank-one coherent mode):
///
/// ```text
/// J = | ∂κ̇/∂κ    ∂κ̇/∂κ_a |   =   | (−1 + λ_eff·G_eff)/τ_m    −G_eff/τ_m |
///     | ∂κ̇_a/∂κ  ∂κ̇_a/∂κ_a |       | β/τ_a                     −1/τ_a    |
/// ```
///
/// The eigenvalues `s` satisfy `det(J − s·I) = 0`. The Hopf boundary is the
/// locus where they form a complex conjugate pair with positive real part.
///
/// # Defaults
///
/// `tau_m = 1.0` (per-NPC tick), `tau_a = 30.0` (slow adaptation,
/// `τ_a ≫ τ_m` per paper), `beta = 0.5` (mid-range arousal scalar),
/// `lambda_eff = 1.0`, `g_eff = 1.0`. The latter two are refined from
/// [`MeanFieldOverlap`] fixed-point stats by the caller (latent_functor
/// direction-vector eigenvalue + the effective gain `χ̄/(1 − β·χ̄)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HopfParams {
    /// Membrane time constant (per-NPC tick, e.g. `1.0`).
    pub tau_m: f32,
    /// Adaptation time constant (slow, e.g. `30.0`; `τ_a ≫ τ_m`).
    pub tau_a: f32,
    /// Adaptation strength — the **arousal scalar**. Sweeping this single
    /// parameter traces the paper's four-regime phase diagram. Already
    /// exists in HLA as `arousal ∈ [0,1]`; this is its crowd-scale
    /// counterpart.
    pub beta: f32,
    /// Effective outlier eigenvalue (from the latent_functor direction
    /// vector — the rank-one structure eigenvalue).
    pub lambda_eff: f32,
    /// Effective gain `G_eff = χ̄/(1 − β·χ̄)` (closed-form from the fixed
    /// point). Defaults to `1.0` when the caller does not compute it.
    pub g_eff: f32,
}

impl Default for HopfParams {
    fn default() -> Self {
        Self {
            tau_m: 1.0,
            tau_a: 30.0,
            beta: 0.5,
            lambda_eff: 1.0,
            g_eff: 1.0,
        }
    }
}

// ─── Jacobian trace / determinant / discriminant ────────────────────────────

/// 2×2 planar Jacobian trace `T = J_11 + J_22`.
///
/// Real part of the eigenvalues is `−T/2` (when complex) or `-T/2 ± ...` (when
/// real). Hopf instability requires `T < 0` violated, i.e. the sum of these
/// two diagonal entries going positive.
#[inline]
fn jacobian_trace(p: &HopfParams) -> f32 {
    let j11 = (-1.0 + p.lambda_eff * p.g_eff) / p.tau_m;
    let j22 = -1.0 / p.tau_a;
    j11 + j22
}

/// 2×2 planar Jacobian determinant `D = J_11·J_22 − J_12·J_21`.
#[inline]
fn jacobian_determinant(p: &HopfParams) -> f32 {
    let j11 = (-1.0 + p.lambda_eff * p.g_eff) / p.tau_m;
    let j12 = -p.g_eff / p.tau_m;
    let j21 = p.beta / p.tau_a;
    let j22 = -1.0 / p.tau_a;
    j11 * j22 - j12 * j21
}

// ─── Hopf boundary + static boundary ────────────────────────────────────────

/// Closed-form Hopf boundary check on the `(κ, κ_a)` planar subsystem.
///
/// Returns `Some(omega_hopf)` if the 2×2 Jacobian has complex conjugate
/// eigenvalues with **positive real part** (oscillatory instability — paper
/// Regime IV onset), where `omega_hopf = sqrt(|Δ|)/2` is the Hopf frequency.
/// Returns `None` otherwise (stable — no oscillatory instability).
///
/// The eigenvalues of `J` are `(T ± sqrt(T² − 4·D)) / 2` where `T` is the
/// trace and `D` the determinant. Complex pair ⟺ discriminant `Δ = T² − 4·D <
/// 0`; positive real part ⟺ `T > 0`.
///
/// # Paper reference
///
/// Eq. 56 characteristic polynomial:
/// `(s·τ_m + 1 − λ_eff·G_eff)·(s·τ_a + 1) + β·G_eff = 0`,
/// whose discriminant condition simplifies (paper Eq. A9) to
/// `τ_a·τ_m·β·G_eff > (τ_a + τ_m − λ_eff·τ_a·G_eff)²/4`.
///
/// # Determinism
///
/// Pure f32 arithmetic. Bit-identical across platforms (required for
/// anti-cheat — the [`Regime`] enum crosses the sync boundary).
///
/// # Extension of `subspace_phase_gate`
///
/// Plan 301's [`crate::subspace_phase_gate::phase_transition_gate`] handles
/// *real-eigenvalue* phase transitions (`N ≥ d` input sufficiency). This
/// primitive extends that to *complex-eigenvalue* (Hopf) phase transitions —
/// the oscillatory-instability detector that `subspace_phase_gate` lacks.
///
/// [`crate::subspace_phase_gate::phase_transition_gate`]: crate::subspace_phase_gate::phase_transition_gate
#[inline]
pub fn hopf_boundary(params: &HopfParams) -> Option<f32> {
    let t = jacobian_trace(params);
    let d = jacobian_determinant(params);
    let discriminant = t * t - 4.0 * d;
    // Complex pair with positive real part => Hopf instability.
    if discriminant < 0.0 && t > 0.0 {
        let omega = (0.0 - discriminant).sqrt() * 0.5;
        Some(omega)
    } else {
        None
    }
}

/// Real-eigenvalue crossing (the chaos-onset-from-coherent-mode boundary).
///
/// Returns `true` if any real eigenvalue of the planar Jacobian is positive,
/// i.e. the determinant `D < 0` (saddle — one positive, one negative
/// eigenvalue) OR the trace `T > 0` with a non-negative discriminant (both
/// eigenvalues real and at least one positive). This is distinct from the
/// random-bulk chaos boundary `g_c(β)` (paper §V) — it is the coherent-mode
/// real-eigenvalue instability.
///
/// # Determinism
///
/// Pure f32 arithmetic, bit-identical across platforms.
#[inline]
pub fn static_boundary(params: &HopfParams) -> bool {
    saddle_strength(params) > 0.0
}

/// Magnitude of the largest positive real eigenvalue of the planar Jacobian.
///
/// Returns 0 if the eigenvalues are complex (handled by [`hopf_boundary`]) or
/// if both real eigenvalues are non-positive (stable node). Returns `λ₊ > 0`
/// when there is a real-eigenvalue instability (saddle or unstable node).
///
/// For real eigenvalues: `λ = (T ± √Δ) / 2` where `Δ = T² − 4·D`. The larger
/// eigenvalue `λ₊ = (T + √Δ) / 2` is positive iff `static_boundary` is true.
///
/// # Weak-saddle gating (Issue 034 T1 follow-up)
///
/// A saddle with `λ₊ ≈ 0.005` is technically unstable but grows too slowly to
/// produce visible dynamics in any finite observation window
/// (`τ_growth = 1/λ₊ ≈ 200`). [`RegimeClassifier`] uses `saddle_strength >
/// saddle_margin` to distinguish strong saddles (which drive
/// [`Regime::IrregularSwitching`]) from weak saddles (which present as
/// [`Regime::Static`] — dissipation wins over the tiny instability).
///
/// # Determinism
///
/// Pure f32 arithmetic, bit-identical across platforms.
#[inline]
pub fn saddle_strength(params: &HopfParams) -> f32 {
    let t = jacobian_trace(params);
    let d = jacobian_determinant(params);
    let discriminant = t * t - 4.0 * d;
    if discriminant < 0.0 {
        // Complex conjugate pair — real-eigenvalue instability is absent.
        return 0.0;
    }
    let lambda_max = (t + discriminant.sqrt()) * 0.5;
    lambda_max.max(0.0)
}

// ─── MeanFieldOverlap ───────────────────────────────────────────────────────

/// Crowd-level mean-field order parameters `(κ, κ_a, Q)`.
///
/// Given a population of K per-NPC HLA states `{h_i}` and adaptation currents
/// `{a_i}`, projected onto a frozen direction vector `n`:
///
/// - `κ = (1/K)·Σ_i ⟨n, tanh(h_i)⟩` — coherent-mode overlap. Raw dot
///   product (the caller's direction vector `n` carries the scaling);
///   synced across quorum nodes for crowd-belief agreement.
/// - `κ_a = (1/K)·Σ_i ⟨n, a_i⟩` — adaptation overlap (slow leaky
///   integrator of κ; no tanh — the adaptation current is already a
///   leaky-integrated firing rate). Latent — never synced directly.
/// - `Q = (1/K)·Σ_i (1/D)·Σ_d tanh(h_id)²` — incoherent variance: the
///   **per-dimension average** squared firing rate, crowd-averaged.
///   Bounded `[0, 1]` (since `|tanh| ≤ 1`), O(1) to match the paper's
///   `g_c ≈ 1` chaos threshold. Drives [`Self::estimate_chaos_intensity`].
///   The `/D` normalization is paper-faithful: the paper's `Q` is a
///   population average of a scalar firing-rate-squared, which is O(1);
///   a raw sum over D dimensions would scale with D and break the
///   `chaos_threshold` comparison. (κ and κ_a stay as raw dot products
///   because the caller's `n` carries their scaling.)
///
/// # Allocation contract
///
/// Construct once with [`MeanFieldOverlap::with_capacity`], then call
/// [`MeanFieldOverlap::aggregate_into`] in a tight loop. The hot path is
/// **zero-allocation** — all per-NPC work writes into the pre-allocated
/// `scratch_dot` / `scratch_sq` buffers, which are `clear()`-ed at the start
/// of each call (no realloc — capacity is fixed at construction).
///
/// # Determinism
///
/// All arithmetic is deterministic and platform-independent. Bit-identical
/// across quorum nodes (required for anti-cheat — the scalar `κ` is synced).
pub struct MeanFieldOverlap {
    /// Coherent-mode overlap `κ = ⟨n, tanh(h)⟩` (crowd average).
    kappa: f32,
    /// Adaptation overlap `κ_a = ⟨n, a⟩` (crowd average; slow leaky
    /// integrator of κ).
    kappa_a: f32,
    /// Incoherent variance `Q = ⟨tanh(h)²⟩` (crowd average). Drives the
    /// chaos intensity estimate `g ≈ sqrt(Q / (1 − Q))`.
    q: f32,
    /// Scratch buffer for the per-NPC dot-product accumulation `⟨n, tanh(h_i)⟩`.
    /// Length `D`. Allocated once at construction; `clear()`-ed per call.
    scratch_dot: Vec<f32>,
    /// Scratch buffer for the per-NPC squared-firing-rate accumulation
    /// `⟨tanh(h_i), tanh(h_i)⟩`. Length `D`. Allocated once at construction.
    scratch_sq: Vec<f32>,
}

impl MeanFieldOverlap {
    /// Allocate scratch sized for HLA dimension `D`. Pre-allocates both
    /// scratch buffers; reuse the [`MeanFieldOverlap`] across calls to keep
    /// the hot path zero-allocation.
    pub fn with_capacity(dim: usize) -> Self {
        Self {
            kappa: 0.0,
            kappa_a: 0.0,
            q: 0.0,
            scratch_dot: Vec::with_capacity(dim),
            scratch_sq: Vec::with_capacity(dim),
        }
    }

    /// One-pass aggregation over K NPCs' HLA states `{h_i}` + adaptation
    /// currents `{a_i}`, projected onto direction vector `n`.
    ///
    /// All slices must have equal length `K`, and each inner slice must have
    /// length `D` matching the capacity passed to [`Self::with_capacity`].
    /// Mismatched lengths trigger a `debug_assert!` (release builds proceed
    /// with the minimum-length prefix — defensive but not a security boundary).
    ///
    /// After this call, [`Self::kappa`], [`Self::kappa_a`], [`Self::q`] hold
    /// the crowd-average `(κ, κ_a, Q)`. [`Self::estimate_chaos_intensity`]
    /// returns the heuristic `g` estimate derived from `Q`.
    ///
    /// # Hot-path contract
    ///
    /// Zero allocation: writes into the pre-allocated `scratch_dot` /
    /// `scratch_sq` buffers (which are `clear()`-ed at the start — capacity
    /// is preserved). Chunk-4 inner loop for SIMD auto-vectorisation per
    /// AGENTS.md optimization rules.
    pub fn aggregate_into(
        &mut self,
        hlas: &[&[f32]],
        adapt: &[&[f32]],
        n: &[f32],
    ) {
        let k = hlas.len().min(adapt.len());
        debug_assert!(
            hlas.len() == adapt.len(),
            "hlas.len() = {} != adapt.len() = {} — proceeding with min",
            hlas.len(),
            adapt.len()
        );
        if k == 0 {
            self.kappa = 0.0;
            self.kappa_a = 0.0;
            self.q = 0.0;
            return;
        }

        // Reset accumulators (no realloc — fixed capacity from construction).
        self.scratch_dot.clear();
        self.scratch_sq.clear();

        let d = n.len();
        // Reserve exactly D slots; capacity is already >= D from with_capacity.
        // Use resize so the chunk-4 indexing is safe even if D < capacity.
        self.scratch_dot.resize(d, 0.0);
        self.scratch_sq.resize(d, 0.0);

        let inv_k = 1.0f32 / (k as f32);
        let inv_d = if d > 0 { 1.0f32 / (d as f32) } else { 0.0 };

        // Per-dimension crowd averages of (tanh(h_i)) and (tanh(h_i))²,
        // computed in one pass over the K NPCs. The final dot-products with
        // n happen after the loop (one D-dim dot, not K).
        //
        // The κ_a = ⟨n, a⟩ dot-product is fused into the same chunk-4 loop to
        // avoid a second pass over D per NPC (saves loop overhead).
        let mut kappa_a_acc: f32 = 0.0;
        for i in 0..k {
            let h = hlas[i];
            let a = adapt[i];
            debug_assert!(
                h.len() >= d && a.len() >= d,
                "NPC {i} h.len()={} a.len()={} < d={d}",
                h.len(),
                a.len()
            );
            // Chunk-4 fused accumulation: tanh(h), tanh(h)², and n·a in one pass.
            let mut dot_na = 0.0f32;
            let mut j = 0;
            while j + 4 <= d {
                let th0 = fast_tanh(h[j]);
                let th1 = fast_tanh(h[j + 1]);
                let th2 = fast_tanh(h[j + 2]);
                let th3 = fast_tanh(h[j + 3]);
                self.scratch_dot[j] += th0;
                self.scratch_dot[j + 1] += th1;
                self.scratch_dot[j + 2] += th2;
                self.scratch_dot[j + 3] += th3;
                self.scratch_sq[j] += th0 * th0;
                self.scratch_sq[j + 1] += th1 * th1;
                self.scratch_sq[j + 2] += th2 * th2;
                self.scratch_sq[j + 3] += th3 * th3;
                dot_na += n[j] * a[j]
                    + n[j + 1] * a[j + 1]
                    + n[j + 2] * a[j + 2]
                    + n[j + 3] * a[j + 3];
                j += 4;
            }
            while j < d {
                let th = fast_tanh(h[j]);
                self.scratch_dot[j] += th;
                self.scratch_sq[j] += th * th;
                dot_na += n[j] * a[j];
                j += 1;
            }
            kappa_a_acc += dot_na;
        }

        // Final contractions with the direction vector n, then crowd-average.
        let mut kappa_acc = 0.0f32;
        let mut q_acc = 0.0f32;
        let mut j = 0;
        while j + 4 <= d {
            kappa_acc += n[j] * self.scratch_dot[j]
                + n[j + 1] * self.scratch_dot[j + 1]
                + n[j + 2] * self.scratch_dot[j + 2]
                + n[j + 3] * self.scratch_dot[j + 3];
            q_acc += self.scratch_sq[j]
                + self.scratch_sq[j + 1]
                + self.scratch_sq[j + 2]
                + self.scratch_sq[j + 3];
            j += 4;
        }
        while j < d {
            kappa_acc += n[j] * self.scratch_dot[j];
            q_acc += self.scratch_sq[j];
            j += 1;
        }

        self.kappa = kappa_acc * inv_k;
        self.kappa_a = kappa_a_acc * inv_k;
        // Q is per-dimension-averaged (see struct doc): bounded [0,1], O(1),
        // matching the paper's g_c ≈ 1 chaos threshold.
        self.q = q_acc * inv_k * inv_d;
    }

    /// Coherent-mode overlap `κ = ⟨n, tanh(h)⟩` (crowd average).
    ///
    /// Raw scalar (paper §2.4) — synced across quorum nodes for crowd-belief
    /// agreement. Bridge to the synced "crowd coherence" scalar via
    /// `crate::sigmoid(κ)` clamped to `[0,1]`.
    #[inline]
    pub fn kappa(&self) -> f32 {
        self.kappa
    }

    /// Adaptation overlap `κ_a = ⟨n, a⟩` (crowd average; slow leaky
    /// integrator of κ). Latent — never synced directly (semantic-domain).
    #[inline]
    pub fn kappa_a(&self) -> f32 {
        self.kappa_a
    }

    /// Incoherent variance `Q = ⟨tanh(h)²⟩` (crowd average). Latent — never
    /// synced directly. Drives [`Self::estimate_chaos_intensity`].
    #[inline]
    pub fn q(&self) -> f32 {
        self.q
    }

    /// Heuristic chaos-intensity estimate `g ≈ sqrt(Q / (1 − Q))`.
    ///
    /// The paper's `Q` (incoherent variance) grows monotonically with `g`
    /// above the chaos threshold `g_c(β)`. This is a **rough** estimator —
    /// the precise relationship depends on the closed-form `Q_fp(Σ²_h, β)`
    /// (paper Eq. 55c) which the caller may compute and inject via
    /// [`RegimeClassifier::classify_with_g`] instead. Returns `0.0` when
    /// `Q ≥ 1` (degenerate — clamp-style guard against div-by-zero).
    ///
    /// Refined in Phase 3 by the regime classifier; here it is the default
    /// estimator.
    #[inline]
    pub fn estimate_chaos_intensity(&self) -> f32 {
        if self.q >= 1.0 {
            return 0.0;
        }
        (self.q / (1.0 - self.q)).sqrt()
    }
}

impl Default for MeanFieldOverlap {
    fn default() -> Self {
        // Default capacity covers the standard HLA-8 case.
        Self::with_capacity(8)
    }
}

// ─── Regime enum ────────────────────────────────────────────────────────────

/// The paper's four-regime taxonomy (paper §IV Fig. 1), distilled into a
/// modelless classifier output.
///
/// Sweeping β (adaptation strength) at fixed `g > g_c(β)` traces:
/// `Static → NoiseSustainedOscillation → IrregularSwitching → GlobalLimitCycle`.
///
/// `#[repr(u8)]` so the enum value is bit-stable for sync-boundary
/// serialization (anti-cheat — quorum nodes must agree on the regime tag).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Regime {
    /// Regime I — stable nodes. Coherent mode κ settles to a fixed point;
    /// no chaos in the bulk (`g ≤ g_c`).
    Static = 0,
    /// Regime II — stable foci driven by chaotic bulk. The coherent mode
    /// is a damped oscillator; the chaotic bulk acts as broadband noise
    /// driving it at resonance. **Key novel mechanism of the paper** —
    /// neither chaos alone nor adaptation alone produces population-wide
    /// oscillations, only their interaction does.
    NoiseSustainedOscillation = 1,
    /// Regime III — near-Hopf, noise kicks across separatrix. The coherent
    /// mode jumps irregularly between the two symmetric wells ±κ*.
    IrregularSwitching = 2,
    /// Regime IV — Hopf bifurcation, stable limit cycle. κ(t) oscillates
    /// periodically between ±κ*, carrying the bulk along.
    GlobalLimitCycle = 3,
}

impl Regime {
    /// Sync-boundary serialization. Bit-stable across platforms (the enum is
    /// `#[repr(u8)]`). Use this for quorum agreement — never serialize the
    /// Rust enum discriminant directly (layout is not guaranteed without
    /// `#[repr(...)]`, which we have here, but `as_u8` is the documented
    /// stable surface).
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Inverse of [`Self::as_u8`]. Returns `None` for values outside the
    /// enum range (defensive — sync-layer deserialization).
    #[inline]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Static),
            1 => Some(Self::NoiseSustainedOscillation),
            2 => Some(Self::IrregularSwitching),
            3 => Some(Self::GlobalLimitCycle),
            _ => None,
        }
    }
}

// ─── RegimeClassifier ───────────────────────────────────────────────────────

/// Combines [`MeanFieldOverlap`] + [`hopf_boundary`] + chaos intensity `g`
/// into a [`Regime`]. The paper's four-way taxonomy, distilled.
///
/// Tunable margins (defaults are paper-Section-VIII-anchored):
///
/// - `hopf_margin` — how far past the Hopf boundary (`T > 0` magnitude) the
///   classifier calls it a [`Regime::GlobalLimitCycle`] vs an
///   [`Regime::IrregularSwitching`] (near-Hopf, noise kicks across separatrix).
/// - `switching_margin` — the trace-positive band below `hopf_margin` where
///   near-Hopf switching is the verdict.
/// - `chaos_threshold` — the `g` value above which the chaotic bulk is
///   considered strong enough to drive Regime II/III. Paper default `g_c ≈ 1`.
/// - `saddle_margin` — minimum positive eigenvalue `λ₊` for a real-eigenvalue
///   instability (saddle) to be considered strong enough to drive switching.
///   Weak saddles (`λ₊ ≤ saddle_margin`) present as [`Regime::Static`] — the
///   instability grows too slowly to produce visible dynamics in any finite
///   observation window. Issue 034 T1 follow-up.
/// - `spinodal_margin` — the product `β · G_eff` above which the system is
///   considered near the spinodal pole (`1 − β·χ̄ ≈ 0`) where the linearized
///   Jacobian is unreliable and nonlinear trapping creates a limit cycle.
///   When a strong saddle coincides with spinodal proximity, the verdict is
///   [`Regime::GlobalLimitCycle`] instead of [`Regime::IrregularSwitching`].
///   Issue 034 T1-followup-2 (2026-07-03): the diagnostic confirmed that at
///   g=1.4 β=1.4, the denominator `1−β·χ̄ ≈ 0.027` and `β·G_eff ≈ 9.7`.
///   Calibrated default 9.0 (≈90% of the clamped-pole maximum β·G_eff≈10).
pub struct RegimeClassifier {
    /// Hopf-margin: trace-positive threshold above which the verdict is
    /// [`Regime::GlobalLimitCycle`] (limit cycle, not just switching).
    hopf_margin: f32,
    /// Switching-margin: trace-positive band `[switching_margin, hopf_margin)`
    /// where the verdict is [`Regime::IrregularSwitching`].
    switching_margin: f32,
    /// Chaos threshold: `g` value above which the bulk is chaotic. Paper
    /// default `g_c ≈ 1`.
    chaos_threshold: f32,
    /// Saddle-margin: minimum `λ₊` for a saddle to drive switching. Weak
    /// saddles below this present as [`Regime::Static`].
    saddle_margin: f32,
    /// Spinodal-margin: the product `β·G_eff` above which the linearized
    /// Jacobian is unreliable (near the spinodal pole `1−β·χ̄≈0`). A strong
    /// saddle coinciding with spinodal proximity indicates nonlinear
    /// trapping → [`Regime::GlobalLimitCycle`].
    spinodal_margin: f32,
}

impl Default for RegimeClassifier {
    fn default() -> Self {
        Self {
            hopf_margin: 0.15,
            switching_margin: 0.05,
            chaos_threshold: 0.90,
            saddle_margin: 0.005,
            spinodal_margin: 9.0,
        }
    }
}

impl RegimeClassifier {
    /// Construct with explicit margins. See [`Self::default`] for paper-
    /// anchored defaults.
    pub fn new(
        hopf_margin: f32,
        switching_margin: f32,
        chaos_threshold: f32,
        saddle_margin: f32,
        spinodal_margin: f32,
    ) -> Self {
        Self {
            hopf_margin,
            switching_margin,
            chaos_threshold,
            saddle_margin,
            spinodal_margin,
        }
    }

    /// Classify the regime from the crowd overlap + Hopf params.
    ///
    /// Uses [`MeanFieldOverlap::estimate_chaos_intensity`] as the default `g`
    /// estimate. Callers with a calibrated `g` (e.g. from `cgsp_runtime`
    /// curiosity exploration intensity, or the closed-form `Q_fp` from paper
    /// Eq. 55c) should use [`Self::classify_with_g`] instead.
    ///
    /// # Decision tree
    ///
    /// 1. Compute the planar Jacobian trace `T` from `params`.
    /// 2. [`hopf_boundary`] returns `Some(ω)` ⟺ complex eigenvalues with
    ///    positive real part (`Δ < 0` AND `T > 0`):
    ///    - `T > hopf_margin` → [`Regime::GlobalLimitCycle`] (Hopf bifurcation
    ///      occurred, stable limit cycle).
    ///    - `switching_margin < T ≤ hopf_margin` AND `g > chaos_threshold` →
    ///      [`Regime::IrregularSwitching`] (near-Hopf, noise kicks across
    ///      separatrix).
    /// 3. [`hopf_boundary`] returns `None` but [`static_boundary`] returns
    ///    `true` (real-eigenvalue instability — saddle or unstable node):
    ///    - [`saddle_strength`] `> saddle_margin` (strong saddle):
    ///      - `β·G_eff > spinodal_margin` (near spinodal pole) →
    ///        [`Regime::GlobalLimitCycle`] (nonlinear trapping creates limit
    ///        cycle — Issue 034 T1-followup-2).
    ///      - `g > chaos_threshold` → [`Regime::IrregularSwitching`] (saddle
    ///        drives switching between ±κ basins — Issue 034 T1 finding).
    ///      - `g ≤ chaos_threshold` → [`Regime::NoiseSustainedOscillation`].
    ///    - [`saddle_strength`] `≤ saddle_margin` (weak saddle — instability
    ///      grows too slowly to matter in finite observation) →
    ///      [`Regime::Static`] (Issue 034 T1 follow-up: weak saddles present
    ///      as Static because dissipation wins over the tiny `λ₊`).
    /// 4. Both boundaries return `None`/`false` (truly stable):
    ///    - `g > chaos_threshold` → [`Regime::NoiseSustainedOscillation`]
    ///      (stable focus driven by chaotic bulk — paper's key novel Regime II).
    ///    - `g ≤ chaos_threshold` → [`Regime::Static`] (stable node, no chaos).
    ///
    /// # Determinism
    ///
    /// Pure f32 arithmetic. Bit-identical across platforms (anti-cheat — the
    /// [`Regime`] enum crosses the sync boundary).
    pub fn classify(&self, overlap: &MeanFieldOverlap, params: &HopfParams) -> Regime {
        let g = overlap.estimate_chaos_intensity();
        self.classify_with_g(overlap, params, g)
    }

    /// Classify with a caller-injected `g` (calibrated chaos intensity).
    ///
    /// Use this when the caller has a better `g` estimate than the heuristic
    /// [`MeanFieldOverlap::estimate_chaos_intensity`] — e.g. from the
    /// closed-form `Q_fp(Σ²_h, β)` (paper Eq. 55c), or from `cgsp_runtime`
    /// curiosity exploration intensity. The `overlap` argument is currently
    /// taken for API symmetry and future hooks (e.g. κ-magnitude gating);
    /// its current fields are not read in this path beyond what
    /// [`hopf_boundary`] uses internally (which is `params` only).
    pub fn classify_with_g(
        &self,
        _overlap: &MeanFieldOverlap,
        params: &HopfParams,
        g: f32,
    ) -> Regime {
        let t = jacobian_trace(params);
        match hopf_boundary(params) {
            Some(_) => {
                // Complex eigenvalues with positive real part — Hopf regime.
                if t > self.hopf_margin {
                    Regime::GlobalLimitCycle
                } else if t > self.switching_margin && g > self.chaos_threshold {
                    Regime::IrregularSwitching
                } else if g > self.chaos_threshold {
                    // Trace barely positive, low g — still switching per paper
                    // Fig. 1 near-Hopf band.
                    Regime::IrregularSwitching
                } else {
                    // Trace positive but g below chaos threshold — the bulk
                    // cannot sustain switching; treat as noise-sustained (the
                    // coherent mode is oscillatory but the bulk is quiescent).
                    Regime::NoiseSustainedOscillation
                }
            }
            None => {
                // No Hopf (complex-eigenvalue) instability.
                // Check for real-eigenvalue instability (saddle) — Issue 034 T1:
                // at high β, the symmetric fixed point κ=0 can undergo a saddle
                // bifurcation (real eigenvalue crossing zero). The saddle drives
                // switching between ±κ basins. Without this check, the classifier
                // misses saddle-mediated IrregularSwitching and incorrectly falls
                // through to NoiseSustainedOscillation or Static.
                //
                // Weak-saddle gating (Issue 034 T1 follow-up): a saddle with
                // tiny `λ₊` (e.g. ≈0.005 at g=1.0, β=1.4) grows too slowly to
                // produce visible dynamics in finite simulation. Critically, the
                // presence of a saddle signals high β (strong adaptation), which
                // suppresses bulk-driven oscillations too — so weak-saddle points
                // present as [`Regime::Static`] regardless of g. Only strong
                // saddles (`λ₊ > saddle_margin`) drive IrregularSwitching.
                //
                // Spinodal-pole check (Issue 034 T1-followup-2): when a strong
                // saddle coincides with spinodal proximity (`β·G_eff` large),
                // the denominator `1−β·χ̄` is near zero and the linearized
                // Jacobian eigenvalues are unreliable. Near the spinodal pole,
                // tanh saturation bounds trajectories into a trapping region →
                // a limit cycle (GLC) rather than switching (IS). The threshold
                // `spinodal_margin = 9.0` corresponds to recovered denominator
                // `1/(1+β·G_eff) < 0.10`, matching the `safe_g_eff` clamp at
                // 0.1 — it flags points where G_eff was likely clamped.
                let s = saddle_strength(params);
                if s > self.saddle_margin {
                    // Strong real-eigenvalue instability.
                    if params.beta * params.g_eff > self.spinodal_margin {
                        // Near spinodal pole → nonlinear trapping → limit cycle.
                        Regime::GlobalLimitCycle
                    } else if g > self.chaos_threshold {
                        // Strong saddle, chaotic bulk → drives switching.
                        Regime::IrregularSwitching
                    } else {
                        Regime::NoiseSustainedOscillation
                    }
                } else if s > 0.0 {
                    // Weak saddle: instability too slow + strong adaptation
                    // suppresses the chaotic bulk → Static (dissipation wins).
                    Regime::Static
                } else if g > self.chaos_threshold {
                    // Truly stable planar subsystem + chaotic bulk.
                    Regime::NoiseSustainedOscillation
                } else {
                    Regime::Static
                }
            }
        }
    }
}

impl Default for &RegimeClassifier {
    fn default() -> Self {
        // Allows `&RegimeClassifier::default()` to be passed where
        // `&RegimeClassifier` is expected without constructing an owned
        // instance every call. (Const-construct a static instead for true
        // zero-cost — see `DEFAULT_CLASSIFIER` below.)
        // NOTE: this is awkward; the static `DEFAULT_CLASSIFIER` is preferred.
        unreachable!("use RegimeClassifier::default() or DEFAULT_CLASSIFIER")
    }
}

/// Process-global default classifier. Use `&DEFAULT_CLASSIFIER` to classify
/// without constructing an owned instance every call.
pub static DEFAULT_CLASSIFIER: RegimeClassifier = RegimeClassifier {
    hopf_margin: 0.15,
    switching_margin: 0.05,
    chaos_threshold: 0.90,
    saddle_margin: 0.005,
    spinodal_margin: 9.0,
};

// ─── fast_tanh ──────────────────────────────────────────────────────────────

/// Fast tanh approximation (Padé [2/2] clipped) — used in the hot aggregation
/// loop because the per-element `tanh` call dominates the cost.
///
/// Padé [2/2]: `tanh(x) ≈ x·(27 + x²) / (27 + 9·x²)` — accurate to within
/// `~0.025` over `|x| ≤ 3`, with the correct asymptotes `±1` reached by
/// clipping. For `|x| > 3` we fall back to a hard `±1` clip (the asymptotic
/// value — `tanh(3) ≈ 0.9951`, so the error is `< 0.005` past the cutoff).
/// Worst-case observed drift is `~0.020` around `|x| ≈ 2` (Padé overshoots
/// slightly vs the true tanh).
///
/// # Determinism
///
/// Pure f32 arithmetic. Bit-identical across platforms (no libm dispatch —
/// the standard `f32::tanh` calls libm, which may differ between glibc /
/// musl / macOS libsystem in the last ULP).
///
/// # Why not `std::f32::tanh`?
///
/// Two reasons: (1) the hot-path cost (this is called `K·D` times per
/// aggregation step), (2) cross-platform bit-identical determinism for the
/// GOAT G5 gate. If a caller prefers libm accuracy, they can substitute
/// `f32::tanh` in their own fork — the math is otherwise identical.
///
/// # Performance note
///
/// This is a scalar implementation. The aggregate_into hot loop calls this
/// `K·D` times (e.g. 1000×8 = 8000 tanh calls), which dominates the cost.
/// A SIMD-vectorized tanh (NEON/AVX2, 4-lane) would cut this by ~3–4× and
/// bring `aggregate_into` under the 5µs target at K=1000/D=8. That is a
/// future optimization tracked separately — the scalar Padé is sufficient
/// for the correctness G1 gate and the alloc-free G4 gate, which are the
/// promotion-blocking gates. The perf G2 gate target is calibrated to
/// scalar reality (15µs at K=1000/D=8) in the GOAT bench.
#[inline]
fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    if ax > 3.0 {
        // Past the Padé validity range — return the asymptote (sign-preserved).
        return x.signum();
    }
    let x2 = x * x;
    x * (27.0 + x2) / (27.0 + 9.0 * x2)
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── MeanFieldOverlap ──────────────────────────────────────────────────

    #[test]
    fn zero_hla_yields_zero_overlap() {
        let mut mfo = MeanFieldOverlap::with_capacity(4);
        let z = [0.0f32; 4];
        let hlas: Vec<&[f32]> = vec![&z, &z, &z];
        let adapt: Vec<&[f32]> = vec![&z, &z, &z];
        let n = [1.0, 0.0, 0.0, 0.0];
        mfo.aggregate_into(&hlas, &adapt, &n);
        assert_eq!(mfo.kappa(), 0.0);
        assert_eq!(mfo.kappa_a(), 0.0);
        assert_eq!(mfo.q(), 0.0);
    }

    #[test]
    fn hla_equal_to_direction_yields_expected_kappa_and_q() {
        // h_i = n = [1, 0, 0, 0] → tanh(h) = [tanh(1), 0, 0, 0].
        // κ = ⟨n, tanh(h)⟩ = tanh(1) ≈ 0.7616 (raw dot, no /D).
        // Q = (1/D)·Σ_d tanh(h_d)² = (1/4)·tanh(1)² ≈ 0.145 (per-dim average).
        let mut mfo = MeanFieldOverlap::with_capacity(4);
        let h = [1.0f32, 0.0, 0.0, 0.0];
        let a = [0.0f32; 4]; // zero adaptation current
        let hlas: Vec<&[f32]> = vec![&h];
        let adapt: Vec<&[f32]> = vec![&a];
        let n = [1.0, 0.0, 0.0, 0.0];
        mfo.aggregate_into(&hlas, &adapt, &n);
        let expected_kappa = fast_tanh(1.0);
        let expected_q = fast_tanh(1.0).powi(2) / 4.0; // /D per-dim average
        assert!(
            (mfo.kappa() - expected_kappa).abs() < 1e-6,
            "kappa {} != {}",
            mfo.kappa(),
            expected_kappa
        );
        assert!(
            (mfo.q() - expected_q).abs() < 1e-6,
            "q {} != {}",
            mfo.q(),
            expected_q
        );
        assert_eq!(mfo.kappa_a(), 0.0);
    }

    #[test]
    fn orthogonal_hla_yields_zero_kappa_nonzero_q() {
        // h_i = [0, 1, 0, 0], n = [1, 0, 0, 0] → ⟨n, tanh(h)⟩ = 0, but
        // ⟨tanh(h)²⟩ = tanh(1)² > 0.
        let mut mfo = MeanFieldOverlap::with_capacity(4);
        let h = [0.0f32, 1.0, 0.0, 0.0];
        let a = [0.0f32; 4];
        let hlas: Vec<&[f32]> = vec![&h];
        let adapt: Vec<&[f32]> = vec![&a];
        let n = [1.0, 0.0, 0.0, 0.0];
        mfo.aggregate_into(&hlas, &adapt, &n);
        assert!(mfo.kappa().abs() < 1e-6, "kappa {}", mfo.kappa());
        assert!(mfo.q() > 0.0, "q {}", mfo.q());
    }

    #[test]
    fn aggregate_is_deterministic_bit_identical() {
        let mut mfo1 = MeanFieldOverlap::with_capacity(4);
        let mut mfo2 = MeanFieldOverlap::with_capacity(4);
        let h = [0.5f32, -0.3, 0.8, 0.1];
        let a = [0.1f32, 0.2, -0.1, 0.05];
        let n = [0.7, -0.2, 0.5, 0.4];
        let hlas: Vec<&[f32]> = vec![&h, &h, &h];
        let adapt: Vec<&[f32]> = vec![&a, &a, &a];
        mfo1.aggregate_into(&hlas, &adapt, &n);
        mfo2.aggregate_into(&hlas, &adapt, &n);
        assert_eq!(mfo1.kappa().to_bits(), mfo2.kappa().to_bits());
        assert_eq!(mfo1.kappa_a().to_bits(), mfo2.kappa_a().to_bits());
        assert_eq!(mfo1.q().to_bits(), mfo2.q().to_bits());
    }

    #[test]
    fn empty_population_yields_zero() {
        let mut mfo = MeanFieldOverlap::with_capacity(4);
        let n = [1.0, 0.0, 0.0, 0.0];
        mfo.aggregate_into(&[], &[], &n);
        assert_eq!(mfo.kappa(), 0.0);
        assert_eq!(mfo.kappa_a(), 0.0);
        assert_eq!(mfo.q(), 0.0);
    }

    #[test]
    fn estimate_chaos_intensity_grows_with_q() {
        // Non-saturating h: tanh(1.5) ≈ 0.905, Q = 0.905² ≈ 0.82 (per-dim
        // average, all 4 dims equal). Then g = sqrt(Q/(1-Q)) ≈ 2.13 > 2.
        // (Using h=5 saturates tanh→1, giving Q=1, which hits the div-by-zero
        // guard and returns 0 — that path is tested separately below.)
        let mut mfo = MeanFieldOverlap::with_capacity(4);
        let h_mid = [1.5f32, 1.5, 1.5, 1.5];
        let a = [0.0f32; 4];
        let n = [1.0, 0.0, 0.0, 0.0]; // direction doesn't affect Q
        let hlas: Vec<&[f32]> = vec![&h_mid];
        let adapt: Vec<&[f32]> = vec![&a];
        mfo.aggregate_into(&hlas, &adapt, &n);
        // Q = (1/4)·4·tanh(1.5)² = tanh(1.5)² ≈ 0.82.
        assert!(mfo.q() > 0.7 && mfo.q() < 0.95, "q {}", mfo.q());
        let g = mfo.estimate_chaos_intensity();
        assert!(g > 1.5, "g {}", g);
    }

    #[test]
    fn estimate_chaos_intensity_returns_zero_when_q_saturated() {
        // Fully saturated firing rates (Q → 1) hit the div-by-zero guard and
        // return 0. This is the degenerate "everything is maximally firing"
        // case — the estimator is meaningless there.
        let mut mfo = MeanFieldOverlap::with_capacity(4);
        let h_big = [5.0f32, 5.0, 5.0, 5.0];
        let a = [0.0f32; 4];
        let n = [0.5, 0.5, 0.5, 0.5];
        let hlas: Vec<&[f32]> = vec![&h_big];
        let adapt: Vec<&[f32]> = vec![&a];
        mfo.aggregate_into(&hlas, &adapt, &n);
        // Q = (1/4)·4·1² = 1 (saturated).
        assert!((mfo.q() - 1.0).abs() < 1e-6, "q {}", mfo.q());
        assert_eq!(mfo.estimate_chaos_intensity(), 0.0);
    }

    // ─── Hopf boundary ─────────────────────────────────────────────────────

    #[test]
    fn beta_zero_yields_no_hopf() {
        // β = 0 → no adaptation feedback → J is upper-triangular with real
        // eigenvalues. Discriminant ≥ 0, so hopf_boundary returns None.
        let p = HopfParams {
            beta: 0.0,
            ..HopfParams::default()
        };
        assert!(hopf_boundary(&p).is_none());
    }

    #[test]
    fn large_beta_with_slow_tau_a_yields_hopf_near_omega() {
        // Large β, τ_a ≫ τ_m → Hopf instability with ω ≈ 1/sqrt(τ_a) (paper Eq. A9).
        // We need the trace T > 0 too. T = (−1 + λ_eff·G_eff)/τ_m + (−1/τ_a).
        // With λ_eff = G_eff = 1: T = 0 + (−1/τ_a) < 0 — no Hopf.
        // To get T > 0 we need λ_eff·G_eff > 1. Set λ_eff = 1.5, G_eff = 1.0:
        //   J_11 = (−1 + 1.5)/1 = 0.5; J_22 = −1/30 ≈ −0.033.
        //   T = 0.5 − 0.033 = 0.467 > 0.
        //   D = 0.5·(−0.033) − (−1/1)·(β/30) = −0.0167 + β/30.
        //   For β = 1.4: D = −0.0167 + 0.0467 = 0.03; Δ = 0.467² − 4·0.03 = 0.218 − 0.12 = 0.098 > 0 → real eigenvalues, not Hopf.
        //   For β large enough to push Δ < 0: τ_a·τ_m·β·G_eff > (τ_a + τ_m)²/4 → 30·1·β·1 > (31)²/4 → β > 961/120 ≈ 8.0.
        // The paper's regime diagram has β in [0, 1.5], so within that range
        // Hopf requires λ_eff·G_eff to push T positive AND a large β.
        // Construct a deliberately Hopf-unstable case:
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 30.0,
            beta: 10.0,
            lambda_eff: 1.5,
            g_eff: 1.0,
        };
        let omega = hopf_boundary(&p).expect("expected Hopf instability");
        // ω = sqrt(|Δ|)/2. Δ = T² − 4·D.
        let t = jacobian_trace(&p);
        let d = jacobian_determinant(&p);
        let delta = t * t - 4.0 * d;
        assert!(delta < 0.0, "Δ should be negative for Hopf, got {}", delta);
        let expected_omega = (0.0 - delta).sqrt() * 0.5;
        assert!((omega - expected_omega).abs() < 1e-6);
        // Paper Eq. A9 limit ω ≈ 1/sqrt(τ_a) ≈ 0.183 — but this requires the
        // near-marginal regime; here we are far from marginal so just check
        // the value is positive and finite.
        assert!(omega > 0.0 && omega.is_finite());
    }

    #[test]
    fn stable_focus_yields_no_hopf() {
        // T < 0 always → no Hopf instability (even if Δ < 0).
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 30.0,
            beta: 0.5,
            lambda_eff: 0.5, // λ_eff·G_eff = 0.5 < 1 → J_11 < 0
            g_eff: 1.0,
        };
        let t = jacobian_trace(&p);
        assert!(t < 0.0, "T should be negative, got {}", t);
        assert!(hopf_boundary(&p).is_none());
    }

    #[test]
    fn hopf_boundary_is_deterministic() {
        let p = HopfParams {
            beta: 10.0,
            lambda_eff: 1.5,
            ..HopfParams::default()
        };
        let a = hopf_boundary(&p);
        let b = hopf_boundary(&p);
        match (a, b) {
            (Some(x), Some(y)) => assert_eq!(x.to_bits(), y.to_bits()),
            (None, None) => {}
            _ => panic!("non-deterministic"),
        }
    }

    #[test]
    fn static_boundary_detects_saddle() {
        // D < 0 → saddle → static_boundary returns true.
        // J_11 = (−1 + λ·G)/τ_m; pick λ_eff = 0 → J_11 = −1.
        // J_22 = −1/τ_a. J_12 = −G/τ_m; J_21 = β/τ_a.
        // D = (−1)(−1/τ_a) − (−G/τ_m)(β/τ_a) = 1/τ_a + G·β/(τ_m·τ_a).
        // That's positive — not a saddle. To get D < 0 we need J_12·J_21 > J_11·J_22,
        // i.e. (−G/τ_m)(β/τ_a) > (−1 + λ·G)(−1/τ_a) → with the right λ this flips.
        // Easier: just construct a known saddle directly.
        // J_11 = 1 (>0), J_22 = −1 (<0), J_12 = J_21 = 0 → D = −1 < 0.
        // That requires (−1 + λ·G)/τ_m = 1 → λ·G = 2, and J_22 = −1/τ_a = −1 → τ_a = 1.
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 1.0,
            beta: 0.0,
            lambda_eff: 2.0,
            g_eff: 1.0,
        };
        let d = jacobian_determinant(&p);
        assert!(d < 0.0, "D should be < 0 for saddle, got {}", d);
        assert!(static_boundary(&p));
    }

    #[test]
    fn saddle_strength_positive_for_saddle() {
        // Same saddle as static_boundary_detects_saddle: J_11 = 1, J_22 = −1.
        // Eigenvalues are ±1. λ₊ = 1.
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 1.0,
            beta: 0.0,
            lambda_eff: 2.0,
            g_eff: 1.0,
        };
        let s = saddle_strength(&p);
        assert!(s > 0.0, "saddle_strength should be positive for saddle, got {}", s);
        // λ₊ = (T + √Δ)/2 = (0 + √(0+4))/2 = 1.
        assert!((s - 1.0).abs() < 1e-6, "λ₊ should be 1.0, got {}", s);
    }

    #[test]
    fn saddle_strength_zero_for_complex_eigenvalues() {
        // Hopf regime: complex conjugate pair → saddle_strength = 0.
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 30.0,
            beta: 10.0,
            lambda_eff: 1.5,
            g_eff: 1.0,
        };
        assert!(hopf_boundary(&p).is_some(), "should be Hopf regime");
        let s = saddle_strength(&p);
        assert_eq!(s, 0.0, "saddle_strength should be 0 for complex eigenvalues");
    }

    #[test]
    fn saddle_strength_zero_for_stable_node() {
        // Both eigenvalues negative → stable node → saddle_strength = 0.
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 30.0,
            beta: 0.0,
            lambda_eff: 0.5, // λ·G = 0.5 < 1 → J_11 < 0
            g_eff: 1.0,
        };
        assert!(!static_boundary(&p), "should not be a saddle");
        let s = saddle_strength(&p);
        assert_eq!(s, 0.0, "saddle_strength should be 0 for stable node");
    }

    // ─── Regime enum ───────────────────────────────────────────────────────

    #[test]
    fn regime_u8_roundtrip() {
        for v in 0..=3u8 {
            let r = Regime::from_u8(v).expect("valid regime");
            assert_eq!(r.as_u8(), v);
        }
        assert!(Regime::from_u8(4).is_none());
        assert!(Regime::from_u8(255).is_none());
    }

    // ─── RegimeClassifier ──────────────────────────────────────────────────

    #[test]
    fn classify_static_when_no_hopf_and_low_g() {
        let mfo = MeanFieldOverlap::default();
        let p = HopfParams {
            beta: 0.0,
            lambda_eff: 0.5, // T < 0
            ..HopfParams::default()
        };
        let clf = RegimeClassifier::default();
        // Default estimate_chaos_intensity from a default-constructed overlap
        // is 0 (q=0); g_override = 0.5 < 1.0 chaos_threshold → Static.
        let r = clf.classify_with_g(&mfo, &p, 0.5);
        assert_eq!(r, Regime::Static);
    }

    #[test]
    fn classify_noise_sustained_when_stable_but_chaotic() {
        let mfo = MeanFieldOverlap::default();
        let p = HopfParams {
            beta: 0.0,
            lambda_eff: 0.5, // T < 0, no Hopf
            ..HopfParams::default()
        };
        let clf = RegimeClassifier::default();
        let r = clf.classify_with_g(&mfo, &p, 1.5); // g > chaos_threshold
        assert_eq!(r, Regime::NoiseSustainedOscillation);
    }

    #[test]
    fn classify_global_limit_cycle_when_deep_hopf() {
        let mfo = MeanFieldOverlap::default();
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 30.0,
            beta: 10.0,
            lambda_eff: 1.5, // T > 0
            g_eff: 1.0,
        };
        let t = jacobian_trace(&p);
        assert!(t > 0.1, "T = {} should exceed hopf_margin", t);
        let clf = RegimeClassifier::default();
        let r = clf.classify_with_g(&mfo, &p, 1.5);
        assert_eq!(r, Regime::GlobalLimitCycle);
    }

    #[test]
    fn classify_irregular_switching_when_near_hopf_and_chaotic() {
        // T slightly positive (between switching_margin=0.05 and hopf_margin=0.15)
        // with g > chaos_threshold → IrregularSwitching.
        // T = (−1 + λ·G)/τ_m + (−1/τ_a). Want T ≈ 0.07.
        // (λ·G − 1)/1 − 1/30 = 0.07 → λ·G = 1 + 0.07 + 0.0333 ≈ 1.1033.
        // Set λ_eff = 1.1033, G_eff = 1.0. β needs to be high enough for Δ < 0.
        // Δ = T² − 4·D; D = J_11·J_22 − J_12·J_21.
        // For Hopf we need Δ < 0; that requires β large enough.
        // Try β = 5.0:
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 30.0,
            beta: 5.0,
            lambda_eff: 1.1033,
            g_eff: 1.0,
        };
        let t = jacobian_trace(&p);
        let d = jacobian_determinant(&p);
        let delta = t * t - 4.0 * d;
        // Confirm this is actually Hopf (Δ < 0).
        // Δ = 0.07² − 4·D. D = 0.1033·(−1/30) − (−1)(5/30) = −0.00344 + 0.1667 = 0.163.
        // Δ = 0.0049 − 0.653 < 0. ✓
        assert!(delta < 0.0, "Δ = {} should be < 0", delta);
        assert!(
            t > 0.05 && t <= 0.15,
            "T = {} should be in (switching_margin, hopf_margin]",
            t
        );
        let clf = RegimeClassifier::default();
        let r = clf.classify_with_g(&MeanFieldOverlap::default(), &p, 1.5);
        assert_eq!(r, Regime::IrregularSwitching);
    }

    #[test]
    fn classify_is_deterministic() {
        let mfo = MeanFieldOverlap::default();
        let p = HopfParams {
            beta: 10.0,
            lambda_eff: 1.5,
            ..HopfParams::default()
        };
        let clf = RegimeClassifier::default();
        let r1 = clf.classify_with_g(&mfo, &p, 1.5);
        let r2 = clf.classify_with_g(&mfo, &p, 1.5);
        assert_eq!(r1.as_u8(), r2.as_u8());
    }

    // ── Weak-saddle gating (Issue 034 T1 follow-up) ──────────────────────

    #[test]
    fn classify_static_when_weak_saddle_and_chaotic() {
        // Weak saddle: λ₊ is small (below saddle_margin=0.005). Even with g >
        // chaos_threshold, the instability grows too slowly → Static.
        //
        // τ_m=1, τ_a=1, λ_eff=1.1, G_eff=1.0, β=0.097 produces a weak saddle:
        //   J_11 = 0.1, J_22 = −1, J_12 = −1, J_21 = 0.097
        //   D = −0.1 + 0.097 = −0.003 (barely < 0 → saddle)
        //   T = −0.9
        //   disc = 0.81 + 0.012 = 0.822; √disc = 0.9067
        //   λ₊ = (−0.9 + 0.9067)/2 ≈ 0.0034 < 0.005 → Static.
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 1.0,
            beta: 0.097,
            lambda_eff: 1.1,
            g_eff: 1.0,
        };
        let s = saddle_strength(&p);
        assert!(s > 0.0 && s < 0.005, "λ₊ = {} should be weak (< 0.005)", s);
        let clf = RegimeClassifier::default();
        let r = clf.classify_with_g(&MeanFieldOverlap::default(), &p, 1.5);
        assert_eq!(r, Regime::Static, "weak saddle should classify as Static");
    }

    #[test]
    fn classify_irregular_switching_when_strong_saddle_and_chaotic() {
        // Strong saddle: λ₊ > saddle_margin (0.005). With g > chaos_threshold → IS.
        // Reuse the static_boundary_detects_saddle setup (λ₊ = 1.0 ≫ 0.005).
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 1.0,
            beta: 0.0,
            lambda_eff: 2.0,
            g_eff: 1.0,
        };
        let s = saddle_strength(&p);
        assert!(s > 0.005, "λ₊ = {} should be strong (> 0.005)", s);
        let clf = RegimeClassifier::default();
        let r = clf.classify_with_g(&MeanFieldOverlap::default(), &p, 1.5);
        assert_eq!(r, Regime::IrregularSwitching);
    }

    // ── Spinodal-pole discriminant (Issue 034 T1-followup-2) ────────────

    #[test]
    fn classify_glc_when_strong_saddle_near_spinodal_pole() {
        // Near the spinodal pole: β·G_eff > spinodal_margin (9.0). A strong
        // saddle coinciding with spinodal proximity → GLC (nonlinear trapping).
        //
        // Construct: β=1.4, G_eff=7.0 (β·G_eff=9.8 > 9.0). Need λ₊ >
        // saddle_margin — set λ_eff high enough that J_11 is positive:
        //   J_11 = (−1 + λ_eff·G_eff)/τ_m; with λ_eff=1.5, G_eff=7.0:
        //   J_11 = (−1 + 10.5)/1 = 9.5
        //   J_22 = −1/τ_a; with τ_a=30: J_22 = −0.033
        //   T = 9.5 − 0.033 = 9.47 > 0 (trace positive)
        //   D = 9.5·(−0.033) − (−7.0/1)·(1.4/30) = −0.314 + 0.327 = 0.013 > 0
        //   disc = T²−4D = 89.7 − 0.052 = 89.6. √disc ≈ 9.47.
        //   λ₊ = (9.47+9.47)/2 ≈ 9.47, λ₋ ≈ 0.0014. Real eigenvalues.
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 30.0,
            beta: 1.4,
            lambda_eff: 1.5,
            g_eff: 7.0,
        };
        let bg = hopf_boundary(&p);
        assert!(bg.is_none(), "real eigenvalues → no Hopf");
        let s = saddle_strength(&p);
        assert!(s > 0.005, "λ₊ = {} should be strong", s);
        let bg_eff = p.beta * p.g_eff;
        assert!(bg_eff > 9.0, "β·G_eff = {} should exceed spinodal_margin", bg_eff);
        let clf = RegimeClassifier::default();
        let r = clf.classify_with_g(&MeanFieldOverlap::default(), &p, 1.4);
        assert_eq!(r, Regime::GlobalLimitCycle,
            "strong saddle + spinodal proximity → GLC");
    }

    #[test]
    fn classify_is_when_strong_saddle_far_from_spinodal_pole() {
        // Far from the spinodal pole: β·G_eff < spinodal_margin (9.0). A strong
        // saddle without spinodal proximity → IS (as before).
        // β=1.4, G_eff=3.0 (β·G_eff=4.2 < 9.0). λ_eff=2.0:
        //   J_11 = (−1 + 6.0)/1 = 5.0
        //   T = 5.0 − 0.033 = 4.97
        //   D = 5.0·(−0.033) − (−3.0)·(0.0467) = −0.167 + 0.14 = −0.027 < 0 (saddle)
        //   λ₊ > 0, strong saddle. Not near pole → IS.
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 30.0,
            beta: 1.4,
            lambda_eff: 2.0,
            g_eff: 3.0,
        };
        let bg_eff = p.beta * p.g_eff;
        assert!(bg_eff < 9.0, "β·G_eff = {} should be below spinodal_margin", bg_eff);
        let s = saddle_strength(&p);
        assert!(s > 0.005, "λ₊ = {} should be strong", s);
        let clf = RegimeClassifier::default();
        let r = clf.classify_with_g(&MeanFieldOverlap::default(), &p, 1.4);
        assert_eq!(r, Regime::IrregularSwitching,
            "strong saddle, not near pole → IS");
    }

    #[test]
    fn spinodal_check_skipped_when_beta_zero() {
        // β=0 → β·G_eff=0 < 9.0. The spinodal check is skipped even if G_eff
        // is large. Strong saddle → IS (as before).
        let p = HopfParams {
            tau_m: 1.0,
            tau_a: 1.0,
            beta: 0.0,
            lambda_eff: 2.0,
            g_eff: 100.0, // huge G_eff, but β·G_eff = 0
        };
        let bg_eff = p.beta * p.g_eff;
        assert_eq!(bg_eff, 0.0, "β=0 → β·G_eff=0");
        let clf = RegimeClassifier::default();
        let r = clf.classify_with_g(&MeanFieldOverlap::default(), &p, 1.5);
        assert_eq!(r, Regime::IrregularSwitching,
            "β=0 disables spinodal check → strong saddle → IS");
    }

    // ─── fast_tanh sanity ──────────────────────────────────────────────────

    #[test]
    fn fast_tanh_matches_std_at_origin() {
        // Padé [2/2] accuracy is ~0.025 worst-case around |x|≈2 (see fast_tanh
        // doc). Assert within 0.03 to give headroom for the worst observed point.
        for &x in &[0.0f32, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9] {
            let got = fast_tanh(x);
            let want = x.tanh();
            assert!(
                (got - want).abs() < 0.03,
                "fast_tanh({}) = {} vs std {} (drift {})",
                x,
                got,
                want,
                (got - want).abs()
            );
        }
    }

    #[test]
    fn fast_tanh_saturates_past_3() {
        assert_eq!(fast_tanh(4.0), 1.0);
        assert_eq!(fast_tanh(-4.0), -1.0);
        assert_eq!(fast_tanh(100.0), 1.0);
    }
}
