//! BoM Trajectory Sampling — Multi-Hypothesis Heat Kernel (Plan 359 Phase 4).
//!
//! The modelless analog of PhysiFormer's generative uncertainty: sample K
//! diverse plausible trajectories by perturbing the initial state `h₀` along
//! the **near-harmonic subspace** (slowest-decaying eigenmodes of the operator
//! `A = -I + Δ + diag(motor)`), then applying the linear heat kernel to each.
//!
//! # Why near-harmonic?
//!
//! Perturbations along eigenmodes of the Laplacian evolve independently under
//! `exp(t·A)` (the modes decouple). The decay rate of mode `k` is
//! `exp(t·a_k)` where `a_k = motor_d - 1 + λ_k`. Modes with `|a_k|` small
//! decay slowly → perturbations along them PERSIST, producing genuinely
//! different futures. Modes with `|a_k|` large decay fast → perturbations
//! vanish, producing identical futures.
//!
//! The **harmonic subspace** (λ_k = 0) is a special case: under pure Laplacian
//! evolution (`exp(t·Δ)`) these modes don't decay at all. Under the full
//! motor-gated operator, they decay at rate `exp(t·(motor_d - 1))`. For
//! `motor_d = 1` they persist exactly; for `motor_d < 1` they decay.
//!
//! On a simply-connected game map the true harmonic subspace is 1-dimensional
//! (the constant vector), so we generalize to **near-harmonic** = the `n`
//! modes with smallest `|a_k|`. This is the modelless analog of perturbing
//! along the leading singular vectors of the propagator (ensemble forecasting
//! / BoM philosophy).
//!
//! # Distilled from
//!
//! - PhysiFormer (arXiv:2606.27364) — single-shot joint trajectory prediction.
//! - BoMSampler (Plan 281) — K-hypothesis diverse sampling via noise injection.
//! - Research 365 §5 — the BoM extension is the "speculative phase"; diversity
//!   depends on the near-harmonic subspace dimension.
//!
//! # Modelless
//!
//! No training, no backprop. The perturbation directions are read from the
//! precomputed eigendecomposition; the noise coefficients are provided by the
//! caller (deterministic RNG). Every step is closed-form algebra over the
//! shipped DEC substrate.
//!
//! # UQ classification (the "Report the Floor" rule, Issue 010)
//!
//! The K trajectory samples are a **diversity-for-exploration** signal (like
//! BoMSampler), NOT calibrated predictive uncertainty. The interval width is
//! controlled by `perturbation_sigma` (a hyperparameter), not by residual
//! calibration. See `conformal_floor_bom_trajectory.rs` for the floor
//! comparison that documents this classification with evidence.

use crate::heat_kernel::{DecEigendecomposition, K_MAX, heat_kernel_trajectory_linear_into};
use crate::types::CochainField;
use core::cmp::Ordering;

// =============================================================================
// Near-harmonic direction selection
// =============================================================================

/// Find the indices of the `n` eigenmodes with smallest `|a_k|` where
/// `a_k = motor_d - 1 + λ_k`.
///
/// These are the **slowest-decaying** modes under the operator
/// `A = -I + Δ + diag(motor)`: perturbations along them persist longest,
/// producing the most diverse futures under heat-kernel evolution.
///
/// Returns indices into `eig.eigenvalues`. The caller passes these to
/// [`heat_kernel_trajectory_bom`] / [`heat_kernel_trajectory_bom_into`] as the
/// `near_harmonic_indices` parameter.
///
/// # Arguments
/// - `eig`: precomputed eigendecomposition of the Laplacian.
/// - `motor_d`: representative motor value for the channel of interest
///   (typically `motor_vec[0]`, or `0.0` if `motor_dim == 0`).
/// - `n`: number of near-harmonic directions to select. Capped at `eig.k()`.
///
/// # Sorting
///
/// `eig.eigenvalues` is sorted descending (largest first). This function
/// re-sorts by `|a_k|` ascending and returns the `n` smallest. The result is
/// NOT necessarily contiguous in the original spectrum.
pub fn near_harmonic_indices(
    eig: &DecEigendecomposition,
    motor_d: f32,
    n: usize,
) -> Vec<usize> {
    let k = eig.k();
    let n_capped = n.min(k);
    let mut scored: Vec<(usize, f32)> = (0..k)
        .map(|i| {
            let a_k = motor_d - 1.0 + eig.eigenvalues[i];
            (i, a_k.abs())
        })
        .collect();
    // Sort by |a_k| ascending (smallest decay rate first). partial_cmp because
    // f32 is not Ord; NaN (shouldn't occur for finite eigenvalues + motor) maps
    // to Equal as a defensive fallback.
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    scored.into_iter().take(n_capped).map(|(i, _)| i).collect()
}

// =============================================================================
// BoM trajectory sampling
// =============================================================================

/// Sample K diverse trajectories via near-harmonic perturbation of `h₀`.
///
/// For each hypothesis `k ∈ 0..K`:
/// 1. Build a spatial perturbation field as a weighted sum of the
///    `near_harmonic_indices` eigenvectors, using the caller-provided noise
///    coefficients: `perturbation[i] = σ · Σ_m noise[k·M+m] · v_{dir_m}[i]`.
/// 2. Add the perturbation to `h₀` (broadcast across all channels — the same
///    spatial pattern perturbs every channel uniformly).
/// 3. Apply the linear heat kernel `exp(t·A)·(h₀ + perturbation)`.
///
/// The K trajectories differ in their near-harmonic content, which persists
/// (decays slowly) under the evolution — producing genuinely diverse futures.
///
/// # Allocating variant
///
/// This allocates K `CochainField`s for the output. For steady-state loops,
/// use [`heat_kernel_trajectory_bom_into`] with a pre-allocated `&mut
/// [CochainField]` scratch.
///
/// # Arguments
/// - `eig`: precomputed eigendecomposition (shared with the linear path).
/// - `h0`: initial field.
/// - `motor_vec`, `motor_dim`, `t`: propagation parameters (same as
///   [`heat_kernel_trajectory_linear`](crate::heat_kernel_trajectory_linear)).
/// - `k_hypotheses`: number of trajectories to sample (`K`).
/// - `perturbation_sigma`: magnitude of each perturbation coefficient.
/// - `near_harmonic_indices`: the eigenmode indices to perturb along
///   (length `M`; use [`near_harmonic_indices`] to compute).
/// - `noise`: pre-generated Gaussian coefficients, length `k_hypotheses · M`.
///   `noise[k · M + m]` is the `m`-th coefficient for hypothesis `k`.
///
/// # Panics (debug)
///
/// Debug-asserts that `near_harmonic_indices` and `noise` have compatible
/// lengths with `k_hypotheses`.
#[allow(clippy::too_many_arguments, reason = "BoM perturbation sweep needs eig + field + motor + perturbation params; a config struct would obscure the math")]
pub fn heat_kernel_trajectory_bom(
    eig: &DecEigendecomposition,
    h0: &CochainField,
    motor_vec: &[f32],
    motor_dim: usize,
    t: f32,
    k_hypotheses: usize,
    perturbation_sigma: f32,
    near_harmonic_indices: &[usize],
    noise: &[f32],
) -> Vec<CochainField> {
    let n = h0.n_cells();
    let dim = h0.dim;
    let mut out: Vec<CochainField> = (0..k_hypotheses)
        .map(|_| CochainField::zeros(h0.rank, n, dim))
        .collect();
    let mut scratch = CochainField::zeros(h0.rank, n, dim);
    let out_slice = out.as_mut_slice();
    heat_kernel_trajectory_bom_into(
        eig,
        h0,
        motor_vec,
        motor_dim,
        t,
        k_hypotheses,
        perturbation_sigma,
        near_harmonic_indices,
        noise,
        out_slice,
        &mut scratch,
    );
    out
}

/// Zero-alloc BoM trajectory sampling — writes into `out` and reuses `scratch`.
///
/// Same as [`heat_kernel_trajectory_bom`] but writes into caller-provided
/// buffers:
/// - `out`: slice of length `k_hypotheses`. Each element is resized to match
///   `h0`'s shape and filled with the trajectory. Reused across calls.
/// - `scratch`: a single `CochainField` used as the perturbed `h₀` (resized to
///   match `h0`'s shape). Reused across all K hypotheses within this call.
///
/// After the eigendecomposition precompute, the per-hypothesis cost is one
/// heat-kernel application (`O(n·k·dim)`) plus the perturbation assembly
/// (`O(n·M)` where `M = near_harmonic_indices.len()`). The total for K
/// hypotheses is `O(K · (n·k·dim + n·M))`.
#[allow(clippy::too_many_arguments, reason = "zero-alloc variant mirrors heat_kernel_trajectory_bom; caller-provided scratch/out add the 2 extra args")]
pub fn heat_kernel_trajectory_bom_into(
    eig: &DecEigendecomposition,
    h0: &CochainField,
    motor_vec: &[f32],
    motor_dim: usize,
    t: f32,
    k_hypotheses: usize,
    perturbation_sigma: f32,
    near_harmonic_indices: &[usize],
    noise: &[f32],
    out: &mut [CochainField],
    scratch: &mut CochainField,
) {
    let n = h0.n_cells();
    let dim = h0.dim;
    let m = near_harmonic_indices.len();

    debug_assert!(
        out.len() >= k_hypotheses,
        "bom: out.len {} < k_hypotheses {k_hypotheses}",
        out.len()
    );
    debug_assert!(
        m <= K_MAX,
        "bom: near_harmonic_indices.len {m} > K_MAX {K_MAX} (stack-alloc limit)"
    );
    debug_assert!(
        noise.len() >= k_hypotheses * m,
        "bom: noise.len {} < k_hypotheses({k_hypotheses}) * M({m})",
        noise.len()
    );

    // Ensure scratch can hold h₀ + perturbation.
    scratch.data.resize(n * dim, 0.0);
    scratch.dim = dim;
    scratch.rank = h0.rank;

    // Stack-allocated accumulation buffer for the perturbation field (spatial,
    // length n_cells). Capped at K_MAX = 64... wait, n_cells can exceed K_MAX.
    // Use a heap Vec for the spatial perturbation (this is the ONE allocation
    // the BoM path allows — analogous to Krylov's basis allocation). For
    // repeated calls the caller can hoist this into a persistent scratch.
    // We re-use `scratch.data` split: no, that conflates with h₀. Allocate one
    // Vec for the perturbation field, reused across all K hypotheses.
    let mut pert_field: Vec<f32> = vec![0.0f32; n];

    for (k_idx, out_k) in out.iter_mut().enumerate().take(k_hypotheses) {
        // 1. Copy h₀ into scratch.
        scratch.data[..n * dim].copy_from_slice(&h0.data[..n * dim]);

        // 2. Build the perturbation field: pert_field[i] = σ · Σ_m noise[k·M+m] · v_{dir_m}[i].
        // Zero the perturbation field (reused across hypotheses).
        pert_field.fill(0.0);
        let noise_offset = k_idx * m;
        for (mi, &eig_idx) in near_harmonic_indices.iter().enumerate() {
            let coeff = perturbation_sigma * noise[noise_offset + mi];
            if coeff == 0.0 {
                continue;
            }
            let v_m = eig.eigenvector(eig_idx);
            for (i, p) in pert_field.iter_mut().enumerate() {
                *p += coeff * v_m[i];
            }
        }

        // 3. Add the perturbation to every channel of h₀ (broadcast spatial).
        for (i, &p) in pert_field.iter().enumerate() {
            for d in 0..dim {
                scratch.data[i * dim + d] += p;
            }
        }

        // 4. Apply the linear heat kernel to the perturbed field.
        heat_kernel_trajectory_linear_into(eig, scratch, motor_vec, motor_dim, t, out_k);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heat_kernel::DecEigendecomposition;
    use crate::types::{CellComplex, CochainField};

    /// Helper: build a rank-0 4×4 grid complex (16 vertices).
    fn grid_4x4() -> CellComplex {
        CellComplex::grid_2d(4, 4)
    }

    /// Helper: a bump field centered at cell 5, amplitude 1.0, dim=1.
    fn bump_field(cx: &CellComplex) -> CochainField {
        let n = cx.n_cells(0);
        let mut h0 = CochainField::zeros(0, n, 1);
        for (i, v) in h0.data.iter_mut().enumerate() {
            let dx = (i as f32) - 5.0;
            *v = (-0.5 * dx * dx).exp();
        }
        h0
    }

    /// Helper: full eigendecomposition on 4×4 (k=16, max_iter=2000).
    fn full_eig(cx: &CellComplex) -> DecEigendecomposition {
        DecEigendecomposition::compute(cx, 0, 16, 2000)
    }

    /// Helper: deterministic Gaussian noise via Box-Muller on a simple LCG.
    /// Produces `k * m` standard-normal coefficients.
    fn gaussian_noise(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            // SplitMix64 step.
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let u1 = ((z ^ (z >> 31)) >> 40) as f32 / ((1u64 << 24) as f32);
            // Second draw.
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z2 = state;
            z2 = (z2 ^ (z2 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z2 = (z2 ^ (z2 >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let u2 = ((z2 ^ (z2 >> 31)) >> 40) as f32 / ((1u64 << 24) as f32);
            // Box-Muller (guard against log(0)).
            let u1 = u1.clamp(1e-7, 1.0 - 1e-7);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * core::f32::consts::PI * u2;
            out.push(r * theta.cos());
        }
        out
    }

    // ── T4.1: near_harmonic_indices ──────────────────────────────────────────

    #[test]
    fn near_harmonic_indices_returns_smallest_abs_a() {
        // On 4×4, eigenvalues of the graph Laplacian range from 0 (constant) to
        // ~8 (highest frequency). With motor_d = 0: a_k = -1 + λ_k. The smallest
        // |a_k| is at λ_k ≈ 1.
        let cx = grid_4x4();
        let eig = full_eig(&cx);
        let motor_d = 0.0_f32;
        let dirs = near_harmonic_indices(&eig, motor_d, 4);
        assert_eq!(dirs.len(), 4, "should return 4 directions");
        // Verify: the returned directions should have the smallest |a_k|.
        let all_abs: Vec<f32> = (0..eig.k())
            .map(|i| (motor_d - 1.0 + eig.eigenvalues[i]).abs())
            .collect();
        let returned_abs: Vec<f32> = dirs.iter().map(|&i| all_abs[i]).collect();
        let max_returned = returned_abs.iter().cloned().fold(0.0f32, f32::max);
        // Every non-returned direction should have |a_k| >= max_returned.
        for (i, &abs) in all_abs.iter().enumerate() {
            if !dirs.contains(&i) {
                assert!(
                    abs >= max_returned - 1e-5,
                    "non-returned direction {i} has |a_k|={abs} < max_returned {max_returned}"
                );
            }
        }
    }

    #[test]
    fn near_harmonic_indices_caps_at_k() {
        let cx = grid_4x4();
        let eig = full_eig(&cx);
        let dirs = near_harmonic_indices(&eig, 0.0, 100);
        assert_eq!(dirs.len(), eig.k(), "should cap at eig.k()");
    }

    // ── T4.1: heat_kernel_trajectory_bom basic ───────────────────────────────

    #[test]
    fn bom_returns_k_trajectories() {
        let cx = grid_4x4();
        let eig = full_eig(&cx);
        let h0 = bump_field(&cx);
        let motor = [-10.0f32]; // stable
        let dirs = near_harmonic_indices(&eig, motor[0], 4);
        let noise = gaussian_noise(42, 8 * dirs.len());
        let trajs = heat_kernel_trajectory_bom(
            &eig, &h0, &motor, 1, 1.0, 8, 0.1, &dirs, &noise,
        );
        assert_eq!(trajs.len(), 8, "should return 8 trajectories");
        for (i, t) in trajs.iter().enumerate() {
            assert_eq!(t.n_cells(), h0.n_cells(), "traj {i} n_cells mismatch");
            assert_eq!(t.dim, h0.dim, "traj {i} dim mismatch");
            assert_eq!(t.rank, h0.rank, "traj {i} rank mismatch");
        }
    }

    #[test]
    fn bom_into_matches_allocating() {
        let cx = grid_4x4();
        let eig = full_eig(&cx);
        let h0 = bump_field(&cx);
        let motor = [-10.0f32];
        let dirs = near_harmonic_indices(&eig, motor[0], 4);
        let noise = gaussian_noise(7, 4 * dirs.len());

        let alloc = heat_kernel_trajectory_bom(
            &eig, &h0, &motor, 1, 1.0, 4, 0.1, &dirs, &noise,
        );
        let mut into: Vec<CochainField> = (0..4)
            .map(|_| CochainField::zeros(0, cx.n_cells(0), 1))
            .collect();
        let mut scratch = CochainField::zeros(0, cx.n_cells(0), 1);
        heat_kernel_trajectory_bom_into(
            &eig, &h0, &motor, 1, 1.0, 4, 0.1, &dirs, &noise, &mut into, &mut scratch,
        );

        for k in 0..4 {
            let (a, b) = (&alloc[k], &into[k]);
            for i in 0..a.data.len() {
                assert!(
                    (a.data[i] - b.data[i]).abs() < 1e-6,
                    "hypothesis {k}, cell {i}: alloc={:.6} vs into={:.6}",
                    a.data[i],
                    b.data[i]
                );
            }
        }
    }

    // ── T4.2: bom_produces_diverse_trajectories ──────────────────────────────

    #[test]
    fn bom_produces_diverse_trajectories() {
        // The K trajectories must have non-trivial L2 spread (not identical).
        let cx = grid_4x4();
        let eig = full_eig(&cx);
        let h0 = bump_field(&cx);
        let motor = [-10.0f32]; // stable
        let dirs = near_harmonic_indices(&eig, motor[0], 4);
        let noise = gaussian_noise(0xCAFE, 8 * dirs.len());
        let trajs = heat_kernel_trajectory_bom(
            &eig, &h0, &motor, 1, 1.0, 8, 0.1, &dirs, &noise,
        );

        // Compute pairwise L2 distances.
        let mut max_dist = 0.0f32;
        let mut sum_dist = 0.0f32;
        let mut n_pairs = 0usize;
        for i in 0..trajs.len() {
            for j in (i + 1)..trajs.len() {
                let mut d2 = 0.0f32;
                for k in 0..trajs[i].data.len() {
                    let diff = trajs[i].data[k] - trajs[j].data[k];
                    d2 += diff * diff;
                }
                let dist = d2.sqrt();
                max_dist = max_dist.max(dist);
                sum_dist += dist;
                n_pairs += 1;
            }
        }
        let mean_dist = sum_dist / n_pairs as f32;

        // Sanity: spread must be positive (trajectories differ).
        assert!(
            max_dist > 1e-6,
            "max pairwise L2 dist {max_dist} should be > 0 (trajectories must differ)"
        );
        assert!(
            mean_dist > 1e-8,
            "mean pairwise L2 dist {mean_dist} should be > 0"
        );
        eprintln!(
            "bom diversity: max_dist={max_dist:.6}, mean_dist={mean_dist:.6} over {n_pairs} pairs"
        );
    }

    // ── T4.2: bom_perturbation_zero_returns_baseline ─────────────────────────

    #[test]
    fn bom_zero_sigma_returns_baseline() {
        // With sigma=0, every perturbation is zero → every trajectory equals
        // the unperturbed linear heat kernel trajectory.
        let cx = grid_4x4();
        let eig = full_eig(&cx);
        let h0 = bump_field(&cx);
        let motor = [-10.0f32];
        let dirs = near_harmonic_indices(&eig, motor[0], 4);
        // noise doesn't matter when sigma=0, but pass valid-length.
        let noise = gaussian_noise(1, 3 * dirs.len());

        let trajs = heat_kernel_trajectory_bom(
            &eig, &h0, &motor, 1, 1.0, 3, 0.0, &dirs, &noise,
        );

        // Baseline: unperturbed linear heat kernel.
        let baseline = crate::heat_kernel::heat_kernel_trajectory_linear(
            &eig, &h0, &motor, 1, 1.0,
        );

        for (k, t) in trajs.iter().enumerate() {
            for i in 0..t.data.len() {
                assert!(
                    (t.data[i] - baseline.data[i]).abs() < 1e-6,
                    "sigma=0 hypothesis {k} cell {i}: traj={:.6} vs baseline={:.6}",
                    t.data[i],
                    baseline.data[i]
                );
            }
        }
    }

    // ── T4.2: bom_trajectories_are_finite_and_bounded (stability) ────────────

    #[test]
    fn bom_trajectories_are_finite_and_bounded() {
        // Under stable motor (all a_k < 0), every perturbed trajectory must
        // decay to a bounded value. No NaN, no inf, no blow-up.
        let cx = grid_4x4();
        let eig = full_eig(&cx);
        let h0 = bump_field(&cx);
        let motor = [-10.0f32];
        let dirs = near_harmonic_indices(&eig, motor[0], 4);
        let noise = gaussian_noise(99, 6 * dirs.len());

        // t=5 — moderately long horizon; field should be well below initial.
        let trajs = heat_kernel_trajectory_bom(
            &eig, &h0, &motor, 1, 5.0, 6, 0.1, &dirs, &noise,
        );

        let h0_norm = h0.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (k, t) in trajs.iter().enumerate() {
            let mut max_val = 0.0f32;
            for &v in &t.data {
                assert!(v.is_finite(), "traj {k} has non-finite value {v}");
                max_val = max_val.max(v.abs());
            }
            // Under stable motor, the trajectory must decay — well below h₀.
            assert!(
                max_val < h0_norm,
                "traj {k} max |val| {max_val} should be < h0 max {h0_norm} (decay under stable motor)"
            );
        }
    }

    // ── T4.2: bom_diversity_grows_with_sigma ─────────────────────────────────

    #[test]
    fn bom_diversity_grows_with_sigma() {
        // Larger perturbation → larger spread between trajectories. This is
        // the σ-controlled diversity signature (analogous to BoMSampler).
        let cx = grid_4x4();
        let eig = full_eig(&cx);
        let h0 = bump_field(&cx);
        let motor = [-10.0f32];
        let dirs = near_harmonic_indices(&eig, motor[0], 4);
        let noise = gaussian_noise(0xBEEF, 8 * dirs.len());

        let spread = |sigma: f32| -> f32 {
            let trajs = heat_kernel_trajectory_bom(
                &eig, &h0, &motor, 1, 1.0, 8, sigma, &dirs, &noise,
            );
            // Max pairwise L2.
            let mut max_d = 0.0f32;
            for i in 0..trajs.len() {
                for j in (i + 1)..trajs.len() {
                    let mut d2 = 0.0f32;
                    for k in 0..trajs[i].data.len() {
                        let diff = trajs[i].data[k] - trajs[j].data[k];
                        d2 += diff * diff;
                    }
                    max_d = max_d.max(d2.sqrt());
                }
            }
            max_d
        };

        let s_small = spread(0.01);
        let s_med = spread(0.1);
        let s_large = spread(1.0);

        eprintln!("σ-sweep spread: 0.01→{s_small:.6}, 0.1→{s_med:.6}, 1.0→{s_large:.6}");
        assert!(s_med > s_small, "σ=0.1 spread {s_med} should exceed σ=0.01 {s_small}");
        assert!(s_large > s_med, "σ=1.0 spread {s_large} should exceed σ=0.1 {s_med}");
    }
}
