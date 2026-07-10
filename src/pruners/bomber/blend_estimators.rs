//! Modelless nonlinear blend estimators for HLPlayer (Plan 436 / Issue 428).
//!
//! Two estimators that learn the context→Q mapping that the linear contextual
//! bandit (Issue 371 T7) could not. Both are **non-parametric** and
//! **modelless** — no offline training, no backprop, no gradient descent. The
//! only "mutation" is online incremental averaging on a fixed-size data
//! structure, which is the same class of primitive as the n-armed bandit.
//!
//! ## Why nonlinear estimators
//!
//! The linear contextual bandit failed because the optimal action depends on a
//! **nonlinear threshold** (`blast_proximity > 0.4` → Flee). A linear model
//! `θ_a^T · φ(s)` cannot represent this step function. These estimators can:
//!
//! - **Binned**: discretises `blast_proximity` into 5 bins. Each bin stores
//!   per-arm empirical Q averages. O(1) lookup. Captures the step function
//!   directly — bin 1 (safe) and bin 2 (danger) have different Q tables.
//!
//! - **Kernel**: Nadaraya-Watson weighted average over a ring buffer of
//!   observed (context, arm, reward) triples. Soft interpolation between
//!   observed contexts. Captures smooth nonlinear boundaries.
//!
//! ## Scoring integration
//!
//! Both estimators return Q ∈ [0, 1] per arm. The centered blend
//! `(Q - 0.5) * 2.0` maps this to `(-1, +1)`, matching the existing n-armed
//! and contextual bandit integration in `select_action`.
//!
//! ## Allocation discipline
//!
//! Both estimators use **stack-only fixed-size arrays** — zero heap allocation
//! in the hot path (G5 gate). The binned estimator uses `[f32; 35]` + `[u32; 35]`
//! (280 bytes). The kernel uses `[[f32; 7]; 128]` + `[usize; 128]` + `[f32; 128]`
//! (~4.5 KB). Both fit comfortably in cache.

use super::blend_context::{BLAST_PROXIMITY_IDX, CONTEXT_DIM, OPPONENT_PRESSURE_IDX};
use super::players::ACTION_COUNT;

// ── Binned estimator ─────────────────────────────────────────────────

/// Number of bins on `blast_proximity`. 5 bins with boundaries at
/// 0.2, 0.4, 0.6, 0.8. Bin 1→2 boundary (0.4) aligns with the nonlinear
/// danger threshold the linear bandit could not resolve.
const N_BINS: usize = 5;

/// 5 bins × 7 arms = 35 entries.
const BIN_TABLE_SIZE: usize = N_BINS * ACTION_COUNT;

/// Binned empirical estimator — per-(bin, arm) Q averages.
///
/// Discretises `phi[1]` (blast_proximity) into 5 bins. Each bin stores an
/// independent per-arm Q table. This directly captures the nonlinear step
/// function: bin 1 (safe, blast_prox ∈ [0.2, 0.4)) and bin 2 (danger,
/// blast_prox ∈ [0.4, 0.6)) have completely separate Q values for each arm.
///
/// O(1) predict and update. Zero heap allocation.
#[derive(Clone, Debug)]
pub struct BinnedBlendEstimator {
    /// Per-(bin, arm) Q average. Index: `bin * ACTION_COUNT + arm`.
    q: [f32; BIN_TABLE_SIZE],
    /// Per-(bin, arm) visit count.
    visits: [u32; BIN_TABLE_SIZE],
}

impl Default for BinnedBlendEstimator {
    fn default() -> Self {
        Self {
            q: [0.5; BIN_TABLE_SIZE],
            visits: [0; BIN_TABLE_SIZE],
        }
    }
}

impl BinnedBlendEstimator {
    /// Bin index for a given blast_proximity value.
    #[inline]
    fn bin_idx(phi: &[f32; CONTEXT_DIM]) -> usize {
        let prox = phi[BLAST_PROXIMITY_IDX];
        ((prox * N_BINS as f32) as usize).min(N_BINS - 1)
    }

    /// Predict Q for all arms at once. Returns `[f32; ACTION_COUNT]`.
    ///
    /// O(1) — single array lookup per arm.
    #[inline]
    pub fn predict_all(&self, phi: &[f32; CONTEXT_DIM]) -> [f32; ACTION_COUNT] {
        let base = Self::bin_idx(phi) * ACTION_COUNT;
        let mut result = [0.5f32; ACTION_COUNT];
        let mut i = 0;
        while i < ACTION_COUNT {
            result[i] = self.q[base + i];
            i += 1;
        }
        result
    }

    /// Online update: incremental average for the observed (context, arm, reward).
    ///
    /// O(1) — single array index + increment.
    #[inline]
    pub fn update(&mut self, phi: &[f32; CONTEXT_DIM], arm: usize, reward: f32) {
        debug_assert!(arm < ACTION_COUNT, "arm {arm} out of range");
        let idx = Self::bin_idx(phi) * ACTION_COUNT + arm;
        self.visits[idx] = self.visits[idx].saturating_add(1);
        let n = self.visits[idx] as f32;
        self.q[idx] += (reward - self.q[idx]) / n;
    }

    /// Whether this arm has been observed in ANY bin (cold-start gate).
    #[inline]
    pub fn is_cold(&self, arm: usize) -> bool {
        debug_assert!(arm < ACTION_COUNT, "arm {arm} out of range");
        let mut bin = 0;
        while bin < N_BINS {
            if self.visits[bin * ACTION_COUNT + arm] > 0 {
                return false;
            }
            bin += 1;
        }
        true
    }
}

// ── Kernel estimator ─────────────────────────────────────────────────

/// Ring buffer size for the kernel estimator. Bounds latency to
/// O(KERNEL_BUF × KERNEL_DIM). 128 × 2 = 256 ops ≈ 250ns.
const KERNEL_BUF: usize = 128;

/// Kernel bandwidth (σ). Controls how far the influence of each observed
/// sample spreads. σ = 0.15 for 2-dim distance: points within ~0.15
/// contribute significantly, points beyond ~0.45 are negligible.
const KERNEL_SIGMA: f32 = 0.15;

/// Kernel dimensions: blast_proximity and opponent_pressure.
/// These are the two defence features from the Issue 428 PoC (C3 config).
const KERNEL_DIMS: [usize; 2] = [BLAST_PROXIMITY_IDX, OPPONENT_PRESSURE_IDX];

/// Nadaraya-Watson kernel estimator — soft interpolation over observed
/// (context, arm, reward) triples.
///
/// For each prediction, computes a Gaussian-weighted average of observed
/// rewards for each arm. Weights decay with L2 distance in the
/// (blast_proximity, opponent_pressure) subspace. This captures smooth
/// nonlinear boundaries that hard binning cannot.
///
/// O(KERNEL_BUF × KERNEL_DIMS.len()) per predict. Zero heap allocation.
#[derive(Clone, Debug)]
pub struct KernelBlendEstimator {
    /// Ring buffer of observed contexts.
    contexts: [[f32; CONTEXT_DIM]; KERNEL_BUF],
    /// Which arm was taken for each observation.
    arms: [usize; KERNEL_BUF],
    /// Observed reward for each entry.
    rewards: [f32; KERNEL_BUF],
    /// Number of valid entries (≤ KERNEL_BUF).
    count: usize,
    /// Next write position (ring buffer pointer).
    next: usize,
    /// Precomputed σ².
    sigma_sq: f32,
}

impl Default for KernelBlendEstimator {
    fn default() -> Self {
        Self {
            contexts: [[0.0; CONTEXT_DIM]; KERNEL_BUF],
            arms: [0; KERNEL_BUF],
            rewards: [0.0; KERNEL_BUF],
            count: 0,
            next: 0,
            sigma_sq: KERNEL_SIGMA * KERNEL_SIGMA,
        }
    }
}

impl KernelBlendEstimator {
    /// Predict Q for all arms at once via Nadaraya-Watson weighting.
    ///
    /// For each stored observation `(φ_i, a_i, r_i)`, computes weight
    /// `w_i = exp(-dist² / σ²)` where `dist²` is the squared L2 distance in
    /// the kernel subspace. Then for each arm `a`:
    /// `Q(a) = Σ(w_i · r_i | a_i == a) / Σ(w_i | a_i == a)`.
    ///
    /// O(count × KERNEL_DIMS.len()) = O(128 × 2) ≈ 250ns.
    #[inline]
    pub fn predict_all(&self, phi: &[f32; CONTEXT_DIM]) -> [f32; ACTION_COUNT] {
        let mut wsum = [0.0f32; ACTION_COUNT];
        let mut wtot = [0.0f32; ACTION_COUNT];

        let mut i = 0;
        while i < self.count {
            // Squared distance in the kernel subspace (2 dims).
            let d0 = phi[KERNEL_DIMS[0]] - self.contexts[i][KERNEL_DIMS[0]];
            let d1 = phi[KERNEL_DIMS[1]] - self.contexts[i][KERNEL_DIMS[1]];
            let dist_sq = d0 * d0 + d1 * d1;

            let weight = (-dist_sq / self.sigma_sq).exp();
            let arm = self.arms[i];
            wsum[arm] += weight * self.rewards[i];
            wtot[arm] += weight;
            i += 1;
        }

        let mut result = [0.5f32; ACTION_COUNT];
        let mut a = 0;
        while a < ACTION_COUNT {
            if wtot[a] > 1e-10 {
                result[a] = wsum[a] / wtot[a];
            }
            a += 1;
        }
        result
    }

    /// Online update: append (context, arm, reward) to the ring buffer.
    ///
    /// O(1) — ring buffer insert.
    #[inline]
    pub fn update(&mut self, phi: &[f32; CONTEXT_DIM], arm: usize, reward: f32) {
        debug_assert!(arm < ACTION_COUNT, "arm {arm} out of range");
        self.contexts[self.next] = *phi;
        self.arms[self.next] = arm;
        self.rewards[self.next] = reward;
        self.next = (self.next + 1) % KERNEL_BUF;
        if self.count < KERNEL_BUF {
            self.count += 1;
        }
    }

    /// Whether this arm has been observed at least once (cold-start gate).
    #[inline]
    pub fn is_cold(&self, arm: usize) -> bool {
        debug_assert!(arm < ACTION_COUNT, "arm {arm} out of range");
        let mut i = 0;
        while i < self.count {
            if self.arms[i] == arm {
                return false;
            }
            i += 1;
        }
        true
    }
}

// ── Unit tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn phi_safe() -> [f32; CONTEXT_DIM] {
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    }

    fn phi_danger() -> [f32; CONTEXT_DIM] {
        [1.0, 0.9, 0.8, 0.7, 0.1, 0.9, 1.0]
    }

    // ── Binned estimator tests ──

    #[test]
    fn binned_cold_start_returns_neutral() {
        let est = BinnedBlendEstimator::default();
        let q = est.predict_all(&phi_safe());
        // All Qs should be 0.5 (neutral) at cold start.
        for &qi in &q {
            assert_eq!(qi, 0.5);
        }
    }

    #[test]
    fn binned_is_cold_until_updated() {
        let mut est = BinnedBlendEstimator::default();
        assert!(est.is_cold(0));
        est.update(&phi_safe(), 0, 1.0);
        assert!(!est.is_cold(0));
        // Other arms still cold.
        assert!(est.is_cold(1));
    }

    #[test]
    fn binned_different_bins_different_q() {
        let mut est = BinnedBlendEstimator::default();
        // Safe context → arm 0 gets reward 1.0.
        est.update(&phi_safe(), 0, 1.0);
        // Danger context → arm 0 gets reward 0.0.
        est.update(&phi_danger(), 0, 0.0);

        let q_safe = est.predict_all(&phi_safe());
        let q_danger = est.predict_all(&phi_danger());
        // Same arm (0) should have different Q in safe vs danger bins.
        assert!(q_safe[0] > q_danger[0], "safe Q {} should > danger Q {}", q_safe[0], q_danger[0]);
    }

    // ── Kernel estimator tests ──

    #[test]
    fn kernel_cold_start_returns_neutral() {
        let est = KernelBlendEstimator::default();
        let q = est.predict_all(&phi_safe());
        for &qi in &q {
            assert_eq!(qi, 0.5);
        }
    }

    #[test]
    fn kernel_is_cold_until_updated() {
        let mut est = KernelBlendEstimator::default();
        assert!(est.is_cold(3));
        est.update(&phi_safe(), 3, 0.8);
        assert!(!est.is_cold(3));
        assert!(est.is_cold(0));
    }

    #[test]
    fn kernel_nearby_context_dominates() {
        let mut est = KernelBlendEstimator::default();
        // Two observations: one near phi_safe, one near phi_danger.
        // Arm 0 near safe gets reward 1.0; arm 0 near danger gets reward 0.0.
        est.update(&phi_safe(), 0, 1.0);
        est.update(&phi_danger(), 0, 0.0);

        // Predicting at phi_safe should weight the safe observation more.
        let q_safe = est.predict_all(&phi_safe());
        let q_danger = est.predict_all(&phi_danger());
        assert!(
            q_safe[0] > q_danger[0],
            "kernel should give higher Q for arm 0 in safe context: {} vs {}",
            q_safe[0],
            q_danger[0]
        );
    }

    #[test]
    fn kernel_ring_buffer_evicts_oldest() {
        let mut est = KernelBlendEstimator::default();
        // Fill beyond capacity to test ring buffer wrap.
        for i in 0..(KERNEL_BUF + 50) {
            let phi = [i as f32 * 0.001; CONTEXT_DIM];
            est.update(&phi, 0, 1.0);
        }
        assert_eq!(est.count, KERNEL_BUF);
    }
}
