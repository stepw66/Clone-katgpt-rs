//! Family C — leaky integrator / delta-rule SSM kernel.
//!
//! This is a **standalone mirror** of the math in
//! `ReconstructionState::evolve_hla()`
//! (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs` L625–648). It
//! exists so that Plan 276's `MicroRecurrentBeliefState` trait has a Family C
//! implementation that can be benchmarked against the Family A attractor kernel
//! on the same footing (G2.1 coherence benchmark).
//!
//! # IMPORTANT — scope note
//!
//! Plan 276 Phase 2 task T2.1 calls for refactoring `evolve_hla` itself into a
//! thin delegate over `LeakyIntegrator::step()`. **That refactor is OUT OF
//! SCOPE for this delegation** — it requires editing
//! `sense/reconstruction.rs`, which is locked. What's here is the standalone
//! kernel with the *same math*, so:
//!
//! 1. The trait has a Family C impl (the G1.1–G1.5 tests can run on it too).
//! 2. Future T2.1 refactor will replace `evolve_hla`'s body with a delegate
//!    call to this kernel — zero behavior change, because the math is identical.
//!
//! When T2.1 lands, the existing `ReconstructionState::evolve_hla()` will
//! become:
//!
//! ```text
//! pub fn evolve_hla(&mut self) {
//!     let kernel = LeakyIntegrator::new(self.config.hla_learning_rate, self.config.max_hla_delta, 8);
//!     kernel.step(&mut self.hla, &self.evidence.kind_activations_padded());
//! }
//! ```
//!
//! (The `KIND_MAP` wrap — `kind_activations[0,1,2,3,4,5,0,1]` — would move
//! into a small helper on `TripleEvidence`. That's a T2.1 concern.)
//!
//! # Math (verbatim from evolve_hla)
//!
//! Given `input` = activation vector of length `dim`:
//!
//! ```text
//! total        = sum(input)
//! if total < 1e-8: return (no update — avoids div-by-zero)
//! t_min        = total.min(1.0)
//! scale        = lr * t_min / total
//! half_total   = 0.5 * total
//! for i in 0..dim:
//!     delta         = scale * (input[i] - half_total)
//!     clamped_delta = delta.clamp(-max_delta, max_delta)
//!     state[i]      = (state[i] + clamped_delta).clamp(-1.0, 1.0)
//! ```
//!
//! Properties (inherited from `evolve_hla`):
//! - Always stable: output clamped to `[-1, 1]`.
//! - Zero allocation: operates on the `&mut [f32]` slice.
//! - No softmax: pure additive update with sigmoid-style bounds.

use crate::micro_belief::bridge::project_to_scalars as bridge_project;
use crate::micro_belief::types::{MicroRecurrentBeliefState, RecurrenceFamily};

/// Family C leaky-integrator kernel — mirrors `ReconstructionState::evolve_hla`.
///
/// Construct with [`new`](Self::new). The kernel is stateless aside from its
/// config (`lr`, `max_delta`, `dim`) — the belief vector lives in the caller's
/// `&mut [f32]`.
#[derive(Clone, Debug)]
pub struct LeakyIntegrator {
    /// Learning rate (`config.hla_learning_rate` in `ReconstructionState`).
    pub lr: f32,
    /// Maximum per-tick delta (`config.max_hla_delta` in `ReconstructionState`).
    pub max_delta: f32,
    /// Belief-vector dimension.
    pub dim: usize,
}

impl LeakyIntegrator {
    /// Construct a new leaky integrator.
    ///
    /// Defaults that match `ReconstructionConfig::default()`:
    /// - `lr = 0.1`
    /// - `max_delta = 0.2`
    pub fn new(lr: f32, max_delta: f32, dim: usize) -> Self {
        Self { lr, max_delta, dim }
    }

    /// Construct with HLA-default config (`lr=0.1`, `max_delta=0.2`).
    pub fn hla_default(dim: usize) -> Self {
        Self::new(0.1, 0.2, dim)
    }
}

impl MicroRecurrentBeliefState for LeakyIntegrator {
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    /// Advance one tick using the `evolve_hla` leaky-integrator update.
    ///
    /// See the module-level docs for the exact formula. This is byte-for-byte
    /// equivalent to `ReconstructionState::evolve_hla()` modulo the `KIND_MAP`
    /// gather (which is a concern of `TripleEvidence`, not the kernel).
    #[inline]
    fn step(&self, state: &mut [f32], input: &[f32]) {
        debug_assert_eq!(state.len(), self.dim, "state/dim mismatch");
        debug_assert_eq!(input.len(), self.dim, "input/dim mismatch");

        // Verbatim from reconstruction.rs L628–647.
        let total_activation: f32 = input.iter().copied().sum();
        if total_activation < 1e-8 {
            return;
        }
        let t_min = total_activation.min(1.0);
        let scale = self.lr * t_min / total_activation;
        let half_total = 0.5 * total_activation;

        for i in 0..self.dim {
            let normalized = input[i];
            let delta = scale * (normalized - half_total);
            let clamped_delta = delta.clamp(-self.max_delta, self.max_delta);
            state[i] = (state[i] + clamped_delta).clamp(-1.0, 1.0);
        }
    }

    #[inline(always)]
    fn project_to_scalars(
        &self,
        state: &[f32],
        directions: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        bridge_project(state, directions, dim, out);
    }

    #[inline]
    fn family(&self) -> RecurrenceFamily {
        RecurrenceFamily::DeltaRule
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_input_is_noop() {
        // Matches evolve_hla: total < 1e-8 → early return, state unchanged.
        let k = LeakyIntegrator::hla_default(8);
        let mut state = [0.5f32; 8];
        let input = [0.0f32; 8];
        k.step(&mut state, &input);
        assert_eq!(state, [0.5f32; 8], "zero input must be a no-op");
    }

    #[test]
    fn state_stays_in_minus_one_to_one() {
        // Mirror of G1.2 for Family C — must always pass (linear, always stable).
        let k = LeakyIntegrator::new(0.5, 1.0, 8);
        let mut state = [0.0f32; 8];
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..10_000 {
            let input: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, rng.f32(), rng.f32(), rng.f32()];
            k.step(&mut state, &input);
            for &v in &state {
                assert!(v >= -1.0 && v <= 1.0, "Family C diverged: {v}");
            }
        }
    }

    #[test]
    fn family_is_delta_rule() {
        let k = LeakyIntegrator::hla_default(8);
        assert_eq!(k.family(), RecurrenceFamily::DeltaRule);
    }

    #[test]
    fn matches_evolve_hla_math_reference() {
        // Reference implementation of the evolve_hla math, computed directly.
        // The kernel MUST produce identical output for the same input.
        let lr = 0.1f32;
        let max_delta = 0.2f32;
        let dim = 8usize;
        let k = LeakyIntegrator::new(lr, max_delta, dim);

        let input: [f32; 8] = [0.1, 0.2, 0.3, 0.0, 0.5, 0.0, 0.0, 0.0];
        let mut state_actual = [0.0f32; 8];
        let mut state_ref = [0.0f32; 8];

        // Reference: verbatim evolve_hla body (without KIND_MAP — direct input).
        let total: f32 = input.iter().sum();
        assert!(total >= 1e-8);
        let t_min = total.min(1.0);
        let scale = lr * t_min / total;
        let half_total = 0.5 * total;
        for i in 0..dim {
            let normalized = input[i];
            let delta = scale * (normalized - half_total);
            let clamped_delta = delta.clamp(-max_delta, max_delta);
            state_ref[i] = (state_ref[i] + clamped_delta).clamp(-1.0, 1.0);
        }

        // Actual: through the kernel.
        k.step(&mut state_actual, &input);

        assert_eq!(state_actual, state_ref, "kernel must match evolve_hla math");
    }

    #[test]
    fn hla_default_matches_reconstruction_defaults() {
        // The defaults here must match ReconstructionConfig's HLA defaults so
        // the future T2.1 refactor is a true zero-behavior-change delegate.
        let k = LeakyIntegrator::hla_default(8);
        assert_eq!(k.lr, 0.1);
        assert_eq!(k.max_delta, 0.2);
        assert_eq!(k.dim, 8);
    }
}
