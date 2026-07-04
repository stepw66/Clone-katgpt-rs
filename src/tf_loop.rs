//! Training-Free Loop Wrapper — root re-export shim (Plan 136, Proposal 003 Phase 9).
//!
//! The bulk of tf_loop (ODE-refined sub-stepping, anchor blend, cache snapshots)
//! moved to `katgpt_transformer::tf_loop`. This file re-exports it under the
//! historical `katgpt_rs::tf_loop::*` paths AND retains the
//! `should_apply_pruner_at_iteration` function here, because that function
//! consumes `katgpt_pruners::PrunerSchedule` — and `katgpt-pruners` already
//! depends non-optionally on `katgpt-transformer`, so adding the reverse dep
//! to katgpt-transformer would create a package cycle.
//!
//! The split is invisible to consumers: `katgpt_rs::tf_loop::*` resolves both
//! the moved substrate and this retained function via the glob below.

pub use katgpt_transformer::tf_loop::*;

/// Decide whether full screening should be applied at a given loop iteration.
///
/// Thin wrapper over `PrunerSchedule::should_screen_full`. Retained at root
/// (not moved to `katgpt-transformer`) because of the
/// `katgpt-pruners → katgpt-transformer` non-optional dep — see the module
/// header comment for details.
///
/// # Arguments
///
/// * `iteration` — Current loop iteration (0-based)
/// * `total_iterations` — Total number of loop iterations (K)
/// * `schedule` — The pruner schedule to use
///
/// # Returns
///
/// `true` if full screening should be applied at this iteration.
#[cfg(feature = "thinking_prune")]
#[inline]
pub fn should_apply_pruner_at_iteration(
    iteration: usize,
    total_iterations: usize,
    schedule: katgpt_pruners::PrunerSchedule,
) -> bool {
    schedule.should_screen_full(iteration, total_iterations)
}
