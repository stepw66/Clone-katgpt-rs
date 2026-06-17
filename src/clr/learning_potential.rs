//! Learning potential + curiosity write gate (Plan 284 T2.4).
//!
//! Distilled from Research 255's curiosity arm. This is the modelless
//! "how surprising was this trajectory under the frozen brain?" signal that
//! drives the write-memory decision in the self-adaptive loop.
//!
//! # Math
//!
//! ```text
//! S_LP(y) = -(1/|y|) * Σ_{t=0..|y|} log π(y_t | y_{<t})
//! ```
//!
//! Higher `S_LP` = more surprising = more to learn. The caller supplies a
//! per-token log-prob accessor `log_prob_at(t)`; katgpt-rs does NOT depend on
//! any model — this stays generic so the same primitive scores LLM continuations,
//! game action sequences, or any other tokenized outcome.
//!
//! # Memory write gate
//!
//! `should_write_memory(reliability, s_lp, config)` returns
//! `reliability > config.tau_reliable && s_lp > config.tau_curiosity`. This is
//! the gateable curiosity predicate — high reliability AND high surprise means
//! "we found something the frozen brain got right but didn't expect" — exactly
//! the signal worth persisting.
//!
//! NO softmax. NO training. Pure arithmetic on caller-supplied log-probs.

use crate::clr::ClrConfig;

/// Compute the learning potential `S_LP` of a tokenized outcome.
///
/// `S_LP = -(1/len) * Σ_{t=0..len} log_prob_at(t)`.
///
/// Higher = more surprising under the current frozen brain. The caller supplies
/// the per-token log-prob accessor; this function does not touch any model.
///
/// # Arguments
///
/// * `len` — number of tokens/steps in the outcome (`|y|`). Must be > 0.
/// * `log_prob_at` — closure `Fn(usize) -> f32` returning
///   `log π(y_t | y_{<t})` for index `t` in `[0, len)`.
///
/// # Returns
///
/// The mean negative log-probability. Bounded below by 0 (when all log-probs
/// are 0, i.e. the model was certain). Unbounded above (when the model
/// assigned near-zero probability).
///
/// # Panics
///
/// Panics if `len == 0` (no outcome to score).
#[inline]
pub fn learning_potential<F: Fn(usize) -> f32>(len: usize, log_prob_at: F) -> f32 {
    assert!(len > 0, "learning_potential: len must be > 0");
    let mut sum = 0.0f32;
    for t in 0..len {
        sum += log_prob_at(t);
    }
    -(1.0 / len as f32) * sum
}

/// Should we persist this trajectory to memory for future freeze/thaw cycles?
///
/// Gateable curiosity predicate: write when the trajectory is BOTH reliable
/// (passed the CLR reliability bar) AND surprising (high `S_LP` under the
/// frozen brain). This selects exactly the trajectories that the current
/// brain got right but didn't expect — the highest-value training signal
/// for the next freeze/thaw direction-vector update.
///
/// # Arguments
///
/// * `reliability` — per-trajectory CLR reliability `r_k` in `(0, 1)`.
/// * `s_lp` — learning potential from [`learning_potential`].
/// * `config` — CLR config supplying `tau_reliable` and `tau_curiosity`.
///
/// # Returns
///
/// `true` iff `reliability > config.tau_reliable && s_lp > config.tau_curiosity`.
#[inline]
pub fn should_write_memory(reliability: f32, s_lp: f32, config: &ClrConfig) -> bool {
    reliability > config.tau_reliable && s_lp > config.tau_curiosity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_log_prob_sequence() {
        // log π = [-1.0, -2.0, -3.0, -4.0] → sum = -10 → mean neg = 10/4 = 2.5
        let log_probs = [-1.0f32, -2.0, -3.0, -4.0];
        let s_lp = learning_potential(log_probs.len(), |t| log_probs[t]);
        assert!((s_lp - 2.5).abs() < 1e-6, "expected 2.5, got {}", s_lp);
    }

    #[test]
    fn zero_when_certain() {
        // All log-probs zero (model was certain) → S_LP = 0.
        let s_lp = learning_potential(5, |_| 0.0);
        assert!(s_lp.abs() < 1e-6);
    }

    #[test]
    fn higher_when_surprising() {
        let certain = learning_potential(3, |_| -0.1);
        let surprising = learning_potential(3, |_| -5.0);
        assert!(surprising > certain);
    }

    #[test]
    fn should_write_memory_gates_correctly() {
        let config = ClrConfig::default(); // tau_reliable=0.5, tau_curiosity=0.7
        // Reliable + surprising → write.
        assert!(should_write_memory(0.8, 1.5, &config));
        // Reliable but not surprising → don't write.
        assert!(!should_write_memory(0.8, 0.3, &config));
        // Surprising but not reliable → don't write.
        assert!(!should_write_memory(0.3, 1.5, &config));
        // Neither → don't write.
        assert!(!should_write_memory(0.3, 0.3, &config));
    }

    #[test]
    #[should_panic(expected = "len must be > 0")]
    fn panics_on_zero_len() {
        let _ = learning_potential(0, |_| 0.0);
    }
}
