//! Reference implementations of `DifficultyFilter` and `BatchQualityGate`
//! for the CGSP loop (Plan 274 T1.6 + reference plumbing).
//!
//! These are game-agnostic, modelless defaults. Production callers (riir-ai
//! Plan 299) swap in domain-specific impls via `CgspLoop::with_difficulty_filter`
//! and `CgspLoop::with_batch_gate`.

use crate::cgsp::traits::{BatchQualityGate, DifficultyFilter};
use crate::cgsp::types::Candidate;

// ── BreakevenDifficultyFilter ─────────────────────────────────────────────

/// Default `DifficultyFilter` — admits only the intermediate-difficulty band.
///
/// Drops candidates whose estimated solve-rate is below `floor` (already-known
/// / trivially-solved) or above `ceiling` (too hard / Solver has no chance).
///
/// This is the modelless katgpt-rs analogue of the SGS paper's
/// "intermediate-difficulty" curriculum filter and the breakeven_complexity
/// router (Plan 250) applied at candidate admission time.
#[derive(Clone, Copy, Debug)]
pub struct BreakevenDifficultyFilter {
    /// Estimated solve-rate floor (default 0.05).
    pub floor: f32,
    /// Estimated solve-rate ceiling (default 0.95).
    pub ceiling: f32,
}

impl Default for BreakevenDifficultyFilter {
    fn default() -> Self {
        Self {
            floor: 0.05,
            ceiling: 0.95,
        }
    }
}

impl BreakevenDifficultyFilter {
    /// Build with explicit floor/ceiling.
    pub fn new(floor: f32, ceiling: f32) -> Self {
        Self { floor, ceiling }
    }
}

impl DifficultyFilter for BreakevenDifficultyFilter {
    #[inline]
    fn admit(&self, _guide_score: f32, estimated_solve_rate: f32) -> bool {
        estimated_solve_rate > self.floor && estimated_solve_rate < self.ceiling
    }
}

// ── ColinearityBatchGate ──────────────────────────────────────────────────

/// Default `BatchQualityGate` — flags batches as degenerate when all
/// admitted candidates are colinear (effectively the same direction) or
/// when no candidates were admitted at all.
///
/// Colinearity is measured via pairwise cosine similarity against a threshold
/// (default `0.95`). When all admitted candidates exceed the threshold
/// against each other, the batch is degenerate.
#[derive(Clone, Copy, Debug)]
pub struct ColinearityBatchGate {
    /// Cosine-similarity threshold above which two candidates are considered
    /// colinear (default 0.95).
    pub colinearity_threshold: f32,
}

impl Default for ColinearityBatchGate {
    fn default() -> Self {
        Self {
            colinearity_threshold: 0.95,
        }
    }
}

impl ColinearityBatchGate {
    /// Build with a custom threshold.
    pub fn new(threshold: f32) -> Self {
        Self {
            colinearity_threshold: threshold.clamp(0.0, 1.0),
        }
    }
}

impl BatchQualityGate for ColinearityBatchGate {
    fn is_degenerate(
        &self,
        candidates: &[Candidate],
        admitted: &[bool],
        _guide_scores: &[f32],
    ) -> bool {
        // Case 1: nobody admitted → degenerate.
        let any_admitted = admitted.iter().any(|&a| a);
        if !any_admitted {
            return true;
        }
        // Case 2: all admitted candidates colinear with each other.
        let admitted_idx: Vec<usize> = admitted
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| if a { Some(i) } else { None })
            .collect();
        // Need at least 2 to check colinearity.
        if admitted_idx.len() < 2 {
            return false;
        }
        let mut all_colinear = true;
        for w in admitted_idx.windows(2) {
            let a = &candidates[w[0]].direction;
            let b = &candidates[w[1]].direction;
            let cos = cosine(a, b);
            if cos < self.colinearity_threshold {
                all_colinear = false;
                break;
            }
        }
        all_colinear
    }
}

/// Cosine similarity between two directions. Returns 0.0 on dim-mismatch.
#[inline]
fn cosine(a: &crate::cgsp::types::Direction, b: &crate::cgsp::types::Direction) -> f32 {
    let denom = (a.norm_sq() * b.norm_sq()).sqrt();
    if denom < 1e-9 {
        return 0.0;
    }
    a.dot(b) / denom
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cgsp::types::{Candidate, Direction};

    fn unit(dim: usize, axis: usize) -> Direction {
        let mut coords = vec![0.0f32; dim];
        coords[axis.min(dim.saturating_sub(1))] = 1.0;
        Direction { coords }
    }

    #[test]
    fn breakeven_admits_intermediate_band() {
        let f = BreakevenDifficultyFilter::default();
        assert!(!f.admit(1.0, 0.0), "trivially solved -> reject");
        assert!(!f.admit(1.0, 1.0), "too hard -> reject");
        assert!(f.admit(1.0, 0.5), "intermediate -> admit");
        assert!(f.admit(1.0, 0.5), "near-floor -> admit");
        assert!(!f.admit(1.0, 0.96), "above ceiling -> reject");
    }

    #[test]
    fn batch_gate_flags_all_rejected() {
        let gate = ColinearityBatchGate::default();
        let cands = vec![
            Candidate::new(unit(4, 0), 0),
            Candidate::new(unit(4, 1), 1),
        ];
        let admitted = vec![false, false];
        let scores = vec![0.5, 0.5];
        assert!(
            gate.is_degenerate(&cands, &admitted, &scores),
            "no admissions -> degenerate"
        );
    }

    #[test]
    fn batch_gate_flags_colinear() {
        let gate = ColinearityBatchGate::default();
        let cands = vec![
            Candidate::new(unit(4, 0), 0),
            Candidate::new(unit(4, 0), 0), // identical to first
        ];
        let admitted = vec![true, true];
        let scores = vec![0.9, 0.9];
        assert!(
            gate.is_degenerate(&cands, &admitted, &scores),
            "all colinear -> degenerate"
        );
    }

    #[test]
    fn batch_gate_passes_diverse() {
        let gate = ColinearityBatchGate::default();
        let cands = vec![
            Candidate::new(unit(4, 0), 0),
            Candidate::new(unit(4, 1), 1),
        ];
        let admitted = vec![true, true];
        let scores = vec![0.5, 0.5];
        assert!(
            !gate.is_degenerate(&cands, &admitted, &scores),
            "diverse batch -> not degenerate"
        );
    }
}
