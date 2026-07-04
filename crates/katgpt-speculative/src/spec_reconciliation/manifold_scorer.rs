//! Manifold-based soft scorer for the Speculative Reconciliation Engine.
//!
//! Computes cosine similarity between client trajectories and a speculative
//! manifold of pre-computed trajectories. Used as a plausibility measure
//! for offline trajectory verification (Plan 177, Task T4).

use super::types::{ReconciliationConfig, TrajectoryPoint};
use crate::ScreeningPruner;
#[cfg(test)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Inlined local copy — the historical `katgpt_rs::benchmark::cosine_similarity`
    // lives in the root crate, which katgpt-speculative cannot depend on (cycle).
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 0.0;
    }
    dot / (na * nb)
}

/// Cosine-similarity scorer over a flattened speculative manifold.
///
/// The manifold is a flat `Vec<TrajectoryPoint>` populated from K speculative
/// trajectories. Scoring a single client point returns the maximum similarity
/// across all manifold points; scoring a full client trajectory returns the
/// average of those per-point maxima.
#[allow(dead_code)] // accept_threshold used by downstream reconciliation pipeline
pub struct ManifoldScorer {
    /// Pre-computed manifold trajectories flattened for comparison.
    manifold: Vec<TrajectoryPoint>,
    /// Cached L2 norms of `manifold` points, indexed in lock-step. Avoids
    /// recomputing the norm M times per `score_trajectory` pass — pre-built
    /// once in `set_manifold`.
    manifold_norms: Vec<f32>,
    /// Best similarity score computed in the last scoring pass.
    best_score: f32,
    /// Accept threshold (from [`ReconciliationConfig`]).
    accept_threshold: f32,
}

impl ManifoldScorer {
    /// Create a new scorer with an empty manifold and thresholds from `config`.
    pub fn new(config: &ReconciliationConfig) -> Self {
        Self {
            manifold: Vec::new(),
            manifold_norms: Vec::new(),
            best_score: 0.0,
            accept_threshold: config.accept_threshold,
        }
    }

    /// Flatten K trajectories into the manifold for comparison.
    ///
    /// Each inner `Vec<TrajectoryPoint>` represents one speculative trajectory.
    /// All points are appended into a single flat vector so that scoring is a
    /// simple max over a contiguous slice. Manifold point norms are pre-computed
    /// here so the per-client-point scoring pass is a tight dot-product + divide.
    pub fn set_manifold(&mut self, trajectories: &[Vec<TrajectoryPoint>]) {
        self.manifold.clear();
        let total: usize = trajectories.iter().map(|t| t.len()).sum();
        self.manifold.reserve(total);
        self.manifold_norms.clear();
        self.manifold_norms.reserve(total);
        for traj in trajectories {
            for mp in traj {
                self.manifold.push(*mp);
                // Fixed 8-wide dot — LLVM vectorizes this cleanly.
                let d = &mp.data;
                let norm_sq = d[0] * d[0]
                    + d[1] * d[1]
                    + d[2] * d[2]
                    + d[3] * d[3]
                    + d[4] * d[4]
                    + d[5] * d[5]
                    + d[6] * d[6]
                    + d[7] * d[7];
                self.manifold_norms.push(norm_sq.sqrt());
            }
        }
    }

    /// Score a single client point against every manifold point.
    ///
    /// Returns `max_j(cosine_similarity(client_point.data, manifold[j].data))`.
    /// Returns `0.0` when the manifold is empty.
    #[inline]
    pub fn score_against_manifold(&self, client_point: &TrajectoryPoint) -> f32 {
        let n = self.manifold.len();
        if n == 0 {
            return 0.0;
        }
        let cd = &client_point.data;
        let client_norm_sq = cd[0] * cd[0]
            + cd[1] * cd[1]
            + cd[2] * cd[2]
            + cd[3] * cd[3]
            + cd[4] * cd[4]
            + cd[5] * cd[5]
            + cd[6] * cd[6]
            + cd[7] * cd[7];
        let client_norm = client_norm_sq.sqrt();
        if client_norm < f32::EPSILON {
            return 0.0;
        }
        // Single fused pass: dot + divide by pre-cached manifold norms.
        // No per-iteration norm computation and no inner iterator allocations.
        let mut best = f32::NEG_INFINITY;
        for (mp, &mn) in self.manifold.iter().zip(self.manifold_norms.iter()) {
            if mn < f32::EPSILON {
                continue;
            }
            let md = &mp.data;
            let dot = cd[0] * md[0]
                + cd[1] * md[1]
                + cd[2] * md[2]
                + cd[3] * md[3]
                + cd[4] * md[4]
                + cd[5] * md[5]
                + cd[6] * md[6]
                + cd[7] * md[7];
            let s = dot / (client_norm * mn);
            if s > best {
                best = s;
            }
        }
        if best.is_finite() {
            best
        } else {
            0.0
        }
    }

    /// Score an entire client trajectory against the manifold.
    ///
    /// For each client point, computes the max cosine similarity across the
    /// manifold, then returns the average of those maxima.
    ///
    /// Returns `0.0` when the manifold or client trajectory is empty.
    pub fn score_trajectory(&self, client_trajectory: &[TrajectoryPoint]) -> f32 {
        if self.manifold.is_empty() || client_trajectory.is_empty() {
            return 0.0;
        }
        let sum: f32 = client_trajectory
            .iter()
            .map(|cp| self.score_against_manifold(cp))
            .sum();
        sum / client_trajectory.len() as f32
    }
}

impl ScreeningPruner for ManifoldScorer {
    /// Compatibility adapter: returns the last computed `best_score`.
    ///
    /// The real scoring happens via [`ManifoldScorer::score_against_manifold`]
    /// and [`ManifoldScorer::score_trajectory`]. This trait impl allows the
    /// scorer to slot into existing screening infrastructure.
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.best_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(data: [f32; 8]) -> TrajectoryPoint {
        TrajectoryPoint::new(data)
    }

    // ── Cosine similarity basics ──────────────────────────────────────

    #[test]
    fn identical_vectors_similarity_one() {
        let v = make_point([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let sim = cosine_similarity(&v.data, &v.data);
        assert!((sim - 1.0).abs() < 1e-5, "expected 1.0, got {sim}");
    }

    #[test]
    fn orthogonal_vectors_similarity_zero() {
        // (1,0,...) · (0,1,0,...) = 0
        let a = make_point([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = make_point([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sim = cosine_similarity(&a.data, &b.data);
        assert!(sim.abs() < 1e-5, "expected ~0.0, got {sim}");
    }

    #[test]
    fn known_similar_vectors() {
        // a = (1,0,...), b = (1,1,0,...) → dot=1, |a|=1, |b|=√2 → cos=1/√2 ≈ 0.7071
        let a = make_point([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = make_point([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sim = cosine_similarity(&a.data, &b.data);
        let expected = 1.0 / 2.0f32.sqrt();
        assert!(
            (sim - expected).abs() < 1e-5,
            "expected {expected}, got {sim}"
        );
    }

    // ── ManifoldScorer ────────────────────────────────────────────────

    #[test]
    fn empty_manifold_returns_zero() {
        let config = ReconciliationConfig::default();
        let scorer = ManifoldScorer::new(&config);
        let point = make_point([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(
            (scorer.score_against_manifold(&point) - 0.0).abs() < f32::EPSILON,
            "empty manifold should return 0.0"
        );
    }

    #[test]
    fn score_point_returns_max_similarity() {
        let config = ReconciliationConfig::default();
        let mut scorer = ManifoldScorer::new(&config);

        let target = make_point([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let close = make_point([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let orthogonal = make_point([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Two manifold trajectories, each with one point.
        scorer.set_manifold(&[vec![close], vec![orthogonal]]);

        let score = scorer.score_against_manifold(&target);
        // Should match `close` (high similarity) not `orthogonal` (~0).
        let expected = cosine_similarity(&target.data, &close.data);
        assert!(
            (score - expected).abs() < 1e-5,
            "expected {expected}, got {score}"
        );
        assert!(score > 0.8, "score should be close to 1.0, got {score}");
    }

    #[test]
    fn score_trajectory_returns_average_max() {
        let config = ReconciliationConfig::default();
        let mut scorer = ManifoldScorer::new(&config);

        // Manifold: one trajectory with one point.
        let manifold_pt = make_point([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        scorer.set_manifold(&[vec![manifold_pt]]);

        // Client: two points — identical and orthogonal.
        let identical = make_point([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let orthogonal = make_point([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let score = scorer.score_trajectory(&[identical, orthogonal]);
        // identical→1.0, orthogonal→0.0, average = 0.5
        assert!((score - 0.5).abs() < 1e-5, "expected 0.5, got {score}");
    }

    #[test]
    fn empty_client_trajectory_returns_zero() {
        let config = ReconciliationConfig::default();
        let mut scorer = ManifoldScorer::new(&config);
        scorer.set_manifold(&[vec![make_point([1.0; 8])]]);
        assert!(
            (scorer.score_trajectory(&[]) - 0.0).abs() < f32::EPSILON,
            "empty client trajectory should return 0.0"
        );
    }

    #[test]
    fn screening_pruner_relevance_returns_best_score() {
        let config = ReconciliationConfig {
            accept_threshold: 0.9,
            ..ReconciliationConfig::default()
        };
        let scorer = ManifoldScorer::new(&config);
        // best_score starts at 0.0
        assert!((scorer.relevance(0, 0, &[]) - 0.0).abs() < f32::EPSILON);
    }
}
