//! Best Buddies Drafting — Mutual NN Filter for Speculative Decoding (Plan 199).
//!
//! Filters draft model marginals to only tokens where draft and target models
//! agree bidirectionally (mutual nearest neighbors / "best buddies").
//!
//! Feature flag: `best_buddies`

use katgpt_core::traits::{BestBuddyAligner, pearson_correlation};

/// Marginal-based best buddy aligner using Pearson correlation.
///
/// Compares draft vs target logit distributions at each position.
/// High correlation → tokens are likely accepted → include in DDTree.
/// Low correlation → tokens likely rejected → filter out.
pub struct MarginalBestBuddyAligner {
    /// Correlation threshold for filtering. Default: 0.3
    pub threshold: f32,
    /// EMA alpha for smoothing agreement scores across steps. Default: 0.1
    pub ema_alpha: f32,
    /// Cached agreement scores from previous step (for EMA smoothing)
    cached_scores: Vec<f32>,
}

impl MarginalBestBuddyAligner {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            ema_alpha: 0.1,
            cached_scores: Vec::new(),
        }
    }

    pub fn with_ema_alpha(mut self, alpha: f32) -> Self {
        self.ema_alpha = alpha;
        self
    }

    /// Filter marginals by mutual agreement between draft and target.
    ///
    /// For each position, computes Pearson correlation between draft and target
    /// marginals. Positions with agreement below `threshold` have their marginals
    /// dampened (scaled by the agreement score). Positions with high agreement
    /// pass through unchanged.
    ///
    /// After dampening, marginals are re-normalized so each position sums to 1.0.
    /// If all marginals at a position are zeroed out (complete disagreement),
    /// the original draft marginals are kept as fallback.
    ///
    /// Returns filtered marginals owned by the caller.
    pub fn filter_marginals<'a>(
        &mut self,
        draft_marginals: &[&[f32]],
        target_marginals: &[&[f32]],
    ) -> Vec<Vec<f32>> {
        let seq_len = draft_marginals.len().min(target_marginals.len());
        let mut filtered = Vec::with_capacity(seq_len);

        // Compute per-position alignment confidence
        let mut confidence = vec![0.0f32; seq_len];
        for i in 0..seq_len {
            let len = draft_marginals[i].len().min(target_marginals[i].len());
            if len == 0 {
                continue;
            }
            confidence[i] =
                pearson_correlation(&draft_marginals[i][..len], &target_marginals[i][..len]);
        }

        // EMA-smooth with cached scores from previous step
        if !self.cached_scores.is_empty() {
            let prev_len = self.cached_scores.len().min(seq_len);
            for i in 0..prev_len {
                confidence[i] =
                    self.ema_alpha * confidence[i] + (1.0 - self.ema_alpha) * self.cached_scores[i];
            }
        }
        self.cached_scores = confidence.clone();

        // Apply agreement-based dampening
        for i in 0..seq_len {
            let agreement = self.mutual_agreement(draft_marginals[i], target_marginals[i]);
            let src = draft_marginals[i];

            if agreement >= self.threshold {
                // High agreement — pass through unchanged
                filtered.push(src.to_vec());
            } else {
                // Low agreement — dampen by agreement score
                let mut row: Vec<f32> = src.iter().map(|&v| v * agreement).collect();
                let sum: f32 = row.iter().sum();
                if sum > f32::EPSILON {
                    for v in &mut row {
                        *v /= sum;
                    }
                } else {
                    // Complete disagreement — fallback to original
                    row = src.to_vec();
                }
                filtered.push(row);
            }
        }

        filtered
    }
}

impl Default for MarginalBestBuddyAligner {
    fn default() -> Self {
        Self::new(0.3)
    }
}

impl BestBuddyAligner for MarginalBestBuddyAligner {
    fn mutual_agreement(&self, draft_top_k: &[f32], target_top_k: &[f32]) -> f32 {
        let corr = pearson_correlation(draft_top_k, target_top_k);
        // Sigmoid maps [-1, 1] → [0, 1] with threshold sensitivity
        1.0 / (1.0 + (-(corr - self.threshold) * 5.0).exp())
    }

    fn batch_alignment_confidence(
        &self,
        draft_logits: &[f32],
        target_logits: &[f32],
        results: &mut [f32],
    ) {
        // logits layout: flat [seq_len * vocab_size]
        // We need vocab_size to stride correctly. Infer it from total length.
        let total = draft_logits.len().min(target_logits.len());
        let seq_len = results.len();
        if seq_len == 0 {
            return;
        }
        let vocab_size = total / seq_len;
        if vocab_size == 0 {
            return;
        }

        for i in 0..seq_len {
            let offset = i * vocab_size;
            let draft_end = draft_logits.len().min(offset + vocab_size);
            let target_end = target_logits.len().min(offset + vocab_size);
            let len = draft_end.min(target_end) - offset;
            if len == 0 {
                results[i] = 0.0;
                continue;
            }
            results[i] = pearson_correlation(
                &draft_logits[offset..offset + len],
                &target_logits[offset..offset + len],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutual_agreement_high_correlation() {
        let aligner = MarginalBestBuddyAligner::default();
        // Identical distributions → correlation = 1.0 → well above threshold 0.3
        let draft = [0.1, 0.2, 0.3, 0.4];
        let target = [0.1, 0.2, 0.3, 0.4];
        let score = aligner.mutual_agreement(&draft, &target);
        assert!(
            score > 0.9,
            "identical distributions should give high agreement, got {score}"
        );
    }

    #[test]
    fn test_mutual_agreement_anti_correlation() {
        let aligner = MarginalBestBuddyAligner::default();
        let draft = [1.0, 2.0, 3.0, 4.0, 5.0];
        let target = [10.0, 8.0, 6.0, 4.0, 2.0];
        let score = aligner.mutual_agreement(&draft, &target);
        // Anti-correlation → corr = -1.0 → far below threshold 0.3 → low score
        assert!(
            score < 0.1,
            "anti-correlated distributions should give low agreement, got {score}"
        );
    }

    #[test]
    fn test_mutual_agreement_range() {
        let aligner = MarginalBestBuddyAligner::default();
        // Any input should produce a score in [0, 1]
        let cases: &[(&[f32], &[f32])] = &[
            (&[0.1, 0.2, 0.3], &[0.3, 0.2, 0.1]),
            (&[1.0, 2.0], &[2.0, 4.0]),
            (&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]),
        ];
        for (draft, target) in cases {
            let score = aligner.mutual_agreement(draft, target);
            assert!(
                (0.0..=1.0).contains(&score),
                "score {score} out of [0,1] for draft={draft:?}, target={target:?}"
            );
        }
    }

    #[test]
    fn test_default_threshold() {
        let aligner = MarginalBestBuddyAligner::default();
        assert!(
            (aligner.threshold - 0.3).abs() < 1e-6,
            "default threshold should be 0.3"
        );
    }

    #[test]
    fn test_batch_alignment_confidence() {
        let aligner = MarginalBestBuddyAligner::default();
        // 2 positions × 4 vocab
        let draft = [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1];
        let target = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4];
        let mut results = [0.0f32; 2];
        aligner.batch_alignment_confidence(&draft, &target, &mut results);

        // Position 0: identical → corr ≈ 1.0
        assert!(
            (results[0] - 1.0).abs() < 1e-6,
            "position 0 should be perfectly correlated, got {}",
            results[0]
        );
        // Position 1: reversed → corr ≈ -1.0
        assert!(
            (results[1] + 1.0).abs() < 1e-6,
            "position 1 should be anti-correlated, got {}",
            results[1]
        );
    }

    #[test]
    fn test_with_ema_alpha() {
        let aligner = MarginalBestBuddyAligner::new(0.5).with_ema_alpha(0.2);
        assert!(
            (aligner.ema_alpha - 0.2).abs() < 1e-6,
            "ema_alpha should be 0.2"
        );
        assert!(
            (aligner.threshold - 0.5).abs() < 1e-6,
            "threshold should be 0.5"
        );
    }
}

// TL;DR: MarginalBestBuddyAligner filters speculative decode candidates via Pearson
// correlation between draft and target marginals. Feature-gated `best_buddies`.
