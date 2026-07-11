//! `MixedRopeSummarizer` — per-frequency-pair RoPE-aware summary construction.
//!
//! Plan 397 T1.4, Research 379 §1.2(b).
//!
//! # The algorithm
//!
//! For a group of `GS` key vectors `{k_0, ..., k_{GS-1}}` at positions
//! `{p_0, ..., p_{GS-1}}`, and RoPE frequency pairs indexed by `i ∈ [0, D/2)`:
//!
//! - Compute `θ_i = inv_freq[i]` (the RoPE frequency for pair `i`).
//! - Compute the phase range across the group: `Δφ_i = θ_i · (p_max - p_min)`.
//!   The paper uses `θ_i · C` as the threshold proxy (C = chunk/group size).
//! - If `θ_i · span ≥ 2π` (high-frequency): **rotate each key at its position,
//!   then average.** The rotated keys span the full circle, so averaging captures
//!   the mean direction without a systematic phase bias.
//! - If `θ_i · span < 2π` (low-frequency): **average raw keys, then rotate at
//!   the group-mid position.** The raw keys are nearly co-linear (small phase
//!   spread), so averaging in raw space preserves direction, then a single
//!   rotation at the midpoint aligns the summary with the query's RoPE frame.
//!
//! # Why this matters
//!
//! Naive mean-pooling of RoPE-rotated keys produces a "rotation wedge" (Plan 245
//! diagnostic) — the average of vectors at different rotation angles points in a
//! meaningless direction. RTPurbo handles this by projecting to a 16-dim pre-RoPE
//! subspace. HGA's mixed rule is the full-dim alternative: it applies the correct
//! averaging strategy per frequency pair.
//!
//! # Threshold derivation
//!
//! `θ_threshold = 2π / C` where C is the group span. Pairs with `θ_i ≥ θ_threshold`
//! are "high-frequency". This must be derived from `rope_theta` and C, not hardcoded.
//! For `rope_theta = 10000` and C = 16: the crossover lands at `i` where
//! `10000^(-2i/D) = 2π/16 ≈ 0.393`, i.e., `-2i/D · log(10000) = log(0.393)`,
//! `i ≈ D/2 · log(0.393) / log(1/10000) ≈ D/2 · 0.0998`.
//! For `rope_theta = 1000000` and C = 16: crossover at `i ≈ D/2 · 0.060`.
//!
//! The summarizer takes `inv_freq` (pre-computed from `rope_theta`) and `span`
//! as parameters, so it works for any `rope_theta`.

use crate::simd::simd_dot_f32;

/// Per-frequency-pair RoPE-aware mixed-frequency summarizer.
///
/// Pre-computes the high/low frequency mask and mid-position rotation tables
/// for a given group span and RoPE frequency set.
pub struct MixedRopeSummarizer {
    /// Head dimension D.
    head_dim: usize,
    /// D/2 — number of RoPE frequency pairs.
    half: usize,
    /// Group span (the position range over which averaging happens, typically
    /// = group_size for contiguous groups).
    span: usize,
    /// `[half]` — `inv_freq[i] = 1.0 / (rope_theta ^ (2i/D))`.
    inv_freq: Vec<f32>,
    /// `[half]` — true if pair `i` is "high-frequency" (rotate-then-average).
    high_freq_mask: Vec<bool>,
}

impl MixedRopeSummarizer {
    /// Construct from RoPE `inv_freq` and group span.
    ///
    /// - `head_dim` — D.
    /// - `inv_freq` — `[D/2]` RoPE inverse frequencies (`inv_freq[i] = 1/theta^(2i/D)`).
    /// - `span` — the position span of the group (typically group_size for
    ///   contiguous groups, or `p_max - p_min` for non-contiguous).
    pub fn new(head_dim: usize, inv_freq: Vec<f32>, span: usize) -> Self {
        let half = head_dim / 2;
        debug_assert_eq!(inv_freq.len(), half);
        // Threshold: θ_i · span ≥ 2π → high-frequency.
        let threshold = 2.0 * std::f32::consts::PI / (span as f32).max(1.0);
        let high_freq_mask: Vec<bool> = inv_freq
            .iter()
            .map(|&freq| freq.abs() >= threshold)
            .collect();

        Self {
            head_dim,
            half,
            span,
            inv_freq,
            high_freq_mask,
        }
    }

    /// Construct from `rope_theta` and group span (computes `inv_freq` internally).
    pub fn from_rope_theta(head_dim: usize, rope_theta: f32, span: usize) -> Self {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let exp = 2.0 * i as f32 / head_dim as f32;
                1.0 / rope_theta.powf(exp)
            })
            .collect();
        Self::new(head_dim, inv_freq, span)
    }

    /// Summarize a group of key vectors at given positions.
    ///
    /// - `keys_flat` — `[n_tokens * D]` flattened key vectors (the chunk or group).
    /// - `positions` — `[n_tokens]` token positions (must align with keys_flat).
    /// - `group_start` — index of the first token in the group (within keys_flat).
    /// - `n_tokens` — number of tokens in the group.
    ///
    /// Returns `[D]` summary key vector.
    pub fn summarize(
        &self,
        keys_flat: &[f32],
        positions: &[usize],
        group_start: usize,
        n_tokens: usize,
    ) -> Vec<f32> {
        let d = self.head_dim;
        let half = self.half;
        let n = n_tokens;

        if n == 0 {
            return vec![0.0; d];
        }

        // Compute the mid position (mean of the group's positions).
        let mid_pos: f32 = (0..n)
            .map(|t| positions[group_start + t] as f32)
            .sum::<f32>()
            / n as f32;

        let mut summary = vec![0.0f32; d];

        // Process each RoPE frequency pair independently.
        for i in 0..half {
            let freq = self.inv_freq[i];
            let pair_offset = 2 * i; // (x_{2i}, x_{2i+1})

            if self.high_freq_mask[i] {
                // High-frequency: rotate each key at its position, then average.
                let mut sum_x = 0.0f32;
                let mut sum_y = 0.0f32;
                for t in 0..n {
                    let key_idx = group_start + t;
                    let pos = positions[key_idx] as f32;
                    let theta = pos * freq;
                    let (sin_t, cos_t) = theta.sin_cos();
                    let x = keys_flat[key_idx * d + pair_offset];
                    let y = keys_flat[key_idx * d + pair_offset + 1];
                    // RoPE rotation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
                    sum_x += x * cos_t - y * sin_t;
                    sum_y += x * sin_t + y * cos_t;
                }
                summary[pair_offset] = sum_x / n as f32;
                summary[pair_offset + 1] = sum_y / n as f32;
            } else {
                // Low-frequency: average raw keys, then rotate at mid position.
                let mut mean_x = 0.0f32;
                let mut mean_y = 0.0f32;
                for t in 0..n {
                    let key_idx = group_start + t;
                    mean_x += keys_flat[key_idx * d + pair_offset];
                    mean_y += keys_flat[key_idx * d + pair_offset + 1];
                }
                mean_x /= n as f32;
                mean_y /= n as f32;
                // Rotate at mid position.
                let theta = mid_pos * freq;
                let (sin_t, cos_t) = theta.sin_cos();
                summary[pair_offset] = mean_x * cos_t - mean_y * sin_t;
                summary[pair_offset + 1] = mean_x * sin_t + mean_y * cos_t;
            }
        }

        summary
    }

    /// Number of high-frequency pairs (for diagnostics).
    pub fn n_high_freq(&self) -> usize {
        self.high_freq_mask.iter().filter(|&&h| h).count()
    }

    /// Number of low-frequency pairs.
    pub fn n_low_freq(&self) -> usize {
        self.half - self.n_high_freq()
    }

    /// The frequency threshold (2π / span).
    pub fn threshold(&self) -> f32 {
        2.0 * std::f32::consts::PI / (self.span as f32).max(1.0)
    }

    /// The `inv_freq` vector (for testing).
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// The high-freq mask (for testing).
    pub fn high_freq_mask(&self) -> &[bool] {
        &self.high_freq_mask
    }

    /// Compute the crossover frequency-pair index — the highest `i` that is
    /// still "high-frequency". Returns `half` if all pairs are high-freq, 0 if none.
    pub fn crossover_index(&self) -> usize {
        let mut last_high = 0;
        for (i, &is_high) in self.high_freq_mask.iter().enumerate() {
            if is_high {
                last_high = i + 1;
            }
        }
        last_high
    }
}

/// Compute the dot-product score between a query and a summary key.
/// Used by `GroupSummaryCache::score_groups`.
#[inline]
pub fn dot_score(query: &[f32], summary: &[f32]) -> f32 {
    simd_dot_f32(query, summary, query.len())
}
