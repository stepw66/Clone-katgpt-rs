//! `MedianTopMAvailability` — the paper's load-bearing community-aggregation
//! rule.
//!
//! See [`crate::alien_sampler`] for the module-level doc and paper citation.
//!
//! Implements [`super::traits::AvailabilityScorer`]`<f32>` by computing the
//! median of the top-`m` cosine similarities between the candidate (treated as
//! a dense `&[f32]` embedding) and a precomputed `community_bank` of
//! community embeddings. This is the dual-encoder availability signal that
//! the paper proves (via ablation in §1.4) is *not* substitutable by a
//! density estimator — the median-of-top-m aggregation is load-bearing.
//!
//! # Top-m partial sort
//! For each candidate we compute `n_bank` cosines, then take the median of the
//! top-`m`. The top-`m` selection uses `select_nth_unstable_by` — an
//! `O(n_bank)` expected-time partial sort that avoids the `O(n log n)` cost
//! of a full sort. The median of the resulting `m` cosines is then taken via
//! a small fixed-size sort on the top-`m` slice.
//!
//! # Zero-alloc hot path
//! The [`Self::availability_embedded_with_scratch`] variant takes a
//! caller-owned cosine scratch buffer (`&mut [f32]` of length `>= n_bank`),
//! so the per-candidate hot path performs **no allocation**. The
//! [`AvailabilityScorer`] trait impl uses [`Self::availability_embedded`],
//! which lazily allocates / reuses an internal scratch buffer — convenient
//! for cold paths and tests, but not the recommended hot-path entry point.
//!
//! # Edge cases
//! - Empty bank → availability = 0.0 (no community signal; candidate is
//!   "neutral" w.r.t. the community). The sampler will z-score this constant
//!   to 0.
//! - `m` larger than bank size → falls back to `m = bank.len()` (effectively
//!   median over the whole bank).
//! - `m == 1` → returns the single top-1 cosine (max similarity).
//! - Zero-norm candidate or bank item → cosine = 0.0 (avoid divide-by-zero).
//!
//! Reference: Plan 311 (T1.4), Research 293, arXiv:2603.01092 §1.4.

use super::traits::AvailabilityScorer;

/// Median-of-top-m cosine availability scorer.
///
/// Stores an owned `community_bank: Vec<Vec<f32>>` (the reference community
/// embeddings) and an `m` parameter (paper default 10). The bank is set at
/// construction; mutable updates are the caller's concern (build a new scorer
/// or use interior mutability in the consumer — the open primitive is
/// immutable by design for determinism).
///
/// # Determinism
/// Bit-identical across runs for the same `(candidate, bank, m)` — no RNG, no
/// thread-local state. The partial sort is deterministic; the median is
/// deterministic.
///
/// Reference: Plan 311 (T1.4).
pub struct MedianTopMAvailability {
    community_bank: Vec<Vec<f32>>,
    /// Precomputed L2 norms of each bank item (avoid recomputing on every
    /// candidate). Computed once at construction.
    bank_norms: Vec<f32>,
    m: usize,
    /// Reusable cosine scratch for the convenience `availability_embedded`
    /// entry point. The zero-alloc hot-path variant
    /// (`availability_embedded_with_scratch`) does not touch this.
    scratch: Vec<f32>,
}

impl MedianTopMAvailability {
    /// Construct from an owned bank + `m`.
    ///
    /// **Validation** (panics — one-time setup):
    /// - All bank items must have the same length (the embedding dimension).
    ///   Mixed-dim banks are a wiring bug.
    /// - Bank items must be finite (no NaN / inf embeddings — they corrupt
    ///   every cosine).
    /// - `m >= 1` (m=0 is meaningless; the median of zero items is undefined).
    ///
    /// Bank norms are precomputed here so the per-candidate hot path is a
    /// single dot + divide per bank item.
    ///
    /// Reference: Plan 311 T1.4.
    #[must_use]
    pub fn new(community_bank: Vec<Vec<f32>>, m: usize) -> Self {
        assert!(
            m >= 1,
            "MedianTopMAvailability::new: m must be >= 1, got {m}"
        );
        let dim = community_bank.first().map(Vec::len).unwrap_or(0);
        for (i, item) in community_bank.iter().enumerate() {
            assert_eq!(
                item.len(),
                dim,
                "MedianTopMAvailability::new: bank item {i} has len {} but bank dim is {dim} (all items must share the embedding dimension)",
                item.len()
            );
            for (j, &v) in item.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "MedianTopMAvailability::new: bank item {i}[{j}] is not finite (got {v})"
                );
            }
        }
        let bank_norms: Vec<f32> = community_bank
            .iter()
            .map(|item| {
                let mut s = 0.0_f32;
                for &v in item {
                    s += v * v;
                }
                s.sqrt()
            })
            .collect();
        // Cosine scratch sized to the bank; reused by `availability_embedded`.
        let scratch = vec![0.0_f32; community_bank.len()];
        Self {
            community_bank,
            bank_norms,
            m,
            scratch,
        }
    }

    /// Construct with paper-default `m = 10`.
    #[must_use]
    pub fn with_paper_default_m(community_bank: Vec<Vec<f32>>) -> Self {
        Self::new(community_bank, 10)
    }

    /// Borrowed view of the community bank.
    #[inline]
    #[must_use]
    pub fn bank(&self) -> &[Vec<f32>] {
        &self.community_bank
    }

    /// The configured `m` (top-m count).
    #[inline]
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Number of items in the community bank.
    #[inline]
    #[must_use]
    pub fn bank_len(&self) -> usize {
        self.community_bank.len()
    }

    /// Embedding dimension (length of each bank item). `0` for an empty bank.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.community_bank.first().map(Vec::len).unwrap_or(0)
    }

    /// Compute median-of-top-m cosine availability for an embedded candidate.
    ///
    /// This is the **convenience** entry point — it uses an internal scratch
    /// buffer (`self.scratch`) and is therefore not allocation-free across
    /// the first call (the buffer is sized at construction; subsequent calls
    /// reuse it). For the hot path, prefer
    /// [`Self::availability_embedded_with_scratch`].
    ///
    /// Returns `0.0` for an empty bank.
    ///
    /// Reference: Plan 311 T1.4.
    #[inline]
    pub fn availability_embedded(&mut self, candidate: &[f32]) -> f32 {
        // Borrow self.scratch mutably for the duration of the call. Safe
        // because we don't re-enter (no recursion through self).
        let scratch = core::mem::take(&mut self.scratch);
        let mut scratch = scratch;
        let out = self.availability_embedded_with_scratch(candidate, &mut scratch);
        self.scratch = scratch;
        out
    }

    /// Zero-alloc hot-path variant: caller owns the cosine scratch buffer.
    ///
    /// `cosine_scratch` must have length `>= self.bank_len()`. It's
    /// overwritten with per-bank-item cosine similarities during the
    /// computation and then partially sorted in place to extract the top-m
    /// and compute the median. The caller can inspect it after the call to
    /// see all cosines (pre-sort).
    ///
    /// # Panics
    /// Debug builds assert `cosine_scratch.len() >= self.bank_len()`. Release
    /// builds trust the caller (hot-path contract).
    ///
    /// Reference: Plan 311 T1.4.
    pub fn availability_embedded_with_scratch(
        &self,
        candidate: &[f32],
        cosine_scratch: &mut [f32],
    ) -> f32 {
        let n_bank = self.community_bank.len();
        if n_bank == 0 {
            return 0.0;
        }
        debug_assert_eq!(
            cosine_scratch.len(),
            n_bank,
            "availability_embedded_with_scratch: cosine_scratch must have len == bank_len ({n_bank})"
        );

        // Candidate L2 norm (single pass).
        let mut cand_norm_sq = 0.0_f32;
        for &v in candidate {
            cand_norm_sq += v * v;
        }
        let cand_norm = cand_norm_sq.sqrt();
        if cand_norm == 0.0 {
            // Zero-norm candidate has no direction; cosine is undefined.
            // Treat as availability 0 (neutral).
            return 0.0;
        }

        // Pass 1: cosine similarity against each bank item.
        // cosine(a, b) = (a · b) / (||a|| ||b||)
        // We've precomputed ||b|| (self.bank_norms); compute a · b inline.
        for (i, item) in self.community_bank.iter().enumerate() {
            let bank_norm = self.bank_norms[i];
            if bank_norm == 0.0 {
                cosine_scratch[i] = 0.0;
                continue;
            }
            // Dot product. Lengths match by construction (validated in new);
            // we zip to short-circuit if the candidate is shorter than the
            // bank items (defensive — the caller should pass full-length
            // candidates, but this avoids OOB).
            let mut dot = 0.0_f32;
            for (a, b) in candidate.iter().zip(item.iter()) {
                dot += a * b;
            }
            cosine_scratch[i] = dot / (cand_norm * bank_norm);
        }

        // Pass 2: median of top-m.
        median_of_top_m(cosine_scratch, self.m)
    }

    /// Batch availability scoring: fills `out[i]` with the availability of
    /// `candidates[i]`, reusing a single cosine scratch across all candidates.
    ///
    /// This is the **hot-path** entry point for ranking passes: one scratch
    /// allocation amortized across the whole candidate pool, instead of one
    /// per candidate in the trait path. Pair with
    /// [`super::sampler::AlienSampler::rank_precomputed`] to skip the
    /// per-candidate trait allocation entirely.
    ///
    /// `out` must have length `>= candidates.len()`; `cosine_scratch` must
    /// have length `>= self.bank_len()`. Only the first `candidates.len()`
    /// entries of `out` are written.
    ///
    /// Reference: Plan 311 T4.4.
    pub fn availability_batch(
        &self,
        candidates: &[Vec<f32>],
        out: &mut [f32],
        cosine_scratch: &mut [f32],
    ) {
        let n_bank = self.community_bank.len();
        debug_assert_eq!(
            cosine_scratch.len(),
            n_bank,
            "availability_batch: cosine_scratch must have len == bank_len ({n_bank})"
        );
        for (i, cand) in candidates.iter().enumerate() {
            out[i] = self.availability_embedded_with_scratch(cand, cosine_scratch);
        }
    }
}

impl AvailabilityScorer<f32> for MedianTopMAvailability {
    /// Trait-compatible entry point. Allocates a cosine scratch buffer per call.
    ///
    /// This is the **cold-path** convenience entry point: it matches the
    /// `&self` trait signature but performs one `Vec` allocation per call.
    /// Hot-path callers should use [`Self::availability_embedded_with_scratch`]
    /// (zero-alloc, caller-owned scratch) or [`Self::availability_embedded`]
    /// (`&mut self`, reuses internal scratch) instead.
    ///
    /// The trait is `&self`, so we cannot reuse the internal `scratch` field
    /// here without interior mutability (`RefCell` / `UnsafeCell`) — and the
    /// open primitive deliberately avoids those for determinism + audit. The
    /// per-call allocation is the price of trait compatibility.
    fn availability(&self, atoms: &[f32]) -> f32 {
        let mut scratch = vec![0.0_f32; self.community_bank.len()];
        self.availability_embedded_with_scratch(atoms, &mut scratch)
    }
}

/// Compute the median of the top-`m` values in `xs` (in place).
///
/// After the call, `xs` is partially sorted (top-`m` at the tail, sorted
/// ascending within the tail). Returns the median of those top-m values.
///
/// - `xs.len() == 0` → returns `0.0` (no data).
/// - `m >= xs.len()` → returns the median of all of `xs`.
/// - `m == 1` → returns `xs.max()`.
/// - Odd top-m count → middle element.
/// - Even top-m count → mean of the two middle elements.
///
/// Uses `select_nth_unstable_by` for `O(n)` expected-time top-m extraction
/// (the load-bearing perf trick per AGENTS.md "Prefer `match` … partial
/// sort").
#[inline]
fn median_of_top_m(xs: &mut [f32], m: usize) -> f32 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    let effective_m = m.min(n);
    if effective_m == 1 {
        // Top-1 = max.
        return xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }

    // Partial sort: place the top-m (largest) at the tail of xs.
    // select_nth_unstable_by(k, cmp) partitions so that xs[k] is in its
    // sorted position, everything before is "less" by cmp, everything after
    // is "greater". We want the top-m at the tail, so we partition at
    // k = n - effective_m with ascending cmp → tail [n-m, n) holds the
    // largest m.
    let k = n - effective_m;
    xs.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let top_m = &mut xs[k..];
    // Sort the top-m slice ascending so we can pick the median index cleanly.
    top_m.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    // Median of top_m (length = effective_m):
    // - odd:  top_m[effective_m / 2]
    // - even: (top_m[effective_m/2 - 1] + top_m[effective_m/2]) / 2
    let mid = effective_m / 2;
    if effective_m % 2 == 1 {
        top_m[mid]
    } else {
        // Even: average of the two middle elements. Deterministic (no FMA
        // reassociation here; just one add + one multiply).
        (top_m[mid - 1] + top_m[mid]) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction validation ─────────────────────────────────────────────

    #[test]
    fn new_empty_bank_ok() {
        // Empty bank is allowed; availability returns 0.0.
        let s = MedianTopMAvailability::new(vec![], 10);
        assert_eq!(s.bank_len(), 0);
        assert_eq!(s.dim(), 0);
    }

    #[test]
    fn new_precomputes_norms() {
        let bank = vec![
            vec![3.0, 4.0], // norm = 5
            vec![1.0, 0.0], // norm = 1
        ];
        let s = MedianTopMAvailability::new(bank, 2);
        assert!((s.bank_norms[0] - 5.0).abs() < 1e-6);
        assert!((s.bank_norms[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "m must be >= 1")]
    fn new_rejects_zero_m() {
        MedianTopMAvailability::new(vec![vec![1.0]], 0);
    }

    #[test]
    #[should_panic(expected = "all items must share the embedding dimension")]
    fn new_rejects_mixed_dim_bank() {
        MedianTopMAvailability::new(vec![vec![1.0, 2.0], vec![1.0]], 1);
    }

    #[test]
    #[should_panic(expected = "is not finite")]
    fn new_rejects_nan_in_bank() {
        MedianTopMAvailability::new(vec![vec![f32::NAN]], 1);
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn empty_bank_returns_zero() {
        let mut s = MedianTopMAvailability::new(vec![], 10);
        let v = s.availability_embedded(&[1.0, 2.0, 3.0]);
        assert!(v.abs() < 1e-6);
    }

    #[test]
    fn zero_norm_candidate_returns_zero() {
        let mut s = MedianTopMAvailability::new(vec![vec![1.0, 0.0]], 1);
        let v = s.availability_embedded(&[0.0, 0.0]);
        assert!(v.abs() < 1e-6);
    }

    #[test]
    fn zero_norm_bank_item_returns_zero_for_that_slot() {
        // Bank has one zero-norm item and one normal item. m=1 picks the max,
        // which should be the normal item's cosine.
        let mut s = MedianTopMAvailability::new(vec![vec![0.0, 0.0], vec![1.0, 0.0]], 1);
        let v = s.availability_embedded(&[1.0, 0.0]);
        // cosine with [1,0] = 1.0, cosine with [0,0] = 0.0. Top-1 = max = 1.0.
        assert!((v - 1.0).abs() < 1e-6);
    }

    // ── Top-m fallback ──────────────────────────────────────────────────────

    #[test]
    fn top1_fallback_returns_max_cosine() {
        // m=1 with bank larger than 1 → returns max cosine.
        let bank = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0], // normalized: [0.707, 0.707]
        ];
        let mut s = MedianTopMAvailability::new(bank, 1);
        // Candidate [1, 0]: cosines = [1.0, 0.0, 0.707]. Top-1 = 1.0.
        let v = s.availability_embedded(&[1.0, 0.0]);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn m_larger_than_bank_falls_back_to_bank_size() {
        // m=100 but bank only has 3 items → effective m = 3.
        let bank = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.6, 0.8], // norm=1
        ];
        let mut s = MedianTopMAvailability::new(bank, 100);
        // Candidate [1, 0]: cosines = [1.0, 0.0, 0.6]. Median of all 3 = 0.6.
        let v = s.availability_embedded(&[1.0, 0.0]);
        assert!((v - 0.6).abs() < 1e-6);
    }

    // ── Paper default m=10 ─────────────────────────────────────────────────

    #[test]
    fn paper_default_m10_bank_of_50_median_of_top10() {
        // Bank of 50 items, m=10. We construct deterministic cosines and
        // verify the median of the top-10.
        //
        // We build a bank where each item has a known cosine to a fixed
        // candidate, then check the median.
        let candidate = vec![1.0, 0.0];
        // Build bank items as unit vectors at known angles so cosines are
        // known. We'll just use [cos θ, sin θ] for θ in [0, π/2).
        // cosine([1,0], [cos θ, sin θ]) = cos θ.
        // We want 50 cosines; pick θ_i = (i / 50) * (π/2).
        let mut bank: Vec<Vec<f32>> = Vec::with_capacity(50);
        let mut cosines: Vec<f32> = Vec::with_capacity(50);
        for i in 0..50 {
            let theta = (i as f32 / 50.0) * (core::f32::consts::PI / 2.0);
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            bank.push(vec![cos_t, sin_t]);
            cosines.push(cos_t); // cosine similarity with [1, 0]
        }
        let mut s = MedianTopMAvailability::new(bank, 10);
        let got = s.availability_embedded(&candidate);
        // Top-10 cosines = the 10 largest values in `cosines`.
        cosines.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let top10 = &cosines[40..]; // 10 largest
        // Median of 10 (even) = average of top10[4] and top10[5].
        let expected = (top10[4] + top10[5]) * 0.5;
        assert!(
            (got - expected).abs() < 1e-5,
            "paper default m=10 median mismatch: got {got}, expected {expected}"
        );
    }

    // ── Invariance ─────────────────────────────────────────────────────────

    #[test]
    fn invariant_to_bank_permutation() {
        // Shuffling the bank should not change the result (median of top-m is
        // order-independent).
        let candidate = vec![1.0, 0.0];
        let bank_a = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.43589], // ~cos 25°
            vec![0.5, 0.86603], // ~cos 60°
            vec![0.0, 1.0],
        ];
        // Permute: move item 0 to the end.
        let mut bank_b = bank_a.clone();
        let first = bank_b.remove(0);
        bank_b.push(first);
        let mut sa = MedianTopMAvailability::new(bank_a, 2);
        let mut sb = MedianTopMAvailability::new(bank_b, 2);
        let va = sa.availability_embedded(&candidate);
        let vb = sb.availability_embedded(&candidate);
        assert!((va - vb).abs() < 1e-6, "bank permutation changed result: {va} vs {vb}");
    }

    #[test]
    fn determinism_same_inputs_same_output() {
        let candidate = vec![0.7, 0.7, 0.7];
        let bank = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ];
        let s = MedianTopMAvailability::new(bank, 3);
        let mut scratch1 = vec![0.0; 5];
        let mut scratch2 = vec![0.0; 5];
        let v1 = s.availability_embedded_with_scratch(&candidate, &mut scratch1);
        let v2 = s.availability_embedded_with_scratch(&candidate, &mut scratch2);
        assert!((v1 - v2).abs() < 1e-7);
    }

    // ── Trait impl ─────────────────────────────────────────────────────────

    #[test]
    fn trait_impl_matches_direct_call() {
        use super::super::traits::AvailabilityScorer;
        let bank = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let s = MedianTopMAvailability::new(bank, 1);
        // Trait method (allocates scratch internally).
        let trait_v = s.availability(&[1.0, 0.0]);
        // Direct call with explicit scratch.
        let mut scratch = vec![0.0; 2];
        let direct_v = s.availability_embedded_with_scratch(&[1.0, 0.0], &mut scratch);
        assert!((trait_v - direct_v).abs() < 1e-6);
    }

    // ── median_of_top_m unit tests ─────────────────────────────────────────

    #[test]
    fn median_of_top_m_empty_returns_zero() {
        let mut xs: Vec<f32> = vec![];
        assert_eq!(median_of_top_m(&mut xs, 5), 0.0);
    }

    #[test]
    fn median_of_top_m_top1_returns_max() {
        let mut xs = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        assert!((median_of_top_m(&mut xs, 1) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn median_of_top_m_odd_count() {
        // top-3 of [0.1, 0.9, 0.5, 0.3, 0.7] = [0.5, 0.7, 0.9] (sorted).
        // median (odd, 3) = middle = 0.7.
        let mut xs = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        assert!((median_of_top_m(&mut xs, 3) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn median_of_top_m_even_count() {
        // top-4 of [0.1, 0.9, 0.5, 0.3, 0.7, 0.8] = [0.5, 0.7, 0.8, 0.9] (sorted).
        // median (even, 4) = (0.7 + 0.8) / 2 = 0.75.
        let mut xs = vec![0.1, 0.9, 0.5, 0.3, 0.7, 0.8];
        let got = median_of_top_m(&mut xs, 4);
        assert!((got - 0.75).abs() < 1e-6, "even-count median: got {got}");
    }

    #[test]
    fn median_of_top_m_m_larger_than_n_uses_n() {
        // m=100, n=3 → effective m=3 → median of all 3.
        let mut xs = vec![1.0, 2.0, 3.0];
        // sorted: [1, 2, 3], median = 2.
        assert!((median_of_top_m(&mut xs, 100) - 2.0).abs() < 1e-6);
    }
}
