//! Core types for the AC-GPT arbitrary-conditional prefix primitive.

/// AC-GPT-style arbitrary-conditional prefix. Borrowed; zero owning
/// allocations.
///
/// See the [module docs](super) for the three-region attention rule and the
/// leakage-prevention argument.
pub struct AcPrefix<'a> {
    base_tokens: &'a [u32],
    /// Sorted ascending; each entry indexes INTO `base_tokens`.
    conditioning_positions: &'a [usize],
}

impl<'a> AcPrefix<'a> {
    /// Empty conditioning set — degenerates to a vanilla causal forward (this
    /// is the G3 invariant: `AcPrefix::empty(tokens)` must be bit-identical to
    /// a forward without `AcPrefix` at all).
    pub fn empty(base_tokens: &'a [u32]) -> Self {
        Self {
            base_tokens,
            conditioning_positions: &[],
        }
    }

    /// Construct from a sorted, in-range conditioning-position slice.
    ///
    /// `debug_assert`s (cheap, stripped in release):
    /// - `conditioning_positions` is sorted strictly ascending.
    /// - Every entry is `< base_tokens.len()`.
    pub fn new(base_tokens: &'a [u32], conditioning_positions: &'a [usize]) -> Self {
        debug_assert!(
            conditioning_positions.windows(2).all(|w| w[0] < w[1]),
            "conditioning_positions must be strictly ascending"
        );
        debug_assert!(
            conditioning_positions
                .iter()
                .all(|&p| p < base_tokens.len()),
            "conditioning_positions must index into base_tokens"
        );
        Self {
            base_tokens,
            conditioning_positions,
        }
    }

    #[inline]
    pub fn base_tokens(&self) -> &'a [u32] {
        self.base_tokens
    }

    #[inline]
    pub fn conditioning_positions(&self) -> &'a [usize] {
        self.conditioning_positions
    }

    /// Number of conditioning copies placed at the front.
    #[inline]
    pub fn xc_len(&self) -> usize {
        self.conditioning_positions.len()
    }

    /// Length of the augmented sequence: `|xc|` copies at the front + `|x|`
    /// original tokens.
    #[inline]
    pub fn augmented_len(&self) -> usize {
        self.base_tokens.len() + self.conditioning_positions.len()
    }

    /// Original position lookup for augmented slot `k`:
    ///   - `k < |xc|`               → `conditioning_positions[k]` (the copy
    ///     carries its source position so RoPE applies the correct rotation).
    ///   - `|xc| <= k < augmented`  → `k - |xc|` (identity position in the
    ///     original sequence).
    ///
    /// Branch-free, zero-allocation. Used by [`Self::attends`] in the r1-r1
    /// case (where it collapses to `k - |xc|`) and by
    /// [`Self::original_positions_into`].
    #[inline]
    pub fn original_pos(&self, k: usize) -> usize {
        let xc = self.conditioning_positions.len();
        if k < xc {
            // SAFETY-equivalent: bounds-checked indexing; debug builds catch OOB.
            self.conditioning_positions[k]
        } else {
            k - xc
        }
    }

    /// Write the original position for each augmented slot into `out`.
    ///
    /// The first `|xc|` slots are the conditioning copies (carry their source
    /// position `conditioning_positions[k]`); the remaining `|x|` slots are
    /// the original tokens (carry identity positions `0..|x|`).
    ///
    /// `debug_assert`s `out.len() == augmented_len()`.
    pub fn original_positions_into(&self, out: &mut [usize]) {
        let xc = self.conditioning_positions.len();
        let base_len = self.base_tokens.len();
        debug_assert_eq!(
            out.len(),
            xc + base_len,
            "out.len() must equal augmented_len"
        );
        out[..xc].copy_from_slice(self.conditioning_positions);
        for k in 0..base_len {
            out[xc + k] = k;
        }
    }

    /// Three-region attention rule — see the [module docs](super).
    ///
    /// Branch-free inner expression (boolean `&` / `|`, no short-circuit, no
    /// allocation, O(1)). In region r1 the `original_pos(k) = k - |xc|` offset
    /// cancels in the causal comparison, so `original_pos(i) >= original_pos(j)`
    /// collapses to `i >= j` — no conditioning_positions lookup needed on the
    /// hot path.
    #[inline]
    pub fn attends(&self, i: usize, j: usize) -> bool {
        // Region partition:
        //   r0 = [0, |xc|)         — conditioning copies
        //   r1 = [|xc|, augmented) — original sequence positions
        //
        // Truth table:
        //   (i ∈ r0, j ∈ r0) → true
        //   (i ∈ r1, j ∈ r0) → true
        //   (i ∈ r0, j ∈ r1) → false
        //   (i ∈ r1, j ∈ r1) → i >= j   (original_pos offset cancels in r1)
        //
        // Compact form: `j_in_r0 OR (both_in_r1 AND i >= j)`.
        // When j ∈ r0, the second clause is false (both_in_r1 requires j ∈ r1),
        // so the result is true regardless of i. When j ∈ r1, the first clause
        // is false; the result is then `both_in_r1 AND i >= j`, which is false
        // if i ∈ r0 (both_in_r1 = false) and `i >= j` if i ∈ r1.
        let xc = self.conditioning_positions.len();
        let j_in_r0 = j < xc;
        let i_in_r1 = i >= xc;
        let j_in_r1 = j >= xc;
        let both_in_r1 = i_in_r1 & j_in_r1;
        let causal_in_r1 = i >= j;
        j_in_r0 | (both_in_r1 & causal_in_r1)
    }

    /// Write the loss mask into `out`:
    ///   - `0.0` for slots in region 0 (the copies — never part of the loss).
    ///   - `0.0` for slots in region 1 whose original position is in
    ///     `conditioning_positions` (these are the in-place conditioning
    ///     tokens, not eval).
    ///   - `1.0` for all other slots in region 1 (the eval positions `xe`).
    ///
    /// Membership check uses `slice::binary_search` on the sorted
    /// `conditioning_positions` — O(log |xc|) per slot, zero allocation.
    /// (Hot-path alternative would be a precomputed `Vec<bool>` lookup table;
    /// not used here because `loss_mask_into` runs once per forward, not per
    /// (i,j) pair.)
    ///
    /// `debug_assert`s `out.len() == augmented_len()`.
    pub fn loss_mask_into(&self, out: &mut [f32]) {
        let xc = self.conditioning_positions.len();
        let base_len = self.base_tokens.len();
        debug_assert_eq!(
            out.len(),
            xc + base_len,
            "out.len() must equal augmented_len"
        );
        // Region 0: copies are never in the loss.
        for slot in 0..xc {
            out[slot] = 0.0;
        }
        // Region 1: original sequence positions.
        let xc_positions = self.conditioning_positions;
        for k in 0..base_len {
            let is_conditioning = xc_positions.binary_search(&k).is_ok();
            out[xc + k] = if is_conditioning { 0.0 } else { 1.0 };
        }
    }
}

/// Bit-packed attention mask for the augmented sequence.
///
/// Layout: `augmented_len × augmented_len` bits, row-major. The bit at offset
/// `(i * augmented_len + j)` encodes `attends(i, j)`. The row length
/// (`augmented_len`) is **not** stored — callers pass it back to [`Self::get`]
/// so the struct stays a single-field transparent wrapper.
#[repr(transparent)]
pub struct AcPrefixMask {
    bits: Box<[u64]>,
}

impl AcPrefixMask {
    /// Bit-pack the [`AcPrefix::attends`] rule over the full
    /// `augmented_len × augmented_len` grid into a `Box<[u64]>` of size
    /// `ceil(augmented_len² / 64)`.
    ///
    /// This is the only allocating call in the module — run it once per
    /// augmented sequence for batched attention kernels that want a
    /// materialized mask. Hot-path callers should prefer
    /// [`AcPrefix::attends`] directly.
    pub fn materialize_from(prefix: &AcPrefix<'_>) -> Self {
        let n = prefix.augmented_len();
        let total_bits = n.checked_mul(n).expect("augmented_len squared overflows");
        let words = total_bits.div_ceil(64);
        let mut bits = vec![0u64; words].into_boxed_slice();
        // Word-stride outer loop so the compiler can hoist the row base.
        for i in 0..n {
            let row_base = i * n;
            for j in 0..n {
                if prefix.attends(i, j) {
                    let bit = row_base + j;
                    // SAFETY: bit < n*n <= words*64, so bit/64 is in bounds.
                    bits[bit / 64] |= 1u64 << (bit % 64);
                }
            }
        }
        Self { bits }
    }

    /// Number of 64-bit words in the packed buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Read the `attends(i, j)` bit. `row_len` must equal the `augmented_len`
    /// passed to [`Self::materialize_from`].
    #[inline]
    pub fn get(&self, i: usize, j: usize, row_len: usize) -> bool {
        let bit = i * row_len + j;
        (self.bits[bit / 64] >> (bit % 64)) & 1 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small fixture: base_len=4, xc_positions=[1,3].
    ///   augmented_len = 6
    ///   r0 = [0, 2)            — copies (original positions 1 and 3)
    ///   r1 = [2, 6)            — original tokens (original positions 0,1,2,3)
    fn small_prefix<'a>(base: &'a [u32]) -> AcPrefix<'a> {
        // base.len() must be >= 4 for [1,3] to be in-range.
        assert!(base.len() >= 4);
        AcPrefix::new(base, &[1, 3])
    }

    #[test]
    fn augmented_len_empty_and_nonempty() {
        let base = [10u32, 20, 30, 40];
        let empty = AcPrefix::empty(&base);
        assert_eq!(empty.augmented_len(), 4);
        assert_eq!(empty.xc_len(), 0);

        let p = small_prefix(&base);
        assert_eq!(p.augmented_len(), 6);
        assert_eq!(p.xc_len(), 2);
    }

    #[test]
    fn original_positions_into_matches_layout() {
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);
        let mut out = [0usize; 6];
        p.original_positions_into(&mut out);
        // First 2 slots are copies: their source positions are conditioning_positions = [1, 3].
        // Remaining 4 slots are original tokens: identity positions 0..4.
        assert_eq!(out, [1, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn original_positions_into_empty_prefix_is_identity() {
        let base = [10u32, 20, 30];
        let p = AcPrefix::empty(&base);
        let mut out = [0usize; 3];
        p.original_positions_into(&mut out);
        assert_eq!(out, [0, 1, 2]);
    }

    #[test]
    fn attends_three_region_rule_small_example() {
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);
        // augmented_len = 6; r0 = [0,2); r1 = [2,6).

        // (i ∈ r0, j ∈ r0) → true  (copies bidirectional)
        assert!(p.attends(0, 0));
        assert!(p.attends(0, 1));
        assert!(p.attends(1, 0));
        assert!(p.attends(1, 1));

        // (i ∈ r1, j ∈ r0) → true  (eval attends to all copies)
        assert!(p.attends(2, 0));
        assert!(p.attends(2, 1));
        assert!(p.attends(5, 0));
        assert!(p.attends(5, 1));

        // (i ∈ r0, j ∈ r1) → false (copies don't attend back to originals)
        assert!(!p.attends(0, 2));
        assert!(!p.attends(0, 5));
        assert!(!p.attends(1, 2));
        assert!(!p.attends(1, 5));

        // (i ∈ r1, j ∈ r1) → i >= j (standard causal in r1; original_pos offset cancels)
        assert!(p.attends(2, 2)); // 2 >= 2
        assert!(p.attends(3, 2)); // 3 >= 2
        assert!(p.attends(5, 2)); // 5 >= 2
        assert!(p.attends(5, 5)); // 5 >= 5
        assert!(!p.attends(2, 3)); // 2 < 3
        assert!(!p.attends(2, 5)); // 2 < 5
        assert!(!p.attends(4, 5)); // 4 < 5
    }

    #[test]
    fn attends_empty_prefix_is_standard_causal() {
        let base = [10u32, 20, 30];
        let p = AcPrefix::empty(&base);
        // augmented_len = 3; r0 is empty so everything is r1.
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(p.attends(i, j), i >= j, "i={i}, j={j}");
            }
        }
    }

    #[test]
    fn loss_mask_into_marks_only_eval_positions() {
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);
        let mut out = [0.0f32; 6];
        p.loss_mask_into(&mut out);
        // r0 copies: always 0.0.
        // r1 positions: original_pos 0 (not in xc) → 1.0
        //               original_pos 1 (in xc)     → 0.0
        //               original_pos 2 (not in xc) → 1.0
        //               original_pos 3 (in xc)     → 0.0
        assert_eq!(out, [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn loss_mask_into_empty_prefix_all_ones() {
        let base = [10u32, 20, 30];
        let p = AcPrefix::empty(&base);
        let mut out = [0.0f32; 3];
        p.loss_mask_into(&mut out);
        assert_eq!(out, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn materialize_from_matches_attends_for_all_pairs() {
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);
        let n = p.augmented_len();
        let mask = AcPrefixMask::materialize_from(&p);

        // Word count = ceil(n*n / 64) = ceil(36/64) = 1.
        assert_eq!(mask.len(), 1);
        assert!(!mask.is_empty());

        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    mask.get(i, j, n),
                    p.attends(i, j),
                    "materialized bit disagrees with attends at (i={i}, j={j})"
                );
            }
        }
    }

    #[test]
    fn materialize_from_empty_prefix_matches_causal() {
        let base = [10u32, 20, 30, 40, 50];
        let p = AcPrefix::empty(&base);
        let n = p.augmented_len();
        let mask = AcPrefixMask::materialize_from(&p);
        // ceil(25/64) = 1 word.
        assert_eq!(mask.len(), 1);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(mask.get(i, j, n), i >= j, "i={i}, j={j}");
            }
        }
    }

    #[test]
    fn materialize_from_large_prefix_spans_multiple_words() {
        // base_len=12, xc=4 → augmented=16 → 256 bits → 4 words.
        let base: Vec<u32> = (0..12).collect();
        let xc: Vec<usize> = vec![1, 4, 7, 10];
        let p = AcPrefix::new(&base, &xc);
        let n = p.augmented_len();
        assert_eq!(n, 16);
        let mask = AcPrefixMask::materialize_from(&p);
        assert_eq!(mask.len(), (16u32 * 16).div_ceil(64) as usize);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    mask.get(i, j, n),
                    p.attends(i, j),
                    "large-case mismatch at (i={i}, j={j})"
                );
            }
        }
    }
}
