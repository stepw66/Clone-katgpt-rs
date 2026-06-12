//! BlockTopK router — simplest VortexFlow implementation.
//!
//! Routes via mean key centroids + dot-product top-k selection.
//! Feature gate: `vortex_flow` (Plan 196, Phase 1).

use katgpt_core::simd::simd_argmax_f32;

use super::vortex_flow::{RoutingDecision, VortexFlow, VortexScratch};

// ---------------------------------------------------------------------------
// BlockTopKCache
// ---------------------------------------------------------------------------

/// Cache for BlockTopK router: mean key centroid per block.
#[derive(Debug, Clone)]
pub struct BlockTopKCache {
    /// Centroid vectors: flat `[n_blocks * head_dim]`.
    pub centroids: Vec<f32>,
    /// Number of blocks currently cached.
    pub n_blocks: usize,
    /// Dimension of each centroid.
    pub head_dim: usize,
}

impl BlockTopKCache {
    /// Create a pre-allocated cache.
    pub fn new(n_blocks_capacity: usize, head_dim: usize) -> Self {
        Self {
            centroids: vec![0.0; n_blocks_capacity * head_dim],
            n_blocks: 0,
            head_dim,
        }
    }

    /// Ensure the cache can hold at least `n` blocks.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.centroids.len() < n * self.head_dim {
            self.centroids.resize(n * self.head_dim, 0.0);
        }
    }

    /// Get centroid for block `i`.
    #[inline]
    pub fn centroid(&self, i: usize) -> &[f32] {
        let start = i * self.head_dim;
        &self.centroids[start..start + self.head_dim]
    }
}

// ---------------------------------------------------------------------------
// BlockTopKRouter
// ---------------------------------------------------------------------------

/// BlockTopK router — routes via mean key centroids + dot product top-k.
#[derive(Debug)]
pub struct BlockTopKRouter {
    /// Apply `1/sqrt(head_dim)` scaling to dot products.
    pub scale: bool,
}

impl BlockTopKRouter {
    /// Create a new router with optional scaling.
    pub fn new(scale: bool) -> Self {
        Self { scale }
    }
}

impl VortexFlow for BlockTopKRouter {
    type Cache = BlockTopKCache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        _values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        cache.ensure_capacity(block_idx + 1);
        let block_size = keys.len() / head_dim;
        if block_size == 0 {
            // Zero-length block: zero centroid
            let start = block_idx * head_dim;
            cache.centroids[start..start + head_dim].fill(0.0);
            cache.n_blocks = cache.n_blocks.max(block_idx + 1);
            return;
        }

        // Compute mean of keys → centroid
        let start = block_idx * head_dim;
        let centroid = &mut cache.centroids[start..start + head_dim];
        centroid.fill(0.0);
        for t in 0..block_size {
            let k_start = t * head_dim;
            for d in 0..head_dim {
                centroid[d] += keys[k_start + d];
            }
        }
        let inv = 1.0 / block_size as f32;
        for d in centroid.iter_mut() {
            *d *= inv;
        }

        cache.n_blocks = cache.n_blocks.max(block_idx + 1);
    }

    fn forward_indexer(
        &self,
        query: &[f32],
        cache: &Self::Cache,
        n_blocks: usize,
        top_k: usize,
        scratch: &mut VortexScratch,
    ) -> RoutingDecision {
        if n_blocks == 0 {
            return RoutingDecision::new();
        }

        let hd = query.len();
        let scale = match self.scale {
            true => 1.0 / (hd as f32).sqrt(),
            false => 1.0,
        };

        scratch.ensure_capacity(n_blocks);
        let scores = &mut scratch.scores[..n_blocks];

        // Compute dot(query, centroid[i]) for each block
        // Chunked accumulation for auto-vectorization
        for (i, score) in scores.iter_mut().enumerate().take(n_blocks) {
            let centroid = cache.centroid(i);
            let mut dot0 = 0.0f32;
            let mut dot1 = 0.0f32;
            let mut dot2 = 0.0f32;
            let mut dot3 = 0.0f32;
            let chunks = hd / 4;
            for c in 0..chunks {
                let base = c * 4;
                dot0 += query[base] * centroid[base];
                dot1 += query[base + 1] * centroid[base + 1];
                dot2 += query[base + 2] * centroid[base + 2];
                dot3 += query[base + 3] * centroid[base + 3];
            }
            let mut dot = dot0 + dot1 + dot2 + dot3;
            let rem = hd % 4;
            for d in (hd - rem)..hd {
                dot += query[d] * centroid[d];
            }
            *score = dot * scale;
        }

        // Partial sort to find top-k (reuses scratch buffer)
        let k = top_k.min(n_blocks);
        scratch.indices.clear();
        argtopk_with_scratch(scores, k, &mut scratch.indices, &mut scratch.argtopk_pairs);

        // Build routing decision with sigmoid weights
        let mut decision = RoutingDecision::with_capacity(k);
        for &idx in &scratch.indices[..k] {
            decision.blocks.push(idx);
            // Sigmoid normalization per-block score
            let w = sigmoid(scores[idx]);
            decision.weights.push(w);
        }

        decision
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        BlockTopKCache::new(n_blocks_capacity, head_dim)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find top-k indices from scores (partial sort, descending).
/// Uses a simple selection-based approach — O(n*k) which is fine for small k.
///
/// Reuses `pairs` scratch buffer across calls to avoid per-call allocation.
pub fn argtopk(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    argtopk_with_scratch(scores, k, indices, &mut Vec::new());
}

/// Zero-alloc variant of [`argtopk`] — reuses `pairs` scratch buffer.
///
/// Uses SIMD-optimized path for k ≤ 16 (Plan 256 Phase 1):
/// - k=1: single-pass `simd_argmax_f32`
/// - k≤16: register-based min-heap with SIMD-parallel comparison
/// - k>16: falls back to selection sort
pub fn argtopk_with_scratch(
    scores: &[f32],
    k: usize,
    indices: &mut Vec<usize>,
    pairs: &mut Vec<(usize, f32)>,
) {
    indices.clear();
    let n = scores.len();
    let k = k.min(n);
    if k == 0 {
        return;
    }

    if k == 1 {
        // Fast path: single argmax via SIMD
        let (idx, _) = simd_argmax_f32(scores);
        indices.push(idx);
        return;
    }

    if k <= 16 {
        // SIMD-optimized register top-k for small k (Plan 256)
        argtopk_simd(scores, k, indices);
        return;
    }

    // Fallback: selection sort O(n*k) for large k
    pairs.clear();
    pairs.extend((0..n).map(|i| (i, scores[i])));

    for i in 0..k {
        let mut best = i;
        for j in (i + 1)..n {
            if pairs[j].1 > pairs[best].1 {
                best = j;
            }
        }
        pairs.swap(i, best);
    }

    indices.extend(pairs[..k].iter().map(|(idx, _)| *idx));
}

/// Insert (val, idx) into sorted register (descending). k ≤ 16.
/// Binary search + shift: O(log k + k) ≈ O(k) for small k.
#[inline(always)]
#[allow(dead_code)] // Used by argtopk_simd (NEON) and argtopk_scalar_heap (non-NEON)
fn insert_sorted(
    heap_vals: &mut [f32; 16],
    heap_idxs: &mut [usize; 16],
    k: usize,
    val: f32,
    idx: usize,
) {
    if val <= heap_vals[k - 1] {
        return;
    }
    // Binary search for insertion position (first element < val)
    let mut lo = 0usize;
    let mut hi = k;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if heap_vals[mid] < val {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    // Shift elements down to make room at position `lo`
    for j in (lo + 1..k).rev() {
        heap_vals[j] = heap_vals[j - 1];
        heap_idxs[j] = heap_idxs[j - 1];
    }
    heap_vals[lo] = val;
    heap_idxs[lo] = idx;
}

/// SIMD-accelerated top-k for k ≤ 16 on NEON.
///
/// Strategy: maintain top-k as a sorted register (descending).
/// Phase 1 optimization (Plan 256): batch NEON threshold comparison +
/// lane-indexed insertion. NEON loads 4 scores, compares against threshold
/// via vcgtq_f32, then only inserts lanes that exceed threshold.
/// The inner insertion uses NEON-parallel comparison across the entire
/// top-k register (4 lanes at a time) to find the insertion point,
/// then scalar shift + write for the final placement.
#[cfg(target_arch = "aarch64")]
fn argtopk_simd(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    use core::arch::aarch64::{vcgtq_f32, vdupq_n_f32, vld1q_f32};
    unsafe {
        let n = scores.len();
        let mut heap_vals = [f32::NEG_INFINITY; 16];
        let mut heap_idxs = [0usize; 16];

        let init = k.min(n);
        for i in 0..init {
            heap_vals[i] = *scores.get_unchecked(i);
            heap_idxs[i] = i;
        }
        // Sort initial k elements descending (insertion sort, k≤16)
        for i in 1..init {
            let val = heap_vals[i];
            let idx = heap_idxs[i];
            let mut j = i;
            while j > 0 && heap_vals[j - 1] < val {
                heap_vals[j] = heap_vals[j - 1];
                heap_idxs[j] = heap_idxs[j - 1];
                j -= 1;
            }
            heap_vals[j] = val;
            heap_idxs[j] = idx;
        }

        // Process remaining elements in NEON chunks of 4.
        // Phase 1: NEON threshold filter — skip entire chunk if no lane exceeds.
        // Phase 2: per-lane insertion using NEON-parallel comparison across
        // the sorted register to find insertion points faster than scalar binary search.
        let remaining = n - init;
        let chunks4 = remaining / 4;
        let mut pos = init;

        for _ in 0..chunks4 {
            let v = vld1q_f32(scores.as_ptr().add(pos));
            let thresh = vdupq_n_f32(heap_vals[k - 1]);
            let mask = vcgtq_f32(v, thresh);

            // Extract mask as u32 lanes — check which elements exceed threshold
            let mask_bits: u32 = core::arch::aarch64::vmaxvq_u32(core::mem::transmute::<
                _,
                core::arch::aarch64::uint32x4_t,
            >(mask));

            if mask_bits != 0 {
                // At least one lane exceeded threshold — process qualifying lanes
                let mask_arr: [u32; 4] = core::mem::transmute(mask);
                let vals: [f32; 4] = core::mem::transmute(v);
                for lane in 0..4 {
                    if mask_arr[lane] != 0 {
                        insert_sorted_simd_neon(
                            &mut heap_vals,
                            &mut heap_idxs,
                            k,
                            vals[lane],
                            pos + lane,
                        );
                    }
                }
            }
            pos += 4;
        }

        // Scalar tail
        while pos < n {
            let val = *scores.get_unchecked(pos);
            if val > heap_vals[k - 1] {
                insert_sorted_simd_neon(&mut heap_vals, &mut heap_idxs, k, val, pos);
            }
            pos += 1;
        }

        indices.extend(heap_idxs[..k].iter());
    }
}

/// NEON-accelerated insertion into sorted register (descending, k ≤ 16).
///
/// Uses NEON to compare the candidate value against 4 heap elements at a time,
/// finding the insertion point via SIMD-parallel comparison instead of scalar
/// binary search. The shift is still scalar (k≤16 → max 16 shifts = negligible).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn insert_sorted_simd_neon(
    heap_vals: &mut [f32; 16],
    heap_idxs: &mut [usize; 16],
    k: usize,
    val: f32,
    idx: usize,
) {
    use core::arch::aarch64::{vcgtq_f32, vdupq_n_f32, vld1q_f32};
    unsafe {
        // SIMD-parallel search: compare val against 4 heap elements at a time
        // to find the first position where heap_vals[pos] < val.
        let val_vec = vdupq_n_f32(val);
        let mut lo = 0usize;
        let mut found_in_simd = false;

        // Process 4 elements at a time via NEON
        let chunks4 = k / 4;
        for c in 0..chunks4 {
            let base = c * 4;
            let heap_chunk = vld1q_f32(heap_vals.as_ptr().add(base));
            // Find which heap elements are < val (val > heap element)
            let gt_mask = vcgtq_f32(val_vec, heap_chunk);
            let mask_arr: [u32; 4] = core::mem::transmute(gt_mask);

            // Find first position in this chunk where val > heap element
            for lane in 0..4 {
                if mask_arr[lane] != 0 {
                    lo = base + lane;
                    found_in_simd = true;
                    break;
                }
            }
            if found_in_simd {
                break;
            }
        }

        // If not found in NEON chunks, start scanning from the tail of SIMD region
        if !found_in_simd {
            lo = chunks4 * 4;
        }
        while lo < k && heap_vals[lo] >= val {
            lo += 1;
        }

        // lo is the insertion point — if it's at or beyond k, value doesn't qualify
        if lo >= k {
            return;
        }

        // Shift elements down to make room at position `lo`
        for j in (lo + 1..k).rev() {
            heap_vals[j] = heap_vals[j - 1];
            heap_idxs[j] = heap_idxs[j - 1];
        }
        heap_vals[lo] = val;
        heap_idxs[lo] = idx;
    }
}

// ---------------------------------------------------------------------------
// AVX2 path (x86_64)

/// SIMD-accelerated top-k for k ≤ 16 on x86_64 AVX2.
///
/// Strategy: same register-based sorted top-k as NEON path, but using
/// 8-wide __m256 comparison via AVX2 intrinsics. Loads 8 scores per
/// iteration, compares against threshold, and inserts qualifying lanes
/// using AVX2-parallel search across the sorted register.
#[cfg(target_arch = "x86_64")]
fn argtopk_simd(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    #[target_feature(enable = "avx2")]
    unsafe fn inner(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
        use core::arch::x86_64::{
            _CMP_GT_OS, _mm256_cmp_ps, _mm256_loadu_ps, _mm256_movemask_ps, _mm256_set1_ps,
        };

        let n = scores.len();
        let mut heap_vals = [f32::NEG_INFINITY; 16];
        let mut heap_idxs = [0usize; 16];

        let init = k.min(n);
        for i in 0..init {
            heap_vals[i] = *scores.get_unchecked(i);
            heap_idxs[i] = i;
        }
        // Sort initial k elements descending (insertion sort, k≤16)
        for i in 1..init {
            let val = heap_vals[i];
            let idx = heap_idxs[i];
            let mut j = i;
            while j > 0 && heap_vals[j - 1] < val {
                heap_vals[j] = heap_vals[j - 1];
                heap_idxs[j] = heap_idxs[j - 1];
                j -= 1;
            }
            heap_vals[j] = val;
            heap_idxs[j] = idx;
        }

        // Process remaining elements in AVX2 chunks of 8
        let remaining = n - init;
        let chunks8 = remaining / 8;
        let mut pos = init;

        for _ in 0..chunks8 {
            let v = _mm256_loadu_ps(scores.as_ptr().add(pos));
            let thresh = _mm256_set1_ps(heap_vals[k - 1]);
            // Compare: v > thresh (ordered, signaling)
            let cmp = _mm256_cmp_ps(v, thresh, _CMP_GT_OS);
            let mask_bits = _mm256_movemask_ps(cmp) as u32;

            if mask_bits != 0 {
                // Extract qualifying lanes
                let vals_arr: [f32; 8] = core::mem::transmute(v);
                for lane in 0..8 {
                    if mask_bits & (1 << lane) != 0 {
                        insert_sorted_simd_avx2(
                            &mut heap_vals,
                            &mut heap_idxs,
                            k,
                            vals_arr[lane],
                            pos + lane,
                        );
                    }
                }
            }
            pos += 8;
        }

        // Handle remaining elements that didn't fill a full 8-wide chunk
        let remaining4 = (n - pos) / 4;
        for _ in 0..remaining4 {
            // Process 4 at a time using SSE-like approach via AVX2
            let v = _mm256_loadu_ps(scores.as_ptr().add(pos));
            let thresh = _mm256_set1_ps(heap_vals[k - 1]);
            let cmp = _mm256_cmp_ps(v, thresh, _CMP_GT_OS);
            let mask_bits = (_mm256_movemask_ps(cmp) as u32) & 0x0F; // Only first 4 lanes

            if mask_bits != 0 {
                let vals_arr: [f32; 8] = core::mem::transmute(v);
                for lane in 0..4 {
                    if mask_bits & (1 << lane) != 0 {
                        insert_sorted_simd_avx2(
                            &mut heap_vals,
                            &mut heap_idxs,
                            k,
                            vals_arr[lane],
                            pos + lane,
                        );
                    }
                }
            }
            pos += 4;
        }

        // Scalar tail
        while pos < n {
            let val = *scores.get_unchecked(pos);
            if val > heap_vals[k - 1] {
                insert_sorted_simd_avx2(&mut heap_vals, &mut heap_idxs, k, val, pos);
            }
            pos += 1;
        }

        indices.extend(heap_idxs[..k].iter());
    }

    // Safety: we check for AVX2 at call time via cfg. The function is
    // target_feature enabled so it's unsafe to call directly.
    if is_x86_feature_detected!("avx2") {
        unsafe { inner(scores, k, indices) };
    } else {
        argtopk_scalar_heap(scores, k, indices);
    }
}

/// AVX2-accelerated insertion into sorted register (descending, k ≤ 16).
///
/// Uses AVX2 to compare the candidate value against 8 heap elements at a time,
/// finding the insertion point via SIMD-parallel comparison.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(always)]
unsafe fn insert_sorted_simd_avx2(
    heap_vals: &mut [f32; 16],
    heap_idxs: &mut [usize; 16],
    k: usize,
    val: f32,
    idx: usize,
) {
    use core::arch::x86_64::{
        _CMP_GT_OS, _mm256_cmp_ps, _mm256_loadu_ps, _mm256_movemask_ps, _mm256_set1_ps,
    };

    // SIMD-parallel search: compare val against up to 8 heap elements at a time
    let val_vec = _mm256_set1_ps(val);
    let mut lo = 0usize;

    // Process 8 elements at a time via AVX2
    let chunks8 = k / 8;
    for c in 0..chunks8 {
        let base = c * 8;
        let heap_chunk = _mm256_loadu_ps(heap_vals.as_ptr().add(base));
        let cmp = _mm256_cmp_ps(val_vec, heap_chunk, _CMP_GT_OS);
        let mask_bits = _mm256_movemask_ps(cmp) as u32;

        if mask_bits != 0 {
            // Find first set bit = first position where val > heap element
            lo = base + mask_bits.trailing_zeros() as usize;
            break;
        }
    }

    // If not found in AVX2 chunks, scan the remainder
    if lo == 0 && chunks8 > 0 {
        lo = chunks8 * 8;
    }
    while lo < k && heap_vals[lo] >= val {
        lo += 1;
    }

    if lo >= k {
        return;
    }

    // Shift elements down to make room at position `lo`
    for j in (lo + 1..k).rev() {
        heap_vals[j] = heap_vals[j - 1];
        heap_idxs[j] = heap_idxs[j - 1];
    }
    heap_vals[lo] = val;
    heap_idxs[lo] = idx;
}

/// Scalar fallback for targets without NEON or AVX2.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn argtopk_simd(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    argtopk_scalar_heap(scores, k, indices);
}

/// Scalar min-heap top-k: O(n * log(k)).
/// Used as fallback when SIMD is not available.
pub fn argtopk_scalar_heap(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    let n = scores.len();
    let init = k.min(n);

    let mut heap_vals = [f32::NEG_INFINITY; 16];
    let mut heap_idxs = [0usize; 16];

    for i in 0..init {
        heap_vals[i] = scores[i];
        heap_idxs[i] = i;
    }
    // Sort initial k elements descending (insertion sort)
    for i in 1..init {
        let val = heap_vals[i];
        let idx = heap_idxs[i];
        let mut j = i;
        while j > 0 && heap_vals[j - 1] < val {
            heap_vals[j] = heap_vals[j - 1];
            heap_idxs[j] = heap_idxs[j - 1];
            j -= 1;
        }
        heap_vals[j] = val;
        heap_idxs[j] = idx;
    }

    // Process remaining elements
    for i in init..n {
        insert_sorted(&mut heap_vals, &mut heap_idxs, k, scores[i], i);
    }

    indices.extend(heap_idxs[..k].iter());
}

/// Standard sigmoid function.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: usize = 4;

    fn make_router() -> BlockTopKRouter {
        BlockTopKRouter::new(true)
    }

    #[test]
    fn test_known_keys_known_centroids_known_topk() {
        let router = make_router();
        let mut cache = router.cache_new(3, HEAD_DIM);
        let mut scratch = VortexScratch::new(3);

        // Block 0: keys = [[1,0,0,0], [1,0,0,0]] → centroid [1,0,0,0]
        let keys0 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let vals0 = vec![0.0; 8];
        router.forward_cache(&mut cache, &keys0, &vals0, 0, HEAD_DIM);

        // Block 1: keys = [[0,1,0,0], [0,1,0,0]] → centroid [0,1,0,0]
        let keys1 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let vals1 = vec![0.0; 8];
        router.forward_cache(&mut cache, &keys1, &vals1, 1, HEAD_DIM);

        // Block 2: keys = [[0,0,1,0], [0,0,1,0]] → centroid [0,0,1,0]
        let keys2 = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let vals2 = vec![0.0; 8];
        router.forward_cache(&mut cache, &keys2, &vals2, 2, HEAD_DIM);

        // Query aligned with block 1 → should select block 1 first
        let query = vec![0.0, 1.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 3, 1, &mut scratch);
        assert_eq!(decision.blocks.len(), 1);
        assert_eq!(decision.blocks[0], 1);
    }

    #[test]
    fn test_single_block_always_selected() {
        let router = make_router();
        let mut cache = router.cache_new(1, HEAD_DIM);
        let mut scratch = VortexScratch::new(1);

        let keys = vec![1.0, 2.0, 3.0, 4.0];
        let vals = vec![5.0, 6.0, 7.0, 8.0];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        let query = vec![0.1, 0.2, 0.3, 0.4];
        let decision = router.forward_indexer(&query, &cache, 1, 1, &mut scratch);
        assert_eq!(decision.blocks, vec![0]);
    }

    #[test]
    fn test_zero_query_all_scores_equal() {
        let router = make_router();
        let mut cache = router.cache_new(3, HEAD_DIM);
        let mut scratch = VortexScratch::new(3);

        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let keys1 = vec![0.0, 1.0, 0.0, 0.0];
        let keys2 = vec![0.0, 0.0, 1.0, 0.0];
        let vals = vec![0.0; HEAD_DIM];

        router.forward_cache(&mut cache, &keys0, &vals, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys1, &vals, 1, HEAD_DIM);
        router.forward_cache(&mut cache, &keys2, &vals, 2, HEAD_DIM);

        // Zero query → all dot products = 0
        let query = vec![0.0; HEAD_DIM];
        let decision = router.forward_indexer(&query, &cache, 3, 2, &mut scratch);
        assert_eq!(decision.blocks.len(), 2);
        // All scores are 0.0, any 2 of 3 blocks selected is valid
        assert!(decision.blocks.iter().all(|&b| b < 3));
    }

    #[test]
    fn test_empty_blocks_returns_empty() {
        let router = make_router();
        let cache = router.cache_new(0, HEAD_DIM);
        let mut scratch = VortexScratch::new(0);

        let query = vec![1.0; HEAD_DIM];
        let decision = router.forward_indexer(&query, &cache, 0, 4, &mut scratch);
        assert!(decision.is_empty());
    }

    #[test]
    fn test_topk_capped_at_n_blocks() {
        let router = make_router();
        let mut cache = router.cache_new(2, HEAD_DIM);
        let mut scratch = VortexScratch::new(2);

        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let vals = vec![0.0; HEAD_DIM];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys, &vals, 1, HEAD_DIM);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        // top_k=10 but only 2 blocks
        let decision = router.forward_indexer(&query, &cache, 2, 10, &mut scratch);
        assert_eq!(decision.blocks.len(), 2);
    }

    #[test]
    fn test_cache_ensure_capacity_grows() {
        let mut cache = BlockTopKCache::new(2, HEAD_DIM);
        assert_eq!(cache.centroids.len(), 8);
        cache.ensure_capacity(5);
        assert!(cache.centroids.len() >= 20);
    }

    #[test]
    fn test_centroid_mean_pooling() {
        let router = make_router();
        let mut cache = router.cache_new(1, HEAD_DIM);

        // 3 tokens: [1,0,0,0], [0,2,0,0], [0,0,3,0] → mean = [1/3, 2/3, 1, 0]
        let keys = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let vals = vec![0.0; 12];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        let c = cache.centroid(0);
        let expected = [1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0];
        for (i, (&got, &exp)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "centroid mismatch at dim {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_argtopk_descending() {
        let scores = [0.5, 2.0, 0.1, 3.0, 1.5];
        let mut indices = Vec::new();
        argtopk(&scores, 3, &mut indices);
        assert_eq!(indices.len(), 3);
        // Descending order by score
        assert_eq!(indices[0], 3); // score 3.0
        assert_eq!(indices[1], 1); // score 2.0
        assert_eq!(indices[2], 4); // score 1.5
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    // ── Plan 256: SIMD argtopk tests ─────────────────────────────────

    #[test]
    fn test_argtopk_k1_uses_simd_path() {
        let scores = [0.1, 0.5, 0.3, 0.8, 0.2];
        let mut indices = Vec::new();
        let mut pairs = Vec::new();
        argtopk_with_scratch(&scores, 1, &mut indices, &mut pairs);
        assert_eq!(indices, vec![3]); // score 0.8
    }

    #[test]
    fn test_argtopk_k4_simd_path() {
        // k=4, n=20 — hits the SIMD register path
        let scores: Vec<f32> = (0..20).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut indices = Vec::new();
        let mut pairs = Vec::new();
        argtopk_with_scratch(&scores, 4, &mut indices, &mut pairs);
        assert_eq!(indices.len(), 4);
        // Verify descending order
        for w in indices.windows(2) {
            assert!(
                scores[w[0]] >= scores[w[1]],
                "not descending: scores[{}] = {} < scores[{}] = {}",
                w[0],
                scores[w[0]],
                w[1],
                scores[w[1]]
            );
        }
    }

    #[test]
    fn test_argtopk_k8_simd_path() {
        // k=8, n=32 — NEON chunk processing
        let scores: Vec<f32> = (0..32).rev().map(|i| i as f32).collect();
        let mut indices = Vec::new();
        let mut pairs = Vec::new();
        argtopk_with_scratch(&scores, 8, &mut indices, &mut pairs);
        assert_eq!(indices.len(), 8);
        // Scores are 31,30,...,0 so top-8 = 31..24 at indices 0..8
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_argtopk_k16_boundary() {
        // k=16 is the SIMD path boundary
        let scores: Vec<f32> = (0..64).map(|i| (i as f32 * 0.5).cos()).collect();
        let mut indices = Vec::new();
        let mut pairs = Vec::new();
        argtopk_with_scratch(&scores, 16, &mut indices, &mut pairs);
        assert_eq!(indices.len(), 16);
        // Verify all selected indices are valid
        assert!(indices.iter().all(|&i| i < 64));
        // Verify descending
        for w in indices.windows(2) {
            assert!(
                scores[w[0]] >= scores[w[1]],
                "not descending: scores[{}] = {} < scores[{}] = {}",
                w[0],
                scores[w[0]],
                w[1],
                scores[w[1]]
            );
        }
    }

    #[test]
    fn test_argtopk_k17_falls_back_to_selection() {
        // k=17 > 16, should use selection sort fallback
        let scores: Vec<f32> = (0..50).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut indices = Vec::new();
        let mut pairs = Vec::new();
        argtopk_with_scratch(&scores, 17, &mut indices, &mut pairs);
        assert_eq!(indices.len(), 17);
        for w in indices.windows(2) {
            assert!(
                scores[w[0]] >= scores[w[1]],
                "not descending: scores[{}] = {} < scores[{}] = {}",
                w[0],
                scores[w[0]],
                w[1],
                scores[w[1]]
            );
        }
    }

    #[test]
    fn test_argtopk_simd_matches_scalar() {
        // Verify SIMD and scalar paths produce identical results
        let scores: Vec<f32> = [
            0.5, -1.2, 3.4, 0.0, 2.1, -0.5, 1.8, 4.2, -3.0, 0.7, 1.1, -2.0, 2.9, 0.3, -1.1, 3.8,
            0.1, 1.5, -0.3, 2.6,
        ]
        .to_vec();
        for k in [1, 2, 4, 8, 16] {
            // SIMD path
            let mut simd_indices = Vec::new();
            let mut pairs = Vec::new();
            argtopk_with_scratch(&scores, k, &mut simd_indices, &mut pairs);

            // Reference: full sort
            let mut ref_pairs: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
            ref_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let ref_indices: Vec<usize> = ref_pairs[..k].iter().map(|(i, _)| *i).collect();

            assert_eq!(
                simd_indices, ref_indices,
                "k={k}: SIMD result {simd_indices:?} != reference {ref_indices:?}"
            );
        }
    }

    #[test]
    fn test_argtopk_ties_stable_first_occurrence() {
        // Duplicate scores — should prefer earlier indices
        let scores = [1.0, 2.0, 2.0, 1.0, 3.0];
        let mut indices = Vec::new();
        argtopk(&scores, 3, &mut indices);
        assert_eq!(indices[0], 4); // 3.0
        // Both 2.0 values — should pick first occurrence
        assert_eq!(indices[1], 1); // first 2.0
        assert_eq!(indices[2], 2); // second 2.0
    }

    #[test]
    fn test_argtopk_n_equals_k() {
        // n == k: all elements selected
        let scores = [3.0, 1.0, 4.0];
        let mut indices = Vec::new();
        argtopk(&scores, 3, &mut indices);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices, vec![2, 0, 1]); // 4.0, 3.0, 1.0
    }
}
