//! Per-capability head scoring + cross-capability fusion + head partition
//! (paper Eq 11–12, Plan 358).
//!
//! [`per_capability_score`] combines a head's strongest role (receiver or
//! sender) with its task-consistency κ. [`fuse_across_capabilities`] min-max
//! normalizes per-capability and weighted-means across capabilities into one
//! ranking. [`partition_by_causal_score`] turns the ranking into a
//! critical/convertible split mirroring RTPurbo's `HeadCalibration` shape.

/// Per-capability head score (paper Eq 11).
///
/// `s_h^(c) = max(IE_recv_h, IE_send_h) · κ_h^(c)`
///
/// where `κ_h^(c)` is task-consistency: the fraction of sub-probes in which
/// the head exceeds the importance threshold (default 0.01) in its strongest role.
/// Down-weights heads that score high on a single sub-probe but are negligible
/// on the rest — favors heads whose contribution is stable across tasks.
#[inline]
pub fn per_capability_score(ie_receiver: f32, ie_sender: f32, task_consistency: f32) -> f32 {
    ie_receiver.max(ie_sender) * task_consistency
}

/// Fuse per-capability scores into a single head ranking (paper Eq 12).
///
/// Min-max normalizes each capability's scores to [0,1] (per-capability drops
/// differ in scale), then takes the weighted mean across capabilities with
/// equal weights by default (equal prior over capabilities when no task pref).
///
/// `per_head_per_capability[h]` = `Vec<(capability_weight, raw_score)>` per head.
/// Returns `Vec<f32>` of length `n_heads`, one fused score per head.
///
/// Min-max normalization is per-capability across heads: for each capability c,
/// `ŝ_h^(c) = (s_h^(c) − min_h s_h^(c)) / (max_h s_h^(c) − min_h s_h^(c))`.
pub fn fuse_across_capabilities(
    per_head_per_capability: &[Vec<(f32, f32)>], // [n_heads] of [(weight, raw_score); n_capabilities]
) -> Vec<f32> {
    let n_heads = per_head_per_capability.len();
    if n_heads == 0 {
        return Vec::new();
    }
    let n_caps = per_head_per_capability[0].len();
    if n_caps == 0 {
        return vec![0.0; n_heads];
    }

    // Per-capability min/max in one pass. Two flat Vecs (3 allocs total)
    // instead of Vec<Vec<f32>> (n_heads + 2 allocs) — the normalized matrix
    // never needs to materialize; we fuse normalize + weighted-mean in the
    // second pass.
    let mut cap_min = vec![f32::INFINITY; n_caps];
    let mut cap_max = vec![f32::NEG_INFINITY; n_caps];
    for h in 0..n_heads {
        for c in 0..n_caps {
            let s = per_head_per_capability[h][c].1;
            if s < cap_min[c] {
                cap_min[c] = s;
            }
            if s > cap_max[c] {
                cap_max[c] = s;
            }
        }
    }

    // Weighted-mean fusion across capabilities with fused min-max normalization.
    let mut out = vec![0.0f32; n_heads];
    for h in 0..n_heads {
        let mut total_w = 0.0f32;
        let mut acc = 0.0f32;
        for c in 0..n_caps {
            let (w, s) = per_head_per_capability[h][c];
            let range = cap_max[c] - cap_min[c];
            let norm = if range.abs() < f32::EPSILON {
                0.0
            } else {
                (s - cap_min[c]) / range
            };
            acc += w * norm;
            total_w += w;
        }
        out[h] = if total_w > 0.0 { acc / total_w } else { 0.0 };
    }
    out
}

/// Rank heads by causal-importance score and partition into critical vs
/// convertible sets, mirroring RTPurbo's `HeadCalibration` shape.
///
/// `critical_ratio` is the fraction of heads to retain (paper default: 0.25
/// for FA in the hybrid; RTPurbo default: 0.15 for retrieval heads).
/// `min_one_per_layer` (paper "Constrained Global Screening" §5.6): if Some,
/// guarantee at least one critical head per layer (caller supplies layer ids).
///
/// Returns `(critical_set, convertible_set)` as sorted `Vec<usize>` of head
/// indices. Ties in score are broken by ascending head index (lower index wins
/// the critical slot), giving a deterministic, reproducible partition.
pub fn partition_by_causal_score(
    scores: &[f32],
    critical_ratio: f32,
    layer_ids: Option<&[usize]>,
    min_one_per_layer: bool,
) -> (Vec<usize>, Vec<usize>) {
    let n = scores.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Rank heads by score descending; ties broken by ascending index.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        // Higher score first; on tie, lower index first.
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    // Top-k by critical_ratio (round up so a nonzero ratio on a small n does
    // not yield zero critical heads). Clamp to [0, n].
    let mut n_critical = ((n as f32) * critical_ratio).ceil() as usize;
    if n_critical > n {
        n_critical = n;
    }

    let mut critical: Vec<usize> = order.split_off(n_critical);
    // `order` now holds the top-n_critical (highest scores); `critical` holds
    // the rest. Swap naming to match intent.
    std::mem::swap(&mut critical, &mut order);
    // Now: `critical` = top-n_critical (unsorted), `order` = convertible tail.
    let mut convertible: Vec<usize> = order;

    // Constrained Global Screening: ensure at least one critical head per layer.
    if min_one_per_layer {
        if let Some(layers) = layer_ids {
            debug_assert_eq!(layers.len(), n, "layer_ids must have one entry per head");
            // Dense layer-coverage bitset indexed by layer id (O(1) lookup,
            // no hashing) instead of HashSet<usize>.
            let max_layer = layers.iter().copied().max().unwrap_or(0);
            let mut layer_covered = vec![false; max_layer + 1];
            for &h in &critical {
                layer_covered[layers[h]] = true;
            }
            // Scan convertible in score-desc order (it's the tail of `order`,
            // still sorted desc). The first head of each uncovered layer we
            // encounter is its highest-scoring head — promote it. Single pass,
            // no swap_remove (preserves scan order).
            let mut promoted: Vec<usize> = Vec::new();
            for &h in &convertible {
                let layer = layers[h];
                if !layer_covered[layer] {
                    layer_covered[layer] = true;
                    promoted.push(h);
                }
            }
            if !promoted.is_empty() {
                // Dense head-id bitset for O(1) retain (no HashSet hashing).
                let mut is_promoted = vec![false; n];
                for &h in &promoted {
                    is_promoted[h] = true;
                }
                convertible.retain(|&h| !is_promoted[h]);
                critical.extend(promoted);
            }
        }
    }

    // Return sorted sets (ascending head index) for stable downstream use.
    critical.sort_unstable();
    convertible.sort_unstable();
    (critical, convertible)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_capability_takes_max_role_times_consistency() {
        // max(0.8, 0.3) = 0.8; × 0.5 consistency = 0.4.
        let s = per_capability_score(0.8, 0.3, 0.5);
        assert!((s - 0.4).abs() < 1e-6);
        // Sender dominates: max(0.1, 0.9) = 0.9; × 1.0 = 0.9.
        let s = per_capability_score(0.1, 0.9, 1.0);
        assert!((s - 0.9).abs() < 1e-6);
    }

    #[test]
    fn fuse_single_capability_single_head_is_zero() {
        // Single capability, single head: range = 0 → normalized to 0.
        let fused = fuse_across_capabilities(&[vec![(1.0, 5.0)]]);
        assert_eq!(fused, vec![0.0]);
    }

    #[test]
    fn fuse_two_capabilities_equal_weight_is_mean() {
        // Two heads, two capabilities, equal weights.
        // cap0 scores: [0.0, 1.0] → normalized [0.0, 1.0]
        // cap1 scores: [1.0, 0.0] → normalized [1.0, 0.0]
        // equal weights → mean: head0 = (0+1)/2 = 0.5, head1 = (1+0)/2 = 0.5
        let per_head = vec![
            vec![(1.0, 0.0), (1.0, 1.0)],
            vec![(1.0, 1.0), (1.0, 0.0)],
        ];
        let fused = fuse_across_capabilities(&per_head);
        assert_eq!(fused.len(), 2);
        assert!((fused[0] - 0.5).abs() < 1e-6, "head0: {}", fused[0]);
        assert!((fused[1] - 0.5).abs() < 1e-6, "head1: {}", fused[1]);
    }

    #[test]
    fn fuse_min_max_normalizes_per_capability() {
        // Two heads, one capability. cap0 scores [2.0, 10.0].
        // normalized: head0 = (2-2)/(10-2) = 0, head1 = (10-2)/(10-2) = 1.
        let per_head = vec![vec![(1.0, 2.0)], vec![(1.0, 10.0)]];
        let fused = fuse_across_capabilities(&per_head);
        assert!((fused[0] - 0.0).abs() < 1e-6, "head0: {}", fused[0]);
        assert!((fused[1] - 1.0).abs() < 1e-6, "head1: {}", fused[1]);
    }

    #[test]
    fn fuse_weighted_mean_respects_weights() {
        // Two capabilities with unequal weights; head0 should weight cap0 more.
        // cap0 normalized [0, 1], cap1 normalized [1, 0]; weights (3.0, 1.0).
        // head0 = (3*0 + 1*1)/(3+1) = 0.25
        let per_head = vec![
            vec![(3.0, 0.0), (1.0, 1.0)],
            vec![(3.0, 1.0), (1.0, 0.0)],
        ];
        let fused = fuse_across_capabilities(&per_head);
        assert!((fused[0] - 0.25).abs() < 1e-6, "head0: {}", fused[0]);
    }

    #[test]
    fn partition_empty_returns_empty() {
        let (crit, conv) = partition_by_causal_score(&[], 0.25, None, false);
        assert!(crit.is_empty());
        assert!(conv.is_empty());
    }

    #[test]
    fn partition_single_head_is_critical() {
        // n=1, ratio=0.25 → ceil(0.25) = 1 critical.
        let (crit, conv) = partition_by_causal_score(&[0.5], 0.25, None, false);
        assert_eq!(crit, vec![0]);
        assert!(conv.is_empty());
    }

    #[test]
    fn partition_top_k_by_score() {
        // 4 heads, ratio 0.5 → 2 critical. Scores: head2 > head0 > head3 > head1.
        let scores = [0.3, 0.1, 0.9, 0.2];
        let (crit, conv) = partition_by_causal_score(&scores, 0.5, None, false);
        // Top-2 by score: head2 (0.9), head0 (0.3).
        assert_eq!(crit, vec![0, 2]);
        assert_eq!(conv, vec![1, 3]);
    }

    #[test]
    fn partition_ties_broken_by_index() {
        // All equal scores → lower indices win critical slots.
        let scores = [0.5, 0.5, 0.5, 0.5];
        let (crit, conv) = partition_by_causal_score(&scores, 0.5, None, false);
        // 2 critical: indices 0, 1 (lowest indices on tie).
        assert_eq!(crit, vec![0, 1]);
        assert_eq!(conv, vec![2, 3]);
    }

    #[test]
    fn partition_min_one_per_layer_rescues_unrepresented_layer() {
        // 4 heads in 2 layers: heads 0,1 in layer 0; heads 2,3 in layer 1.
        // ratio 0.25 → ceil(4*0.25) = 1 critical by score.
        // Score ranking: head0 (0.9) > head1 (0.8) > head2 (0.4) > head3 (0.3).
        // So the top-1 critical is head0 (layer 0). Layer 1 is unrepresented.
        // min_one_per_layer should promote head2 (highest-scoring layer-1 head)
        // from convertible into critical.
        let scores = [0.9, 0.8, 0.4, 0.3];
        let layers = [0usize, 0, 1, 1];
        let (crit, conv) =
            partition_by_causal_score(&scores, 0.25, Some(&layers), true);
        // head0 (layer 0) is critical by score; head2 (layer 1) promoted.
        assert!(crit.contains(&0), "head0 missing from critical: {crit:?}");
        assert!(crit.contains(&2), "head2 (promoted) missing: {crit:?}");
        // Every layer has ≥1 critical head.
        let crit_layers: std::collections::HashSet<usize> =
            crit.iter().map(|&h| layers[h]).collect();
        assert!(crit_layers.contains(&0) && crit_layers.contains(&1));
        // head2 was promoted OUT of convertible.
        assert!(!conv.contains(&2));
    }

    #[test]
    fn partition_min_one_per_layer_noop_when_all_represented() {
        // All layers already represented after top-k → no promotion.
        let scores = [0.9, 0.8, 0.7, 0.6];
        let layers = [0, 1, 2, 3];
        let (crit, conv) =
            partition_by_causal_score(&scores, 0.5, Some(&layers), true);
        // Top-2: head0, head1 (layers 0, 1). Layers 2, 3 unrepresented →
        // promote head2 (layer 2), head3 (layer 3). All 4 critical.
        assert_eq!(crit.len(), 4, "expected all critical after promotion");
        assert!(conv.is_empty());
    }

    #[test]
    fn partition_sets_are_sorted() {
        let scores = [0.4, 0.1, 0.9, 0.2, 0.7];
        let (crit, conv) = partition_by_causal_score(&scores, 0.4, None, false);
        // Verify ascending order.
        assert!(crit.windows(2).all(|w| w[0] < w[1]), "crit not sorted: {crit:?}");
        assert!(conv.windows(2).all(|w| w[0] < w[1]), "conv not sorted: {conv:?}");
    }
}
