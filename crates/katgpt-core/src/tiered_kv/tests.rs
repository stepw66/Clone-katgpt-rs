use super::*;
use crate::tiered_kv::in_memory::InMemoryTieredKvStore;

/// Simple group summarizer: just mean-pool the group's keys (no RoPE awareness).
/// Used as the injection function for the reference store.
fn mean_summarizer(
    keys_flat: &[f32],
    _positions: &[usize],
    group_start: usize,
    n_tokens: usize,
) -> Vec<f32> {
    let d = keys_flat.len() / _positions.len().max(1);
    // NOTE: d is inferred from the keys_flat length / positions length. In
    // practice the caller passes the correct d. For tests we use d=8.
    let d = if d == 0 { 8 } else { d };
    let mut summary = vec![0.0f32; d];
    for t in 0..n_tokens {
        let offset = (group_start + t) * d;
        for i in 0..d {
            summary[i] += keys_flat[offset + i];
        }
    }
    let inv = 1.0 / n_tokens as f32;
    for x in summary.iter_mut() {
        *x *= inv;
    }
    summary
}

#[test]
fn sink_local_set_dedup() {
    let sl = SinkLocalSet::new(vec![0, 1], vec![1, 2, 3]);
    let all = sl.all_chunks();
    assert_eq!(all, vec![0, 1, 2, 3]);
    assert!(sl.contains(0));
    assert!(sl.contains(3));
    assert!(!sl.contains(4));
}

#[test]
fn route_budget_full_is_max() {
    let b = RouteBudget::FULL;
    assert_eq!(b.k_c, usize::MAX);
    assert_eq!(b.k_g, usize::MAX);
}

#[test]
fn working_set_push_token() {
    let mut ws = WorkingSet::empty();
    ws.push_token(&[1.0, 2.0], &[3.0, 4.0], 5);
    assert_eq!(ws.n_tokens, 1);
    assert_eq!(ws.keys, vec![1.0, 2.0]);
    assert_eq!(ws.values, vec![3.0, 4.0]);
    assert_eq!(ws.positions, vec![5]);
}

#[test]
fn group_selection_all_groups() {
    let gs = GroupSelection::all_groups(3, 4);
    assert_eq!(gs.selections.len(), 3);
    assert_eq!(gs.selections[0], (0, 0, 4));
    assert_eq!(gs.selections[2], (2, 0, 4));
}

#[test]
fn in_memory_store_append_and_fetch_full() {
    let d = 8;
    let c = 4; // chunk_size
    let gs = 2; // group_size
    let mut store = InMemoryTieredKvStore::new(d, c, gs, mean_summarizer);

    // Append 3 chunks (12 tokens total).
    for chunk_idx in 0..3 {
        let keys: Vec<f32> = (0..c * d)
            .map(|i| chunk_idx as f32 * 100.0 + i as f32)
            .collect();
        let values: Vec<f32> = (0..c * d)
            .map(|i| chunk_idx as f32 * 200.0 + i as f32)
            .collect();
        let positions: Vec<usize> = (0..c).map(|t| chunk_idx * c + t).collect();
        store.append_chunk(&keys, &values, &positions);
    }

    assert_eq!(store.n_chunks(), 3);
    assert_eq!(store.head_dim(), d);
    assert_eq!(store.chunk_size(), c);
    assert_eq!(store.group_size(), gs);
    assert_eq!(store.n_groups_per_chunk(), 2);

    // Full coverage fetch: all chunks, all groups.
    let sink_local = SinkLocalSet::new(vec![], vec![]);
    let group_sel = GroupSelection::all_groups(3, 2);
    let ws = store.fetch_working_set(&sink_local, &[0, 1, 2], &group_sel);
    assert_eq!(ws.n_tokens, 12); // 3 chunks * 4 tokens
    assert_eq!(ws.keys.len(), 12 * d);
}

#[test]
fn in_memory_store_partial_fetch_by_group() {
    let d = 4;
    let c = 4;
    let gs = 2;
    let mut store = InMemoryTieredKvStore::new(d, c, gs, mean_summarizer);

    let keys: Vec<f32> = (0..c * d).map(|i| i as f32).collect();
    let values: Vec<f32> = (0..c * d).map(|i| (i as f32) * 10.0).collect();
    let positions: Vec<usize> = (0..c).collect();
    store.append_chunk(&keys, &values, &positions);

    // Fetch only group 0 of chunk 0 (tokens 0 and 1).
    let sink_local = SinkLocalSet::new(vec![], vec![]);
    let mut group_sel = GroupSelection::empty();
    group_sel.add_range(0, 0, 1); // chunk 0, group 0 only

    let ws = store.fetch_working_set(&sink_local, &[0], &group_sel);
    assert_eq!(ws.n_tokens, 2); // group 0 = tokens 0,1
    // Token 0 key = [0,1,2,3], token 1 key = [4,5,6,7]
    assert_eq!(&ws.keys[..d], &[0.0f32, 1.0, 2.0, 3.0]);
    assert_eq!(&ws.keys[d..2 * d], &[4.0f32, 5.0, 6.0, 7.0]);
}

#[test]
fn in_memory_store_sink_local_always_fetched() {
    let d = 4;
    let c = 2;
    let gs = 1;
    let mut store = InMemoryTieredKvStore::new(d, c, gs, mean_summarizer);

    for chunk_idx in 0..5 {
        let keys: Vec<f32> = vec![chunk_idx as f32; c * d];
        let values: Vec<f32> = vec![chunk_idx as f32; c * d];
        let positions: Vec<usize> = (chunk_idx * c..).take(c).collect();
        store.append_chunk(&keys, &values, &positions);
    }

    // Sink = chunk 0, Local = chunk 4. Selected = chunk 2.
    let sink_local = SinkLocalSet::new(vec![0], vec![4]);
    let mut group_sel = GroupSelection::empty();
    group_sel.add_range(2, 0, 2); // all groups of chunk 2

    let ws = store.fetch_working_set(&sink_local, &[2], &group_sel);
    // Should fetch: chunk 0 (2 tokens) + chunk 2 (2 tokens) + chunk 4 (2 tokens) = 6 tokens.
    assert_eq!(ws.n_tokens, 6);
}
