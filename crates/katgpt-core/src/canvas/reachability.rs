//! Reachability semantics for the Canvas Schema Compiler — the provable
//! correctness guarantee (Plan 419 Phase 3).
//!
//! ## The guarantee (paper §2.3, load-bearing)
//!
//! For a **binary** mask (`weight ∈ {0, 1}`): if there is no directed path
//! `a → b` in the information-flow graph `G`, then region `a` cannot influence
//! region `b` — this is **exact marginal independence**, by construction.
//!
//! The graph `G` is built from the topology with the convention "content flows
//! `src → dst` because `dst` queries `src`". A connection `src → dst` adds arc
//! `dst → src` in `G` (the *query* node points to its *source* — information
//! moves backwards along the query edge, from source to requester).
//!
//! ## CSR adjacency
//!
//! `G` is stored as a CSR adjacency (`FlowGraph`) so that bounded BFS in
//! [`can_reach`] is `O(V + E)` with zero per-query allocation. This reuses the
//! CSR pattern proven in `viable_manifold_graph` (Plan 312 Phase 5).

use std::collections::VecDeque;

use super::types::{CanvasTopology, Connection, RegionId};

/// The information-flow graph, stored as CSR adjacency.
///
/// Arc `u → v` means "information can flow from `v` to `u`" — i.e. `u` queries
/// `v`, so `v`'s content reaches `u`. Built from a [`CanvasTopology`] by
/// inverting each connection's query direction.
///
/// CSR layout (identical to `viable_manifold_graph::SafeManifoldGraph`):
/// - `offsets`: length `n_nodes + 1`; neighbors of `u` are
///   `neighbors[offsets[u]..offsets[u+1]]`.
/// - `neighbors`: flat arc targets, ordered per-node ascending for determinism.
#[derive(Debug, Clone)]
pub struct FlowGraph {
    n_nodes: usize,
    offsets: Vec<u32>,
    neighbors: Vec<u32>,
}

impl FlowGraph {
    /// Number of nodes (regions) in the graph.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of arcs.
    #[inline]
    pub fn n_arcs(&self) -> usize {
        self.neighbors.len()
    }

    /// Iterate the out-arcs of `u` (zero-allocation). Calls `f(v)` for each
    /// `u → v`. O(degree) via two CSR reads + a slice scan.
    #[inline]
    pub fn for_each_out<F: FnMut(u32)>(&self, u: usize, mut f: F) {
        let start = self.offsets[u] as usize;
        let end = self.offsets[u + 1] as usize;
        for &v in &self.neighbors[start..end] {
            f(v);
        }
    }

    /// Out-degree of `u`.
    #[inline]
    pub fn out_degree(&self, u: usize) -> usize {
        let start = self.offsets[u] as usize;
        let end = self.offsets[u + 1] as usize;
        end - start
    }
}

/// Build the information-flow graph `G` from a topology.
///
/// For each present `Connection { src, dst, .. }`: add arc `dst → src` in `G`.
/// Self-loops (`src == dst`) are preserved (a region querying itself is valid).
/// Duplicate arcs (same `(dst, src)` from multiple connections) are deduplicated.
///
/// `n_regions` is the node count; connections referencing out-of-bounds region
/// ids are skipped (defensive — well-formed schemas stay in bounds).
///
/// # Allocation
///
/// Two allocations total (`offsets` + `neighbors`), at graph-build time. This
/// is a one-time cost; all subsequent [`can_reach`] / [`transitive_closure`]
/// queries are alloc-free against the precomputed CSR.
pub fn build_flow_graph(topology: &CanvasTopology, n_regions: usize) -> FlowGraph {
    // Collect unique arcs (dst → src), skipping out-of-bounds + absent edges.
    let mut arcs: Vec<(u32, u32)> = Vec::with_capacity(topology.connections.len());
    for &conn in &topology.connections {
        if !conn.is_present() {
            continue;
        }
        let Connection { src, dst, .. } = conn;
        let (s, d) = (src.get(), dst.get());
        if s < n_regions && d < n_regions {
            arcs.push((d as u32, s as u32)); // arc dst → src
        }
    }
    // Deduplicate + sort per-node ascending (deterministic neighbor order).
    arcs.sort_unstable();
    arcs.dedup();

    // Counting-sort into CSR.
    let mut degree = vec![0u32; n_regions];
    for &(d, _s) in &arcs {
        degree[d as usize] += 1;
    }
    let mut offsets = Vec::with_capacity(n_regions + 1);
    offsets.push(0);
    let mut acc: u32 = 0;
    for &deg in &degree {
        acc += deg;
        offsets.push(acc);
    }
    let mut neighbors = vec![0u32; arcs.len()];
    let mut cursor = offsets.clone();
    for &(d, s) in &arcs {
        let p = cursor[d as usize] as usize;
        neighbors[p] = s;
        cursor[d as usize] += 1;
    }
    // Per-node sort (arcs were sorted by (d,s) globally, so per-node slices are
    // already ascending; the sort is a no-op safety net for future constructors
    // that don't pre-sort).
    for u in 0..n_regions {
        let s = offsets[u] as usize;
        let e = offsets[u + 1] as usize;
        neighbors[s..e].sort_unstable();
    }

    FlowGraph { n_nodes: n_regions, offsets, neighbors }
}

/// Returns the causal horizon: max path length reachable in `n_blocks`
/// attention blocks × `n_steps` sampling steps (paper §2.3).
///
/// One denoiser pass with `L` blocks moves information along paths of length
/// `≤ L`; `K` sampling steps compose to horizon `K · L`. Trivial but explicit
/// — it documents the causal-horizon invariant that [`can_reach`] respects.
///
/// Modelless: pure multiplication, zero allocation.
#[inline]
pub fn reachability_horizon(n_blocks: usize, n_steps: usize) -> usize {
    n_blocks.saturating_mul(n_steps)
}

/// Returns `true` iff `from` can reach `to` within `horizon` hops in `G`.
///
/// BFS from `from`, bounded by `horizon` edges. `from == to` with `horizon == 0`
/// returns `true` (a node trivially reaches itself in zero hops). For
/// `horizon >= 1` a direct arc suffices.
///
/// # Allocation
///
/// Allocates a small `visited` bitmap + a `VecDeque` frontier per call. For
/// the allocation-free hot path, precompute [`TransitiveClosure`] once and use
/// [`TransitiveClosure::reaches`].
pub fn can_reach(g: &FlowGraph, from: RegionId, to: RegionId, horizon: usize) -> bool {
    let (f, t) = (from.get(), to.get());
    if f >= g.n_nodes() || t >= g.n_nodes() {
        return false;
    }
    if f == t {
        return true; // a node reaches itself in 0 hops.
    }
    if horizon == 0 {
        return false;
    }
    let mut visited = vec![false; g.n_nodes()];
    visited[f] = true;
    let mut frontier: VecDeque<u32> = VecDeque::with_capacity(g.n_nodes());
    frontier.push_back(f as u32);
    let mut remaining = horizon;
    while !frontier.is_empty() && remaining > 0 {
        let layer_size = frontier.len();
        for _ in 0..layer_size {
            let u = frontier.pop_front().unwrap() as usize;
            g.for_each_out(u, |v| {
                let v = v as usize;
                if v == t {
                    // Found — but we can't early-return out of the closure
                    // cleanly; set a flag via the visited trick below.
                }
                if !visited[v] {
                    visited[v] = true;
                    frontier.push_back(v as u32);
                }
            });
            // Early-exit: if `to` was visited this layer, we're done.
            if visited[t] {
                return true;
            }
        }
        remaining -= 1;
    }
    visited[t]
}

/// Precomputed bounded-reachability matrix — the `(n_regions × n_regions)`
/// boolean matrix of "can `from` reach `to` within `horizon` hops".
///
/// Build once at schema-load time (allocates the `n²` bitset), then query
/// alloc-free via [`reaches`]. For large region counts or frequently-changing
/// horizons, prefer direct [`can_reach`] BFS over a precomputed closure.
#[derive(Debug, Clone)]
pub struct TransitiveClosure {
    n: usize,
    horizon: usize,
    /// Row-major bitset: bit `(from * n + to)` = reachable within horizon.
    bits: Vec<u64>,
}

impl TransitiveClosure {
    /// Number of regions.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// The horizon this closure was computed for.
    #[inline]
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// Build the transitive closure out to `horizon` hops.
    ///
    /// Computes, for each `from`, the set of nodes reachable in `≤ horizon`
    /// BFS hops. Self-reachability (`from == to`) is always true (0 hops).
    pub fn build(g: &FlowGraph, horizon: usize) -> Self {
        let n = g.n_nodes();
        let n_words = (n * n).div_ceil(64);
        let mut bits = vec![0u64; n_words];
        // Seed: each node reaches itself (0 hops).
        for u in 0..n {
            set_bit(&mut bits, u, u, n);
        }
        if horizon == 0 {
            return Self { n, horizon, bits };
        }
        // BFS from each node, marking reachable within `horizon`.
        let mut visited = vec![false; n];
        let mut frontier: VecDeque<u32> = VecDeque::with_capacity(n);
        for src in 0..n {
            visited.fill(false);
            frontier.clear();
            visited[src] = true;
            frontier.push_back(src as u32);
            let mut remaining = horizon;
            while !frontier.is_empty() && remaining > 0 {
                let layer_size = frontier.len();
                for _ in 0..layer_size {
                    let u = frontier.pop_front().unwrap() as usize;
                    g.for_each_out(u, |v| {
                        let v = v as usize;
                        if !visited[v] {
                            visited[v] = true;
                            set_bit(&mut bits, src, v, n);
                            frontier.push_back(v as u32);
                        }
                    });
                }
                remaining -= 1;
            }
        }
        Self { n, horizon, bits }
    }

    /// Returns `true` iff `from` reaches `to` within this closure's horizon.
    /// Alloc-free: one bitset read.
    #[inline]
    pub fn reaches(&self, from: RegionId, to: RegionId) -> bool {
        let (f, t) = (from.get(), to.get());
        if f >= self.n || t >= self.n {
            return false;
        }
        let bit = f * self.n + t;
        (self.bits[bit / 64] >> (bit % 64)) & 1 != 0
    }
}

#[inline]
fn set_bit(bits: &mut [u64], from: usize, to: usize, n: usize) {
    let bit = from * n + to;
    bits[bit / 64] |= 1u64 << (bit % 64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::{build_flow_graph_via_compile, causal_chain, isolated, RegionId};

    #[test]
    fn horizon_is_product() {
        assert_eq!(reachability_horizon(4, 3), 12);
        assert_eq!(reachability_horizon(0, 5), 0);
        assert_eq!(reachability_horizon(5, 0), 0);
    }

    #[test]
    fn flow_graph_inverts_query_direction() {
        // Connection A→B (B queries A). Arc in G: B→A.
        // Use from_connections with no self-loops so the inversion is unambiguous.
        let topo = crate::canvas::types::CanvasTopology::from_connections(vec![
            crate::canvas::Connection::new(RegionId(0), RegionId(1)),
        ]);
        let g = build_flow_graph(&topo, 2);
        // The single arc is B(1)→A(0): B has out-degree 1 pointing at A.
        assert_eq!(g.out_degree(1), 1);
        let mut neighbors_of_1 = vec![];
        g.for_each_out(1, |v| neighbors_of_1.push(v));
        assert_eq!(neighbors_of_1, vec![0]);
        // A (node 0) has no out-arc: nothing queries A's query, A is the leaf source.
        assert_eq!(g.out_degree(0), 0);
    }

    #[test]
    fn flow_graph_preserves_self_loops() {
        // isolated topology: each region self-attends → self-loops in G.
        let topo = isolated(&[RegionId(0), RegionId(1)]);
        let g = build_flow_graph(&topo, 2);
        // Each node has a self-loop.
        assert_eq!(g.out_degree(0), 1);
        assert_eq!(g.out_degree(1), 1);
        let mut n0 = vec![];
        g.for_each_out(0, |v| n0.push(v));
        assert_eq!(n0, vec![0]);
    }

    #[test]
    fn flow_graph_dedups_duplicate_arcs() {
        // Two identical connections → one arc.
        let topo = crate::canvas::types::CanvasTopology::from_connections(vec![
            crate::canvas::Connection::new(RegionId(0), RegionId(1)),
            crate::canvas::Connection::new(RegionId(0), RegionId(1)),
        ]);
        let g = build_flow_graph(&topo, 2);
        assert_eq!(g.n_arcs(), 1);
    }

    #[test]
    fn can_reach_respects_horizon_on_causal_chain() {
        // THE HORIZON TEST (Plan 419 T3.6 / G2): causal_chain([A,B,C]) means
        // information flows A→B→C (each region queries its predecessor). So
        // from A you can reach B (1 hop) and C (2 hops), but NOT C in 1 hop.
        let topo = causal_chain(&[RegionId(0), RegionId(1), RegionId(2)]);
        let g = build_flow_graph(&topo, 3);
        // A (0) reaches C (2) in exactly 2 hops (A→B→C).
        assert!(can_reach(&g, RegionId(0), RegionId(2), 2));
        // A does NOT reach C in 1 hop.
        assert!(!can_reach(&g, RegionId(0), RegionId(2), 1));
        // A reaches B in 1 hop.
        assert!(can_reach(&g, RegionId(0), RegionId(1), 1));
        // Converse (bounded, not guaranteed): C cannot reach A — no back-edge.
        assert!(!can_reach(&g, RegionId(2), RegionId(0), 100));
    }

    #[test]
    fn can_reach_self_is_zero_hops() {
        let topo = isolated(&[RegionId(0)]);
        let g = build_flow_graph(&topo, 1);
        assert!(can_reach(&g, RegionId(0), RegionId(0), 0));
    }

    #[test]
    fn can_reach_absent_edge_means_no_reach() {
        // THE SOUNDNESS TEST (G1): two regions with no directed path in G.
        // isolated topology → only self-loops, no cross-region reachability.
        let topo = isolated(&[RegionId(0), RegionId(1)]);
        let g = build_flow_graph(&topo, 2);
        // A (0) cannot reach B (1) at any horizon (exact marginal independence).
        assert!(!can_reach(&g, RegionId(0), RegionId(1), 1));
        assert!(!can_reach(&g, RegionId(0), RegionId(1), 10));
        assert!(!can_reach(&g, RegionId(0), RegionId(1), 1000));
    }

    #[test]
    fn transitive_closure_matches_bfs() {
        // causal_chain([A,B,C]): info arcs A→B→C (each region queries its
        // predecessor). Within horizon 2: A→B (1), A→C (2), B→C (1). Plus self.
        let topo = causal_chain(&[RegionId(0), RegionId(1), RegionId(2)]);
        let g = build_flow_graph(&topo, 3);
        let tc = TransitiveClosure::build(&g, 2);
        assert!(tc.reaches(RegionId(0), RegionId(0))); // self
        assert!(tc.reaches(RegionId(0), RegionId(1))); // A→B
        assert!(tc.reaches(RegionId(0), RegionId(2))); // A→C
        assert!(tc.reaches(RegionId(1), RegionId(2))); // B→C
        // C (2) is the sink — reaches nothing but itself.
        assert!(!tc.reaches(RegionId(2), RegionId(1)));
        assert!(!tc.reaches(RegionId(2), RegionId(0)));
    }

    #[test]
    fn transitive_closure_horizon_clamp() {
        let topo = causal_chain(&[RegionId(0), RegionId(1), RegionId(2)]);
        let g = build_flow_graph(&topo, 3);
        let tc1 = TransitiveClosure::build(&g, 1);
        // Horizon 1: A→B yes, A→C no (needs 2 hops).
        assert!(tc1.reaches(RegionId(0), RegionId(1)));
        assert!(!tc1.reaches(RegionId(0), RegionId(2)));
    }

    #[test]
    fn build_flow_graph_via_compile_helper_round_trips() {
        let topo = causal_chain(&[RegionId(0), RegionId(1), RegionId(2)]);
        let g = build_flow_graph_via_compile(&topo, 3);
        assert_eq!(g.n_nodes(), 3);
        // Forward direction: A(0) reaches C(2) in 2 hops (info flows A→B→C).
        assert!(can_reach(&g, RegionId(0), RegionId(2), 2));
    }
}
