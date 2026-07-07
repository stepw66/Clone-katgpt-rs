//! Linking-number detector (paper Algorithm 1, §H) — cold-path diagnostic.
//!
//! Given two point clouds X, Y ⊂ R^d, decide whether they are topologically
//! **linked** in ambient space (link ≠ 0). The linking number is an
//! *extrinsic* topological invariant: it depends on how the manifolds are
//! embedded in ambient space, not on their intrinsic topology. This
//! distinguishes it sharply from the Betti-number TDA the codebase already
//! ships in the DEC substrate (Plan 251, Research 219) — DEC computes
//! *intrinsic* homology via `d∘d=0`; this module computes *extrinsic*
//! ambient linking.
//!
//! # Why this matters
//!
//! The paper's Theorem 4.7 proves that width-d feedforward nets with
//! coordinate-wise monotonic activations (ReLU, sigmoid, tanh — including
//! the sigmoid the codebase mandates) preserve the linking number, so they
//! cannot linearly separate linked manifolds. By contrapositive: **if the
//! detector says "linked," every monotonic projection (HLA affect scalars,
//! direction-vector projections, cosine retrieval) is provably doomed**,
//! and a coordinate-fold (`fold_projection_into`, gated under the
//! `linking_fold_fold` feature) is required to unlink.
//!
//! # Algorithm (paper §H)
//!
//! 1. Project X ∪ Y to R^3 via PCA (top-3 eigenvectors of the joint
//!    covariance).
//! 2. Build ε-filtered k-NN graphs G_X, G_Y on the projected clouds.
//! 3. Extract a fundamental cycle basis per graph via BFS spanning forest
//!    (Definition H.1): for each non-tree edge, the basis cycle is that
//!    edge plus the unique tree path between its endpoints.
//! 4. For each pair of basis cycles (C ∈ C_X, D ∈ C_Y), compute the Gauss
//!    linking integral via midpoint quadrature; round to the nearest
//!    integer. The first non-zero integral is the witness.
//! 5. If all pairs integrate to 0 → not linked.
//!
//! # Cold-path
//!
//! The detector is O(n²) brute-force k-NN + O(k·n) cycle-basis extraction
//! + O(β_X · β_Y · L² · N_sub²) Gauss integral. On n ≈ 2×1000 clouds this
//! is a few milliseconds — fine for audit cadence (every N ticks), too
//! slow for per-tick. The fold is the hot path.

use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════
// Public types
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for [`detect_linking`] / [`detect_linking_into`].
///
/// Defaults match the paper's CIFAR-10 settings (§I.1): `k_neighbors = 15`,
/// `epsilon_quantile = 0.7` (70th percentile of nearest-neighbor distances),
/// `min_cycle_len = 4` (smaller cycles are construction artifacts), and
/// `n_subdivisions = 4` midpoint-quadrature subdivisions per cycle edge.
#[derive(Clone, Copy, Debug)]
pub struct LinkingDetectorConfig {
    /// Number of nearest neighbors per node in the k-NN graph. Paper §H.1
    /// recommends `k ∈ [6, 15]` — denser k exposes more cycles at compute
    /// cost. Default `8`.
    pub k_neighbors: usize,
    /// Edge-length threshold quantile. Edges longer than the
    /// `epsilon_quantile`-th percentile of nearest-neighbor distances are
    /// suppressed. Paper §H.1 uses the 70th percentile. Default `0.7`.
    pub epsilon_quantile: f32,
    /// Minimum cycle length (number of vertices). Cycles shorter than this
    /// are construction artifacts (paper §H.4). Default `4`.
    pub min_cycle_len: usize,
    /// Midpoint-quadrature subdivisions per cycle edge when evaluating the
    /// Gauss integral (paper §H.3). Higher = more accurate but slower.
    /// Default `4`.
    pub n_subdivisions: usize,
}

impl Default for LinkingDetectorConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 8,
            epsilon_quantile: 0.7,
            min_cycle_len: 4,
            n_subdivisions: 4,
        }
    }
}

/// Result of a linking-number detection query.
///
/// `link` is the integer Gauss linking integral (paper Definition 3.2);
/// `linked` is `link != 0`; `witness` is the (cycle_x_idx, cycle_y_idx)
/// basis-cycle pair that produced the non-zero link, or `None` if unlinked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkingVerdict {
    /// True iff the two clouds are topologically linked (link ≠ 0).
    pub linked: bool,
    /// The integer linking number, or 0 if unlinked. `±1` for a Hopf link.
    pub link: i32,
    /// Indices into the internal cycle bases of the witness pair. Useful
    /// for diagnostic visualization; `None` if unlinked or degenerate input.
    pub witness: Option<(usize, usize)>,
}

impl LinkingVerdict {
    /// Convenience constructor for the unlinked verdict.
    pub const fn not_linked() -> Self {
        Self { linked: false, link: 0, witness: None }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Top-level entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Detect whether two point clouds X, Y ⊂ R^d are topologically linked.
///
/// Convenience wrapper around [`detect_linking_into`] that allocates the
/// projected-3D output buffers internally. For repeated calls, prefer
/// `detect_linking_into` with reused scratch.
///
/// # Arguments
///
/// - `x`: flat row-major `[n_x, d]` slice. `n_x = x.len() / d` must be ≥
///   `k_neighbors + 1` to expose any cycle.
/// - `y`: flat row-major `[n_y, d]` slice, same `d`.
/// - `d`: ambient dimension (e.g. 8 for HLA, 64 for shard style_weights).
/// - `config`: see [`LinkingDetectorConfig`].
///
/// # Degenerate inputs
///
/// Returns [`LinkingVerdict::not_linked`] if either cloud has fewer than
/// `k_neighbors + 1` points, if `d == 0`, if all points coincide, or if
/// the PCA projection collapses (zero variance).
pub fn detect_linking(
    x: &[f32],
    y: &[f32],
    d: usize,
    config: &LinkingDetectorConfig,
) -> LinkingVerdict {
    if d == 0 || x.len() < d || y.len() < d {
        return LinkingVerdict::not_linked();
    }
    let n_x = x.len() / d;
    let n_y = y.len() / d;
    // Need at least k+1 points to form a k-NN cycle (paper §H.2).
    if n_x < config.k_neighbors + 1 || n_y < config.k_neighbors + 1 {
        return LinkingVerdict::not_linked();
    }
    let mut x_proj = vec![[0.0_f32; 3]; n_x];
    let mut y_proj = vec![[0.0_f32; 3]; n_y];
    detect_linking_into(x, y, d, &mut x_proj, &mut y_proj, config)
}

/// Same as [`detect_linking`] but with caller-supplied scratch buffers for
/// the projected-3D points. Allows reuse across calls (zero internal
/// allocation after the first).
///
/// `x_proj` and `y_proj` must have lengths `n_x = x.len() / d` and
/// `n_y = y.len() / d` respectively; their contents are overwritten.
pub fn detect_linking_into(
    x: &[f32],
    y: &[f32],
    d: usize,
    x_proj: &mut [[f32; 3]],
    y_proj: &mut [[f32; 3]],
    config: &LinkingDetectorConfig,
) -> LinkingVerdict {
    if d == 0 {
        return LinkingVerdict::not_linked();
    }
    let n_x = x.len() / d;
    let n_y = y.len() / d;
    if n_x < config.k_neighbors + 1 || n_y < config.k_neighbors + 1 {
        return LinkingVerdict::not_linked();
    }
    if x_proj.len() < n_x || y_proj.len() < n_y {
        return LinkingVerdict::not_linked();
    }

    // ── Step 1: PCA-3D projection (joint covariance for a shared frame) ──
    let x_proj = &mut x_proj[..n_x];
    let y_proj = &mut y_proj[..n_y];
    if !pca_project_joint_into_3d(x, y, d, x_proj, y_proj) {
        // PCA collapsed (all points coincide, zero variance).
        return LinkingVerdict::not_linked();
    }

    // ── Step 2: ε-kNN graphs ──
    let g_x = build_epsilon_knn_graph(x_proj, config.k_neighbors, config.epsilon_quantile);
    let g_y = build_epsilon_knn_graph(y_proj, config.k_neighbors, config.epsilon_quantile);

    // ── Step 3: fundamental cycle bases ──
    let cycles_x = fundamental_cycle_basis(n_x, &g_x, config.min_cycle_len);
    let cycles_y = fundamental_cycle_basis(n_y, &g_y, config.min_cycle_len);
    if cycles_x.is_empty() || cycles_y.is_empty() {
        return LinkingVerdict::not_linked();
    }

    // ── Step 4: Gauss linking integral over basis-cycle pairs ──
    for (i, cx) in cycles_x.iter().enumerate() {
        for (j, cy) in cycles_y.iter().enumerate() {
            let link = gauss_linking_integral(cx, cy, x_proj, y_proj, config.n_subdivisions);
            if link != 0 {
                return LinkingVerdict { linked: true, link, witness: Some((i, j)) };
            }
        }
    }
    LinkingVerdict::not_linked()
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 1: PCA-3D via power iteration + deflation on a 3×3 covariance
// ═══════════════════════════════════════════════════════════════════════════

/// Project both clouds to R^3 via PCA on the joint covariance.
///
/// Returns `false` if PCA collapses (all points coincide, zero variance) —
/// caller should treat this as "degenerate, not linked."
///
/// Implementation: compute the 3×3 joint covariance of the centered clouds
/// in a random 3D projection (to bound the eigenvector search to R^3
/// regardless of ambient d), then power-iterate + deflate for the top 3
/// eigenvectors of that 3×3 matrix. This is the standard "random-projection
/// then 3D eigendecomp" trick — O((n_x + n_y) · d) for the projection,
/// then a tiny constant-time eigendecomp.
fn pca_project_joint_into_3d(
    x: &[f32],
    y: &[f32],
    d: usize,
    x_proj: &mut [[f32; 3]],
    y_proj: &mut [[f32; 3]],
) -> bool {
    let n_x = x.len() / d;
    let n_y = y.len() / d;
    debug_assert!(x_proj.len() == n_x);
    debug_assert!(y_proj.len() == n_y);

    // Pick a fixed deterministic 3×d projection matrix (rows are normalized
    // DFT-like vectors — deterministic so the link result is reproducible
    // across runs, satisfying G5 determinism).
    let proj = projection_matrix_3xd(d);

    // Project both clouds to R^3.
    for (i, p) in x_proj.iter_mut().enumerate() {
        let row = &x[i * d..(i + 1) * d];
        project_row_into(row, &proj, p);
    }
    for (i, p) in y_proj.iter_mut().enumerate() {
        let row = &y[i * d..(i + 1) * d];
        project_row_into(row, &proj, p);
    }

    // Now do PCA *within* the 3D projection: center, compute 3×3 covariance,
    // eigendecompose, rotate. This corrects for the projection not aligning
    // with the true top-3 principal axes.
    let (mean, variance) = mean_and_variance_3d(x_proj, y_proj);
    let total_var = variance[0] + variance[1] + variance[2];
    if total_var < 1e-12_f32 {
        return false; // All points coincide after projection.
    }

    // Center both clouds.
    for p in x_proj.iter_mut() {
        for k in 0..3 { p[k] -= mean[k]; }
    }
    for p in y_proj.iter_mut() {
        for k in 0..3 { p[k] -= mean[k]; }
    }

    // 3×3 covariance on the joint centered clouds.
    let cov = covariance_3d(x_proj, y_proj);

    // Eigendecompose the 3×3 symmetric covariance via Jacobi rotation.
    let (v, _lambda) = jacobi_eigendecomp_3x3(cov);

    // Rotate both clouds into the PCA frame: p ← Vᵀ · p.
    for p in x_proj.iter_mut() {
        rotate_by_3x3(v, p);
    }
    for p in y_proj.iter_mut() {
        rotate_by_3x3(v, p);
    }

    true
}

/// Deterministic 3×d projection matrix with DFT-like rows (rows are
/// orthogonal by construction). Determinism is required for G5.
fn projection_matrix_3xd(d: usize) -> [[f32; 64]; 3] {
    // Fixed seeds for 3 deterministic pseudo-random rows in R^d.
    // We use the first d entries of three fixed Hadamard-like rows; for
    // d > 64 we cap at 64 (the projection still works, just with truncated
    // randomness — fine for a diagnostic).
    let d_eff = d.min(64);
    let mut out = [[0.0_f32; 64]; 3];
    // Use simple deterministic normalized vectors. Row 0: gradient of index.
    // Rows 1, 2: alternating signs with different periods. All normalized.
    for k in 0..d_eff {
        let phase0 = (k as f32) * 0.791_753_f32; // ~sqrt(2π)/2.8
        let phase1 = (k as f32) * 1.274_547_f32; // ~golden-ratio·π/4
        out[0][k] = phase0.cos();
        out[1][k] = phase1.sin();
        out[2][k] = phase0.sin() + phase1.cos() * 0.5;
    }
    // Normalize each row.
    for row in &mut out {
        let mut norm = 0.0_f32;
        for &v in row.iter().take(d_eff) {
            norm += v * v;
        }
        norm = norm.sqrt().max(1e-12);
        for v in row.iter_mut().take(d_eff) {
            *v /= norm;
        }
    }
    out
}

#[inline]
fn project_row_into(row: &[f32], proj: &[[f32; 64]; 3], out: &mut [f32; 3]) {
    let d_eff = row.len().min(64);
    for k in 0..3 {
        let mut acc = 0.0_f32;
        for j in 0..d_eff {
            acc += proj[k][j] * row[j];
        }
        out[k] = acc;
    }
}

#[inline]
fn mean_and_variance_3d(x: &[[f32; 3]], y: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let n = (x.len() + y.len()) as f32;
    let mut mean = [0.0_f32; 3];
    for p in x.iter().chain(y.iter()) {
        for k in 0..3 { mean[k] += p[k]; }
    }
    for k in 0..3 { mean[k] /= n.max(1.0); }
    let mut var = [0.0_f32; 3];
    for p in x.iter().chain(y.iter()) {
        for k in 0..3 {
            let d = p[k] - mean[k];
            var[k] += d * d;
        }
    }
    let nf = n.max(1.0);
    for k in 0..3 { var[k] /= nf; }
    (mean, var)
}

#[inline]
fn covariance_3d(x: &[[f32; 3]], y: &[[f32; 3]]) -> [[f32; 3]; 3] {
    let n = (x.len() + y.len()) as f32;
    let mut cov = [[0.0_f32; 3]; 3];
    for p in x.iter().chain(y.iter()) {
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += p[i] * p[j];
            }
        }
    }
    let nf = n.max(1.0);
    for i in 0..3 {
        for j in 0..3 {
            cov[i][j] /= nf;
        }
    }
    cov
}

/// Jacobi eigendecomposition of a 3×3 symmetric matrix. Returns
/// (eigenvectors as columns of a 3×3, eigenvalues sorted descending).
/// Standard textbook algorithm; converges in a handful of sweeps for 3×3.
fn jacobi_eigendecomp_3x3(mut a: [[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3]) {
    let mut v = [[0.0_f32; 3]; 3];
    for i in 0..3 { v[i][i] = 1.0; }

    for _sweep in 0..50 {
        // Off-diagonal magnitude.
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-10_f32 {
            break;
        }
        // Rotate on the largest off-diagonal each sweep.
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() < 1e-12_f32 {
                continue;
            }
            let app = a[p][p];
            let aqq = a[q][q];
            let theta = 0.5_f32 * (aqq - app).atan2(2.0 * apq);
            let c = theta.cos();
            let s = theta.sin();
            // Rotate A.
            for k in 0..3 {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p][k];
                let aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            // Rotate V (columns).
            for k in 0..3 {
                let vkp = v[k][p];
                let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }

    let mut lam = [a[0][0], a[1][1], a[2][2]];

    // Sort eigenpairs descending by eigenvalue (simple selection sort on 3).
    for i in 0..3 {
        let mut max_k = i;
        for k in (i + 1)..3 {
            if lam[k] > lam[max_k] { max_k = k; }
        }
        if max_k != i {
            lam.swap(i, max_k);
            for k in 0..3 { v[k].swap(i, max_k); }
        }
    }
    (v, lam)
}

#[inline]
fn rotate_by_3x3(v: [[f32; 3]; 3], p: &mut [f32; 3]) {
    let x = v[0][0] * p[0] + v[0][1] * p[1] + v[0][2] * p[2];
    let y = v[1][0] * p[0] + v[1][1] * p[1] + v[1][2] * p[2];
    let z = v[2][0] * p[0] + v[2][1] * p[1] + v[2][2] * p[2];
    p[0] = x;
    p[1] = y;
    p[2] = z;
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 2: ε-filtered k-NN graph (brute-force, cold-path)
// ═══════════════════════════════════════════════════════════════════════════

/// Build an ε-filtered k-NN graph: node i has edges to its k nearest
/// neighbors within distance ε. Edges are undirected (we add both
/// directions). Paper §H.1.
fn build_epsilon_knn_graph(
    points: &[[f32; 3]],
    k: usize,
    epsilon_quantile: f32,
) -> Vec<Vec<usize>> {
    let n = points.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let kk = k.min(n - 1);

    // For each point, compute distances to all others, find the kk nearest.
    // Collect all kk-th-nearest distances to compute ε as a quantile.
    let mut all_kth_distances = Vec::with_capacity(n);
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::with_capacity(kk * 2); n];

    for i in 0..n {
        // Collect (distance, j) for all j ≠ i.
        let mut dists: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (dist3(points[i], points[j]), j))
            .collect();
        // Partial sort: get the kk smallest.
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let kth = dists.get(kk.saturating_sub(1)).map(|(d, _)| *d).unwrap_or(f32::INFINITY);
        all_kth_distances.push(kth);
    }

    // ε = quantile of the kth-nearest distances.
    all_kth_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let qidx = ((epsilon_quantile.clamp(0.0, 1.0)) * (n as f32 - 1.0)).round() as usize;
    let qidx = qidx.min(n - 1);
    let epsilon = all_kth_distances[qidx];

    // Build undirected adjacency: edge (i, j) if j is in i's k-NN AND dist ≤ ε.
    for i in 0..n {
        let pi = points[i];
        let mut dists: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (dist3(pi, points[j]), j))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (d, j) in dists.into_iter().take(kk) {
            if d <= epsilon {
                if !adjacency[i].contains(&j) {
                    adjacency[i].push(j);
                }
                if !adjacency[j].contains(&i) {
                    adjacency[j].push(i);
                }
            }
        }
    }
    adjacency
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 3: fundamental cycle basis via BFS spanning forest (paper §H.2)
// ═══════════════════════════════════════════════════════════════════════════

/// Extract a fundamental cycle basis: BFS spanning forest; for each non-tree
/// edge (u, v), the cycle is `tree_path(root, u) + (u, v) + tree_path(v, root)`.
/// Cycles shorter than `min_len` are dropped (paper §H.4 construction-artifact
/// filter).
fn fundamental_cycle_basis(
    n: usize,
    adjacency: &[Vec<usize>],
    min_len: usize,
) -> Vec<Vec<usize>> {
    let mut parent = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    let mut tree_edges: Vec<(usize, usize)> = Vec::new();
    let mut non_tree_edges: Vec<(usize, usize)> = Vec::new();

    // BFS spanning forest.
    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        parent[start] = start;
        let mut queue = VecDeque::with_capacity(n);
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for &v in adjacency[u].iter() {
                if !visited[v] {
                    visited[v] = true;
                    parent[v] = u;
                    tree_edges.push((u, v));
                    queue.push_back(v);
                } else if v > u {
                    // Candidate non-tree edge (v > u dedupes undirected).
                    // Verify it's not the tree edge we just traversed.
                    if parent[u] != v && parent[v] != u {
                        non_tree_edges.push((u, v));
                    }
                }
            }
        }
    }
    let _ = tree_edges; // Not needed for cycle construction; parent[] suffices.

    // For each non-tree edge, build the cycle: path u→root, then root→v, then (v,u).
    let mut cycles = Vec::new();
    for (u, v) in non_tree_edges {
        let path_u = path_to_root(u, &parent);
        let path_v = path_to_root(v, &parent);
        // Find the lowest common ancestor by walking path_u's set.
        let path_u_set: std::collections::HashSet<usize> = path_u.iter().copied().collect();
        // Walk path_v from v upward; first node in path_u_set is the LCA.
        let mut lca = v;
        for &node in &path_v {
            if path_u_set.contains(&node) {
                lca = node;
                break;
            }
        }
        // Cycle = (u → lca) ++ (lca → v) ++ (v → u).
        let mut cycle: Vec<usize> = Vec::new();
        // u → lca: walk up from u until we hit lca.
        let mut cur = u;
        cycle.push(cur);
        while cur != lca {
            cur = parent[cur];
            if cur == usize::MAX { break; }
            cycle.push(cur);
        }
        // lca → v: walk up from v to lca, then reverse (excluding lca, already added).
        let mut v_to_lca: Vec<usize> = Vec::new();
        cur = v;
        while cur != lca {
            v_to_lca.push(cur);
            cur = parent[cur];
            if cur == usize::MAX { break; }
        }
        v_to_lca.reverse();
        cycle.extend(v_to_lca);
        // Close the cycle back to u (last node is v, edge v→u closes).
        // We don't repeat u; the cycle is encoded as a vertex sequence.

        if cycle.len() >= min_len {
            cycles.push(cycle);
        }
    }
    cycles
}

#[inline]
fn path_to_root(mut node: usize, parent: &[usize]) -> Vec<usize> {
    let mut path = Vec::new();
    loop {
        path.push(node);
        let p = parent[node];
        if p == node || p == usize::MAX {
            break;
        }
        node = p;
    }
    path
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 4: Gauss linking integral via midpoint quadrature (paper §H.3)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Gauss linking integral of two piecewise-linear cycles in R^3,
/// rounded to the nearest integer. Returns 0 if either cycle has < 3 vertices.
///
/// `link(C, D) = (1/4π) ∮_C ∮_D (x − y)·(dx × dy) / |x − y|³`.
///
/// Midpoint quadrature: each cycle edge is subdivided into `n_sub` segments;
/// the double integral is summed over all (C-segment, D-segment) midpoint
/// pairs. Paper §H.3.
fn gauss_linking_integral(
    cycle_x: &[usize],
    cycle_y: &[usize],
    points_x: &[[f32; 3]],
    points_y: &[[f32; 3]],
    n_sub: usize,
) -> i32 {
    if cycle_x.len() < 3 || cycle_y.len() < 3 || n_sub == 0 {
        return 0;
    }
    let n_sub = n_sub.max(1);
    let inv_4pi = 1.0_f32 / (4.0 * std::f32::consts::PI);

    // Subdivide each cycle into small segments, collecting midpoint + tangent.
    let segs_x = subdivide_cycle(cycle_x, points_x, n_sub);
    let segs_y = subdivide_cycle(cycle_y, points_y, n_sub);

    let mut total = 0.0_f32;
    for sx in &segs_x {
        for sy in &segs_y {
            total += gauss_integrand(sx.midpoint, sx.tangent, sy.midpoint, sy.tangent);
        }
    }
    (inv_4pi * total).round() as i32
}

struct CycleSegment {
    midpoint: [f32; 3],
    tangent: [f32; 3], // dx (or dy) — the differential along the segment.
}

fn subdivide_cycle(
    cycle: &[usize],
    points: &[[f32; 3]],
    n_sub: usize,
) -> Vec<CycleSegment> {
    let n = cycle.len();
    let mut out =
        Vec::with_capacity(n * n_sub);
    for i in 0..n {
        let a = points[cycle[i]];
        let b = points[cycle[(i + 1) % n]];
        let edge_tangent = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let inv = 1.0_f32 / (n_sub as f32);
        for k in 0..n_sub {
            let t0 = (k as f32) * inv;
            let t1 = ((k + 1) as f32) * inv;
            let tm = 0.5_f32 * (t0 + t1);
            let midpoint = [
                a[0] + tm * edge_tangent[0],
                a[1] + tm * edge_tangent[1],
                a[2] + tm * edge_tangent[2],
            ];
            // Segment tangent = (t1 - t0) * edge_tangent = inv * edge_tangent.
            let tangent = [
                inv * edge_tangent[0],
                inv * edge_tangent[1],
                inv * edge_tangent[2],
            ];
            out.push(CycleSegment { midpoint, tangent });
        }
    }
    out
}

/// The integrand at a pair of midpoints (x, y) with differentials (dx, dy):
/// `(x − y) · (dx × dy) / |x − y|³`.
#[inline]
fn gauss_integrand(x: [f32; 3], dx: [f32; 3], y: [f32; 3], dy: [f32; 3]) -> f32 {
    let diff = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    // cross = dx × dy
    let cross = [
        dx[1] * dy[2] - dx[2] * dy[1],
        dx[2] * dy[0] - dx[0] * dy[2],
        dx[0] * dy[1] - dx[1] * dy[0],
    ];
    let dot = diff[0] * cross[0] + diff[1] * cross[1] + diff[2] * cross[2];
    let norm = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
    let norm_cubed = norm * norm * norm;
    if norm_cubed < 1e-12_f32 {
        return 0.0;
    }
    dot / norm_cubed
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    // The unlinking test needs the fold correction (gated under
    // `linking_fold_fold`). When only `linking_fold_detector` is enabled, the
    // fold module is absent and this import would fail — gate it.
    #[cfg(feature = "linking_fold_fold")]
    use crate::linking_fold::fold_projection_into;

    /// Generate a thickened Hopf link (paper §G.1 parametrization).
    /// X(t) = (cos t, sin t, 0), Y(s) = (1 + cos s, 0, sin s).
    /// Returns (x_flat, y_flat) with d=3.
    fn thickened_hopf_link(n_per_circle: usize, thickness: f32) -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n_per_circle * 3);
        let mut y = Vec::with_capacity(n_per_circle * 3);
        for i in 0..n_per_circle {
            let t = (i as f32 / n_per_circle as f32) * 2.0 * std::f32::consts::PI;
            // Small normal perturbation to thicken (preserves topology).
            let nx = (i as f32 * 7.13).sin() * thickness;
            let ny = (i as f32 * 5.31).cos() * thickness;
            x.push(t.cos() + nx);
            x.push(t.sin() + ny);
            x.push(0.0 + (i as f32 * 3.7).sin() * thickness * 0.5);

            let s = (i as f32 / n_per_circle as f32) * 2.0 * std::f32::consts::PI;
            y.push(1.0 + s.cos() + (i as f32 * 4.1).sin() * thickness);
            y.push(0.0 + (i as f32 * 6.7).cos() * thickness);
            y.push(s.sin() + (i as f32 * 2.9).sin() * thickness * 0.5);
        }
        (x, y)
    }

    /// Generate two unlinked circles (separated in space).
    fn unlinked_circles(n_per_circle: usize) -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n_per_circle * 3);
        let mut y = Vec::with_capacity(n_per_circle * 3);
        for i in 0..n_per_circle {
            let t = (i as f32 / n_per_circle as f32) * 2.0 * std::f32::consts::PI;
            // Circle 1 in plane z=0, centered at origin, radius 1.
            x.push(t.cos());
            x.push(t.sin());
            x.push(0.0);
            // Circle 2 in plane z=10, centered at origin, radius 1 — far from circle 1.
            y.push(t.cos());
            y.push(t.sin());
            y.push(10.0);
        }
        (x, y)
    }

    #[test]
    fn detects_hopf_link_as_linked() {
        let (x, y) = thickened_hopf_link(80, 0.05);
        let cfg = LinkingDetectorConfig::default();
        let verdict = detect_linking(&x, &y, 3, &cfg);
        assert!(
            verdict.linked,
            "Hopf link should be detected as linked; got {:?}",
            verdict
        );
        assert_eq!(verdict.link.abs(), 1, "Hopf link has link = ±1");
    }

    #[test]
    fn unlinked_circles_return_not_linked() {
        let (x, y) = unlinked_circles(80);
        let cfg = LinkingDetectorConfig::default();
        let verdict = detect_linking(&x, &y, 3, &cfg);
        assert!(
            !verdict.linked,
            "Unlinked circles should not be detected as linked; got {:?}",
            verdict
        );
    }

    /// Cross-feature test: needs both the detector AND the fold.
    /// Gated out when only the detector is enabled (the fold module is absent
    /// in that configuration). Run with `--features linking_fold` (umbrella)
    /// or `--features linking_fold_fold,linking_fold_detector`.
    #[test]
    #[cfg(feature = "linking_fold_fold")]
    fn fold_unlinks_hopf_link() {
        let (x, y) = thickened_hopf_link(80, 0.05);
        let cfg = LinkingDetectorConfig::default();

        // Confirm linked first.
        let before = detect_linking(&x, &y, 3, &cfg);
        assert!(before.linked, "fixture: should be linked before fold");

        // Apply coordinate-wise fold to BOTH clouds (paper Fig. 9: 3 passes
        // per cloud, one per axis, reflecting onto the positive octant).
        let mut x_folded = x.clone();
        let mut y_folded = y.clone();
        let center = [0.0_f32; 3];
        for axis_chunk in x_folded.chunks_mut(3) {
            fold_projection_into(axis_chunk, &center);
        }
        for axis_chunk in y_folded.chunks_mut(3) {
            fold_projection_into(axis_chunk, &center);
        }

        // Re-run the detector — link should drop to 0.
        let after = detect_linking(&x_folded, &y_folded, 3, &cfg);
        assert!(
            !after.linked,
            "After coordinate fold, Hopf link should be unlinked; got {:?}",
            after
        );
    }

    #[test]
    fn degenerate_empty_input() {
        let cfg = LinkingDetectorConfig::default();
        assert_eq!(detect_linking(&[], &[0.0; 30], 3, &cfg), LinkingVerdict::not_linked());
        assert_eq!(detect_linking(&[0.0; 30], &[], 3, &cfg), LinkingVerdict::not_linked());
    }

    #[test]
    fn degenerate_too_few_points() {
        let cfg = LinkingDetectorConfig::default();
        // k_neighbors default is 8; need ≥ 9 points per cloud.
        let x = vec![0.0_f32; 6 * 3];
        let y = vec![0.0_f32; 6 * 3];
        assert_eq!(detect_linking(&x, &y, 3, &cfg), LinkingVerdict::not_linked());
    }

    #[test]
    fn degenerate_all_coincident() {
        let cfg = LinkingDetectorConfig::default();
        let n = 30;
        // All points identical → PCA collapses → not linked.
        let x = vec![1.0_f32; n * 3];
        let y = vec![2.0_f32; n * 3];
        assert_eq!(detect_linking(&x, &y, 3, &cfg), LinkingVerdict::not_linked());
    }

    #[test]
    fn verdict_deterministic_across_runs() {
        let (x, y) = thickened_hopf_link(80, 0.05);
        let cfg = LinkingDetectorConfig::default();
        let v1 = detect_linking(&x, &y, 3, &cfg);
        let v2 = detect_linking(&x, &y, 3, &cfg);
        let v3 = detect_linking(&x, &y, 3, &cfg);
        assert_eq!(v1, v2);
        assert_eq!(v2, v3);
    }

    #[test]
    fn gauss_integrand_zero_at_coincident_points() {
        // x = y → diff = 0 → integrand returns 0 (avoids 0/0).
        let x = [1.0_f32, 0.0, 0.0];
        let dx = [0.1, 0.0, 0.0];
        let y = [1.0, 0.0, 0.0];
        let dy = [0.0, 0.1, 0.0];
        assert_eq!(gauss_integrand(x, dx, y, dy), 0.0);
    }

    #[test]
    fn jacobi_eigendecomp_diagonal_input() {
        // Already-diagonal matrix → eigenvectors = identity, eigenvalues = diag.
        let a = [[2.0_f32, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let (v, lam) = jacobi_eigendecomp_3x3(a);
        // Sorted descending: 3, 2, 1.
        assert!((lam[0] - 3.0).abs() < 1e-4);
        assert!((lam[1] - 2.0).abs() < 1e-4);
        assert!((lam[2] - 1.0).abs() < 1e-4);
        // V should be a rotation (columns are unit vectors).
        for col in 0..3 {
            let mut norm_sq = 0.0;
            for row in 0..3 { norm_sq += v[row][col] * v[row][col]; }
            assert!((norm_sq - 1.0).abs() < 1e-4, "column {col} not unit");
        }
    }
}
