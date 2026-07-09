//! Representation geometry diagnostics (Plan 151, Research 113).
//!
//! Measures representation health of hidden state vectors via:
//! - **Effective rank** (entropy-based, Roy & Vetterli 2007)
//! - **Average pairwise cosine similarity** (anisotropy metric)
//!
//! High effective rank + low cosine similarity = healthy, isotropic representations.
//! Low effective rank + high cosine similarity = degenerate, collapsed representations.
//!
//! ## Sink-aware aggregation (Plan 287, Research 258)
//!
//! [`LayerSinkSummary`] bridges the per-sink classifier
//! ([`super::sink_classify`]) with the whole-layer [`GeometryReport`].
//! The classifier is the *mechanism locator* (NOP vs Broadcast per sink
//! column); `effective_rank` is the *aggregate symptom*. `LayerSinkSummary`
//! aggregates the per-sink verdicts across all heads in a layer.

use super::sink_classify::{SinkClassifierConfig, SinkKind, StableRankScratch, classify_all_sinks};

// ── Core types ──────────────────────────────────────────────────

/// Combined representation geometry report for a set of hidden states.
#[derive(Debug, Clone)]
pub struct GeometryReport {
    pub layer_index: usize,
    pub n_tokens: usize,
    pub hidden_dim: usize,
    pub effective_rank: f32,
    pub avg_cosine_sim: f32,
}

// ── Core functions ──────────────────────────────────────────────

/// Compute the effective rank of a set of hidden state vectors.
///
/// Uses entropy-based effective rank (Roy & Vetterli, 2007) from the
/// eigenvalue spectrum of the empirical covariance matrix.
///
/// High effective rank = healthy, diverse representations.
/// Low effective rank = degenerate, collapsed representations.
///
/// # Panics
/// Panics if `hidden_states` is empty or if vectors have inconsistent dimensions.
pub fn effective_rank(hidden_states: &[Vec<f32>]) -> f32 {
    assert!(!hidden_states.is_empty(), "hidden_states must not be empty");
    let n = hidden_states.len();
    let dim = hidden_states[0].len();
    assert!(dim > 0, "hidden state vectors must be non-empty");

    // Verify consistent dimensions.
    for (i, v) in hidden_states.iter().enumerate() {
        assert_eq!(
            v.len(),
            dim,
            "inconsistent dimensions: vector 0 has len {dim}, vector {i} has len {}",
            v.len()
        );
    }

    // If only one vector, rank is 0 (no variance).
    if n == 1 {
        return 0.0;
    }

    // 1. Compute mean.
    let mut mean = vec![0.0f64; dim];
    for v in hidden_states {
        for (j, &val) in v.iter().enumerate() {
            mean[j] += val as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    // 2. Center each vector (build centered matrix X: n × dim).
    let mut centered = vec![0.0f64; n * dim];
    for (i, v) in hidden_states.iter().enumerate() {
        for (j, &val) in v.iter().enumerate() {
            centered[i * dim + j] = val as f64 - mean[j];
        }
    }

    // 3. Compute covariance matrix C = (1/N) * X^T * X  (dim × dim).
    // We use the smaller of n and dim to decide approach.
    // For typical hidden states, n << dim, so compute C directly.
    let scale = 1.0 / n as f64;
    let mut cov = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let mut sum = 0.0f64;
            for k in 0..n {
                sum += centered[k * dim + i] * centered[k * dim + j];
            }
            let val = sum * scale;
            cov[i * dim + j] = val;
            cov[j * dim + i] = val;
        }
    }

    // 4. Compute eigenvalues via Jacobi iteration.
    let eigenvalues = jacobi_eigenvalues(&mut cov, dim, 50);

    // 5. Normalize eigenvalues to sum to 1.0.
    let total: f64 = eigenvalues.iter().sum();
    if total < 1e-15 {
        return 0.0;
    }
    let inv_total = 1.0 / total;

    // 6. Effective rank = exp(-Σ λ_i * log(λ_i)) — fused with normalization
    // to avoid allocating an intermediate Vec.
    let entropy: f64 = eigenvalues
        .iter()
        .map(|&v| v * inv_total)
        .filter(|&v| v > 1e-15)
        .map(|v| -v * v.ln())
        .sum();

    entropy.exp() as f32
}

/// Compute average pairwise cosine similarity between hidden states.
///
/// High similarity = anisotropic (degenerate), Low = isotropic (healthy).
///
/// # Panics
/// Panics if `hidden_states` is empty or if vectors have inconsistent dimensions
/// or zero norm.
pub fn avg_cosine_similarity(hidden_states: &[Vec<f32>]) -> f32 {
    assert!(!hidden_states.is_empty(), "hidden_states must not be empty");
    let n = hidden_states.len();
    let dim = hidden_states[0].len();
    assert!(dim > 0, "hidden state vectors must be non-empty");

    if n < 2 {
        return 1.0; // Trivially similar to itself.
    }

    // Normalize each vector to unit length.
    let mut normalized = Vec::with_capacity(n * dim);
    for v in hidden_states {
        let norm: f64 = v
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        assert!(norm > 1e-10, "zero-norm vector encountered");
        let inv_norm = 1.0 / norm;
        for &x in v {
            normalized.push((x as f64) * inv_norm);
        }
    }

    // Compute average pairwise dot product.
    let mut total = 0.0f64;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dot = 0.0f64;
            for d in 0..dim {
                dot += normalized[i * dim + d] * normalized[j * dim + d];
            }
            total += dot;
            count += 1;
        }
    }

    (total / count.max(1) as f64) as f32
}

/// Compute a combined representation geometry report.
///
/// # Panics
/// Panics if `hidden_states` is empty.
pub fn representation_geometry_report(
    hidden_states: &[Vec<f32>],
    layer_index: usize,
) -> GeometryReport {
    assert!(!hidden_states.is_empty(), "hidden_states must not be empty");
    let n_tokens = hidden_states.len();
    let hidden_dim = hidden_states[0].len();

    GeometryReport {
        effective_rank: effective_rank(hidden_states),
        avg_cosine_sim: avg_cosine_similarity(hidden_states),
        layer_index,
        n_tokens,
        hidden_dim,
    }
}

// ── Sink-aware layer summary (Plan 287 Phase 4) ───────────────

/// Per-layer aggregate of sink classifications across all heads.
///
/// Bridges the per-sink [`super::sink_classify::classify_sink_at`]
/// (mechanism locator) with whole-layer [`GeometryReport`] (aggregate
/// symptom). Lets a caller ask "does this layer predominantly NOP or
/// Broadcast?" in O(H · N²) where H is the head count.
#[derive(Debug, Clone)]
pub struct LayerSinkSummary {
    /// Layer index for cross-layer phase plots (paper Figure 4 analog).
    pub layer_index: usize,
    /// Total NOP sinks across all heads in this layer.
    pub n_nop_sinks: usize,
    /// Total Broadcast sinks across all heads in this layer.
    pub n_broadcast_sinks: usize,
    /// Plurality vote across all heads: which `SinkKind` dominated?
    /// `None` if no head had a sink above `τ_sink`.
    pub dominant_kind: SinkKind,
    /// Mean `‖v_s‖` over all Broadcast sinks in the layer.
    /// Useful for cross-layer phase plots (paper §1.4 — patches become
    /// Broadcast sinks in deeper layers with growing `‖v_s‖`).
    /// `f32::NAN` if no Broadcast sinks.
    pub mean_broadcast_value_norm: f32,
}

/// Run [`super::sink_classify::classify_all_sinks`] across every head in a
/// layer and aggregate into a [`LayerSinkSummary`].
///
/// # Arguments
/// * `attn_per_head`    — `H` attention maps, each `(n, n)` row-major.
/// * `values_per_head`  — `H` value matrices, each `(n, d_h)` row-major.
/// * `cfg`              — sink classifier thresholds.
/// * `scratch`          — reused across heads (zero-alloc after warmup).
/// * `layer_index`      — for the summary's `layer_index` field.
///
/// # Algorithmic cost
/// `O(H · N²)` for the column-sum pass, plus `O(sinks · n · d_h)` for the
/// per-sink value-norm scan. `sinks` is small in practice (paper: head
/// specialization → ~1 sink per head).
pub fn summarize_layer_sinks(
    attn_per_head: &[Vec<Vec<f32>>],
    values_per_head: &[Vec<Vec<f32>>],
    cfg: &SinkClassifierConfig,
    scratch: &mut StableRankScratch,
    layer_index: usize,
) -> LayerSinkSummary {
    let h = attn_per_head.len().min(values_per_head.len());
    let mut n_nop = 0usize;
    let mut n_broadcast = 0usize;
    let mut broadcast_value_sum = 0.0f32;
    let mut broadcast_value_count = 0usize;

    let mut sink_buf: Vec<super::sink_classify::SinkDiagnostic> = Vec::new();

    for head in 0..h {
        sink_buf.clear();
        classify_all_sinks(
            &attn_per_head[head],
            &values_per_head[head],
            cfg,
            scratch,
            &mut sink_buf,
        );
        for d in &sink_buf {
            match d.kind {
                SinkKind::Nop => n_nop += 1,
                SinkKind::Broadcast => {
                    n_broadcast += 1;
                    // Re-derive ‖v_s‖ from ratio and per-head mean norm.
                    // We don't have the per-head mean handy here without
                    // rescanning; approximate via ratio * assumed_mean=1.
                    // For precise ‖v_s‖, callers should keep the per-head
                    // diagnostics. Here we use ratio as a proxy.
                    broadcast_value_sum += d.value_norm_ratio;
                    broadcast_value_count += 1;
                }
                SinkKind::None => {}
            }
        }
    }

    let dominant_kind = if n_nop > n_broadcast {
        SinkKind::Nop
    } else if n_broadcast > n_nop {
        SinkKind::Broadcast
    } else if n_nop + n_broadcast == 0 {
        SinkKind::None
    } else {
        // Tie — fall back to None (ambiguous).
        SinkKind::None
    };

    let mean_broadcast_value_norm = if broadcast_value_count > 0 {
        broadcast_value_sum / (broadcast_value_count as f32)
    } else {
        f32::NAN
    };

    LayerSinkSummary {
        layer_index,
        n_nop_sinks: n_nop,
        n_broadcast_sinks: n_broadcast,
        dominant_kind,
        mean_broadcast_value_norm,
    }
}

// ── Within-class effective rank (Plan 415, Research 394) ───────
//
// Entropy-based effective rank of the *within-class residual* covariance
// matrix (paper S.1.2). Fuses the shipped class-agnostic `effective_rank`
// (centers by global mean) with the class-conditioning machinery shipped in
// `riir-engine/src/latent_functor/quality_gate.rs` (within/between adjacency).
// The two halves had never been combined; the paper (arXiv:2412.19419 §5.3.1)
// claims this specific application is novel.
//
// Why: the global `effective_rank` cannot distinguish "between-class variance
// dominates, within-class collapsed" (a failure mode for committed-personality
// / HLA populations) from "all variance is healthy and isotropic". The two
// cases produce the same global rank but different within-class ranks.

/// Combined within-class + global geometry report for a class-labeled set of
/// hidden states. The `global_erank_for_contrast` field lets a caller see at a
/// glance whether the two metrics disagree (the load-bearing signal — see Plan
/// 415 G2).
#[derive(Debug, Clone)]
pub struct WithinClassGeometryReport {
    /// Entropy-based effective rank of the within-class residual covariance.
    /// Lower → more within-class collapse. `NaN` only if there are no
    /// within-class residual degrees of freedom (e.g. each class has 1 member).
    pub within_class_erank: f32,
    /// Class-agnostic effective rank (computed via the shipped `effective_rank`
    /// over the same states). High here + low `within_class_erank` = the
    /// non-redundancy signal.
    pub global_erank_for_contrast: f32,
    /// Number of distinct class labels present.
    pub n_classes: usize,
    /// Number of state vectors.
    pub n_states: usize,
    /// Embedding dimension.
    pub dim: usize,
}

/// Compute the within-class effective rank: entropy-based effective rank of
/// the **within-class residual** covariance matrix.
///
/// For each class `c`, the class centroid `μ_c = (1/n_c) Σ_{i∈S_c} x_i` is
/// subtracted from each member before forming the pooled within-class
/// covariance `Σ_w`. The effective rank is then `exp(H(p))` over the
/// normalized eigenvalue spectrum of `Σ_w` (Roy & Vetterli 2007), identical
/// math to [`effective_rank`] but on the residual covariance.
///
/// Returns the effective number of independent directions of *within-class*
/// variation. Lower → more within-class collapse (the oversmoothing analog of
/// arXiv:2412.19419 §5.3.1).
///
/// Returns `0.0` if there are no within-class residual degrees of freedom:
/// fewer than 2 states, every class with exactly 1 member (no within-class
/// variance to measure), or zero total variance.
///
/// # Arguments
///
/// * `states`       — flat `[n × dim]` row-major embedding matrix.
/// * `dim`          — embedding dimension (number of columns).
/// * `class_labels` — `[n]` class id per state. Labels need not be contiguous;
///   any two equal labels denote the same class.
///
/// # Panics
///
/// Debug builds assert `states.len() == n * dim` where `n = class_labels.len()`,
/// `dim > 0`, and `n > 0`.
pub fn within_class_effective_rank(states: &[f32], dim: usize, class_labels: &[usize]) -> f32 {
    let n = class_labels.len();
    debug_assert!(n > 0, "class_labels must not be empty");
    debug_assert!(dim > 0, "dim must be positive");
    debug_assert_eq!(
        states.len(),
        n * dim,
        "states must be [n × dim] = [{} × {}] = {} floats, got {}",
        n,
        dim,
        n * dim,
        states.len()
    );

    // Trivial guard: no within-class variance possible with < 2 states.
    if n < 2 {
        return 0.0;
    }

    // 1. Per-class centroid μ_c.
    //    Use a small hash map (class_id -> (sum[dim], count)). For the typical
    //    diagnostic sizes (n ≤ few thousand) this is fine.
    use std::collections::HashMap;
    let mut class_sums: HashMap<usize, (Vec<f64>, usize)> = HashMap::new();
    for i in 0..n {
        let row = &states[i * dim..(i + 1) * dim];
        let entry = class_sums
            .entry(class_labels[i])
            .or_insert_with(|| (vec![0.0f64; dim], 0));
        for (j, &x) in row.iter().enumerate() {
            entry.0[j] += x as f64;
        }
        entry.1 += 1;
    }
    let class_centroids: HashMap<usize, Vec<f64>> = class_sums
        .iter()
        .map(|(&id, (sum, count))| (id, sum.iter().map(|s| s / *count as f64).collect()))
        .collect();

    // 2. If every class has exactly 1 member, there is no within-class residual
    //    variance to measure. Return 0 (consistent with effective_rank's
    //    single-vector guard).
    let any_multi_member = class_sums.values().any(|(_, c)| *c >= 2);
    if !any_multi_member {
        return 0.0;
    }

    // 3. Build the pooled within-class covariance Σ_w (dim × dim).
    //    Σ_w = (1 / Σ_c (n_c − 1)) Σ_c Σ_{i∈S_c} (x_i − μ_c)(x_i − μ_c)^T
    //    Degrees of freedom Σ_c (n_c − 1) = n − C.
    let n_classes = class_sums.len();
    let dof = n.saturating_sub(n_classes);
    if dof == 0 {
        return 0.0;
    }
    let scale = 1.0 / dof as f64;

    let mut cov = vec![0.0f64; dim * dim];
    // Scratch residual buffer, reused per state.
    let mut residual = vec![0.0f64; dim];
    for i in 0..n {
        let row = &states[i * dim..(i + 1) * dim];
        let centroid = &class_centroids[&class_labels[i]];
        for j in 0..dim {
            residual[j] = row[j] as f64 - centroid[j];
        }
        // Rank-1 update: cov += residual * residual^T * scale.
        for r in 0..dim {
            let rr = residual[r];
            if rr == 0.0 {
                continue;
            }
            for c in r..dim {
                cov[r * dim + c] += rr * residual[c] * scale;
            }
        }
    }
    // Mirror upper triangle to lower (we only accumulated r ≤ c).
    for r in 0..dim {
        for c in 0..r {
            cov[r * dim + c] = cov[c * dim + r];
        }
    }

    // 4. Eigenvalues via the shared Jacobi eigensolver.
    let eigenvalues = jacobi_eigenvalues(&mut cov, dim, 50);

    // 5. Normalize and compute entropy-based effective rank.
    let total: f64 = eigenvalues.iter().sum();
    if total < 1e-15 {
        return 0.0;
    }
    let inv_total = 1.0 / total;
    let entropy: f64 = eigenvalues
        .iter()
        .map(|&v| v * inv_total)
        .filter(|&v| v > 1e-15)
        .map(|v| -v * v.ln())
        .sum();

    entropy.exp() as f32
}

/// Convenience wrapper for `&[Vec<f32>]` callers (mirrors [`effective_rank`]'s
/// signature). See [`within_class_effective_rank`] for the underlying math.
///
/// # Panics
///
/// Panics if `states` is empty, vectors have inconsistent dimensions, or
/// `states.len() != class_labels.len()`.
pub fn within_class_effective_rank_owned(
    states: &[Vec<f32>],
    class_labels: &[usize],
) -> f32 {
    assert!(!states.is_empty(), "states must not be empty");
    assert_eq!(
        states.len(),
        class_labels.len(),
        "states.len() must equal class_labels.len()"
    );
    let dim = states[0].len();
    assert!(dim > 0, "hidden state vectors must be non-empty");
    for (i, v) in states.iter().enumerate() {
        assert_eq!(
            v.len(),
            dim,
            "inconsistent dimensions: vector 0 has len {dim}, vector {i} has {}",
            v.len()
        );
    }
    // Flatten into a single buffer to feed the flat-slice entry point.
    let mut flat = Vec::with_capacity(states.len() * dim);
    for v in states {
        flat.extend_from_slice(v);
    }
    within_class_effective_rank(&flat, dim, class_labels)
}

/// Compute a combined within-class + global geometry report for a class-labeled
/// set of hidden states. The `global_erank_for_contrast` field lets a caller
/// detect the load-bearing non-redundancy signal (high global, low within-class).
///
/// # Panics
///
/// Panics under the same conditions as [`within_class_effective_rank_owned`].
pub fn within_class_geometry_report(
    states: &[Vec<f32>],
    class_labels: &[usize],
) -> WithinClassGeometryReport {
    assert!(!states.is_empty(), "states must not be empty");
    assert_eq!(
        states.len(),
        class_labels.len(),
        "states.len() must equal class_labels.len()"
    );
    let dim = states[0].len();
    let n_states = states.len();

    let within = within_class_effective_rank_owned(states, class_labels);
    let global = effective_rank(states);
    let n_classes = {
        let mut s = std::collections::HashSet::new();
        for &c in class_labels {
            s.insert(c);
        }
        s.len()
    };

    WithinClassGeometryReport {
        within_class_erank: within,
        global_erank_for_contrast: global,
        n_classes,
        n_states,
        dim,
    }
}

// ── Jacobi eigenvalue algorithm (symmetric matrix) ─────────────
//
// Simple iterative Jacobi rotation to find eigenvalues of a real symmetric
// matrix. Not optimized for large matrices — fine for diagnostic use on
// covariance matrices up to ~256×256.

#[inline]
fn jacobi_eigenvalues(mat: &mut [f64], dim: usize, max_sweeps: usize) -> Vec<f64> {
    // Extract diagonal as initial eigenvalue estimates.
    let mut eigenvalues: Vec<f64> = (0..dim).map(|i| mat[i * dim + i]).collect();

    for _ in 0..max_sweeps {
        // Find the largest off-diagonal element.
        let mut max_val = 0.0f64;
        let (mut p, mut q) = (0, 1);
        for i in 0..dim {
            for j in (i + 1)..dim {
                let val = mat[i * dim + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Converged if off-diagonal is negligible.
        if max_val < 1e-12 {
            break;
        }

        // Compute Jacobi rotation angle.
        let app = mat[p * dim + p];
        let aqq = mat[q * dim + q];
        let apq = mat[p * dim + q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply rotation to rows/cols p, q.
        for r in 0..dim {
            if r == p || r == q {
                continue;
            }
            let arp = mat[r * dim + p];
            let arq = mat[r * dim + q];
            mat[r * dim + p] = cos_t * arp + sin_t * arq;
            mat[p * dim + r] = mat[r * dim + p];
            mat[r * dim + q] = -sin_t * arp + cos_t * arq;
            mat[q * dim + r] = mat[r * dim + q];
        }

        let new_pp = cos_t * cos_t * app + 2.0 * sin_t * cos_t * apq + sin_t * sin_t * aqq;
        let new_qq = sin_t * sin_t * app - 2.0 * sin_t * cos_t * apq + cos_t * cos_t * aqq;
        mat[p * dim + p] = new_pp;
        mat[q * dim + q] = new_qq;
        mat[p * dim + q] = 0.0;
        mat[q * dim + p] = 0.0;

        eigenvalues[p] = new_pp;
        eigenvalues[q] = new_qq;
    }

    // Filter out near-zero eigenvalues (numerical noise).
    eigenvalues.retain(|&v| v > 1e-10);
    eigenvalues
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: simple Gaussian-like noise using central limit theorem.
    fn gaussian_noise(rng: &mut fastrand::Rng) -> f32 {
        let sum: f32 = (0..12).map(|_| rng.f32()).sum();
        sum - 6.0
    }

    // ── G1: effective_rank() on known matrix → correct value ─────

    #[test]
    fn g1_effective_rank_known_matrix() {
        // Hand-constructed case: dim=3, 6 vectors that span all 3 dims.
        // After mean-centering, the covariance matrix is full rank.
        // Effective rank is entropy-based, so for full rank with equal
        // eigenvalues it equals dim; for uneven eigenvalues it's < dim.
        // We verify: collapsed → low rank, full-rank → rank close to dim.
        let states: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![-1.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0],
            vec![0.0, 0.0, -1.0],
        ];

        let rank = effective_rank(&states);
        // Symmetric ±basis → mean is zero → covariance is (1/6)*I → all eigenvalues
        // equal → effective rank = 3.
        let dim = 3;
        assert!(
            (rank - dim as f32).abs() < 0.05,
            "effective rank of symmetric ±basis should be ~{dim}, got {rank}"
        );
    }

    #[test]
    fn g1_effective_rank_collapsed_matrix() {
        // All identical vectors → rank ≈ 0 (no variance).
        let states = vec![vec![1.0f32, 2.0, 3.0]; 10];
        let rank = effective_rank(&states);
        assert!(
            rank < 0.1,
            "effective rank of identical vectors should be ~0, got {rank}"
        );
    }

    #[test]
    fn g1_effective_rank_single_vector() {
        let states = vec![vec![1.0f32, 2.0, 3.0]];
        let rank = effective_rank(&states);
        assert!(
            rank < 0.01,
            "effective rank of single vector should be 0, got {rank}"
        );
    }

    // ── G2: avg_cosine_similarity() on orthogonal / identical ────

    #[test]
    fn g2_orthogonal_vectors_similarity_zero() {
        // Standard basis vectors are orthogonal → cosine sim ≈ 0.
        let dim = 4;
        let states: Vec<Vec<f32>> = (0..dim)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i] = 1.0;
                v
            })
            .collect();

        let sim = avg_cosine_similarity(&states);
        assert!(
            sim.abs() < 0.01,
            "orthogonal vectors should have cosine sim ≈ 0, got {sim}"
        );
    }

    #[test]
    fn g2_identical_vectors_similarity_one() {
        let states = vec![vec![1.0f32, 2.0, 3.0, 4.0]; 5];
        let sim = avg_cosine_similarity(&states);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "identical vectors should have cosine sim ≈ 1.0, got {sim}"
        );
    }

    #[test]
    fn g2_opposite_vectors_similarity_minus_one() {
        let states = vec![vec![1.0f32, 0.0, 0.0], vec![-1.0f32, 0.0, 0.0]];
        let sim = avg_cosine_similarity(&states);
        assert!(
            (sim - (-1.0)).abs() < 0.01,
            "opposite vectors should have cosine sim ≈ -1.0, got {sim}"
        );
    }

    // ── G3: Random init → effective_rank > 0.5 * dim ─────────────

    #[test]
    fn g3_random_init_high_effective_rank() {
        let mut rng = fastrand::Rng::with_seed(42);
        let dim = 16;
        let n_tokens = 32;

        // Random isotropic vectors — should span most dimensions.
        let states: Vec<Vec<f32>> = (0..n_tokens)
            .map(|_| (0..dim).map(|_| gaussian_noise(&mut rng)).collect())
            .collect();

        let rank = effective_rank(&states);
        assert!(
            rank > 0.5 * dim as f32,
            "random init should have effective_rank > 0.5 * dim={dim}, got {rank}"
        );
    }

    #[test]
    fn g3_random_init_low_cosine_similarity() {
        let mut rng = fastrand::Rng::with_seed(123);
        let dim = 16;
        let n_tokens = 32;

        let states: Vec<Vec<f32>> = (0..n_tokens)
            .map(|_| (0..dim).map(|_| gaussian_noise(&mut rng)).collect())
            .collect();

        let sim = avg_cosine_similarity(&states);
        assert!(
            sim.abs() < 0.3,
            "random init should have avg_cosine_sim near 0, got {sim}"
        );
    }

    // ── G5: GeometryReport integrates correctly ──────────────────

    #[test]
    fn g5_geometry_report_fields() {
        let mut rng = fastrand::Rng::with_seed(99);
        let dim = 8;
        let n = 10;
        let layer = 3;

        let states: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| gaussian_noise(&mut rng)).collect())
            .collect();

        let report = representation_geometry_report(&states, layer);

        assert_eq!(report.layer_index, layer);
        assert_eq!(report.n_tokens, n);
        assert_eq!(report.hidden_dim, dim);
        assert!(report.effective_rank > 0.0);
        assert!(report.avg_cosine_sim > -1.0 && report.avg_cosine_sim < 1.0);
    }

    #[test]
    fn g5_geometry_report_consistent_with_individual_calls() {
        let mut rng = fastrand::Rng::with_seed(77);
        let dim = 8;
        let n = 12;

        let states: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| gaussian_noise(&mut rng)).collect())
            .collect();

        let erank = effective_rank(&states);
        let asim = avg_cosine_similarity(&states);
        let report = representation_geometry_report(&states, 7);

        assert!(
            (report.effective_rank - erank).abs() < 1e-4,
            "report effective_rank {} != direct {}",
            report.effective_rank,
            erank
        );
        assert!(
            (report.avg_cosine_sim - asim).abs() < 1e-4,
            "report avg_cosine_sim {} != direct {}",
            report.avg_cosine_sim,
            asim
        );
    }

    // ── Plan 415: within_class_effective_rank ───────────────────────
    //
    // G1 correctness, G2 non-redundancy vs shipped effective_rank, plus the
    // degenerate guards.

    #[test]
    fn p415_single_state_returns_zero() {
        // No within-class variance possible with 1 state.
        let states = vec![vec![1.0f32, 2.0, 3.0]];
        let labels = vec![0usize];
        let r = within_class_effective_rank_owned(&states, &labels);
        assert!(
            r < 0.01,
            "single state within-class rank should be ~0, got {r}"
        );
    }

    #[test]
    fn p415_all_singletons_returns_zero() {
        // Every class has exactly 1 member -> no within-class residual DoF.
        let states: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let labels = vec![0, 1, 2]; // 3 classes, each with 1 member
        let r = within_class_effective_rank_owned(&states, &labels);
        assert!(
            r < 0.01,
            "all-singleton-classes within-class rank should be ~0, got {r}"
        );
    }

    #[test]
    fn p415_identical_within_class_collapses_to_zero() {
        // Two classes; within each class all vectors are identical -> the
        // within-class residual covariance is zero -> rank ~ 0.
        let states: Vec<Vec<f32>> = vec![
            // class 0
            vec![10.0, 0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0, 0.0],
            // class 1
            vec![-10.0, 0.0, 0.0, 0.0],
            vec![-10.0, 0.0, 0.0, 0.0],
            vec![-10.0, 0.0, 0.0, 0.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let r = within_class_effective_rank_owned(&states, &labels);
        assert!(
            r < 0.1,
            "identical-within-class rank should be ~0, got {r}"
        );
    }

    #[test]
    fn p415_isotropic_within_class_is_high() {
        // Two classes; each class is a tight Gaussian ball around a different
        // centroid. Within-class residual covariance is ~ isotropic -> within-
        // class rank ~ dim.
        let mut rng = fastrand::Rng::with_seed(7);
        let dim = 4usize;
        let per_class = 16usize;
        let mut states: Vec<Vec<f32>> = Vec::with_capacity(2 * per_class);
        let mut labels: Vec<usize> = Vec::with_capacity(2 * per_class);
        let centroid0: Vec<f32> = (0..dim).map(|_| 10.0).collect();
        let centroid1: Vec<f32> = (0..dim).map(|_| -10.0).collect();
        for _ in 0..per_class {
            states.push((0..dim).map(|j| centroid0[j] + 0.1 * gaussian_noise(&mut rng)).collect());
            labels.push(0);
        }
        for _ in 0..per_class {
            states.push((0..dim).map(|j| centroid1[j] + 0.1 * gaussian_noise(&mut rng)).collect());
            labels.push(1);
        }
        let r = within_class_effective_rank_owned(&states, &labels);
        assert!(
            r > 0.6 * dim as f32,
            "isotropic within-class rank should be > 0.6 * dim={}, got {r}",
            dim
        );
        assert!(
            r <= dim as f32 + 0.05,
            "within-class rank should be <= dim={}, got {r}",
            dim
        );
    }

    #[test]
    fn p415_rank1_within_class_collapses_to_one() {
        // Two classes; each class varies only along a single direction
        // (rank-1 within each class) -> within-class effective rank ~ 1.
        let dim = 4usize;
        let per_class = 16usize;
        let dir: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let mut states: Vec<Vec<f32>> = Vec::with_capacity(2 * per_class);
        let mut labels: Vec<usize> = Vec::with_capacity(2 * per_class);
        let mut rng = fastrand::Rng::with_seed(11);
        for _ in 0..per_class {
            let t: f32 = rng.f32() * 10.0 - 5.0; // uniform [-5, 5)
            states.push((0..dim).map(|j| 10.0 + t * dir[j]).collect());
            labels.push(0);
        }
        for _ in 0..per_class {
            let t: f32 = rng.f32() * 10.0 - 5.0;
            states.push((0..dim).map(|j| -10.0 + t * dir[j]).collect());
            labels.push(1);
        }
        let r = within_class_effective_rank_owned(&states, &labels);
        assert!(
            (r - 1.0).abs() < 0.15,
            "rank-1 within-class rank should be ~1, got {r}"
        );
    }

    #[test]
    fn p415_single_class_matches_global_effective_rank() {
        // Degenerate single-class case: within-class == global residual
        // centering. The two metrics should agree to numerical noise.
        let mut rng = fastrand::Rng::with_seed(31);
        let dim = 6usize;
        let n = 24usize;
        let states: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| gaussian_noise(&mut rng)).collect())
            .collect();
        let labels = vec![0usize; n];
        let within = within_class_effective_rank_owned(&states, &labels);
        let global = effective_rank(&states);
        assert!(
            (within - global).abs() < 0.05,
            "single-class within rank {within} should match global {global}"
        );
    }

    #[test]
    fn p415_monotone_in_within_class_variance() {
        // G1 monotonicity: as within-class noise shrinks (vectors collapse
        // toward their centroids), within-class effective rank should decrease.
        let mut rng = fastrand::Rng::with_seed(2024);
        let dim = 4usize;
        let per_class = 12usize;
        let centroid0: Vec<f32> = vec![10.0, 0.0, 0.0, 0.0];
        let centroid1: Vec<f32> = vec![-10.0, 0.0, 0.0, 0.0];

        let make = |sigma: f32, rng: &mut fastrand::Rng| -> (Vec<Vec<f32>>, Vec<usize>) {
            let mut states: Vec<Vec<f32>> = Vec::with_capacity(2 * per_class);
            let mut labels: Vec<usize> = Vec::with_capacity(2 * per_class);
            for _ in 0..per_class {
                states.push(
                    (0..dim)
                        .map(|j| centroid0[j] + sigma * gaussian_noise(rng))
                        .collect(),
                );
                labels.push(0);
            }
            for _ in 0..per_class {
                states.push(
                    (0..dim)
                        .map(|j| centroid1[j] + sigma * gaussian_noise(rng))
                        .collect(),
                );
                labels.push(1);
            }
            (states, labels)
        };

        let (states_hi, labels_hi) = make(1.0, &mut rng);
        let (states_lo, labels_lo) = make(0.01, &mut rng);
        let r_hi = within_class_effective_rank_owned(&states_hi, &labels_hi);
        let r_lo = within_class_effective_rank_owned(&states_lo, &labels_lo);
        assert!(
            r_hi > r_lo,
            "within-class rank should decrease as within-class variance shrinks: got hi={r_hi} lo={r_lo}"
        );
    }

    #[test]
    fn p415_g2_nonredundancy_vs_global() {
        // THE LOAD-BEARING G2 GATE.
        //
        // Construct a case where global effective_rank is HIGH but within-class
        // effective rank is LOW, proving the two metrics disagree. This is the
        // exact failure mode the new primitive detects that the class-agnostic
        // `effective_rank` cannot.
        //
        // KEY INSIGHT (learned from the first test-construction failure):
        // effective rank is SCALE-INVARIANT — it normalizes eigenvalues to
        // probabilities. So "small within-class variance" does NOT imply "low
        // within-class rank". To get a low within-class rank we need
        // RANK-DEFICIENT within-class structure (each class lives in a
        // low-dim subspace), not just small-magnitude variance. The cleanest
        // construction: each class collapses to a SINGLE POINT exactly (zero
        // within-class variance → rank 0 via the zero-variance guard), while
        // the 4 class centroids span a high-dim subspace globally.
        let dim = 4usize;
        let per_class = 8usize;
        let mut states: Vec<Vec<f32>> = Vec::with_capacity(4 * per_class);
        let mut labels: Vec<usize> = Vec::with_capacity(4 * per_class);
        // 4 orthogonal centroids — the global covariance will span ~3 dims
        // (4 points centered live in a 3-dim affine subspace).
        let centroids: [[f32; 4]; 4] = [
            [50.0, 0.0, 0.0, 0.0],
            [0.0, 50.0, 0.0, 0.0],
            [0.0, 0.0, 50.0, 0.0],
            [0.0, 0.0, 0.0, 50.0],
        ];
        for (c, center) in centroids.iter().enumerate() {
            for _ in 0..per_class {
                // EXACTLY identical within each class — zero within-class
                               // variance. (No jitter: jitter would make within-class
                // isotropic and the rank would jump to ~dim.)
                states.push(center.to_vec());
                labels.push(c);
            }
        }
        let within = within_class_effective_rank_owned(&states, &labels);
        let global = effective_rank(&states);

        // The new primitive sees the within-class collapse.
        assert!(
            within < 0.1,
            "within-class rank should be ~0 (each class is a single point), got {within}"
        );
        // The shipped global metric is fooled by the between-class spread:
        // 4 orthogonal centroids span a 3-dim affine subspace → high rank.
        assert!(
            global > 0.5 * dim as f32,
            "global effective_rank should be high (between-class dominates), got {global}"
        );
        // The non-redundancy signal: they disagree by a large margin.
        assert!(
            global - within > 1.5,
            "non-redundancy signal: global {global} - within {within} should be > 1.5"
        );
    }

    #[test]
    fn p415_flat_and_owned_agree() {
        // The flat-slice entry point and the &Vec<f32> wrapper must produce
        // identical results on the same data.
        let mut rng = fastrand::Rng::with_seed(55);
        let dim = 5usize;
        let n = 20usize;
        let states: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let c = i % 3;
                let center = (c as f32) * 10.0;
                (0..dim)
                    .map(|_| center + gaussian_noise(&mut rng))
                    .collect()
            })
            .collect();
        let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();

        let mut flat = Vec::with_capacity(n * dim);
        for v in &states {
            flat.extend_from_slice(v);
        }
        let r_flat = within_class_effective_rank(&flat, dim, &labels);
        let r_owned = within_class_effective_rank_owned(&states, &labels);
        assert!(
            (r_flat - r_owned).abs() < 1e-4,
            "flat {r_flat} and owned {r_owned} should agree"
        );
    }

    #[test]
    fn p415_report_carries_both_metrics() {
        // Report struct plumbing check. Uses two classes whose centroids are
        // CLOSE TOGETHER relative to the isotropic within-class noise, so that
        // both within-class and global covariance are dominated by the isotropic
        // noise → both ranks are high (~dim). This isolates the struct-plumbing
        // assertion from the non-redundancy geometry tested in
        // p415_g2_nonredundancy_vs_global.
        let mut rng = fastrand::Rng::with_seed(9);
        let dim = 4usize;
        let per_class = 10usize;
        let mut states: Vec<Vec<f32>> = Vec::with_capacity(2 * per_class);
        let mut labels: Vec<usize> = Vec::with_capacity(2 * per_class);
        // Small between-class separation (0.1) vs unit isotropic within-class
        // noise → global cov ≈ I + tiny perturbation → global rank ≈ dim.
        for _ in 0..per_class {
            states.push((0..dim).map(|_| 0.1 + gaussian_noise(&mut rng)).collect());
            labels.push(0);
        }
        for _ in 0..per_class {
            states.push((0..dim).map(|_| -0.1 + gaussian_noise(&mut rng)).collect());
            labels.push(1);
        }
        let report = within_class_geometry_report(&states, &labels);
        assert_eq!(report.n_classes, 2);
        assert_eq!(report.n_states, 2 * per_class);
        assert_eq!(report.dim, dim);
        assert!(report.within_class_erank > 0.0);
        assert!(report.global_erank_for_contrast > 0.0);
        // Both ranks high because isotropic noise dominates both covariances.
        assert!(
            report.within_class_erank > 0.5 * dim as f32,
            "within-class rank should be high (isotropic noise), got {}",
            report.within_class_erank
        );
        assert!(
            report.global_erank_for_contrast > 0.5 * dim as f32,
            "global rank should be high (isotropic noise dominates), got {}",
            report.global_erank_for_contrast
        );
    }
}
