//! MAG transfer prediction: modelless ranking of candidate datasets/experiences
//! by predicted transfer to a target capability (arXiv:2607.04222 §4).
//!
//! The headline capability: given a target capability `T` and a pool of
//! candidate experiences/datasets, predict which candidate will most improve `T`
//! — **without** running any training. The paper achieves 94.7% Top-1 accuracy
//! on their 18-dataset corpus; raw centroid cosine achieves only ρ ≈ 0.03 (near
//! random). The class-conditional MAG metrics (cosine on the positive/negative
//! class centroids) carry the signal; raw cosine does not.
//!
//! This is the **directed curiosity** signal: not "what's novel?" (entropy) but
//! "what transfers to my goal?" (MAG geometry). Consumed by CGSP curiosity
//! routing and the AnyRAG escalation gateway in riir-ai / riir-neuron-db.

use super::types::{check_dim, cosine, MagError, TransferMetric};

// ── Dataset view ───────────────────────────────────────────────────

/// A borrowed dataset view: activation samples + per-sample verdict labels.
///
/// Generic over sample storage (`Vec<f32>`, `[f32; D]`, `&[f32]`, etc.) via
/// `AsRef<[f32]>`. Labels correspond to the model/runtime's own verdict `y_M`
/// for each sample — the same unsupervised label source used by
/// [`mine_contrast_direction`](super::mining::mine_contrast_direction).
///
/// `labels.len()` must equal `activations.len()`; this is checked by
/// [`transfer_score`] and [`rank_candidates`].
#[derive(Debug, Clone, Copy)]
pub struct DataSet<'a, S: AsRef<[f32]>> {
    /// `N × d` activation readouts. These should already be operator-applied
    /// (via [`apply_operator`](super::mining::apply_operator)) if the caller
    /// wants a MAG-operator-based score rather than a raw-activation score.
    pub activations: &'a [S],
    /// Per-sample verdict labels (`true` = positive/benign class, `false` =
    /// negative/malicious class).
    pub labels: &'a [bool],
}

impl<'a, S: AsRef<[f32]>> DataSet<'a, S> {
    /// Construct a dataset view from activations + labels.
    #[inline]
    pub fn new(activations: &'a [S], labels: &'a [bool]) -> Self {
        Self { activations, labels }
    }

    /// Number of samples.
    #[inline]
    pub fn len(&self) -> usize {
        self.activations.len()
    }

    /// Whether the dataset is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.activations.is_empty()
    }
}

// ── Single-metric transfer score ───────────────────────────────────

/// Compute the transfer score of `candidate` relative to `target` under `metric`.
///
/// All metrics return **higher = better predicted transfer** — distance-based
/// metrics are negated so that `0` = identical, negative = dissimilar. See
/// [`TransferMetric`] for the semantics of each.
///
/// The activation sets may have different sample counts but must share the same
/// per-sample dimensionality `d`. Labels must match the activation count in each
/// set.
///
/// # Example
///
/// ```
/// # use katgpt_core::mag::{DataSet, TransferMetric, transfer_score};
/// // Identical candidate and target → centroid cosine = 1.0 (perfect transfer).
/// let acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// let labels = [true, false, true];
/// let ds = DataSet::new(&acts, &labels);
/// let score = transfer_score(&ds, &ds, TransferMetric::CentroidCosine).unwrap();
/// assert!((score - 1.0).abs() < 1e-5);
/// ```
pub fn transfer_score<S: AsRef<[f32]>>(
    candidate: &DataSet<'_, S>,
    target: &DataSet<'_, S>,
    metric: TransferMetric,
) -> Result<f32, MagError> {
    let d = check_dim(candidate.activations)?;
    let d2 = check_dim(target.activations)?;
    if d != d2 {
        return Err(MagError::DimMismatch);
    }
    if candidate.labels.len() != candidate.activations.len()
        || target.labels.len() != target.activations.len()
    {
        return Err(MagError::DimMismatch);
    }

    match metric {
        TransferMetric::CentroidCosine => {
            let c = centroid(candidate.activations, d);
            let t = centroid(target.activations, d);
            Ok(cosine(&c, &t))
        }
        TransferMetric::Euclidean => {
            let c = centroid(candidate.activations, d);
            let t = centroid(target.activations, d);
            let mut dist_sq = 0.0;
            for j in 0..d {
                let diff = c[j] - t[j];
                dist_sq += diff * diff;
            }
            Ok(-dist_sq.sqrt())
        }
        TransferMetric::Correlation => {
            let c = centroid(candidate.activations, d);
            let t = centroid(target.activations, d);
            Ok(pearson(&c, &t))
        }
        TransferMetric::RbfMmd => {
            let gamma = 1.0 / d as f32;
            Ok(-rbf_mmd_sq(candidate.activations, target.activations, gamma))
        }
        TransferMetric::Wasserstein1d => {
            Ok(-wasserstein1d(candidate.activations, target.activations, d))
        }
        TransferMetric::CkaLinear => Ok(cka_linear(candidate.activations, target.activations, d)),
        TransferMetric::ClassConditionalCosineMalicious => {
            let c = class_centroid(candidate.activations, candidate.labels, false, d)?;
            let t = class_centroid(target.activations, target.labels, false, d)?;
            Ok(cosine(&c, &t))
        }
        TransferMetric::ClassConditionalCosineBenign => {
            let c = class_centroid(candidate.activations, candidate.labels, true, d)?;
            let t = class_centroid(target.activations, target.labels, true, d)?;
            Ok(cosine(&c, &t))
        }
    }
}

// ── Multi-candidate ranking ────────────────────────────────────────

/// A ranked candidate entry from [`rank_candidates`].
#[derive(Debug, Clone)]
pub struct RankEntry {
    /// Index into the input `candidates` slice.
    pub candidate_idx: usize,
    /// Mean percentile rank across all metrics (`0.0` = worst, `1.0` = best).
    pub mean_percentile: f32,
    /// Raw per-metric scores (same order as the input `metrics` slice).
    pub per_metric_scores: Vec<f32>,
}

/// Rank candidates by aggregate transfer score across multiple metrics.
///
/// For each metric, candidates are assigned a percentile rank (the fraction of
/// other candidates they outscore). The mean percentile rank across all metrics
/// is the aggregate score (arXiv:2607.04222 §4 protocol). The returned vec is
/// sorted by `mean_percentile` descending (best predicted transfer first).
///
/// This is the modelless "which experience teaches the most" scorer. At least
/// one candidate and one metric are required; otherwise returns
/// [`MagError::Empty`]. With a single candidate the percentile is trivially
/// `1.0`; the score is still useful as a per-metric diagnostic.
pub fn rank_candidates<S: AsRef<[f32]>>(
    candidates: &[DataSet<'_, S>],
    target: &DataSet<'_, S>,
    metrics: &[TransferMetric],
) -> Result<Vec<RankEntry>, MagError> {
    if candidates.is_empty() || metrics.is_empty() {
        return Err(MagError::Empty);
    }

    let n_cand = candidates.len();
    let n_metrics = metrics.len();

    // Compute raw scores: scores[i * n_metrics + m].
    let mut scores = Vec::with_capacity(n_cand * n_metrics);
    for cand in candidates {
        for &metric in metrics {
            scores.push(transfer_score(cand, target, metric)?);
        }
    }

    // Convert to percentile ranks per metric (fraction of candidates beaten).
    let mut percentiles = vec![0.0_f32; n_cand * n_metrics];
    for m in 0..n_metrics {
        let mut ranked: Vec<(f32, usize)> =
            (0..n_cand).map(|i| (scores[i * n_metrics + m], i)).collect();
        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let denom = (n_cand - 1).max(1) as f32;
        for (rank, &(_, idx)) in ranked.iter().enumerate() {
            percentiles[idx * n_metrics + m] = rank as f32 / denom;
        }
    }

    // Mean percentile per candidate.
    let mut entries: Vec<RankEntry> = (0..n_cand)
        .map(|i| {
            let mut sum = 0.0;
            let mut per_metric = Vec::with_capacity(n_metrics);
            for m in 0..n_metrics {
                sum += percentiles[i * n_metrics + m];
                per_metric.push(scores[i * n_metrics + m]);
            }
            RankEntry {
                candidate_idx: i,
                mean_percentile: sum / n_metrics as f32,
                per_metric_scores: per_metric,
            }
        })
        .collect();

    entries.sort_by(|a, b| {
        b.mean_percentile
            .partial_cmp(&a.mean_percentile)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(entries)
}

// ── Internal metric helpers ────────────────────────────────────────

/// Compute the centroid (per-dimension mean) of a sample set. Allocates `d` f32s.
fn centroid<S: AsRef<[f32]>>(samples: &[S], d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; d];
    let inv_n = 1.0 / samples.len() as f32;
    for s in samples {
        let s = s.as_ref();
        for (acc, &v) in out.iter_mut().zip(s) {
            *acc += v * inv_n;
        }
    }
    out
}

/// Compute the centroid of samples whose label matches `target_class`.
fn class_centroid<S: AsRef<[f32]>>(
    samples: &[S],
    labels: &[bool],
    target_class: bool,
    d: usize,
) -> Result<Vec<f32>, MagError> {
    let mut out = vec![0.0_f32; d];
    let mut count = 0;
    for (s, &label) in samples.iter().zip(labels) {
        if label == target_class {
            let s = s.as_ref();
            for (acc, &v) in out.iter_mut().zip(s) {
                *acc += v;
            }
            count += 1;
        }
    }
    if count == 0 {
        return Err(MagError::EmptyClass);
    }
    let inv_n = 1.0 / count as f32;
    for x in out.iter_mut() {
        *x *= inv_n;
    }
    Ok(out)
}

/// Pearson correlation between two equal-length vectors.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for j in 0..a.len() {
        let da = a[j] - mean_a;
        let db = b[j] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        cov / denom
    }
}

/// RBF-kernel MMD² between two sample sets (biased estimator).
///
/// `mmd² = (1/m²)ΣΣ k(x_i,x_j) + (1/n²)ΣΣ k(y_i,y_j) − (2/(mn))ΣΣ k(x_i,y_j)`
fn rbf_mmd_sq<S: AsRef<[f32]>>(x: &[S], y: &[S], gamma: f32) -> f32 {
    let m = x.len();
    let n = y.len();

    let mut sum_xx = 0.0;
    for i in 0..m {
        let xi = x[i].as_ref();
        for j in 0..m {
            sum_xx += rbf_kernel(xi, x[j].as_ref(), gamma);
        }
    }
    let mut sum_yy = 0.0;
    for i in 0..n {
        let yi = y[i].as_ref();
        for j in 0..n {
            sum_yy += rbf_kernel(yi, y[j].as_ref(), gamma);
        }
    }
    let mut sum_xy = 0.0;
    for i in 0..m {
        let xi = x[i].as_ref();
        for j in 0..n {
            sum_xy += rbf_kernel(xi, y[j].as_ref(), gamma);
        }
    }

    let mmd = sum_xx / (m * m) as f32
        + sum_yy / (n * n) as f32
        - 2.0 * sum_xy / (m * n) as f32;
    mmd.max(0.0) // MMD² is non-negative by construction (guard against f32 drift)
}

#[inline]
fn rbf_kernel(a: &[f32], b: &[f32], gamma: f32) -> f32 {
    let mut dist_sq = 0.0;
    for j in 0..a.len() {
        let diff = a[j] - b[j];
        dist_sq += diff * diff;
    }
    (-gamma * dist_sq).exp()
}

/// 1D Wasserstein-1 distance averaged over dimensions.
///
/// For each dimension, both sets are sorted and their empirical quantile
/// functions are compared on a common grid of `max(m, n)` points (handles
/// unequal sample counts via linear interpolation).
fn wasserstein1d<S: AsRef<[f32]>>(x: &[S], y: &[S], d: usize) -> f32 {
    let t = x.len().max(y.len());
    let mut total = 0.0;
    for j in 0..d {
        let mut col_x: Vec<f32> = x.iter().map(|s| s.as_ref()[j]).collect();
        let mut col_y: Vec<f32> = y.iter().map(|s| s.as_ref()[j]).collect();
        col_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        col_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut dim_dist = 0.0;
        for k in 0..t {
            let qx = quantile_interp(&col_x, (k as f32 + 0.5) / t as f32);
            let qy = quantile_interp(&col_y, (k as f32 + 0.5) / t as f32);
            dim_dist += (qx - qy).abs();
        }
        total += dim_dist / t as f32;
    }
    total / d as f32
}

/// Linear-interpolated quantile from a sorted slice at fraction `f ∈ [0, 1]`.
#[inline]
fn quantile_interp(sorted: &[f32], f: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = f.clamp(0.0, 0.9999) * sorted.len() as f32;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Linear CKA in feature space: `trace(Cx·Cy) / (‖Cx‖_F · ‖Cy‖_F)`.
///
/// Uses d×d feature-space Gram matrices (`Cx = XᵀX/m`, `Cy = YᵀY/n`) so
/// candidate and target may have different sample counts (sample-space CKA
/// requires equal `n`).
fn cka_linear<S: AsRef<[f32]>>(x: &[S], y: &[S], d: usize) -> f32 {
    let cx = feature_gram(x, d);
    let cy = feature_gram(y, d);

    // trace(Cx·Cy) = Σ_i Σ_j Cx[i,j]·Cy[j,i]; both symmetric so Cy[j,i]=Cy[i,j].
    let mut trace_val = 0.0;
    let mut norm_cx = 0.0;
    let mut norm_cy = 0.0;
    for i in 0..d {
        for j in 0..d {
            trace_val += cx[i * d + j] * cy[j * d + i];
            norm_cx += cx[i * d + j] * cx[i * d + j];
            norm_cy += cy[i * d + j] * cy[i * d + j];
        }
    }
    let denom = (norm_cx * norm_cy).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        trace_val / denom
    }
}

/// Compute the d×d feature Gram matrix `XᵀX/n` (flattened row-major).
fn feature_gram<S: AsRef<[f32]>>(samples: &[S], d: usize) -> Vec<f32> {
    let n = samples.len();
    let mut gram = vec![0.0_f32; d * d];
    for s in samples {
        let s = s.as_ref();
        for i in 0..d {
            for j in 0..d {
                gram[i * d + j] += s[i] * s[j];
            }
        }
    }
    let inv_n = 1.0 / n as f32;
    for g in gram.iter_mut() {
        *g *= inv_n;
    }
    gram
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn centroid_cosine_identical_is_one() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let labels = [true, false, true];
        let ds = DataSet::new(&acts, &labels);
        let score = transfer_score(&ds, &ds, TransferMetric::CentroidCosine).unwrap();
        assert!(approx_eq(score, 1.0, 1e-5), "got {score}");
    }

    #[test]
    fn euclidean_identical_is_zero() {
        let acts: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let labels = [true, false];
        let ds = DataSet::new(&acts, &labels);
        let score = transfer_score(&ds, &ds, TransferMetric::Euclidean).unwrap();
        assert!(approx_eq(score, 0.0, 1e-4), "got {score}");
    }

    #[test]
    fn class_conditional_separates_by_label() {
        // candidate: class-false centroid at [1,0], class-true at [10,0]
        let cand_acts: Vec<[f32; 2]> = vec![[1.0, 0.1], [10.0, 0.0], [1.0, -0.1], [10.0, 0.1]];
        let cand_labels = [false, true, false, true];
        let cand = DataSet::new(&cand_acts, &cand_labels);

        // target: same structure → class-conditional cosine should be ~1.0
        let target = DataSet::new(&cand_acts, &cand_labels);
        let score_mal = transfer_score(&cand, &target, TransferMetric::ClassConditionalCosineMalicious).unwrap();
        let score_ben = transfer_score(&cand, &target, TransferMetric::ClassConditionalCosineBenign).unwrap();
        assert!(approx_eq(score_mal, 1.0, 1e-5), "malicious cos got {score_mal}");
        assert!(approx_eq(score_ben, 1.0, 1e-5), "benign cos got {score_ben}");
    }

    #[test]
    fn class_conditional_empty_class_errors() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [2.0, 0.0]];
        let labels_all_true = [true, true];
        let ds = DataSet::new(&acts, &labels_all_true);
        // Requesting malicious class (false) but no false labels → EmptyClass
        assert_eq!(
            transfer_score(&ds, &ds, TransferMetric::ClassConditionalCosineMalicious),
            Err(MagError::EmptyClass)
        );
    }

    #[test]
    fn cka_identical_is_one() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = [true, false, true];
        let ds = DataSet::new(&acts, &labels);
        let score = transfer_score(&ds, &ds, TransferMetric::CkaLinear).unwrap();
        assert!(approx_eq(score, 1.0, 1e-4), "got {score}");
    }

    #[test]
    fn mmd_identical_is_zero() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let labels = [true, false, true];
        let ds = DataSet::new(&acts, &labels);
        let score = transfer_score(&ds, &ds, TransferMetric::RbfMmd).unwrap();
        assert!(approx_eq(score, 0.0, 1e-4), "got {score}");
    }

    #[test]
    fn wasserstein_identical_is_zero() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = [true, false, true];
        let ds = DataSet::new(&acts, &labels);
        let score = transfer_score(&ds, &ds, TransferMetric::Wasserstein1d).unwrap();
        assert!(approx_eq(score, 0.0, 1e-4), "got {score}");
    }

    #[test]
    fn rank_candidates_returns_sorted_desc() {
        // 3 candidates, 1 metric. Candidate 1 is identical to target (best),
        // candidate 2 is orthogonal, candidate 3 is anti-correlated.
        let target_acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [1.0, 0.1]];
        let target_labels = [true, false];
        let target = DataSet::new(&target_acts, &target_labels);

        let c0_acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [1.0, 0.1]]; // identical
        let c1_acts: Vec<[f32; 2]> = vec![[0.0, 1.0], [0.1, 1.0]]; // orthogonal
        let c2_acts: Vec<[f32; 2]> = vec![[-1.0, 0.0], [-1.0, -0.1]]; // anti
        let labels = [true, false];
        let candidates = vec![
            DataSet::new(&c0_acts, &labels),
            DataSet::new(&c1_acts, &labels),
            DataSet::new(&c2_acts, &labels),
        ];

        let ranked = rank_candidates(
            &candidates,
            &target,
            &[TransferMetric::CentroidCosine],
        )
        .unwrap();

        assert_eq!(ranked.len(), 3);
        // Best (identical) should be first.
        assert_eq!(ranked[0].candidate_idx, 0);
        assert!(approx_eq(ranked[0].mean_percentile, 1.0, 1e-5));
        // Descending order.
        assert!(ranked[0].mean_percentile >= ranked[1].mean_percentile);
        assert!(ranked[1].mean_percentile >= ranked[2].mean_percentile);
    }

    #[test]
    fn rank_candidates_multi_metric_aggregates() {
        let target_acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0]];
        let target_labels = [true, false];
        let target = DataSet::new(&target_acts, &target_labels);

        let c0_acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0]]; // identical to target
        let c1_acts: Vec<[f32; 2]> = vec![[5.0, 5.0], [5.0, 5.0]]; // different
        let labels = [true, false];
        let candidates = vec![
            DataSet::new(&c0_acts, &labels),
            DataSet::new(&c1_acts, &labels),
        ];

        let ranked = rank_candidates(
            &candidates,
            &target,
            &[TransferMetric::CentroidCosine, TransferMetric::Euclidean],
        )
        .unwrap();

        // Identical candidate should win on both metrics.
        assert_eq!(ranked[0].candidate_idx, 0);
        assert_eq!(ranked[0].per_metric_scores.len(), 2);
    }

    #[test]
    fn transfer_score_label_length_mismatch_errors() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0]];
        let labels_ok = [true, false];
        let labels_bad = [true]; // wrong length
        let a = DataSet::new(&acts, &labels_ok);
        let b = DataSet::new(&acts, &labels_bad);
        assert_eq!(
            transfer_score(&a, &b, TransferMetric::CentroidCosine),
            Err(MagError::DimMismatch)
        );
    }
}
