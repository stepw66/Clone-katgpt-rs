//! Core types for MAG (Mining via Activation Geometry).
//!
//! Decoupled from the mining/transfer algorithms so a frozen [`MagDirection`]
//! artifact can be referenced (loaded, verified, committed) without pulling in
//! the mining or transfer code paths.

use blake3::Hasher;

// ── Math helpers (shared by mining.rs + transfer.rs) ──────────────

/// L2 norm of an f32 slice.
#[inline]
pub(crate) fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize `v` to unit L2 norm in place. Returns the pre-normalization norm
/// (or `0.0` if the vector is zero, leaving it unchanged).
#[inline]
pub(crate) fn normalize_in_place(v: &mut [f32]) -> f32 {
    let n = norm(v);
    if n > 0.0 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    n
}

/// Dot product of two equal-length f32 slices.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Cosine similarity with a zero-norm guard (returns `0.0` if either norm is zero).
#[inline]
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

// ── Direction ──────────────────────────────────────────────────────

/// A mined feature direction vector with diagnostics + BLAKE3 commitment.
///
/// Distilled from MAG (arXiv:2607.04222). `direction` is unit-norm.
/// `recon_error` (ϵ_Q) and `cosine` are populated by
/// [`reconstruction_error`](super::mining::reconstruction_error) and are `NaN`
/// immediately after mining (they require the original paired data + an alpha).
///
/// `blake3` is `BLAKE3(direction_le)` — little-endian `f32` per element —
/// mirroring the `LatentSteeringVector` commitment pattern (Plan 309) and the
/// `MerkleFrozenEnvelope` pattern in riir-neuron-db. This makes a mined
/// direction a freeze/thaw-versionable, tamper-evident artifact that the
/// injection side (`apply_latent_steering`) can consume directly.
#[derive(Debug, Clone)]
pub struct MagDirection {
    /// Unit-norm direction `v_Q` (mean-shift) or `u_Q` (contrast) in ℝ^d.
    /// Stored as `Box<[f32]>` (no spare capacity — a frozen artifact, not a
    /// growable buffer).
    pub direction: Box<[f32]>,
    /// Reconstruction error ϵ_Q. ≈0 ⇒ linear shift (single direction suffices,
    /// steerable); ≈1 ⇒ no shift on average; >1 ⇒ overshoot. `NaN` until
    /// [`reconstruction_error`] populates it.
    ///
    /// [`reconstruction_error`]: super::mining::reconstruction_error
    pub recon_error: f32,
    /// Mean cosine of the predicted shift (`α·direction`) vs the actual
    /// per-sample shift. High (→1.0) ⇒ a single direction aligns with
    /// individual shifts, not just their mean. `NaN` until
    /// [`reconstruction_error`] populates it.
    ///
    /// [`reconstruction_error`]: super::mining::reconstruction_error
    pub cosine: f32,
    /// `BLAKE3(direction_le)` — content-addressed commitment.
    pub blake3: [u8; 32],
}

impl MagDirection {
    /// Dimensionality `d` of the direction vector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.direction.len()
    }

    /// The unit-norm direction as a borrowed slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.direction
    }

    /// Stamp the linearity diagnostic fields onto a mined direction. Builder
    /// style — typically called after [`reconstruction_error`] returns the values.
    ///
    /// [`reconstruction_error`]: super::mining::reconstruction_error
    #[inline]
    pub fn with_diagnostics(mut self, recon_error: f32, cosine: f32) -> Self {
        self.recon_error = recon_error;
        self.cosine = cosine;
        self
    }
}

// ── Operators ──────────────────────────────────────────────────────

/// The 8 MAG readout operators (arXiv:2607.04222 §2.2).
///
/// Each defines a different vector summary of the activation readout, isolating
/// a different aspect of the prefix-induced shift. Applied via
/// [`apply_operator`](super::mining::apply_operator) or the zero-alloc
/// [`apply_operator_into`](super::mining::apply_operator_into).
///
/// The paper found operators `Interaction` (Y6) and `Verdict` (Y7) are
/// near-zero on average; `Prefixed` / `InputDelta` / `Answered` / `FewShot`
/// carry the signal. The primitive ships all 8 for completeness; the GOAT gate
/// (Plan 418 G2) focuses on the load-bearing operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MagOperator {
    /// `m(p)` — plain input activation (baseline).
    Direct = 0,
    /// `m(Q‖p)` — prefixed activation.
    Prefixed = 1,
    /// `m(Q‖p, y_M)` — answered activation (prefix + model verdict).
    Answered = 2,
    /// `m(Q‖p) − m(p)` — input-conditioned prefix delta.
    InputDelta = 3,
    /// `m(Q‖p) − m(Q)` — question-conditioned delta.
    QuestionDelta = 4,
    /// `m(Q‖p, y_M) − m(Q‖p) − m(p, y_M) + m(p)` — Q×y interaction contrast.
    Interaction = 5,
    /// `m(p, y_M) − m(p)` — verdict-only delta.
    Verdict = 6,
    /// `m(E‖Q‖p) − m(p)` — few-shot examples + prefix delta.
    FewShot = 7,
}

// ── Transfer metrics ───────────────────────────────────────────────

/// Geometric metric for transfer prediction (arXiv:2607.04222 §4).
///
/// All variants return **higher = better predicted transfer** via
/// [`transfer_score`](super::transfer::transfer_score) — distance-based metrics
/// are negated so that `0` = identical and negative = dissimilar. This
/// convention lets all metrics aggregate uniformly in
/// [`rank_candidates`](super::transfer::rank_candidates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TransferMetric {
    /// Cosine between overall centroids. The paper's near-uninformative baseline
    /// (raw cosine ρ ≈ 0.03 on their 18-dataset corpus).
    CentroidCosine = 0,
    /// Negative Euclidean distance between centroids (`0` = identical centroids).
    Euclidean = 1,
    /// Pearson correlation between the centroid (mean-activation) vectors.
    Correlation = 2,
    /// Negative RBF-kernel MMD² (`0` = identical distributions). γ = 1/d.
    RbfMmd = 3,
    /// Negative 1D Wasserstein distance averaged over dimensions (`0` = identical
    /// per-dimension marginals).
    Wasserstein1d = 4,
    /// Linear CKA in feature space (d×d Gram). `1` = identical second-moment
    /// structure. Uses feature-space (not sample-space) CKA so candidate and
    /// target may have different sample counts.
    CkaLinear = 5,
    /// Cosine between negative-class centroids (the "malicious"/refusal class,
    /// `label == false`). The paper's most informative single metric.
    ClassConditionalCosineMalicious = 6,
    /// Cosine between positive-class centroids (the "benign"/comply class,
    /// `label == true`).
    ClassConditionalCosineBenign = 7,
}

// ── Errors ─────────────────────────────────────────────────────────

/// Errors returned by MAG operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagError {
    /// Input slices have mismatched per-sample dimensionality.
    DimMismatch,
    /// A sample set is empty (zero samples) or has zero-dimensionality samples.
    Empty,
    /// A computed direction has zero norm (cannot normalize).
    ZeroNorm,
    /// A class is empty in one of the sets (class-conditional metric requested
    /// but no samples match the requested class).
    EmptyClass,
}

impl std::fmt::Display for MagError {
    #[cold]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MagError::DimMismatch => write!(f, "MAG: sample dimensionality mismatch"),
            MagError::Empty => write!(f, "MAG: empty sample set"),
            MagError::ZeroNorm => write!(f, "MAG: zero-norm direction (cannot normalize)"),
            MagError::EmptyClass => write!(f, "MAG: empty class in class-conditional metric"),
        }
    }
}

impl std::error::Error for MagError {}

// ── BLAKE3 commitment ──────────────────────────────────────────────

/// Compute `BLAKE3(direction_le)` — little-endian `f32` per element.
///
/// Mirrors `latent_steering::compute_commitment` (Plan 309) and the
/// `MerkleFrozenEnvelope` pattern in riir-neuron-db. Used by the mining
/// functions to stamp a content-addressed commitment onto a mined direction.
pub(crate) fn compute_direction_commitment(direction: &[f32]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    for &f in direction {
        hasher.update(&f.to_le_bytes());
    }
    let mut out = [0u8; 32];
    hasher.finalize_xof().fill(&mut out);
    out
}

// ── Validation helpers ─────────────────────────────────────────────

/// Check that a sample set is non-empty and all samples share the same
/// per-sample dimensionality. Returns the dimensionality `d`.
pub(crate) fn check_dim<S: AsRef<[f32]>>(samples: &[S]) -> Result<usize, MagError> {
    if samples.is_empty() {
        return Err(MagError::Empty);
    }
    let d = samples[0].as_ref().len();
    if d == 0 {
        return Err(MagError::Empty);
    }
    for s in samples {
        if s.as_ref().len() != d {
            return Err(MagError::DimMismatch);
        }
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_and_normalize_roundtrip() {
        let mut v = vec![3.0, 4.0];
        let pre = normalize_in_place(&mut v);
        assert!((pre - 5.0).abs() < 1e-5);
        assert!((norm(&v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normalize_zero_is_noop() {
        let mut v = vec![0.0, 0.0, 0.0];
        let pre = normalize_in_place(&mut v);
        assert_eq!(pre, 0.0);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_norm_guard() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn check_dim_rejects_inconsistent() {
        let good: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(check_dim(&good).unwrap(), 2);
        let bad: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0]];
        assert_eq!(check_dim(&bad), Err(MagError::DimMismatch));
        let empty: Vec<Vec<f32>> = vec![];
        assert_eq!(check_dim(&empty), Err(MagError::Empty));
    }

    #[test]
    fn commitment_is_deterministic() {
        let a = compute_direction_commitment(&[1.0, 2.0, 3.0]);
        let b = compute_direction_commitment(&[1.0, 2.0, 3.0]);
        assert_eq!(a, b);
        let c = compute_direction_commitment(&[1.0, 2.0, 3.001]);
        assert_ne!(a, c);
    }

    #[test]
    fn mag_direction_with_diagnostics() {
        let d = MagDirection {
            direction: vec![1.0, 0.0].into_boxed_slice(),
            recon_error: f32::NAN,
            cosine: f32::NAN,
            blake3: [0u8; 32],
        };
        let d = d.with_diagnostics(0.1, 0.95);
        assert!((d.recon_error - 0.1).abs() < 1e-6);
        assert!((d.cosine - 0.95).abs() < 1e-6);
        assert_eq!(d.dim(), 2);
    }

    #[test]
    fn operator_repr_u8() {
        assert_eq!(MagOperator::Direct as u8, 0);
        assert_eq!(MagOperator::FewShot as u8, 7);
        assert_eq!(TransferMetric::CentroidCosine as u8, 0);
        assert_eq!(TransferMetric::ClassConditionalCosineBenign as u8, 7);
    }
}
