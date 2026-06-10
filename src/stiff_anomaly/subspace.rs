//! Stiff/soft subspace decomposition.
//!
//! Partitions eigenvalue spectrum into "stiff" (high-energy, top-k) and
//! "soft" (low-energy, remaining d-k) subspaces via trace-mass thresholding.
//! The soft alignment ratio measures how much a state change projects onto
//! soft axes — α ≈ 1 implies elastic absorption, α ≈ 0 implies stiff
//! collision (potential anomaly).

/// Result of stiff/soft subspace decomposition on eigenvalues.
#[derive(Debug, Clone, Default)]
pub struct StiffSoftDecomposition {
    /// Top-k eigenvalues by magnitude (stiff subspace).
    pub stiff_eigenvalues: Vec<f32>,
    /// Remaining d-k eigenvalues (soft subspace).
    pub soft_eigenvalues: Vec<f32>,
    /// Stiff eigenvectors: k × d matrix.
    pub stiff_eigenvectors: Vec<Vec<f32>>,
    /// Soft eigenvectors: (d-k) × d matrix.
    pub soft_eigenvectors: Vec<Vec<f32>>,
    /// All eigenvalues sorted descending.
    pub eigenvalues: Vec<f32>,
    /// Stiff dimension count.
    pub k: usize,
}

impl StiffSoftDecomposition {
    /// Pre-allocate a reusable decomposition buffer for `dim`-sized data.
    ///
    /// Use with [`decompose_into`] to avoid per-call allocations.
    pub fn with_capacity(dim: usize) -> Self {
        Self {
            stiff_eigenvalues: Vec::with_capacity(dim),
            soft_eigenvalues: Vec::with_capacity(dim),
            stiff_eigenvectors: Vec::with_capacity(dim),
            soft_eigenvectors: Vec::with_capacity(dim),
            eigenvalues: Vec::with_capacity(dim),
            k: 0,
        }
    }

    /// Clear all internal buffers, retaining capacity for reuse.
    pub fn clear(&mut self) {
        self.stiff_eigenvalues.clear();
        self.soft_eigenvalues.clear();
        self.stiff_eigenvectors.clear();
        self.soft_eigenvectors.clear();
        self.eigenvalues.clear();
        self.k = 0;
    }
}

/// Find invariant k at given trace mass threshold (e.g., 0.90).
///
/// Returns the smallest k such that the cumulative sum of the top-k
/// eigenvalues accounts for at least `trace_mass` fraction of total.
pub fn stiff_subspace_k(eigenvalues: &[f32], trace_mass: f32) -> usize {
    let total: f64 = eigenvalues.iter().map(|&x| x as f64).sum();
    if total < 1e-12 {
        return 0;
    }
    let mut cumsum = 0.0f64;
    for (i, &ev) in eigenvalues.iter().enumerate() {
        cumsum += ev as f64;
        if cumsum / total >= trace_mass as f64 {
            return i + 1;
        }
    }
    eigenvalues.len()
}

/// Perform stiff/soft decomposition from eigenvalues and eigenvectors.
///
/// `eigenvalues` must be sorted descending. `eigenvectors` is a d × d matrix
/// where row i is the eigenvector corresponding to `eigenvalues[i]`.
/// `trace_mass` is the fraction of total variance captured by the stiff
/// subspace (e.g., 0.90).
///
/// Allocates a new [`StiffSoftDecomposition`]. For zero-alloc hot paths, use
/// [`decompose_into`] instead.
pub fn decompose(
    eigenvalues: Vec<f32>,
    eigenvectors: Vec<Vec<f32>>,
    trace_mass: f32,
) -> StiffSoftDecomposition {
    let k = stiff_subspace_k(&eigenvalues, trace_mass);
    let (stiff_eigenvalues, soft_eigenvalues) = eigenvalues.split_at(k);
    let (stiff_eigenvectors, soft_eigenvectors) = eigenvectors.split_at(k);
    StiffSoftDecomposition {
        stiff_eigenvalues: stiff_eigenvalues.to_vec(),
        soft_eigenvalues: soft_eigenvalues.to_vec(),
        stiff_eigenvectors: stiff_eigenvectors.to_vec(),
        soft_eigenvectors: soft_eigenvectors.to_vec(),
        eigenvalues,
        k,
    }
}

/// Zero-allocation variant of [`decompose`] that writes into a pre-allocated buffer.
///
/// Populates `buf` in-place by copying slices from the borrowed inputs.
/// Clears `buf` first, then extends from the input references.
pub fn decompose_into(
    eigenvalues: &[f32],
    eigenvectors: &[Vec<f32>],
    trace_mass: f32,
    buf: &mut StiffSoftDecomposition,
) {
    buf.clear();
    let k = stiff_subspace_k(eigenvalues, trace_mass);
    buf.stiff_eigenvalues.extend_from_slice(&eigenvalues[..k]);
    buf.soft_eigenvalues.extend_from_slice(&eigenvalues[k..]);
    buf.stiff_eigenvectors
        .extend(eigenvectors[..k].iter().cloned());
    buf.soft_eigenvectors
        .extend(eigenvectors[k..].iter().cloned());
    buf.eigenvalues.extend_from_slice(eigenvalues);
    buf.k = k;
}

/// Compute soft alignment ratio: how much of `delta_x` projects onto soft axes.
///
/// α = ‖P_soft · δx‖² / ‖δx‖²
///
/// - α ≈ 1 → elastic (benign): change is absorbed by soft directions.
/// - α ≈ 0 → stiff collision (anomaly): change resisted by stiff subspace.
///
/// Returns 0.0 if `delta_x` is zero-length or dimension mismatch.
pub fn soft_alignment_ratio(decomp: &StiffSoftDecomposition, delta_x: &[f32]) -> f32 {
    if delta_x.is_empty() || decomp.soft_eigenvectors.is_empty() {
        return 0.0;
    }
    let d = delta_x.len();
    // Check dimension compatibility
    if decomp.soft_eigenvectors[0].len() != d {
        return 0.0;
    }

    // ‖δx‖²
    let dx_sq: f64 = delta_x.iter().map(|&x| (x as f64) * (x as f64)).sum();
    if dx_sq < 1e-30 {
        return 0.0;
    }

    // ‖P_soft · δx‖² = Σ_j (v_j · δx)² over soft eigenvectors
    let mut soft_proj_sq = 0.0f64;
    for v in &decomp.soft_eigenvectors {
        let dot: f64 = v
            .iter()
            .zip(delta_x.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        soft_proj_sq += dot * dot;
    }

    (soft_proj_sq / dx_sq) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: identity eigenvectors for dimension d.
    fn identity_eigenvectors(d: usize) -> Vec<Vec<f32>> {
        (0..d)
            .map(|i| {
                let mut v = vec![0.0f32; d];
                v[i] = 1.0;
                v
            })
            .collect()
    }

    /// G1: Known rotation matrix → k matches rank at 90% trace mass;
    ///     isotropic → k = d; rank-3 → k = 3.
    #[test]
    fn test_g1_stiff_subspace_k() {
        // Rank-3 in 5D: first 3 eigenvalues dominate
        let rank3 = vec![10.0, 8.0, 6.0, 0.001, 0.001];
        let k = stiff_subspace_k(&rank3, 0.90);
        assert!(k <= 3, "rank-3 spectrum: k should be ≤ 3, got {k}");
        assert!(k >= 2, "rank-3 spectrum: k should capture 90%, got {k}");

        // Isotropic (all equal): k = d
        let iso = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let k_iso = stiff_subspace_k(&iso, 0.90);
        assert_eq!(k_iso, 5, "isotropic spectrum: k should be d=5");

        // Exact rank-3: 3 large + 2 zero
        let exact = vec![5.0, 4.0, 3.0, 0.0, 0.0];
        let k_exact = stiff_subspace_k(&exact, 0.90);
        assert_eq!(k_exact, 3, "exact rank-3: k should be 3");
    }

    /// G1 extended: decompose on rank-3 data.
    #[test]
    fn test_g1_decompose_rank3() {
        let eigenvalues = vec![5.0, 4.0, 3.0, 0.0, 0.0];
        let eigenvectors = identity_eigenvectors(5);
        let decomp = decompose(eigenvalues, eigenvectors, 0.90);
        assert_eq!(decomp.k, 3);
        assert_eq!(decomp.stiff_eigenvalues.len(), 3);
        assert_eq!(decomp.soft_eigenvalues.len(), 2);
        assert_eq!(decomp.stiff_eigenvectors.len(), 3);
        assert_eq!(decomp.soft_eigenvectors.len(), 2);
    }

    /// Soft alignment ratio: known projection.
    #[test]
    fn test_soft_alignment_ratio_pure_soft() {
        // 3D: stiff gets eigenvalue [3.0], soft gets [1.0, 1.0]
        let eigenvalues = vec![3.0, 1.0, 1.0];
        let eigenvectors = identity_eigenvectors(3);
        let decomp = decompose(eigenvalues, eigenvectors, 0.50);
        // delta_x = [0, 1, 0] is entirely in soft subspace (eigenvector 1)
        let alpha = soft_alignment_ratio(&decomp, &[0.0, 1.0, 0.0]);
        assert!(
            (alpha - 1.0).abs() < 0.01,
            "pure soft projection: α should be ~1.0, got {alpha}"
        );
    }

    #[test]
    fn test_soft_alignment_ratio_pure_stiff() {
        let eigenvalues = vec![3.0, 1.0, 1.0];
        let eigenvectors = identity_eigenvectors(3);
        let decomp = decompose(eigenvalues, eigenvectors, 0.50);
        // delta_x = [1, 0, 0] is entirely in stiff subspace (eigenvector 0)
        let alpha = soft_alignment_ratio(&decomp, &[1.0, 0.0, 0.0]);
        assert!(
            alpha.abs() < 0.01,
            "pure stiff projection: α should be ~0.0, got {alpha}"
        );
    }

    #[test]
    fn test_soft_alignment_ratio_zero_delta() {
        let eigenvalues = vec![3.0, 1.0, 1.0];
        let eigenvectors = identity_eigenvectors(3);
        let decomp = decompose(eigenvalues, eigenvectors, 0.50);
        let alpha = soft_alignment_ratio(&decomp, &[0.0, 0.0, 0.0]);
        assert_eq!(alpha, 0.0, "zero delta should return 0.0");
    }

    /// G5: Integration with eigenvalue data → k matches expected effective dimension.
    #[test]
    fn test_g5_effective_dimension() {
        // Simulate 10-dim spectrum with effective dim ~4
        let eigenvalues = vec![50.0, 30.0, 15.0, 5.0, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001];
        let eigenvectors = identity_eigenvectors(10);
        let decomp = decompose(eigenvalues, eigenvectors, 0.99);
        // Top 4 eigenvalues: 50+30+15+5 = 100, total = 100.186, ratio ≈ 0.998
        assert!(
            decomp.k <= 5 && decomp.k >= 3,
            "k should capture effective dimension, got {}",
            decomp.k
        );

        // At 90%: top 2 are 80/100.186 ≈ 0.799, top 3 = 95/100.186 ≈ 0.948
        let decomp90 = decompose(
            vec![50.0, 30.0, 15.0, 5.0, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
            identity_eigenvectors(10),
            0.90,
        );
        assert_eq!(decomp90.k, 3, "90% trace mass should need top 3");
    }
}
