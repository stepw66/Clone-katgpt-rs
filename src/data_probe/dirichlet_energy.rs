//! Dirichlet Energy structural alignment diagnostic.
//!
//! Computes E(E) = Σ_{i,j} A_{ij} ‖h_{e_i} - h_{e_j}‖² over embeddings
//! w.r.t. a sparse adjacency graph. Lower energy means structurally aligned
//! entities (connected by edges have similar embeddings).
//!
//! This is the core measurable from Research 111 — it quantifies whether
//! embeddings are **structurally aligned** across entities/positions,
//! which is a prerequisite for analogical reasoning.

// ── Core computation ──────────────────────────────────────────

/// Compute Dirichlet Energy over embeddings w.r.t. adjacency graph.
///
/// E(E) = Σ_{i,j} A_{ij} ‖h_{e_i} - h_{e_j}‖²
///
/// Lower energy = more structurally aligned (entities connected by edges
/// have similar embeddings).
///
/// # Arguments
/// * `embeddings` — flat slice of embeddings, shape [n_entities × dim]
/// * `dim` — embedding dimension
/// * `adjacency` — sparse adjacency pairs [(i, j), ...] where A_{ij} = 1
///
/// # Returns
/// Total Dirichlet Energy (f32).
///
/// # Panics
/// Panics if any adjacency index is out of bounds for the embedding matrix.
pub fn dirichlet_energy(embeddings: &[f32], dim: usize, adjacency: &[(usize, usize)]) -> f32 {
    let n_entities = embeddings.len() / dim;
    let mut energy = 0.0f32;

    for &(i, j) in adjacency {
        debug_assert!(
            i < n_entities,
            "adjacency index {i} out of bounds ({n_entities} entities)"
        );
        debug_assert!(
            j < n_entities,
            "adjacency index {j} out of bounds ({n_entities} entities)"
        );

        let row_i = i * dim;
        let row_j = j * dim;
        let mut dist_sq = 0.0f32;
        for d in 0..dim {
            let diff = embeddings[row_i + d] - embeddings[row_j + d];
            dist_sq += diff * diff;
        }
        energy += dist_sq;
    }

    energy
}

// ── Adjacency construction helpers ────────────────────────────

/// Build functor adjacency from paired entity indices.
///
/// For N pairs (a_i, b_i), creates edges: (a_0, b_0), (a_1, b_1), ...
/// This is the paper's A_{ij} = 1 iff entities i,j are related by functor.
pub fn functor_adjacency(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    pairs.to_vec()
}

/// Build position-neighbor adjacency from consecutive positions.
///
/// For positions 0..n, creates edges (p, p+1) for each consecutive pair.
/// This is the default structural graph when no domain-specific adjacency
/// is available.
pub fn consecutive_adjacency(n_positions: usize) -> Vec<(usize, usize)> {
    (0..n_positions.saturating_sub(1))
        .map(|p| (p, p + 1))
        .collect()
}

// ── KV cache probe ────────────────────────────────────────────

/// Probe KV cache key embeddings for structural alignment.
///
/// Computes Dirichlet Energy over KV cache keys at a given layer,
/// using position-adjacency (user-specified pairs).
///
/// # Arguments
/// * `keys` — flat slice of key embeddings, shape [n_positions × kv_dim]
/// * `kv_dim` — dimension of each key vector
/// * `adjacency` — sparse adjacency pairs
///
/// # Returns
/// (energy, normalized_energy) where normalized = energy / max(n_edges, 1).
pub fn kv_cache_dirichlet_energy(
    keys: &[f32],
    kv_dim: usize,
    adjacency: &[(usize, usize)],
) -> (f32, f32) {
    let energy = dirichlet_energy(keys, kv_dim, adjacency);
    let n_edges = adjacency.len().max(1) as f32;
    (energy, energy / n_edges)
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a flat embedding matrix from per-entity vectors.
    fn flat_embeddings(entity_vecs: &[Vec<f32>]) -> Vec<f32> {
        entity_vecs.iter().flat_map(|v| v.iter().copied()).collect()
    }

    /// Helper: simple Gaussian-like noise using central limit theorem.
    fn gaussian_noise(rng: &mut fastrand::Rng) -> f32 {
        // Sum of 12 uniform ≈ N(6, 1), shift to N(0, 1).
        let sum: f32 = (0..12).map(|_| rng.f32()).sum();
        sum - 6.0
    }

    // ── G1: Identical embeddings → E < 0.01 ───────────────────

    #[test]
    fn g1_identical_embeddings_near_zero() {
        let dim = 64;
        let n = 5;
        let emb = vec![1.0f32; n * dim];
        // Fully connected adjacency.
        let adjacency: Vec<(usize, usize)> =
            (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).collect();

        let e = dirichlet_energy(&emb, dim, &adjacency);
        assert!(
            e < 0.01,
            "identical embeddings should have E < 0.01, got {e}"
        );
    }

    // ── G2: Random embeddings → E > 1.0 (dim=128, 10 entities) ─

    #[test]
    fn g2_random_embeddings_high_energy() {
        let mut rng = fastrand::Rng::new();
        let dim = 128;
        let n = 10;
        let emb: Vec<f32> = (0..n * dim).map(|_| rng.f32()).collect();

        // Fully connected.
        let adjacency: Vec<(usize, usize)> =
            (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).collect();

        let e = dirichlet_energy(&emb, dim, &adjacency);
        assert!(
            e > 1.0,
            "random embeddings (dim=128, n=10) should have E > 1.0, got {e}"
        );
    }

    // ── G3: Energy increases monotonically with noise ───────────

    #[test]
    fn g3_energy_increases_monotonically_with_noise() {
        let mut rng = fastrand::Rng::new();
        let dim = 64;
        let n = 8;
        // Start from identical embeddings.
        let base: Vec<f32> = vec![0.5f32; n * dim];
        let adjacency: Vec<(usize, usize)> =
            (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).collect();

        let noise_levels = [0.0f32, 0.1, 0.5, 1.0, 2.0];
        let mut energies: Vec<f32> = Vec::new();

        for &sigma in &noise_levels {
            let noisy: Vec<f32> = base
                .iter()
                .map(|&v| v + sigma * gaussian_noise(&mut rng))
                .collect();
            let e = dirichlet_energy(&noisy, dim, &adjacency);
            energies.push(e);
        }

        for window in energies.windows(2) {
            assert!(
                window[1] >= window[0] || (window[1] - window[0]).abs() < 1.0,
                "energy should trend upward with noise: {:?}",
                energies
            );
        }
    }

    // ── G4: Scalar implementation is correct (hand-computed) ───

    #[test]
    fn g4_scalar_correctness_hand_computed() {
        // 3 entities, dim=2.
        // e0 = [1.0, 0.0], e1 = [0.0, 1.0], e2 = [1.0, 1.0]
        let emb = flat_embeddings(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
        let adjacency = vec![(0, 1), (1, 2)];

        // ‖e0-e1‖² = (1-0)² + (0-1)² = 2
        // ‖e1-e2‖² = (0-1)² + (1-1)² = 1
        // E = 2 + 1 = 3
        let e = dirichlet_energy(&emb, 2, &adjacency);
        let expected = 3.0f32;
        assert!(
            (e - expected).abs() < 1e-5,
            "expected E={expected}, got {e}"
        );
    }

    // ── G5: Aligned embeddings < random embeddings ─────────────

    #[test]
    fn g5_aligned_less_than_random() {
        let mut rng = fastrand::Rng::new();
        let dim = 64;
        let n = 10;

        // Aligned: pairs of entities offset by a constant.
        // Category A (entities 0..5): base vector + small noise
        // Category B (entities 5..10): base vector + offset + small noise
        let base: Vec<f32> = (0..dim).map(|_| rng.f32()).collect();
        let offset: Vec<f32> = (0..dim).map(|_| 0.01 * rng.f32()).collect();

        let aligned: Vec<f32> = (0..n)
            .flat_map(|i| {
                let base_vec = if i < 5 { &base } else { &base };
                base_vec
                    .iter()
                    .zip(offset.iter())
                    .map(|(&b, &o)| if i < 5 { b } else { b + o })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Random: completely random embeddings.
        let random: Vec<f32> = (0..n * dim).map(|_| rng.f32()).collect();

        // Functor pairs: match category A to category B.
        let adjacency: Vec<(usize, usize)> = (0..5).map(|i| (i, i + 5)).collect();

        let e_aligned = dirichlet_energy(&aligned, dim, &adjacency);
        let e_random = dirichlet_energy(&random, dim, &adjacency);

        assert!(
            e_aligned < 0.5 * e_random,
            "aligned E ({e_aligned}) should be < 0.5 × random E ({e_random})"
        );
    }

    // ── G6: KV cache probe — random keys baseline ──────────────

    #[test]
    fn g6_kv_cache_random_keys_high_energy() {
        let mut rng = fastrand::Rng::new();
        let kv_dim = 64;
        let n_positions = 20;
        let keys: Vec<f32> = (0..n_positions * kv_dim).map(|_| rng.f32()).collect();
        let adjacency = consecutive_adjacency(n_positions);

        let (energy, normalized) = kv_cache_dirichlet_energy(&keys, kv_dim, &adjacency);
        assert!(
            energy > 1.0,
            "random KV keys should have E > 1.0, got {energy}"
        );
        assert!(
            (normalized - energy / adjacency.len() as f32).abs() < 1e-5,
            "normalized energy mismatch"
        );
    }

    // ── Adjacency helpers ───────────────────────────────────────

    #[test]
    fn test_functor_adjacency_passthrough() {
        let pairs = vec![(0, 5), (1, 6), (2, 7)];
        let adj = functor_adjacency(&pairs);
        assert_eq!(adj, pairs);
    }

    #[test]
    fn test_consecutive_adjacency() {
        let adj = consecutive_adjacency(5);
        assert_eq!(adj, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn test_consecutive_adjacency_empty() {
        let adj = consecutive_adjacency(0);
        assert!(adj.is_empty());
        let adj = consecutive_adjacency(1);
        assert!(adj.is_empty());
    }
}
