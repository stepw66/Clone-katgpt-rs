//! Query bank: generates latent queries for cross-attention compaction.
//!
//! Each compaction strategy produces different initial queries that bias
//! the perceiver toward capturing strategy-relevant information.

use crate::still_kv::compact_cache::CompactionStrategy;

/// Trait for generating latent queries for perceiver cross-attention.
pub trait QueryBank: Send + Sync {
    /// Generate latent queries for the given KV cache and token budget.
    ///
    /// # Arguments
    /// * `kv_cache` - Flat f32 KV cache buffer
    /// * `budget` - Number of compact tokens to produce
    ///
    /// # Returns
    /// Flat f32 buffer of latent queries, shape `[budget * latent_dim]`.
    fn generate_queries(&self, kv_cache: &[f32], budget: usize) -> Vec<f32>;
}

/// Cluster centroid query bank — initializes queries as random centroids.
#[derive(Debug, Clone)]
pub struct ClusterQueryBank {
    /// Latent dimension per query.
    pub latent_dim: usize,
}

impl QueryBank for ClusterQueryBank {
    fn generate_queries(&self, _kv_cache: &[f32], budget: usize) -> Vec<f32> {
        // TODO: Initialize as k-means++ centroids from KV cache.
        vec![0.0f32; budget * self.latent_dim]
    }
}

/// Attention-weighted query bank — places queries at high-attention positions.
#[derive(Debug, Clone)]
pub struct AttentionQueryBank {
    /// Latent dimension per query.
    pub latent_dim: usize,
}

impl QueryBank for AttentionQueryBank {
    fn generate_queries(&self, _kv_cache: &[f32], budget: usize) -> Vec<f32> {
        // TODO: Sample positions proportional to attention weight magnitude.
        vec![0.0f32; budget * self.latent_dim]
    }
}

/// Spectral projection query bank — top eigenvectors as initial queries.
#[derive(Debug, Clone)]
pub struct SpectralQueryBank {
    /// Latent dimension per query.
    pub latent_dim: usize,
}

impl QueryBank for SpectralQueryBank {
    fn generate_queries(&self, _kv_cache: &[f32], budget: usize) -> Vec<f32> {
        // TODO: PCA top-k eigenvectors as query initialization.
        vec![0.0f32; budget * self.latent_dim]
    }
}

/// BFCF region-blend query bank — region-weighted initialization.
#[derive(Debug, Clone)]
pub struct BfcfQueryBank {
    /// Latent dimension per query.
    pub latent_dim: usize,
}

impl QueryBank for BfcfQueryBank {
    fn generate_queries(&self, _kv_cache: &[f32], budget: usize) -> Vec<f32> {
        // TODO: BFCF region-weighted centroid initialization.
        vec![0.0f32; budget * self.latent_dim]
    }
}

/// Mux superposition query bank — multiplexed encoding initialization.
#[derive(Debug, Clone)]
pub struct MuxQueryBank {
    /// Latent dimension per query.
    pub latent_dim: usize,
}

impl QueryBank for MuxQueryBank {
    fn generate_queries(&self, _kv_cache: &[f32], budget: usize) -> Vec<f32> {
        // TODO: Mux superposition random phase initialization.
        vec![0.0f32; budget * self.latent_dim]
    }
}

/// Create a query bank for the given strategy.
pub fn create_query_bank(strategy: CompactionStrategy, latent_dim: usize) -> Box<dyn QueryBank> {
    match strategy {
        CompactionStrategy::ClusterCentroids => Box::new(ClusterQueryBank { latent_dim }),
        CompactionStrategy::AttentionWeighted => Box::new(AttentionQueryBank { latent_dim }),
        CompactionStrategy::SpectralProjection => Box::new(SpectralQueryBank { latent_dim }),
        CompactionStrategy::BfcfRegionBlend => Box::new(BfcfQueryBank { latent_dim }),
        CompactionStrategy::MuxSuperposition => Box::new(MuxQueryBank { latent_dim }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_query_bank_all_strategies() {
        let strategies = [
            CompactionStrategy::ClusterCentroids,
            CompactionStrategy::AttentionWeighted,
            CompactionStrategy::SpectralProjection,
            CompactionStrategy::BfcfRegionBlend,
            CompactionStrategy::MuxSuperposition,
        ];
        for strategy in strategies {
            let bank = create_query_bank(strategy, 32);
            let queries = bank.generate_queries(&[1.0f32; 64], 4);
            assert_eq!(queries.len(), 4 * 32);
        }
    }
}
