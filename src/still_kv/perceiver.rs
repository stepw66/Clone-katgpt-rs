//! StillPerceiver: Cross-attention based KV cache compaction.
//!
//! A lightweight Perceiver-style architecture that uses learned latent queries
//! to cross-attend into the full KV cache, producing a compact representation.

/// Configuration for the StillPerceiver compactor.
#[derive(Debug, Clone)]
pub struct StillPerceiverConfig {
    /// Latent dimension for the perceiver bottleneck.
    pub latent_dim: usize,
    /// Target compact sequence length (number of latent tokens).
    pub compact_length: usize,
    /// Number of perceiver blocks (cross-attn + self-attn pairs).
    pub num_blocks: usize,
    /// Number of heads for cross-attention.
    pub cross_attn_heads: usize,
    /// Number of heads for self-attention within latents.
    pub self_attn_heads: usize,
}

impl Default for StillPerceiverConfig {
    fn default() -> Self {
        Self {
            latent_dim: 64,
            compact_length: 128,
            num_blocks: 2,
            cross_attn_heads: 4,
            self_attn_heads: 4,
        }
    }
}

impl StillPerceiverConfig {
    /// Create a new config with the given latent dimension and compact length.
    pub fn new(latent_dim: usize, compact_length: usize) -> Self {
        Self {
            latent_dim,
            compact_length,
            ..Default::default()
        }
    }
}

/// Perceiver-based KV cache compactor.
///
/// Uses cross-attention from learned latent queries into the KV cache
/// to produce a compact representation. Multiple blocks of
/// cross-attention + self-attention refine the compact output.
#[derive(Debug, Clone)]
pub struct StillPerceiver {
    /// Configuration.
    pub config: StillPerceiverConfig,
}

impl StillPerceiver {
    /// Create a new perceiver with the given configuration.
    pub fn new(config: StillPerceiverConfig) -> Self {
        Self { config }
    }

    /// Cross-attend latent queries into the KV cache.
    ///
    /// # Arguments
    /// * `latent_queries` - Shape `[compact_length, latent_dim]`
    /// * `kv_cache` - Flat f32 buffer, shape `[seq_len * num_heads * head_dim]`
    ///
    /// # Returns
    /// Updated latent queries after cross-attention, shape `[compact_length, latent_dim]`.
    pub fn cross_attention(&self, latent_queries: &[f32], _kv_cache: &[f32]) -> Vec<f32> {
        // TODO: Implement cross-attention.
        // Q = latent_queries  [compact_length, latent_dim]
        // K,V = kv_cache      [seq_len, kv_dim]
        // output = softmax(Q @ K^T / sqrt(d)) @ V
        latent_queries.to_vec()
    }

    /// Self-attention among latent tokens to refine representations.
    ///
    /// # Arguments
    /// * `latents` - Shape `[compact_length, latent_dim]`
    ///
    /// # Returns
    /// Refined latents after self-attention.
    pub fn self_attention(&self, latents: &[f32]) -> Vec<f32> {
        // TODO: Implement self-attention among latent tokens.
        // Q = K = V = latents
        // output = softmax(Q @ K^T / sqrt(d)) @ V
        latents.to_vec()
    }

    /// Run the full perceiver forward pass: num_blocks × (cross-attn + self-attn).
    ///
    /// # Arguments
    /// * `kv_cache` - Flat f32 KV cache buffer
    /// * `query_bank` - Initial latent queries from query bank
    ///
    /// # Returns
    /// Compacted latent representation.
    pub fn forward(&self, kv_cache: &[f32], query_bank: &[f32]) -> Vec<f32> {
        let mut latents = query_bank.to_vec();
        for _ in 0..self.config.num_blocks {
            latents = self.cross_attention(&latents, kv_cache);
            latents = self.self_attention(&latents);
        }
        latents
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = StillPerceiverConfig::default();
        assert_eq!(config.latent_dim, 64);
        assert_eq!(config.compact_length, 128);
        assert_eq!(config.num_blocks, 2);
    }

    #[test]
    fn test_perceiver_new() {
        let config = StillPerceiverConfig::new(32, 64);
        let perceiver = StillPerceiver::new(config);
        assert_eq!(perceiver.config.latent_dim, 32);
        assert_eq!(perceiver.config.compact_length, 64);
    }

    #[test]
    fn test_forward_identity_passthrough() {
        let config = StillPerceiverConfig::new(4, 2);
        let perceiver = StillPerceiver::new(config);
        let kv = vec![1.0f32; 16];
        let queries = vec![0.5f32; 8];
        let result = perceiver.forward(&kv, &queries);
        // With stub (identity), output == queries
        assert_eq!(result, queries);
    }
}
