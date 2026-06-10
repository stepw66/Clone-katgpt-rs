//! StillKV: Perceiver-based KV cache compaction — modelless (Plan 245).
//!
//! Compacts KV caches via learned cross-attention without model-specific knowledge.
//! Key insight: position-free compaction — un-rotate RoPE, compact in latent space,
//! re-rotate on retrieval.
//!
//! Strategies:
//! - **ClusterCentroids**: k-means-style cluster representatives
//! - **AttentionWeighted**: attention-score-weighted importance sampling
//! - **SpectralProjection**: PCA/SVD low-rank projection
//! - **BfcfRegionBlend**: BFCF region-weighted blending
//! - **MuxSuperposition**: multiplexed superposition encoding

pub mod compact_cache;
pub mod iterative;
pub mod perceiver;
pub mod position_free;
pub mod query_bank;

pub use compact_cache::{CompactKVCache, CompactionMeta, CompactionStrategy};
pub use iterative::{IterativeChunkCompactor, KVChunk};
pub use perceiver::{StillPerceiver, StillPerceiverConfig};
pub use position_free::PositionFreeCompactor;
pub use query_bank::QueryBank;
