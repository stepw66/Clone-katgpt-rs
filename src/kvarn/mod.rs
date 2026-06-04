//! KVarN — Variance-Normalized KV-Cache Quantization (Research 159).
//!
//! Phase 1 core implementation: Sinkhorn-style iterative dual-scaling variance
//! normalization combined with asymmetric RTN quantization for KV cache compression.
//!
//! Pipeline per tile:
//!   Key [D, group]:  Hadamard → variance normalize → RTN with dual scales (per-channel × per-token)
//!   Value [group, D]: Hadamard → variance normalize → RTN with dual scales (per-token × per-channel)
//!
//! The variance normalization equalizes per-row and per-column standard deviations
//! via iterative Sinkhorn-style log-space scaling, reducing quantization error from
//! heterogenous magnitude distributions.
//!
//! Binary bloat verification:
//!   cargo build --release 2>/dev/null && ls -la target/release/katgpt-rs
//!   cargo build --release --features kvarn 2>/dev/null && ls -la target/release/katgpt-rs
//!   The two binary sizes should be identical when kvarn is off by default.

pub mod eval;
pub mod hadamard;
pub mod kv_cache;
pub mod var_norm;

pub use eval::pseudo_decode_eval;
pub use kv_cache::KVarNKVCache;
pub use var_norm::{VarianceNormScales, variance_normalize};
