//! DenseMesh — Latent Node Network for Modelless Inference (root re-export shim).
//!
//! Substrate modules (adaptive_width, edge_bandit, topology, traits, types,
//! etc.) moved to `katgpt_transformer::dense_mesh` per Proposal 003 Phase 9.
//! This file re-exports them under the historical `katgpt_rs::dense_mesh::*`
//! paths.
//!
//! Plan 385 (2026-07-05): `node_transformer.rs` moved to
//! `katgpt_forward::dense_mesh_node_transformer`. The cycle that kept it in
//! root (it consumes `forward`, which lived in root) is dissolved now that
//! `forward` lives in katgpt-forward too. See
//! `crates/katgpt-forward/src/forward.rs` for the story.
//!
//! See `crates/katgpt-transformer/src/dense_mesh/mod.rs` for the architecture
//! doc and `katgpt-rs/.plans/266_densemesh_latent_node_network.md` Phase 5
//! for the deferred-node history.

pub use katgpt_transformer::dense_mesh::*;

#[cfg(feature = "dense_mesh")]
pub use katgpt_forward::TransformerNode;
