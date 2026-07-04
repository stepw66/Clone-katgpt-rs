//! DenseMesh — Latent Node Network for Modelless Inference (root re-export shim).
//!
//! Substrate modules (adaptive_width, edge_bandit, topology, traits, types,
//! etc.) moved to `katgpt_transformer::dense_mesh` per Proposal 003 Phase 9.
//! This file re-exports them under the historical `katgpt_rs::dense_mesh::*`
//! paths AND keeps `node_transformer` as a root sibling, because it consumes
//! `crate::transformer::forward` (the cognitive-primitive composer) which
//! cannot move to `katgpt-transformer` without a cycle.
//!
//! See `crates/katgpt-transformer/src/dense_mesh/mod.rs` for the architecture
//! doc and `katgpt-rs/.plans/266_densemesh_latent_node_network.md` Phase 5
//! for the deferred-node history.

pub use katgpt_transformer::dense_mesh::*;

pub mod node_transformer;
pub use node_transformer::TransformerNode;
