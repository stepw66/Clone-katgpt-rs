//! DenseMesh — Latent Node Network for Modelless Inference.
//!
//! Distillation of LMNet (arXiv:2505.12741, ICML 2026). Treats multiple forward
//! passes through the same LLM as nodes in a directed graph, communicating via
//! dense hidden-state vectors instead of natural-language tokens. Edges are
//! pluggable: identity (baseline), LoRA adapter, fixed projection.
//!
//! # Architecture
//!
//! - [`DenseNode`] — a stripped transformer forward pass (no embed/de-embed)
//! - [`DenseEdge`] — a transformation applied to hidden state between nodes
//! - [`LayerwiseTopology`] — the graph orchestration (layer widths, edge matrix)
//! - [`EdgeBandit`] — Thompson-sampling bandit over (topology, edge_set)
//!
//! # Latent / Raw Compliance (AGENTS.md)
//!
//! `DenseHidden` is a **latent** channel — it never crosses `SyncBlock` or chain
//! quorum. Only the input and output boundary nodes touch tokens (raw values).
//! Bridge functions ([`crate::dense_mesh::types::latent_to_raw_scalar`],
//! [`crate::dense_mesh::types::raw_to_latent_projection`]) convert at the seam.
//!
//! # Anti-Patterns (do NOT do)
//!
//! - Never encode `MapPos` as a `DenseHidden` then decode for sync (lossy)
//! - Never validate a movement claim by latent similarity (need exact x,y)
//! - Never sync the full hidden vector when a scalar projection suffices
//!
//! Reference: katgpt-rs/.research/234_DenseMesh_Latent_Node_Network.md

// Adaptive width controller — Plan 266 Phase 5 CollapseAware + BreakevenRouter
// integration. Picks between narrow/wide topology per query, driven by
// external CollapseDetector and BreakevenBandit signals. Sub-modules are
// feature-gated internally so callers without `collapse_aware_thinking` or
// `breakeven_routing` still get the base `AdaptiveWidthConfig` API.
pub mod adaptive_width;
pub mod compute_router;
pub mod edge_bandit;
pub mod edge_identity;
pub mod edge_lora;
pub mod edge_projection;
pub mod handoff;
pub mod node_transformer;
pub mod topology;
pub mod traits;
pub mod types;

pub use adaptive_width::{AdaptiveWidthConfig, WidthDecision};
pub use edge_bandit::{EdgeBandit, EdgeBanditArm};
pub use edge_identity::IdentityEdge;
pub use edge_lora::LoraEdge;
pub use edge_projection::ProjectionEdge;
pub use handoff::HiddenHandoff;
pub use node_transformer::TransformerNode;
pub use topology::{LayerwiseTopology, TopologyError};
pub use traits::{DenseEdge, DenseNode};
pub use types::{
    latent_to_raw_scalar, raw_to_latent_projection, ComputeTarget, DenseHidden, LayerRole,
    MeshConfig, MeshScratch, Topology,
};
