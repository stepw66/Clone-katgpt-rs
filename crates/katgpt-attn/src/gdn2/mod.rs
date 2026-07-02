//! Gated DeltaNet-2 (GDN2) — O(1) decode kernel + types.
//!
//! This module owns the GDN2 recurrent attention substrate (kernel + types).
//! The composition layer (`forward_gdn2`, which takes `ForwardContext`) stays
//! in the root crate (`katgpt_rs::gdn2::forward`).
//!
//! See the root `gdn2/mod.rs` for the full architecture documentation.
//! Reference: Yang, Zhang, Kautz (2024). "Gated Delta Networks."

pub mod kernel;
pub mod types;
// Composition layer (Issue 007 Phase F.4a, 2026-07-02): forward_gdn2 moved
// here from root `src/gdn2/forward.rs` now that ForwardContext lives in
// katgpt-forward. Gated by the parent `gdn2_attention` feature in lib.rs.
pub mod forward;

pub use kernel::{gdn2_recurrent_step, gdn2_state_readout, gdn2_state_update, l2_normalize, sigmoid};
pub use types::{Gdn2GateConfig, Gdn2HeadState, Gdn2LayerState, MultiLayerGdn2Cache};
pub use forward::{forward_gdn2, generate_gdn2_into};
