//! SwiR Switch-Thinking — root re-export shim (Proposal 003 Phase 9).
//!
//! Substrate modules (controller, entropy, signal_mix, soft_embedding, types,
//! etc.) moved to `katgpt_transformer::swir`. This file re-exports them under
//! the historical `katgpt_rs::swir::*` paths AND keeps `strategy_adapter` as a
//! root sibling, because it consumes
//! `crate::thinking_cot::{ThinkingStrategy, StepContext, StepDirective,
//! ControlTokenIds}` — and `thinking_cot` is a root module with its own deep
//! dep tree that cannot yet move to a crate.
//!
//! See `crates/katgpt-transformer/src/swir/mod.rs` for the architecture doc
//! and `katgpt-rs/.plans/275_swir_switch_thinking.md` for the full plan.

pub use katgpt_transformer::swir::*;

pub mod strategy_adapter;
pub use strategy_adapter::SwiRStrategyAdapter;
