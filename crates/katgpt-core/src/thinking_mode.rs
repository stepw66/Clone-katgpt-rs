//! Per-query thinking mode tag (Plan 388 Phase 3 extraction).
//!
//! Extracted from `katgpt-pruners` (where it was the canonical definition) to
//! `katgpt-core` to break the katgpt-pruners ↔ katgpt-speculative cycle.
//! `katgpt-pruners::ThinkingMode` and `katgpt_rs::speculative::ThinkingMode`
//! both re-export this type for backwards compatibility.
//!
//! This is the SINGLE canonical definition — both
//! `katgpt_pruners::collapse_detector` and `katgpt_speculative::thinking_controller`
//! reference this type. Previously duplicated/re-exported across the boundary
//! to break an older dependency cycle; the cycle is now resolved by defining
//! the shared tag here (the lowest crate) and having both consumers re-export it.
//!
//! Crosses the crate boundary as plain `u8` via `#[repr(u8)]` for FFI/persistence.

/// Per-query thinking mode — controls whether latent reasoning is invoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ThinkingMode {
    /// Answer directly — no latent reasoning (baseline for benchmarks).
    Direct,
    /// Full latent reasoning via RiM buffer slots.
    Latent,
    /// CPU-side PPoT resample (cheaper than full GPU RiM decode).
    CpuResample,
    /// Dendritic-gated reasoning (Plan 194 variant).
    Dendritic,
}
