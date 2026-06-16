//! FaithfulnessProbe — Causal Intervention Diagnostic for Injected Memory (Modelless)
//!
//! Plan 278, Research 244. Feature flag: `faithfulness_probe` (and `triggered_injection`).
//!
//! Implements the open half of the Cognitive Integrity Layer:
//! - [`types`] — `Intervention`, `FaithfulnessProfile`, `ConsumerContext`, `MemorySlice`.
//! - [`probe`] — `FaithfulnessProbe` trait + `DefaultFaithfulnessProbe`.
//! - [`perturb`] — causal perturbation strategies (empty/shuffle/corrupt/irrelevant/filler).
//! - [`attribution`] — `AttributionProbe` (finite-difference IG surrogate).
//! - [`gate`] — `TriggeredInjectionGate` + `EntropyThresholdGate` + `UncertaintySignal`.
//!
//! All modelless: zero training, zero backprop through base weights. Zero-allocation
//! on hot paths (`EntropyThresholdGate::should_inject` <10ns). The probe suite runs
//! at audit cadence (every N ticks), not per-tick.
//!
//! Based on Zhao et al. 2026 (arxiv 2601.22436), "Large Language Model Agents Are
//! Not Always Faithful Self-Evolvers".

pub mod attribution;
pub mod gate;
pub mod perturb;
pub mod probe;
pub mod types;

// Convenience re-exports for the most-used items.
pub use attribution::{AttributionProbe, FiniteDifferenceAttributionProbe};
pub use gate::{EntropyThresholdGate, TriggeredInjectionGate, UncertaintySignal};
pub use probe::{DefaultFaithfulnessProbe, FaithfulnessProbe};
pub use types::{ConsumerContext, FaithfulnessProfile, Intervention, MemorySlice};
