//! Causal Head-Importance Calibration & Scale-Normalized Heterogeneous Fusion
//! (Plan 358, Research 362, arXiv:2606.20097 HydraHead).
//!
//! Two modelless, inference-time primitives distilled from HydraHead:
//!
//! - **Causal head-importance scoring** — a forward-pass-only causal-intervention
//!   scorer that ranks attention heads by their *necessity* for a target
//!   capability (activation patching Eq 10 + path patching Eq 11 + span-level
//!   logit-diff readout Eq 9). Strictly stronger than RTPurbo's observational
//!   attention-mass scoring: catches *correlated bystander* heads (attend
//!   strongly to the needle but are overridden downstream) that attention-mass
//!   wrongly promotes.
//! - **`ScaleNormalizedFusion`** — fuses outputs from two heterogeneous
//!   attention branches via independent RMSNorm per branch + a learnable
//!   per-head scalar γ (paper Eq 13–14). Ships ready for any future
//!   head-mixing runtime (currently unused — Plan 182 is layer-wise).
//!
//! # Modelless discipline
//!
//! Zero training, zero backprop through base weights. Both primitives are pure
//! functions / offline calibration tools. The head-importance score runs as
//! forward passes only (the paper emphasizes "lightweight and one-shot,
//! requiring only a few forward passes over a small calibration set"). The
//! architecture HydraHead trains (head-wise FA/LA mixing + three-stage
//! transfer) is out of scope — noted "→ riir-train" in Research 362 §3.5.
//!
//! # Caller responsibility
//!
//! The patched forward pass (selective head-output substitution +
//! downstream-attention freezing) needed to produce `m_patched` is the caller's
//! responsibility — it requires a full transformer forward and lives in
//! riir-engine / riir-games. This module is the *scorer*; the patched forward
//! pass is supplied by the caller via a closure. This keeps katgpt-core
//! leaf-clean (no transformer dep) and matches the FaithfulnessProbe pattern
//! (probe is generic, consumer supplies the behavior metric).

pub mod fusion;
pub mod patching;
pub mod readout;
pub mod scorer;

pub use fusion::ScaleNormalizedFusion;
pub use patching::{direct_effect_importance, indirect_effect_importance};
pub use readout::SpanLogitDiffReadout;
pub use scorer::{fuse_across_capabilities, partition_by_causal_score, per_capability_score};
