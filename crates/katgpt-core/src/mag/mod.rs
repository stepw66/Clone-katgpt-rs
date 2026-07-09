//! MAG — Mining via Activation Geometry (arXiv:2607.04222).
//!
//! Unsupervised, modelless direction-mining + modelless transfer-prediction
//! primitive. Distilled from LeVi, David, Fomin (ICML 2026 FAGEN Workshop).
//!
//! ## What this is
//!
//! **The missing acquisition step** for the direction-vector ecosystem. Today
//! every direction vector in the codebase is either designer-authored
//! (LatentFieldSteering Plan 309) or supervised-extracted (EmotionDirections
//! Plan 162, KG Latent Octree R196 — mean-difference on labeled data). MAG mines
//! directions **unsupervised** from the host's own runtime verdict `y_M` — no
//! human labels, no gradient descent.
//!
//! Plus the §4 transfer-prediction experiment is a genuinely new capability:
//! modelless "which experience teaches the most". The paper achieves 94.7%
//! Top-1 accuracy predicting dataset transfer; raw centroid cosine achieves
//! only ρ ≈ 0.03 (near random).
//!
//! ## The two halves
//!
//! 1. **Mining** ([`mining`]): `mine_direction` / `mine_contrast_direction`
//!    extract unit-norm feature directions from activation shifts, using the
//!    model's own verdict `y_M` as the label. `reconstruction_error` computes
//!    the linearity diagnostic ϵ_Q (≈0 ⇒ steerable, ≈1 ⇒ entrenchied).
//!    `calibrate_alpha` makes injection strength substrate-invariant.
//!    `apply_operator` computes the 8 MAG readout summaries.
//!
//! 2. **Transfer** ([`transfer`]): `transfer_score` / `rank_candidates` predict
//!    which candidate dataset/experience best improves a target capability —
//!    modellessly, via geometric comparison of activation sets.
//!
//! ## Why modelless
//!
//! The "label" is the model/runtime's own verdict `y_M` — a runtime observation
//! (did the NPC succeed? did the claim pass the rubric?), NOT a training target.
//! The math is mean-difference (identical to EmotionDirections) + cosine
//! geometry. No gradients, no backprop, no weight mutation. Mined directions are
//! frozen as `BLAKE3`-committed artifacts (same envelope as
//! `MerkleFrozenEnvelope` in riir-neuron-db).
//!
//! ## §3.5 modelless-unblock relevance
//!
//! MAG direction mining is a **path-3** (latent-space correction) tool per the
//! research skill's modelless-unblock protocol. A systematically biased verdict
//! (e.g., "signal doubled", "position offset") can potentially be corrected by
//! mining the bias direction and projecting it out — before deferring to
//! riir-train. The `ϵ_Q ≈ 1` diagnostic predicts non-steerability (entrenched
//! bias), flagging when a latent correction won't work.
//!
//! ## Fusion (the Super-GOAT angle)
//!
//! - **F1**: MAG mines directions → Latent Field Steering (P309) injects them →
//!   NPCs discover reasoning directions from their own experience.
//! - **F2**: MAG transfer prediction = directed curiosity ("what transfers to my
//!   goal?") → CGSP (R126) + AnyRAG escalation.
//! - **F3**: MAG mines archetype directions unsupervised → CommittedFieldBlend
//!   (P321) + PersonalityWeightedComposition (P297) blend them.
//! - **F4**: MAG transfer prediction ranks which experiences to consolidate →
//!   Raven/δ-Mem (riir-neuron-db).
//!
//! ## Status
//!
//! Phase 2 COMPLETE (2026-07-09). GOAT G1–G6 ALL PASS. Promoted to DEFAULT-ON.
//! G2 (the headline kill-it gate) verified: contrast directions mined from
//! self-labeled classes ARE linearly separable (LOO accuracy 0.925 at σ=1.5,
//! 0.810 at σ=3.0). G4: MAG class-conditional transfer Top-1 0.720 vs raw
//! centroid cosine 0.220 (3.3×). Phase 2 added `mine_direction_into` +
//! `transfer_score_into` zero-alloc hot-path variants.
//!
//! See: `katgpt-rs/.research/397_Mining_via_Activation_Geometry.md`
//! See: `katgpt-rs/.plans/418_mag_activation_geometry_primitive.md`
//! See: `katgpt-rs/.benchmarks/418_mag_goat.md`

pub mod mining;
pub mod transfer;
pub mod types;

// Re-export the public API at the module root for ergonomic access
// (`katgpt_core::mag::mine_direction` instead of `...::mag::mining::mine_direction`).
pub use mining::{
    apply_operator, apply_operator_into, calibrate_alpha, mine_contrast_direction,
    mine_direction, mine_direction_into, reconstruction_error,
};
pub use transfer::{rank_candidates, transfer_score, transfer_score_into, DataSet, RankEntry};
pub use types::{MagDirection, MagError, MagOperator, TransferMetric};
