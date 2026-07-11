//! Claim Rubric Runtime — L1/L2/L3 evidence ladder validator (Plan 307,
//! Research 287, arxiv 2606.07612).
//!
//! Materializes the R287 §2.2 rubric as executable Rust. A generic
//! meta-discipline that grades probe / steering claims by evidence level:
//!
//! - **L1 — Behavioral** — "primitive P *reads/detects/projects* signal D".
//! - **L2 — Functional** — "signal from P *induces* downstream effect E".
//! - **L3 — Causal-mechanistic** — "direction w_B *causally controls* behavior B".
//!
//! Vocabulary must match evidence (R287 §2.3): an L3 verb ("causally
//! controls") in a claim with only L1 evidence is flagged as an
//! overclaiming `VocabularyViolation`, and the claim's `honest_level`
//! reflects what the evidence actually supports.
//!
//! ## Scope
//!
//! This module operates on **claim text + metadata**, not on latent
//! embeddings or raw physical values. There is no sigmoid, no projection,
//! no sync crossing. `FeatureClass` is re-used as a *tag*, not a math
//! operation (R287 §3). R287 §6 anti-pattern #3 (sync-boundary leakage) is
//! encoded as the S2 `LatentFreshnessCheck` checklist item, not as a code
//! path.
//!
//! ## Out of scope
//!
//! - No LLM-judge integration. The validator is rule-based over text +
//!   metadata. R287 §5 S3 "if LLM judge" checklist items are present as
//!   [`EvidenceItemId`] variants but the validator does not call a judge.
//! - No markdown linter that scans `.research/*.md` files — that is a
//!   future CLI tool layered on top of [`ClaimValidator`].
//! - No runtime invocation by hot-path kernels — this is a development- /
//!   CI- / GOAT-gate-time tool, not a 20Hz-tick tool.
//!
//! See `katgpt-rs/.plans/307_claim_rubric_runtime.md` for the full design.
//! See `katgpt-rs/.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md`
//! §2.2 (rubric), §2.3 (vocabulary), §5 (S1–S4 checklist).

pub mod checklist;
pub mod types;
pub mod validator;
pub mod vocabulary;

pub use checklist::{full_checklist, requirements, section_items};
pub use types::{
    ChecklistSection, Claim, EvidenceItem, EvidenceItemId, EvidenceLevel, Grade,
    VocabularyViolation,
};
pub use validator::ClaimValidator;

// Re-export `FeatureClass` from `katgpt-core` (Plan 292 T1.5 shim pattern —
// same as `src/pruners/feature_class.rs`). Do NOT duplicate the type. R287 §3
// uses this enum as a *claim tag*, not a math operation.
pub use katgpt_core::traits::FeatureClass;
