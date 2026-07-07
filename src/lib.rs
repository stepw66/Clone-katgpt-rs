#![allow(unexpected_cfgs)]
#[cfg(all(target_os = "macos", feature = "ane"))]
pub mod ane_backend;
// Issue 359: `attn_match` extracted to the katgpt-attn-match leaf. The root
// re-exports the leaf as `attn_match` so all historical `katgpt_rs::attn_match::*`
// paths continue to resolve (Issue 014/015 re-export contract). The
// `adaptive_cot` glue stays in root (composes root-only `freq_bandit`).
#[cfg(feature = "attn_match")]
pub use katgpt_attn_match as attn_match;
/// Adaptive CoT compaction glue — composes the leaf's online compactor with the
/// root-only `freq_bandit` bandit threshold tuner. Stays in root per Issue 359
/// (freq_bandit depends on root-only `trigger_gate`).
#[cfg(feature = "adaptive_cot_compaction")]
pub mod attn_match_adaptive_cot;
// Phase 5 absorption (Proposal 003, 2026-07-04): module moved to katgpt-kv.
// Re-export preserves `katgpt_rs::async_qdq::*` paths.
#[cfg(feature = "async_qdq_overlap")]
pub use katgpt_kv::async_qdq;
pub mod benchmark;
// Phase 11 absorption (Proposal 003, 2026-07-04): module moved to katgpt-band.
// Re-export preserves `katgpt_rs::band_conditioner::*` paths.
#[cfg(feature = "band_conditioner")]
pub use katgpt_band::band_conditioner;
#[cfg(feature = "bckvss")]
pub use katgpt_band::bckvss;
#[cfg(feature = "breakeven_routing")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::breakeven::*` paths.
pub use katgpt_core::breakeven;
// Phase 5 absorption (Proposal 003, 2026-07-04): module moved to katgpt-kv.
// Re-export preserves `katgpt_rs::cache_prune::*` paths.
#[cfg(feature = "cache_prune")]
pub use katgpt_kv::cache_prune;
// Phase 10 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::channel_simd::*` paths.
#[cfg(feature = "channel_simd_align")]
pub use katgpt_core::channel_simd;
// CGSP inlined from src/cgsp.rs (Proposal 003 Phase 0.3, 2026-07-01): the
// 37-line shim file is replaced by a direct module re-export. `katgpt::cgsp`
// resolves to `katgpt_core::cgsp`, so all public types, the `traits` / `types`
// submodules, and `sigmoid` are accessible unchanged. The `cgsp_dual_pool`
// items resolve the same way when that feature forwards to katgpt-core.
#[cfg(feature = "cgsp")]
pub use katgpt_core::cgsp;
// Phase 11 absorption (Proposal 003, 2026-07-04): modules moved to katgpt-claim.
// Re-exports preserve `katgpt_rs::clr::*` and the flat symbol surface.
#[cfg(feature = "clr")]
pub use katgpt_claim::clr;
#[cfg(feature = "clr")]
pub use katgpt_claim::clr::{
    Claim, ClaimExtractor, ClaimVerifier, ClrConfig, ClrScratch, Cluster, DirectionVectorSource,
    FnClaimExtractor, ReliabilityScore, SigmoidProjectionVerifier, Trajectory, Verdict, VoteResult,
    allocate_budget, brevity_tiebreak, clr_vote, clr_vote_minimal, learning_potential,
    mgpo_sampling_weight, should_write_memory,
};

// Claim Rubric Runtime — L1/L2/L3 evidence ladder validator (Plan 307,
// Research 287, arxiv 2606.07612). Generic meta-discipline that grades
// probe/steering claims by evidence level: L1 (Behavioral) / L2
// (Functional) / L3 (Causal-mechanistic). Vocabulary must match evidence
// — "causally controls" requires L3 evidence; "reads" is L1-safe. Opt-in
// until Phase 2 round-trip tests pass on R287 §4 scores.
//
// Phase 11 absorption (Proposal 003, 2026-07-04): module moved to katgpt-claim.
// Re-exports preserve `katgpt_rs::claim_rubric::*` and the flat symbol surface.
#[cfg(feature = "claim_rubric")]
pub use katgpt_claim::claim_rubric;
#[cfg(feature = "claim_rubric")]
pub use katgpt_claim::claim_rubric::{
    ChecklistSection, ClaimValidator, EvidenceItem, EvidenceItemId, EvidenceLevel, Grade,
    VocabularyViolation,
};
// Phase 10 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::cumprodsum::*` paths.
pub use katgpt_core::cumprodsum;
// CUCG — Closed-Unit Compaction Gate (Plan 333, Research 300, arxiv 2606.23525).
// Generic rubric-gated trajectory compaction primitive. DEFAULT-ON since
// Phase 6 (2026-06-25): 7/7 GOAT gates PASS. Re-exports the public surface
// for ergonomic use.
#[cfg(feature = "closed_unit_compaction")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::compaction::*` paths.
pub use katgpt_core::compaction;
#[cfg(feature = "closed_unit_compaction")]
pub use katgpt_core::compaction::{
    Backstop, ClosedUnitCompactionGate, ClosedUnitCompactionGateBuilder, CombineOp,
    CompactionAuditRecord, CompactionDecision, DecisionKind, FireRule, FireRuleEval,
    PredicateAudit, PredicateReason, PredicateResult, Rubric, RubricScratch, RubricVerdict,
};
// Phase 10 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::ssd_block::*` paths.
#[cfg(feature = "ssd_block")]
pub use katgpt_core::ssd_block;
#[cfg(feature = "dash_attn")]
pub mod dash_attn;
#[cfg(feature = "data_probe")]
pub mod data_probe;
// Issue 007 Phase C: `npc_ane_backend` and `npc_brain_router` moved to
// riir-engine (NPC runtime IP). They depended on `katgpt_core::sense::backend`
// which moved, and are themselves gameplay-runtime IP per the 5-repo strategy.
// Shared diagonal gate abstraction (GDN2 + Wall).
// Available when either gdn2_attention or wall_attention is enabled.
#[cfg(feature = "cubical_nerve")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::cubical_nerve::*` paths.
pub use katgpt_core::cubical_nerve;
// Phase 11 absorption (Proposal 003, 2026-07-04): module moved to katgpt-band.
// Re-export preserves `katgpt_rs::collider_pruner::*` paths.
#[cfg(feature = "collider_consistency")]
pub use katgpt_band::collider_pruner;
// CompressionDrafter — corpus-as-model quest grammar drafter (Plan 285, Research 256).
// Re-exports katgpt-core's compression_drafter module for downstream consumers (riir-games).
// Opt-in behind the `compression_drafter` feature until GOAT gate passes.
#[cfg(any(feature = "gdn2_attention", feature = "wall_attention"))]
pub use katgpt_attn::diagonal_gate;
#[cfg(feature = "compression_drafter")]
pub use katgpt_core::compression_drafter;
// Phase 6 absorption (Proposal 003, 2026-07-04): `ilc` moved to katgpt-speculative;
// `trd` stays root (depends on `crate::fold` — transformer-bound glue). peira
// already re-exports katgpt-spectral (Phase 4). The distill/mod.rs shim re-exports
// ilc from katgpt-speculative so `katgpt_rs::distill::ilc::*` paths still resolve.
#[cfg(any(
    feature = "peira_distill",
    feature = "ilc_distill",
    feature = "trd_refined_draft"
))]
pub mod distill;
#[cfg(feature = "dllm")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_range_loop)]
pub mod dllm;
#[cfg(feature = "ega_attn")]
pub use katgpt_attn::ega_attn;
#[cfg(feature = "critical_interval_gate")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::dllm_solver::*` paths.
pub use katgpt_core::dllm_solver;
// `feedback` module exiled to `katgpt-deprecated` (Phase 3a, Proposal 003).
// Re-export preserved for back-compat: `katgpt_rs::feedback::*` still resolves.
#[cfg(feature = "feedback")]
pub use katgpt_deprecated::feedback;
#[cfg(feature = "chain_fold")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-speculative.
// Re-export preserves `katgpt_rs::fold::*` paths.
pub use katgpt_speculative::fold;
// CCE — Coarse Correlated Equilibria moderator primitives (Plan 295 + Plan 300, Research 274, arxiv 2606.20062).
// Generic, game-agnostic LP-CCE formulation + external-regret functional +
// heterogeneous (subjective-CCE) extension + primal-dual iterator.
// DEFAULT-ON after GOAT gates all PASS (G1+G2+G3+G4): G1 homogeneous
// equivalence regression, G2 regret transfer on synthetic heterogeneous
// CWMs (er_heterogeneous(ρ⋆) ≤ 1e-3), G3 primal-dual convergence at log-log
// slope -1.0 (beats paper's -0.5 O(N⁻¹ᐟ²) bound — Plan 300 T4.3b), G4 16-player
// latency = 33.97ms < 50ms target.
// Phase 10 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::cce::*` paths.
#[cfg(feature = "cce_moderator")]
pub use katgpt_core::cce;
#[cfg(feature = "freq_bandit")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-pruners.
// Re-export preserves `katgpt_rs::freq_bandit::*` paths.
pub use katgpt_pruners::freq_bandit;
#[cfg(feature = "gdn2_attention")]
pub mod gdn2;
#[cfg(all(target_os = "macos", feature = "gpu_inference"))]
pub mod gpu_backend;
#[cfg(feature = "hla_attention")]
pub mod hla;
// katgpt-quant re-export (Proposal 003 Phase 1, 2026-07-01): quantization codecs
// moved to crates/katgpt-quant/. Re-exported here so historical `katgpt_rs::*`
// paths resolve.
#[cfg(feature = "hybrid_oct_pq")]
pub use katgpt_quant::hybrid_oct_pq;
pub mod inference_backend;
pub mod inference_router;
#[cfg(feature = "interval_pruner")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-pruners.
// Re-export preserves `katgpt_rs::interval_pruner::*` paths.
pub use katgpt_pruners::interval_pruner;
#[cfg(feature = "lattice_operad")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-pruners.
// Re-export preserves `katgpt_rs::lattice_operad::*` paths.
pub use katgpt_pruners::lattice_operad;
#[cfg(feature = "iso_quant")]
pub use katgpt_quant::iso_quant;
#[cfg(feature = "gauge_invariant")]
pub use katgpt_spectral::gauge_invariant;
#[cfg(feature = "manifold_power_iter_router")]
pub use katgpt_spectral::manifold_power_iter_router;
pub use katgpt_spectral::spectral_retract;
// Kog CPU Fusion — RMSNorm gamma folding + QKV interleaving (Plan 160 GOAT 3/3 Gemma 2 scale).
// Phase 9: mbu moved to katgpt-transformer; re-export preserves `katgpt_rs::mbu::*`.
#[cfg(feature = "newton_schulz")]
pub use katgpt_core::newton_schulz; // Extracted to katgpt-core per Issue 355 Phase 1a; re-export preserves historical `katgpt_rs::newton_schulz::*` paths.
#[cfg(feature = "modality_pruned_load")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::pipeline_pruner::*` paths.
pub use katgpt_core::pipeline_pruner;
#[cfg(feature = "octopus")]
pub use katgpt_quant::octopus;
#[cfg(feature = "planar_quant")]
pub use katgpt_quant::planar_quant;
#[cfg(feature = "hla_eigenbasis_recovery")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-spectral.
// Re-export preserves `katgpt_rs::hla_eigenbasis::*` paths.
pub use katgpt_spectral::hla_eigenbasis;
#[cfg(feature = "off_principal_retrieval")]
pub use katgpt_spectral::off_principal;
#[cfg(feature = "kog_cpu_fusion")]
pub use katgpt_transformer::mbu;
#[cfg(feature = "plot")]
pub mod plot; // Issue 355 Phase 2a: gated behind `plot` feature (plotters is now optional). DEFAULT-ON.
// Orthogonal Procrustes — cross-frame embedding alignment via polar
// decomposition (Newton-Schulz on B^T A). Issue 001 (katgpt-rs). GOAT
// candidate — gated behind `orthogonal_procrustes` until benchmark gates
// G1–G6 (Issue 001) pass. Promotes to default-on if GOAT, demoted if not.
// Substrate moved to katgpt-spectral (Proposal 003 Phase 4).
#[cfg(feature = "orthogonal_procrustes")]
pub use katgpt_spectral::procrustes;
// Phase 6 absorption (Proposal 003, 2026-07-04): module moved to katgpt-speculative.
// Re-export preserves `katgpt_rs::precision_aware_draft::*` paths.
#[cfg(feature = "precision_aware_draft")]
pub use katgpt_speculative::precision_aware_draft;
#[cfg(feature = "progressive_mcgs")]
#[doc(alias = "mcts")]
#[doc(alias = "mcgs")]
#[doc(alias = "graph_search")]
#[doc(alias = "monte_carlo")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-speculative.
// Re-export preserves `katgpt_rs::progressive_mcgs::*` paths.
pub use katgpt_speculative::progressive_mcgs;
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-proof-cert.
// Re-export preserves `katgpt_rs::proof_cert::*` paths. The `conditional_proof!`
// macro is also re-exported via `#[macro_export]` at the leaf crate root.
#[cfg(feature = "proof_cert")]
pub use katgpt_proof_cert as proof_cert;
// Macro re-export so `katgpt_rs::conditional_proof!` continues to resolve after
// the move. `#[macro_export]` at the leaf places it at the leaf crate root; the
// `pub use` here mirrors the leaf's macro into the root crate's macro namespace.
#[cfg(feature = "proof_cert")]
pub use katgpt_proof_cert::conditional_proof;
pub mod pruners;
// DenseMesh — latent node network for modelless inference (Plan 266, Research 234).
#[cfg(feature = "dense_mesh")]
pub mod dense_mesh;
#[cfg(feature = "rat_plus_bridge")]
pub use katgpt_attn::rat_bridge;
// Phase 8 absorption (Proposal 003, 2026-07-04): module moved to katgpt-attn-match.
// Re-export preserves `katgpt_rs::rerank::*` paths.
#[cfg(feature = "maxsim")]
pub use katgpt_attn_match::rerank;
#[cfg(feature = "river_valley")]
pub use katgpt_spectral::river_valley;
// Phase 6 absorption (Proposal 003, 2026-07-04): module moved to katgpt-speculative.
// Re-export preserves `katgpt_rs::rt_turbo::*` paths.
#[cfg(feature = "rt_turbo")]
pub use katgpt_speculative::rt_turbo;
// Phase 11 absorption (Proposal 003, 2026-07-04): module moved to katgpt-ruliology.
// Re-export preserves `katgpt_rs::ruliology::*` paths.
#[cfg(feature = "ruliology")]
pub use katgpt_ruliology as ruliology;
// Phase 5 absorption (Proposal 003, 2026-07-04): module moved to katgpt-kv.
// Re-export preserves `katgpt_rs::segment_checkpoint::*` paths.
#[cfg(feature = "chiaroscuro")]
pub use katgpt_attn::chiaroscuro;
#[cfg(feature = "segment_checkpoint")]
pub use katgpt_kv::segment_checkpoint;
// Functional Attention composition layer — Plan 286 Phase 5 (T5.1–T5.3). Each
// submodule is independently feature-gated; the module root compiles when any
// of the three composition features is on.
#[cfg(any(
    feature = "funcattn_spectral_pre_rotate",
    feature = "funcattn_chiar_blend",
    feature = "funcattn_freeze_thaw"
))]
pub use katgpt_attn::funcattn_compose;
// Phase 11 absorption (Proposal 003, 2026-07-04): modules moved to katgpt-sparse.
// Re-exports preserve `katgpt_rs::specialist_projection::*` and
// `katgpt_rs::sparse_task_vector::*` paths.
#[cfg(feature = "sparse_task_vector")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-sparse.
// Re-export preserves `katgpt_rs::sparse_compose::*` paths.
pub use katgpt_sparse::sparse_compose;
#[cfg(feature = "sparse_task_vector")]
pub use katgpt_sparse::sparse_task_vector;
#[cfg(feature = "specialist_projection")]
pub use katgpt_sparse::specialist_projection;
// Phase 10 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::skill_opt::*` paths.
#[cfg(feature = "skill_opt")]
pub use katgpt_core::skill_opt;
#[cfg(feature = "sleep_consolidation")]
pub mod sleep;
#[cfg(feature = "cs_kv_probe")]
pub use katgpt_kv::cs_kv_probe;
#[cfg(feature = "kv_share")]
pub use katgpt_kv::kv_share;
#[cfg(feature = "kvarn")]
pub use katgpt_kv::kvarn;
#[cfg(feature = "osc_kv")]
pub use katgpt_kv::osc_kv;
#[cfg(feature = "shard_kv")]
pub use katgpt_kv::shard_kv;
#[cfg(feature = "still_kv")]
pub use katgpt_kv::still_kv;
#[cfg(feature = "targeted_precision")]
pub use katgpt_kv::targeted_precision;
#[cfg(feature = "sp_kv")]
pub mod sp_kv {
    //! SP-KV re-export bridge (Issue 015 Phase 5).
    //!
    //! Combines `katgpt_kv::sp_kv` (types + utility predictor) with the local
    //! `sp_kv_forward_mod` (transformer glue that depends on ForwardContext).
    pub use katgpt_kv::sp_kv::*;
    pub mod forward {
        pub use crate::sp_kv_forward_mod::*;
    }
    // Re-export the forward-module building blocks at the sp_kv top level so
    // historical `katgpt_rs::sp_kv::{GateBias, attention_head_core, ...}` paths
    // still resolve (back-compat with tests/examples).
    pub use crate::sp_kv_forward_mod::{
        GateBias, NoBias, SpKvForwardContext, attention_head_core, attention_head_gated,
        forward_sp_kv, forward_sp_kv_quant,
    };
}
/// SP-KV transformer glue — full pipeline functions that take ForwardContext.
/// Kept private; surfaced via the `sp_kv::forward` re-export above.
#[cfg(feature = "sp_kv")]
mod sp_kv_forward_mod;
// Phase 6 absorption (Proposal 003, 2026-07-04): module moved to katgpt-speculative.
// Re-export preserves `katgpt_rs::spec_reconciliation::*` paths. Originally ungated
// (the `spec_reconciliation = []` feature was vestigial); preserved as ungated.
pub use katgpt_speculative::spec_reconciliation;
// Phase 6 absorption (Proposal 003, 2026-07-04): module moved to katgpt-speculative.
// Re-export preserves `katgpt_rs::spechop::*` paths.
#[cfg(feature = "spectral_budget")]
pub use katgpt_spectral::spectral_budget;
#[cfg(feature = "spectral_rank")]
pub use katgpt_spectral::spectral_concentration;
#[cfg(feature = "spechop")]
pub use katgpt_speculative::spechop;
#[cfg(feature = "spectral_quant")]
pub mod spectralquant {
    //! Spectralquant re-export shim (Issue 015 Phase 5).
    //!
    //! The substrate physically lives in `crates/katgpt-spectral/`. This
    //! module re-exports it so all historical `katgpt_rs::spectralquant::*`
    //! paths continue to resolve for `funcattn_compose`, `chiaroscuro`,
    //! `benchmark/infrastructure`, and all tests/examples.
    pub use katgpt_spectral::*;
}
pub mod speculative;
// SwiR Switch-Thinking — Explicit↔Latent mode controller (Plan 275, Research 241).
// Phase 12 T4.6 (2026-07-04): strategy_adapter moved to katgpt-transformer
// (the thinking_cot cycle is resolved). This shim re-exports everything.
#[cfg(feature = "static_cal_tables")]
pub use katgpt_attn::static_cal;
#[cfg(feature = "stiff_anomaly")]
pub use katgpt_spectral::stiff_anomaly;
#[cfg(feature = "swir_switch_thinking")]
pub use katgpt_transformer::swir;
// thinking_cot — adaptive CoT framework (Plan 194). The feature is a
// meta-feature that pulls in the bandit/prune/probe machinery required by
// speculative::thinking_controller; the module itself owns the shared
// ThinkingStrategy trait (Plan 275 Phase 2).
pub use katgpt_tokenizer as tokenizer;
#[cfg(feature = "thinking_cot")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-transformer.
// Re-export preserves `katgpt_rs::thinking_cot::*` paths.
pub use katgpt_transformer::thinking_cot; // re-export (Issue 014): preserves `katgpt_rs::tokenizer::*` paths for tests/examples/validator
pub mod transformer;
// Phase 10 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::trigger_gate::*` paths.
pub use katgpt_core::trigger_gate;
#[cfg(feature = "turboquant")]
pub use katgpt_quant::turboquant;
pub mod types;
#[cfg(feature = "unit_distance")]
pub use katgpt_deprecated::unit_distance;

// Plan 008 Step 2: weight-packing substrate now lives in `katgpt-transformer`.
// Historical `crate::weights::ContiguousWeights` / `load_ternary_bits` callers
// resolve through this re-export unchanged.
pub use katgpt_transformer::{ContiguousWeights, load_ternary_bits};

// Plan 265 Phase 4: Adaptive CoT stopping criterion (depends on band_conditioner).
// Phase 11 absorption (Proposal 003, 2026-07-04): module moved to katgpt-band.
// Re-export preserves `katgpt_rs::adaptive_cot_stopper::*` paths.
#[cfg(feature = "adaptive_cot_identifiability")]
pub use katgpt_band::adaptive_cot_stopper;

// Phase 10 absorption (Proposal 003, 2026-07-04): the TrackingAllocator struct
// + reset_alloc_stats/get_alloc_stats helpers + tests moved to katgpt-core.
// Re-export preserves `katgpt_rs::alloc::*` paths. The `#[global_allocator]`
// static MUST stay in the root crate (binary/library crate boundary — the
// global allocator is process-global and can only be declared once, in the
// final crate that links into a binary). The katgpt-core module exposes the
// type + helper fns; root instantiates the static.
#[cfg(debug_assertions)]
pub use katgpt_core::alloc;

/// Debug-only global allocator that tracks allocation count and bytes.
#[cfg(debug_assertions)]
#[global_allocator]
static GLOBAL_ALLOC: katgpt_core::alloc::TrackingAllocator = katgpt_core::alloc::TrackingAllocator;

// Phase 10 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::mux_demux::*` paths.
#[cfg(feature = "mux_demux")]
pub use katgpt_core::mux_demux;

#[cfg(feature = "mux_latent_context")]
// Phase 12 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::mux_latent::*` paths.
pub use katgpt_core::mux_latent;

// Memory Soup LoRA Artifact Importer (Plan 253 T19 G5).
// Standalone MSP0 binary format parser — uses only std + blake3, no riir-gpu dep.
// Proves katgpt-rs can consume riir-gpu's exported Memory Soup artifacts.
// Phase 10 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::memory_soup_lora::*` paths.
#[cfg(feature = "memory_soup_lora")]
pub use katgpt_core::memory_soup_lora;

// Phase 10 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::llmexec_guard::*` paths.
#[cfg(feature = "llmexec_guard")]
pub use katgpt_core::llmexec_guard;

// Phase 11 absorption (Proposal 003, 2026-07-04): module moved to katgpt-validator.
// Re-export preserves `katgpt_rs::validator::*` paths.
#[cfg(feature = "validator")]
pub use katgpt_validator as validator;

#[cfg(feature = "breakeven_routing")]
pub use katgpt_core::breakeven::{
    BreakevenBandit, BreakevenStats, BreakevenTierPair, BreakevenTracker,
};

#[cfg(feature = "tf_loop")]
pub mod tf_loop;

// Closure-Expansion Instrument — runtime wiring (Plan 290 Phase 4 T4.2/T4.3).
// `closure_wire` decorates any ScreeningPruner (BanditPruner / AbsorbCompressLayer)
// with PTG recording (wake-phase decorator — Phase 8 katgpt-pruners absorption
// target). `closure_mining` runs motif mining + admission at sleep-cycle
// boundaries — Proposal 003 Phase 7 (2026-07-04) hoisted this module into
// `katgpt-core::closure::mining`. The original proposal targeted katgpt-sleep,
// but that triggered a cyclic package dep (katgpt-core → katgpt-sleep →
// katgpt-core, because katgpt-core already depends on katgpt-sleep for the
// sleep_time_anticipation re-export). katgpt-core is the natural home — the
// instrument is a thin wrapper around `closure::{MotifMiner, MotifAdmitter,
// compute_pri, compute_cdg}` which already live there. The historical
// `katgpt_rs::closure_mining::*` API path is preserved by the `pub use`
// re-export below; external consumers (riir-engine::closure_bridge) are
// unaffected. Both `closure_wire` and `closure_mining` are gated on
// `closure_instrument`; the AbsorbCompress auto-tracing impl in `closure_wire`
// additionally needs `bandit`.
// Phase 8 absorption (Proposal 003, 2026-07-04): module moved to katgpt-pruners.
// Re-export preserves `katgpt_rs::closure_wire::*` paths.
#[cfg(feature = "closure_instrument")]
pub use katgpt_pruners::closure_wire;

#[cfg(feature = "closure_instrument")]
pub use katgpt_core::closure::mining as closure_mining;

// Salience Tri-Gate Primitive — open 3-way per-tick emit gate (Speak / Silent /
// Delegate) distilled from JoyAI-VL-Interaction (Plan 303, Research 281,
// arxiv 2606.14777). Two stacked sigmoids (never softmax); silence is a
// first-class variant, not a threshold-suppression default; zero-allocation
// hot path; deterministic for replay/sync. Game-side NPC wiring lives in
// riir-ai Plan 330 — this crate stays math-only, MIT, no game IP.
// Opt-in until G1 (determinism + monotonicity) + G2 (two-sigmoid ablation
// parity) + <50ns decide() latency gates pass.
// Phase 10 absorption (Proposal 003, 2026-07-04): module moved to katgpt-core.
// Re-export preserves `katgpt_rs::salience::*` paths. The flat type re-export
// below resolves through this shim.
#[cfg(feature = "salience_tri_gate")]
pub use katgpt_core::salience;
#[cfg(feature = "salience_tri_gate")]
pub use salience::{
    DelegateToken, FoldbackTarget, SalienceDecision, SalienceTriGate, SilenceToken,
};

// Algorithmic-Probability Sampler + Coincidence Gate — two open primitives
// distilled from Dingle & Hutter 2026 (Plan 305, Research 284, Entropy
// 28(2):226). `CompressionPriorSampler<K>` replaces uniform candidate sampling
// in MCTS / bandits / DDTree / speculative drafters with a simplicity-biased
// prior (sigmoid per candidate, never softmax as the public API). Pluggable
// K̃: RLE ratio, Shannon entropy, L1 norm (LZ4 / BLAKE3 stubs gated behind
// sub-features). `CoincidenceGate` probes a found optimum against other simple
// objectives for theorem-backed cross-task transfer. riir-ai Plan 331 wires
// this to HLA / functor / shard vectors (private).
// Opt-in until G1 (sampler safety) + G2 (exponential speedup) gates pass.
// Phase 8 absorption (Proposal 003, 2026-07-04): screening moved to
// katgpt-pruners. Re-export preserves `katgpt_rs::screening::*` paths.
// External consumers (algorithmic_probability_sampler_demo / _bench examples)
// are unaffected.
#[cfg(feature = "complexity_prior_sampler")]
pub use katgpt_pruners::screening;
#[cfg(feature = "complexity_prior_sampler")]
pub use katgpt_pruners::screening::{
    CoincidenceGate, ComplexityProxy, CompressionPriorSampler, EntropyComplexity, L1Complexity,
    LatentCompressionPriorSampler, RleComplexity, quantize_latent,
};

// Alien Sampler Primitive — Coherence × Availability Frontier Ranking
// (Plan 311, Research 293, arxiv 2603.01092, Artiles et al. "The Alien Space
// of Science" May 2026). Generic, modelless within-pool ranking: z-scored
// linear fusion `(1−β)·zC + β·zU` of a coherence score and an unavailability
// score. `MedianTopMAvailability` implements the paper's load-bearing
// median-of-top-m cosine rule. Open math only — NPC population banks, CGSP
// binding, and zone emission feeds live in riir-ai Plan 312+.
// Opt-in until Phase 3 GOAT gate (G1 motif collapse ≤50% of OPUS baseline,
// G2 quality ≥90% of coherence-only, G3 perf ≤5× baseline, G4 no Vec<f32>
// escapes rank()) passes.
#[cfg(feature = "alien_sampler")]
pub use katgpt_deprecated::alien_sampler;
#[cfg(feature = "alien_sampler")]
pub use katgpt_deprecated::alien_sampler::{
    AlienConfig, AlienSampler, AlienSamplerError, AvailabilityScorer, CoherenceScorer,
    MedianTopMAvailability, ScoredCandidate,
};

// Vessel — Extract-Once Secure Wire Format Primitive (Plan 315 / Research 297).
//
// MOVED to riir-neuron-db/src/vessel/ (Plan 006, 2026-06-29). The primitive
// is now PRIVATE — publishing the security-enforcement wire format (magic
// bytes, BLAKE3 protocol, projector signatures, fuel-gating budgets, exact
// 52-byte header layout) handed attackers a free threat-model blueprint with
// zero public adoption value. No external dev builds on "secure WASM vessel
// format"; the only consumers were the private repos (riir-neuron-db
// NeuronVesselSidecar, riir-chain delivery, riir-ai runtime). See
// riir-neuron-db/.plans/006_vessel_primitive_migration.md.
//
// The `secure_vessel` Cargo feature, the `vessel_extract_bench`, and the
// `vessel_minimal` / `vessel_project` examples were removed in the same
// migration. Historical docs remain in katgpt-rs/.docs, .benchmarks, .plans,
// .research as the public record of what existed.
