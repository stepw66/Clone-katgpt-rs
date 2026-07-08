//! katgpt-core: Shared types and SIMD kernels for katgpt-rs and riir-engine.
//!
//! This crate contains the common core shared between the two projects:
//! - **types**: Config, Rng, math utilities, LoRA, DomainLatent
//! - **simd**: NEON/AVX2 accelerated linear algebra kernels
//! - **hla**: Higher-order Linear Attention substrate (cache types + kernels)
//! - **mcts**: Generic Monte Carlo Tree Search over any `GameState`
//! - **delta_mem**: δ-mem associative memory substrate (state, hasher, multi-domain)
//! - **traits**: Shared traits for game AI and speculative decoding
//! - **speculative**: Speculative-decoding substrate types + sampling primitives
//!   (TreeNode, DraftResult, configs, LDT conflict detector, TES credit
//!   assignment, CDF/residual samplers)
//!
//! No feature flags on types/simd/hla/mcts/delta_mem/speculative — both projects
//! get the full substrate. Composition layers (root-only types like
//! `BanditRolloutPolicy`, `MemorySteeredPruner<P>`) stay in the consuming crate.

/// Standard logistic sigmoid: `σ(x) = 1 / (1 + e^{-x})`.
///
/// Numerically stable (branches on sign of `x` to avoid `e^{-x}` overflow).
/// Returns a value in `(0, 1)` for finite inputs. Always available — no feature
/// gate — because it's a pure math utility consumed across many domains (band
/// conditioning, CGSP, faithfulness gates, personality composition, etc.).
/// Hoisted here from `band_conditioner::sigmoid` (Proposal 003 Phase 0.1) so the
/// upcoming `katgpt-band` extraction doesn't drag a math utility into the band
/// crate. Per the project rule: sigmoid, never softmax.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[cfg(feature = "tiled_attention")]
pub mod attention;

// Newton-Schulz orthogonalization + Muon momentum (Plan 152, Research 114,
// GOAT 25/25 Bench 050). Pure substrate — self-contained f32 linear algebra,
// zero crate:: deps, zero external deps. Extracted from katgpt-rs/src/ per
// Issue 355 Phase 1a. Re-exported by katgpt-rs root so historical
// `katgpt_rs::newton_schulz::*` paths continue to resolve.
#[cfg(feature = "newton_schulz")]
pub mod newton_schulz;

// linking_fold — Linking-Number Detector + Fold Correction (Plan 410,
// Research 391, arXiv:2606.31856 Ren & Lim ICML 2026). SPLIT (Plan 410 T4.4
// Option C, 2026-07-07) into two independently-gated sub-features:
//   - linking_fold_fold     (hot-path |x−c| fold correction) — DEFAULT-ON
//   - linking_fold_detector (cold-path Algorithm-1 linking detector) — opt-in
//   - linking_fold          (umbrella = fold + detector) — opt-in
// The fold passes every GOAT gate modellessly and ships default-on; the
// detector's G2 budget is the audit-cadence-appropriate 500 ms @ n=2×200
// (Issue 050 Option A, resolved 2026-07-07) and it stays opt-in. The
// module root exists when EITHER sub-feature is on; submodules gate their own
// parts.
#[cfg(any(feature = "linking_fold_fold", feature = "linking_fold_detector"))]
pub mod linking_fold;
// best_belief — ε-quantile Beta lower bound for conservative selection
// (Plan 336, Research 320, RQGM arXiv:2606.26294 Prop. 4). Complements
// `sample_beta` (Thompson sampling for EXPLORATION) with a conservative
// EXPLOITATION / SELECTION counterpart. Opt-in until the G1+G2+G4 GOAT gate
// passes.
#[cfg(feature = "best_belief")]
pub mod best_belief;
#[cfg(feature = "best_belief")]
pub use best_belief::{best_belief_score, best_belief_scores, select_best_belief};
// Conformal Predictive Intervals — modelless UQ overlay (Plan 340, Research
// 322, arXiv:2605.03789 CSP + arXiv:2606.09473 "Report the Floor"). Wraps any
// PointForecaster with a per-channel × per-horizon-bucket exp-recency-
// weighted residual ring buffer, reads empirical quantiles to produce
// coverage-guaranteed predictive intervals. The
// ConformalIntervalCalibrator<SeasonalNaiveForecaster> with m=1 is the
// canonical conformal-naive floor per the "Report the Floor" rule (Issue 010,
// AGENTS.md Feature Flag Discipline). Opt-in until G1–G4 GOAT gate passes.
#[cfg(feature = "conformal_predictive_intervals")]
pub mod conformal;
#[cfg(feature = "conformal_predictive_intervals")]
pub use conformal::metrics::{
    crps, crps_interval, empirical_coverage, mean_crps_interval, mean_winkler, winkler_score,
};
#[cfg(feature = "conformal_predictive_intervals")]
pub use conformal::{
    ConformalIntervalCalibrator, DecayUnit, PointForecaster, PredictiveInterval, ResidualMode,
    ResidualRingBuffer, RingBuffer, SeasonalNaiveForecaster, SeasonalPoolForecaster,
    seasonal_naive_floor,
};
// Plan 340 Phase 2 (T2.1) — KARC adapter for the conformal overlay.
// Gated on BOTH features: needs the conformal substrate AND the KARC forecaster.
#[cfg(all(
    feature = "conformal_predictive_intervals",
    feature = "karc_forecaster"
))]
pub use conformal::KarcChannelForecaster;
// Issue 010 T2 — "Report the Floor" comparison harness. Re-exported for
// T3–T7 (BoMSampler, Sleep-Time, Best-Belief, Alien Sampler adapters).
#[cfg(feature = "conformal_predictive_intervals")]
pub use conformal::{
    FloorAdapter, FloorComparisonReport, OverallVerdict, PredictiveOutput, TrajectoryCorpus,
    UqMetrics, UqPrimitiveUnderTest, empirical_quantile_interval, run_floor_comparison,
};
#[cfg(feature = "coda_fusion")]
pub mod coda;
#[cfg(feature = "dec_operators")]
pub use katgpt_dec as dec;
pub mod delta_mem;
// Higher-order Linear Attention (HLA) substrate — cache types + streaming
// kernels. Spun out to the `katgpt-hla` crate (Issue 007 Phase E Tier 2 #4)
// and re-exported here as `katgpt_core::hla` for backwards compatibility.
// All `crate::hla::*` and `katgpt_core::hla::*` paths resolve unchanged. The
// composition layer (`forward_hla` / `forward_ahla`, depends on ForwardContext)
// stays in katgpt-core; the cognitive role-aware variants stay in riir-engine.
pub use katgpt_hla as hla;
// Shared leaky-integrator primitive. Spun out to the `katgpt-types` leaf
// (Issue 007 Phase E Tier 1 #3) so both katgpt-core (`sense::reconstruction`)
// and `katgpt-micro-belief` (`LeakyIntegrator::step`) can consume it without
// a cycle. Re-exported here as `katgpt_core::leaky_core` for backwards
// compatibility.
pub use katgpt_types::leaky_core;
/// Generic Monte Carlo Tree Search over any [`crate::traits::GameState`].
///
/// Always-on substrate. Composition that needs root-only types
/// (`BanditRolloutPolicy` depends on `crate::pruners::bandit::BanditStats`)
/// stays in the consuming crate.
pub mod mcts;
/// State-Action Pair Cache for MCTS over Deterministic Inference Actions
/// (Plan 390, Research 386, arXiv:2602.04344 UnMaskFork).
///
/// Opt-in extension to [`mcts`]: a standalone search over an opaque
/// `InferenceActionSpace` (no `GameState` / game-IP coupling), backed by a
/// lock-free `StateActionCache` keyed on `(blake3::Hash, InferenceAction)`.
/// Gated behind `mcts_state_action_cache` so the always-on `mcts` substrate
/// stays dep-free.
#[cfg(feature = "mcts_state_action_cache")]
pub mod mcts_state_action_cache;
// Shared freeze/thaw disk I/O for `repr(C)` knowledge structs.
// Extracted from `katgpt-pruners::freeze` (Plan 388 Phase 1) to break the
// katgpt-pruners ↔ katgpt-speculative cycle. Pure stdlib (Path + fs + mem).
// Re-exported by katgpt-pruners::freeze for backwards compatibility.
pub mod freeze;
// Proof goal deduplication cache core types (GoalHash, GoalResult,
// GoalVerifier, ProofGoalCache). Extracted from `katgpt-pruners::proof::goal_cache`
// (Plan 388 Phase 2) to break the katgpt-pruners ↔ katgpt-speculative cycle.
// Pure substrate (blake3 + HashMap + AtomicU64). Re-exported by
// katgpt-pruners::proof::goal_cache for backwards compatibility.
pub mod proof_cache;
// Per-query thinking mode tag. Extracted from `katgpt-pruners` (Plan 388
// Phase 3) to break the katgpt-pruners ↔ katgpt-speculative cycle. Pure
// 4-variant `#[repr(u8)]` enum, no pruners-specific knowledge. Re-exported
// by katgpt-pruners and katgpt_rs::speculative for backwards compatibility.
#[cfg(feature = "parallax_attn")]
pub mod parallax_attn;
pub mod thinking_mode;
// Algebraic-structure primitives. Currently home to the tropical (max, +)
// semiring (Plan 337, Research 321). Opt-in via `tropical_algebra`.
#[cfg(feature = "tropical_algebra")]
pub mod algebra;
pub mod shard_embedding;
// SSMax — length-aware log-N attention temperature (Plan 411, Research 392,
// arxiv 2607.01538 Gollapudi et al. *Drowning in Documents at Million Token
// Scale*). Multiplicative pre-attention logit rescaling that cancels the
// attention dilution at large N. Default `s_L = 1.0` is truly modelless.
// Composes with parallax_attn (sigmoid) and attention.rs (SDPA); does NOT
// apply to funcattn (Research 261 closed negative: basis-mode has no (n,n)
// attention matrix → no dilution). Opt-in until G1+G2 GOAT gate passes.
#[cfg(feature = "ssmax_temperature")]
pub mod ssmax;
// Position-Offset Reveal-Time Schedule for Set Diffusion (Research 376).
// Canonical source for `PositionOffsetSchedule` — pure math (CDF/inverse-CDF/
// ordering), RNG-agnostic via closure-based sampling. No feature gate because
// it's a zero-dep math substrate consumed by both katgpt-rs (runtime) and
// riir-train (training). Eliminates the 3-way DRY violation that previously
// had copies in katgpt-rs/src/dllm.rs, riir-train/.../set_diffusion_schedule.rs,
// and riir-ai/crates/riir-poc/.
pub mod set_diffusion_schedule;
pub use set_diffusion_schedule::{
    PositionOffsetSchedule, ar_order, block_causal_gen_steps, mdlm_gen_steps, order_to_gen_steps,
    uniform_order, uniform_order_with,
};
// SIMD-accelerated linear algebra kernels (NEON / AVX2 / WASM-SIMD128 /
// scalar fallback). Spun out to the `katgpt-types` crate (Issue 007 Phase E
// Tier 1 #2) and re-exported here as `katgpt_core::simd` for backwards
// compatibility. All `crate::simd::*` paths resolve unchanged.
pub use katgpt_types::simd;
pub mod speculative;
pub mod traits;
// Shared configuration, RNG, math utilities, LoRA, domain embeddings, and
// inference types. Spun out to the `katgpt-types` crate (Issue 007 Phase E
// Tier 1 #2) and re-exported here as `katgpt_core::types` for backwards
// compatibility. All `crate::types::*` paths resolve unchanged.
pub use katgpt_types as types;

// CGSP — Curiosity-Guided Self-Play modelless triad (Plan 274, Research 240).
// Self-contained: Direction/Target/Candidate, CgspLoop, PoolConjecturer,
// HlaProjectionGuide, BreakevenDifficultyFilter, ColinearityBatchGate,
// EntropyCollapse, CuriosityPrioritySnapshot (BLAKE3-committed).
// Consumed by riir-engine Plan 299 (NPC curiosity runtime).
#[cfg(feature = "cgsp")]
pub mod cgsp;
#[cfg(feature = "cgsp")]
pub use cgsp::{
    BatchQualityGate,
    BreakevenDifficultyFilter,
    Candidate,
    CgspConfig,
    CgspLoop,
    ColinearityBatchGate,
    CollapseSignal,
    ComplexityWeights,
    CuriosityConjecturer,
    CuriosityPrioritySnapshot,
    CycleResult,
    CycleStats,
    DEFAULT_HLA_DIM,
    DEFAULT_K,
    DEFAULT_POOL_SIZE,
    DifficultyFilter,
    Direction,
    EntropyCollapse,
    HintDeltaBandit,
    HlaProjectionGuide,
    NoOpBatchGate,
    NoOpDifficultyFilter,
    PoolConjecturer,
    Priority,
    QualityGuide,
    ScratchBuffers,
    SolveRate,
    Solver,
    Target,
    entropy_nats,
    structural_complexity,
    // Note: `sigmoid` is no longer re-exported here — it's now an always-on
    // top-level `katgpt_core::sigmoid` (Proposal 003 Phase 0.1). The module-local
    // `katgpt_core::cgsp::sigmoid` (in cgsp/types.rs) remains for `cgsp::*` paths.
};

// CGSP dual-pool extension — DecentMem distillation (Plan 282, Research 249).
#[cfg(feature = "cgsp_dual_pool")]
pub use cgsp::{DualPoolBandit, DualPoolConfig, PoolId, ReachableDualPoolRouter};

// Issue 364 T4 — modelless k_npc selector (wraps GainCostLoopHalter, Plan 304).
// Needs both cgsp (the host module) and gain_cost_halt (the halter kernel).
// Consumed by riir-ai's per-NPC CLR cadence wiring (Phase 30 of tick_map).
#[cfg(all(feature = "cgsp", feature = "gain_cost_halt"))]
pub use cgsp::{KnpcDecision, KnpcSelector};

// ActionBridge — generic latent→raw action bridge (Plan 262).
#[cfg(feature = "action_bridge")]
pub mod bridge;
#[cfg(feature = "action_bridge")]
pub use bridge::ActionBridge;

// Re-export consolidated traits (Plan 107 Phase 0)
pub use traits::{
    ActionSpaceLog, BestBuddyAligner, BinaryScreeningPruner, ConstraintPruner, DominoPruner,
    FeatureClass, GameState, NoPruner, NoScreeningPruner, RandomRolloutPolicy, RolloutPolicy,
    ScreeningPruner, StateHeuristic, best_buddies, pearson_correlation,
};
pub use traits::{GenerativeConstraintPruner, SpeculativeGenerator};

// RecursionLogits — opt-in trait for generators that expose pre/post recursion
// logits so AdvantageMarginGate can wrap them (Plan 283 T2.3, arxiv:2511.16886).
// Opt-in: not in default feature list. Non-recursing generators do not implement it.
#[cfg(feature = "recursion_logits")]
pub use traits::RecursionLogits;

// Q-Guided Flow (Plan 268) — test-time Q-gradient guidance primitive.
#[cfg(feature = "qgf_oracle")]
pub use traits::{NoGuidanceOracle, QGradientOracle};
#[cfg(feature = "qgf")]
pub mod qgf;

// MicroRecurrentBeliefState — per-entity recurrent state kernel (Plan 276, Research 242).
// Trait + Family A (attractor) + Family C (leaky) + BLAKE3 snapshot + sigmoid bridge.
// Spun out to the `katgpt-micro-belief` crate (Issue 007 Phase E Tier 1 #3) and
// re-exported here as `katgpt_core::micro_belief` for backwards compatibility.
#[cfg(feature = "micro_belief")]
pub use katgpt_micro_belief as micro_belief;
#[cfg(feature = "micro_belief")]
pub use micro_belief::{
    AttractorKernel, KernelConfig, LeakyIntegrator, MicroRecurrentBeliefState,
    MicroRecurrentKernelSnapshot, RecurrenceFamily, SNAPSHOT_VERSION, project_to_scalars,
};

// BoMSampler — K-hypothesis single-pass belief sampling (Plan 281, Research 248).
// Opt-in extension of MicroRecurrentBeliefState; gated on bom_sampling which implies micro_belief.
#[cfg(feature = "bom_sampling")]
pub use micro_belief::{BoMSampler, NoiseQueryConfig, QmcMethod, SeedStrategy, dot_product_scorer};

// Plan 370 — QMC noise-fill convenience entry point (constructs the right
// QmcSource from a QmcMethod tag + seed, zero-alloc). Used by
// MultiHypothesisBoMMinimaxPlanner when NoiseQueryConfig::qmc_method is Some.
#[cfg(all(feature = "qmc_sampling", feature = "bom_sampling"))]
pub use speculative::fill_noise_queries_gaussian_qmc_by_method;

// BoM G2 arena harness — Plan 281 T2.3.
// Engine-side traits + synthetic reference env. riir-ai implements the traits
// over a real bomber/go sim to produce the empirical G2 gate.
#[cfg(feature = "bom_sampling")]
pub use micro_belief::{
    ArenaAction, ArenaEnvironment, BeliefPlanner, BoMMeanPlanner, BoMMinimaxPlanner,
    ComparisonResult, DeterministicPlanner, EnvHint, PlannerOutcome, SyntheticThreatArena,
    bom_mean_attractor, bom_minimax_attractor, bom_minimax_leaky, run_arena_comparison,
};

// FaithfulnessProbe — causal intervention diagnostic for injected memory (Plan 278, Research 244).
// Moved from katgpt root to katgpt-core so riir-engine (Plan 308) can consume via katgpt-core.
// Two features:
// - `triggered_injection` (default-ON after GOAT G3): sigmoid-thresholded inject/skip hot-path gate.
// - `faithfulness_probe` (opt-in, audit cadence): full intervention suite + perturbation + attribution.
// The module is compiled when EITHER feature is on; submodules are individually gated in `mod.rs`.
#[cfg(any(feature = "faithfulness_probe", feature = "triggered_injection"))]
pub mod faithfulness;

// Pruners module (Plan 054 review_metrics + Plan 320 indicator_probe_bank, etc.).
// Parent module is always compiled; individual submodules gate their own features.
// (Previously the whole `pruners` module was gated behind `review_metrics`; that
// coupling was broken in Plan 320 so indicator_probe_bank can gate independently.)
pub mod pruners;

// Temporal Derivative Kernel — dual fast/slow EMA surprise signal (Plan 277, Research 243).
// Turns any streaming latent vector into a signed "surprise" signal — the implicit
// prediction-error channel for credit assignment, computed locally with no backprop.
// Opt-in until ≥2 fusion gates (G2–G5) pass.
#[cfg(feature = "temporal_deriv")]
pub mod temporal_deriv;
#[cfg(feature = "temporal_deriv")]
pub use temporal_deriv::{TemporalDerivativeKernel, sigmoid_surprise_gate};

// HOLA Hippocampal Exact KV Cache — surprise-evicted (β·‖e‖) bounded KV cache with
// decoupled RMSNorm-γ read (Plan 395, Research 378, arxiv 2607.02303). Complements
// the GDN2 fixed-size recurrent state with a top-w exact KV set for long-range
// retrieval. Opt-in until G1–G4 GOAT gate passes. Pure stdlib + katgpt-types.
#[cfg(feature = "hippocampal_cache")]
pub mod hippocampal_cache;
#[cfg(feature = "hippocampal_cache")]
pub use hippocampal_cache::{HippocampalCache, SortedSlotCache};

// Tiered Hot/Warm/Cold K/V Store — the route-and-fetch substrate for sparse
// long-context attention (Plan 397, Research 379, arxiv 2606.30709). Generic
// trait + in-memory reference impl. Always-on (no feature gate) because it's
// a generic primitive with no attention-layer deps; the HGA-specific consumer
// is gated by `hga`.
pub mod tiered_kv;

// Hierarchical Global Attention (HGA) — chunk→group→token routing with
// RoPE-aware mixed-frequency summaries (Plan 397, Research 379, arxiv 2606.30709).
// Three refinements of the sparse-attention routing slot: group middle tier,
// mixed-RoPE summarizer, tiered route-and-fetch consumer. Opt-in until the
// Phase 2 GOAT gate (G2 head-to-head vs DashAttention) passes.
//
// NOTE: the HGA forward path (which needs dash_attn::entmax_1p5) lives in
// katgpt-attn/src/hga_forward.rs, not here — katgpt-core cannot import
// katgpt-attn without a circular dependency.
#[cfg(feature = "hga")]
pub mod hga;
#[cfg(feature = "hga")]
pub use hga::{GroupSummaryCache, MixedRopeSummarizer};

// Renoise-CE Self-Verifier — perturb a completed state, re-resolve through the
// same operator, measure drift as a verifier-free correctness score (Plan 406,
// Research 369, arxiv 2606.29150). Third orthogonal self-eval signal alongside
// CLR (claim-vote) and CoE (trajectory-shape). Operator-agnostic trait over any
// state->state map. Opt-in until G1+G2 GOAT gate passes. NOT a UQ primitive
// (raw ranking signal; conformal wrapping required for any UQ claim).
#[cfg(feature = "renoise_ce")]
pub mod renoise_ce;
#[cfg(feature = "renoise_ce")]
pub use renoise_ce::{
    Proposer, RenoiseCeConfig, RenoiseCeProbe, RenoiseCeScore, best_of_n_stability,
    renoise_ce_score, verify_and_restart,
};

#[cfg(feature = "dual_leo")]
pub use traits::{
    ActingMode, AlphaSchedule, AutocurriculumSampler, BcConfig, BcTarget, DualLeoMixer,
};
#[cfg(feature = "leo_all_goals")]
pub use traits::{AllGoalsUpdate, LeoHead, sigmoid_bounded_q};

// Re-export key types at crate root for convenience
pub use shard_embedding::{EMBED_DIM, JlProjectionMatrix, STYLE_DIM as JL_STYLE_DIM};
#[allow(deprecated)]
pub use types::sample_token;
pub use types::{
    AttentionMode, AttentionProjection, CacheLayout, CalibrationMode, Config, ConvergenceSelector,
    DashAttnConfig, DilationConfig, HlaMode, HybridPattern, InferenceOverrides, InferenceResult,
    LoopMode, LoraAdapter, LoraPair, ModelArchitecture, ResidualGate, RetrievalHeadRole, Rng,
    RtTurboConfig, SdpaOutputGate, ShardEmbedding, WeightDtype, kv_dim, lora_apply, matmul,
    matmul_f16, matmul_f16_parallel, matmul_parallel, matmul_relu, rmsnorm, sample_token_into,
    softmax, softmax_scaled,
};

#[cfg(feature = "domain_latent")]
pub use types::DomainLatent;

#[cfg(feature = "sr2am_configurator")]
pub use types::{ConfiguratorContext, PlanningDecision};

#[cfg(feature = "data_gate")]
pub use types::{DataGate, GateDecision, ProposerTask, TaskType};

#[cfg(feature = "sparse_mlp")]
pub use types::sparse_matmul;

#[cfg(feature = "coda_fusion")]
pub use coda::{
    GateActivation, MoaConfig, compute_rstd, simd_matmul_residual,
    simd_matmul_residual_partial_rms, simd_matmul_rmsnorm_activation, simd_matmul_rmsnorm_rope,
    simd_matmul_rmsnorm_swiglu,
};

#[cfg(all(feature = "coda_fusion", feature = "moa_inference"))]
pub use coda::{MoaActivation, moa_swiglu, simd_matmul_rmsnorm_moa_swiglu};

#[cfg(feature = "tiled_attention")]
pub use attention::{
    tiled_attention_batched, tiled_attention_forward, tiled_attention_forward_with_scores,
};

#[cfg(feature = "parallax_attn")]
pub use parallax_attn::{
    ParallaxActivation, ParallaxConfig, ParallaxScratch, compute_rho, parallax_correction,
    tiled_attention_parallax_forward, tiled_attention_parallax_forward_retaining,
};

// Sink-aware composition (Plan 289). Requires both parallax_attn (for the
// forward) and sink_aware_attn (for the classifier + flat gate). The
// `tiled_attention_parallax_forward_sink_aware` entry point short-circuits to
// vanilla parallax when policy = Uniform, so this is a zero-cost abstraction
// for callers who construct the scratch but never enable DualPolicy.
#[cfg(all(feature = "parallax_attn", feature = "sink_aware_attn"))]
pub use parallax_attn::{SinkAwareParallaxScratch, tiled_attention_parallax_forward_sink_aware};

pub use simd::SimdLevel;

#[cfg(feature = "hydra_budget")]
pub use types::{HydraBudgetConfig, HydraLayerProfile};

#[cfg(feature = "collapse_aware_thinking")]
pub use types::ThinkingBudget;

#[cfg(feature = "questbench")]
pub mod questbench;
#[cfg(feature = "questbench")]
pub use questbench::{
    CspDomain, MemoryTier, QuestBenchDecision, SyntheticCsp, UnderspecConfig, find_sufficient_set,
    generate_synthetic_csps, tier_from_score, underspecification_score,
};

#[cfg(feature = "tf_loop")]
pub use types::{CacheStrategy, IterationMode, SubStepStrategy, TrainingFreeLoopConfig};

#[cfg(feature = "plasma_path")]
pub use simd::{simd_ternary_matmul_batch, simd_ternary_matvec, ternary_matvec_scalar};
#[cfg(feature = "plasma_path")]
pub use types::TernaryWeights;

#[cfg(feature = "peira_distill")]
pub mod peira;
#[cfg(feature = "peira_distill")]
pub use peira::{PeiraConfig, PeiraCovariance, peira_aux_loss};

#[cfg(feature = "dirichlet_energy")]
pub mod dirichlet;
#[cfg(feature = "dirichlet_energy")]
pub use dirichlet::{
    consecutive_adjacency, dirichlet_energy, functor_adjacency, kv_cache_dirichlet_energy,
};

#[cfg(feature = "spectral_hierarchy")]
pub mod spectral_hierarchy;
#[cfg(feature = "spectral_hierarchy")]
pub use spectral_hierarchy::{cauchy_interlacing_check, eigenspace_alignment, haar_wavelet_basis};

#[cfg(feature = "sigmoid_margin")]
pub use simd::{compute_retrieval_margin, dim_sufficiency_bound, sigmoid_margin_loss};

#[cfg(feature = "dual_gram_pca")]
pub use simd::simd_gram_f32;

#[cfg(feature = "roofline_cost")]
pub mod roofline;
#[cfg(feature = "roofline_cost")]
pub use roofline::{
    ComputeBound, Dtype, HardwarePeaks, OpType, RooflineCost, gemm_cost, gemv_cost, gram_cost,
    roofline_estimate,
};

#[cfg(feature = "ane_roofline")]
pub mod ane_roofline;
#[cfg(feature = "ane_roofline")]
pub use ane_roofline::{
    AneBound, AneCost, AneFamily, AneOpShape, AnePeaks, Device, ane_conv3x3_cost, ane_estimate,
    ane_gemm_cost, ane_gemv_cost,
};

#[cfg(feature = "and_or_dtree")]
pub mod and_or;
#[cfg(feature = "and_or_dtree")]
pub use and_or::AndOrNode;

#[cfg(feature = "partial_scoring")]
pub use traits::{GameTrace, PartialScorer};

#[cfg(feature = "problem_mutator")]
pub use traits::{GameConfig, MutantConfig, MutationKind, ProblemMutator};

#[cfg(feature = "modal_spec")]
pub mod linoss;
#[cfg(feature = "mux_pruner")]
pub mod mux;

// Sense substrate was spun out to the `katgpt-sense` crate (Issue 007 Phase E
// Tier 2 #7, Plan 338). `spectral_threat` stayed local (depends on `linoss`);
// it lives at `crate::sense_threat` and is re-exported through the
// `sense::spectral_threat` shim path below to preserve external consumers'
// `katgpt_core::sense::spectral_threat::*` paths bit-for-bit.
#[cfg(feature = "sense_composition")]
pub mod sense {
    pub use katgpt_sense::*;
    #[cfg(feature = "spectral_threat")]
    pub mod spectral_threat {
        pub use crate::sense_threat::*;
    }
}
#[cfg(feature = "spectral_threat")]
pub mod sense_threat;

#[cfg(feature = "slod")]
pub mod slod;
#[cfg(feature = "slod")]
pub use slod::{
    ScaleBoundary, SlodConfig, SlodOperator, SlodPruner, exp_map, frechet_mean,
    heat_kernel_weights, log_map, poincare_distance,
};

// Spectral Irrep Pruner - spectral flatness-based speculative decoding pruning (Plan 246).
// Prunes tokens when logit spectrum shows competing modes (high spectral flatness).
// GOAT PASS: +3.6% overhead, default-ON.
#[cfg(feature = "spectral_pruner")]
pub mod irrep_pruner;

// Subspace phase-gate primitives — participation ratio, numerical rank, N≥d
// phase-transition gate (Wang et al. Thm 4, arXiv:2409.02426), and runtime
// Jacobian SVD via forward differences (Plan 301, Research 279). Pure numeric,
// no game/shard/chain semantics. Consumers (riir-neuron-db Plan 002, future
// riir-ai HLA self-discovery plan) apply these to their own maps.
// Opt-in until G1 GOAT gate passes.
#[cfg(feature = "subspace_phase_gate")]
pub mod subspace_phase_gate;

// Group Invariance Probe — modelless symmetry discovery on a hypothesis Lie
// group (Plan 356, Research 355 — distilled from LieFlow, arXiv:2512.20043).
// Generalizes `subspace_phase_gate` from "subspace of R^d" to "subgroup of G":
// score each sampled g ∈ G by direct invariance testing σ(β·(1−d(q, g·q))),
// then classify the discovered H as Discrete / Continuous / Partial / None via
// a participation-ratio-style concentration measure on the score histogram.
// Pure numeric, no game/shard/chain semantics, zero deps. Sibling of
// `subspace_phase_gate`. Opt-in until G1 GOAT gate passes (Plan 356 Phase 1).
#[cfg(feature = "group_invariance_probe")]
pub mod group_invariance_probe;

// Latent Trajectory Geometry — probe-free geometric diagnostic (length +
// mean turning-angle curvature + min adjacent cosine + bifurcation ratio).
// Distilled from Pandey et al., arXiv:2606.09287 (Plan 342, Research 324).
// Pure numeric over `&[&[f32]]`, no extra deps. Opt-in until the Phase 3
// game-related gate (curvature catches the oscillation failure mode entropy
// misses) passes; promotion to a routing role is a separate follow-up plan.
#[cfg(feature = "latent_trajectory_geometry")]
pub mod latent_trajectory_geometry;

// Viable Manifold Graph — discrete safe-manifold navigation primitive.
// Distillation of arXiv:2206.00106 (González-Duque et al., *Mario Plays on a
// Manifold*, 2022). Generic over any smooth map `f: R^n → R^m` (closure) and
// a viability predicate `V(z)`. Computes the pullback volume field
// `log det(J_f^T J_f)` (via Plan 301's `jacobian_svd_at`), filters a latent
// sample to a discrete safe-manifold subgraph, and runs A* / random-walk
// navigation that stays inside the viable set by construction. Game / shard /
// chain wiring lives in riir-ai (R154). Opt-in until G1–G6 GOAT gates pass.
#[cfg(feature = "viable_manifold_graph")]
pub mod viable_manifold_graph;

// Zone Affective Manifold — crowd-scale PCA via power iteration + deflation
// on the (N, D) crowd-activation covariance (Issue 001). Top-k principal
// directions ("zone mood axes") + per-NPC projections. Rayon-parallel for
// N > parallel_threshold, cold-start identity fallback for small crowds,
// sign-fixed for temporal continuity. Pure modelless. Opt-in until G1-G6 pass.
#[cfg(feature = "zone_affective_manifold")]
pub mod zone_manifold;

// Zone Density Routing — modelless per-zone physical compute scheduler (Plan
// 351, Research 350 — Treuille Continuum Crowds + Fokker-Planck-on-cochains).
// Three primitives: zone_density_classify (mobility = fast_sigmoid(-β·(ρ−ρ₀))
// → tier + composite cache_key), schedule_outer_first (stable ascending-density
// sort — outer/sparse zones compute first), ZoneDensityCache<V> (papaya-backed
// LRU with tier-transition / density-drift / TTL invalidation rules). Sibling to
// Plan 305 cognitive gating (Plan 305 gates learning compute; this gates
// movement compute) — they compose orthogonally, NOT overlap. Population is
// raw/synced; mobility/tier/cache_key are latent/local. Opt-in until G5a
// (Shannon entropy ≥+15% vs mean-agg) + G5b (≥50% compute saved on dense-dominated)
// + G5c (zero stale reads during stampede) all pass. No UQ claim — mobility is
// a deterministic [0,1] weight, not a probability/interval/coverage.
#[cfg(feature = "zone_density_routing")]
pub mod zone_density;

// AC-GPT Arbitrary-Conditional Prefix — modelless mask builder + sequence
// augmenter that turns any causal Transformer forward into a single-pass
// arbitrary-conditional forward p(xe | xc) via position-aware copies of xc at
// the front and a [xc-bidirectional | causal-everywhere-else] attention mask
// (Lu et al., Mila, arXiv:2606.14943, Plan 313, Research 295). Phase 1 ships
// types + bit math only — no attention kernel dep, no SVD. Opt-in until G1–G4
// GOAT gates pass.
#[cfg(feature = "ac_prefix")]
pub mod ac_prefix;

// Causal Head-Importance Calibration & Scale-Normalized Heterogeneous Fusion
// (Plan 358, Research 362, arXiv:2606.20097 HydraHead). Modelless
// causal-intervention head scorer: activation patching (Eq 10) + path patching
// (Eq 11) + span-level logit-diff readout (Eq 9) + cross-capability fusion
// (Eq 12) + head partition mirroring RTPurbo's HeadCalibration. Plus
// scale-normalized heterogeneous-branch fusion (Eq 13–14, currently unused).
// Pure numeric over `&[f32]` + a caller-supplied patched-forward-pass closure;
// the patched forward pass itself lives in riir-engine. Sibling of
// `faithfulness_probe` (causal-intervention diagnostic pattern). Opt-in until
// G1–G4 GOAT gate passes; competes for the RTPurbo calibration slot.
#[cfg(feature = "causal_head_importance")]
pub mod causal_head_importance;
#[cfg(feature = "spectral_pruner")]
pub use irrep_pruner::{
    IrrepPruner, IrrepPrunerConfig, irrep_pruner_from_config, spectral_flatness,
};
#[cfg(feature = "subspace_phase_gate")]
pub use subspace_phase_gate::{
    IntrinsicDimMethod, JacobianSvdScratch, SvdResult, SvdResultScratch, SvdScratch,
    estimate_intrinsic_dim, jacobian_svd_at, jacobian_svd_at_into, numerical_rank,
    participation_ratio, phase_transition_gate, thin_svd, thin_svd_into,
};

#[cfg(feature = "group_invariance_probe")]
pub use group_invariance_probe::{
    GroupAction, SubgroupClass, SubgroupReport, classify_subgroup, classify_subgroup_with,
    discover_subgroup, discover_subgroup_into, invariance_score, score_concentration,
    score_variance,
};

#[cfg(feature = "causal_head_importance")]
pub use causal_head_importance::{
    ScaleNormalizedFusion, SpanLogitDiffReadout, direct_effect_importance,
    fuse_across_capabilities, indirect_effect_importance, partition_by_causal_score,
    per_capability_score,
};
// Adaptive Causal Calibration (Proposal 004) — cheap-proxy escalate. Opt-in.
// Re-exported alongside the causal head-importance primitives it builds on.
#[cfg(feature = "adaptive_causal_calibration")]
pub use causal_head_importance::{adaptive_partition, suspect_indices};

#[cfg(feature = "latent_trajectory_geometry")]
pub use latent_trajectory_geometry::{
    BifurcationResult, LatentTrajectoryGeometry, bifurcation_ratio, fast_acos, from_states,
};

#[cfg(feature = "viable_manifold_graph")]
pub use viable_manifold_graph::{
    ClosurePredicate, GraphBuildConfig, SafeManifoldGraph, ViabilityPredicate, VolumeFieldConfig,
    build_safe_manifold_graph, manifold_curiosity_walk, manifold_geodesic, manifold_random_walk,
    pullback_volume,
};

#[cfg(feature = "ac_prefix")]
pub use ac_prefix::{AcPrefix, AcPrefixMask};

#[cfg(feature = "flow_field_nav")]
pub mod flow;
#[cfg(feature = "flow_field_nav")]
pub use flow::{
    FlowField, FlowFieldCache, FlowFieldConfig, LeoPotentialGrid, blend_steering, fft_smooth,
    fft_smooth_into, flow_steering, inflate_obstacles, should_use_flow_field,
};

// Spectral primitives — Fourier-basis algebra on discrete samples.
// Distilled from the FNO practical-perspective survey (Research 307).
// Each operator ships behind its own feature flag and is independently GOAT-gated.
// - `continuation` (feature `fourier_continuation`, Plan 323): Fourier
//   continuation for non-periodic latent fields — closed-form polynomial
//   periodic extension so the FFT does not produce Gibbs ringing at the
//   boundaries. The one modelless FNO primitive the codebase genuinely
//   lacked (Research 307 §3 candidate plan #1). Opt-in until G1–G4 pass.
// - `differentiation` (feature `spectral_differentiation`, Plan 325):
//   standalone FFT-based spectral differentiation on periodic uniform 1D
//   grids — multiply FFT coefficients by `(iω)^m`, IFFT back. The
//   specialized 1D-periodic case where DEC `exterior_derivative` is
//   overkill. Opt-in until G1–G4 pass.
#[cfg(any(feature = "fourier_continuation", feature = "spectral_differentiation"))]
pub mod spectral;
#[cfg(feature = "fourier_continuation")]
pub use spectral::continuation::{
    FcConfig, FcScratch, FourierContinuationError, MAX_POLY_ORDER, fourier_continue,
    fourier_continue_into,
};
#[cfg(feature = "spectral_differentiation")]
pub use spectral::differentiation::{
    MAX_ORDER, SpecDiffConfig, SpecDiffError, SpecDiffScratch, spectral_differentiate,
    spectral_differentiate_into,
};

// Merkle octree — hierarchical BLAKE3 commitment for KG latent octree nodes (Plan 221-M).
#[cfg(feature = "merkle_octree")]
pub mod merkle;
#[cfg(feature = "merkle_octree")]
pub use merkle::{
    HASH_SIZE, MERKLE_OCTREE_DEPTH, MERKLE_OCTREE_INTERNAL, MERKLE_OCTREE_LEAVES,
    MERKLE_OCTREE_NODES, MerkleOctree, MerkleProof,
};

// Curator verification layer for Merkle octree (Plan 253).
#[cfg(feature = "merkle_octree")]
pub mod curator;
#[cfg(feature = "merkle_octree")]
pub use curator::{
    CuratorArm, CuratorBandit, CuratorVerdict, FrozenTarget, MerkleEnvelope, MerkleFrozenStore,
    verification_weight,
};

// RTDC — Resolution-Tiered Deterministic Commitment (Plan 302, Research 280).
// Wraps `MerkleOctree` with 3 per-depth roots aligned to SLoD σ-boundaries,
// enabling trust-minimized semantic zoom: a light client verifies its
// fog-of-war view is a faithful sub-summation of the chain-committed full KG,
// with O(log n) proof at the abstraction level it operates at.
//
// Phase 1 ships the open primitive (types + trait + depth-2 sound proofs).
// Cross-depth soundness (`subtree_inclusion`) is Phase 3: Candidate C
// (probabilistic sampling) shipped behind `rtdc_subtree_inclusion`.
// Candidate A (Pedersen deterministic) research closed dormant — see
// `riir-chain/.research/006_RTDC_Candidate_A_Pedersen_Resolution.md`.
// LatCal-backed `DeterministicLeafEncode` impl lives in riir-chain (Plan 003).
#[cfg(feature = "rtdc")]
pub mod rtdc;
#[cfg(feature = "rtdc")]
pub use rtdc::{
    DepthSelector, DepthTieredMerkleOctree, DepthTieredRoots, DeterministicLeafEncode, RtdcError,
    RtdcProof,
};
#[cfg(feature = "rtdc_subtree_inclusion")]
pub use rtdc::{RTDC_SUBTREE_DEFAULT_K, SubtreeProof, min_k_for_95pct_confidence};

// GPart isometric partition adapter — replaces LoRA's bilinear BA with single isometric Pθ_d (Plan 257).
#[cfg(feature = "gpart_adapter")]
pub use types::{GPART_MAGIC, GPART_VERSION, GpartAdapter, GpartPair, GpartPrepared};

#[cfg(feature = "dendritic_gate")]
pub mod dendritic_gate;
#[cfg(feature = "dendritic_gate")]
pub use dendritic_gate::{DendriticGate, dendritic_sigmoid};
#[cfg(feature = "dendritic_gate")]
pub use simd::{coincidence_score, entropy_f32};

// CompressionDrafter — Hot-tier modelless LZ4 corpus-as-model drafter (Plan 285,
// Research 256, nathan.rs/gzip-lm). The compressor IS the model: score candidate
// continuations by compressed length against a frozen corpus. Corpus is appendable
// for online learning and is itself the wired format (bytes + BLAKE3).
// Opt-in until G1–G3 GOAT gate passes.
#[cfg(feature = "compression_drafter")]
pub mod compression_drafter;
#[cfg(feature = "compression_drafter")]
pub use compression_drafter::{CompressionDrafter, Lz4FlexDrafter};

// BabelCodec — Readability-relaxed semantic codec (Plan 331, Research 312,
// arXiv:2606.19857 BabelTele). Successor text codec to CompressionDrafter:
// where CompressionDrafter failed G2 twice on the Seal corpus (byte-level LZ4
// matching on short quest-grammar strings), BabelCodec operates on semantic
// STRUCTURE (BT-P8 fixed symbolic mapping rules) — purpose-built for KG-triple
// / entity-attribute / config / quest-grammar surfaces. Ships three pieces:
// (1) generic `BabelCodec` trait, (2) `FixedRuleTextCodec` (deterministic BT-P8
// text codec, the modelless subset of BabelTele), (3) `SigmoidLatentCodec<D>`
// (generic-trait facade over existing DensityBudget infrastructure, latent-level
// analog — value is API uniformity, NOT new capability), plus BLAKE3 commitment
// for the future LatCal chain bridge (.issues/002). Sigmoid, not softmax.
// Opt-in until the G1–G5 GOAT gate passes — the same G2 (≥ 2× on real corpus)
// gate that killed CompressionDrafter twice.
#[cfg(feature = "babel_codec")]
pub mod babel_codec;
#[cfg(feature = "babel_codec")]
pub use babel_codec::{
    BabelCodec, BabelCommitment, BabelPair, CompressedLatent, FixedRuleTextCodec,
    SigmoidLatentCodec,
};

// Analytic Lattice — k×k transport operator chain composer + ASOC trait shapes
// + direction-vector SIMD decoder + spectral audit (Plan 330, Research 311).
// katgpt-core half: pure math primitives + generic trait shapes (NO GpuFuture
// import — leaf-clean). The ComposerTick: GpuFuture impl + Join3 combinator
// ship in riir-engine under the `analytic_lattice_runtime` feature (Phase 1b).
// Opt-in until G1–G6 GOAT gate passes.
#[cfg(feature = "analytic_lattice")]
pub mod analytic_lattice;
#[cfg(feature = "analytic_lattice")]
pub use analytic_lattice::{
    ChainError, ComposerCtx, LatticeVector, PlasmaDraft, RederiveOp, TransportOperator,
    apply_operator_into, audit::AuditReport, audit::spectral_audit, batch_compose_chain,
    batch_compose_chain_into, compose_chain, compose_chain_into, decoder::direction_vector_decode,
    decoder::direction_vector_decode_into, decoder::direction_vector_decode_slice,
};

// Functional Attention — closed-form Tikhonov spectral transport operator
// (Plan 286, Research 257, arxiv 2605.31559, Xiao et al. ICML 2026). DUAL FORM
// matching the reference implementation (`.raw/FUNCATTN/PDE-StandardBenchmark/model/
// Functional_attention.py`): convex-combo regularization `(1-α)·K̃ᵀK̃ + α·I_d`,
// column-normalized slice tokens, per-slice-token to_q/to_k/to_v linear
// projections. Sigmoid-basis default per AGENTS.md (partition-of-unity holds
// for any row-normalized non-negative kernel). Gain-tier open primitive:
// paper itself defers NLP validation (§6); promote only after G1–G5 GOAT
// gate passes.
#[cfg(feature = "funcattn")]
pub mod funcattn;
#[cfg(feature = "funcattn")]
pub use funcattn::{
    FuncAttnBasis, FuncAttnConfig, FuncAttnError, FuncAttnScratch, compute_basis_into,
    funcattn_forward, pre_rotate_basis_weights_into, solve_convex_combo_dual,
};
// Plan 332 — principled multi-scale basis constructors (DCT-log, Haar-packet).
// gated by the dedicated `funcattn_structured_basis` feature (implies funcattn).
#[cfg(feature = "funcattn_structured_basis")]
pub use funcattn::{make_dct_log_basis, make_haar_packet_basis};

// Plan 353 — Head Substitution Gate (Gain-tier, opt-in). Small decision
// struct that decides when a FuncAttn-style surrogate should substitute for
// a real attention head, using the paper's IoU cheap-proxy (§3 Fig 5b r>0.9)
// + cached FaithfulnessProfile veto (Plan 287 SinkAware cadence). NOT a new
// primitive — the original draft proposed a redundant ProgramSynthesizedHead
// primitive that was dropped after re-review identified FuncAttn (above) as
// the existing primitive surface. Stays opt-in: Gain-tier, and the plan's
// own Risk note flags it as borderline-thin for a feature flag.
#[cfg(feature = "functional_substitution_gate")]
pub mod functional_substitution;
#[cfg(feature = "functional_substitution_gate")]
pub use functional_substitution::{HeadSubstitutionGate, iou, worst_case_behavior_delta};

// Cross-Resolution Spectral Transport — asymmetric-basis FUNCATTN (Plan 310,
// Research 291, arxiv 2605.31559). Generalizes FUNCATTN to d_src ≠ d_dst,
// enabling train-on-small-deploy-on-large latent transfer without retraining.
// Open primitive: frozen BLAKE3-committed bases + zero-alloc transport.
// Opt-in until G1–G4 GOAT gate passes.
#[cfg(feature = "cross_resolution_transport")]
pub mod cross_resolution;
#[cfg(feature = "cross_resolution_transport")]
pub use cross_resolution::{
    CrossResScratch, CrossResolutionBases, CrossResolutionError, project_to_spectral_into,
    reconstruct_from_spectral_into, transport_cross_domain_cross_resolution_into,
    transport_cross_resolution, transport_cross_resolution_into,
};

// Latent Field Steering — top-down direction-vector injection into mutable
// latent state (Plan 309, Research 290, CAA + functional emotions). The missing
// fourth quadrant: CNA mutates neurons, EmotionDirections is read-only, FPCG
// refuses mutation — this injects directly into the latent state on the hot
// path. Zero-alloc SIMD SAXPY + sigmoid-falloff localized support.
// Opt-in until G1–G5 GOAT gate passes (G2 make-or-break: rank preservation ≥0.95).
#[cfg(feature = "latent_field_steering")]
pub mod latent_steering;
#[cfg(feature = "latent_field_steering")]
pub use latent_steering::{
    FieldSupport, HLA_AROUSAL, HLA_CALM, HLA_DESPERATION, HLA_DIM, HLA_FEAR, HLA_VALENCE,
    LatentField, LatentSteeringError, LatentSteeringVector, apply_field_to_crowd,
    apply_latent_steering, apply_latent_steering_weighted, kernel_weight,
};

// Subspace Steering Field — k-dim manifold steering (Plan 412, Research 393,
// arxiv 2606.25234 Goodfire BSF). The k-dim generalization of Plan 309: an
// orthonormal block `{u_1..u_k}` + per-axis strengths `{α_1..α_k}`, math
// `s' = s + Σ_j α_j · u_j`. At K=1 bit-identical to Plan 309; at K≥2 enables
// manifold walking (sweep alphas over a grid → concept variations). Pure
// modelless consumer of pre-discovered blocks (Plan 301 Jacobian SVD,
// SpectralQuant eigenbasis, or hand-constructed sets). Opt-in until G1–G5
// GOAT gate passes (G1 K=1 parity with Plan 309 is the load-bearing gate).
#[cfg(feature = "subspace_steering")]
pub mod subspace_steering;
#[cfg(feature = "subspace_steering")]
pub use subspace_steering::{
    SubspaceSteeringError, SubspaceSteeringField, apply_subspace_steering,
    block_energy, compute_block_commitment, walk_manifold,
};

// Phase-Modulated Subspace Rotation Gate — norm-preserving latent coupling
// `cos α ⊙ a + sin α ⊙ b` with phase from a sigmoid projection onto a frozen
// direction vector (Plan 322, Research 305, arxiv 2605.12700 UFO). The
// genuinely-new operation class: every other latent op in the crate is
// additive / convex-combo / dot-projection / wedge-detection / linear-transport
// / spatial-sum — none has the `sin²α+cos²α=1` Pythagorean norm-preservation
// invariant. §3.5 modelless Path 2 unblock: the trained `γ_θ` is replaced with
// `α = sigmoid(⟨state, direction⟩ · λ) · π/2` (closed-form). Opt-in until the
// G1–G4 GOAT gate passes (G1 norm-preservation <1e-4 is the kill switch).
#[cfg(feature = "phase_rotation_coupling")]
pub mod phase_rotation;
#[cfg(feature = "phase_rotation_coupling")]
pub use phase_rotation::{
    PhaseRotationError, PhaseRotationGate, PhaseRotationScratch, compute_phase_from_projection,
    compute_phase_per_channel_into, phase_rotation_gate_into,
};

// Spherical Steering — single-target geodesic Slerp rotation
// `sin((1−t)θ)/sin θ · ĥ + sin(tθ)/sin θ · μ_T` toward a unit-norm target
// direction on S^{d-1}, with sigmoid-translated vMF confidence gate (Plan 405,
// Research 382, arxiv 2602.08169 You/Deng/Chen ICML 2026). Sibling to Plan 322's
// 2-subspace phase rotation — same norm-preservation thesis, different
// parameterization: Plan 322 rotates *within* the (a,b) plane; Plan 405 rotates
// *toward* a target outside the input's direction (Slerp identity holds for all
// θ ∈ (0,π)). vMF gate reduces to sigmoid via Eq 17:
// `δ = -tanh(κ·s_T) = 1 − 2·sigmoid(2κ·s_T)`. §3.5 modelless Path 3 (closed-form
// trig + sigmoid; no training). Opt-in until G1–G6 GOAT gate passes (G1
// norm-preservation <1e-4 is the kill switch, mirroring Plan 322's G1).
#[cfg(feature = "spherical_steering")]
pub mod spherical_steering;
#[cfg(feature = "spherical_steering")]
pub use spherical_steering::{
    SlerpError, SlerpScratch, slerp_steering_into, spherical_steering_into, vmf_confidence_gate,
};

// ChunkedContentStore — Lore-distilled chunked content-addressed Merkle store (Plan 272, Research 262).
// Open primitive: chunks → BLAKE3 → dedup via papaya → binary Merkle root. No game/chain IP.
// Consumed by riir-ai Plan 319 (Executable Asset Vessel + Quorum Gitflow).
//
// NOTE: the binary-Merkle `MerkleProof` here is renamed on re-export to
// `BinaryMerkleProof` to avoid colliding with `merkle_octree::MerkleProof`
// when both features are active simultaneously (caught by `cargo check
// --all-features`). Internal callers still reach the type via
// `crate::content_store::MerkleProof`.
#[cfg(feature = "chunked_content_store")]
pub mod content_store;
#[cfg(feature = "chunked_content_store")]
pub use content_store::{
    BlobId, ChunkFetcher, ChunkRange, ChunkedContentStore, ChunkerConfig, ChunkingStrategy,
    FastCdcChunker, FixedSizeChunker, InMemoryChunkedStore, MerkleProof as BinaryMerkleProof,
    StoreStats, build_binary_merkle_proof, build_binary_merkle_root, verify_binary_merkle_proof,
};

// Closure-Expansion Instrument (CEI) — PTG recorder + motif miner + PRI/CDG/TaR metrics
// (Plan 290, Research 264, arxiv 2606.15386, Momennejad & Raileanu). Open measurement
// layer: turns open-ended inference into observable metrics. Opt-in until G1–G4 GOAT
// gate passes; G5 demotes to opt-in diagnostic if metrics don't correlate with quality.
#[cfg(feature = "closure_instrument")]
pub mod closure;
#[cfg(feature = "closure_instrument")]
pub use closure::{
    OperatorKind, PrimitiveKind, PrimitiveTransitionGraph, PtgEdge, PtgNode,
    admit::{GateResult, MotifAdmitter, RejectionReason},
    bridge::{
        DEFAULT_MOTIF_DIRS, MotifDirections, motif_embedding_to_tar_score, ptg_to_motif_embedding,
    },
    commitment, deserialize_postcard,
    metrics::{CdgScore, PriScores, compute_cdg, compute_pri, compute_tar_score, motif_multiset},
    mining::{SleepCycleClosureReport, fold_cdg_at_sleep_cycle, mine_motifs_at_sleep_cycle},
    motif::{
        FixedU32Set, MAX_MOTIF_EDGES, MAX_MOTIF_NODES, Motif, MotifMiner, RING_BUFFER_K,
        enumerate_subgraph_hashes,
    },
    serialize_postcard,
    trace::{DEFAULT_TRACE_CAPACITY, NodeId, PtgRecorder},
};

// Issue 040 — PTG × latent_functor edge composition. Ships `FunctorPtg`
// (composite wrapper over an unchanged `PrimitiveTransitionGraph`),
// `FunctorEdgeParams` (per-edge continuous-functor params), and
// `apply_functor_edge_into` (zero-alloc sigmoid-gated apply path). Gated by
// `ptg_functor_edges` (implies `closure_instrument`). Wire-format safe: the
// inner PTG is byte-identical to a bare PTG.
#[cfg(feature = "ptg_functor_edges")]
pub use closure::{FunctorEdgeParams, FunctorPtg, apply_functor_edge_into, functor_edge_gate};

// Sink-Aware Attention — NOP/Broadcast classifier + dual-policy sigmoid gate
// (Plan 287, Research 258, arxiv 2606.08105, Fesser et al.). Per-head
// classifier (value-norm-ratio + stable-rank-of-update) decides whether a
// sink is Adaptive NOP (gate it via sigmoid) or Broadcast (preserve it).
// Staged integration: the policy enum + standalone apply_dual_policy_gate
// ship here; direct wiring into parallax_attn / funcattn forward paths is
// deferred until synthetic G2 + latency G3 gates pass on a real model
// (validation fallback per Plan 287 §Validation).
//
// Plan 404 (2026-07-06): the parent module is now always-on. The pure
// information-theoretic substrate (markov/nll/typical_set/dirichlet_energy/
// claim) moved here from root `src/data_probe/`. The sink-aware classifier
// (`sink_classify`) + `geometry` remain gated `sink_aware_attn` inside the
// module. The gated re-exports below preserve `crate::data_probe::SinkKind`
// etc. for internal consumers (notably `parallax_attn.rs`). The always-on
// re-exports (markov/nll/typical_set/dirichlet_energy/claim) live in
// `data_probe/mod.rs`.
pub mod data_probe;
#[cfg(feature = "sink_aware_attn")]
pub use data_probe::{
    CachedSinkClassification, SinkAwarePolicy, SinkClassifierConfig, SinkDiagnostic, SinkKind,
    StableRankScratch, apply_dual_policy_gate, apply_dual_policy_gate_cached,
    apply_dual_policy_gate_cached_flat, apply_dual_policy_gate_flat, classify_all_sinks,
    classify_all_sinks_flat, classify_sink_at, classify_sink_at_flat, stable_rank_update_into,
    stable_rank_update_into_flat,
};

// ICT Distributional Branching-Point Detector — open generic math (Plan 294,
// Research 270, arxiv 2606.19771). Collision purity β(π) = Σ π² (proven
// unconditionally monotone, ICT §A.2.5 — H₁ is wrong below π > e⁻¹ ≈ 0.37),
// Rényi H₂, Jensen-Shannon divergence to group mean, BranchingDetector
// (top-k% selector over K candidate trajectories + per-step β EMA), and the
// Bebop H₁→H₂ acceptance-forecast upgrade. No game semantics, no chain;
// runtime fusion (CLR gating, HLA updates, KG emission) is riir-ai Plan 324.
// Opt-in until G3 (Spearman ρ(H₁, JS-uniqueness) < 0.5) AND G8 (riir-ai
// Plan 324 runtime validation) pass.
#[cfg(feature = "ict_branching")]
pub mod ict;
#[cfg(feature = "ict_branching")]
pub use ict::{
    AcceptanceForecastH2, BranchingDetector, BranchingReport, branching_point_mask,
    branching_point_mask_into, collision_purity, collision_purity_into, is_critical_branching,
    js_divergence, js_divergence_batch, renyi_h2, shannon_h1,
};

// ── Induced Code World Model (Plan 296, Research 275, arxiv 2510.04542) ───────
//
// Open half of the CWM Super-GOAT: a marker trait over `GameState` for forward
// models that are verifiable, BLAKE3-committable, and hot-swappable. The
// LLM-induction pipeline that *produces* an `InducedCwmKernel` impl is private
// (riir-ai Plan 326). The runtime never sees the LLM — only the frozen kernel.
#[cfg(feature = "induced_cwm")]
pub mod induced_cwm;
#[cfg(feature = "induced_cwm")]
pub use induced_cwm::{
    BeliefInferenceFn, CwmCommitment, InducedCwmKernel, TransitionTestFailure, TransitionUnitTest,
    make_transition_tests_from_trajectory, verify_transition,
};

// Phase 2 (Plan 296 T2.1–T2.5): Information-Set MCTS over an induced CWM +
// belief fn. Self-contained search tree (does NOT reuse root-crate
// `mcts_search` — that lives in katgpt-rs/src, katgpt-core cannot depend on the
// root). Gated by `induced_cwm_ismcts` (which auto-enables
// `induced_cwm`).
#[cfg(feature = "induced_cwm_ismcts")]
pub use induced_cwm::{InformationSet, NodeStats, ismcts_search_with_inference};

// ── Bisimulation Operator Inference (Plan 324, Research 308, arxiv 2602.19260) ─
//
// Open primitive: quotient an observed transition graph into bisimulation-
// equivalent state classes (signature-based partition refinement, O((S+E)
// log² S log d)) and infer an abstract PDDL-like operator schema. The
// lighter-weight PDDL-side counterpart to Induced CWM (Plan 296): where CWM
// induces executable *code* via an LLM, this induces an *operator schema* via
// a deterministic graph algorithm. Closes Research 264 §2.2 gaps #1 (PTG) +
// #2 (motif mining). Opt-in by design — downstream pipelines (riir-ai NPC
// runtime, riir-chain LatCal consumer) opt in by enabling the feature.
#[cfg(feature = "bisimulation_operator_inference")]
pub mod bisimulation;
#[cfg(feature = "bisimulation_operator_inference")]
pub use bisimulation::{
    BisimulationQuotient, OperatorDef, OperatorLabel, OperatorSchema, Plan, QuotientEdge,
    StateClassId, StateId, Transition, TransitionGraph, TransitionGraphBuilder, partition_refine,
    plan as bisimulation_plan,
};

// ── Personality-Weighted Layer Composition (Plan 297, Research 276) ──────
//
// Open MIT-licensed primitive for the Entity Cognition Stack Super-GOAT.
// A `PersonalityWeightedComposition<N, D>` kernel composes `N` latent
// direction vectors via per-layer sigmoid-gated weights, then drifts those
// weights via a reward-surprise Hebbian update. Zero-allocation, sigmoid-gated
// (NOT softmax — per AGENTS.md), belief-gated, BLAKE3-snapshot-integrated.
// Entity-agnostic (NPC, player, predator, prey, robot, recommender user).
//
// Consumed by riir-ai Plan 327 (runtime wiring) — the game-specific 7-layer
// mapping, archetype table, taming transition stay private in riir-ai.
// Opt-in until G4 (<1µs/entity) + G5 (zero alloc) GOAT gate passes.
//
// Substrate lives in the katgpt-personality crate (Issue 007 Phase E Tier 2
// #5, 2026-06-28). Re-exported here as `katgpt_core::personality_composition`
// for backwards compatibility — all `crate::personality_composition::*` paths
// resolve unchanged. The `personality_composition` Cargo feature turns on the
// `dep:katgpt-personality` dependency; the substrate compiles unconditionally
// inside the crate itself.
#[cfg(feature = "personality_composition")]
pub use katgpt_personality as personality_composition;
#[cfg(feature = "personality_composition")]
pub use personality_composition::{
    ArchetypeLabel, EntityCognitionComposition, LayerDirectionSource, PersonalityConfig,
    PersonalitySnapshot, PersonalityWeightedComposition, sigmoid as personality_sigmoid,
    sigmoid_into as personality_sigmoid_into,
};

// ── Committed Field Blend (Plan 321, Research 302) ───────────────────────
//
// Open MIT-licensed primitive: the sampling-invariant half of the FAME
// Super-GOAT. A `CommittedFieldBlend<N, D>` computes blend weights pi ONCE
// from a trajectory summary via sigmoid projection, then FREEZES them for
// the entity's lifetime. The blended field f_pi(z) = Σ_k sigmoid(pi_k/tau) ·
// f_k(z) governs dynamics; because both pi and the fields are frozen, the
// trajectory is sampling-invariant (FAME Proposition 3 / Young-integral).
// Zero-alloc apply + BLAKE3-committed. Reuses personality_composition's
// sigmoid + simd::simd_fused_scale_acc (DRY).
// Opt-in until G1–G5 GOAT gate passes; G2 (sampling invariance) is the
// make-or-break gate. Private selling-point guide at riir-ai/.research/158.
#[cfg(feature = "committed_field_blend")]
pub mod committed_field_blend;
#[cfg(feature = "committed_field_blend")]
pub use committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend, TriArchetypeBlend};

// ── Engram — Hash-Addressed Pattern Memory (Plan 299, Research 278) ───────
//
// Open MIT-licensed primitive: the first conditional-MEMORY axis in the
// katgpt stack (complementary to Raven's conditional-COMPUTATION axis).
// N-gram-suffix → multi-head hash → O(1) slot lookup → sigmoid gate (RMSNorm
// dot σ) → residual-fuse into hidden state. Frozen table, atomic swaps for
// updates, BLAKE3 commitment as sync-boundary audit artifact.
//
// CRITICAL: sigmoid, not softmax — per AGENTS.md. No `softmax` symbol here.
//
// Open half of the Engram Super-GOAT: private selling-point guide lives in
// riir-ai Guide 147; chain commitment bridge is riir-chain R001 (TODO).
// Opt-in until G1–G7 GOAT gate passes.
#[cfg(feature = "engram")]
pub mod engram;
#[cfg(feature = "engram")]
pub use engram::{
    CacheResult, CacheTier, ColdFetcher, EngramConfig, EngramHash, EngramHotSwap, EngramTable,
    EngramTableBuilder, EngramTableId, HashHead, IDENTITY_KERNEL, InMemoryEngramTable, K_MAX,
    SigmoidFusionConfig, StagingEngramTable, StagingError, SurjectiveMap, SurjectiveMapLoadError,
    TokenId, TokenizerSpec, ZipfianCacheHierarchy, ZipfianStats, ZipfianStatsSnapshot,
    build_merkle_root, build_surjective_map, compress_token, conv_causal_into,
    fuse_into_hidden_state, multi_head_hash, rmsnorm_into, sigmoid_fuse_into,
    sigmoid_fuse_multi_branch_into, try_compress_token,
};

// ── Product Key Memory — O(√N) Factored Retrieval (Plan 408, Research 387) ─
//
// Open MIT-licensed primitive: the fourth complexity class in the retrieval
// stack (Raven O(1) / Engram O(1)-hash / δ-Mem O(r) / PKM O(√N)). Splits a
// d_k-dim query into two halves, scores two √N codebooks, takes top-k of the
// k² Cartesian product — yielding `2√N + k²` cost instead of `N`. Scales to
// ~10⁶ slots at sub-linear retrieval cost.
//
// Modelless (constraint #1): the FwPKM paper's gradient-descent half (L_mem
// GD on V, L_addr GD on K, n-iter TTT) is forbidden. Replaced by shipped
// δ-rule analog (Plan 053). This primitive ships ONLY the inference-time
// factored retrieval; the optional δ-rule write path lands in Phase 5
// (product_key_memory_episodic).
//
// Phase 1 (this commit): types only — const-generic
// `ProductKeyMemory<SQRT_N, D_K, D_V>`, `ScoreFn` (Dot/Idw), fixed-size
// `PkQuery<K>`. Leaf-clean (zero deps). Phase 2 ships the kernel + GOAT gate.
// Opt-in until G1+G2+G4 GOAT gate passes.
#[cfg(feature = "product_key_memory")]
pub mod product_key_memory;
#[cfg(feature = "product_key_memory")]
pub use product_key_memory::{
    D_K_FLOOR, PkEntry, PkQuery, PkmScratch, ProductKeyMemory, SQRT_N_FLOOR, ScoreFn, score_dot,
    score_idw,
};
// Phase 4 (F4 fusion) — freeze/thaw wrapper around ProductKeyMemory. Gated
// separately so the leaf-clean retrieval primitive (above) stays usable
// without the Arc<RwLock<Arc<...>>> + BLAKE3 commitment machinery. See
// `product_key_memory/freeze.rs` for the pattern rationale (mirrors
// `induced_cwm/hot_swap.rs`).
#[cfg(feature = "product_key_memory_freeze")]
pub use product_key_memory::FrozenProductKeyMemory;

// Plan 408 Phase 5 — δ-rule write gate over PKM (F1 fusion: PKM × δ-Mem).
// PkmEpisodicStore wraps FrozenProductKeyMemory + a mutable working copy.
// Gated on `product_key_memory_episodic` (implies `product_key_memory_freeze`).
#[cfg(feature = "product_key_memory_episodic")]
pub use product_key_memory::PkmEpisodicStore;

// Gain/Cost Loop Halting Primitive — open substrate-agnostic kernel for per-loop
// halting decisions (Plan 304, Research 282, arXiv:2606.18023, LoopCoder-v2).
//
// halt when marginal refinement gain < marginal drift cost × τ; oscillation
// early-halt via cos θ < 0; L_min floor protects representational capacity.
// Composes with the shipped elastic-loop override (Issue 035) — Phase 2 will wire
// this into `forward_looped()` (separate scope). Phase 1 ships the kernel only.
//
// Latent vs Raw: gain/cost signals are local latent (per-loop hidden-state
// deltas); the halt count L is a deterministic raw scalar safe to sync/replay.
//
// Opt-in until G1–G5 GOAT gate (Research 149 §5) passes.
#[cfg(feature = "gain_cost_halt")]
pub mod gain_cost_halt;
#[cfg(feature = "gain_cost_halt")]
pub use gain_cost_halt::{
    GainCostLoopHalter, HaltDecision, HaltReason, angular_change, hidden_erank, step_size,
};

// Cross-Datapoint Set Attention — sigmoid-gated, permutation-equivariant
// cross-entity refinement kernel (Plan 354, Research 354, arXiv:2106.02584
// Kossen et al. NeurIPS 2021, Non-Parametric Transformers). The inference-time
// operator only — training of Q/K/V via BERT-style masking stays in riir-train.
// Substrate-agnostic: `&[f32]` → `&mut [f32]`, no opinion on what the vectors
// mean. The riir-ai runtime (Plan 355) wires it onto HLA belief states for
// crowd-scale NPC joint inference; the open primitive is just the math.
//
// Sigmoid gates (NEVER softmax per AGENTS.md §2) — each pair α_ij ∈ (0,1)
// independently, so an entity may attend to 0 peers (lonely), 1 peer (paired),
// or many peers (formation). Softmax would force artificial competition.
//
// Permutation-equivariant by construction (NPT Lemma 4, Appendix A) —
// shuffling input rows shuffles output rows identically. The G1 test
// verifies this bit-exactly.
//
// Latent vs Raw: the primitive is substrate-agnostic. The sync boundary is
// the caller's responsibility (see the riir-ai runtime plan 355 for the
// HLA-specific wiring + the unchanged 5-scalar bridge).
//
// Opt-in until G1–G5 GOAT gate (Research 354 §5) passes; Super-GOAT promotion
// also requires riir-ai Plan 355 G6 (CS-ranking fusion adds value).
#[cfg(feature = "set_attention")]
pub mod set_attention;
#[cfg(feature = "set_attention")]
pub use set_attention::{
    SetAttentionConfig, SetAttentionError, identity, identity_into, identity_projection,
    identity_projection_into, set_sigmoid_attention_into,
};

// Depth-Invariance Diagnostic + Magnitude-Regularized Residual — the
// root-cause counterpart to four symptom-only detectors (BeliefRankPruner,
// GainCostLoopHalter, latent_functor/reestimation.rs,
// micro_belief/coherence_bench.rs). Modelless math, no game semantics.
// Classifies recursive latent-state chains as DepthInvariant /
// DepthSpecificRefinement / Collapsed / Insufficient. The MagnitudeReg
// wrapper is the modelless fix for kernels we own (HLA, functor,
// micro_belief, engram, Raven); for frozen MLPs (BeliefDrafter) the fix
// requires retraining → riir-train.
// Plan 306 Phase 1+5; Research 286; arXiv:2605.09992 Eldenk et al.
// Opt-in until G1 (8 correctness tests) passes.
#[cfg(feature = "depth_invariance")]
pub use katgpt_types::depth_invariance;
#[cfg(feature = "depth_invariance")]
pub use katgpt_types::depth_invariance::{
    DepthInvarianceConfig, DepthInvarianceDiagnostic, DepthInvarianceKind, MagnitudeRegularization,
    Scratch, apply_magnitude_regularization, classify_chain, classify_chain_batched,
};

// Shared linear-algebra kernels. Originally extracted for `karc`'s ridge-style
// solvers (Plan 308); the f32 Cholesky/ridge path lives here as a standalone
// extraction of the PEIRA `(N + λI)⁻¹` pattern — see the module note for why
// PEIRA's f64 path is left untouched. Plan 319 (Clifford geometric product)
// and Plan 326 (Tucker/HOSVD tensor factorization) ship peers under `linalg::`
// — each must also gate this `pub mod` so the crate compiles when only that
// feature is on.
#[cfg(any(
    feature = "karc_forecaster",
    feature = "geometric_product",
    feature = "tucker_factorization"
))]
pub mod linalg;

// KARC — Kolmogorov-Arnold Reservoir Computing delay-basis-ridge forecaster
// (Plan 308, Research 288, arXiv:2606.19984). Modelless, inference-time
// trajectory forecaster: delay-embedding × sealed KarcBasis (Fourier/Chebyshev/
// BSpline) × closed-form ridge readout, with a zero-alloc forecast matvec.
// Opt-in until G1–G4 GOAT gate passes (no root-feature alias in Phase 1).
#[cfg(feature = "karc_forecaster")]
pub mod karc;
#[cfg(feature = "karc_forecaster")]
pub use karc::{
    BSplineBasis, ChebyshevBasis, DelayRing, FitError, FourierBasis, KarcBasis, KarcForecaster,
    KarcScratch, LowRankFitScratch, chunked_gram_into, feature_expand, feature_expand_higher_order,
    forecast_low_rank_apply, higher_order_feature_count, low_rank_fit,
};

// KarcShard DP Output Perturbation (Issue 370 T4) — post-hoc Gaussian noise
// on a fitted ridge Wout matrix to provide formal (ε,δ)-DP for the committed
// KarcShard parameters. Defends PARAMETER-INSPECTION MI (attacker reads Wout
// to detect memorized patterns). Does NOT defend Yeom loss-threshold MI —
// see karc_dp module docs and riir-ai/.benchmarks/399 for the structural
// insufficiency analysis. Modelless (post-hoc noise on a closed-form solve).
// Gated on karc_forecaster since it operates on the Wout produced by
// KarcForecaster::fit_ridge.
#[cfg(feature = "karc_forecaster")]
pub mod karc_dp;
#[cfg(feature = "karc_forecaster")]
pub use karc_dp::{KarcDpNoiseConfig, apply_dp_noise_to_wout};

// ARG Protocol Primitives — open half of the ARG × Latent Substrate Super-GOAT
// fusion (Plan 327 Phases 1-3, Research 309, Guide 160 private). Five generic
// protocol primitives distilled from the ARG Standard
// (https://protocol.airistech.ai/arg-core.html, Iris Technologies 2026):
// `PolicyEnvelope` (Step 1 hard gate), `TaxonomyValidator` (Step 3 deterministic
// label-set validator), `LifecycleState` + `RedirectTable` (Step E ontology
// lifecycle continuity), `TypedOfflineCandidate` + `CandidateIntent` (Step C
// typed offline candidate), `OfflineCandidateScorer` (Step C scoring with the
// G5 silence-bias penalty), `InfoRegistry` (Step 9 + Step C two-phase dedup
// with grey-zone review). Private runtime composition with HLA / Entity
// Cognition Stack / VMG / Sub-Goal Compaction lives in riir-ai Plan 337.
// No game/chain/shard semantics. DEFAULT-ON (Plan 327 Phase 4, 2026-06-25):
#[cfg(feature = "arg_protocol")]
pub mod arg;
#[cfg(feature = "arg_protocol")]
pub use arg::{
    AccessScope, CandidateIntent, CandidateKind, CompareFn, CompareResult,
    DEFAULT_AUTO_COMMIT_THRESHOLD, Evidence, EvidenceId, GainComponents, InfoKey,
    InfoOutcomeStatus, InfoRegistry, InfoType, InfoUnit, LabelId, LabelSet, LabelSignature,
    LifecycleState, MatchResult, MatchScratch, OfflineCandidateScorer, PayloadHash,
    PayloadHashCompare, PolicyConstraints, PolicyDecision, PolicyEnvelope, PolicyState, Provenance,
    RedirectTable, ResponseMode, ScoredCandidate, ShouldProceed, TaxonomyKind, TaxonomyNode,
    TaxonomyValidator, TypedOfflineCandidate, ValidationError, ValidationResult, ValidationScratch,
};

// Non-Interference Memory Branches — Super-GOAT fusion (Plan 329, Research 310,
// arXiv:2606.20638 Goel et al. Oxford Jun 2026). Five generic open primitives:
// BranchBank (bounded persistent CognitiveBranch bank with spawn/merge/prune
// lifecycle), BranchRouter (dot-product snap + Jaccard fallback), VerifierGate
// (reward + curiosity + centroid-quarantine write gate, composes with CLR),
// NonInterferenceProjection (orthogonal latent subspace per branch),
// BudgetCompiler (priority-cascade context compiler under fixed budget). Fuses
// BAKE × CLR × MCGS × Engram × ARG × closure-instrument × Salience into a new
// capability class: per-NPC continual adaptation without catastrophic
// forgetting. Composes with arg_protocol LifecycleState when both features on.
// Opt-in until G1–G5 GOAT gate passes (Phase 3).
#[cfg(feature = "non_interference_branches")]
pub mod branching;
#[cfg(feature = "non_interference_branches")]
pub use branching::{
    AssignError, AssignResult, BranchBank, BranchId, BranchLifecycle, BranchRouter, BranchStats,
    BudgetCompiler, CognitiveBranch, CompiledContext, CompiledItem,
    DEFAULT_ASSIGN_MAX_INTERFERENCE, DEFAULT_BUDGET_BYTES, DEFAULT_MAX_BRANCHES,
    DEFAULT_ORTHOGONAL_EPSILON, DEFAULT_PROJECTION_DIM, DEFAULT_QUARANTINE_CENTROID_THRESH,
    DEFAULT_TAU_CURIOSITY, DEFAULT_TAU_JACCARD, DEFAULT_TAU_SNAP, DEFAULT_TAU_SPAWN,
    DEFAULT_TAU_WRITE, EpisodicEntry, FailureEntry, NonInterferenceProjection, PriorityTier,
    ProceduralRule, RetrievedMaterials, RouteMode, RouteResult, VerifierGate, WriteDecision,
    max_orthogonal_branches,
};

// Post-Candidate Branch Router — distilled from Local Branch Routing
// (arXiv:2606.25354, Yin et al. June 2026). The modelless inference mechanism
// distilled to its open primitive: forward K candidate next-tokens, score each
// post-candidate hidden state by dot-product onto a frozen direction, commit
// the argmax (or perturbed-argmax sample with Logistic noise — the sigmoid
// analog of Gumbel-max).
//
// Generalizes the shipped ColliderPruner::batch_is_valid_with_hidden from
// binary prune/keep to relative route-and-commit. PoC-confirmed modelless
// quality gain of +9pp to +26pp across 5 noise cells (Plan 377 Phase 1,
// riir-ai/crates/riir-poc). Set-attention variant adds zero modelless value
// (PoC §8 — within ±1pp of the dot-product router across v1 and v2) and stays
// a riir-train follow-up (needs trained Q/K/V projections).
//
// Sigmoid (NEVER softmax) per AGENTS.md §2: sampling uses Logistic(0, β)
// noise whose CDF is sigmoid(x/β), making the categorical sample a
// sigmoid-family operation without any exp/softmax normalization.
//
// Opt-in until Plan 377 Phase 3 GOAT gate (G1 correctness ≥90%, G2 router
// latency <1µs at K=3 D=64, G3 K=1 bit-identical to standard decode, G4
// alloc-free hot path, G5 modelless, G6 sigmoid-not-softmax).
#[cfg(feature = "local_branch_routing")]
pub mod branch_routing;
#[cfg(feature = "local_branch_routing")]
pub use branch_routing::{
    ColliderRouterAdapter, DotProductRouter, PostCandidateRouter, PreservationScorer,
};

// Sleep-Time Query Anticipator — open primitive for offline query anticipation
// (Plan 334, Research 318, arXiv:2504.13171 Lin et al. Letta/Berkeley).
// Implements the open math half: SleepTimeAnticipator orchestrates per-direction
// sleep-time compute → emits reusable AnticipatedQuerySet (the c' artifact,
// BLAKE3-committed) → wake-time consume() does cheap dot-product + sigmoid-
// gated lookup, falling through to fresh compute on low-predictability queries.
// PredictabilityScorer trait + DotPredictabilityScorer default
// (p = sigmoid(α·dot(c,dir)+β)); AmortizationCostModel operationalizes the
// paper's §5.3 cost model. Game-specific direction-vector catalogs, NPC tiering,
// HLA wiring, and chain commitment live in riir-ai Plan 341 (private).
// Phase 1 ships traits + types + IdentityFunctorOp (synthetic-test default);
// Phase 2 ships synthetic gates G1/G2/G5/G6/G7. G2/G3/G4 quality gates require
// a real predictability-labeled corpus → deferred to riir-ai Plan 341.
// Opt-in until G1–G5 GOAT gate passes; promotion to default-on requires
// Plan 341 G1–G5 to clear on a real game corpus.
//
// Substrate lives in the katgpt-sleep crate (Issue 007 Phase E Tier 2 #6,
// 2026-06-28). Re-exported here as `katgpt_core::sleep_time` for backwards
// compatibility — all `crate::sleep_time::*` paths resolve unchanged. The
// `sleep_time_anticipation` Cargo feature turns on the `dep:katgpt-sleep`
// dependency; the substrate compiles unconditionally inside the crate itself.
#[cfg(feature = "sleep_time_anticipation")]
pub use katgpt_sleep as sleep_time;
#[cfg(feature = "sleep_time_anticipation")]
pub use sleep_time::{
    AmortizationCostModel, AnticipatedQueryDir, AnticipatedQuerySet, AnticipatedSlot,
    ConsumeMatchMode, DEFAULT_LATENCY_PREMIUM, DotPredictabilityScorer, IdentityFunctorOp,
    PredictabilityScorer, SLEEP_TIME_DEFAULT_K, SleepTimeAnticipator, SleepTimeComputeOp,
    SleepTimeScratch, commit_direction, consume, consume_gate, consume_gate_with_match_mode,
    consume_with_match_mode,
};

// PairedLossGap — generic modelless paired token-level loss gap diagnostic
// (Plan 335, Research 319, arXiv:2606.20936 Li & Merrill AI2). Pure
// measurement tool: given two log-prob traces over the same prefixes, compute
// per-token Δ_i = ℓ_A − ℓ_B, stratify by token class, report filtered
// aggregates (ALL / TOP-K∩NO-COPY / COPY-N-ONLY) that amplify small
// architecture gaps aggregate loss hides. ClassSizeBound exposes Proposition 1
// (DKL ≤ log|V_τ|) — the volume-of-support bound justifying raw-vs-latent
// sync. Generic math, no game/chain/shard semantics — legitimately public.
// NOT an inference mechanism (measurement tool only) → not Super-GOAT.
// Opt-in until G1–G4 GOAT gate passes.
#[cfg(feature = "paired_loss_diagnostic")]
pub mod paired_loss;
#[cfg(feature = "paired_loss_diagnostic")]
pub use paired_loss::{
    ClassGapReport, ClassGapRow, ClassSizeBound, CopyNGramTagger, FilterKind, FilterScratch,
    PairedLossGap, TokenClass, TokenTagger,
};

// TEMP — Perturbed-Loss-Vector Diversity Fingerprint (Plan 341, Research 323,
// arXiv:2606.26797 Jin et al. ICML 2026). Modelless diversity selector: given
// two committed snapshots S_0, S_1, extrapolate K checkpoints along v = S_1 − S_0,
// compute per-candidate short-prefix loss vectors, and select the K-subset with
// maximal Lipschitz-bound spread — gradient-diversity ranking without gradients.
// Theorem 3.1 modelless reframe: similar loss vectors across K extrapolated
// checkpoints ⇒ similar gradients along v during the next weight-mutation cycle.
// Composes with ac_prefix::ConditionalLogprob, HLA surprise, RavenSlotLossKernel
// (riir-neuron-db Plan 005). Opt-in until G1–G5 GOAT gate passes.
#[cfg(feature = "temp_loss_fingerprint")]
pub mod diversity;
#[cfg(feature = "temp_loss_fingerprint")]
pub use diversity::temp::{
    LossKernel, extrapolated_snapshot_schedule, lipschitz_gradient_bound, pairwise_bound,
    perturbed_loss_vector, select_diverse_subset,
};
// Plan 367 Fusion C — QMC variant of `extrapolated_snapshot_schedule`.
// Low-discrepancy noise coverage → more diverse loss vectors per unit K.
// Requires both TEMP substrate and the QMC source trait.
#[cfg(all(feature = "temp_loss_fingerprint", feature = "qmc_sampling"))]
pub use diversity::temp::extrapolated_snapshot_schedule_qmc;

// Manifold Bandits — Latent Task Tree + Hierarchical Thompson Sampler +
// BayesianFilterArm (Plan 370, Research 370, arXiv:2606.19750 McKenzie et al.
// UCSD 2026). Modelless inference-time routing primitive: frozen, BLAKE3-
// committable hierarchical clustering of an arm space + top-down Beta posterior
// descent + per-arm non-stationary Bayesian filtering. Closes the contextual +
// non-stationary bandit gap (Plans 030/032/025). The BMC training curriculum
// routes to riir-train; this ships the modelless inference-time routing
// primitive. Opt-in until G1–G5 GOAT gate passes.
#[cfg(feature = "manifold_bandit")]
pub mod manifold_bandit;

// Mean-Field Crowd Oscillation Regime Classifier — crowd-level (κ, κ_a, Q)
// order-parameter aggregator + closed-form 2×2 Jacobian Hopf boundary check +
// four-way regime taxonomy (Static / NoiseSustainedOscillation /
// IrregularSwitching / GlobalLimitCycle). Distilled from Zheng, Miller, Fiete
// (arXiv:2606.30366, MIT, Jun 2026). The paper's algorithmic content is ~80%
// covered by shipped primitives (LinOSS, `subspace_phase_gate`, `temporal_deriv`,
// `MicroRecurrentBeliefState`, `ict::BranchingDetector`); this ships the
// missing 20% — the crowd-scale mean-field view + oscillatory-instability
// detector + regime taxonomy. Extends Plan 301's `subspace_phase_gate` from
// real-eigenvalue phase transitions (`N ≥ d` input sufficiency) to complex-
// eigenvalue (Hopf) phase transitions. Opt-in until the G1–G5 GOAT gate +
// mandatory defend-wrong PoC (Plan 371 Phase 5 T5.1) pass.
#[cfg(feature = "mean_field_regime")]
pub mod mean_field;
#[cfg(feature = "mean_field_regime")]
pub use mean_field::{
    DEFAULT_CLASSIFIER, HopfParams, MeanFieldOverlap, Regime, RegimeClassifier, hopf_boundary,
    static_boundary,
};

// Factorized Transition Action Abstraction — modelless compositional action
// latent primitive distilled from Nam et al., *Latent Actions from Factorized
// Transition Effects under Agent Ambiguity* (arXiv:2606.30544, Brown, 2026-06-30).
// Research 374, Plan 375. The factorized/compositional cousin of the shipped
// monolithic `latent_functor` (riir-ai Plan 273): frozen codebook of K D-dim
// effect primitives + Top-1 patch assignment + sigmoid relevance gate +
// normalized weighted average → compact action latent. Codebook constructed
// modellessly via Lloyd's k-means (Path 2 of AGENTS.md §3.5 — deterministic,
// no gradient descent). Sigmoid gating throughout (NEVER softmax per AGENTS.md
// §2, verified in `otf_lam/model.py::GateNetwork.forward()`). Opt-in until the
// G1–G6 GOAT gate (bench_375_factorized_action_goat) passes.
#[cfg(feature = "factorized_action")]
pub mod factorized_action;
#[cfg(feature = "factorized_action")]
pub use factorized_action::{
    AggregatorType, EffectCodebook, FactorizedActionLatent, FilmProjectionBank, MAX_K, MAX_PATCHES,
    TransitionFactors, aggregate_action_latent_into, factor_token_into, finalize_factors,
    fit_codebook_kmeans_into, motion_input_velocity_into, patchify_1d, relevance_score,
};

// Velocity-Field Ensemble — Algebraic Combination of Pre-Trained Models
// (Plan 376, Research 375, arXiv:2602.20070 Coeurdoux et al. ICML 2026 SPIGM).
// Combine P frozen pre-trained velocity fields (any forward model: LLM
// drafter, HLA forecaster, KARC forecaster, archetype operator field) into a
// single regression-optimal combined drift b̂(x) = Σ_i η_i · b_i(x), where η
// is solved once from N data pairs via the existing linalg::ridge_solve
// P×P Cholesky path (the SAME math KARC uses — KARC's basis is delay-embedded
// features; this primitive's basis is P frozen model forward outputs).
//
// The contribution is the *basis construction*, NOT the ridge solve — anyone
// reviewing should grep `ridge_solve_direct_f32` and confirm KARC's `fit_direct`
// is the same linear-algebra operation. No duplicate math; pure DRY reuse.
//
// η CAN be negative (signed combination, not probabilistic mixture). No
// softmax anywhere; no sigmoid on η either (η is regression-solved, not
// projected). The sigmoid-not-softmax rule applies to *gating*, not to
// regression-optimal weights.
//
// Includes the optimal-diffusion SDE integrator (paper Algorithm 1, eq. 14
// with D*_t = α_t γ_t / β_t) as a decoupled utility — composes with any drift
// source, not just the ensemble.
//
// Opt-in until the G1–G4 GOAT gate (Plan 376 Phase 3) passes. G2 (cross-domain
// quality) is the make-or-break gate — the paper proves cross-domain
// composition for image generation only; Phase 2 PoC is mandatory before any
// quality-parity claim for game AI.
#[cfg(feature = "velocity_field_ensemble")]
pub mod velocity_field_ensemble;
#[cfg(feature = "velocity_field_ensemble")]
pub use velocity_field_ensemble::{
    ClosureField, EnsembleFitScratch, Schedule, VelocityField, VelocityFieldEnsemble,
    accumulate_pair_into, stochastic_interpolant_step_into,
};

// ── Phase 10 absorption (Proposal 003, 2026-07-04): modules moved from katgpt-rs/src/.
// Always-on (no feature gate):
pub mod alloc; // Debug-only TrackingAllocator (consumer gates via #[cfg(debug_assertions)])
pub mod cumprodsum; // Cumprodsum primitive (Plan 263) — always-on
pub mod trigger_gate; // Compute-tier trigger gate — always-on
// ── Phase 12 absorption (Proposal 003, 2026-07-04): more modules moved from katgpt-rs/src/.
// Feature-gated (mirror root feature names):
#[cfg(feature = "critical_interval_gate")]
pub mod dllm_solver; // Discrete Critical Interval Solver Switching (Plan 222)
#[cfg(feature = "modality_pruned_load")]
pub mod pipeline_pruner; // Pipeline Pruner — modality-aware inference pipeline selection (Plan 227 Phase 3)
// ── Phase 12 T4.3: folder moves from katgpt-rs/src/.
#[cfg(feature = "breakeven_routing")]
pub mod breakeven;
#[cfg(feature = "closed_unit_compaction")]
pub mod compaction; // Closed-Unit Compaction Gate — CUCG (Plan 333)
#[cfg(feature = "cubical_nerve")]
pub mod cubical_nerve; // CubicalNerve CAT(0) cubical complexes (Plan 252 Phase 3)
#[cfg(feature = "mux_latent_context")]
pub mod mux_latent; // MUX-Latent Context Compression (Research 211, Plan 238) // Breakeven complexity cost-aware routing (Plan 250)
// Feature-gated (mirror root feature names):
#[cfg(feature = "cce_moderator")]
pub mod cce;
#[cfg(feature = "llmexec_guard")]
pub mod llmexec_guard;
#[cfg(feature = "memory_soup_lora")]
pub mod memory_soup_lora;
#[cfg(feature = "mux_demux")]
pub mod mux_demux;
#[cfg(feature = "salience_tri_gate")]
pub mod salience;
#[cfg(feature = "salience_tri_gate")]
pub use salience::{
    DelegateToken, FoldbackTarget, SalienceDecision, SalienceTriGate, SilenceToken,
};
#[cfg(feature = "channel_simd_align")]
pub mod channel_simd;
#[cfg(feature = "skill_opt")]
pub mod skill_opt;
#[cfg(feature = "ssd_block")]
pub mod ssd_block;

// Test-only `#[global_allocator]` so `alloc::tests::*` pass when running
// `cargo test -p katgpt-core --lib`. Downstream consumers (katgpt-rs root,
// riir-engine, etc.) install their OWN `#[global_allocator]`; this static is
// `cfg(test)` so it does not exist when katgpt-core is consumed as a library
// dep — no double-declare conflict. Mirrors the root crate's
// `static GLOBAL_ALLOC: TrackingAllocator` (src/lib.rs:356).
#[cfg(all(test, debug_assertions))]
#[global_allocator]
static TEST_GLOBAL_ALLOC: alloc::TrackingAllocator = alloc::TrackingAllocator;
