//! CGSP re-export from `katgpt-core` (Plan 274, Research 240).
//!
//! The `cgsp` module was moved into `katgpt-core` so that `riir-engine`
//! (Plan 299) can consume the modelless Solver / Conjecturer / Guide triad
//! via `katgpt-core/cgsp` without depending on the root application crate.
//! This file preserves the historical `katgpt::cgsp::...` import paths
//! (including the `traits` / `types` submodules used by the Plan 274 GOAT
//! benchmarks) by re-exporting katgpt-core's module verbatim.
//!
//! See `crates/katgpt-core/src/cgsp/` for the actual implementation.

pub use katgpt_core::cgsp::{
    BatchQualityGate, BreakevenDifficultyFilter, Candidate, CgspConfig, CgspLoop,
    ColinearityBatchGate, CollapseSignal, ComplexityWeights, CuriosityConjecturer,
    CuriosityPrioritySnapshot, CycleResult, CycleStats, Direction, DifficultyFilter,
    EntropyCollapse, HlaProjectionGuide, HintDeltaBandit, NoOpBatchGate, NoOpDifficultyFilter,
    PoolConjecturer, Priority, QualityGuide, ScratchBuffers, Solver, SolveRate, Target,
    DEFAULT_HLA_DIM, DEFAULT_K, DEFAULT_POOL_SIZE, entropy_nats, sigmoid, structural_complexity,
};

// Dual-pool reachable memory router (Plan 282, Research 249 — DecentMem).
// Re-exported under the `cgsp_dual_pool` feature so `katgpt::cgsp::DualPoolBandit`
// resolves from the root crate.
#[cfg(feature = "cgsp_dual_pool")]
pub use katgpt_core::cgsp::{DualPoolBandit, DualPoolConfig, PoolId, ReachableDualPoolRouter};

/// Re-export the `traits` submodule so `katgpt::cgsp::traits::*` paths from
/// Plan 274's GOAT benchmark keep resolving.
pub mod traits {
    pub use katgpt_core::cgsp::traits::*;
}

/// Re-export the `types` submodule so `katgpt::cgsp::types::*` paths keep
/// resolving.
pub mod types {
    pub use katgpt_core::cgsp::types::*;
}
