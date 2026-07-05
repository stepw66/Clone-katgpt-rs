//! Decision-Diffusion Tree (DDTree) for speculative decoding — root shim.
//!
//! Plan 396 (2026-07-05): the full dd_tree module (production + ~2380 LOC of
//! integration tests) moved to `katgpt_forward::dd_tree`. The core algorithm
//! lives in `katgpt_speculative::dd_tree` (re-exported via the glob below).
//! This file is a thin re-export shim that preserves every historical
//! `crate::speculative::dd_tree::*` import path.
//!
//! The two feature-gated production fns (`build_dd_tree_screened_with_schedule`,
//! `build_dd_tree_gdsd`) are re-exported from katgpt-forward; the ~2380-LOC
//! integration test module moved with them (the tests exercise the full
//! dd_tree + dflash_predict pipeline, both resident in katgpt-forward).

pub use katgpt_forward::dd_tree::*;

// The two feature-gated wrappers are re-exported explicitly so the historical
// `crate::speculative::dd_tree::build_dd_tree_screened_with_schedule` /
// `build_dd_tree_gdsd` paths resolve without callers needing to know they now
// live in katgpt-forward.
#[cfg(feature = "thinking_prune")]
pub use katgpt_forward::build_dd_tree_screened_with_schedule;
#[cfg(feature = "gdsd_distill")]
pub use katgpt_forward::build_dd_tree_gdsd;
