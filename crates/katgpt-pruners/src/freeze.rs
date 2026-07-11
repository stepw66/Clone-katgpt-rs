//! Re-export of [`katgpt_core::freeze`] for backwards compatibility.
//!
//! Extracted to `katgpt-core` (Plan 388 Phase 1) to break the
//! katgpt-pruners ↔ katgpt-speculative cycle. Pure stdlib (Path + fs + mem),
//! no pruners-specific knowledge. Existing `katgpt_pruners::freeze::*` and
//! `crate::pruners::freeze::*` paths resolve unchanged.

pub use katgpt_core::freeze::{load_frozen, save_frozen};
