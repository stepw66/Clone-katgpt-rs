//! D2F Drafter Verifier — root re-export shim.
//!
//! Plan 399 (2026-07-05): the full module (production + tests) moved to
//! `katgpt_forward::d2f_verifier`. This file is a thin re-export shim that
//! preserves every historical `crate::speculative::d2f_verifier::*` import
//! path. The historical "Root-resident by design (Issue 033 §C)" comment
//! was obsolete — all three blockers (root-only `crate::dllm`,
//! `crate::transformer::forward`, `crate::speculative::verifier`) now resolve
//! to leaf crates. See `katgpt-forward/src/d2f_verifier.rs` for the
//! implementation.

pub use katgpt_forward::d2f_verifier::D2fDrafterVerifier;
