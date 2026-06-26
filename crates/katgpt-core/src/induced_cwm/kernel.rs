//! `InducedCwmKernel` — marker trait for verifiable, committable forward models.
//!
//! Paper §3.1 (arxiv 2510.04542): a Code World Model is an LLM-induced executable
//! Python module implementing the game's forward model (`apply_action`,
//! `get_legal_actions`). In our port, the analogue is a `GameState` impl whose
//! transition function was produced by some offline process (LLM, hand-coded,
//! distilled — the kernel trait is agnostic).
//!
//! This trait adds three capabilities on top of `GameState`:
//!
//! 1. **Verifiable** — passes the auto-generated `TransitionUnitTest`s produced
//!    by [`crate::induced_cwm::unit_test::make_transition_tests_from_trajectory`].
//! 2. **Committable** — BLAKE3 over a canonical byte representation, via
//!    [`commitment`](InducedCwmKernel::commitment).
//! 3. **Hot-swappable** — atomic `Arc` swap is handled at the slot layer
//!    (Phase 4, mirrors `micro_belief::MicroRecurrentKernelSnapshot` / Plan 092).
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! The kernel's *transition function* is raw and deterministic — `advance()`
//! produces a bit-identical successor for `(state, action, player_id)` across
//! re-runs. The *commitment bytes* are also raw (they go through chain consensus
//! in riir-ai Plan 326). The LLM prompts/intermediate artifacts that produced
//! the kernel are latent and never cross this trait boundary.
//!
//! # What this trait does NOT require
//!
//! - It does NOT require the implementor to expose the LLM, prompts, or
//!   refinement tree. Those are cold-tier concerns (riir-ai Plan 326).
//! - It does NOT require serialisable state — only a canonical form for the
//!   *kernel itself* (rule schema, action enum, source/bytecode), not for
//!   every game state. Two distinct game states in the same kernel share the
//!   same `canonical_bytes`.
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/296_induced_cwm_kernel_primitive.md`]
//! - Research: [`katgpt-rs/.research/275_Code_World_Model_Induced_Forward_Model.md`]
//! - Source paper: [arxiv 2510.04542](https://arxiv.org/pdf/2510.04542)

use crate::traits::GameState;

/// Marker trait for forward-model impls that are verifiable, committable, and
/// hot-swappable.
///
/// See the [module docs](self) for the full contract. The trait surface is
/// intentionally minimal: implementors only need to provide
/// [`canonical_bytes`](InducedCwmKernel::canonical_bytes) plus the existing
/// `GameState` impl. How the impl was produced (hand-written, LLM-induced,
/// distilled) is the integrator's concern.
///
/// # Determinism contract
///
/// `canonical_bytes` MUST be deterministic across runs: the same logical kernel
/// (same rule schema + same action enum + same rule source/bytecode) MUST hash
/// to the same 32-byte BLAKE3. This is what makes
/// [`commitment`](InducedCwmKernel::commitment) usable as a chain-consensus
/// commitment in riir-ai Plan 326. Non-deterministic serialisation (e.g.
/// `HashMap` iteration order leaking into bytes) breaks quorum — see the test
/// `canonical_bytes_determinism` in [`crate::induced_cwm::tests`].
pub trait InducedCwmKernel: GameState {
    /// Canonical byte representation of the *kernel* (not the state).
    ///
    /// SHOULD cover: state schema, action enum, rule source/bytecode. MUST be
    /// deterministic. Integrators decide the canonical form — this trait only
    /// enforces "same logical kernel → same bytes" via the BLAKE3 wrapper below.
    ///
    /// Returning a freshly-allocated `Vec` is acceptable: this is a cold-tier
    /// call (one per induction event), never on the 20Hz hot path. The hot
    /// path uses [`commitment`](Self::commitment) on a pre-cached `[u8; 32]`.
    fn canonical_bytes(&self) -> Vec<u8>;

    /// BLAKE3 over [`canonical_bytes`](Self::canonical_bytes).
    ///
    /// Convenience wrapper; implementors normally do not override this. The
    /// default impl is a single `blake3::hash` call — O(|canonical_bytes|), no
    /// allocation beyond the hasher's internal buffer.
    fn commitment(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_bytes()).as_bytes()
    }
}
