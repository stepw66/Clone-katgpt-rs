//! `CwmCommitment` — the BLAKE3-committed induction-event artifact.
//!
//! Produced when an [`InducedCwmKernel`](crate::induced_cwm::InducedCwmKernel)
//! is induced (offline / cold-tier). Crosses the sync boundary as an audit
//! event in riir-ai Plan 326 — clients can verify "the entity I observe is
//! running kernel X" without learning the kernel's source/weights.
//!
//! # Field semantics
//!
//! - `blake3` is the canonical commitment over the kernel's
//!   [`canonical_bytes`](crate::induced_cwm::InducedCwmKernel::canonical_bytes).
//!   Two kernels with identical canonical bytes MUST produce identical `blake3`.
//! - `version` is a caller-managed monotonic counter for the *contents ordinal*
//!   of a personality. NOT part of the BLAKE3 input — see the
//!   [`micro_belief::MicroRecurrentKernelSnapshot`] precedent: two snapshots
//!   with identical bytes but different versions are the *same* kernel at
//!   different points in time.
//! - `created_at_tick` is the GameState tick at which the induction event
//!   occurred. Raw, deterministic, replayable. Distinct from `version` (which
//!   is per-entity monotonic) — `created_at_tick` is global clock.
//!
//! # Deviation from Plan 296 T1.4 (UUID)
//!
//! The plan called for `snapshot_id: Uuid` using `Uuid::now_v7()` per AGENTS.md.
//! We follow the established codebase convention instead — `micro_belief/
//! snapshot.rs`, `funcattn_compose/freeze_thaw.rs`, and `pruners/proof/
//! sketch_types.rs` all defer UUID to the *swap event layer* (riir-ai Plan 326
//! `KernelHotSwap`) and use a plain `u64 version` for the contents ordinal.
//! The `uuid` crate is not currently a dependency of katgpt-core or katgpt-rs;
//! adding it for a single field that nobody reads at this layer would be
//! scope-creep. The hot-swap slot (Phase 4) is the natural home for a v7 UUID
//! event tag — same place `micro_belief` puts it.
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/296_induced_cwm_kernel_primitive.md`] §Phase 4
//! - Precedent: `katgpt-rs/crates/katgpt-core/src/micro_belief/snapshot.rs`
//! - AGENTS.md rule cited: "Use `Uuid::now_v7()` not `Uuid::new_v4()`"

/// BLAKE3-committed induction-event artifact.
///
/// Construct via [`from_kernel`](Self::from_kernel) (computes BLAKE3) or
/// [`from_parts`](Self::from_parts) (raw fields, for deserialisation paths).
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CwmCommitment {
    /// BLAKE3 over the kernel's canonical bytes. Two kernels with identical
    /// canonical bytes produce identical `blake3`.
    pub blake3: [u8; 32],
    /// Caller-managed per-entity monotonic contents ordinal. NOT part of the
    /// BLAKE3 input.
    pub version: u64,
    /// Global GameState tick at which the induction event occurred. Raw,
    /// deterministic, replayable.
    pub created_at_tick: u64,
}

impl CwmCommitment {
    /// Build a commitment from a kernel.
    ///
    /// `kernel` is any `InducedCwmKernel`; we only call
    /// [`commitment`](crate::induced_cwm::InducedCwmKernel::commitment) on it.
    /// `version` and `created_at_tick` are caller-managed.
    pub fn from_kernel<K: crate::induced_cwm::InducedCwmKernel>(
        kernel: &K,
        version: u64,
        created_at_tick: u64,
    ) -> Self {
        Self {
            blake3: kernel.commitment(),
            version,
            created_at_tick,
        }
    }

    /// Build a commitment from raw parts WITHOUT recomputing BLAKE3.
    ///
    /// Useful for deserialisation paths where the hash is already known
    /// (e.g. loading from disk). The caller is responsible for ensuring
    /// `blake3` actually matches the kernel's canonical bytes — there is no
    /// `verify()` here because this struct does not store the bytes themselves
    /// (the kernel does).
    pub fn from_parts(blake3: [u8; 32], version: u64, created_at_tick: u64) -> Self {
        Self {
            blake3,
            version,
            created_at_tick,
        }
    }

    /// Returns `true` iff this commitment matches the given kernel's current
    /// canonical bytes.
    ///
    /// Cheap (one BLAKE3 over `canonical_bytes`). Use this when verifying
    /// that an externally-loaded commitment still corresponds to the kernel
    /// the entity claims to be running.
    pub fn matches_kernel<K: crate::induced_cwm::InducedCwmKernel>(&self, kernel: &K) -> bool {
        kernel.commitment() == self.blake3
    }
}
