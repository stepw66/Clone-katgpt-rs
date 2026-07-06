//! `product_key_memory` — O(√N) Factored Retrieval Memory (open primitive).
//!
//! Plan 408 / Research 387. Implements the **retrieval factorization** half of
//! the FwPKM paper (Lample et al. 2019 §2.2; Zhao & Jones 2026 distillation).
//! The gradient-descent half of the paper is forbidden per AGENTS.md
//! constraint #1 and replaced by the shipped δ-rule analog (Plan 053) —
//! this primitive implements ONLY the inference-time factored retrieval.
//!
//! # The complexity-class gap this fills
//!
//! The katgpt retrieval stack today has four distinct complexity classes:
//!
//! | Retriever | Cost | Slot ceiling | Sparsity axis |
//! |---|---|---|---|
//! | Raven RSM | O(1) routing | ~10³ experts | conditional computation |
//! | Engram | O(1) hash | ~10⁵ slots (hash-collides above) | content-addressed |
//! | δ-Mem | O(r) associative | rank-r bounded | associative |
//! | **PKM (this crate)** | **O(√N) factored** | **~10⁶ slots** | **similarity-ranked** |
//!
//! PKM is the only retriever that scales to ~10⁶ slots at sub-linear cost. It
//! splits a `d_k`-dim query into two `d_k/2`-dim halves, scores two √N-row
//! codebooks, and takes the top-k of the `k×k` Cartesian product — yielding
//! `2√N + k²` scoring cost instead of `N`.
//!
//! # Modelless mandate (§3.5)
//!
//! The FwPKM paper's `L_mem` GD on value rows, `L_addr` GD on keys, and the
//! n-iter TTT loop are ALL forbidden. They are replaced by shipped substrates:
//!
//! | Forbidden paper mechanism | Modelless replacement (shipped) |
//! |---|---|
//! | `L_mem` GD on V | `DeltaMemoryState::write_segment` δ-rule (Plan 053) |
//! | `L_addr` GD on K | TEMP `sleep_diverse` diversity selector (Plan 005) |
//! | n-iter TTT loop | Sleep Consolidation N-pass (Plan 154) |
//!
//! The optional δ-rule write path over the PKM value table lands in Phase 5
//! (`product_key_memory_episodic`) and is bit-identical to one GD step at
//! η=1 — but is NOT iterated, so it is modelless.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! - Slot keys + values (latent patterns) → latent, frozen, BLAKE3-committed.
//! - PKM table commitment root (Phase 4) → raw, syncable audit artifact.
//! - Top-k weights → latent scalar; bridge at the sync boundary.
//!
//! # Phase status (this module)
//!
//! - ✅ Phase 1 (skeleton + types) — [`types`].
//! - ⏳ Phase 2 (retrieval kernel) — [`kernel`] (stub).
//! - ⏳ Phase 3 (GOAT gate G1–G4) — `.benchmarks/408_pkm_goat.md`.
//! - ⏳ Phase 4 (freeze/thaw wrapper) — `freeze.rs`.
//! - ⏳ Phase 5 (δ-rule write gate, F1 fusion) — `episodic.rs`.
//!
//! # CRITICAL — never softmax at the *gate* level
//!
//! Per AGENTS.md, every *probability/relevance gate* in this codebase uses
//! sigmoid, not softmax. The top-k *normalization* within PKM (paper §2.2) is
//! a different concern: it is a ranking normalization over the k²-restricted
//! candidate set, NOT a probability claim. Plan 408 T2.1 step 6 documents the
//! deviation from the global sigmoid rule for this restricted case and keeps
//! softmax there for ranking fidelity vs the paper's reference implementation.
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/408_Product_Key_Memory_Primitive.md`]
//! - Research: [`katgpt-rs/.research/387_Fast_Weight_Product_Key_Memory_PKM.md`]
//! - Source paper: [arXiv:2601.00671](https://arxiv.org/abs/2601.00671) —
//!   Zhao & Jones, "Fast-weight Product Key Memory", Sakana AI, Feb 2026
//!   (distills the PKM factorization from Lample et al. 2019 §2.2).

pub mod kernel;
pub mod types;

pub use kernel::{PkmScratch, score_dot, score_idw};
pub use types::{
    ProductKeyMemory, PkEntry, PkQuery, ScoreFn, D_K_FLOOR, SQRT_N_FLOOR,
};
