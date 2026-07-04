//! Hierarchical GOAT Proof Certificates (Plan 145, Research 106).
//!
//! Standalone, serializable proof certificates with dependency chains,
//! topological verification, and blake3 checksum integrity.
//!
//! Extracted from `katgpt-rs/src/proof_cert/` per Proposal 003 Phase 12
//! (2026-07-04). All five files moved as a unit; deps are `serde` +
//! `postcard` + `blake3` (all always-on — the binary persistence format
//! includes a blake3 checksum). The `wasm_proof_witness` feature gates
//! the witness-generation subset (no extra deps — blake3 already on).
//!
//! # Module map
//!
//! - `certificate` — `ProofCertificate`, `ProofEvidence`, `ProofProperty`,
//!   `ProofResult` core types.
//! - `chain` — `verify_proof_chain` topological verifier.
//! - `serde_impls` — `load_certificates` / `save_certificates` /
//!   `verify_checksum` BLAKE3-integrity-checked file I/O.
//! - `wasm_certificates` — `generate_wasm_validator_certificates` for
//!   validator proof bundles.
//! - `wasm_proof_witness` (gated `wasm_proof_witness` feature) —
//!   `WasmProofWitness` + `generate_wasm_witness_certificates`. The gated
//!   feature adds the `blake3` dep.
//! - `macros` — `conditional_proof!` declarative macro (exported at crate
//!   root via `#[macro_export]`).

#![allow(unexpected_cfgs)]  // root may pass-through aggregate features like `full`

mod certificate;
mod chain;
mod macros;
mod serde_impls;
mod wasm_certificates;

#[cfg(feature = "wasm_proof_witness")]
mod wasm_proof_witness;

pub use certificate::{ProofCertificate, ProofEvidence, ProofProperty, ProofResult};
pub use chain::{ProofChainResult, verify_proof_chain};
pub use serde_impls::{load_certificates, save_certificates, verify_checksum};
pub use wasm_certificates::generate_wasm_validator_certificates;

#[cfg(feature = "wasm_proof_witness")]
pub use wasm_proof_witness::{WasmProofWitness, generate_wasm_witness_certificates};
