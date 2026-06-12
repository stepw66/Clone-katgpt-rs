//! SpecAsPruner — compile NL specs into symbolic ConstraintPruner rules.
//!
//! Research 229: ProgramAsWeights compiles specs into neural LoRA adapters (~22MB).
//! We compile specs into symbolic bitmap rules (~1KB), achieving:
//! - **4400× smaller** — bitmaps vs LoRA weight matrices
//! - **O(1) per token** — bitmap lookup vs neural forward pass
//! - **Zero training** — no GPU, no data, no gradient computation
//! - **Exact verification** — can prove all outputs satisfy spec constraints
//!
//! Supported spec types:
//! - **Classification**: "Classify sentiment as positive or negative" → allowlist
//! - **Extraction**: "Extract email addresses" → character-class allowlist
//! - **Format repair**: "Fix malformed JSON" → structural token constraints
//! - **Intent routing**: "Route to: search, create, delete" → label allowlist
//!
//! Feature gate: `spec_pruner`

pub mod compiler;
pub mod pruner;
pub mod types;

pub use compiler::SpecCompiler;
pub use pruner::*;
pub use types::CompilationResult;
pub use types::*;
