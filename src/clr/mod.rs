//! CLR — Claim-Level Reliability runtime (Plan 284, Research 255).
//!
//! Phase 1 scaffolding: primitive types, traits, a closure-backed claim
//! extractor, a sigmoid-projection verifier, and the Long2Short brevity
//! tiebreak. The full voter / MGPO / curiosity arms live in later phases
//! (Plan 284 Phase 2+) and are intentionally out of scope here.
//!
//! ## Design
//!
//! The runtime is generic over the claim payload type `T`. The only thing CLR
//! actually needs to compute a verdict is `claim.embedding: &[f32]`; the
//! payload is carried through opaquely so the same voter can score reasoning
//! traces (T = CoT text), game episodes (T = action sequence), tool calls
//! (T = call/response pair), etc.
//!
//! ## Sigmoid-only rule
//!
//! All activations in this module are sigmoid. There is no softmax anywhere
//! (per project convention + the user's `AGENTS.md` rule). This matters for
//! freeze/thaw stability: softmax is not separable across directions, while
//! sigmoid projections are.
//!
//! ## Latent-space contract
//!
//! Claim embeddings and direction vectors are latent (semantic domain). They
//! are NEVER used as raw physical values, NEVER synced across nodes directly,
//! and NEVER fed into deterministic replay. See the user's `AGENTS.md`
//! "Latent vs Raw Space Rules" for the full contract.

pub mod brevity;
pub mod extractor;
pub mod learning_potential;
pub mod mgpo;
pub mod scratch;
pub mod traits;
pub mod types;
pub mod verifier;
pub mod vote;

pub use brevity::brevity_tiebreak;
pub use extractor::FnClaimExtractor;
pub use learning_potential::{learning_potential, should_write_memory};
pub use mgpo::{allocate_budget, mgpo_sampling_weight};
pub use scratch::ClrScratch;
pub use traits::{ClaimExtractor, ClaimVerifier, DirectionVectorSource};
pub use types::{Claim, ClrConfig, Cluster, ReliabilityScore, Trajectory, Verdict, VoteResult};
pub use verifier::SigmoidProjectionVerifier;
pub use vote::{clr_vote, clr_vote_minimal};
