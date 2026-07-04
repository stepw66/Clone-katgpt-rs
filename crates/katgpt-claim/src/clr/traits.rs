//! CLR trait surface — extractor, verifier, direction source (Plan 284).
//!
//! These three traits are the seams between the CLR runtime and the
//! domain-specific code that supplies claim embeddings and direction vectors.
//! The runtime itself is generic over `T` (the claim payload type) and is
//! deliberately decoupled from any specific model or tokenizer.

use crate::clr::types::{Claim, Trajectory, Verdict};

/// Extracts exactly M claims from a trajectory.
///
/// Returns exactly `M` claims (where `M` is configured on the implementing
/// type — see [`crate::clr::extractor::FnClaimExtractor`]). The caller asserts
/// length; mis-sized output is a programmer error and trips a `debug_assert`.
pub trait ClaimExtractor<T> {
    /// Extract claims from `trajectory`. Length must equal the configured `M`.
    fn extract(&self, trajectory: &Trajectory<T>) -> Vec<Claim<T>>;
}

/// Verifies a single claim against one projection direction.
///
/// Returns `sigmoid(dot(claim.embedding, direction_vec[direction_idx]))`.
/// `direction_idx` must be in `[0, M)`. The scalar output is bounded in
/// `(0, 1)` and is the atomic unit of reliability aggregation.
pub trait ClaimVerifier<T> {
    /// Verdict for `claim` projected onto direction `direction_idx`.
    fn verify(&self, claim: &Claim<T>, direction_idx: usize) -> Verdict;
}

/// Freeze/thaw-versioned direction vector pool.
///
/// Supplies the `M` projection directions used by [`ClaimVerifier`]. The
/// `blake3` + `version` pair lets downstream consumers detect direction-vector
/// drift across freeze/thaw cycles without re-reading the full pool.
///
/// Implementors MUST guarantee that `direction(idx)` for a fixed `version`
/// returns a byte-identical slice — verdict reproducibility depends on it.
pub trait DirectionVectorSource {
    /// Borrow the direction vector at `idx`. Length must equal configured `k`.
    fn direction(&self, idx: usize) -> &[f32];
    /// BLAKE3 hash of the full direction pool (all `M` vectors concatenated).
    fn blake3(&self) -> [u8; 32];
    /// Monotonic freeze/thaw version. Bumps on every direction-vector update.
    fn version(&self) -> u64;
}
