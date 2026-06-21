//! The trait that any "latent direction source" must implement to participate
//! in [`crate::personality_composition::kernel::PersonalityWeightedComposition`].
//!
//! # Why a trait, not a struct?
//!
//! The composition kernel is generic over its inputs — it doesn't care
//! whether a direction comes from:
//!
//! - a `micro_belief::AttractorKernel` (R242 — implicit microcognition)
//! - a `LatentThoughtKernel` (R242 Family B — K-hypothesis attention)
//! - a frozen text embedding (e.g. a zone description, in the host)
//! - a hard-coded "compass" direction (e.g. `COMPANION(player)` in the
//!   Entity Cognition Stack guide R146)
//! - a runtime-computed blend (e.g. flock centroid direction)
//!
//! As long as the source can produce a D-dim unit-ish direction vector on
//! demand and report a belief confidence, it can be composed. This matches
//! the broader crate philosophy: modelless primitives, host fills in
//! semantics.
//!
//! # File-name note
//!
//! This file is named `direction_source.rs`, not `trait.rs`, because `trait`
//! is a reserved keyword in Rust 2021 and cannot be used as a module path
//! component. `mod.rs` re-exports the trait at the module root so callers
//! write `crate::personality_composition::LayerDirectionSource` — the
//! file-name choice is invisible from outside.

/// A source of D-dimensional latent direction vectors that can be composed
/// by [`crate::personality_composition::kernel::PersonalityWeightedComposition`].
///
/// Each call to [`direction`](Self::direction) must produce a vector that the
/// caller treats as having *some* magnitude; the kernel does NOT normalize
/// (the host decides whether unit-norm, magnitude-weighted, or raw
/// direction-magnitude semantics apply). The vector lives in the `scratch`
/// buffer provided by the caller — this is the **zero-alloc hot path**
/// requirement (Plan 297 T4.1).
///
/// `belief_confidence` is a scalar in `[0, 1]` that gates how much each
/// layer's contribution makes it into the output. The kernel multiplies it
/// in: `out[j] += sigmoid(wᵢ/τ) · belief_confidence · d[j]`. Defaults to
/// `1.0` for plasma-tier layers (those that the host always trusts).
pub trait LayerDirectionSource {
    /// Write the current direction into `scratch` and return a slice of it.
    ///
    /// The returned slice MUST be `&scratch[..D]` (or a prefix of `scratch`).
    /// Returning a slice that aliases a different buffer is undefined
    /// behaviour for the kernel's borrow-check story.
    ///
    /// `scratch` must be at least `D` elements long; the kernel passes a
    /// buffer of exactly `D` so any excess capacity is unused.
    fn direction(&self, scratch: &mut [f32]) -> &[f32];

    /// Most recent direction the source produced.
    ///
    /// Used by the drift rule (`Δwᵢ = α · surprise · d_recent`) so weights
    /// move in the direction that *was* active when reward was observed,
    /// not the direction that *is* active now.
    ///
    /// Default impl returns an empty slice — sources that don't track
    /// history will get zero drift. Concrete sources should override.
    fn recent_direction(&self) -> &[f32] {
        &[]
    }

    /// Belief confidence scalar in `[0, 1]`. Default `1.0` (plasma tier).
    ///
    /// The host uses this to attenuate layers it doesn't currently trust
    /// (e.g. a perception layer when the entity is blinded, or a social
    /// layer when the entity is alone). The kernel multiplies it in
    /// verbatim — it is NOT passed through sigmoid, so `belief_confidence = 0`
    /// genuinely zeros the layer's contribution (unlike `w → -∞`, which the
    /// `w_max` clamp prevents).
    fn belief_confidence(&self) -> f32 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial source that always reports the same direction. Useful as a
    /// base for testing the trait's default methods.
    struct ConstDir {
        d: [f32; 4],
    }

    impl LayerDirectionSource for ConstDir {
        fn direction(&self, scratch: &mut [f32]) -> &[f32] {
            scratch[..self.d.len()].copy_from_slice(&self.d);
            &scratch[..self.d.len()]
        }
    }

    #[test]
    fn default_recent_direction_is_empty() {
        let src = ConstDir {
            d: [1.0, 0.0, 0.0, 0.0],
        };
        assert!(src.recent_direction().is_empty());
    }

    #[test]
    fn default_belief_confidence_is_one() {
        let src = ConstDir {
            d: [1.0, 0.0, 0.0, 0.0],
        };
        assert_eq!(src.belief_confidence(), 1.0);
    }

    #[test]
    fn direction_writes_into_scratch() {
        let src = ConstDir {
            d: [0.5, -0.5, 1.0, 0.25],
        };
        let mut scratch = [0.0f32; 4];
        let out = src.direction(&mut scratch);
        assert_eq!(out, &[0.5, -0.5, 1.0, 0.25]);
        // Slice must alias scratch.
        assert_eq!(out.as_ptr(), scratch.as_ptr());
    }
}
