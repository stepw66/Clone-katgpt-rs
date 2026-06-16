//! Core types for the FaithfulnessProbe causal intervention suite.
//!
//! See Research 244 §4 and Plan 278 for methodology.

/// Causal intervention variant applied to an injected memory segment.
///
/// Each variant destroys a different aspect of the memory:
/// - `Empty` — content removed (zero-fill), format/length preserved.
/// - `Shuffle` — temporal/causal structure destroyed (Fisher-Yates).
/// - `Corrupt` — internal coherence broken (random element displacement).
/// - `Irrelevant` — replaced with same-format unrelated content (external pool).
/// - `Filler` — replaced with a semantically-empty placeholder constant.
///
/// `#[repr(u8)]` guarantees 1-byte size — zero-allocation on hot paths.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Intervention {
    Empty = 0,
    Shuffle = 1,
    Corrupt = 2,
    Irrelevant = 3,
    Filler = 4,
}

/// Aggregated behavioral deltas across the intervention suite.
///
/// POD — fixed-size, `Copy`, no heap. Generic over the delta metric `D`
/// (typically `f32` or `f64`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaithfulnessProfile<D> {
    /// Delta from the `Empty` intervention (content zeroed). A faithful consumer
    /// falls back to baseline when content is absent, so this should be small.
    pub empty_delta: D,
    /// Max of `Shuffle` and `Corrupt` deltas — structural/coherence disruption.
    pub shuffle_or_corrupt_delta: D,
    /// Delta from the `Irrelevant` intervention (unrelated content substituted).
    pub irrelevant_delta: D,
    /// Delta from the `Filler` intervention (placeholder constant injected).
    pub filler_delta: D,
}

impl<D> FaithfulnessProfile<D>
where
    D: PartialOrd + Copy + Default,
{
    /// Strict faithfulness verdict per Plan 278 / Research 244 §4.
    ///
    /// Returns `true` iff ALL of:
    /// - `empty_delta < threshold` — consumer ignores emptying (graceful absence).
    /// - `shuffle_or_corrupt_delta > threshold` — reacts to structural disruption.
    /// - `irrelevant_delta > threshold` — reacts to unrelated content.
    /// - `filler_delta > threshold` — reacts to placeholder injection.
    ///
    /// A faithfully-used memory produces large deltas for meaningful
    /// interventions but a small delta for emptying (the consumer falls back
    /// to baseline behavior when content is absent).
    #[inline]
    pub fn is_faithfully_used(&self, threshold: D) -> bool {
        self.empty_delta < threshold
            && self.shuffle_or_corrupt_delta > threshold
            && self.irrelevant_delta > threshold
            && self.filler_delta > threshold
    }
}

/// Minimal interface for a memory consumer to expose its behavioral surface.
///
/// Uses associated types for `Behavior`, `Delta`, and `Memory` to keep the
/// type story clean (Plan 278 T1.4: "associated type Memory"). This also lets
/// `DefaultFaithfulnessProbe<C>` stay generic over only `C` — generic params
/// for B/D would leak into every call site.
pub trait ConsumerContext {
    /// Behavior representation (e.g. `f32` logits, action enum, `Vec<f32>`).
    type Behavior;
    /// Delta metric — must support ordering and be `Copy` (e.g. `f32`).
    type Delta: PartialOrd + Copy + Default;
    /// Memory representation this consumer reads (e.g. `Vec<f32>`).
    type Memory;

    /// Behavior with no memory injected (the prior / fallback).
    fn baseline_behavior(&self) -> Self::Behavior;

    /// Behavior given a memory segment.
    fn behavior_with_memory(&self, memory: &Self::Memory) -> Self::Behavior;

    /// Distance/delta between two behaviors. Should be non-negative for
    /// meaningful threshold comparison.
    fn behavior_delta(&self, a: &Self::Behavior, b: &Self::Behavior) -> Self::Delta;
}

/// Trait for memory types that expose a mutable element slice for perturbation.
///
/// Implement this for your memory representation to enable
/// [`DefaultFaithfulnessProbe`](super::probe::DefaultFaithfulnessProbe).
/// `Vec<T>` implements this out of the box.
pub trait MemorySlice {
    /// Element type (e.g. `f32`, token id, latent dimension value).
    type Elem: Clone + Default;

    /// Borrow as an immutable element slice.
    fn mem_as_slice(&self) -> &[Self::Elem];

    /// Borrow as a mutable element slice for in-place perturbation.
    fn mem_as_mut_slice(&mut self) -> &mut [Self::Elem];
}

impl<T> MemorySlice for Vec<T>
where
    T: Clone + Default,
{
    type Elem = T;

    #[inline]
    fn mem_as_slice(&self) -> &[T] {
        self.as_slice()
    }

    #[inline]
    fn mem_as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

// ---------------------------------------------------------------------------
// Unit tests — Plan 278 T1.8
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn test_intervention_enum_repr_u8() {
        // Plan 278 T1.8: size is 1 byte.
        assert_eq!(size_of::<Intervention>(), 1);
        // Verify discriminant values for stable serialization.
        assert_eq!(Intervention::Empty as u8, 0);
        assert_eq!(Intervention::Shuffle as u8, 1);
        assert_eq!(Intervention::Corrupt as u8, 2);
        assert_eq!(Intervention::Irrelevant as u8, 3);
        assert_eq!(Intervention::Filler as u8, 4);
    }

    #[test]
    fn test_profile_pod_size() {
        // Plan 278 T1.8: FaithfulnessProfile<f32> is 16 bytes (4 × f32, no padding).
        assert_eq!(size_of::<FaithfulnessProfile<f32>>(), 16);
    }

    #[test]
    fn test_is_faithfully_used_strict_all_conditions() {
        let threshold = 1.0_f32;

        // All conditions satisfied: empty small, others large.
        let faithful = FaithfulnessProfile {
            empty_delta: 0.0,
            shuffle_or_corrupt_delta: 5.0,
            irrelevant_delta: 4.0,
            filler_delta: 3.0,
        };
        assert!(faithful.is_faithfully_used(threshold));

        // Fails: shuffle/corrupt too small.
        let low_shuffle = FaithfulnessProfile {
            empty_delta: 0.0,
            shuffle_or_corrupt_delta: 0.5,
            irrelevant_delta: 4.0,
            filler_delta: 3.0,
        };
        assert!(!low_shuffle.is_faithfully_used(threshold));

        // Fails: reacts to empty (empty_delta above threshold).
        let reacts_to_empty = FaithfulnessProfile {
            empty_delta: 5.0,
            shuffle_or_corrupt_delta: 5.0,
            irrelevant_delta: 4.0,
            filler_delta: 3.0,
        };
        assert!(!reacts_to_empty.is_faithfully_used(threshold));

        // Fails: irrelevant too small.
        let low_irrelevant = FaithfulnessProfile {
            empty_delta: 0.0,
            shuffle_or_corrupt_delta: 5.0,
            irrelevant_delta: 0.5,
            filler_delta: 3.0,
        };
        assert!(!low_irrelevant.is_faithfully_used(threshold));

        // Fails: filler too small.
        let low_filler = FaithfulnessProfile {
            empty_delta: 0.0,
            shuffle_or_corrupt_delta: 5.0,
            irrelevant_delta: 4.0,
            filler_delta: 0.5,
        };
        assert!(!low_filler.is_faithfully_used(threshold));
    }

    #[test]
    fn test_vec_implements_memory_slice() {
        let mut v: Vec<f32> = vec![1.0, 2.0, 3.0];
        assert_eq!(v.mem_as_slice(), &[1.0, 2.0, 3.0]);
        v.mem_as_mut_slice()[0] = 99.0;
        assert_eq!(v.mem_as_slice(), &[99.0, 2.0, 3.0]);
    }
}
