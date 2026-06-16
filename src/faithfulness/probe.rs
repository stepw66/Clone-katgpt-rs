//! [`FaithfulnessProbe`] trait + [`DefaultFaithfulnessProbe`] implementation.
//!
//! The probe runs the causal intervention suite from Research 244 §4 / Plan 278:
//! for each [`Intervention`](super::types::Intervention), it perturbs a clone of
//! the memory, queries the consumer's behavior, and measures the delta from
//! baseline (no-memory). The aggregated deltas form a [`FaithfulnessProfile`]
//! whose `is_faithfully_used(threshold)` gives the verdict.

use fastrand::Rng;

use super::perturb;
use super::types::{ConsumerContext, FaithfulnessProfile, Intervention, MemorySlice};

/// Causal-intervention faithfulness probe for injected memory segments.
///
/// Verifies that a consumer's behavior is causally bound to an injected
/// memory segment. Modelless: zero training, zero backprop through base weights.
///
/// Based on Zhao et al. 2026 (arxiv 2601.22436), Research 244 §4.
pub trait FaithfulnessProbe {
    /// Memory representation under audit.
    type Memory;
    /// Behavior representation produced by the consumer.
    type Behavior;
    /// Delta metric (must support ordering, be `Copy`).
    type Delta: PartialOrd + Copy + Default;

    /// Run a single causal intervention and report the behavioral delta
    /// between baseline (no memory) and behavior with the perturbed memory.
    fn probe_intervention(
        &mut self,
        memory: &Self::Memory,
        intervention: Intervention,
        rng: &mut Rng,
    ) -> Self::Delta;

    /// Run the full intervention suite {Empty, Shuffle, Corrupt, Irrelevant,
    /// Filler} and aggregate into a [`FaithfulnessProfile`].
    fn faithfulness_profile(
        &mut self,
        memory: &Self::Memory,
        rng: &mut Rng,
    ) -> FaithfulnessProfile<Self::Delta>;
}

/// Default probe that runs the full intervention suite using the
/// [`perturb`] strategies on a slice view of memory.
///
/// Generic over `C: ConsumerContext`. Requires `C::Memory: MemorySlice + Clone`
/// so the probe can clone-and-perturb.
///
/// # Fields
///
/// - `consumer` — the behavioral surface under audit.
/// - `irrelevant_pool` — source of unrelated content for the `Irrelevant` intervention.
/// - `filler_elem` — placeholder for the `Filler` intervention.
///
/// Runs at **audit cadence** (every N ticks), not per-tick. May allocate
/// (clones memory per intervention). See Plan 278 ADR-2.
pub struct DefaultFaithfulnessProbe<C>
where
    C: ConsumerContext,
    C::Memory: MemorySlice,
{
    pub consumer: C,
    pub irrelevant_pool: Vec<<C::Memory as MemorySlice>::Elem>,
    pub filler_elem: <C::Memory as MemorySlice>::Elem,
}

impl<C> DefaultFaithfulnessProbe<C>
where
    C: ConsumerContext,
    C::Memory: MemorySlice + Clone,
{
    /// Create a probe with the given consumer, irrelevant pool, and filler element.
    pub fn new(
        consumer: C,
        irrelevant_pool: Vec<<C::Memory as MemorySlice>::Elem>,
        filler_elem: <C::Memory as MemorySlice>::Elem,
    ) -> Self {
        Self {
            consumer,
            irrelevant_pool,
            filler_elem,
        }
    }
}

impl<C> FaithfulnessProbe for DefaultFaithfulnessProbe<C>
where
    C: ConsumerContext,
    C::Memory: MemorySlice + Clone,
{
    type Memory = C::Memory;
    type Behavior = C::Behavior;
    type Delta = C::Delta;

    fn probe_intervention(
        &mut self,
        memory: &Self::Memory,
        intervention: Intervention,
        rng: &mut Rng,
    ) -> Self::Delta {
        let baseline = self.consumer.baseline_behavior();

        // Clone + perturb in-place. The mutable borrow from mem_as_mut_slice
        // ends before we hand &perturbed to behavior_with_memory (NLL).
        let mut perturbed = memory.clone();
        {
            let slice = perturbed.mem_as_mut_slice();
            match intervention {
                Intervention::Empty => perturb::perturb_empty(slice),
                Intervention::Shuffle => perturb::perturb_shuffle(slice, rng),
                Intervention::Corrupt => perturb::perturb_corrupt(slice, rng),
                Intervention::Irrelevant => {
                    perturb::perturb_irrelevant(slice, rng, &self.irrelevant_pool)
                }
                Intervention::Filler => perturb::perturb_filler(slice, &self.filler_elem),
            }
        }

        let behavior = self.consumer.behavior_with_memory(&perturbed);
        self.consumer.behavior_delta(&baseline, &behavior)
    }

    fn faithfulness_profile(
        &mut self,
        memory: &Self::Memory,
        rng: &mut Rng,
    ) -> FaithfulnessProfile<Self::Delta> {
        // Each intervention clones from the ORIGINAL memory — perturbations
        // do not compound. 5 clones total (audit cadence, acceptable).
        let empty_delta = self.probe_intervention(memory, Intervention::Empty, rng);
        let shuffle_delta = self.probe_intervention(memory, Intervention::Shuffle, rng);
        let corrupt_delta = self.probe_intervention(memory, Intervention::Corrupt, rng);
        let irrelevant_delta = self.probe_intervention(memory, Intervention::Irrelevant, rng);
        let filler_delta = self.probe_intervention(memory, Intervention::Filler, rng);

        // Aggregate shuffle + corrupt into the max (whichever disrupted more).
        let shuffle_or_corrupt_delta = if shuffle_delta >= corrupt_delta {
            shuffle_delta
        } else {
            corrupt_delta
        };

        FaithfulnessProfile {
            empty_delta,
            shuffle_or_corrupt_delta,
            irrelevant_delta,
            filler_delta,
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests — Plan 278 T1.8 (G1 + G1b synthetic consumers)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faithfulness::types::ConsumerContext;

    /// Faithful consumer: behavior = weighted dot product with position-dependent
    /// weights. Empty memory → behavior 0 (= baseline). Meaningful perturbations
    /// → non-zero behavior.
    struct FaithfulConsumer {
        weights: Vec<f32>,
    }

    impl ConsumerContext for FaithfulConsumer {
        type Behavior = f32;
        type Delta = f32;
        type Memory = Vec<f32>;

        fn baseline_behavior(&self) -> f32 {
            0.0
        }

        fn behavior_with_memory(&self, memory: &Vec<f32>) -> f32 {
            memory
                .iter()
                .zip(self.weights.iter())
                .map(|(&v, &w)| v * w)
                .sum()
        }

        fn behavior_delta(&self, a: &f32, b: &f32) -> f32 {
            (a - b).abs()
        }
    }

    /// Unfaithful consumer: ignores memory entirely, always returns a constant.
    struct UnfaithfulConsumer;

    impl ConsumerContext for UnfaithfulConsumer {
        type Behavior = f32;
        type Delta = f32;
        type Memory = Vec<f32>;

        fn baseline_behavior(&self) -> f32 {
            42.0
        }

        fn behavior_with_memory(&self, _memory: &Vec<f32>) -> f32 {
            42.0 // ignores memory
        }

        fn behavior_delta(&self, a: &f32, b: &f32) -> f32 {
            (a - b).abs()
        }
    }

    #[test]
    fn test_faithful_consumer_detected() {
        // Position-dependent weights so shuffle matters.
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let consumer = FaithfulConsumer { weights };
        // Non-zero pool so irrelevant produces non-zero behavior.
        let irrelevant_pool = vec![0.1_f32, 0.2, 0.3, 0.4];
        // Non-zero filler so filler behavior = sum(weights) != 0.
        let filler = 1.0_f32;

        let mut probe = DefaultFaithfulnessProbe::new(consumer, irrelevant_pool, filler);

        // Distinct values so shuffle/corrupt change the dot product.
        let memory = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut rng = Rng::with_seed(42);

        let profile = probe.faithfulness_profile(&memory, &mut rng);

        // empty_delta should be ~0 (zeros -> behavior 0 = baseline).
        assert!(
            profile.empty_delta < 0.001,
            "empty_delta should be ~0, got {}",
            profile.empty_delta
        );

        let threshold = 0.5;
        assert!(
            profile.is_faithfully_used(threshold),
            "faithful consumer should be detected as faithfully used: {:?}",
            profile
        );
    }

    #[test]
    fn test_unfaithful_consumer_detected() {
        let consumer = UnfaithfulConsumer;
        let irrelevant_pool = vec![1.0_f32, 2.0, 3.0];
        let filler = 1.0_f32;

        let mut probe = DefaultFaithfulnessProbe::new(consumer, irrelevant_pool, filler);

        let memory = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mut rng = Rng::with_seed(42);

        let profile = probe.faithfulness_profile(&memory, &mut rng);

        // All deltas should be 0 (behavior always 42.0 = baseline).
        assert_eq!(profile.empty_delta, 0.0);
        assert_eq!(profile.shuffle_or_corrupt_delta, 0.0);
        assert_eq!(profile.irrelevant_delta, 0.0);
        assert_eq!(profile.filler_delta, 0.0);

        let threshold = 0.5;
        assert!(
            !profile.is_faithfully_used(threshold),
            "unfaithful consumer should NOT be detected as faithfully used: {:?}",
            profile
        );
    }
}
