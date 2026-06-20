//! `BeliefInferenceFn<S>` — stochastic hidden-state sampler for IIGs.
//!
//! Paper §4.2: for imperfect-information games, the CWM is paired with a
//! "belief inference function" that samples plausible hidden states from
//! observation histories. The paper asks the inducing LLM to also synthesise
//! this function; in our port, the function is whatever the integrator plugs
//! in (LLM-induced, hand-coded enumeration, particle filter, etc.).
//!
//! # Posterior-support guarantee (paper §4.2, hidden-history variant)
//!
//! The contract: if the offline unit tests all pass, each emitted sample is
//! (a) a valid CWM state AND (b) reproduces the observed observation sequence.
//! The guarantee is about SUPPORT, not DISTRIBUTION — ISMCTS only needs
//! samples from the posterior support set; it does not need the samples to be
//! distributed exactly proportional to posterior probability.
//!
//! This is the same relaxation the paper exploits: as long as every plausible
//! hidden state is *reachable* by sampling, ISMCTS's UCB1 aggregation will
//! find the equilibrium action.
//!
//! # Latent boundary (AGENTS.md)
//!
//! `Sample` is opaque — the trait does not constrain it. Integrators define
//! the hidden-state representation. Per AGENTS.md, samples are latent and
//! local: they never cross the sync boundary as embeddings. Only scalar
//! projections (e.g. "posterior P(strong hand)") cross.
//!
//! # Determinism for tests
//!
//! Implementations SHOULD be deterministic given a fixed `seed`. The mock
//! enumerators used in unit tests rely on this — see
//! [`crate::induced_cwm::tests`].

use crate::traits::GameState;

/// Stochastic belief-state sampler for imperfect-information games.
///
/// See the [module docs](self) for the posterior-support contract.
pub trait BeliefInferenceFn<S: GameState> {
    /// Hidden-state sample — opaque to this trait; integrators define the type.
    type Sample;

    /// Draw `n` samples from the belief at the current observation horizon.
    ///
    /// # Arguments
    /// * `obs_history` — observations visible to `player_id` so far
    ///   (representation is integrator-defined; typed as `S::Action` for
    ///   convenience, but the meaning is "what the player saw", not "what the
    ///   player did"). Integrators that need richer observation types should
    ///   wrap them — this trait intentionally avoids parameterising on a
    ///   second observation type to keep the surface minimal.
    /// * `action_history` — actions taken by all players so far
    /// * `player_id` — which player's belief we're sampling
    /// * `n` — number of samples to draw
    /// * `seed` — deterministic RNG seed (for unit-test reproducibility)
    ///
    /// # Returns
    /// `Vec<Self::Sample>` of length ≤ `n`. Implementations MAY return fewer
    /// than `n` samples if the posterior support is smaller than `n`.
    fn sample(
        &self,
        obs_history: &[S::Action],
        action_history: &[S::Action],
        player_id: u8,
        n: usize,
        seed: u64,
    ) -> Vec<Self::Sample>;
}
