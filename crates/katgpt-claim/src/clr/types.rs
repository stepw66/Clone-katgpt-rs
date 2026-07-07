//! CLR primitive data types — claim, trajectory, vote, cluster (Plan 284).
//!
//! These types are the wire/algorithm surface for Claim-Level Reliability voting.
//! They are intentionally generic over the payload `T` so the same runtime can
//! score reasoning traces, game episodes, tool calls, or any other outcome
//! whose quality decomposes into M projection directions.

/// CLR runtime configuration. All thresholds are sigmoid-bounded (in `(0, 1)`).
///
/// Field naming follows Plan 284 §3.2 + Research 255 notation:
///   - `k`                 embedding dimension per claim
///   - `m`                 number of projection directions (claims per trajectory)
///   - `tau_v`             verdict threshold (sigmoid gate per claim)
///   - `tau_reliable`      per-trajectory reliability threshold
///   - `tau_curiosity`     exploration threshold (curiosity arm)
///   - `alpha_freeze_thaw` direction vector freeze/thaw learning rate
///   - `gamma_mgpo`        MGPO advantage exponent
///   - `lambda_long2short` Long2Short tiebreak weight
///   - `tiebreak_eps`      reliability tie tolerance for the brevity tiebreak
#[derive(Clone, Debug)]
pub struct ClrConfig {
    /// Embedding dimension per claim (`k` in the paper).
    pub k: usize,
    /// Number of projection directions / claims per trajectory (`m`).
    pub m: usize,
    /// Verdict threshold `τ_v` — sigmoid gate per claim.
    pub tau_v: f32,
    /// Per-trajectory reliability threshold `τ_reliable`.
    pub tau_reliable: f32,
    /// Exploration (curiosity) threshold `τ_curiosity`.
    pub tau_curiosity: f32,
    /// Freeze/thaw learning rate `α` for direction vector updates.
    pub alpha_freeze_thaw: f32,
    /// MGPO advantage exponent `γ`.
    pub gamma_mgpo: f32,
    /// Long2Short tiebreak weight `λ`.
    pub lambda_long2short: f32,
    /// Reliability tie tolerance `ε` for the brevity tiebreak.
    pub tiebreak_eps: f32,
}

impl Default for ClrConfig {
    fn default() -> Self {
        // Paper defaults (Plan 284 §3.2, Research 255 Table 1).
        Self {
            k: 32,
            m: 5,
            tau_v: 0.5,
            tau_reliable: 0.5,
            tau_curiosity: 0.7,
            alpha_freeze_thaw: 0.01,
            gamma_mgpo: 2.0,
            lambda_long2short: 0.2,
            tiebreak_eps: 1e-3,
        }
    }
}

/// A single claim extracted from a trajectory.
///
/// `embedding` is the latent representation projected onto direction vectors
/// during verification; `payload` carries the domain-specific claim body.
#[derive(Clone, Debug)]
pub struct Claim<T> {
    /// Latent embedding of the claim, length == [`ClrConfig::k`].
    pub embedding: Vec<f32>,
    /// Domain-specific claim payload (e.g. extracted text, action, tool call).
    pub payload: T,
}

/// A scored trajectory whose claims have been (or will be) verified.
///
/// Generic over the payload type `T`. `T: Clone + Debug` so votes and clusters
/// can be cloned for downstream consumers (logging, replay, KG emission).
#[derive(Clone, Debug)]
pub struct Trajectory<T> {
    /// Final outcome of the trajectory (the thing being voted on).
    pub outcome: T,
    /// Token count (LLM) or step count (agent/game) — feeds Long2Short tiebreak.
    pub tokens_or_steps: usize,
    /// Extracted claims, length should equal [`ClrConfig::m`] (caller-enforced).
    pub claims: Vec<Claim<T>>,
    /// Optional per-token log-probabilities for downstream MGPO/curiosity arms.
    pub log_probs: Option<Vec<f32>>,
}

/// Verdict = sigmoid(dot(claim_embedding, direction_vec)). Bounded in `(0, 1)`.
pub type Verdict = f32;

/// Reliability score = aggregate of verdicts for a trajectory. Bounded in `(0, 1)`.
pub type ReliabilityScore = f32;

/// A cluster of trajectories that agreed (vote-bucket).
///
/// `representative_idx` indexes into the slice passed to the voter;
/// `member_indices` are the trajectory indices that fell into this cluster.
#[derive(Clone, Debug)]
pub struct Cluster<T> {
    /// Outcome carried by the cluster representative.
    pub outcome: T,
    /// Sum of per-trajectory reliability scores for cluster members.
    pub total_reliability: ReliabilityScore,
    /// Index of the representative trajectory in the original input slice.
    pub representative_idx: usize,
    /// Indices of all member trajectories in the original input slice.
    pub member_indices: Vec<usize>,
}

/// Result of a CLR vote over `M` trajectories.
///
/// `per_trajectory_verdicts` is flattened `[trajectory_idx][direction_idx]`
/// (length = `trajectories.len() * m`). The plan's fixed `[Verdict; M_DYNAMIC]`
/// is replaced with a `Vec<Verdict>` because `m` is dynamic at runtime.
#[derive(Clone, Debug)]
pub struct VoteResult<T> {
    /// Winning cluster (highest `total_reliability`, brevity-tiebroken).
    pub winner: Cluster<T>,
    /// All clusters discovered during voting.
    pub all_clusters: Vec<Cluster<T>>,
    /// Per-trajectory reliability score (one per input trajectory).
    pub per_trajectory_reliability: Vec<ReliabilityScore>,
    /// Per-trajectory per-direction verdicts, flattened row-major.
    pub per_trajectory_verdicts: Vec<Verdict>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_returns_spec_defaults() {
        // Paper defaults from Plan 284 §3.2 + Research 255 Table 1.
        let c = ClrConfig::default();
        assert_eq!(c.k, 32, "k default");
        assert_eq!(c.m, 5, "m default");
        assert!((c.tau_v - 0.5).abs() < 1e-6, "tau_v default");
        assert!((c.tau_reliable - 0.5).abs() < 1e-6, "tau_reliable default");
        assert!(
            (c.tau_curiosity - 0.7).abs() < 1e-6,
            "tau_curiosity default"
        );
        assert!((c.alpha_freeze_thaw - 0.01).abs() < 1e-6, "alpha default");
        assert!((c.gamma_mgpo - 2.0).abs() < 1e-6, "gamma_mgpo default");
        assert!((c.lambda_long2short - 0.2).abs() < 1e-6, "lambda default");
        assert!((c.tiebreak_eps - 1e-3).abs() < 1e-9, "tiebreak_eps default");
    }

    #[test]
    fn claim_carries_embedding_and_payload() {
        let c = Claim {
            embedding: vec![1.0, 2.0],
            payload: 42usize,
        };
        assert_eq!(c.embedding.len(), 2);
        assert_eq!(c.payload, 42);
    }

    #[test]
    fn trajectory_default_fields() {
        let t: Trajectory<i32> = Trajectory {
            outcome: 7,
            tokens_or_steps: 10,
            claims: vec![],
            log_probs: None,
        };
        assert_eq!(t.outcome, 7);
        assert_eq!(t.tokens_or_steps, 10);
        assert!(t.claims.is_empty());
        assert!(t.log_probs.is_none());
    }
}
