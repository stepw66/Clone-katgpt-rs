use core::fmt;

/// Outcome of verifying a speculative thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SpecOutcome {
    /// Speculative observation matches target → commit branch.
    Commit,
    /// Speculative observation differs → rollback to verified state.
    Rollback,
}

/// State of a single hop in the speculative pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HopState {
    /// Waiting for target tool to return.
    AwaitingTarget,
    /// Speculator has predicted observation, LLM continuing.
    Speculating,
    /// Verification passed, observation committed.
    Committed,
    /// Verification failed, rolled back.
    RolledBack,
}

/// Error type for speculator operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecError {
    /// No cached/speculated observation available for this action.
    CacheMiss { action: String },
    /// Speculator confidence below threshold.
    LowConfidence { action: String, score: u32 },
    /// Pipeline capacity exhausted (max k threads active).
    CapacityExhausted { active: usize, max: usize },
}

impl fmt::Display for SpecError {
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CacheMiss { action } => write!(f, "cache miss for action: {action}"),
            Self::LowConfidence { action, score } => {
                write!(f, "low confidence ({score}) for action: {action}")
            }
            Self::CapacityExhausted { active, max } => {
                write!(f, "capacity exhausted: {active}/{max} threads active")
            }
        }
    }
}

impl core::error::Error for SpecError {}

/// Configuration for SpecHop pipeline.
/// Parameters from SpecHop paper Section 3.
///
/// - α (alpha): relative speculator latency — `E[T_spec] / E[T_target]`, must be < 1.0
/// - β (beta): decode-to-tool ratio — `E[T_seg] / E[T_target]`
/// - p: speculator success probability per hop
/// - k: max active speculative threads; `None` → auto-compute via `optimal_k()`
/// - volatility: bound for starvation probability (Theorem 4). Default: 0.4
#[derive(Clone, Debug)]
pub struct SpecHopConfig {
    /// Relative speculator latency: `E[T_spec] / E[T_target]`. Must be in (0, 1).
    pub alpha: f64,
    /// Decode-to-tool ratio: `E[T_seg] / E[T_target]`. Must be > 0.
    pub beta: f64,
    /// Speculator success probability per hop. Must be in (0, 1].
    pub p: f64,
    /// Maximum active speculative threads. `None` = auto-compute from α and β.
    pub k: Option<usize>,
    /// Volatility bound for starvation probability (Theorem 4). Default: 0.4.
    pub volatility: f64,
}

impl Default for SpecHopConfig {
    fn default() -> Self {
        Self {
            alpha: 0.2,
            beta: 0.15,
            p: 0.7,
            k: None,
            volatility: 0.4,
        }
    }
}

impl SpecHopConfig {
    /// Validate config parameters. Returns `Ok(())` if valid.
    pub fn validate(&self) -> Result<(), String> {
        if self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(format!("alpha must be in (0, 1), got {}", self.alpha));
        }
        if self.beta <= 0.0 {
            return Err(format!("beta must be > 0, got {}", self.beta));
        }
        if self.p <= 0.0 || self.p > 1.0 {
            return Err(format!("p must be in (0, 1], got {}", self.p));
        }
        if self.volatility <= 0.0 {
            return Err(format!("volatility must be > 0, got {}", self.volatility));
        }
        if let Some(k) = self.k
            && k == 0
        {
            return Err("k must be >= 1 when specified".to_string());
        }
        Ok(())
    }

    /// Effective thread count: user-specified k or auto-computed optimal_k.
    pub fn effective_k(&self) -> usize {
        self.k.unwrap_or_else(|| self.optimal_k())
    }

    /// Compute optimal thread count: `k* = ⌈(1 + β) / (α + β)⌉` (Theorem 2).
    pub fn optimal_k(&self) -> usize {
        let k_det = (1.0 + self.beta) / (self.alpha + self.beta);
        k_det.ceil() as usize
    }
}

/// A single hop observation pair (target vs speculative).
#[derive(Clone, Debug)]
pub struct HopObservation {
    /// The action that triggered this hop (e.g., a search query, tool call).
    pub action: String,
    /// Target tool observation (may be pending until tool returns).
    pub o_target: Option<String>,
    /// Speculative observation (from speculator).
    pub o_spec: Option<String>,
    /// Current state of this hop.
    pub state: HopState,
}

impl HopObservation {
    /// Create a new hop awaiting target, with a speculative prediction.
    pub fn speculating(action: impl Into<String>, o_spec: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            o_target: None,
            o_spec: Some(o_spec.into()),
            state: HopState::Speculating,
        }
    }

    /// Create a new hop awaiting target, without any speculation yet.
    pub fn awaiting(action: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            o_target: None,
            o_spec: None,
            state: HopState::AwaitingTarget,
        }
    }

    /// Transition to committed state with the target observation.
    pub fn commit(&mut self, o_target: impl Into<String>) {
        self.o_target = Some(o_target.into());
        self.state = HopState::Committed;
    }

    /// Transition to rolled-back state.
    pub fn rollback(&mut self) {
        self.o_spec = None;
        self.state = HopState::RolledBack;
    }

    /// Whether this hop has a speculative prediction ready.
    pub fn has_spec(&self) -> bool {
        self.o_spec.is_some()
    }

    /// Whether this hop has received the target observation.
    pub fn has_target(&self) -> bool {
        self.o_target.is_some()
    }

    /// Whether this hop is in a terminal state (committed or rolled back).
    pub fn is_terminal(&self) -> bool {
        matches!(self.state, HopState::Committed | HopState::RolledBack)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_outcome_variants() {
        assert_eq!(SpecOutcome::Commit as u8, 0);
        assert_eq!(SpecOutcome::Rollback as u8, 1);
        assert!(SpecOutcome::Commit != SpecOutcome::Rollback);
    }

    #[test]
    fn test_hop_state_variants() {
        assert_eq!(HopState::AwaitingTarget as u8, 0);
        assert_eq!(HopState::Speculating as u8, 1);
        assert_eq!(HopState::Committed as u8, 2);
        assert_eq!(HopState::RolledBack as u8, 3);
    }

    #[test]
    fn test_spec_error_display() {
        let err = SpecError::CacheMiss {
            action: "search".to_string(),
        };
        assert_eq!(format!("{err}"), "cache miss for action: search");

        let err = SpecError::LowConfidence {
            action: "query".to_string(),
            score: 42,
        };
        assert_eq!(format!("{err}"), "low confidence (42) for action: query");

        let err = SpecError::CapacityExhausted { active: 4, max: 4 };
        assert_eq!(format!("{err}"), "capacity exhausted: 4/4 threads active");
    }

    #[test]
    fn test_spec_hop_config_default_validates() {
        let config = SpecHopConfig::default();
        assert!(config.validate().is_ok());
        assert!((config.alpha - 0.2).abs() < f64::EPSILON);
        assert!((config.beta - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spec_hop_config_validate_rejects_bad_alpha() {
        let config = SpecHopConfig {
            alpha: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = SpecHopConfig {
            alpha: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_spec_hop_config_validate_rejects_bad_beta() {
        let config = SpecHopConfig {
            beta: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_spec_hop_config_validate_rejects_bad_p() {
        let config = SpecHopConfig {
            p: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = SpecHopConfig {
            p: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_spec_hop_config_validate_rejects_zero_k() {
        let config = SpecHopConfig {
            k: Some(0),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_optimal_k_paper_examples() {
        // Paper example 1: α=0.2, β=0.15 → k* = ⌈1.15/0.35⌉ = ⌈3.286⌉ = 4
        let config = SpecHopConfig {
            alpha: 0.2,
            beta: 0.15,
            ..Default::default()
        };
        assert_eq!(config.optimal_k(), 4);

        // Paper example 2: α=0.3, β=0.75 → k* = ⌈1.75/1.05⌉ = ⌈1.667⌉ = 2
        let config = SpecHopConfig {
            alpha: 0.3,
            beta: 0.75,
            ..Default::default()
        };
        assert_eq!(config.optimal_k(), 2);
    }

    #[test]
    fn test_effective_k_uses_explicit_when_set() {
        let config = SpecHopConfig {
            k: Some(8),
            ..Default::default()
        };
        assert_eq!(config.effective_k(), 8);
    }

    #[test]
    fn test_effective_k_auto_computes_when_none() {
        let config = SpecHopConfig {
            alpha: 0.2,
            beta: 0.15,
            k: None,
            ..Default::default()
        };
        assert_eq!(config.effective_k(), 4);
    }

    #[test]
    fn test_hop_observation_speculating() {
        let hop = HopObservation::speculating("search_rust", "result1");
        assert_eq!(hop.action, "search_rust");
        assert!(hop.o_target.is_none());
        assert_eq!(hop.o_spec.as_deref(), Some("result1"));
        assert_eq!(hop.state, HopState::Speculating);
        assert!(hop.has_spec());
        assert!(!hop.has_target());
        assert!(!hop.is_terminal());
    }

    #[test]
    fn test_hop_observation_awaiting() {
        let hop = HopObservation::awaiting("search_go");
        assert_eq!(hop.action, "search_go");
        assert!(hop.o_target.is_none());
        assert!(hop.o_spec.is_none());
        assert_eq!(hop.state, HopState::AwaitingTarget);
        assert!(!hop.has_spec());
        assert!(!hop.has_target());
        assert!(!hop.is_terminal());
    }

    #[test]
    fn test_hop_observation_commit() {
        let mut hop = HopObservation::speculating("search", "spec_result");
        hop.commit("real_result");
        assert_eq!(hop.o_target.as_deref(), Some("real_result"));
        assert_eq!(hop.state, HopState::Committed);
        assert!(hop.has_target());
        assert!(hop.is_terminal());
    }

    #[test]
    fn test_hop_observation_rollback() {
        let mut hop = HopObservation::speculating("search", "spec_result");
        hop.rollback();
        assert!(hop.o_spec.is_none());
        assert_eq!(hop.state, HopState::RolledBack);
        assert!(!hop.has_spec());
        assert!(hop.is_terminal());
    }
}
