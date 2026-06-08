//! LLMExecGuard — entropy-driven verification budgeting (Plan 223, Phase 1).
//!
//! Distills Lean4Agent's verification layer into a modelless confidence gate.
//! Uses sigmoid on entropy + depth to route: high confidence (skip verification),
//! medium (screening), low (full verify).

/// Verification tier for DDTree expansion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum VerifyTier {
    /// High confidence — skip extra verification.
    #[default]
    Skip = 0,
    /// Medium confidence — use ScreeningPruner relevance gate only.
    Screening = 1,
    /// Low confidence — engage full verification pipeline.
    FullVerify = 2,
}

/// LLMExecGuard configuration.
#[derive(Clone, Debug)]
pub struct LlmExecGuardConfig {
    /// Sigmoid steepness for confidence computation. Default: 2.0.
    pub steepness: f32,
    /// Entropy threshold below which we skip verification. Default: 0.3.
    pub skip_threshold: f32,
    /// Entropy threshold above which we full-verify. Default: 0.7.
    pub full_verify_threshold: f32,
    /// Maximum DDTree depth (depths beyond this get less confident). Default: 8.
    pub max_depth: f32,
}

impl Default for LlmExecGuardConfig {
    fn default() -> Self {
        Self {
            steepness: 2.0,
            skip_threshold: 0.3,
            full_verify_threshold: 0.7,
            max_depth: 8.0,
        }
    }
}

/// Compute LLMExec confidence using sigmoid on entropy and depth.
/// Returns value in (0, 1) where higher = more confident.
///
/// confidence = sigmoid(-steepness * (entropy - 0.5) + depth_bonus)
#[inline]
pub fn llmexec_confidence(entropy: f32, depth: usize, config: &LlmExecGuardConfig) -> f32 {
    let depth_bonus = -0.1 * (depth as f32 / config.max_depth);
    let x = -config.steepness * (entropy - 0.5) + depth_bonus;
    1.0 / (1.0 + (-x).exp())
}

/// Route to verification tier based on confidence.
#[inline]
pub fn route_verify_tier(confidence: f32, config: &LlmExecGuardConfig) -> VerifyTier {
    if confidence > (1.0 - config.skip_threshold) {
        VerifyTier::Skip
    } else if confidence < (1.0 - config.full_verify_threshold) {
        VerifyTier::FullVerify
    } else {
        VerifyTier::Screening
    }
}

/// Convenience: compute confidence and route in one call.
#[inline]
pub fn verify_tier(entropy: f32, depth: usize, config: &LlmExecGuardConfig) -> VerifyTier {
    let conf = llmexec_confidence(entropy, depth, config);
    route_verify_tier(conf, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_entropy_high_confidence() {
        let config = LlmExecGuardConfig::default();
        let conf = llmexec_confidence(0.1, 0, &config);
        assert!(
            conf > 0.5,
            "low entropy should give high confidence, got {}",
            conf
        );
    }

    #[test]
    fn test_high_entropy_low_confidence() {
        let config = LlmExecGuardConfig::default();
        let conf = llmexec_confidence(0.9, 4, &config);
        assert!(
            conf < 0.5,
            "high entropy should give low confidence, got {}",
            conf
        );
    }

    #[test]
    fn test_routing_skip() {
        let config = LlmExecGuardConfig::default();
        let tier = verify_tier(0.1, 0, &config);
        assert_eq!(tier, VerifyTier::Skip);
    }

    #[test]
    fn test_routing_full_verify() {
        let config = LlmExecGuardConfig::default();
        let tier = verify_tier(0.95, 6, &config);
        assert_eq!(tier, VerifyTier::FullVerify);
    }

    #[test]
    fn test_depth_reduces_confidence() {
        let config = LlmExecGuardConfig::default();
        let shallow = llmexec_confidence(0.5, 0, &config);
        let deep = llmexec_confidence(0.5, 8, &config);
        assert!(
            shallow > deep,
            "shallow depth should have higher confidence"
        );
    }
}
