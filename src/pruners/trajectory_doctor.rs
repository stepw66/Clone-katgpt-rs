//! TrajectoryDoctor — failure localization in DDTree paths (Plan 223, Phase 3).
//!
//! Traces back from rejected output to first violation, enabling
//! localized repair instead of full retry.

/// A specific failure site in a DDTree trajectory.
#[derive(Clone, Debug)]
pub struct FailureSite {
    /// Depth in DDTree where violation was detected.
    pub depth: usize,
    /// Token index where violation was detected.
    pub token_idx: usize,
    /// Description of violated predicate.
    pub violated_predicate: String,
    /// Alternative tokens that would have satisfied the predicate.
    pub alternatives: Vec<String>,
}

impl FailureSite {
    pub fn new(depth: usize, token_idx: usize, violated: &str, alts: Vec<String>) -> Self {
        Self {
            depth,
            token_idx,
            violated_predicate: violated.to_string(),
            alternatives: alts,
        }
    }
}

/// TrajectoryDoctor trait for failure localization.
pub trait TrajectoryDoctor {
    /// Given a rejected trajectory, find the first failure site.
    fn localize_failure(&self, tokens: &[String], depth_limit: usize) -> Option<FailureSite>;
}

/// Simple bracket-tracking TrajectoryDoctor.
#[derive(Clone, Debug, Default)]
pub struct BracketTrajectoryDoctor {
    /// Maximum allowed bracket nesting depth.
    pub max_depth: u32,
}

impl BracketTrajectoryDoctor {
    pub fn new(max_depth: u32) -> Self {
        Self { max_depth }
    }
}

impl TrajectoryDoctor for BracketTrajectoryDoctor {
    fn localize_failure(&self, tokens: &[String], _depth_limit: usize) -> Option<FailureSite> {
        let mut depth = 0u32;
        for (idx, token) in tokens.iter().enumerate() {
            match token.as_str() {
                "(" | "[" | "{" => {
                    depth += 1;
                    if depth > self.max_depth {
                        return Some(FailureSite::new(
                            idx,
                            idx,
                            &format!("bracket depth {} exceeds max {}", depth, self.max_depth),
                            vec![")".to_string(), "]".to_string(), "}".to_string()],
                        ));
                    }
                }
                ")" | "]" | "}" => {
                    depth = depth.saturating_sub(1);
                }
                _ => {}
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_localize_bracket_failure() {
        let doctor = BracketTrajectoryDoctor::new(2);
        let tokens: Vec<String> = ["(", "(", "(", "x", ")"].iter().map(|s| s.to_string()).collect();
        let failure = doctor.localize_failure(&tokens, 10);
        assert!(failure.is_some());
        let f = failure.unwrap();
        assert_eq!(f.token_idx, 2);
        assert_eq!(f.depth, 2);
    }

    #[test]
    fn test_no_failure() {
        let doctor = BracketTrajectoryDoctor::new(3);
        let tokens: Vec<String> = ["(", "(", "x", ")", ")"].iter().map(|s| s.to_string()).collect();
        assert!(doctor.localize_failure(&tokens, 10).is_none());
    }
}
