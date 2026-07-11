//! HiddenHandoff — stripped forward pass for drafter -> verifier.
//!
//! The paper's key insight (§3.1.1): the wasteful `D1 . E2` round-trip between
//! an LLM and itself (or between drafter and verifier) can be eliminated.
//! Instead of decoding tokens then re-embedding them, pass the hidden state
//! directly. This is the "stripped intermediate node" pattern applied to our
//! existing speculative pipeline.

use super::types::DenseHidden;

/// A handoff buffer for passing hidden state between a drafter and verifier.
///
/// **LATENT** (AGENTS.md): this never crosses `SyncBlock`. The drafter's last
/// hidden state is passed to the verifier's first layer directly, skipping the
/// `argmax + embed` round-trip. The verifier's output tokens are **raw** and
/// commit normally.
///
/// Use case: drafter speculates K tokens, then hands off its final hidden
/// state to the verifier for the next chunk. Saves one decode + one embed.
pub struct HiddenHandoff {
    /// The drafter's final hidden state (latent channel).
    pub hidden: DenseHidden,
    /// How many tokens the drafter has already produced (for KV cache continuity).
    pub draft_position: usize,
    /// Drafter's confidence in the handoff (for EdgeBandit reward).
    pub confidence: f32,
}

impl HiddenHandoff {
    /// Capture a drafter's final hidden state at the given position.
    pub fn new(hidden: DenseHidden, draft_position: usize, confidence: f32) -> Self {
        Self {
            hidden,
            draft_position,
            confidence,
        }
    }

    /// Reset the handoff (plasma-tier reuse).
    pub fn clear(&mut self) {
        self.hidden.clear();
        self.draft_position = 0;
        self.confidence = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handoff_lifecycle() {
        let hidden = DenseHidden::zeros(2, 4);
        let mut handoff = HiddenHandoff::new(hidden, 10, 0.8);
        assert_eq!(handoff.draft_position, 10);
        assert!((handoff.confidence - 0.8).abs() < 1e-6);
        handoff.clear();
        assert_eq!(handoff.draft_position, 0);
        assert!((handoff.confidence).abs() < 1e-6);
    }
}
