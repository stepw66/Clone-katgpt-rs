//! `mux_demux` — deterministic superposition recovery.
//!
//! Given a set of weighted token hypotheses, verifies that they can be
//! uniquely recovered (demultiplexed) from their index.

/// Result of demultiplexing a superposition back to concrete token IDs.
#[derive(Debug, Clone, PartialEq)]
pub struct DemuxResult {
    /// Ordered token IDs recovered from the superposition.
    pub tokens: Vec<u32>,
    /// Whether the recovery was unique (no duplicate tokens).
    pub is_unique: bool,
}

/// Verifies that a superposition span can be uniquely demultiplexed.
#[derive(Debug, Clone)]
pub struct MuxDemuxVerifier {
    /// Expected superposition width.
    pub k: usize,
}

impl MuxDemuxVerifier {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Demultiplex a superposition: given token IDs and weights,
    /// return them sorted by weight (descending) and verify uniqueness.
    pub fn demux(&self, tokens: &[u32], weights: &[f32]) -> DemuxResult {
        assert_eq!(tokens.len(), weights.len());
        let mut pairs: Vec<(u32, f32)> = tokens.iter().zip(weights.iter()).map(|(&t, &w)| (t, w)).collect();
        pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let sorted_tokens: Vec<u32> = pairs.iter().map(|(t, _)| *t).collect();
        let is_unique = {
            let mut seen = std::collections::HashSet::new();
            sorted_tokens.iter().all(|t| seen.insert(*t))
        };

        DemuxResult {
            tokens: sorted_tokens,
            is_unique,
        }
    }
}

/// Convenience function: demultiplex a superposition span.
pub fn mux_demux(tokens: &[u32], weights: &[f32]) -> DemuxResult {
    let verifier = MuxDemuxVerifier::new(tokens.len());
    verifier.demux(tokens, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demux_sorts_by_weight() {
        let tokens = vec![3, 1, 2, 0];
        let weights = vec![0.1, 0.9, 0.5, 0.3];
        let result = mux_demux(&tokens, &weights);
        assert_eq!(result.tokens, vec![1, 2, 0, 3]);
        assert!(result.is_unique);
    }

    #[test]
    fn demux_detects_duplicates() {
        let tokens = vec![1, 1, 2];
        let weights = vec![0.5, 0.3, 0.2];
        let result = mux_demux(&tokens, &weights);
        assert!(!result.is_unique);
    }

    #[test]
    fn demux_empty() {
        let result = mux_demux(&[], &[]);
        assert!(result.tokens.is_empty());
        assert!(result.is_unique);
    }
}
