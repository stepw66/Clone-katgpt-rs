//! `mux_demux` — deterministic superposition recovery.
//!
//! Given a set of weighted token hypotheses, verifies that they can be
//! uniquely recovered (demultiplexed) from their index.

/// Maximum supported superposition width for stack-allocated demux.
const MAX_DEMUX_K: usize = 32;

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
    ///
    /// Performs a single allocation (no intermediate buffer).
    /// For zero-alloc path, use `demux_into`.
    pub fn demux(&self, tokens: &[u32], weights: &[f32]) -> DemuxResult {
        assert_eq!(tokens.len(), weights.len());
        let n = tokens.len();
        if n == 0 {
            return DemuxResult {
                tokens: Vec::new(),
                is_unique: true,
            };
        }

        // Stack-allocated sort: copy to stack, sort descending by weight.
        let mut pairs: [(u32, f32); MAX_DEMUX_K] = [(0, 0.0); MAX_DEMUX_K];
        let len = n.min(MAX_DEMUX_K);
        for i in 0..len {
            pairs[i] = (tokens[i], weights[i]);
        }
        pairs[..len].sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // O(k) uniqueness check via u64 bitmap for token IDs < 64.
        // Falls back to O(k log k) sort for larger IDs (rare for small k ≤ 32).
        let mut is_unique = true;
        let all_below_64 = pairs[..len].iter().all(|&(t, _)| t < 64);
        if all_below_64 {
            let mut bitmap: u64 = 0;
            for &(token, _) in &pairs[..len] {
                let mask = 1u64 << token;
                if bitmap & mask != 0 {
                    is_unique = false;
                    break;
                }
                bitmap |= mask;
            }
        } else {
            let mut sorted_tokens: [u32; MAX_DEMUX_K] = [0; MAX_DEMUX_K];
            for i in 0..len {
                sorted_tokens[i] = pairs[i].0;
            }
            sorted_tokens[..len].sort_unstable();
            for i in 1..len {
                if sorted_tokens[i] == sorted_tokens[i - 1] {
                    is_unique = false;
                    break;
                }
            }
        }

        let mut out_tokens = Vec::with_capacity(len);
        for &(token, _) in &pairs[..len] {
            out_tokens.push(token);
        }

        DemuxResult {
            tokens: out_tokens,
            is_unique,
        }
    }

    /// Zero-alloc demultiplexing into a caller-provided buffer.
    ///
    /// `out_tokens` must have capacity >= `tokens.len()`.
    /// Returns `is_unique` and writes sorted tokens into `out_tokens`.
    /// Caller reads the result directly from `out_tokens` — no clone.
    pub fn demux_into(&self, tokens: &[u32], weights: &[f32], out_tokens: &mut Vec<u32>) -> bool {
        assert_eq!(tokens.len(), weights.len());
        let n = tokens.len();
        if n == 0 {
            out_tokens.clear();
            return true;
        }

        // Stack-allocated sort: copy to stack, sort descending by weight.
        let mut pairs: [(u32, f32); MAX_DEMUX_K] = [(0, 0.0); MAX_DEMUX_K];
        let len = n.min(MAX_DEMUX_K);
        for i in 0..len {
            pairs[i] = (tokens[i], weights[i]);
        }
        pairs[..len].sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // O(k log k) uniqueness check: sort a copy by token ID, scan adjacent.
        let mut sorted_tokens: [u32; MAX_DEMUX_K] = [0; MAX_DEMUX_K];
        for i in 0..len {
            sorted_tokens[i] = pairs[i].0;
        }
        sorted_tokens[..len].sort_unstable();
        let mut is_unique = true;
        for i in 1..len {
            if sorted_tokens[i] == sorted_tokens[i - 1] {
                is_unique = false;
                break;
            }
        }

        out_tokens.clear();
        for &(token, _) in &pairs[..len] {
            out_tokens.push(token);
        }

        is_unique
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
