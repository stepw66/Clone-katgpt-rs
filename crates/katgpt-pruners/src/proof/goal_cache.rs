//! Proof goal cache — pruners-specific snapshot + re-exports (Plan 388 Phase 2).
//!
//! The core types (GoalHash, GoalResult, GoalVerifier, ProofGoalCache) were
//! extracted to `katgpt_core::proof_cache` to break the katgpt-pruners ↔
//! katgpt-speculative cycle. They're pure substrate (blake3 + HashMap +
//! AtomicU64) with zero pruners-specific knowledge.
//!
//! This module retains `ProofGoalSnapshot` — the pruners-specific metric
//! snapshot used by GOAT reporting — and re-exports the core types for
//! backwards compatibility. Existing `katgpt_pruners::proof::goal_cache::*`
//! paths resolve unchanged.
//!
//! # Origin
//!
//! Distilled from AlphaProof Nexus (arXiv:2605.22763):
//! "The global goal cache reduces redundant verification calls by 3×
//! across DDTree branches using deep hashes of exact formal state."

use std::fmt;

// Re-export the extracted core types for backwards compatibility.
pub use katgpt_core::proof_cache::{GoalHash, GoalResult, GoalVerifier, ProofGoalCache};

// ── ProofGoalSnapshot ──────────────────────────────────────────

/// Immutable snapshot of cache metrics for GOAT reporting.
///
/// Captures hit/miss counters and cache size at a point in time.
/// Safe to send across threads and store in benchmark results.
#[derive(Clone, Debug, PartialEq)]
pub struct ProofGoalSnapshot {
    /// Number of unique goals cached.
    pub entries: usize,
    /// Cache hit count.
    pub hits: u64,
    /// Cache miss count.
    pub misses: u64,
    /// Hit rate as ratio [0.0, 1.0].
    pub hit_rate: f64,
    /// Estimated memory usage in bytes.
    pub estimated_memory_bytes: usize,
}

impl ProofGoalSnapshot {
    /// Capture a snapshot of the current cache state.
    pub fn from_cache(cache: &ProofGoalCache) -> Self {
        Self {
            entries: cache.len(),
            hits: cache.hits(),
            misses: cache.misses(),
            hit_rate: cache.hit_rate(),
            estimated_memory_bytes: cache.estimated_memory_bytes(),
        }
    }
}

impl fmt::Display for ProofGoalSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProofGoalSnapshot(entries={}, hits={}, misses={}, hit_rate={:.1}%, mem={}B)",
            self.entries,
            self.hits,
            self.misses,
            self.hit_rate * 100.0,
            self.estimated_memory_bytes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proved_verifier(_bytes: &[u8]) -> GoalResult {
        GoalResult::Proved
    }

    // ── ProofGoalSnapshot Tests ────────────────────────────────

    #[test]
    fn snapshot_from_cache() {
        let mut cache = ProofGoalCache::new();
        cache.get_or_verify(b"a", proved_verifier);
        cache.get_or_verify(b"a", proved_verifier);

        let snap = ProofGoalSnapshot::from_cache(&cache);
        assert_eq!(snap.entries, 1);
        assert_eq!(snap.hits, 1);
        assert_eq!(snap.misses, 1);
        assert!((snap.hit_rate - 0.5).abs() < 1e-9);
        assert!(snap.estimated_memory_bytes > 0);
    }

    #[test]
    fn snapshot_display() {
        let snap = ProofGoalSnapshot {
            entries: 10,
            hits: 50,
            misses: 25,
            hit_rate: 0.667,
            estimated_memory_bytes: 1024,
        };
        let display = format!("{snap}");
        assert!(display.contains("entries=10"));
        assert!(display.contains("hits=50"));
        assert!(display.contains("mem=1024B"));
    }
}
