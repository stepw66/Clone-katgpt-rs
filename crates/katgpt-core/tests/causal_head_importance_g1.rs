//! G1 correctness gate for CausalHeadImportance (Plan 358 Phase 2).
//!
//! Synthetic harness: `N_HEADS` heads, of which `K_LOAD_BEARING` write the
//! signal into the readout (projection = 1.0) and the rest are *correlated
//! bystanders* that attend to the needle but project to zero (projection = 0.0,
//! orthogonal to the readout direction). The mock forward pass is a linear map
//! so `m_patched` is exactly computable:
//!
//! ```text
//! m(x)   = Σ_h p_h · a_h(input)
//! m_clean   = Σ_h p_h · 1      (clean activations = 1)
//! m_corrupt = Σ_h p_h · 0      (corrupt activations = 0, answer replaced)
//! m_patched(h) = p_h · 0 + Σ_{h'≠h} p_h' · 1 = m_clean − p_h
//! IE(h) = (m_clean − m_patched(h)) / (m_clean − m_corrupt) = p_h / m_clean
//! ```
//!
//! So load-bearing heads (p_h=1) get `IE = 1/K_LOAD_BEARING` and bystanders
//! (p_h=0) get `IE = 0` — perfect, exactly-computable separation.

use katgpt_core::causal_head_importance::{
    direct_effect_importance, partition_by_causal_score,
};

const N_HEADS: usize = 32;
const K_LOAD_BEARING: usize = 4;
/// Importance threshold separating load-bearing from bystander heads (paper §4.1).
const THRESHOLD: f32 = 0.01;

/// Mock forward-pass harness. Each head `h` has projection `p_h` into the
/// readout direction. The forward pass is `m(x) = Σ_h p_h · a_h(input)`.
struct LinearHarness {
    /// Readout projection per head (1.0 load-bearing, 0.0 bystander).
    projections: Vec<f32>,
}

impl LinearHarness {
    fn new(n_heads: usize, k_load_bearing: usize) -> Self {
        assert!(k_load_bearing <= n_heads);
        // Heads [0, k_load_bearing) are load-bearing; the rest are bystanders.
        let projections = (0..n_heads)
            .map(|h| if h < k_load_bearing { 1.0 } else { 0.0 })
            .collect();
        Self { projections }
    }

    /// Clean run: all heads active with activation 1.0 → `m = Σ p_h`.
    fn m_clean(&self) -> f32 {
        self.projections.iter().copied().sum()
    }

    /// Corrupt run: all heads active with activation 0.0 (answer replaced).
    fn m_corrupt(&self) -> f32 {
        0.0
    }

    /// Patched run: head `h`'s output replaced by its corrupt-run value
    /// (activation 0.0); all other heads remain at clean (activation 1.0).
    fn m_patched(&self, h: usize) -> f32 {
        // m = Σ_{h'≠h} p_{h'} · 1 + p_h · 0 = m_clean − p_h
        self.m_clean() - self.projections[h]
    }

    /// Readout after knocking out `knocked` heads (their activations zeroed),
    /// with all others at clean activation 1.0.
    fn m_after_knockout(&self, knocked: &[usize]) -> f32 {
        let mut m = 0.0f32;
        for (h, p) in self.projections.iter().enumerate() {
            if !knocked.contains(&h) {
                m += p;
            }
        }
        m
    }

    /// Compute IE for every head.
    fn ies(&self) -> Vec<f32> {
        let m_clean = self.m_clean();
        let m_corrupt = self.m_corrupt();
        (0..self.projections.len())
            .map(|h| direct_effect_importance(m_clean, m_corrupt, self.m_patched(h)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// G1.1 — IE discrimination
// ---------------------------------------------------------------------------

#[test]
fn ie_discriminates_load_bearing_from_bystanders() {
    let harness = LinearHarness::new(N_HEADS, K_LOAD_BEARING);
    let ies = harness.ies();

    // Every load-bearing head (indices 0..K) has IE = 1/K_LOAD_BEARING > threshold.
    let expected_lb_ie = 1.0 / K_LOAD_BEARING as f32;
    for (h, ie) in ies.iter().enumerate().take(K_LOAD_BEARING) {
        assert!(
            *ie > THRESHOLD,
            "load-bearing head {h} has IE {} <= threshold {THRESHOLD}",
            ie
        );
        assert!(
            (*ie - expected_lb_ie).abs() < 1e-6,
            "load-bearing head {h} IE {} != expected {expected_lb_ie}",
            ie
        );
    }
    // Every bystander head (indices K..N) has IE = 0 < threshold.
    for (h, ie) in ies.iter().enumerate().skip(K_LOAD_BEARING).take(N_HEADS - K_LOAD_BEARING) {
        assert!(
            *ie < THRESHOLD,
            "bystander head {h} has IE {} >= threshold {THRESHOLD}",
            ie
        );
        assert!(
            ie.abs() < 1e-6,
            "bystander head {h} IE {} != 0",
            ie
        );
    }
}

// ---------------------------------------------------------------------------
// G1.2 — Ranking puts load-bearing above bystanders (perfect separation)
// ---------------------------------------------------------------------------

#[test]
fn ranking_load_bearing_all_above_bystanders() {
    let harness = LinearHarness::new(N_HEADS, K_LOAD_BEARING);
    let ies = harness.ies();

    // Rank by IE descending (ties broken by index ascending, matching
    // partition_by_causal_score's tiebreak).
    let mut order: Vec<usize> = (0..N_HEADS).collect();
    order.sort_by(|&a, &b| {
        ies[b]
            .partial_cmp(&ies[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    // The top-K_LOAD_BEARING positions must be exactly the load-bearing heads
    // (indices 0..K). Bystanders occupy the tail.
    let top_set: std::collections::HashSet<usize> =
        order[..K_LOAD_BEARING].iter().copied().collect();
    for h in 0..K_LOAD_BEARING {
        assert!(
            top_set.contains(&h),
            "load-bearing head {h} missing from top-{K_LOAD_BEARING}: {top_set:?}"
        );
    }

    // Equivalent Spearman-ρ=1.0 statement on this clean synthetic: the IE-based
    // partition's top-K set is *exactly* the load-bearing set (Jaccard = 1.0).
    let (critical, _) = partition_by_causal_score(&ies, K_LOAD_BEARING as f32 / N_HEADS as f32, None, false);
    let load_bearing_set: std::collections::HashSet<usize> = (0..K_LOAD_BEARING).collect();
    let critical_set: std::collections::HashSet<usize> = critical.iter().copied().collect();
    assert_eq!(
        critical_set, load_bearing_set,
        "causal partition top-K != load-bearing set (Jaccard < 1.0)"
    );
}

// ---------------------------------------------------------------------------
// G1.3 — Knockout faithfulness (paper Fig 9b reproduction)
// ---------------------------------------------------------------------------

/// Simple deterministic xorshift RNG for reproducible random knockout trials.
struct XorShiftRng(u64);

impl XorShiftRng {
    fn new(seed: u64) -> Self {
        Self(seed | 1) // ensure nonzero
    }
    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 32) as u32
    }
    /// Fisher-Yates sample `k` distinct indices from `0..n`.
    fn sample_k(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + (self.next_u32() as usize % (n - i));
            perm.swap(i, j);
        }
        perm[..k].to_vec()
    }
}

#[test]
fn knockout_ie_ordered_collapses_random_stays_high() {
    let harness = LinearHarness::new(N_HEADS, K_LOAD_BEARING);
    let ies = harness.ies();
    let m_baseline = harness.m_clean();
    assert!(m_baseline > 0.0);

    // IE-ordered knockout: knock out the top-K_LOAD_BEARING heads (all load-bearing).
    let mut order: Vec<usize> = (0..N_HEADS).collect();
    order.sort_by(|&a, &b| {
        ies[b]
            .partial_cmp(&ies[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let ie_knocked: Vec<usize> = order[..K_LOAD_BEARING].to_vec();
    let m_after_ie = harness.m_after_knockout(&ie_knocked);
    let ratio_ie = m_after_ie / m_baseline;
    // All load-bearing heads knocked out → readout collapses to 0.
    assert!(
        ratio_ie < 0.2,
        "IE-ordered top-{K_LOAD_BEARING} knockout did not collapse: ratio {ratio_ie:.4}"
    );

    // Random knockout control: mean ratio over many trials.
    // Expected mean ratio = (N − K) / N = 28/32 = 0.875 > 0.8.
    let mut rng = XorShiftRng::new(0xCAFE_BABE_DEAD_BEEF);
    const N_TRIALS: usize = 2000;
    let mut ratio_sum = 0.0f64;
    for _ in 0..N_TRIALS {
        let random_knocked = rng.sample_k(N_HEADS, K_LOAD_BEARING);
        let m_after = harness.m_after_knockout(&random_knocked);
        ratio_sum += (m_after / m_baseline) as f64;
    }
    let mean_random_ratio = ratio_sum / N_TRIALS as f64;
    assert!(
        mean_random_ratio > 0.8,
        "random knockout mean ratio {mean_random_ratio:.4} not > 0.8"
    );
    // The contrast is the point: IE-ordered collapses, random does not.
    assert!(
        mean_random_ratio as f32 - ratio_ie > 0.6,
        "knockout contrast too small: IE {ratio_ie:.4} vs random {mean_random_ratio:.4}"
    );
}

// ---------------------------------------------------------------------------
// G1.4 — partition_by_causal_score matches the IE ranking on this harness
// ---------------------------------------------------------------------------

#[test]
fn partition_matches_load_bearing_set_on_harness() {
    // The partition helper must reproduce the load-bearing/bystander split when
    // given the IE scores and the correct ratio. This is the integration of the
    // scorer (direct_effect_importance) with the RTPurbo-style partition.
    let harness = LinearHarness::new(N_HEADS, K_LOAD_BEARING);
    let ies = harness.ies();
    let ratio = K_LOAD_BEARING as f32 / N_HEADS as f32;
    let (critical, convertible) = partition_by_causal_score(&ies, ratio, None, false);

    // Critical set = load-bearing (0..K), convertible = bystanders (K..N).
    let expected_critical: Vec<usize> = (0..K_LOAD_BEARING).collect();
    let expected_convertible: Vec<usize> = (K_LOAD_BEARING..N_HEADS).collect();
    assert_eq!(critical, expected_critical, "critical set mismatch");
    assert_eq!(convertible, expected_convertible, "convertible set mismatch");
}

// ---------------------------------------------------------------------------
// G1.5 — readout integration: SpanLogitDiffReadout feeds the IE pipeline
// ---------------------------------------------------------------------------

#[test]
fn readout_feeds_ie_pipeline() {
    // End-to-end: use SpanLogitDiffReadout to produce m values, then feed them
    // to direct_effect_importance. Demonstrates the measurement primitives
    // compose correctly. A 2-position answer span, single load-bearing head.
    use katgpt_core::causal_head_importance::SpanLogitDiffReadout;
    let readout = SpanLogitDiffReadout::default();

    // Clean: correct token logits dominate → high m.
    let m_clean = readout.readout(&[(5.0, 1.0), (4.0, 0.0)]);
    // Corrupt: counterfactual dominates → low m.
    let m_corrupt = readout.readout(&[(1.0, 5.0), (0.0, 4.0)]);
    // Patched: head swapped to corrupt → m drops toward corrupt.
    let m_patched = readout.readout(&[(1.0, 5.0), (4.0, 0.0)]);

    assert!(m_clean > m_corrupt, "clean must exceed corrupt");
    let ie = direct_effect_importance(m_clean, m_corrupt, m_patched);
    assert!(
        (0.0..=1.0).contains(&ie),
        "IE {ie} outside [0,1]"
    );
    // The patch moved the first position to corrupt → substantial but partial drop.
    assert!(ie > 0.0, "IE should be positive for a contributing head");
}
