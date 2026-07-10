//! Plan 311 Phase 2 T2.2 — Alien Sampler latency + throughput microbench.
//!
//! Follows the crate's bench convention (`std::time::Instant` + `harness = false`,
//! matches `salience_tri_gate_bench.rs` / `procrustes_bench.rs`). No Criterion
//! dep — DRY with existing benches.
//!
//! Run:
//! ```bash
//! cargo run --release --bench alien_sampler_bench --features alien_sampler
//! ```
//!
//! Gates measured (Plan 311 Phase 2 T2.2 targets):
//! - `rank` 1k candidates × 4 atoms × 16-dim bank of 100  → ≤ 500 µs.
//! - `rank` 10k candidates (warm-tier batch)               → ≤ 5 ms.
//! - `median_top_m` m=10, bank=100  (single call)          → ≤ 5 µs.
//! - `median_top_m` m=10, bank=10k (large bank)            → ≤ 500 µs.

#![cfg(feature = "alien_sampler")]

use katgpt_deprecated::alien_sampler::{
    AlienConfig, AlienSampler, CoherenceScorer, MedianTopMAvailability,
};
use std::time::{Duration, Instant};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Candidate pool sizes to sweep for `rank()`.
const RANK_SIZES: &[usize] = &[1_000, 10_000];

/// Atom count per candidate (paper-motivated: 4 atoms × 16-dim bank).
const ATOMS_PER_CANDIDATE: usize = 4;

/// Embedding dimension (matches the paper's 16-dim repertoires).
const EMBED_DIM: usize = 16;

/// Bank sizes to sweep for `median_top_m`.
const BANK_SIZES: &[usize] = &[100, 10_000];

/// Paper-default m.
const M: usize = 10;

/// Batched-latency outer samples (median-of-N).
const OUTER: usize = 32;

/// Warmup iterations to prime caches + branch predictor.
const WARMUP: usize = 50;

// ─── Deterministic LCG (mirrors salience_tri_gate_bench convention) ─────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn next_f32(&mut self) -> f32 {
        // Divide by 2^31 (matches the salience bench fix; the original
        // u32::MAX division biased to [0, 0.5)).
        (self.next() as f32) / ((1u64 << 31) as f32)
    }
}

// ─── Reference scorers (mirrors integration_tests.rs) ───────────────────────

/// Coherence = dot product against a fixed "personality direction".
struct DotCoherence {
    direction: Vec<f32>,
}

impl CoherenceScorer<f32> for DotCoherence {
    #[inline]
    fn coherence(&self, atoms: &[f32]) -> f32 {
        // Defensive: zip in case the candidate is shorter than the direction
        // (production code trusts the caller; bench mirrors production).
        let mut s = 0.0_f32;
        for (a, b) in atoms.iter().zip(self.direction.iter()) {
            s += a * b;
        }
        s
    }
}

// ─── Bench data generation ──────────────────────────────────────────────────

/// Generate `n` candidates, each `ATOMS_PER_CANDIDATE × EMBED_DIM` f32 values
/// in `[-1, 1]`. Returned as `Vec<Vec<f32>>` (one Vec per candidate).
fn make_candidates(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let atom_count = ATOMS_PER_CANDIDATE * EMBED_DIM;
    (0..n)
        .map(|_| {
            (0..atom_count)
                .map(|_| rng.next_f32() * 2.0 - 1.0)
                .collect()
        })
        .collect()
}

/// Generate a community bank of `n` embeddings, each `EMBED_DIM` f32 values
/// in `[-1, 1]`.
fn make_bank(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| (0..EMBED_DIM).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
        .collect()
}

/// Generate a personality direction (unit vector in `EMBED_DIM`).
fn make_direction(seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let mut d: Vec<f32> = (0..EMBED_DIM).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let norm: f32 = d.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut d {
            *v /= norm;
        }
    }
    d
}

// ─── Timing helpers ─────────────────────────────────────────────────────────

/// Median of `OUTER` batched wall-time measurements of `op`. Each batch runs
/// `op` once; we take the median duration to filter noise on a shared dev
/// machine. The `sink` accumulator defeats dead-code elimination.
fn median_batched<F: FnMut() -> u64>(mut op: F, sink: u64) -> Duration {
    // Warmup.
    let mut warm_sink = sink;
    for _ in 0..WARMUP {
        warm_sink = warm_sink.wrapping_add(op());
    }
    if warm_sink == u64::MAX {
        std::process::abort();
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(OUTER);
    let mut live_sink = warm_sink;
    for _ in 0..OUTER {
        let t0 = Instant::now();
        live_sink = live_sink.wrapping_add(op());
        samples.push(t0.elapsed());
    }
    if live_sink == u64::MAX {
        std::process::abort();
    }
    samples.sort();
    let mid = OUTER / 2;
    // Average the two middle samples for an even-count median.
    (samples[mid] + samples[mid - 1]) / 2
}

/// Median batched latency for a single `rank()` call over `n` candidates.
/// Uses the trait path (`rank_into` — allocates cosine scratch per candidate
/// inside `MedianTopMAvailability::availability`).
fn bench_rank_trait(
    sampler: &AlienSampler<f32, DotCoherence, MedianTopMAvailability>,
    candidates: &[Vec<f32>],
    n: usize,
) -> Duration {
    let mut sc = vec![0.0_f32; n];
    let mut sa = vec![0.0_f32; n];
    let mut out: Vec<katgpt_deprecated::alien_sampler::ScoredCandidate> = Vec::with_capacity(n);
    let op = || {
        sampler
            .rank_into(candidates, &mut sc, &mut sa, &mut out)
            .ok();
        if let Some(first) = out.first() {
            (first.score.to_bits() as u64) ^ (first.idx as u64)
        } else {
            0
        }
    };
    median_batched(op, 0)
}

/// Median batched latency for the **hot path**: batch availability scoring
/// via `MedianTopMAvailability::availability_batch` (reuses one cosine
/// scratch across all candidates) + `AlienSampler::rank_precomputed`. Zero
/// per-candidate allocation.
fn bench_rank_batch(
    sampler: &AlienSampler<f32, DotCoherence, MedianTopMAvailability>,
    avail: &MedianTopMAvailability,
    coh: &DotCoherence,
    candidates: &[Vec<f32>],
    n: usize,
    bank_len: usize,
) -> Duration {
    let mut sc = vec![0.0_f32; n];
    let mut sa = vec![0.0_f32; n];
    let mut cosine_scratch = vec![0.0_f32; bank_len];
    let mut out: Vec<katgpt_deprecated::alien_sampler::ScoredCandidate> = Vec::with_capacity(n);
    let op = || {
        // Batch-fill coherence (trivial dot per candidate).
        for (i, cand) in candidates.iter().enumerate() {
            sc[i] = coh.coherence(cand);
        }
        // Batch-fill availability (one shared cosine scratch).
        avail.availability_batch(candidates, &mut sa, &mut cosine_scratch);
        // Fuse + sort.
        sampler.rank_precomputed(&mut sc, &mut sa, &mut out).ok();
        if let Some(first) = out.first() {
            (first.score.to_bits() as u64) ^ (first.idx as u64)
        } else {
            0
        }
    };
    median_batched(op, 0)
}

/// Median batched latency for a single `median_top_m` call against a bank.
fn bench_median_top_m(
    avail: &MedianTopMAvailability,
    candidate: &[f32],
    bank_len: usize,
) -> Duration {
    let mut scratch = vec![0.0_f32; bank_len];
    let op = || {
        let v = avail.availability_embedded_with_scratch(candidate, &mut scratch);
        v.to_bits() as u64
    };
    median_batched(op, 0)
}

fn pass_str(p: bool) -> &'static str {
    if p { "PASS" } else { "FAIL" }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 311 Alien Sampler Bench (Phase 2 T2.2) ===");
    println!(
        "  candidates: {{atoms={ATOMS_PER_CANDIDATE}, embed_dim={EMBED_DIM}}}, \
         bank dim={EMBED_DIM}, m={M}, β=0.7"
    );
    println!();

    // ── rank() latency ────────────────────────────────────────────────
    println!("--- rank() latency (median of {OUTER} samples, warmup {WARMUP}) ---");
    let mut rank_results: Vec<(usize, Duration, bool, Duration)> =
        Vec::with_capacity(RANK_SIZES.len());
    for &n in RANK_SIZES {
        let candidates = make_candidates(n, 0xBEEF);
        let bank = make_bank(100, 0xCAFE);
        let direction = make_direction(0xD0D0);
        let bank_len = bank.len();
        // Trait-path sampler (owns the scorer).
        let avail_for_trait = MedianTopMAvailability::new(bank.clone(), M);
        let sampler = AlienSampler::new(
            DotCoherence {
                direction: direction.clone(),
            },
            avail_for_trait,
            AlienConfig::paper_default(),
        );
        let dt_trait = bench_rank_trait(&sampler, &candidates, n);

        // Batch-path: rebuild the scorer pair for the hot-path measurement.
        // (We can't borrow `sampler.availability()` immutably while also
        // borrowing `sampler` for `rank_precomputed`, so we construct fresh
        // instances. The perf cost being measured is the same.)
        let avail_batch = MedianTopMAvailability::new(bank.clone(), M);
        let coh_batch = DotCoherence {
            direction: direction.clone(),
        };
        let sampler_batch = AlienSampler::new(
            DotCoherence {
                direction: direction.clone(),
            },
            MedianTopMAvailability::new(bank.clone(), M),
            AlienConfig::paper_default(),
        );
        let dt_batch = bench_rank_batch(
            &sampler_batch,
            &avail_batch,
            &coh_batch,
            &candidates,
            n,
            bank_len,
        );

        let target_us = if n <= 1_000 { 500.0 } else { 5_000.0 };
        let pass = dt_batch.as_secs_f64() * 1e6 <= target_us;
        println!(
            "  n={n:>6}: trait={:>8.2} µs  batch={:>8.2} µs  (target ≤ {target_us:.0} µs)  [{}]",
            dt_trait.as_secs_f64() * 1e6,
            dt_batch.as_secs_f64() * 1e6,
            pass_str(pass)
        );
        rank_results.push((n, dt_trait, pass, dt_batch));
    }
    println!();

    // ── median_top_m() latency ────────────────────────────────────────
    println!("--- median_top_m() latency (single call, median of {OUTER}, warmup {WARMUP}) ---");
    let mut mtm_results: Vec<(usize, Duration, bool)> = Vec::with_capacity(BANK_SIZES.len());
    for &bank_n in BANK_SIZES {
        let bank = make_bank(bank_n, 0xF00D);
        let avail = MedianTopMAvailability::new(bank, M);
        // Use a fixed candidate vector for the call.
        let candidate: Vec<f32> = make_candidates(1, 0x1234)[0].clone();
        let dt = bench_median_top_m(&avail, &candidate, bank_n);
        let target_us = if bank_n <= 100 { 5.0 } else { 500.0 };
        let pass = dt.as_secs_f64() * 1e6 <= target_us;
        println!(
            "  bank={bank_n:>6}: {:>8.2} µs  (target ≤ {target_us:.0} µs)  [{}]",
            dt.as_secs_f64() * 1e6,
            pass_str(pass)
        );
        mtm_results.push((bank_n, dt, pass));
    }
    println!();

    // ── Verdict ─────────────────────────────────────────────────────────
    let all_pass =
        rank_results.iter().all(|&(_, _, p, _)| p) && mtm_results.iter().all(|&(_, _, p)| p);
    println!("=== Phase 2 Microbench Verdict ===");
    for (n, dt_trait, p, dt_batch) in &rank_results {
        println!(
            "  rank n={n:>6}: trait={:.2} µs  batch={:.2} µs  [{}]",
            dt_trait.as_secs_f64() * 1e6,
            dt_batch.as_secs_f64() * 1e6,
            pass_str(*p)
        );
    }
    for (bank_n, dt, p) in &mtm_results {
        println!(
            "  median_top_m bank={bank_n:>6}: {:.2} µs  [{}]",
            dt.as_secs_f64() * 1e6,
            pass_str(*p)
        );
    }
    println!();
    if all_pass {
        println!(
            "  → All Phase 2 microbench targets met (batch path). Proceed to Phase 3 GOAT gate."
        );
    } else {
        println!("  → Some Phase 2 targets missed. Profile hot loops before GOAT gate.");
    }
}
