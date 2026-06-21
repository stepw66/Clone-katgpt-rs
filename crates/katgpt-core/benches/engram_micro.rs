//! Plan 299 T2.6 — Engram micro-benchmarks.
//!
//! Canonical criterion wiring for the G1 latency gate and the lower-level
//! primitives that feed it. Mirrors the convention of
//! [`micro_belief_bench.rs`] (`criterion_group!` + `criterion_main!`).
//!
//! # Benches covered
//!
//! - **T2.6 lookup** — `EngramTable::lookup_into` on a 1M-slot table with
//!   D=128 (paper's large-table regime). Target: < 50 ns amortized per K=16
//!   retrieval (i.e., the whole `lookup_into` call should take < 800 ns). The
//!   G1 gate in `tests/bench_299_engram_goat.rs` is the strict check; this
//!   bench is the regression watch.
//! - **multi_head_hash** — the suffix→K_MAX keys step. Should be a few ns;
//!   dominant cost is the K_MAX prime moduli.
//! - **sigmoid_fuse_into** — the per-pattern gate. Target: < 30 ns at D=128
//!   with SIMD engaged (NEON/AVX2).
//! - **hotswap_with_table** — `EngramHotSwap::with_table` steady-state read
//!   latency. Target: < 10 ns (single relaxed load + indirect call).
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench engram_micro --features engram
//! ```
//!
//! # Feature gate
//!
//! Requires `engram` (Plan 299).

#![cfg(feature = "engram")]
#![allow(clippy::needless_range_loop)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::engram::{
    CanonicalId, EngramConfig, EngramHash, EngramTable, EngramTableBuilder, HashHead, K_MAX,
    SigmoidFusionConfig, fuse_into_hidden_state, multi_head_hash, sigmoid_fuse_into,
};

/// Large-table regime per Plan 299 T2.6 — 1M slots.
const LOOKUP_N_SLOTS: usize = 1_000_000;

/// Hidden-state dimension used by the G1 gate and the lookup bench. Matches
/// the paper's "large model" config (D=128).
const LOOKUP_D: usize = 128;

// ─── T2.6: lookup_into @ 1M slots × D=128 ───────────────────────────────────

fn bench_lookup_into_1m(c: &mut Criterion) {
    let mut group = c.benchmark_group("engram/lookup_into");
    // criterion default sample_size is 100; bump to 500 for tighter CI on the
    // mean. Each iteration is one `lookup_into` call (~50 ns), so 500 iters
    // is well under criterion's per-bench time budget.
    group.sample_size(500);

    // Build the table outside the timed region. Populate ~1% of slots so hit
    // rate is realistic (matches G1 gate setup).
    let mut builder = EngramTableBuilder::new(LOOKUP_N_SLOTS, LOOKUP_D);
    let mut state = 0xC0FFEE_1234u64;
    let mut lcg = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    let n_populated = LOOKUP_N_SLOTS / 100;
    for _ in 0..n_populated {
        let slot = (lcg() as usize) % LOOKUP_N_SLOTS;
        let pat: Vec<f32> = (0..LOOKUP_D).map(|_| (lcg() >> 40) as f32 / (1u64 << 24) as f32).collect();
        builder.add_pattern(EngramHash(slot as u64), &pat);
    }
    let table = builder.build();

    // Pre-compute one set of K_MAX hash keys for the steady-state measurement.
    // (Inside the bench closure, we want to time ONLY `lookup_into`, not the
    // hash — `bench_multi_head_hash` measures that part separately.)
    let heads: [HashHead; K_MAX] = *table.heads();
    let suffix = [CanonicalId(1), CanonicalId(2), CanonicalId(3)];
    let keys = multi_head_hash(&suffix, &heads);

    // Pre-allocated output buffer — `lookup_into` is zero-allocation, and the
    // bench should reflect that.
    let mut out = vec![0.0f32; K_MAX * LOOKUP_D];

    group.bench_function("1m_slots_d128_k16", |b| {
        b.iter(|| {
            let hits = table.lookup_into(black_box(&keys), black_box(&mut out));
            black_box(hits);
        });
    });

    group.finish();
}

// ─── multi_head_hash ────────────────────────────────────────────────────────

fn bench_multi_head_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("engram/multi_head_hash");
    group.sample_size(500);

    // Re-use the table's heads so the bench reflects realistic head configs.
    let table = EngramTableBuilder::new(1024, 16).build();
    let heads = *table.heads();

    // Three suffix lengths to characterize the fold cost: 1 (unigram),
    // 3 (trigram — the paper default), 8 (long context).
    let suffix_1 = vec![CanonicalId(7)];
    let suffix_3 = vec![CanonicalId(7), CanonicalId(11), CanonicalId(13)];
    let suffix_8: Vec<CanonicalId> = (0..8).map(|i| CanonicalId(100 + i as u64)).collect();

    group.bench_function("suffix_1", |b| {
        b.iter(|| multi_head_hash(black_box(&suffix_1), black_box(&heads)));
    });
    group.bench_function("suffix_3", |b| {
        b.iter(|| multi_head_hash(black_box(&suffix_3), black_box(&heads)));
    });
    group.bench_function("suffix_8", |b| {
        b.iter(|| multi_head_hash(black_box(&suffix_8), black_box(&heads)));
    });

    group.finish();
}

// ─── sigmoid_fuse_into ──────────────────────────────────────────────────────

fn bench_sigmoid_fuse_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("engram/sigmoid_fuse_into");
    group.sample_size(500);

    // D=128 matches the G1 gate's table dim. The SIMD path engages when
    // D % 8 == 0; 128 satisfies that.
    let d = 128usize;
    let cfg = SigmoidFusionConfig {
        tau: (d as f32).sqrt(),
        rmsnorm_eps: 1e-6,
    };

    let q: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
    let k: Vec<f32> = (0..d).map(|i| (i as f32 * 0.07).cos()).collect();
    let v: Vec<f32> = (0..d).map(|i| (i as f32 * 0.05).tan() * 0.1).collect();
    let mut out = vec![0.0f32; d];

    group.bench_function("d128", |b| {
        b.iter(|| {
            sigmoid_fuse_into(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(&mut out),
                black_box(&cfg),
            );
        });
    });

    group.finish();
}

// ─── fuse_into_hidden_state (end-to-end, K=16 retrievals + K gates) ─────────

fn bench_fuse_into_hidden_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("engram/fuse_into_hidden_state");
    group.sample_size(500);

    let d = 128usize;
    let mut builder = EngramTableBuilder::new(4096, d);
    let mut state = 0xDEAD_BEEFu64;
    let mut lcg = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    for _ in 0..64 {
        let slot = (lcg() as usize) % 4096;
        let pat: Vec<f32> = (0..d).map(|_| (lcg() >> 40) as f32 / (1u64 << 24) as f32).collect();
        builder.add_pattern(EngramHash(slot as u64), &pat);
    }
    let table = builder.build();
    let heads = *table.heads();
    let keys = multi_head_hash(&[CanonicalId(1), CanonicalId(2), CanonicalId(3)], &heads);

    let mut hidden = vec![0.0f32; d];
    let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
    let cfg = EngramConfig::for_dim(d);

    let mut scratch_lookup = vec![0.0f32; K_MAX * d];
    let mut scratch_norm = vec![0.0f32; d];
    let mut scratch_out = vec![0.0f32; d];

    group.bench_function("d128_k16", |b| {
        b.iter(|| {
            // Reset hidden to keep the gate math identical across iterations.
            for x in hidden.iter_mut() {
                *x = 0.0;
            }
            fuse_into_hidden_state(
                black_box(&mut hidden),
                black_box(&query),
                black_box(&table),
                black_box(&keys),
                black_box(&cfg),
                black_box(&mut scratch_lookup),
                black_box(&mut scratch_norm),
                black_box(&mut scratch_out),
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_lookup_into_1m,
    bench_multi_head_hash,
    bench_sigmoid_fuse_into,
    bench_fuse_into_hidden_state,
);
criterion_main!(benches);
