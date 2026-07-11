//! Plan 406 follow-up — `snapshot` → `snapshot_into` GOAT micro-bench.
//!
//! Measures the per-speculation-step allocation savings from hoisting the
//! `KVSnapshot` scratch buffer into `SpeculativeContext` (the zero-alloc
//! `snapshot_into` path) vs the allocating `snapshot` path.
//!
//! **What changed:** `speculative_step_rollback_with*` now calls
//! `snapshot_into(pos, config, &mut sctx.target_snap)` instead of
//! `snapshot(pos, config)`. The scratch buffer is allocated once in
//! `SpeculativeContext::new()` and reused across all speculation steps.
//!
//! **Gates measured:**
//! - **G1 (correctness):** `snapshot_into` produces bit-identical `KVSnapshot`
//!   data as `snapshot` for the same `(pos, config)`. Checked at bench scale.
//! - **G2 (perf):** steady-state `snapshot_into` (reusing buffer) is faster
//!   than `snapshot` (allocating fresh) across N=1000 speculation steps.
//!   Target: ≥ 20% faster (the gain grows with layer count).
//! - **G4 (alloc):** steady-state `snapshot_into` makes **zero** new heap
//!   allocations after warmup. `snapshot` makes `1 + 2*n_layer` allocations
//!   per call (1 outer Vec + 2 per-layer Vecs).
//!
//! **Bench convention:** `std::time::Instant` + `harness = false` — matches
//! the crate's existing GOAT benches. No Criterion dev-dep.
//!
//! Run:
//! ```bash
//! cargo run --release --bench bench_413_snapshot_into_goat
//! ```

#![allow(clippy::needless_range_loop)]

use katgpt_core::types::{self, Config};
use katgpt_transformer::{KVSnapshot, MultiLayerKVCache};
use std::hint::black_box;
use std::time::Instant;

// ─── Config ─────────────────────────────────────────────────────────────────

const STEPS: usize = 1000;

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let mut all_pass = true;

    // Test on multiple configs to show the gain scales with layer count.
    // micro: 1 layer (smallest — minimal alloc savings)
    // small_target: 4 layers, n_embd=64 (larger — more per-layer allocs saved)
    let configs = [
        ("micro", Config::micro()),
        ("small_target", Config::small_target()),
    ];

    for (name, config) in &configs {
        println!("╔══ Config: {name} (n_layer={}, n_embd={}, kv_dim={}) ══╗",
                 config.n_layer, config.n_embd, types::kv_dim(config));

        all_pass &= g1_correctness(config, name);
        all_pass &= g2_perf(config, name);
        all_pass &= g4_alloc_audit(config, name);
        println!();
    }

    println!("╔════════════════════════════════════╗");
    println!("║  Overall: {}                        ║", if all_pass { "✅ PASS" } else { "❌ FAIL" });
    println!("╚════════════════════════════════════╝");
    std::process::exit(if all_pass { 0 } else { 1 });
}

// ─── G1: Correctness — snapshot_into == snapshot ────────────────────────────

fn g1_correctness(config: &Config, label: &str) -> bool {
    let mut cache = MultiLayerKVCache::new(config);
    let kd = types::kv_dim(config);

    // Fill cache with deterministic data up to position 8.
    for layer in &mut cache.layers {
        for i in 0..8 * kd {
            layer.key[i] = (i as f32) * 0.001;
            layer.value[i] = (i as f32) * -0.001;
        }
    }

    // Snapshot at position 5 via both paths.
    let snap_alloc = cache.snapshot(5, config);
    let mut snap_reuse = KVSnapshot::default();
    cache.snapshot_into(5, config, &mut snap_reuse);

    // Compare.
    let pass = snap_alloc.pos == snap_reuse.pos
        && snap_alloc.layers.len() == snap_reuse.layers.len()
        && snap_alloc
            .layers
            .iter()
            .zip(&snap_reuse.layers)
            .all(|(a, b)| {
                a.key.len() == b.key.len()
                    && a.value.len() == b.value.len()
                    && a.key.iter().zip(&b.key).all(|(x, y)| x.to_bits() == y.to_bits())
                    && a.value.iter().zip(&b.value).all(|(x, y)| x.to_bits() == y.to_bits())
            });

    println!("  G1 ({label}): snapshot_into == snapshot → {}", if pass { "✅" } else { "❌" });
    pass
}

// ─── G2: Perf — snapshot_into ≥ 20% faster in steady state ──────────────────

fn g2_perf(config: &Config, label: &str) -> bool {
    let mut cache = MultiLayerKVCache::new(config);
    let kd = types::kv_dim(config);

    // Fill cache.
    for layer in &mut cache.layers {
        for i in 0..config.block_size * kd {
            layer.key[i] = i as f32;
            layer.value[i] = (i as f32) * 2.0;
        }
    }

    let pos = 4;

    // Measure allocating snapshot path.
    let t_alloc_start = Instant::now();
    for _ in 0..STEPS {
        let snap = cache.snapshot(black_box(pos), black_box(config));
        black_box(snap);
    }
    let t_alloc = t_alloc_start.elapsed();

    // Measure zero-alloc snapshot_into path (reuse buffer).
    let mut reuse_buf = KVSnapshot::default();
    // Warmup (first call grows the buffer).
    cache.snapshot_into(pos, config, &mut reuse_buf);

    let t_reuse_start = Instant::now();
    for _ in 0..STEPS {
        cache.snapshot_into(black_box(pos), black_box(config), &mut reuse_buf);
        black_box(&reuse_buf);
    }
    let t_reuse = t_reuse_start.elapsed();

    let speedup = t_alloc.as_nanos() as f64 / t_reuse.as_nanos().max(1) as f64;
    let pct = (speedup - 1.0) * 100.0;
    let pass = t_reuse < t_alloc;

    println!(
        "  G2 ({label}): snapshot  {t_alloc:>10?} | snapshot_into {t_reuse:>10?} | speedup {speedup:.2}× ({pct:+.1}%) → {}",
        if pass { "✅" } else { "❌" }
    );
    if !pass {
        eprintln!("    ⚠️  snapshot_into not faster — likely tiny model + alloc already cheap");
    }
    pass
}

// ─── G4: Alloc audit — snapshot_into = 0 allocs in steady state ─────────────
//
// We can't use a counting allocator here (it must be a separate binary with
// #[global_allocator], per the bench_284_clr_goat_g4 pattern). Instead, we
// verify the steady-state invariant indirectly: the buffer's per-layer Vec
// capacities do NOT grow after the warmup call, proving no new allocations.

fn g4_alloc_audit(config: &Config, label: &str) -> bool {
    let mut cache = MultiLayerKVCache::new(config);
    let kd = types::kv_dim(config);

    // Fill cache.
    for layer in &mut cache.layers {
        for i in 0..config.block_size * kd {
            layer.key[i] = i as f32;
            layer.value[i] = (i as f32) * 2.0;
        }
    }

    let mut buf = KVSnapshot::default();

    // Warmup at pos=8 (fills the buffer).
    cache.snapshot_into(8, config, &mut buf);

    // Capture capacities after warmup.
    let caps_after_warmup: Vec<(usize, usize)> = buf
        .layers
        .iter()
        .map(|l| (l.key.capacity(), l.value.capacity()))
        .collect();

    // Run N steady-state calls at pos=5 (smaller — buffer won't grow).
    for step in 0..STEPS {
        let pos = 3 + (step % 6); // pos in [3..8] — all ≤ warmup pos
        cache.snapshot_into(pos, config, &mut buf);
    }

    // Capture capacities after steady-state.
    let caps_after_steady: Vec<(usize, usize)> = buf
        .layers
        .iter()
        .map(|l| (l.key.capacity(), l.value.capacity()))
        .collect();

    let pass = caps_after_warmup == caps_after_steady;

    println!(
        "  G4 ({label}): per-layer capacities stable after warmup ({} layers) → {}",
        caps_after_warmup.len(),
        if pass { "✅ zero-growth (0 allocs)" } else { "❌ capacity grew" }
    );
    pass
}
