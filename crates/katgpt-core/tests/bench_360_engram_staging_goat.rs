//! StagingEngramTable GOAT gate (Plan 360 Phase 2).
//!
//! G1 — mutation isolation: source table untouched (COW), new table has the
//!       5 UPDATEs + 2 DELETEs applied, unaffected slots match source bit-for-bit.
//! G2 — perf: surgical update vs whole-table rebuild (1M-slot × D=64 table).
//!       Path A (staging COW) vs Path B (rebuild-from-scratch) vs Path C
//!       (rebuild-from-source via builder). Pass: A/C < 0.1 (≥10× faster).
//! G3 — no regression (verified externally via the feature-flag build matrix).
//! G4 — allocation accounting: update_slot = 1 alloc/call (pattern copy),
//!       delete_slot = 0 allocs/call (None), commit = 2 allocs (slots COW + heads).
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features engram --test bench_360_engram_staging_goat --release -- --nocapture
//! ```
//!
//! Or, working around the intermittent macOS dyld/trustd launch stall:
//!
//! ```bash
//! target/release/deps/bench_360_engram_staging_goat-* --nocapture
//! ```
//!
//! # Memory
//!
//! G2 builds a 1M-slot × D=64 table (~256 MB f32). Peak memory ~512 MB
//! (source + one new table alive at a time). Requires a machine with
//! ≥ 2 GB free RAM. Run with `--release` — debug mode is too slow for the
//! 1M-slot comparison.

#![cfg(feature = "engram")]

use katgpt_core::{
    EngramHash, EngramTable, EngramTableBuilder, InMemoryEngramTable, K_MAX, StagingEngramTable,
};
use std::hint::black_box;
use std::time::Instant;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

// ─── GateResult ─────────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

impl GateResult {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, passed: true, detail: detail.into() }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, passed: false, detail: detail.into() }
    }
}

// ─── Lcg (deterministic fixture RNG — no rand dep) ──────────────────────────

struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = self.state;
        ((x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9)) >> 32 ^ (x >> 17)
    }
    fn next_u32(&mut self, bound: u32) -> u32 {
        (self.next_u64() as u32) % bound
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Build a table with all slots populated with distinct non-zero patterns.
/// Slot `i` gets pattern `[(i+1) as f32; d]` — the `+1` ensures slot 0 is NOT
/// all-zeros (which would be indistinguishable from an empty/unpopulated slot,
/// breaking the G1 mutation-isolation check). Mirrors `make_distinct_table`
/// in `src/engram/staging.rs` unit tests, but uses the PUBLIC API only
/// (`EngramTableBuilder::add_pattern`) since this is an integration test crate
/// and `InMemoryEngramTable::slots()` is `pub(crate)`.
fn make_distinct_table(n_slots: usize, d: usize) -> InMemoryEngramTable {
    let mut b = EngramTableBuilder::new(n_slots, d);
    let mut pat = vec![0.0f32; d];
    for i in 0..n_slots as u64 {
        let v = (i + 1) as f32;
        pat.fill(v);
        b.add_pattern(EngramHash(i), &pat);
    }
    b.build()
}

/// Read a single slot's pattern via the public `lookup_into` API.
///
/// `InMemoryEngramTable::slots()` is `pub(crate)`, so the integration test
/// cannot index into the raw slot array. Instead we pass `EngramHash(slot_idx)`
/// as head 0 — since `slot_idx < num_slots`, `hash % num_slots == slot_idx`,
/// and the first head's output (`out[0..d]`) is the slot's pattern.
fn read_slot(table: &InMemoryEngramTable, slot_idx: usize) -> Vec<f32> {
    let d = table.dim();
    let mut keys = [EngramHash(0); K_MAX];
    keys[0] = EngramHash(slot_idx as u64);
    let mut out = vec![0.0f32; K_MAX * d];
    table.lookup_into(&keys, &mut out);
    out[..d].to_vec()
}

/// Read `K_MAX` contiguous slots `[start, start+K_MAX-1]` into `out` in one
/// `lookup_into` call. `out` must be at least `K_MAX * d` long. Used by G2
/// Path C to read all 1M slots from the source efficiently (62500 calls vs
/// 1M single-slot reads).
fn read_k_slots(table: &InMemoryEngramTable, start: usize, out: &mut [f32]) {
    let d = table.dim();
    let keys = std::array::from_fn(|k| EngramHash((start + k) as u64));
    table.lookup_into(&keys, out);
    debug_assert!(out.len() >= K_MAX * d);
}

// ─── G1: mutation isolation ─────────────────────────────────────────────────

fn gate_g1_mutation_isolation() -> GateResult {
    const N: usize = 1024;
    const D: usize = 32;

    let source = make_distinct_table(N, D);

    // LCG-random slot picks for 5 UPDATEs + 2 DELETEs.
    let mut rng = Lcg::new(42);
    let update_slots: Vec<usize> = (0..5).map(|_| rng.next_u32(N as u32) as usize).collect();
    let delete_slots: Vec<usize> = (0..2).map(|_| rng.next_u32(N as u32) as usize).collect();

    // New patterns for the 5 updates — distinct values not in the original table.
    let new_patterns: Vec<Vec<f32>> =
        update_slots.iter().map(|&i| vec![1000.0 + i as f32; D]).collect();

    // Stage + commit.
    let mut staging = StagingEngramTable::from_table(&source);
    for (&slot, pat) in update_slots.iter().zip(&new_patterns) {
        staging.update_slot(slot, pat).unwrap();
    }
    for &slot in &delete_slots {
        staging.delete_slot(slot).unwrap();
    }
    let new_table = staging.commit().unwrap();

    // ── Sub-check (a): source untouched ──────────────────────────────────
    //
    // COW is a compile-time guarantee (StagingEngramTable borrows &source
    // immutably), but verify empirically: every source slot must still hold
    // its original distinct pattern [(i+1) as f32; d]. A mutation leak would
    // show up as a changed value here.
    let mut source_ok = true;
    for i in 0..N {
        let actual = read_slot(&source, i);
        if !actual.iter().all(|&v| v == (i as f32 + 1.0)) {
            source_ok = false;
            break;
        }
    }

    // ── Sub-check (b): updated slots match new patterns ──────────────────
    let mut updates_ok = true;
    for (&slot, pat) in update_slots.iter().zip(&new_patterns) {
        if read_slot(&new_table, slot) != *pat {
            updates_ok = false;
            break;
        }
    }

    // ── Sub-check (c): deleted slots are all-zero ────────────────────────
    let mut deletes_ok = true;
    for &slot in &delete_slots {
        if read_slot(&new_table, slot).iter().any(|&v| v != 0.0) {
            deletes_ok = false;
            break;
        }
    }

    // ── Sub-check (d): unaffected slots match source bit-for-bit ─────────
    let mutated: Vec<usize> = update_slots.iter().chain(delete_slots.iter()).copied().collect();
    let mut unaffected_ok = true;
    let mut checked = 0usize;
    for i in 0..N {
        if mutated.contains(&i) {
            continue;
        }
        if read_slot(&new_table, i) != read_slot(&source, i) {
            unaffected_ok = false;
            break;
        }
        checked += 1;
    }

    if source_ok && updates_ok && deletes_ok && unaffected_ok {
        GateResult::pass(
            "G1",
            format!(
                "mutation isolation: source untouched (compile-time COW), {} updates applied, \
                 {} deletes zeroed, {}/{} unaffected slots bit-for-bit match",
                update_slots.len(),
                delete_slots.len(),
                checked,
                N,
            ),
        )
    } else {
        let mut fails = Vec::new();
        if !source_ok {
            fails.push("source mutated");
        }
        if !updates_ok {
            fails.push("update mismatch");
        }
        if !deletes_ok {
            fails.push("delete not zeroed");
        }
        if !unaffected_ok {
            fails.push("unaffected slot mismatch");
        }
        GateResult::fail("G1", format!("sub-checks failed: {}", fails.join(", ")))
    }
}

// ─── G2: surgical update vs whole-table rebuild ─────────────────────────────

fn gate_g2_surgical_vs_rebuild() -> GateResult {
    const N: usize = 1_000_000;
    const D: usize = 64;
    const NEW_VAL: f32 = 999.0;
    const MUTATE_SLOT: usize = 42;
    // Each path is run WARMUP_ITERS + 1 times. The warmup runs prime the page
    // allocator (fresh 256MB pages faulted in), stabilize the CPU frequency
    // (turbo boost ramp), and warm the branch predictor. Without warmup, the
    // first-measured path (always Path A, since it runs first) is penalized by
    // cold-start page-fault costs that have nothing to do with the algorithm.
    const WARMUP_ITERS: usize = 2;

    println!(
        "    G2: building {N}-slot × D={D} source table (~{:.0} MB)...",
        (N * D * 4) as f64 / 1_048_576.0
    );
    let source = make_distinct_table(N, D);
    let new_pat = vec![NEW_VAL; D];

    // ── Path A: staging (COW copy + 1 mutation) ──────────────────────────
    let path_a = {
        // Warmup runs — untimed, just to prime the allocator + CPU.
        for _ in 0..WARMUP_ITERS {
            let _ = StagingEngramTable::from_table(&source)
                .update_slot(MUTATE_SLOT, &new_pat)
                .expect("slot 42 in bounds")
                .commit()
                .expect("1 pending mutation");
        }
        // Measured run.
        let t0 = Instant::now();
        let new = StagingEngramTable::from_table(&source)
            .update_slot(MUTATE_SLOT, &new_pat)
            .expect("slot 42 in bounds")
            .commit()
            .expect("1 pending mutation");
        let elapsed = t0.elapsed();
        let _ = black_box(&new);
        elapsed
    };

    // ── Path B: rebuild from scratch (re-derive every pattern) ──────────
    //
    // Worst case: the caller has lost the original patterns and must
    // re-derive each one before writing to a fresh builder.
    let path_b = {
        for _ in 0..WARMUP_ITERS {
            let mut b = EngramTableBuilder::new(N, D);
            let mut pat = vec![0.0f32; D];
            for i in 0..N as u64 {
                let v = (i + 1) as f32;
                pat.fill(v);
                b.add_pattern(EngramHash(i), &pat);
            }
            pat.fill(NEW_VAL);
            b.add_pattern(EngramHash(MUTATE_SLOT as u64), &pat);
            let _ = b.build();
        }
        let t0 = Instant::now();
        let mut b = EngramTableBuilder::new(N, D);
        let mut pat = vec![0.0f32; D];
        for i in 0..N as u64 {
            let v = (i + 1) as f32;
            pat.fill(v);
            b.add_pattern(EngramHash(i), &pat);
        }
        pat.fill(NEW_VAL);
        b.add_pattern(EngramHash(MUTATE_SLOT as u64), &pat);
        let new = b.build();
        let elapsed = t0.elapsed();
        let _ = black_box(&new);
        elapsed
    };

    // ── Path C: rebuild from source (read + re-add every slot) ──────────
    //
    // Realistic rebuild: the caller has the source table but no staging
    // primitive, so they must read every slot via lookup_into and write it
    // to a fresh builder.
    let path_c = {
        for _ in 0..WARMUP_ITERS {
            let mut b = EngramTableBuilder::new(N, D);
            let mut out = vec![0.0f32; K_MAX * D];
            for chunk_start in (0..N).step_by(K_MAX) {
                read_k_slots(&source, chunk_start, &mut out);
                for k in 0..K_MAX {
                    let slot = chunk_start + k;
                    b.add_pattern(EngramHash(slot as u64), &out[k * D..(k + 1) * D]);
                }
            }
            b.add_pattern(EngramHash(MUTATE_SLOT as u64), &new_pat);
            let _ = b.build();
        }
        let t0 = Instant::now();
        let mut b = EngramTableBuilder::new(N, D);
        let mut out = vec![0.0f32; K_MAX * D];
        for chunk_start in (0..N).step_by(K_MAX) {
            read_k_slots(&source, chunk_start, &mut out);
            for k in 0..K_MAX {
                let slot = chunk_start + k;
                b.add_pattern(EngramHash(slot as u64), &out[k * D..(k + 1) * D]);
            }
        }
        b.add_pattern(EngramHash(MUTATE_SLOT as u64), &new_pat);
        let new = b.build();
        let elapsed = t0.elapsed();
        let _ = black_box(&new);
        elapsed
    };

    let a_ms = path_a.as_secs_f64() * 1e3;
    let b_ms = path_b.as_secs_f64() * 1e3;
    let c_ms = path_c.as_secs_f64() * 1e3;
    let a_over_c = path_a.as_secs_f64() / path_c.as_secs_f64();
    let a_over_b = path_a.as_secs_f64() / path_b.as_secs_f64();
    let c_speedup = 1.0 / a_over_c;
    let b_speedup = 1.0 / a_over_b;

    let stretch = a_over_c < 0.01;
    let passed = a_over_c < 0.1; // ≥10× faster than realistic rebuild

    if passed {
        let bar = if stretch { "≥100× STRETCH" } else { "≥10× bar" };
        GateResult::pass(
            "G2",
            format!(
                "staging {a_ms:.1}ms vs rebuild-from-source {c_ms:.1}ms ({c_speedup:.1}× faster, \
                 A/C={a_over_c:.4}, {bar} PASSED) | rebuild-from-scratch {b_ms:.1}ms \
                 ({b_speedup:.1}× faster, A/B={a_over_b:.4})"
            ),
        )
    } else {
        GateResult::fail(
            "G2",
            format!(
                "staging {a_ms:.1}ms vs rebuild-from-source {c_ms:.1}ms ({c_speedup:.1}× faster, \
                 A/C={a_over_c:.4}, ≥10× bar FAILED) | rebuild-from-scratch {b_ms:.1}ms \
                 ({b_speedup:.1}× faster, A/B={a_over_b:.4})"
            ),
        )
    }
}

// ─── G4: allocation accounting ──────────────────────────────────────────────

fn gate_g4_allocation_accounting() -> GateResult {
    const N: usize = 1024;
    const D: usize = 32;
    const ITERS: usize = 1000;

    let table = make_distinct_table(N, D);
    let pattern = vec![1.0f32; D];

    // G4a: update_slot — expect exactly 1 alloc per call (pattern.to_vec()).
    // The pending.push is amortized O(1) via with_capacity → 0 allocs within capacity.
    // NOTE: the staging table is created OUTSIDE alloc_delta so its with_capacity
    // allocation isn't counted — we're measuring the mutation hot path only.
    let mut s_update = StagingEngramTable::with_capacity(&table, ITERS);
    let (_, update_allocs) = alloc_delta(|| {
        for _ in 0..ITERS {
            s_update.update_slot(0, black_box(&pattern)).expect("slot 0 in bounds");
        }
    });
    let _ = black_box(&s_update);

    // G4b: delete_slot — expect 0 allocs per call.
    // None doesn't allocate; push is within capacity.
    let mut s_delete = StagingEngramTable::with_capacity(&table, ITERS);
    let (_, delete_allocs) = alloc_delta(|| {
        for _ in 0..ITERS {
            s_delete.delete_slot(0).expect("slot 0 in bounds");
        }
    });
    let _ = black_box(&s_delete);

    // G4c: commit — expect exactly 2 allocs (slots COW copy + heads Box copy).
    let mut s = StagingEngramTable::from_table(&table);
    s.update_slot(0, &pattern).expect("slot 0 in bounds");
    let (_, commit_allocs) = alloc_delta(|| {
        let _ = black_box(s.commit());
    });

    let update_ok = update_allocs == ITERS; // 1 per call
    let delete_ok = delete_allocs == 0;
    let commit_ok = commit_allocs == 2; // slots + heads

    if update_ok && delete_ok && commit_ok {
        GateResult::pass(
            "G4",
            format!(
                "update_slot: {update_allocs} allocs/{ITERS} (1/call — pattern copy) | \
                 delete_slot: {delete_allocs} allocs/{ITERS} (0/call — None) | \
                 commit: {commit_allocs} allocs (slots COW + heads)"
            ),
        )
    } else {
        GateResult::fail(
            "G4",
            format!(
                "alloc budget violated: update_slot={update_allocs}/{ITERS} (exp {ITERS}), \
                 delete_slot={delete_allocs}/{ITERS} (exp 0), commit={commit_allocs} (exp 2)"
            ),
        )
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 360 - StagingEngramTable GOAT Gate (Phase 2) ===\n");

    let g1 = gate_g1_mutation_isolation();
    let g2 = gate_g2_surgical_vs_rebuild();
    let g4 = gate_g4_allocation_accounting();

    let gates = [&g1, &g2, &g4];
    let mut all_pass = true;
    for g in &gates {
        let status = if g.passed { "PASS" } else { "FAIL" };
        println!("[{status}] {}: {}", g.name, g.detail);
        if !g.passed {
            all_pass = false;
        }
    }

    println!();
    println!("[NOTE] G3: no-regression build matrix verified externally via:");
    println!("    cargo test -p katgpt-core --features engram --lib");
    println!("    cargo test -p katgpt-core --lib  (default features, engram OFF)");
    println!("    cargo check -p katgpt-core --all-features");
    println!();

    if all_pass {
        println!("=== G1, G2, G4 ALL PASS — StagingEngramTable is GOAT-gated ===");
        println!("    (eligible for promotion per Phase 4 decision)");
    } else {
        println!("=== ONE OR MORE GATES FAILED — keep opt-in, document honest result ===");
        let failed: Vec<&str> = gates.iter().filter(|g| !g.passed).map(|g| g.name).collect();
        println!("    Failed gates: {failed:?}");
    }
    // Exit 0 regardless — the gate verdict is communicated via the printed
    // PASS/FAIL table and the benchmark doc, not via the process exit code.
    // (Matches the bench_331 convention: an honest negative result is itself
    // a successful bench run.)
    std::process::exit(0);
}
