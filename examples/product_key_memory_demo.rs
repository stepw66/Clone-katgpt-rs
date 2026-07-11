//! Product Key Memory (PKM) — O(√N) factored retrieval demo.
//!
//! Plan 408 Phase 6 T6.1. Three parts:
//!   1. Build a PKM table from random key-value pairs, query 100 random
//!      queries, print top-k + latency.
//!   2. Scale to N=10⁶ slots and show the O(√N) vs O(N) latency cliff
//!      (PKM vs brute-force).
//!   3. IDW vs Dot scoring comparison on a clustered dataset.
//!
//! Run: `cargo run --example product_key_memory_demo` (PKM is DEFAULT-ON in
//! katgpt-core since 2026-07-07, Plan 408 Phase 3 GOAT G1+G2+G4 ALL PASS).

use std::time::Instant;

use katgpt_core::product_key_memory::{PkmScratch, ProductKeyMemory, ScoreFn};

// ─── Const generics (match the Plan 408 G1 bench configuration) ─────────────
//
// SQRT_N=1000 → N = SQRT_N² = 1,000,000 slots. D_K=64 (split into two 32-dim
// halves). D_V=4 (retrieval cost is dominated by the codebook scans, not the
// value fetches — see .benchmarks/408_pkm_goat.md "Why D_V=4" note). Per-
// codebook top-K=8, final k=8 (the Cartesian product is 8×8=64 candidates).
const SQRT_N: usize = 1000;
const D_K: usize = 64;
const D_V: usize = 4;
const K: usize = 8;
const FINAL_K: usize = 8;

fn main() {
    println!("=== Product Key Memory (PKM) — O(√N) factored retrieval ===\n");
    part1_basic_retrieval();
    part2_latency_cliff();
    part3_idw_vs_dot();
}

// ─── Part 1: basic retrieval ─────────────────────────────────────────────────
//
// Build a table from a seed, run 100 random queries, report mean latency and
// a sample top-k result. Establishes that the retrieval works and is fast.

fn part1_basic_retrieval() {
    println!("--- Part 1: basic retrieval (100 queries) ---");
    let table: ProductKeyMemory<SQRT_N, D_K, D_V> = ProductKeyMemory::from_random(42);
    let mut scratch: PkmScratch<SQRT_N, K> = PkmScratch::new();
    let mut out = [(0usize, 0.0f32); FINAL_K];

    // Deterministic query stream from the same splitmix64 family.
    let mut qrng = SplitMix64::new(0x5eed_5eed);
    let mut total_ns = 0u64;
    let n_queries = 100usize;
    let mut sample_printed = false;

    for _ in 0..n_queries {
        let q = random_query(&mut qrng);
        let start = Instant::now();
        let n = table.query_into(&q, ScoreFn::Dot, FINAL_K, &mut out, &mut scratch);
        total_ns += start.elapsed().as_nanos() as u64;
        debug_assert_eq!(n, FINAL_K);

        if !sample_printed {
            println!(
                "  sample query top-{K}: {}",
                out[..FINAL_K]
                    .iter()
                    .map(|(idx, w)| format!("slot={idx:6} w={w:.4}"))
                    .collect::<Vec<_>>()
                    .join(",  ")
            );
            sample_printed = true;
        }
    }

    let mean_ns = total_ns / n_queries as u64;
    println!(
        "  N = {} slots (SQRT_N={SQRT_N}, D_K={D_K}, D_V={D_V})",
        SQRT_N * SQRT_N
    );
    println!("  {n_queries} queries, mean latency = {mean_ns} ns/query");
    println!();
}

// ─── Part 2: the O(√N) vs O(N) latency cliff ─────────────────────────────────
//
// At N=10⁶, PKM scores 2·SQRT_N = 2000 codebook rows per query; a brute-force
// O(N) scan scores N=10⁶ flat indices. The Plan 408 G1 gate measured a 1670×
// speedup (17.5µs PKM vs 29.2ms brute-force). This part demonstrates the same
// effect on a smaller brute-force sample (brute-force at full N is ~50ms/query,
// so we bound the brute-force iterations and extrapolate).

fn part2_latency_cliff() {
    println!("--- Part 2: O(√N) vs O(N) latency cliff (N=10⁶) ---");
    let table: ProductKeyMemory<SQRT_N, D_K, D_V> = ProductKeyMemory::from_random(7);
    let mut scratch: PkmScratch<SQRT_N, K> = PkmScratch::new();
    let mut out = [(0usize, 0.0f32); FINAL_K];
    let mut qrng = SplitMix64::new(0xc1ff_c1ff);

    // PKM path: time it directly (fast).
    const PKM_ITERS: usize = 1000;
    let mut pkm_ns = 0u64;
    for _ in 0..PKM_ITERS {
        let q = random_query(&mut qrng);
        let start = Instant::now();
        table.query_into(&q, ScoreFn::Dot, FINAL_K, &mut out, &mut scratch);
        pkm_ns += start.elapsed().as_nanos() as u64;
    }
    let pkm_mean_ns = pkm_ns / PKM_ITERS as u64;

    // Brute-force O(N) path: score every flat index, keep top-k via a simple
    // insertion sort. Bounded to BF_ITERS because each call is ~tens of ms.
    const BF_ITERS: usize = 5;
    let mut bf_ns = 0u64;
    let mut bf_top = [(0usize, f32::NEG_INFINITY); FINAL_K];
    let half = D_K / 2;
    for _ in 0..BF_ITERS {
        let q = random_query(&mut qrng);
        let (q1, q2) = q.split_at(half);
        let start = Instant::now();
        brute_force_top_k(&table, q1, q2, &mut bf_top);
        bf_ns += start.elapsed().as_nanos() as u64;
    }
    let bf_mean_ns = bf_ns / BF_ITERS as u64;
    let speedup = bf_mean_ns as f64 / pkm_mean_ns as f64;

    println!(
        "  PKM   O(√N): {PKM_ITERS} queries, mean = {pkm_mean_ns:>10} ns/query ({:.3} µs)",
        pkm_mean_ns as f64 / 1000.0
    );
    println!(
        "  BF    O(N):  {BF_ITERS} queries, mean = {bf_mean_ns:>10} ns/query ({:.3} ms)",
        bf_mean_ns as f64 / 1_000_000.0
    );
    println!("  speedup    : {speedup:.0}×  (Plan 408 G1 target ≥ 100×; bench measured 1670×)");
    println!();
}

// ─── Part 3: IDW vs Dot on a clustered dataset ───────────────────────────────
//
// Dot-product scoring rewards high-magnitude keys (a key can inflate its dot
// score by growing its norm). IDW scoring `−log(ε + ‖q−k‖²)` is magnitude-
// invariant — it attracts to the *nearest centroid* regardless of magnitude.
//
// Build a clustered table: codebook-1 has one low-magnitude cluster (near
// origin) and several high-magnitude clusters. A query near the origin should
// be retrieved by IDW (nearest) but distracted by Dot (high-magnitude). The
// Plan 408 G3 advisory gate + Phase 2 unit test `t27_idw_attracts_to_closer_centroids`
// formalize this; here we show it empirically.

fn part3_idw_vs_dot() {
    println!("--- Part 3: IDW vs Dot on a clustered dataset ---");
    let table = build_clustered_table();
    let mut scratch: PkmScratch<SQRT_N, K> = PkmScratch::new();
    let mut out_dot = [(0usize, 0.0f32); FINAL_K];
    let mut out_idw = [(0usize, 0.0f32); FINAL_K];

    // Query near the origin (cluster-0 centroid).
    let mut q = [0.0f32; D_K];
    // Small random perturbation so the query isn't exactly the centroid.
    let mut qrng = SplitMix64::new(0x5eed_1234);
    for x in q.iter_mut() {
        *x = next_f32_in_range(&mut qrng, -0.05, 0.05);
    }

    table.query_into(&q, ScoreFn::Dot, FINAL_K, &mut out_dot, &mut scratch);
    table.query_into(
        &q,
        ScoreFn::idw_default(),
        FINAL_K,
        &mut out_idw,
        &mut scratch,
    );

    // Measure how close (in Euclidean distance over the codebook-1 half-key)
    // the retrieved slots are to the query's first half. IDW should retrieve
    // slots whose codebook-1 rows are nearer to q1 than Dot's picks.
    let half = D_K / 2;
    let q1 = &q[..half];
    let dot_mean_dist = mean_euclidean_to_q1(&table, &out_dot, q1);
    let idw_mean_dist = mean_euclidean_to_q1(&table, &out_idw, q1);

    println!("  query near origin (cluster-0 centroid), {D_K}-dim");
    println!(
        "  Dot top-{FINAL_K} mean ‖q₁ − key₁‖ = {dot_mean_dist:.4}   (distracted by high-magnitude clusters)"
    );
    println!(
        "  IDW top-{FINAL_K} mean ‖q₁ − key₁‖ = {idw_mean_dist:.4}   (attracted to near low-magnitude cluster)"
    );
    if idw_mean_dist < dot_mean_dist {
        println!(
            "  → IDW wins by {:.4} ( nearer keys) — confirms the centroid-attraction property",
            dot_mean_dist - idw_mean_dist
        );
    } else {
        println!(
            "  → Dot happens to win on this seed (geometry can favor Dot; the property is statistical, see Phase 2 unit test t27)"
        );
    }
    println!();
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn random_query(rng: &mut SplitMix64) -> [f32; D_K] {
    let mut q = [0.0f32; D_K];
    for x in q.iter_mut() {
        *x = next_f32_in_range(rng, -1.0, 1.0);
    }
    q
}

fn brute_force_top_k(
    table: &ProductKeyMemory<SQRT_N, D_K, D_V>,
    q1: &[f32],
    q2: &[f32],
    out: &mut [(usize, f32); FINAL_K],
) {
    // Reset to sentinel.
    for e in out.iter_mut() {
        *e = (0, f32::NEG_INFINITY);
    }
    // Score every flat index = i * SQRT_N + j.
    for i in 0..SQRT_N {
        let k1 = table.keys_1_row(i);
        let s1 = katgpt_core::product_key_memory::score_dot(q1, k1);
        for j in 0..SQRT_N {
            let k2 = table.keys_2_row(j);
            let s2 = katgpt_core::product_key_memory::score_dot(q2, k2);
            let combined = s1 + s2;
            if combined <= out[FINAL_K - 1].1 {
                continue;
            }
            let flat = i * SQRT_N + j;
            let mut pos = 0;
            while pos < FINAL_K && combined <= out[pos].1 {
                pos += 1;
            }
            if pos < FINAL_K {
                out.copy_within(pos..FINAL_K - 1, pos + 1);
                out[pos] = (flat, combined);
            }
        }
    }
}

/// Build a clustered table: codebook-1 has cluster 0 near the origin (low
/// magnitude) and the remaining rows in a few high-magnitude clusters. This is
/// the fixture from the Plan 408 G3 advisory gate.
fn build_clustered_table() -> ProductKeyMemory<SQRT_N, D_K, D_V> {
    let mut rng = SplitMix64::new(0xbeef_babe);
    let half = D_K / 2;

    // Codebook 1: first 100 rows near origin (low magnitude), rest in a
    // high-magnitude cluster around radius 5 in dims [0,1].
    let mut keys_1 = vec![0.0f32; SQRT_N * half].into_boxed_slice();
    for i in 0..SQRT_N {
        let row = &mut keys_1[i * half..(i + 1) * half];
        if i < 100 {
            // Low-magnitude cluster near origin.
            for x in row.iter_mut() {
                *x = next_f32_in_range(&mut rng, -0.1, 0.1);
            }
        } else {
            // High-magnitude cluster: big values in dims [0,1].
            row[0] = next_f32_in_range(&mut rng, 4.0, 6.0);
            row[1] = next_f32_in_range(&mut rng, 4.0, 6.0);
            for x in row[2..].iter_mut() {
                *x = next_f32_in_range(&mut rng, -1.0, 1.0);
            }
        }
    }

    // Codebook 2: random (not the load-bearing axis for this demo).
    let mut keys_2 = vec![0.0f32; SQRT_N * half].into_boxed_slice();
    for x in keys_2.iter_mut() {
        *x = next_f32_in_range(&mut rng, -1.0, 1.0);
    }

    // Values: random.
    let mut values = vec![0.0f32; SQRT_N * SQRT_N * D_V].into_boxed_slice();
    for x in values.iter_mut() {
        *x = next_f32_in_range(&mut rng, -1.0, 1.0);
    }

    ProductKeyMemory::from_centroids(keys_1, keys_2, values)
}

fn mean_euclidean_to_q1(
    table: &ProductKeyMemory<SQRT_N, D_K, D_V>,
    top: &[(usize, f32)],
    q1: &[f32],
) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for &(flat, _w) in top {
        if flat == usize::MAX {
            continue;
        }
        let i = flat / SQRT_N;
        let k1 = table.keys_1_row(i);
        let mut ssd = 0.0f32;
        for (a, b) in q1.iter().zip(k1.iter()) {
            let d = a - b;
            ssd += d * d;
        }
        sum += ssd.sqrt();
        count += 1;
    }
    if count == 0 {
        f32::INFINITY
    } else {
        sum / count as f32
    }
}

// ─── tiny splitmix64 PRNG (mirrors the crate-internal SeededRng) ──────────────

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        // Top 24 bits → [0,1).
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn next_f32_in_range(rng: &mut SplitMix64, lo: f32, hi: f32) -> f32 {
    lo + rng.next_f32() * (hi - lo)
}
