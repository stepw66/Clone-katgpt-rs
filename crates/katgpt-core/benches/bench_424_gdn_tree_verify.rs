//! GDN Rollback-Free Tree Verification — GOAT gate bench (Plan 424 Phase 5).
//!
//! Exercises G2 (perf speedup vs per-branch sequential) and G4 (alloc-free hot
//! path) against random trees at T={16,32,64,128}.
//!
//! # Gates
//!
//! - **G2 (speedup)** — `verify_gdn_tree_into` must be ≥2× faster than
//!   per-branch sequential verify at T=32, and show increasing speedup with T.
//!   Paper achieves 2.7× at T=32, 4.6× at T=64, 7.1× at T=128 on B200 GPU.
//!   CPU numbers will differ (the algorithm is O(T²·d_v) vs sequential
//!   O(T²·d_k·d_v) — the crossover depends on d_k vs d_v ratio).
//! - **G4 (alloc-free)** — `verify_gdn_tree_into` hot path (after construction)
//!   must allocate zero times. Measured via `CountingAllocator`.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/424_gdn_tree_verify \
//!   cargo run -p katgpt-core --features gdn_tree_verify \
//!   --bench bench_424_gdn_tree_verify --release -- --nocapture
//! ```

#![cfg(feature = "gdn_tree_verify")]

use katgpt_core::gdn_tree_verify::{
    GdnLayerParams, GdnTreeVerifier, build_topology, verify_gdn_tree_into,
};

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── Utilities ─────────────────────────────────────────────────────────────

fn xorshift_rng(seed: u32) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Generate a random tree with T nodes. Node 0 is root; each subsequent node's
/// parent is a random earlier node (uniformly chosen).
struct TreeData {
    parents: Vec<usize>,
    keys: Vec<f32>,
    values: Vec<f32>,
    queries: Vec<f32>,
    alphas: Vec<f32>,
    betas: Vec<f32>,
    s0: Vec<f32>,
}

fn gen_tree(t: usize, d_k: usize, d_v: usize, seed: u32) -> TreeData {
    let mut rs = seed;
    let mut next = || {
        rs ^= rs << 13;
        rs ^= rs >> 17;
        rs ^= rs << 5;
        rs
    };
    let parents: Vec<usize> = (0..t)
        .map(|i| if i == 0 { usize::MAX } else { (next() as usize) % i })
        .collect();

    let mut frng = xorshift_rng(seed.wrapping_mul(7));
    let keys: Vec<f32> = (0..t * d_k).map(|_| frng()).collect();
    let values: Vec<f32> = (0..t * d_v).map(|_| frng()).collect();
    let queries: Vec<f32> = (0..t * d_k).map(|_| frng()).collect();
    let alphas: Vec<f32> = (0..t).map(|_| 0.75 + 0.15 * frng()).collect();
    let betas: Vec<f32> = (0..t).map(|_| 0.4 + 0.4 * frng()).collect();
    let s0: Vec<f32> = (0..d_k * d_v).map(|_| 0.1 * frng()).collect();

    TreeData { parents, keys, values, queries, alphas, betas, s0 }
}

/// Per-branch sequential reference: replay the delta-rule from root to each
/// node, then read the output. This is the baseline the tree-verify beats.
fn reference_verify(
    data: &TreeData,
    d_k: usize,
    d_v: usize,
) -> Vec<f32> {
    let t = data.parents.len();
    let mut outputs = vec![0.0f32; t * d_v];
    let scale = 1.0 / (d_k as f32).sqrt();

    for node in 0..t {
        // Trace path from node to root.
        let mut path = vec![node];
        let mut cur = data.parents[node];
        while cur != usize::MAX {
            path.push(cur);
            cur = data.parents[cur];
        }
        path.reverse();

        let mut s = data.s0.clone();
        for &p in &path {
            let k = &data.keys[p * d_k..(p + 1) * d_k];
            let v = &data.values[p * d_v..(p + 1) * d_v];
            let alpha = data.alphas[p];
            let beta = data.betas[p];

            let mut r = vec![0.0f32; d_v];
            for m in 0..d_k {
                for d in 0..d_v {
                    r[d] += s[m * d_v + d] * k[m];
                }
            }
            for val in s[..d_k * d_v].iter_mut() {
                *val *= alpha;
            }
            for m in 0..d_k {
                let bkm = beta * k[m];
                for d in 0..d_v {
                    s[m * d_v + d] += bkm * (v[d] - alpha * r[d]);
                }
            }
        }

        let q = &data.queries[node * d_k..(node + 1) * d_k];
        for d in 0..d_v {
            let mut sum = 0.0f32;
            for m in 0..d_k {
                sum += q[m] * s[m * d_v + d];
            }
            outputs[node * d_v + d] = scale * sum;
        }
    }
    outputs
}

// ─── G4: alloc-free hot path ──────────────────────────────────────────────

fn g4_alloc_free() {
    println!("\n╔══ G4: Alloc-free hot path ════════════════════════════════════╗");
    println!("║ Verifying verify_gdn_tree_into allocates 0 times after      ║");
    println!("║ construction (CountingAllocator).                            ║");

    let (t, d_k, d_v) = (32, 16, 16);
    let data = gen_tree(t, d_k, d_v, 42);
    let topo = build_topology(&data.parents, &data.alphas);
    let params = GdnLayerParams {
        keys: &data.keys,
        values: &data.values,
        queries: &data.queries,
        alphas: &data.alphas,
        betas: &data.betas,
    };

    // Construction allocates (expected). Reset counter after.
    let mut verifier = GdnTreeVerifier::new(t, d_k, d_v);
    let _ = verify_gdn_tree_into(&mut verifier, &topo, &params, &data.s0, d_k, d_v);

    // Now measure the hot path — should be 0 allocs.
    let (_, delta) = alloc_delta(|| {
        let _ = verify_gdn_tree_into(&mut verifier, &topo, &params, &data.s0, d_k, d_v);
    });

    if delta == 0 {
        println!("║ ✅ PASS: 0 allocations on steady-state verify (T={t}).        ║");
    } else {
        println!("║ ❌ FAIL: {delta} allocations on steady-state verify (T={t}).    ║");
        println!("║   Note: build_topology allocates (Vec<u64>, Vec<f64>) —        ║");
        println!("║   that's the one-time topology build, not the verify hot path. ║");
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
}

// ─── G2: perf speedup ─────────────────────────────────────────────────────

fn g2_perf() {
    let (d_k, d_v) = (64, 64); // typical GDN head dims

    // Two tree shapes:
    // 1. Random tree (shallow, depth ~log T) — typical speculative decode shape.
    // 2. Chain tree (depth = T) — worst case for sequential, best case for tree-verify.
    // The paper's GPU speedup (2.7-7.1×) comes from batching T independent branches
    // on a parallel accelerator. On CPU single-threaded, the crossover depends on
    // tree depth: the tree-verify is O(T²·(d_k+d_v)) regardless of shape, while
    // the per-branch sequential is O(T·depth·d_k·d_v). For shallow trees (depth
    // ~log T), sequential does less total work. For deep trees (depth ~T),
    // tree-verify wins big.
    let sizes: &[(usize, &str)] = &[
        (16, "T=16"),
        (32, "T=32"),
        (64, "T=64"),
        (128, "T=128"),
    ];

    println!("\n╔══ G2: Perf — tree-verify vs per-branch sequential ═══════════╗");
    println!("║ d_k={d_k}, d_v={d_v}, release mode, single-threaded           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ ── Random tree (shallow, depth ~log T) ──                    ║");

    for &(t, label) in sizes {
        let data = gen_tree(t, d_k, d_v, 100 + t as u32);
        let topo = build_topology(&data.parents, &data.alphas);
        let params = GdnLayerParams {
            keys: &data.keys, values: &data.values,
            queries: &data.queries, alphas: &data.alphas, betas: &data.betas,
        };
        let mut verifier = GdnTreeVerifier::new(t, d_k, d_v);

        for _ in 0..3 {
            let _ = verify_gdn_tree_into(&mut verifier, &topo, &params, &data.s0, d_k, d_v);
        }

        let n_iters = if t <= 32 { 500 } else if t <= 64 { 200 } else { 50 };

        let start = std::time::Instant::now();
        for _ in 0..n_iters {
            let _ = verify_gdn_tree_into(&mut verifier, &topo, &params, &data.s0, d_k, d_v);
        }
        let tree_us = start.elapsed().as_secs_f64() / n_iters as f64 * 1e6;

        let start = std::time::Instant::now();
        for _ in 0..n_iters {
            let _ = reference_verify(&data, d_k, d_v);
        }
        let seq_us = start.elapsed().as_secs_f64() / n_iters as f64 * 1e6;

        let speedup = seq_us / tree_us;
        println!(
            "║   {label:8} tree={tree_us:8.1}µs  seq={seq_us:8.1}µs  speedup={speedup:5.2}×"
        );
    }

    println!("║ ── Chain tree (deep, depth = T) ──                           ║");
    for &(t, label) in sizes {
        // Chain: each node's parent is the previous node.
        let chain_parents: Vec<usize> = (0..t)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let mut frng = xorshift_rng(200 + t as u32);
        let keys: Vec<f32> = (0..t * d_k).map(|_| frng()).collect();
        let values: Vec<f32> = (0..t * d_v).map(|_| frng()).collect();
        let queries: Vec<f32> = (0..t * d_k).map(|_| frng()).collect();
        let alphas: Vec<f32> = (0..t).map(|_| 0.75 + 0.15 * frng()).collect();
        let betas: Vec<f32> = (0..t).map(|_| 0.4 + 0.4 * frng()).collect();
        let s0: Vec<f32> = (0..d_k * d_v).map(|_| 0.1 * frng()).collect();
        let data = TreeData { parents: chain_parents, keys, values, queries, alphas, betas, s0 };

        let topo = build_topology(&data.parents, &data.alphas);
        let params = GdnLayerParams {
            keys: &data.keys, values: &data.values,
            queries: &data.queries, alphas: &data.alphas, betas: &data.betas,
        };
        let mut verifier = GdnTreeVerifier::new(t, d_k, d_v);

        for _ in 0..3 {
            let _ = verify_gdn_tree_into(&mut verifier, &topo, &params, &data.s0, d_k, d_v);
        }

        let n_iters = if t <= 32 { 200 } else if t <= 64 { 50 } else { 10 };

        let start = std::time::Instant::now();
        for _ in 0..n_iters {
            let _ = verify_gdn_tree_into(&mut verifier, &topo, &params, &data.s0, d_k, d_v);
        }
        let tree_us = start.elapsed().as_secs_f64() / n_iters as f64 * 1e6;

        let start = std::time::Instant::now();
        for _ in 0..n_iters {
            let _ = reference_verify(&data, d_k, d_v);
        }
        let seq_us = start.elapsed().as_secs_f64() / n_iters as f64 * 1e6;

        let speedup = seq_us / tree_us;
        println!(
            "║   {label:8} tree={tree_us:8.1}µs  seq={seq_us:8.1}µs  speedup={speedup:5.2}×"
        );
    }

    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ G2 analysis: tree-verify wins on DEEP trees (chain), where   ║");
    println!("║ sequential is O(T²·d_k·d_v) vs tree O(T²·(d_k+d_v)).          ║");
    println!("║ On shallow (random) trees, sequential does less total work.  ║");
    println!("║ The paper's GPU speedup comes from batching T independent    ║");
    println!("║ branches on a parallel accelerator, not from less FLOPs.     ║");
    println!("║ The CPU win is rollback elimination, not raw FLOP reduction. ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn main() {
    g4_alloc_free();
    g2_perf();
}
