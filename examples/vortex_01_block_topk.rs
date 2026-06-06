//! BlockTopK VortexFlow router demo (Plan 196 T9).
//!
//! Demonstrates the simplest VortexFlow implementation: mean key centroid routing
//! with dot-product top-k selection and sigmoid-normalized weights.
//!
//! Run: `cargo run --example vortex_01_block_topk --features vortex_flow`

#![cfg(feature = "vortex_flow")]

use katgpt_rs::dash_attn::{BlockTopKRouter, VortexFlow, VortexScratch};

const HEAD_DIM: usize = 8;
const N_BLOCKS: usize = 16;
const BLOCK_SIZE: usize = 4;
const TOP_K: usize = 4;

fn main() {
    println!("=== BlockTopK VortexFlow Router Demo (Plan 196 T9) ===\n");

    // ── 1. Setup ─────────────────────────────────────────────
    println!("── Setup ──");
    let router = BlockTopKRouter::new(true); // scale = 1/sqrt(head_dim)
    println!("BlockTopKRouter: scale=true, head_dim={HEAD_DIM}, block_size={BLOCK_SIZE}");

    // ── 2. Synthetic KV Cache ────────────────────────────────
    // Each block has BLOCK_SIZE keys of dimension HEAD_DIM.
    // Block i uses a one-hot direction: centroid for block i is 1.0 at position i % HEAD_DIM.
    // This ensures each block is maximally distinct in direction, not magnitude.
    println!("\n── Building Synthetic KV Cache ({N_BLOCKS} blocks) ──");
    let mut all_keys = vec![0.0f32; N_BLOCKS * BLOCK_SIZE * HEAD_DIM];
    let all_values = vec![0.0f32; N_BLOCKS * BLOCK_SIZE * HEAD_DIM]; // values unused by BlockTopK

    for block in 0..N_BLOCKS {
        // Use a cyclic direction: each block's centroid points along dim (block % HEAD_DIM)
        let direction = block % HEAD_DIM;
        let magnitude = 1.0 + (block as f32 / N_BLOCKS as f32); // slight magnitude variation
        for token in 0..BLOCK_SIZE {
            let base = (block * BLOCK_SIZE + token) * HEAD_DIM;
            // Main direction + small noise
            for dim in 0..HEAD_DIM {
                let noise = 0.05 * (token as f32 * 0.3 - dim as f32 * 0.1);
                all_keys[base + dim] = if dim == direction {
                    magnitude + noise
                } else {
                    noise
                };
            }
        }
        println!(
            "  Block {block:>2}: direction=dim{direction}, keys[0..4]=[{}]",
            all_keys[block * BLOCK_SIZE * HEAD_DIM..][..4]
                .iter()
                .map(|v| format!("{v:.2}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // ── 3. Build Cache ───────────────────────────────────────
    println!("\n── Building BlockTopK Cache ──");
    let mut cache = router.cache_new(N_BLOCKS, HEAD_DIM);
    for block in 0..N_BLOCKS {
        let start = block * BLOCK_SIZE * HEAD_DIM;
        let end = start + BLOCK_SIZE * HEAD_DIM;
        router.forward_cache(
            &mut cache,
            &all_keys[start..end],
            &all_values[start..end],
            block,
            HEAD_DIM,
        );
    }
    println!("Cached {N_BLOCKS} block centroids");

    // Print centroids
    println!("\n  Centroids:");
    for block in 0..N_BLOCKS {
        let c = cache.centroid(block);
        println!(
            "    Block {block:>2}: [{}]",
            c.iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // ── 4. Query close to block 5 ────────────────────────────
    // Block 5 has direction = dim 5 (5 % 8 = 5).
    // Query points strongly along dim 5 to target block 5.
    // Blocks 5 and 13 (13 % 8 = 5) share the same direction,
    // so both should appear in the top-k results.
    println!("\n── Routing Query (aligned with block 5 direction) ──");
    let target_block = 5usize;
    let target_dir = target_block % HEAD_DIM;
    let mut query = vec![0.0f32; HEAD_DIM];
    query[target_dir] = 1.0;
    println!(
        "Query vector: [{}] (targets dim {target_dir})",
        query
            .iter()
            .map(|v| format!("{v:.1}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // ── 5. Route ─────────────────────────────────────────────
    let mut scratch = VortexScratch::new(N_BLOCKS);
    let decision = router.forward_indexer(&query, &cache, N_BLOCKS, TOP_K, &mut scratch);

    println!("\n── Routing Results (top_k={TOP_K}) ──");
    println!("  Selected blocks (sorted by relevance):");
    for (rank, (&block, &weight)) in decision
        .blocks
        .iter()
        .zip(decision.weights.iter())
        .enumerate()
    {
        let marker = if block == target_block {
            " ← target"
        } else {
            ""
        };
        println!("    #{rank}: block {block:>2}, weight={weight:.4}{marker}");
    }

    let found = decision.blocks.contains(&target_block);
    println!(
        "\n  Block {target_block} in selected set: {}",
        if found { "✓ YES" } else { "✗ NO" }
    );

    // ── 6. Comparison with Full Attention ─────────────────────
    println!("\n── Full Attention Comparison ──");
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut full_scores: Vec<(usize, f32)> = (0..N_BLOCKS)
        .map(|i| {
            let centroid = cache.centroid(i);
            let dot: f32 = query.iter().zip(centroid.iter()).map(|(a, b)| a * b).sum();
            (i, dot * scale)
        })
        .collect();
    full_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Full attention top-{TOP_K}:");
    for (rank, &(block, score)) in full_scores.iter().take(TOP_K).enumerate() {
        println!("    #{rank}: block {block:>2}, score={score:.4}");
    }

    let full_top_k: Vec<usize> = full_scores.iter().take(TOP_K).map(|&(b, _)| b).collect();
    let overlap = decision
        .blocks
        .iter()
        .filter(|b| full_top_k.contains(b))
        .count();
    let overlap_pct = overlap as f32 / TOP_K as f32 * 100.0;
    println!("\n  Overlap: {overlap}/{TOP_K} = {overlap_pct:.0}% (BlockTopK vs full attention)");

    // ── 7. Scratch Reuse Demo ────────────────────────────────
    println!("\n── RoutingScratch Reuse Demo (3 queries, 1 scratch buffer) ──");

    let queries: Vec<(&str, Vec<f32>)> = vec![
        // Each query targets a specific direction dimension
        ("dim 0 → block 0", {
            let mut q = vec![0.0; HEAD_DIM];
            q[0] = 1.0;
            q
        }),
        ("dim 2 → block 2", {
            let mut q = vec![0.0; HEAD_DIM];
            q[2] = 1.0;
            q
        }),
        ("dim 7 → block 7", {
            let mut q = vec![0.0; HEAD_DIM];
            q[7] = 1.0;
            q
        }),
    ];

    let scores_ptr = scratch.scores.as_ptr();

    for (label, q) in &queries {
        let decision = router.forward_indexer(q, &cache, N_BLOCKS, TOP_K, &mut scratch);
        println!("  Query ({label}): blocks = {:?}", decision.blocks);
    }

    // Verify scratch buffer was reused (no reallocation)
    let reused = scratch.scores.as_ptr() == scores_ptr;
    println!(
        "\n  Scratch buffer reused across 3 queries: {} (same allocation)",
        if reused { "✓ YES" } else { "✗ reallocated" }
    );

    // ── Summary ──────────────────────────────────────────────
    println!("\n=== Summary ===");
    println!("  BlockTopKRouter routes via mean centroid dot-product");
    println!("  Target block {target_block} correctly in top-{TOP_K}: {found}");
    println!("  Full attention overlap: {overlap_pct:.0}%");
    println!("  Scratch buffer zero-allocation reuse: {reused}");

    // GOAT gate
    if found && overlap_pct >= 50.0 && reused {
        println!("\n  🐐 GOAT gate: PASS");
    } else {
        println!(
            "\n  ⚠ GOAT gate: FAIL (found={found}, overlap={overlap_pct:.0}%, reused={reused})"
        );
    }
}
