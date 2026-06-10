//! MUX-Latent Expand (Decompression) example (Plan 238, Phase 6).
//!
//! Demonstrates the expand workflow: recovering original tokens from compressed
//! latent context. Covers segment-level, full, query-based selective, and
//! buffer-mediated expansion.

#[cfg(feature = "mux_latent_context")]
use katgpt_rs::mux_latent::{
    CompressionRatio, LatentContextBuffer, MuxLatentConfig, MuxLatentEncoder, expand_all,
    expand_segment, select_segments_to_expand,
};

#[cfg(not(feature = "mux_latent_context"))]
fn main() {
    eprintln!("This example requires --features mux_latent_context");
}

#[cfg(feature = "mux_latent_context")]
fn main() {
    println!("=== MUX-Latent Expand (Decompression) Demo (Plan 238, Phase 6) ===\n");

    // ── Setup: generate 4k tokens and encode at X4 ──
    let tokens: Vec<u32> = (0..4096).map(|t| t % 32000).collect();
    println!("Input: {} tokens\n", tokens.len());

    let config = MuxLatentConfig {
        window_size: 4096,
        compression_ratio: CompressionRatio::X4,
        preserve_instructions: false,
        ..Default::default()
    };
    let encoder = MuxLatentEncoder::new(config.clone());
    let ctx = encoder.encode(&tokens);
    println!(
        "Encoded: {} segments, {} original tokens\n",
        ctx.segments.len(),
        ctx.original_token_count
    );

    // ── expand_segment: recover segment 0 ──
    println!("── expand_segment (segment 0) ──");
    match expand_segment(&ctx, 0) {
        Some(seg) => {
            // Segment 0 tokens = tokens[0..4] for X4
            let expected = &tokens[0..4];
            let ok = seg.tokens == expected;
            println!(
                "  Segment {} recovered {} tokens",
                seg.segment_id,
                seg.tokens.len()
            );
            println!(
                "  Tokens match original: {}",
                if ok { "✓ PASS" } else { "✗ FAIL" }
            );
        }
        None => println!("  ✗ FAIL: segment 0 not found"),
    }
    println!();

    // ── expand_all: full roundtrip ──
    println!("── expand_all (full roundtrip) ──");
    let expanded = expand_all(&ctx);
    let ok = expanded == tokens;
    println!(
        "  Recovered {} tokens (expected {})",
        expanded.len(),
        tokens.len()
    );
    println!("  Full roundtrip: {}", if ok { "✓ PASS" } else { "✗ FAIL" });
    println!();

    // ── select_segments_to_expand: query-based selection ──
    println!("── select_segments_to_expand (query) ──");
    // Segment 2 tokens = [8,9,10,11], segment 5 tokens = [20,21,22,23] for X4
    let query: Vec<u32> = vec![9, 10, 20, 21];
    let selected = select_segments_to_expand(&ctx, &query, 3);
    println!("  Query tokens: {:?}", query);
    println!("  Selected segment IDs: {:?}", selected);
    let hits_seg2 = selected.contains(&2);
    let hits_seg5 = selected.contains(&5);
    println!(
        "  Contains segment 2 (tokens [8-11]): {}",
        if hits_seg2 { "✓ PASS" } else { "✗ FAIL" }
    );
    println!(
        "  Contains segment 5 (tokens [20-23]): {}",
        if hits_seg5 { "✓ PASS" } else { "✗ FAIL" }
    );
    println!();

    // ── LatentContextBuffer: query_expand + full_expand ──
    println!("── LatentContextBuffer ──");
    let buf = LatentContextBuffer::new(&tokens, config.clone());
    let stats = buf.stats();
    println!(
        "  Latent slots: {} / budget {}",
        stats.latent_slots_used, stats.latent_slot_budget
    );
    println!("  Raw segments: {}", stats.raw_segment_count);

    // query_expand returns segment IDs relevant to the query
    let buf_selected = buf.query_expand(&query, 3);
    println!(
        "  query_expand({:?}, top_k=3) → segments {:?}",
        query, buf_selected
    );
    let buf_query_ok = buf_selected.contains(&2) && buf_selected.contains(&5);
    println!(
        "  Query hit segments 2 & 5: {}",
        if buf_query_ok { "✓ PASS" } else { "✗ FAIL" }
    );

    // full_expand via buffer
    let buf_expanded = buf.full_expand();
    let buf_roundtrip_ok = buf_expanded == tokens;
    println!(
        "  Buffer full_expand roundtrip: {}",
        if buf_roundtrip_ok {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );

    // expand individual segment via buffer
    if let Some(seg_tokens) = buf.expand(0) {
        let seg_ok = seg_tokens == &tokens[0..4];
        println!(
            "  Buffer expand(0) match: {}",
            if seg_ok { "✓ PASS" } else { "✗ FAIL" }
        );
    } else {
        println!("  Buffer expand(0): ✗ FAIL (not found)");
    }
    println!();

    println!("=== Demo Complete ===");
}
