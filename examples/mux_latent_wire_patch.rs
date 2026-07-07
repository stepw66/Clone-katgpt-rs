//! Example: MUX-Latent Wire Patch — patch latent segments over the wire.
//!
//! Demonstrates the core workflow:
//! 1. Encode 256 tokens into 32 latent slots (X8 compression)
//! 2. Create patches for 3 segments with new weights
//! 3. Verify BLAKE3 commitments
//! 4. Apply patches to context in-place
//! 5. Show before/after weights
//!
//! Run: cargo run --example mux_latent_wire_patch --features mux_latent_wire

#[cfg(feature = "mux_latent_wire")]
use katgpt_rs::mux_latent::{
    CompressionRatio, LatentPatch, LatentPatchBatch, LatentPatcher, MuxLatentConfig,
    MuxLatentEncoder,
};

#[cfg(not(feature = "mux_latent_wire"))]
fn main() {
    eprintln!("This example requires the `mux_latent_wire` feature.");
    eprintln!("Run: cargo run --example mux_latent_wire_patch --features mux_latent_wire");
}

#[cfg(feature = "mux_latent_wire")]
fn main() {
    use katgpt_rs::mux_latent::LatentSegment;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   MUX-Latent Wire Patch — Latent-to-Latent on Wire     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── 1. Encode 256 tokens → 32 latent slots ──────────────
    let config = MuxLatentConfig {
        window_size: 1024,
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        ..Default::default()
    };
    let encoder = MuxLatentEncoder::new(config);
    let tokens: Vec<u32> = (0..256).collect();
    let mut context = encoder.encode(&tokens);

    println!(
        "Encoded {} tokens → {} latent slots",
        context.original_token_count, context.latent_slot_count
    );
    println!();

    // ── 2. Show before state ────────────────────────────────
    println!("── Before Patch ──");
    for sid in [0u32, 5, 15] {
        if let Some(seg) = context
            .segments
            .iter()
            .find(|s| s.segment_id() == Some(sid))
            && let LatentSegment::Compressed { weights, .. } = seg
        {
            let first_3 = &weights[..3.min(weights.len())];
            println!(
                "  Segment {sid}: [{:.3}, {:.3}, {:.3}, ...] ({} weights)",
                first_3[0],
                first_3.get(1).unwrap_or(&0.0),
                first_3.get(2).unwrap_or(&0.0),
                weights.len()
            );
        }
    }
    println!();

    // ── 3. Create patches ───────────────────────────────────
    let patches = vec![
        LatentPatch::new(0, [0.5f32; 8]),
        LatentPatch::new(5, [0.3f32; 8]),
        LatentPatch::new(15, [0.7f32; 8]),
    ];

    println!(
        "Created {} patches for segments {:?}",
        patches.len(),
        patches.iter().map(|p| p.segment_id).collect::<Vec<_>>()
    );
    println!();

    // ── 4. Verify BLAKE3 commitments ────────────────────────
    let batch = LatentPatchBatch::new(
        patches,
        context.latent_slot_count as u32,
        CompressionRatio::X8,
        0,
    );
    match batch.verify_all_commitments() {
        Ok(receipt) => {
            println!(
                "BLAKE3 verification: ✅ {} patches committed",
                receipt.committed.len()
            );
        }
        Err(receipt) => {
            println!(
                "BLAKE3 verification: ❌ {} rejected",
                receipt.rejected.len()
            );
            return;
        }
    }
    println!();

    // ── 5. Apply patches ────────────────────────────────────
    let receipt = LatentPatcher::patch_batch(&mut context, &batch);
    println!("── After Patch ──");
    println!("  Committed: {} segments", receipt.committed.len());
    println!("  Rejected: {} segments", receipt.rejected.len());
    println!();

    // Show updated weights
    for sid in [0u32, 5, 15] {
        if let Some(seg) = context
            .segments
            .iter()
            .find(|s| s.segment_id() == Some(sid))
            && let LatentSegment::Compressed { weights, .. } = seg
        {
            let first_3 = &weights[..3.min(weights.len())];
            println!(
                "  Segment {sid}: [{:.3}, {:.3}, {:.3}, ...] ({} weights)",
                first_3[0],
                first_3.get(1).unwrap_or(&0.0),
                first_3.get(2).unwrap_or(&0.0),
                weights.len()
            );
        }
    }
    println!();

    // ── 6. Summary ──────────────────────────────────────────
    println!("── Summary ──");
    println!("  Latent-to-latent patch: no decompress/recompress round-trip");
    println!("  Each patch: 68 bytes (4B segment_id + 32B weights + 32B BLAKE3)");
    println!("  Throughput target: ≥ 100K patches/sec SIMD batch");
    println!("  Security: BLAKE3 commitment + scalar projections only on wire");
}
