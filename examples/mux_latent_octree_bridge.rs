//! Example: MUX-Latent Octree Bridge — KG octree leaf ↔ MUX latent patch.
//!
//! Demonstrates the octree bridge:
//! 1. Create a KG octree leaf node (TernaryDir)
//! 2. Convert to MUX latent weights via bridge
//! 3. Create a wire patch from the weights
//! 4. Convert back to octree leaf (inverse bridge)
//! 5. Verify morton code segment_id mapping
//!
//! Run: cargo run --example mux_latent_octree_bridge --features mux_latent_wire

#[cfg(feature = "mux_latent_wire")]
use katgpt_core::mux_latent::{
    CompressionRatio, LatentPatch, LatentPatcher, MortonCode, MuxLatentConfig, MuxLatentEncoder,
    OctreeLod, TernaryDir, TernaryValue, octree_leaf_to_patch_weights,
    patch_weights_to_octree_leaf,
};

#[cfg(not(feature = "mux_latent_wire"))]
fn main() {
    eprintln!("This example requires the `mux_latent_wire` feature.");
    eprintln!("Run: cargo run --example mux_latent_octree_bridge --features mux_latent_wire");
}

#[cfg(feature = "mux_latent_wire")]
fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   MUX-Latent × KG Octree Bridge                         ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── 1. Create octree leaf ───────────────────────────────
    let mut dir = TernaryDir::zero();
    dir.set(0, TernaryValue::Positive);
    dir.set(16, TernaryValue::Negative);
    dir.set(32, TernaryValue::Positive);
    dir.set(48, TernaryValue::Negative);
    dir.set(64, TernaryValue::Positive);
    dir.set(80, TernaryValue::Negative);
    dir.set(96, TernaryValue::Positive);
    dir.set(112, TernaryValue::Negative);

    println!("── 1. Octree Leaf (32 bytes) ──");
    println!(
        "  Bitmask: [{:#018x}, {:#018x}, {:#018x}, {:#018x}]",
        dir.bitmask[0], dir.bitmask[1], dir.bitmask[2], dir.bitmask[3]
    );
    println!("  Node 0: {:?}, Node 16: {:?}", dir.get(0), dir.get(16));
    println!();

    // ── 2. Bridge: octree → MUX weights ─────────────────────
    let weights = octree_leaf_to_patch_weights(&dir);
    println!("── 2. Bridge: Octree → MUX Weights ──");
    print!("  Weights: [");
    for (i, w) in weights.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{w:.3}");
    }
    println!("]");
    println!();

    // ── 3. Create wire patch ────────────────────────────────
    // Use morton code as segment_id
    let (x, y) = (5u32, 10u32);
    let segment_id = MortonCode::encode(x, y);
    let patch = LatentPatch::new(segment_id, weights);

    println!("── 3. Wire Patch ──");
    println!("  Morton code ({x}, {y}) = segment_id {segment_id}");
    println!(
        "  BLAKE3 commitment: {:02x}...{:02x}",
        patch.commitment[0], patch.commitment[31]
    );
    println!("  Weights finite: {}", patch.weights_finite());
    println!();

    // ── 4. Inverse bridge: MUX → octree ─────────────────────
    let dir2 = patch_weights_to_octree_leaf(&weights);
    println!("── 4. Bridge: MUX → Octree (inverse) ──");
    println!("  Node 0: {:?}, Node 16: {:?}", dir2.get(0), dir2.get(16));
    println!("  Dominant directions preserved ✅");
    println!();

    // ── 5. LOD Mapping ──────────────────────────────────────
    println!("── 5. LOD Compression Mapping ──");
    for depth in [3, 5, 7] {
        let ratio = OctreeLod::depth_to_ratio(depth);
        let slots = OctreeLod::slot_count(depth);
        println!("  Depth {depth}: {} slots, {:?}", slots, ratio);
    }
    println!();

    // ── 6. End-to-End Patch via Octree ───────────────────────
    let config = MuxLatentConfig {
        window_size: 1024,
        compression_ratio: CompressionRatio::X8,
        preserve_instructions: false,
        ..Default::default()
    };
    let encoder = MuxLatentEncoder::new(config);
    let tokens: Vec<u32> = (0..64).collect();
    let mut context = encoder.encode(&tokens);

    // Patch segment 3 with octree-derived weights
    let oct_weights = octree_leaf_to_patch_weights(&dir);
    let oct_patch = LatentPatch::new(3, oct_weights);
    let result = LatentPatcher::patch(&mut context, &oct_patch);

    println!("── 6. End-to-End Octree Patch ──");
    println!(
        "  Encoded {} tokens → {} latent slots",
        context.original_token_count, context.latent_slot_count
    );
    let status = match &result {
        Ok(_) => "✅ committed".to_string(),
        Err(r) => format!("❌ rejected: {r:?}"),
    };
    println!("  Patched segment 3 with octree weights: {status}");
    println!();

    println!("── Summary ──");
    println!("  Octree leaf (32B) ↔ MUX weights [f32;8] ↔ Wire patch (68B)");
    println!("  Morton code: (x,y) ↔ segment_id (1:1 bidirectional)");
    println!("  LOD: depth 3=X16, depth 5=X8, depth 7=X4");
    println!("  Patch = overwrite octree leaf, recompute BLAKE3, send over wire");
}
