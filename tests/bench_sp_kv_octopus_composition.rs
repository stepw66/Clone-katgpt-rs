//! SP-KV + OCTOPUS cross-crate composition proof.
//!
//! Verifies `SpKvQuantCache<OctopusKVCache>` (from katgpt-kv) composes correctly
//! with `OctopusKVCache` (from katgpt-quant) — the GOAT 022 acceptance criterion.
//!
//! Originally inlined in `katgpt-quant/src/octopus/kv_cache.rs` as
//! `crate::sp_kv::types::{...}` references, which became dangling after the
//! Issue 015 Phase 3 crate split moved `sp_kv` to `katgpt-kv`. Relocated here
//! (root integration tests) because this is the natural home for cross-crate
//! composition proofs — the root crate re-exports both `sp_kv` (via katgpt-kv)
//! and `octopus` (via katgpt-quant).
//!
//! Run with: cargo test --features "sp_kv octopus" bench_sp_kv_octopus_composition -- --nocapture

#![cfg(all(feature = "sp_kv", feature = "octopus"))]

use katgpt_rs::octopus::{OctopusConfig, OctopusKVCache};
use katgpt_rs::sp_kv::types::{SpKvConfig, SpKvQuantCache};

/// Build an OCTOPUS cache matching the historical `make_cache(dim, kb, vb)` shape.
fn make_cache(kv_dim: usize, key_bits: u8, val_bits: u8) -> OctopusKVCache {
    let cfg = OctopusConfig {
        key_bits,
        val_bits,
        seed: 42,
        n_layers: 2,
        kv_dim,
        max_seq_len: 32,
        use_qjl_residual: false,
        use_joint_rounding: true,
    };
    OctopusKVCache::with_config(&cfg)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    dot / (na * nb)
}

// ── SP-KV + OCTOPUS Composition Proof ────────────────────
// Verifies SpKvQuantCache<OctopusKVCache> compiles and works end-to-end.
// Acceptance criterion: "SpKvQuantCache<OctopusKVCache> compiles (composition proof)".

#[test]
fn test_sp_kv_octopus_composition_compiles() {
    let octopus = make_cache(64, 3, 3);
    let sp_cfg = SpKvConfig::default();
    let mut hybrid: SpKvQuantCache<OctopusKVCache> = SpKvQuantCache::new(sp_cfg, octopus, 2, 32);

    let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1 - 3.2).sin()).collect();
    let val: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).cos()).collect();

    // High utility → should be retained and written
    let written = hybrid.write_gated(0, &key, &val, 0.9, 0, false, 0.5);
    assert!(written, "high utility should be retained");
    assert!(hybrid.meta[0].retained[0]);
    assert_eq!(hybrid.meta[0].retained_count, 1);

    // Low utility, not in window → should be pruned
    let written = hybrid.write_gated(0, &key, &val, 0.1, 1, false, 0.5);
    assert!(!written, "low utility should be pruned");
    assert!(!hybrid.meta[0].retained[1]);
    assert_eq!(hybrid.meta[0].retained_count, 1);

    // In window → should be retained regardless of utility
    let written = hybrid.write_gated(0, &key, &val, 0.01, 2, true, 0.5);
    assert!(written, "in-window should always be retained");
    assert!(hybrid.meta[0].retained[2]);
    assert_eq!(hybrid.meta[0].retained_count, 2);
}

#[test]
fn test_sp_kv_octopus_roundtrip() {
    let octopus = make_cache(64, 3, 3);
    let sp_cfg = SpKvConfig::default();
    let mut hybrid: SpKvQuantCache<OctopusKVCache> = SpKvQuantCache::new(sp_cfg, octopus, 2, 32);

    let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1 - 3.2).sin()).collect();
    let val: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).cos()).collect();

    // Write with high utility so it's retained
    hybrid.write_gated(0, &key, &val, 0.99, 0, false, 0.5);

    // Dequantize through the quant backend
    let mut recon_key = vec![0.0f32; 64];
    hybrid.quant.dequantize_key_into(0, 0, &mut recon_key);

    let cos = cosine_sim(&key, &recon_key);
    assert!(cos > 0.95, "SP-KV + OCTOPUS roundtrip cosine = {cos}");

    let mut recon_val = vec![0.0f32; 64];
    hybrid.quant.dequantize_value_into(0, 0, &mut recon_val);

    let cos_v = cosine_sim(&val, &recon_val);
    assert!(
        cos_v > 0.95,
        "SP-KV + OCTOPUS value roundtrip cosine = {cos_v}"
    );
}
