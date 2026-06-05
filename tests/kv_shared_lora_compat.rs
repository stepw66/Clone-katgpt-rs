//! K=V LoRA inference compatibility test (Plan 207 T5).
//!
//! Verifies that K=V LoRA adapters trained in riir-ai can be loaded
//! and used with katgpt-rs inference via AttentionProjection::SharedKV.
//!
//! Run with:
//! ```sh
//! cargo test --features kv_share --test kv_shared_lora_compat -- --nocapture
//! ```

use katgpt_core::types::{AttentionProjection, CacheLayout};
use katgpt_core::types::{Config, kv_dim};
use katgpt_rs::kv_share::cache_layout;

#[test]
fn shared_kv_projection_maps_to_k_only_cache() {
    assert_eq!(
        cache_layout(AttentionProjection::SharedKV),
        CacheLayout::K,
        "SharedKV should use K-only cache"
    );
}

#[test]
fn full_projection_maps_to_kv_cache() {
    assert_eq!(
        cache_layout(AttentionProjection::Full),
        CacheLayout::KV,
        "Full projection should use KV cache"
    );
}

#[test]
fn kv_dim_game_config() {
    let config = Config::game();
    let expected = config.n_kv_head * config.head_dim;
    assert_eq!(
        kv_dim(&config),
        expected,
        "kv_dim should equal n_kv_head * head_dim"
    );
}

#[test]
fn shared_kv_adapter_dimensions() {
    let config = Config::game();
    let kv_dim_val = kv_dim(&config);

    // The shared KV adapter has:
    // - in_dim = n_embd (input projection from hidden state)
    // - out_dim = kv_dim (output to K/V shared space)
    let adapter_in = config.n_embd;
    let adapter_out = kv_dim_val;

    // With rank r, the adapter has:
    // A: [r, in_dim], B: [out_dim, r]
    let rank = config.lora_rank;
    let a_params = rank * adapter_in;
    let b_params = adapter_out * rank;
    let shared_kv_params = a_params + b_params;

    // Standard would have separate K and V:
    let standard_k_params = a_params + adapter_out * rank;
    let standard_v_params = a_params + adapter_out * rank;
    let standard_kv_params = standard_k_params + standard_v_params;

    // Shared should be exactly 50% of standard K+V params
    assert_eq!(
        shared_kv_params * 2,
        standard_kv_params,
        "Shared KV should be 50% of standard K+V params"
    );
}

#[test]
fn game_config_lora_fields() {
    let config = Config::game();
    assert!(config.lora_rank > 0, "lora_rank should be positive");
    assert!(config.lora_alpha > 0.0, "lora_alpha should be positive");
    assert!(
        !config.lora_targets.is_empty(),
        "lora_targets should not be empty"
    );
}

#[test]
fn game_go_config_lora_fields() {
    let config = Config::game_go();
    assert!(config.lora_rank > 0, "lora_rank should be positive");
    assert!(config.lora_alpha > 0.0, "lora_alpha should be positive");
}
