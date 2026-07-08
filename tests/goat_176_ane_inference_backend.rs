//! GOAT Proof: ANE Inference Backend (Plan 176)
//!
//! Tests that:
//! - InferenceBackend trait works correctly with CpuBackend
//! - auto_backend() selects correctly
//! - BackendKind enum works
//! - CPU backend produces same results as direct transformer::forward

use katgpt_rs::inference_backend::{BackendKind, CpuBackend, InferenceBackend, auto_backend};
use katgpt_rs::transformer::{self, ForwardContext, MultiLayerKVCache, TransformerWeights};
use katgpt_rs::types::{Config, Rng, sample_token_into, softmax_scaled};

fn setup() -> (Config, TransformerWeights) {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    (config, weights)
}

// P1: CpuBackend produces identical logits to direct forward()
#[test]
fn goat_p1_cpu_backend_matches_direct_forward() {
    let (config, weights) = setup();

    // Direct forward
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let direct_logits = transformer::forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config);
    let direct: Vec<f32> = direct_logits.to_vec();

    // Backend forward
    let mut backend = CpuBackend::new();
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let backend_logits = backend.forward(&mut ctx2, &weights, &mut cache2, 0, 0, &config);
    let backend_result: Vec<f32> = backend_logits.to_vec();

    assert_eq!(direct.len(), backend_result.len(), "logits length mismatch");
    for (i, (a, b)) in direct.iter().zip(backend_result.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "logits mismatch at index {i}: {a} vs {b}"
        );
    }
}

// P2: auto_backend with Cpu forced always returns CPU
#[test]
fn goat_p2_auto_backend_cpu_forced() {
    let backend = auto_backend(BackendKind::Cpu, None);
    assert_eq!(backend.device_name(), "CPU");
}

// P3: auto_backend with Auto selects best available backend
// (ANE when macOS + `ane` feature is active, else falls back to CPU).
// The unconditional `== "CPU"` assertion was a pre-existing test bug (Issue 413
// follow-up): default features pull `ane` in transitively via
// `async_qdq_overlap` → `inference_router` → `ane`, so on macOS the default
// build selects ANE, not CPU.
#[test]
fn goat_p3_auto_backend_auto_fallback() {
    let backend = auto_backend(BackendKind::Auto, None);
    #[cfg(all(target_os = "macos", feature = "ane"))]
    {
        // ANE feature is active on macOS — auto selects ANE.
        assert_eq!(backend.device_name(), "ANE");
    }
    #[cfg(not(all(target_os = "macos", feature = "ane")))]
    {
        // No ANE available — auto falls back to CPU.
        assert_eq!(backend.device_name(), "CPU");
    }
}

// P4: BackendKind default is Auto
#[test]
fn goat_p4_backend_kind_default_is_auto() {
    assert_eq!(BackendKind::default(), BackendKind::Auto);
}

// P5: CpuBackend determinism — same input produces same output
#[test]
fn goat_p5_cpu_backend_deterministic() {
    let (config, weights) = setup();

    let mut backend = CpuBackend::new();

    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = backend
        .forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config)
        .to_vec();

    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let logits2 = backend
        .forward(&mut ctx2, &weights, &mut cache2, 0, 0, &config)
        .to_vec();

    assert_eq!(logits1, logits2);
}

// P6: Multi-token generation through backend produces valid tokens
#[test]
fn goat_p6_cpu_backend_generation_valid_tokens() {
    let (config, weights) = setup();
    let mut backend = CpuBackend::new();
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut rng = Rng::new(42);
    let mut cdf = Vec::with_capacity(config.vocab_size);

    let mut token = config.bos_token;
    let mut generated = Vec::new();

    for pos in 0..10 {
        let logits = backend.forward(&mut ctx, &weights, &mut cache, token, pos, &config);
        softmax_scaled(logits, 1.0 / config.temperature);
        let next = sample_token_into(&ctx.logits, &mut rng, &mut cdf);
        assert!(
            next < config.vocab_size,
            "token {next} >= vocab_size {}",
            config.vocab_size
        );
        generated.push(next);
        token = next;
    }

    assert_eq!(generated.len(), 10);
}

// P7: CpuBackend supports_stateful is false
#[test]
fn goat_p7_cpu_backend_not_stateful() {
    let backend = CpuBackend::new();
    assert!(!backend.supports_stateful());
}

// P8: Cosine similarity between CPU forward at different positions (sanity check)
#[test]
fn goat_p8_logits_cosine_similarity_sanity() {
    let (config, weights) = setup();
    let mut backend = CpuBackend::new();

    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = backend
        .forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config)
        .to_vec();

    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let logits2 = backend
        .forward(&mut ctx2, &weights, &mut cache2, 1, 1, &config)
        .to_vec();

    // Different inputs should produce different logits (cosine sim < 1.0)
    let dot: f32 = logits1.iter().zip(logits2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = logits1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = logits2.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos_sim = dot / (norm1 * norm2 + 1e-10);

    assert!(
        cos_sim < 0.9999,
        "different inputs should produce different logits, cos_sim={cos_sim}"
    );
    assert!(cos_sim > -1.0, "cosine similarity should be > -1");
}
