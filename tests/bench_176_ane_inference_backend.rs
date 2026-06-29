//! Bench 176: ANE Inference Backend Performance (Plan 176)
//!
//! Benchmarks:
//! - Single-token decode latency: CpuBackend
//! - Full generation (50 tokens): CpuBackend
//! - Backend selection overhead
//!
//! Run with:
//!   cargo test --test bench_176_ane_inference_backend --release -- --nocapture

use std::time::Instant;

use katgpt_rs::inference_backend::{BackendKind, CpuBackend, InferenceBackend, auto_backend};
use katgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use katgpt_rs::types::{Config, Rng, sample_token_into, softmax_scaled};

fn setup() -> (Config, TransformerWeights) {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    (config, weights)
}

#[test]
fn bench_176_single_token_decode_cpu() {
    let (config, weights) = setup();
    let mut backend = CpuBackend::new();
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    // Warmup
    let _ = backend.forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

    // Benchmark
    let n_iters = 1000;
    let start = Instant::now();
    for pos in 0..n_iters {
        if pos >= config.block_size {
            cache.reset();
        }
        let pos_clamped = pos % config.block_size;
        let _ = backend.forward(&mut ctx, &weights, &mut cache, 0, pos_clamped, &config);
    }
    let elapsed = start.elapsed();
    let us_per_token = elapsed.as_secs_f64() * 1e6 / n_iters as f64;

    println!(
        "Bench 176: Single-token decode (CpuBackend): {:.2} µs/token",
        us_per_token
    );
    assert!(
        us_per_token < 5000.0,
        "single-token decode too slow: {us_per_token} µs"
    );
}

#[test]
fn bench_176_generation_50_tokens_cpu() {
    let (config, weights) = setup();
    let mut backend = CpuBackend::new();
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut rng = Rng::new(42);
    let mut cdf = Vec::with_capacity(config.vocab_size);

    let n_tokens = 50;
    let start = Instant::now();

    let mut token = config.bos_token;
    for pos in 0..n_tokens {
        if pos >= config.block_size {
            cache.reset();
        }
        let pos_clamped = pos % config.block_size;
        let logits = backend.forward(&mut ctx, &weights, &mut cache, token, pos_clamped, &config);
        softmax_scaled(logits, 1.0 / config.temperature);
        let next = sample_token_into(&ctx.logits, &mut rng, &mut cdf);
        token = next;
    }

    let elapsed = start.elapsed();
    let us_total = elapsed.as_secs_f64() * 1e6;
    let us_per_token = us_total / n_tokens as f64;

    println!(
        "Bench 176: Generation 50 tokens (CpuBackend): {:.2} µs total, {:.2} µs/token",
        us_total, us_per_token
    );
}

#[test]
fn bench_176_backend_selection_overhead() {
    let start = Instant::now();
    for _ in 0..100 {
        let _backend = auto_backend(BackendKind::Cpu, None);
    }
    let elapsed = start.elapsed();
    let us_per_selection = elapsed.as_secs_f64() * 1e6 / 100.0;

    println!(
        "Bench 176: Backend selection overhead: {:.2} µs/selection",
        us_per_selection
    );
    assert!(
        us_per_selection < 100.0,
        "backend selection too slow: {us_per_selection} µs"
    );
}
