#![cfg(feature = "kog_cpu_fusion")]
//! Bench 160: Kog CPU Monokernel Fusion — Forward Pass Throughput
//!
//! Benchmarks transformer forward pass with and without kog_cpu_fusion optimizations:
//!   1. RMSNorm gamma folding — fold MLP gamma into weight matrices at init
//!   2. QKV weight interleaving — repack separate Q/K/V into one contiguous buffer
//!
//! Run (release for meaningful numbers):
//!   cargo test --features kog_cpu_fusion --test bench_160_kog_cpu_fusion --release -- --nocapture

use std::time::Instant;

use katgpt_rs::mbu;
use katgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use katgpt_rs::types::{Config, Rng};

#[test]
fn bench_kog_cpu_fusion() {
    let config = Config::micro();

    // ── Create baseline weights (seed 42) ──────────────────────────
    let mut rng1 = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng1);
    // Note: gamma is identity (1.0) by default. Gamma folding is GOAT-proven
    // separately in proof_gamma_folding_forward_base() using manual forward.
    // This benchmark focuses on the QKV interleave cache locality effect.

    // ── Create optimized weights (same seed, then apply optimizations) ─
    let mut rng2 = Rng::new(42);
    let mut weights_opt = TransformerWeights::new(&config, &mut rng2);

    // Apply kog_cpu_fusion optimizations
    weights_opt.fold_gamma(&config); // no-op with identity gamma, but tests the code path
    weights_opt.interleave_qkv(&config);

    // ── Pre-allocate contexts and caches ───────────────────────────
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx_opt = ForwardContext::new(&config);
    let mut cache_opt = MultiLayerKVCache::new(&config);

    let n_warmup = 100;
    let n_measure = 10_000;
    let tokens_per_iter = 8;

    // ── Baseline warmup ────────────────────────────────────────────
    for _ in 0..n_warmup {
        cache.reset();
        for pos in 0..tokens_per_iter {
            forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
        }
    }

    // ── Baseline measurement ───────────────────────────────────────
    let t_base_start = Instant::now();
    for _ in 0..n_measure {
        cache.reset();
        for pos in 0..tokens_per_iter {
            forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
        }
    }
    let t_base_elapsed = t_base_start.elapsed();

    // ── Optimized warmup ───────────────────────────────────────────
    for _ in 0..n_warmup {
        cache_opt.reset();
        for pos in 0..tokens_per_iter {
            forward(&mut ctx_opt, &weights_opt, &mut cache_opt, 0, pos, &config);
        }
    }

    // ── Optimized measurement ──────────────────────────────────────
    let t_opt_start = Instant::now();
    for _ in 0..n_measure {
        cache_opt.reset();
        for pos in 0..tokens_per_iter {
            forward(&mut ctx_opt, &weights_opt, &mut cache_opt, 0, pos, &config);
        }
    }
    let t_opt_elapsed = t_opt_start.elapsed();

    // ── Compute metrics ────────────────────────────────────────────
    let total_tokens = (n_measure * tokens_per_iter) as f64;
    let base_tok_per_s = total_tokens / t_base_elapsed.as_secs_f64();
    let opt_tok_per_s = total_tokens / t_opt_elapsed.as_secs_f64();
    let base_us_per_tok = t_base_elapsed.as_micros() as f64 / total_tokens;
    let opt_us_per_tok = t_opt_elapsed.as_micros() as f64 / total_tokens;

    let bytes_per_token = mbu::per_token_weight_bytes(&config) as f64;
    let base_bw_gbps = bytes_per_token * base_tok_per_s / 1e9;
    let opt_bw_gbps = bytes_per_token * opt_tok_per_s / 1e9;
    let peak = mbu::peak_bandwidth_gbps();
    let base_mbu = base_bw_gbps / peak * 100.0;
    let opt_mbu = opt_bw_gbps / peak * 100.0;

    // ── Print results ──────────────────────────────────────────────
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Bench 160: Kog CPU Monokernel Fusion — Forward Pass Throughput       ║");
    eprintln!("║  Config: micro (n_embd=16, n_layer=1, head_dim=4)                     ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════════════╣");
    eprintln!("║ Path          │ tok/s     │ µs/tok   │ MBU%   │ Notes                 ║");
    eprintln!("╟───────────────╫───────────╫──────────╫────────╫───────────────────────╢");
    eprintln!(
        "║ {:<13} │ {:>7.0}   │ {:>6.1}   │ {:>5.1}  │ Separate Q/K/V, gamma  ║",
        "Baseline", base_tok_per_s, base_us_per_tok, base_mbu
    );
    eprintln!(
        "║ {:<13} │ {:>7.0}   │ {:>6.1}   │ {:>5.1}  │ Fused QKV, folded MLP  ║",
        "Optimized", opt_tok_per_s, opt_us_per_tok, opt_mbu
    );
    eprintln!("╚══════════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── Correctness assertion ───────────────────────────────────────
    let mut ctx_base = ForwardContext::new(&config);
    let mut cache_base = MultiLayerKVCache::new(&config);
    let mut ctx_vopt = ForwardContext::new(&config);
    let mut cache_vopt = MultiLayerKVCache::new(&config);

    let logits_base = forward(&mut ctx_base, &weights, &mut cache_base, 0, 0, &config);
    let logits_opt = forward(&mut ctx_vopt, &weights_opt, &mut cache_vopt, 0, 0, &config);

    let max_diff = logits_base
        .iter()
        .zip(logits_opt.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-5,
        "Optimized path diverged from baseline: max |Δ| = {max_diff:.e}"
    );
    eprintln!("  ✓ Correctness: max |Δlogits| = {max_diff:.e} (< 1e-5)");
}
