//! GOAT Benchmark 189: OscKV — Oscillatory KV Cache with IMEX Discretization.
//!
//! Plan 189 Phase 2: compares OscKV vs SpectralQuant vs Raw (identity f32) backends.
//!
//! Tests:
//! G1: Reconstruction quality on cyclic sequences (cosine > 0.85)
//! G2: Reconstruction quality on non-cyclic sequences (cosine > 0.70)
//! G3: Per-token store latency comparison (< 5000ns for OscKV)
//! G4: Memory usage comparison (bytes per position)
//! G5: Quality vs sequence length scaling
//!
//!
//! Run:
//!   cargo test -p katgpt-rs --features "osc_kv,spectral_quant" --test bench_189_osc_kv_goat -- --nocapture
//!   cargo test -p katgpt-rs --features "osc_kv" --test bench_189_osc_kv_goat -- --nocapture  (no SQ)

#![cfg(feature = "osc_kv")]

use katgpt_core::Rng;
use katgpt_rs::osc_kv::{OscKVCache, OscKVConfig};
#[cfg(feature = "spectral_quant")]
use katgpt_rs::spectralquant::{SpectralQuantKVCache, SpectralQuantKVCacheConfig};
use std::time::Instant;

// ── Helpers ──

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn gaussian_vec(dim: usize, rng: &mut Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.normal()).collect()
}

/// Cyclic key: sine wave at given frequency, phase-shifted per position.
fn cyclic_key(kv_dim: usize, pos: usize, freq: f32) -> Vec<f32> {
    (0..kv_dim)
        .map(|i| (i as f32 * freq + pos as f32 * 0.1).sin())
        .collect()
}

fn osc_config() -> OscKVConfig {
    OscKVConfig {
        n_layers: 1,
        kv_dim: 64,
        max_seq_len: 1024,
        dt: 0.01,
        beta_default: 0.1,
    }
}

#[cfg(feature = "spectral_quant")]
fn sq_config() -> SpectralQuantKVCacheConfig {
    SpectralQuantKVCacheConfig {
        seed: 42,
        n_layers: 1,
        kv_dim: 64,
        max_seq_len: 1024,
        lloyd_max_iter: 30,
        calibration_samples: 100,
        qjl_dim: 16,
        avg_bits: 3.0,
        min_tail_bits: 1,
        max_bits: 8,
        wf_min_bits: 1,
        wf_max_bits: 6,
        use_water_fill: false,
    }
}

struct QualityStats {
    mean: f64,
    min: f64,
    max: f64,
}

impl QualityStats {
    fn from_values(values: &[f64]) -> Self {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Self { mean, min, max }
    }
}

// ── G1: Cyclic reconstruction quality ──

#[test]
fn g1_cyclic_reconstruction_quality() {
    let config = osc_config();
    let (n_pos, freq) = (256, 0.3f32);
    println!("\n🧪 GOAT 189 G1: Cyclic Reconstruction Quality");
    println!("{}", "═".repeat(72));
    println!(
        "Config: kv_dim={}, dt={}, beta={}, freq={freq}",
        config.kv_dim, config.dt, config.beta_default
    );
    println!("Positions: {n_pos}\n");

    let keys: Vec<Vec<f32>> = (0..n_pos)
        .map(|p| cyclic_key(config.kv_dim, p, freq))
        .collect();

    // ── OscKV ──
    let mut osc = OscKVCache::with_config(&config);
    for (pos, key) in keys.iter().enumerate() {
        osc.store_key(0, pos, key);
    }
    let mut osc_cosines: Vec<f64> = Vec::with_capacity(n_pos);
    let mut buf = vec![0.0f32; config.kv_dim];
    for (pos, key) in keys.iter().enumerate() {
        osc.dequantize_key_into(0, pos, &mut buf);
        osc_cosines.push(cosine_sim(key, &buf) as f64);
    }
    let osc_st = QualityStats::from_values(&osc_cosines);

    // ── SpectralQuant ──
    #[cfg(feature = "spectral_quant")]
    let sq_st = {
        let vals: Vec<Vec<f32>> = (0..n_pos)
            .map(|p| cyclic_key(config.kv_dim, p, freq * 0.7))
            .collect();
        let mut sq = SpectralQuantKVCache::from_keys(&sq_config(), &keys, &vals);
        for (pos, key) in keys.iter().enumerate() {
            sq.store_key(0, pos, key);
        }
        let mut cosines: Vec<f64> = Vec::with_capacity(n_pos);
        let mut b = vec![0.0f32; config.kv_dim];
        for (pos, key) in keys.iter().enumerate() {
            sq.dequantize_key_into(0, pos, &mut b);
            cosines.push(cosine_sim(key, &b) as f64);
        }
        Some(QualityStats::from_values(&cosines))
    };

    // Raw (identity)
    let raw_st = QualityStats::from_values(&vec![1.0f64; n_pos]);

    println!(
        "{:<15} │ {:>10} {:>10} {:>10}",
        "Backend", "Mean Cos", "Min Cos", "Max Cos"
    );
    println!("{}", "─".repeat(52));
    println!(
        "{:<15} │ {:>10.6} {:>10.6} {:>10.6}",
        "Raw (identity)", raw_st.mean, raw_st.min, raw_st.max
    );
    println!(
        "{:<15} │ {:>10.6} {:>10.6} {:>10.6}",
        "OscKV", osc_st.mean, osc_st.min, osc_st.max
    );
    #[cfg(feature = "spectral_quant")]
    if let Some(ref sq) = sq_st {
        println!(
            "{:<15} │ {:>10.6} {:>10.6} {:>10.6}",
            "SpectralQuant", sq.mean, sq.min, sq.max
        );
    }
    #[cfg(not(feature = "spectral_quant"))]
    println!(
        "{:<15} │ {:>10} {:>10} {:>10}",
        "SpectralQuant", "—", "—", "—"
    );
    println!();

    assert!(
        osc_st.mean > 0.85,
        "OscKV cyclic cosine = {:.6}, expected > 0.85",
        osc_st.mean
    );
    println!(
        "✅ G1 PASS: OscKV mean cosine on cyclic = {:.4}",
        osc_st.mean
    );
}

// ── G2: Non-cyclic (Gaussian random) reconstruction quality ──

#[test]
fn g2_random_reconstruction_quality() {
    let config = osc_config();
    let n_pos = 256;
    println!("\n🧪 GOAT 189 G2: Random (Gaussian) Reconstruction Quality");
    println!("{}", "═".repeat(72));
    println!("Config: kv_dim={}, n_pos={n_pos}\n", config.kv_dim);

    let mut rng = Rng::new(12345);
    let keys: Vec<Vec<f32>> = (0..n_pos)
        .map(|_| gaussian_vec(config.kv_dim, &mut rng))
        .collect();

    // ── OscKV ──
    let mut osc = OscKVCache::with_config(&config);
    for (pos, key) in keys.iter().enumerate() {
        osc.store_key(0, pos, key);
    }
    let mut osc_cosines: Vec<f64> = Vec::with_capacity(n_pos);
    let mut buf = vec![0.0f32; config.kv_dim];
    for (pos, key) in keys.iter().enumerate() {
        osc.dequantize_key_into(0, pos, &mut buf);
        osc_cosines.push(cosine_sim(key, &buf) as f64);
    }
    let osc_st = QualityStats::from_values(&osc_cosines);

    // ── SpectralQuant ──
    #[cfg(feature = "spectral_quant")]
    let sq_st = {
        let vals: Vec<Vec<f32>> = (0..n_pos)
            .map(|_| gaussian_vec(config.kv_dim, &mut rng))
            .collect();
        let mut sq = SpectralQuantKVCache::from_keys(&sq_config(), &keys, &vals);
        for (pos, key) in keys.iter().enumerate() {
            sq.store_key(0, pos, key);
        }
        let mut cosines: Vec<f64> = Vec::with_capacity(n_pos);
        let mut b = vec![0.0f32; config.kv_dim];
        for (pos, key) in keys.iter().enumerate() {
            sq.dequantize_key_into(0, pos, &mut b);
            cosines.push(cosine_sim(key, &b) as f64);
        }
        Some(QualityStats::from_values(&cosines))
    };

    println!(
        "{:<15} │ {:>10} {:>10} {:>10}",
        "Backend", "Mean Cos", "Min Cos", "Max Cos"
    );
    println!("{}", "─".repeat(52));
    println!(
        "{:<15} │ {:>10.6} {:>10.6} {:>10.6}",
        "OscKV", osc_st.mean, osc_st.min, osc_st.max
    );
    #[cfg(feature = "spectral_quant")]
    if let Some(ref sq) = sq_st {
        println!(
            "{:<15} │ {:>10.6} {:>10.6} {:>10.6}",
            "SpectralQuant", sq.mean, sq.min, sq.max
        );
    }
    #[cfg(not(feature = "spectral_quant"))]
    println!(
        "{:<15} │ {:>10} {:>10} {:>10}",
        "SpectralQuant", "—", "—", "—"
    );
    println!();

    assert!(
        osc_st.mean > 0.70,
        "OscKV random cosine = {:.6}, expected > 0.70",
        osc_st.mean
    );
    println!(
        "✅ G2 PASS: OscKV mean cosine on random = {:.4}",
        osc_st.mean
    );
}

// ── G3: Per-token store latency ──

#[test]
fn g3_store_latency_comparison() {
    let (n_pos, n_warmup) = (1000usize, 50usize);
    let config = OscKVConfig {
        n_layers: 1,
        kv_dim: 64,
        max_seq_len: n_pos + n_warmup + 10,
        dt: 0.01,
        beta_default: 0.1,
    };
    println!("\n🧪 GOAT 189 G3: Per-Token Store Latency");
    println!("{}", "═".repeat(72));
    println!(
        "Config: kv_dim={}, n_pos={n_pos}, warmup={n_warmup}\n",
        config.kv_dim
    );

    let mut rng = Rng::new(9999);
    let keys: Vec<Vec<f32>> = (0..n_pos + n_warmup)
        .map(|_| gaussian_vec(config.kv_dim, &mut rng))
        .collect();

    // ── OscKV latency ──
    let mut osc = OscKVCache::with_config(&config);
    for (pos, key) in keys.iter().take(n_warmup).enumerate() {
        osc.store_key(0, pos, key);
    }
    let mut osc_buf = vec![0.0f32; config.kv_dim];
    let t0 = Instant::now();
    for (pos, key) in keys.iter().enumerate().skip(n_warmup) {
        osc.store_key(0, pos, key);
        osc.dequantize_key_into(0, pos, &mut osc_buf);
    }
    let osc_elapsed = t0.elapsed();
    let osc_ns = osc_elapsed.as_nanos() as f64 / n_pos as f64;

    // ── SpectralQuant latency ──
    #[cfg(feature = "spectral_quant")]
    let sq_ns = {
        let mut sq_cfg = sq_config();
        sq_cfg.max_seq_len = config.max_seq_len;
        let ck = &keys[..100.min(keys.len())];
        let cv: Vec<Vec<f32>> = (0..ck.len())
            .map(|_| gaussian_vec(config.kv_dim, &mut rng))
            .collect();
        let mut sq = SpectralQuantKVCache::from_keys(&sq_cfg, ck, &cv);
        for (pos, key) in keys.iter().take(n_warmup).enumerate() {
            sq.store_key(0, pos, key);
        }
        let mut b = vec![0.0f32; config.kv_dim];
        let t0 = Instant::now();
        for (pos, key) in keys.iter().enumerate().skip(n_warmup) {
            sq.store_key(0, pos, key);
            sq.dequantize_key_into(0, pos, &mut b);
        }
        t0.elapsed().as_nanos() as f64 / n_pos as f64
    };

    // ── Raw latency (memcpy baseline) ──
    let mut raw: Vec<Vec<f32>> = vec![vec![0.0f32; config.kv_dim]; n_pos + n_warmup];
    let t0 = Instant::now();
    for pos in n_warmup..(n_warmup + n_pos) {
        raw[pos].copy_from_slice(&keys[pos]);
    }
    let raw_elapsed = t0.elapsed();
    let raw_ns = raw_elapsed.as_nanos() as f64 / n_pos as f64;

    println!(
        "{:<15} │ {:>12} │ {:>10}",
        "Backend", "ns/token", "Total ms"
    );
    println!("{}", "─".repeat(44));
    println!(
        "{:<15} │ {:>12.1} │ {:>10.2}",
        "Raw (memcpy)",
        raw_ns,
        raw_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "{:<15} │ {:>12.1} │ {:>10.2}",
        "OscKV",
        osc_ns,
        osc_elapsed.as_secs_f64() * 1000.0
    );
    #[cfg(feature = "spectral_quant")]
    println!("{:<15} │ {:>12.1} │ {:>10}", "SpectralQuant", sq_ns, "—");
    #[cfg(not(feature = "spectral_quant"))]
    println!("{:<15} │ {:>12} │ {:>10}", "SpectralQuant", "—", "—");
    println!();

    assert!(
        osc_ns < 5000.0,
        "OscKV latency = {osc_ns:.1} ns/token, expected < 5000 ns"
    );
    println!("✅ G3 PASS: OscKV store+dequantize = {osc_ns:.1} ns/token");
}

// ── G4: Memory usage comparison ──

#[test]
fn g4_memory_usage_comparison() {
    let (kv_dim, max_seq_len) = (64usize, 1024usize);
    println!("\n🧪 GOAT 189 G4: Memory Usage Per Position");
    println!("{}", "═".repeat(72));
    println!("Config: kv_dim={kv_dim}, max_seq_len={max_seq_len}\n");

    let raw_bytes = kv_dim * 4; // f32 per dim
    let osckv_per_pos = 2 * kv_dim * 4; // y + z
    let osckv_overhead = 2 * kv_dim * 4; // omega_sq + beta per layer
    let osckv_amortized =
        (osckv_per_pos * max_seq_len + osckv_overhead) as f64 / max_seq_len as f64;
    let sq_bits = 3.0f32;
    let sq_per_pos = (sq_bits * kv_dim as f32 / 8.0).ceil() as usize + 4; // indices + norm

    println!(
        "{:<15} │ {:>12} │ {:>12} │ {:>10}",
        "Backend", "Bytes/Pos", "Bytes/Pos*2", "vs Raw"
    );
    println!("{}", "─".repeat(58));
    println!(
        "{:<15} │ {:>12} │ {:>12} │ {:>10}",
        "Raw (f32)",
        raw_bytes,
        raw_bytes * 2,
        "1.00x"
    );
    println!(
        "{:<15} │ {:>12.1} │ {:>12.1} │ {:>10.2}x",
        "OscKV",
        osckv_amortized,
        osckv_amortized * 2.0,
        osckv_amortized / raw_bytes as f64
    );
    println!(
        "{:<15} │ {:>12} │ {:>12} │ {:>10.2}x",
        "SpectralQuant",
        sq_per_pos,
        sq_per_pos * 2,
        sq_per_pos as f64 / raw_bytes as f64
    );
    println!();
    println!("📋 OscKV: y+z (2×kv_dim f32) + per-channel ω²+β overhead");
    println!(
        "📋 SpectralQuant: ~{sq_bits:.0} bits/coord packed + f32 norm = {sq_per_pos} bytes/pos"
    );
    println!("📋 Raw: kv_dim × 4 = {raw_bytes} bytes/pos (*2 for K+V)");
    println!();
    println!("✅ G4 PASS: Memory comparison complete");
}

// ── G5: Quality vs sequence length scaling ──

#[test]
fn g5_quality_vs_sequence_length() {
    let lengths = [32, 64, 128, 256, 512];
    let freq = 0.3f32;
    println!("\n🧪 GOAT 189 G5: Quality vs Sequence Length Scaling");
    println!("{}", "═".repeat(72));
    println!("Cyclic input, freq={freq}\n");
    println!(
        "{:>6} │ {:>10} │ {:>10} │ {:>10}",
        "Len", "Raw Cos", "OscKV Cos", "SQ Cos"
    );
    println!("{}", "─".repeat(46));

    let mut osc_cosines: Vec<f64> = Vec::new();
    for &seq_len in &lengths {
        let config = OscKVConfig {
            n_layers: 1,
            kv_dim: 64,
            max_seq_len: seq_len * 2,
            dt: 0.01,
            beta_default: 0.1,
        };
        let keys: Vec<Vec<f32>> = (0..seq_len)
            .map(|p| cyclic_key(config.kv_dim, p, freq))
            .collect();

        // OscKV
        let mut osc = OscKVCache::with_config(&config);
        for (pos, key) in keys.iter().enumerate() {
            osc.store_key(0, pos, key);
        }
        let mut buf = vec![0.0f32; config.kv_dim];
        let mut cos_sum = 0.0f64;
        for (pos, key) in keys.iter().enumerate() {
            osc.dequantize_key_into(0, pos, &mut buf);
            cos_sum += cosine_sim(key, &buf) as f64;
        }
        let osc_cos = cos_sum / seq_len as f64;
        osc_cosines.push(osc_cos);

        // SpectralQuant
        #[cfg(feature = "spectral_quant")]
        let sq_cos = {
            let sq_cfg = SpectralQuantKVCacheConfig {
                seed: 42,
                n_layers: 1,
                kv_dim: 64,
                max_seq_len: config.max_seq_len,
                lloyd_max_iter: 30,
                calibration_samples: 100,
                qjl_dim: 16,
                avg_bits: 3.0,
                min_tail_bits: 1,
                max_bits: 8,
                wf_min_bits: 1,
                wf_max_bits: 6,
                use_water_fill: false,
            };
            let vals: Vec<Vec<f32>> = (0..seq_len)
                .map(|p| cyclic_key(64, p, freq * 0.7))
                .collect();
            let mut sq = SpectralQuantKVCache::from_keys(&sq_cfg, &keys, &vals);
            for (pos, key) in keys.iter().enumerate() {
                sq.store_key(0, pos, key);
            }
            let mut b = vec![0.0f32; 64];
            let mut s = 0.0f64;
            for (pos, key) in keys.iter().enumerate() {
                sq.dequantize_key_into(0, pos, &mut b);
                s += cosine_sim(key, &b) as f64;
            }
            s / seq_len as f64
        };

        #[cfg(feature = "spectral_quant")]
        println!(
            "{:>6} │ {:>10.6} │ {:>10.6} │ {:>10.6}",
            seq_len, 1.0f64, osc_cos, sq_cos
        );
        #[cfg(not(feature = "spectral_quant"))]
        println!(
            "{:>6} │ {:>10.6} │ {:>10.6} │ {:>10}",
            seq_len, 1.0f64, osc_cos, "—"
        );
    }

    println!();
    let longest = osc_cosines.last().unwrap();
    assert!(
        *longest > 0.5,
        "OscKV collapsed at longest: cosine = {longest:.6}, expected > 0.5"
    );
    if osc_cosines.len() >= 2 {
        let shortest = *osc_cosines.first().unwrap();
        let degradation = shortest - longest;
        println!("📉 OscKV degradation (shortest→longest): {degradation:.4}");
        assert!(degradation < 0.5, "Too much degradation: {degradation:.4}");
    }
    println!("✅ G5 PASS: OscKV quality degrades gracefully across lengths");
}

// ── Summary ──

#[test]
fn goat_189_summary() {
    println!("\n📋 GOAT 189 OscKV Benchmark Summary");
    println!("{}", "═".repeat(72));
    println!("Plan 189 Phase 2: Oscillatory KV Cache with IMEX Discretization\n");
    println!("Backend         │ Strength                       │ Weakness");
    println!("─────────────────┼────────────────────────────────┼──────────────────────────");
    println!("Raw (identity)   │ Perfect reconstruction (cos≈1) │ No compression, O(n) mem");
    println!("OscKV            │ Good on cyclic (cos>0.85)      │ 2× f32 storage vs raw");
    println!("SpectralQuant    │ Best compression (~3 bit/elem) │ Requires calibration data\n");
    println!("GOAT Gate: OscKV → opt-in feature gate `osc_kv`");
    println!(
        "Use OscKV for oscillatory/periodic signals, SpectralQuant for general compression.\n"
    );
}
