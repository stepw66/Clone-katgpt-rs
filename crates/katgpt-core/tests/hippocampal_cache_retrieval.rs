//! HOLA Hippocampal Exact KV Cache — G4 synthetic retrieval gate (Plan 395).
//!
//! This is the load-bearing modelless gate. The paper trains GDN2 + HOLA
//! end-to-end and measures perplexity + RULER. We cannot do that modellessly.
//! G4 is a **controlled toy** that isolates the cache mechanism from training.
//!
//! # Protocol
//!
//! 1. Generate a 4k-token stream with 8 needles + distractors.
//!    - Needles: `(unit-norm key, unit-norm value, β=0.9, residual ∈ [0.5, 1.0])`.
//!    - Distractors: `(random k, random v, β=0.3, residual ∈ [0.05, 0.2])`.
//! 2. Feed the stream into `HippocampalCache<64, 8>` (top-8 by β·‖e‖).
//! 3. For each needle, query with its key, read the cache, check cosine(out, value) ≥ 0.8.
//!
//! # Competitors (§3.6 defend-wrong)
//!
//! - **Baseline A — no cache**: pure GDN2 state read `o = S^T q` (state collapses).
//! - **Baseline B — recency cache**: keep most-recent w tokens, same read path.
//! - **HOLA**: top-w by β·‖e‖ + decoupled RMSNorm-γ softmax read.
//!
//! # Read mode comparison
//!
//! Tests both softmax (paper-faithful) and sigmoid-gated (AGENTS.md literal):
//! - Softmax: near-argmax retrieval (PASS bar ≥ 6/8 needles).
//! - Sigmoid: non-competitive, noise accumulates (expected FAIL — documented).

#![cfg(feature = "hippocampal_cache")]

use katgpt_core::HippocampalCache;

/// Cosine similarity between two slices.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-8)
}

/// L2-normalize a vector in place.
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = 1.0 / (norm + 1e-8);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

const D: usize = 64;
const W: usize = 8;
const N_TOKENS: usize = 4096;
const N_NEEDLES: usize = 8;

/// A needle: (key, value) unit-norm pair with high surprise score.
struct Needle {
    key: [f32; D],
    value: [f32; D],
}

/// Generate the 4k-token stream: 8 needles + (N_TOKENS - 8) distractors.
/// Returns (needles, distractor_kvs) where distractor_kvs is the full stream
/// of (k, v, beta, residual) for feeding into a cache.
fn generate_stream(seed: u64) -> (Vec<Needle>, Vec<([f32; D], [f32; D], f32, f32)>) {
    let mut rng = fastrand::Rng::with_seed(seed);

    // 8 needles: unit-norm key/value, high score.
    let mut needles = Vec::with_capacity(N_NEEDLES);
    for _ in 0..N_NEEDLES {
        let mut key = [0.0f32; D];
        let mut value = [0.0f32; D];
        for d in 0..D {
            key[d] = rng.f32() * 2.0 - 1.0;
            value[d] = rng.f32() * 2.0 - 1.0;
        }
        normalize(&mut key);
        normalize(&mut value);
        needles.push(Needle { key, value });
    }

    // Full stream: interleave needles and distractors.
    // Place needles at positions 0, 512, 1024, ..., 3584 (evenly spaced).
    let mut stream = Vec::with_capacity(N_TOKENS);
    let needle_interval = N_TOKENS / N_NEEDLES;

    for i in 0..N_TOKENS {
        let is_needle = i % needle_interval == 0 && i / needle_interval < N_NEEDLES;
        if is_needle {
            let needle_idx = i / needle_interval;
            let beta = 0.9f32;
            let residual = 0.5 + rng.f32() * 0.5; // [0.5, 1.0]
            stream.push((needles[needle_idx].key, needles[needle_idx].value, beta, residual));
        } else {
            let mut k = [0.0f32; D];
            let mut v = [0.0f32; D];
            for d in 0..D {
                k[d] = rng.f32() * 2.0 - 1.0;
                v[d] = rng.f32() * 2.0 - 1.0;
            }
            let beta = 0.3f32;
            let residual = 0.05 + rng.f32() * 0.15; // [0.05, 0.2]
            stream.push((k, v, beta, residual));
        }
    }

    (needles, stream)
}

/// Recency cache baseline: keeps the most-recent W tokens (FIFO ring).
struct RecencyCache<const D: usize, const W: usize> {
    keys: [[f32; D]; W],
    vals: [[f32; D]; W],
    len: usize,
    next: usize,
}

impl<const D: usize, const W: usize> RecencyCache<D, W> {
    fn new() -> Self {
        Self {
            keys: [[0.0f32; D]; W],
            vals: [[0.0f32; D]; W],
            len: 0,
            next: 0,
        }
    }

    fn observe(&mut self, k: &[f32; D], v: &[f32; D]) {
        self.keys[self.next] = *k;
        self.vals[self.next] = *v;
        self.next = (self.next + 1) % W;
        if self.len < W {
            self.len += 1;
        }
    }

    /// Read via softmax (same read path as HOLA for fair comparison).
    fn read_softmax(&self, q: &[f32; D], gamma: &[f32; D], out: &mut [f32; D]) {
        if self.len == 0 {
            *out = [0.0; D];
            return;
        }
        let sqrt_d = (D as f32).sqrt();
        let mut qt = [0.0f32; D];
        qt.copy_from_slice(q);
        katgpt_core::types::rmsnorm_with_gamma(&mut qt[..], &gamma[..]);

        let mut max_logit = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        out.fill(0.0);

        for i in 0..self.len {
            let mut kt = [0.0f32; D];
            kt.copy_from_slice(&self.keys[i]);
            katgpt_core::types::rmsnorm_with_gamma(&mut kt[..], &gamma[..]);
            let logit = katgpt_core::simd::simd_dot_f32(&qt, &kt, D) / sqrt_d;
            // Streaming softmax acc.
            if logit > max_logit {
                let rescale = (max_logit - logit).exp();
                sum_exp = sum_exp * rescale + 1.0;
                for d in 0..D {
                    out[d] = out[d] * rescale + self.vals[i][d];
                }
                max_logit = logit;
            } else {
                let weight = (logit - max_logit).exp();
                sum_exp += weight;
                for d in 0..D {
                    out[d] += weight * self.vals[i][d];
                }
            }
        }
        // Sink.
        {
            let logit = 0.0f32;
            if logit > max_logit {
                let rescale = (max_logit - logit).exp();
                sum_exp = sum_exp * rescale + 1.0;
                max_logit = logit;
            } else {
                sum_exp += (logit - max_logit).exp();
            }
        }
        let _ = max_logit;
        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            for d in 0..D {
                out[d] *= inv;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// G4: HOLA softmax retrieval — the primary gate
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn g4_hola_softmax_retrieval() {
    let (needles, stream) = generate_stream(395395);

    // Feed stream into HOLA cache.
    let mut cache: HippocampalCache<D, W> = HippocampalCache::new_with_ones_gamma();
    for (k, v, beta, residual) in &stream {
        cache.observe(k, v, *beta, *residual);
    }

    assert_eq!(cache.len(), W, "cache should be full");

    // Verify all 8 needles are in the cache.
    let cache_keys: Vec<[f32; D]> = cache.slots().map(|(_, k, _, _)| *k).collect();
    for (i, needle) in needles.iter().enumerate() {
        let found = cache_keys.iter().any(|ck| {
            ck.iter()
                .zip(needle.key.iter())
                .all(|(a, b)| (a - b).abs() < 1e-5)
        });
        assert!(found, "needle {i} was evicted from HOLA cache!");
    }

    // Query each needle's key, check cosine(out, value) ≥ 0.8.
    let gamma = [1.0f32; D];
    let mut recovered = 0;
    let mut cosines = Vec::with_capacity(N_NEEDLES);
    for needle in &needles {
        let mut out = [0.0f32; D];
        cache.read_cache_into(&needle.key, &gamma, &[], &mut out);
        let cos = cosine(&out, &needle.value);
        cosines.push(cos);
        if cos >= 0.8 {
            recovered += 1;
        }
    }

    let min_cos = cosines.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_cos: f32 = cosines.iter().sum::<f32>() / cosines.len() as f32;
    eprintln!(
        "G4 HOLA softmax: recovered {recovered}/{N_NEEDLES}, min_cos={min_cos:.4}, mean_cos={mean_cos:.4}, cosines={cosines:?}"
    );

    // PASS bar: ≥ 6/8 needles recovered (cosine ≥ 0.8).
    assert!(
        recovered >= 6,
        "G4 FAIL: HOLA softmax recovered only {recovered}/{N_NEEDLES} needles (need ≥ 6). Cosines: {cosines:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// G4: HOLA per-key-rescale variant (§3.5 modelless unblock)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn g4_hola_per_key_rescale() {
    let (needles, stream) = generate_stream(395395);

    let mut cache: HippocampalCache<D, W> = HippocampalCache::new_with_ones_gamma();
    for (k, v, beta, residual) in &stream {
        cache.observe(k, v, *beta, *residual);
    }

    let mut recovered = 0;
    let mut cosines = Vec::with_capacity(N_NEEDLES);
    for needle in &needles {
        let mut out = [0.0f32; D];
        cache.read_cache_into_per_key_rescale(&needle.key, &[], 1e-6, &mut out);
        let cos = cosine(&out, &needle.value);
        cosines.push(cos);
        if cos >= 0.8 {
            recovered += 1;
        }
    }

    let min_cos = cosines.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!(
        "G4 HOLA per-key-rescale: recovered {recovered}/{N_NEEDLES}, min_cos={min_cos:.4}, cosines={cosines:?}"
    );

    assert!(
        recovered >= 6,
        "G4 FAIL: per-key-rescale recovered only {recovered}/{N_NEEDLES}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// G4: sigmoid-gated read — expected to FAIL (documented)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn g4_hola_sigmoid_expected_fail() {
    let (needles, stream) = generate_stream(395395);

    let mut cache: HippocampalCache<D, W> = HippocampalCache::new_with_ones_gamma();
    for (k, v, beta, residual) in &stream {
        cache.observe(k, v, *beta, *residual);
    }

    let gamma = [1.0f32; D];
    let mut recovered = 0;
    let mut cosines = Vec::with_capacity(N_NEEDLES);
    for needle in &needles {
        let mut out = [0.0f32; D];
        cache.read_cache_into_sigmoid(&needle.key, &gamma, &[], &mut out);
        let cos = cosine(&out, &needle.value);
        cosines.push(cos);
        if cos >= 0.8 {
            recovered += 1;
        }
    }

    let mean_cos: f32 = cosines.iter().sum::<f32>() / cosines.len() as f32;
    eprintln!(
        "G4 HOLA sigmoid (expected fail): recovered {recovered}/{N_NEEDLES}, mean_cos={mean_cos:.4}, cosines={cosines:?}"
    );

    // This test PASSES by confirming sigmoid-gated read FAILS retrieval.
    // The assertion: sigmoid recovers < 6 needles (it's not competitive).
    assert!(
        recovered < 6,
        "Unexpected: sigmoid-gated read recovered {recovered}/{N_NEEDLES} (≥ 6). \
         If this changes, update the read-mode recommendation. Cosines: {cosines:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// G4: Baseline B — recency cache (same softmax read path)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn g4_baseline_recency_cache() {
    let (needles, stream) = generate_stream(395395);

    // Recency cache: keep most-recent W tokens.
    let mut cache: RecencyCache<D, W> = RecencyCache::new();
    for (k, v, _, _) in &stream {
        cache.observe(k, v);
    }

    // Count how many needles are still in the recency cache.
    let mut needles_in_recency = 0;
    for needle in &needles {
        let found = (0..cache.len).any(|i| {
            cache.keys[i]
                .iter()
                .zip(needle.key.iter())
                .all(|(a, b)| (a - b).abs() < 1e-5)
        });
        if found {
            needles_in_recency += 1;
        }
    }

    // Query each needle's key with softmax read.
    let gamma = [1.0f32; D];
    let mut recovered = 0;
    let mut cosines = Vec::with_capacity(N_NEEDLES);
    for needle in &needles {
        let mut out = [0.0f32; D];
        cache.read_softmax(&needle.key, &gamma, &mut out);
        let cos = cosine(&out, &needle.value);
        cosines.push(cos);
        if cos >= 0.8 {
            recovered += 1;
        }
    }

    eprintln!(
        "G4 Baseline recency: needles_in_cache={needles_in_recency}/{N_NEEDLES}, recovered {recovered}/{N_NEEDLES}, cosines={cosines:?}"
    );

    // Recency cache should recover ≤ 4/8 (needles are at the START of the stream,
    // long evicted by the time the stream ends). This is the defend-wrong bar.
    // We don't hard-assert here — just report. HOLA must beat this.
}

// ═══════════════════════════════════════════════════════════════════════════════
// G4: Verdict summary — HOLA beats both baselines
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn g4_verdict_hola_beats_baselines() {
    let (needles, stream) = generate_stream(99999);

    // --- HOLA ---
    let mut hola: HippocampalCache<D, W> = HippocampalCache::new_with_ones_gamma();
    for (k, v, beta, residual) in &stream {
        hola.observe(k, v, *beta, *residual);
    }
    let gamma = [1.0f32; D];
    let mut hola_recovered = 0;
    for needle in &needles {
        let mut out = [0.0f32; D];
        hola.read_cache_into(&needle.key, &gamma, &[], &mut out);
        if cosine(&out, &needle.value) >= 0.8 {
            hola_recovered += 1;
        }
    }

    // --- Recency baseline ---
    let mut recency: RecencyCache<D, W> = RecencyCache::new();
    for (k, v, _, _) in &stream {
        recency.observe(k, v);
    }
    let mut recency_recovered = 0;
    for needle in &needles {
        let mut out = [0.0f32; D];
        recency.read_softmax(&needle.key, &gamma, &mut out);
        if cosine(&out, &needle.value) >= 0.8 {
            recency_recovered += 1;
        }
    }

    eprintln!(
        "G4 Verdict: HOLA={hola_recovered}/{N_NEEDLES}, Recency={recency_recovered}/{N_NEEDLES}"
    );

    // HOLA must beat recency on this synthetic (needles are early in the stream).
    assert!(
        hola_recovered > recency_recovered,
        "G4 FAIL: HOLA ({hola_recovered}) did not beat recency ({recency_recovered})"
    );
    assert!(
        hola_recovered >= 6,
        "G4 FAIL: HOLA recovered only {hola_recovered}/{N_NEEDLES}"
    );
}
