#![cfg(feature = "parallax_attn")]
//! Experiment — Sigmoid Parallax under AdamW optimization (Research 140)
//!
//! Tests the hypothesis: sigmoid's sink-free property keeps the Parallax
//! correction branch active under AdamW, where softmax's correction collapses.
//!
//! Setup:
//!   - Freeze Q, K, V (pretrained backbone)
//!   - Train only W_R (R projection) via AdamW on reconstruction loss
//!   - Compare sigmoid vs softmax attention over 200 steps
//!   - Track: COR (correction-to-output ratio), ||W_R||, loss
//!
//! Run: `cargo test --features parallax_attn --test bench_140_sigmoid_parallax_adamw --release -- --nocapture`

use katgpt_core::{
    ParallaxActivation, ParallaxConfig, ParallaxScratch, compute_rho, parallax_correction,
    tiled_attention_parallax_forward,
};

// ── Config ────────────────────────────────────────────────────

const DIM: usize = 32; // Small dim for fast iteration
const SEQ_LEN: usize = 16; // Short sequence
const STEPS: usize = 200;
const LR: f32 = 1e-3;
const WEIGHT_DECAY: f32 = 0.01; // AdamW decay
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;
const SEED: u64 = 140;

// ── Helpers ───────────────────────────────────────────────────

fn rand_vec(len: usize, rng: &mut fastrand::Rng) -> Vec<f32> {
    (0..len).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute Σ_KV for frozen Q,K,V with given activation.
/// Extracted from tiled_attention_parallax_forward Phase 1-2.
#[allow(clippy::too_many_arguments)] // test helper: fixed Σ_KV I/O shape
fn compute_sigma_kv(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    activation: ParallaxActivation,
    scores_buf: &mut [f32],
    col_sums: &mut [f32],
    sigma_kv: &mut [f32],
) {
    use katgpt_core::simd;
    let d = head_dim;
    let n = seq_len;
    col_sums[..n].fill(0.0);
    sigma_kv[..d * d].fill(0.0);

    for i in 0..n {
        let q_off = i * d;
        // j needed for stride k_off = j*d
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            let k_off = j * d;
            scores_buf[j] =
                simd::simd_dot_f32(&q[q_off..q_off + d], &k[k_off..k_off + d], d) * scale;
        }
        // Normalize
        let row = &mut scores_buf[..n];
        normalize_weights(row, activation);
        // Accumulate column sums
        simd::simd_add_inplace(&mut col_sums[..n], &row[..n]);
    }

    // Phase 2: Σ_KV = Σ_j c_j · v_j ⊗ k_j^T
    let mut pv_buf = vec![0.0f32; d];
    // j needed for stride v_off/k_off = j*d
    #[allow(clippy::needless_range_loop)]
    for j in 0..n {
        let c_j = col_sums[j];
        if c_j == 0.0 {
            continue;
        }
        let v_off = j * d;
        let k_off = j * d;
        simd::simd_fused_decay_write(&mut pv_buf, 0.0, &v[v_off..v_off + d], c_j);
        simd::simd_outer_product_acc(sigma_kv, &pv_buf, &k[k_off..k_off + d], d, d);
    }
}

/// Normalize attention weights (replicates parallax_attn logic without owning the module).
fn normalize_weights(row: &mut [f32], activation: ParallaxActivation) {
    use katgpt_core::simd;
    match activation {
        ParallaxActivation::Softmax => {
            let max_score = simd::simd_max_f32(row);
            simd::simd_add_scalar_inplace(row, -max_score);
            simd::simd_exp_inplace(row);
            let rowsum = simd::simd_sum_f32(row);
            simd::simd_scale_inplace(row, 1.0 / rowsum);
        }
        ParallaxActivation::Sigmoid => {
            simd::simd_scale_inplace(row, -1.0);
            simd::simd_exp_inplace(row);
            simd::simd_add_scalar_inplace(row, 1.0);
            simd::simd_reciprocal_inplace(row);
            let rowsum = simd::simd_sum_f32(row);
            simd::simd_scale_inplace(row, 1.0 / rowsum);
        }
    }
}

// ── AdamW State ───────────────────────────────────────────────

struct AdamWState {
    m: Vec<f32>, // first moment
    v: Vec<f32>, // second moment
    t: usize,    // step count
}

impl AdamWState {
    fn new(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
            t: 0,
        }
    }

    /// AdamW update: w -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)
    fn step(&mut self, w: &mut [f32], grad: &[f32], lr: f32, wd: f32) {
        self.t += 1;
        let bc1 = 1.0 - BETA1.powi(self.t as i32);
        let bc2 = 1.0 - BETA2.powi(self.t as i32);
        for i in 0..w.len() {
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * grad[i];
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            // AdamW: decoupled weight decay
            w[i] -= lr * (m_hat / (v_hat.sqrt() + EPS) + wd * w[i]);
        }
    }
}

// ── Forward + Loss + Gradient ─────────────────────────────────

/// Forward pass: compute o_PLX and return (output, correction, sigma_kv).
#[allow(clippy::too_many_arguments)] // test helper: fixed parallax forward I/O shape
fn forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    x: &[f32],
    w_r: &[f32],
    gate_scale: f32,
    activation: ParallaxActivation,
    scratch: &mut ParallaxScratch,
) -> (Vec<f32>, Vec<f32>) {
    let n = SEQ_LEN;
    let d = DIM;
    let scale = 1.0 / (d as f32).sqrt();
    let config = ParallaxConfig {
        gate_scale,
        zero_init: false,
        activation,
    };
    let mut output = vec![0.0f32; n * d];
    tiled_attention_parallax_forward(
        q,
        k,
        v,
        &mut output,
        n,
        d,
        scale,
        w_r,
        x,
        &config,
        Some(scratch),
    );
    // correction is in scratch after forward — but the sign is flipped by Phase 4.
    // We need the raw correction magnitude: ||Σ_KV · ρ|| (before sign flip and scaling).
    // Re-compute from scratch.sigma_kv and scratch.rho.
    let mut raw_correction = vec![0.0f32; d];
    parallax_correction(&scratch.sigma_kv, &scratch.rho, &mut raw_correction);

    (output, raw_correction)
}

/// Compute gradient of L = ||o_PLX - target||^2 w.r.t. W_R.
///
/// o_PLX = o_SA - gs · Σ_KV · ρ     where ρ = W_R · x
///
/// ∂L/∂ρ = -gs · Σ_KV^T · residual    where residual = o_PLX - target
/// ∂L/∂W_R[i,j] = (∂L/∂ρ[i]) · x[j]
///
/// So: ∂L/∂W_R = outer(∂L/∂ρ, x)
fn compute_grad_w_r(
    sigma_kv: &[f32],
    _rho: &[f32],
    x: &[f32],
    output: &[f32],
    target: &[f32],
    gate_scale: f32,
    head_dim: usize,
) -> Vec<f32> {
    let d = head_dim;
    let n = output.len() / d;
    let mut grad = vec![0.0f32; d * d];

    // Compute residual per query row, accumulate gradient
    let _ = n; // used below

    // Actually, let's be more precise. The loss is averaged over all query rows:
    // L = (1/N) Σ_i ||o_PLX_i - target_i||^2
    //
    // o_PLX_i = o_SA_i - gs · correction   (correction is the same for all rows!)
    // because Σ_KV and ρ are shared across all query positions.
    //
    // So residual_i = o_PLX_i - target_i, and
    // ∂L/∂correction = (-gs / N) · Σ_i 2 · residual_i = (-2gs/N) · Σ_i residual_i
    //
    // ∂L/∂ρ = Σ_KV^T · ∂L/∂correction = (-2gs/N) · Σ_KV^T · Σ_i residual_i
    //
    // ∂L/∂W_R[a,b] = (∂L/∂ρ[a]) · x[b]

    // Compute mean residual
    let mut mean_residual = vec![0.0f32; d];
    for i in 0..n {
        let off = i * d;
        for j in 0..d {
            mean_residual[j] += output[off + j] - target[off + j];
        }
    }
    let scale = 2.0 / (n as f32);
    for r in mean_residual.iter_mut() {
        *r *= scale;
    }

    // d_L/d_correction = -gs · mean_residual
    let mut dl_dc = vec![0.0f32; d];
    for j in 0..d {
        dl_dc[j] = -gate_scale * mean_residual[j];
    }

    // d_L/d_rho = Σ_KV^T · d_L/d_correction
    // Σ_KV is d×d row-major, so Σ_KV^T[a,b] = Σ_KV[b,a]
    let mut dl_drho = vec![0.0f32; d];
    for a in 0..d {
        let mut sum = 0.0f32;
        for b in 0..d {
            // Σ_KV^T[a,b] = Σ_KV[b,a] = sigma_kv[b * d + a]
            sum += sigma_kv[b * d + a] * dl_dc[b];
        }
        dl_drho[a] = sum;
    }

    // d_L/d_W_R = outer(dl_drho, x)
    for a in 0..d {
        for b in 0..d {
            grad[a * d + b] = dl_drho[a] * x[b];
        }
    }

    grad
}

// ── The Experiment ────────────────────────────────────────────

#[test]
fn experiment_adamw_sigmoid_vs_softmax() {
    eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Experiment 140: Sigmoid Parallax under AdamW                               ║");
    eprintln!("║  Hypothesis: sigmoid maintains COR, softmax collapses                       ║");
    eprintln!(
        "║  dim={DIM}, seq_len={SEQ_LEN}, steps={STEPS}, lr={LR}                              ║"
    );
    eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");
    eprintln!("║ step │ SM loss  │ Sig loss │ SM COR   │ Sig COR  │ SM ‖W_R‖ │ Sig ‖W_R‖ ║");
    eprintln!("╟──────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────────╢");

    let d = DIM;
    let n = SEQ_LEN;
    let scale = 1.0 / (d as f32).sqrt();
    let gate_scale = 1.0f32;

    // Generate frozen backbone
    let mut rng = fastrand::Rng::with_seed(SEED);
    let q = rand_vec(n * d, &mut rng);
    let k = rand_vec(n * d, &mut rng);
    let v = rand_vec(n * d, &mut rng);
    let x = rand_vec(d, &mut rng); // layer input (shared across both)

    // Target: base softmax attention output (simulating a teacher/pretrained target)
    // The correction should try to push towards this target.
    // We'll use a slightly different target to give the correction branch something to learn.
    let target_config = ParallaxConfig {
        gate_scale: 0.0, // base attention, no correction
        zero_init: true,
        activation: ParallaxActivation::Softmax,
    };
    let mut target = vec![0.0f32; n * d];
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut target,
        n,
        d,
        scale,
        &vec![0.0f32; d * d], // zero R
        &x,
        &target_config,
        None,
    );

    // Add small noise to target so there's something for the correction to fit
    let mut target_rng = fastrand::Rng::with_seed(SEED + 999);
    for t in target.iter_mut() {
        *t += (target_rng.f32() - 0.5) * 0.1;
    }

    // Initialize W_R for both runs (same init)
    let w_r_init = rand_vec(d * d, &mut fastrand::Rng::with_seed(SEED + 42));

    // ── Run Softmax Parallax with AdamW ──
    let mut w_r_sm = w_r_init.clone();
    let mut adam_sm = AdamWState::new(d * d);
    let mut scratch_sm = ParallaxScratch::new(n, d);

    // ── Run Sigmoid Parallax with AdamW ──
    let mut w_r_sig = w_r_init.clone();
    let mut adam_sig = AdamWState::new(d * d);
    let mut scratch_sig = ParallaxScratch::new(n, d);

    for step in 0..=STEPS {
        // Forward: softmax
        let (out_sm, corr_sm) = forward(
            &q,
            &k,
            &v,
            &x,
            &w_r_sm,
            gate_scale,
            ParallaxActivation::Softmax,
            &mut scratch_sm,
        );
        // Forward: sigmoid
        let (out_sig, corr_sig) = forward(
            &q,
            &k,
            &v,
            &x,
            &w_r_sig,
            gate_scale,
            ParallaxActivation::Sigmoid,
            &mut scratch_sig,
        );

        // Compute loss (MSE)
        let loss_sm: f32 = out_sm
            .chunks(d)
            .zip(target.chunks(d))
            .map(|(o, t)| {
                o.iter()
                    .zip(t.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / (n as f32);
        let loss_sig: f32 = out_sig
            .chunks(d)
            .zip(target.chunks(d))
            .map(|(o, t)| {
                o.iter()
                    .zip(t.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / (n as f32);

        // COR = ||correction|| / ||output|| (averaged over rows)
        let mut cor_sm = 0.0f32;
        let mut cor_sig = 0.0f32;
        let mut out_norm_sm = 0.0f32;
        let mut out_norm_sig = 0.0f32;
        for i in 0..n {
            let off = i * d;
            let on_sm = vec_norm(&out_sm[off..off + d]);
            let on_sig = vec_norm(&out_sig[off..off + d]);
            out_norm_sm += on_sm;
            out_norm_sig += on_sig;
        }
        let corr_sm_norm = vec_norm(&corr_sm);
        let corr_sig_norm = vec_norm(&corr_sig);
        if out_norm_sm > 1e-12 {
            cor_sm = corr_sm_norm * gate_scale / out_norm_sm;
        }
        if out_norm_sig > 1e-12 {
            cor_sig = corr_sig_norm * gate_scale / out_norm_sig;
        }

        let wr_sm_norm = vec_norm(&w_r_sm);
        let wr_sig_norm = vec_norm(&w_r_sig);

        if step % 20 == 0 || step == STEPS {
            eprintln!(
                "║ {step:>4} │ {loss_sm:>8.4} │ {loss_sig:>8.4} │ {cor_sm:>8.4} │ {cor_sig:>8.4} │ {wr_sm_norm:>8.4} │ {wr_sig_norm:>8.4} ║"
            );
        }

        // Compute gradients and update
        if step < STEPS {
            let grad_sm = compute_grad_w_r(
                &scratch_sm.sigma_kv,
                &scratch_sm.rho,
                &x,
                &out_sm,
                &target,
                gate_scale,
                d,
            );
            let grad_sig = compute_grad_w_r(
                &scratch_sig.sigma_kv,
                &scratch_sig.rho,
                &x,
                &out_sig,
                &target,
                gate_scale,
                d,
            );
            adam_sm.step(&mut w_r_sm, &grad_sm, LR, WEIGHT_DECAY);
            adam_sig.step(&mut w_r_sig, &grad_sig, LR, WEIGHT_DECAY);
        }
    }

    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // ── Verdict ──────────────────────────────────────────────────
    eprintln!("\n── Verdict ──");

    // Final forward for diagnostics
    let (out_sm_final, corr_sm_final) = forward(
        &q,
        &k,
        &v,
        &x,
        &w_r_sm,
        gate_scale,
        ParallaxActivation::Softmax,
        &mut scratch_sm,
    );
    let (out_sig_final, corr_sig_final) = forward(
        &q,
        &k,
        &v,
        &x,
        &w_r_sig,
        gate_scale,
        ParallaxActivation::Sigmoid,
        &mut scratch_sig,
    );

    let loss_sm_final: f32 = out_sm_final
        .chunks(d)
        .zip(target.chunks(d))
        .map(|(o, t)| {
            o.iter()
                .zip(t.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
        })
        .sum::<f32>()
        / (n as f32);
    let loss_sig_final: f32 = out_sig_final
        .chunks(d)
        .zip(target.chunks(d))
        .map(|(o, t)| {
            o.iter()
                .zip(t.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
        })
        .sum::<f32>()
        / (n as f32);

    let wr_sm_final = vec_norm(&w_r_sm);
    let wr_sig_final = vec_norm(&w_r_sig);
    let corr_sm_final_norm = vec_norm(&corr_sm_final);
    let corr_sig_final_norm = vec_norm(&corr_sig_final);

    eprintln!(
        "  Softmax:  final_loss={loss_sm_final:.6}, ‖W_R‖={wr_sm_final:.4}, ‖correction‖={corr_sm_final_norm:.4}"
    );
    eprintln!(
        "  Sigmoid:  final_loss={loss_sig_final:.6}, ‖W_R‖={wr_sig_final:.4}, ‖correction‖={corr_sig_final_norm:.4}"
    );

    // Key metric: did sigmoid maintain a larger correction than softmax?
    let correction_ratio = if corr_sm_final_norm > 1e-12 {
        corr_sig_final_norm / corr_sm_final_norm
    } else if corr_sig_final_norm > 1e-12 {
        f32::INFINITY
    } else {
        1.0
    };
    eprintln!("  Correction ratio (sig/sm): {correction_ratio:.4}");

    if correction_ratio > 1.2 {
        eprintln!(
            "  → Sigmoid maintains {correction_ratio:.1}× larger correction than softmax under AdamW"
        );
        eprintln!(
            "  → HYPOTHESIS CONFIRMED: sigmoid's sink-free property keeps correction branch active"
        );
    } else if correction_ratio > 1.0 {
        eprintln!(
            "  → Sigmoid correction is slightly larger ({correction_ratio:.2}×), but not dramatically"
        );
        eprintln!(
            "  → Partial evidence for hypothesis, may need real language data to see full effect"
        );
    } else {
        eprintln!(
            "  → No significant difference between sigmoid and softmax correction under AdamW"
        );
        eprintln!("  → Hypothesis not supported on random data (may need real language data)");
    }

    // Both should at least reduce loss from initial
    let initial_loss = {
        let (out_init, _) = forward(
            &q,
            &k,
            &v,
            &x,
            &w_r_init,
            gate_scale,
            ParallaxActivation::Softmax,
            &mut ParallaxScratch::new(n, d),
        );
        out_init
            .chunks(d)
            .zip(target.chunks(d))
            .map(|(o, t)| {
                o.iter()
                    .zip(t.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / (n as f32)
    };
    eprintln!("  Initial loss: {initial_loss:.6}");
    eprintln!(
        "  Loss reduction: softmax={:.1}%, sigmoid={:.1}%",
        (1.0 - loss_sm_final / initial_loss) * 100.0,
        (1.0 - loss_sig_final / initial_loss) * 100.0,
    );
}

// ── Complementary: gate_scale dynamics ────────────────────────

/// Also train gate_scale as a learnable parameter to see if AdamW
/// collapses it to zero (softmax) or keeps it open (sigmoid).
#[test]
fn experiment_adamw_learnable_gate() {
    eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Experiment 140b: Learnable gate_scale under AdamW                          ║");
    eprintln!("║  Does AdamW collapse gate_scale → 0 for softmax but not sigmoid?           ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");
    eprintln!("║ step │ SM gate  │ Sig gate │ SM loss  │ Sig loss                           ║");
    eprintln!("╟──────┼──────────┼──────────┼──────────┼───────────────────────────────────╢");

    let d = DIM;
    let n = SEQ_LEN;
    let scale = 1.0 / (d as f32).sqrt();

    let mut rng = fastrand::Rng::with_seed(SEED + 77);
    let q = rand_vec(n * d, &mut rng);
    let k = rand_vec(n * d, &mut rng);
    let v = rand_vec(n * d, &mut rng);
    let x = rand_vec(d, &mut rng);

    // Target: slightly noisy base attention
    let target_config = ParallaxConfig {
        gate_scale: 0.0,
        zero_init: true,
        activation: ParallaxActivation::Softmax,
    };
    let mut target = vec![0.0f32; n * d];
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut target,
        n,
        d,
        scale,
        &vec![0.0f32; d * d],
        &x,
        &target_config,
        None,
    );
    let mut target_rng = fastrand::Rng::with_seed(SEED + 1111);
    for t in target.iter_mut() {
        *t += (target_rng.f32() - 0.5) * 0.1;
    }

    // Fixed W_R (don't train, only train gate_scale)
    let w_r = rand_vec(d * d, &mut fastrand::Rng::with_seed(SEED + 42));

    // Both start with gate_scale = 0.95 (open gate)
    let mut gate_sm: f32 = 0.95;
    let mut gate_sig: f32 = 0.95;
    // AdamW state for single scalar
    let mut m_sm = 0.0f32;
    let mut v_sm = 0.0f32;
    let mut m_sig = 0.0f32;
    let mut v_sig = 0.0f32;
    let mut t_adam = 0usize;

    for step in 0..=STEPS {
        // Forward softmax
        let sm_config = ParallaxConfig {
            gate_scale: gate_sm,
            zero_init: false,
            activation: ParallaxActivation::Softmax,
        };
        let mut out_sm = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_sm,
            n,
            d,
            scale,
            &w_r,
            &x,
            &sm_config,
            None,
        );

        // Forward sigmoid
        let sig_config = ParallaxConfig {
            gate_scale: gate_sig,
            zero_init: false,
            activation: ParallaxActivation::Sigmoid,
        };
        let mut out_sig = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_sig,
            n,
            d,
            scale,
            &w_r,
            &x,
            &sig_config,
            None,
        );

        // Loss
        let loss_sm: f32 = out_sm
            .chunks(d)
            .zip(target.chunks(d))
            .map(|(o, t)| {
                o.iter()
                    .zip(t.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / (n as f32);
        let loss_sig: f32 = out_sig
            .chunks(d)
            .zip(target.chunks(d))
            .map(|(o, t)| {
                o.iter()
                    .zip(t.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / (n as f32);

        if step % 20 == 0 || step == STEPS {
            eprintln!(
                "║ {step:>4} │ {gate_sm:>8.4} │ {gate_sig:>8.4} │ {loss_sm:>8.4} │ {loss_sig:>8.4}"
            );
        }

        // Gradient of loss w.r.t. gate_scale:
        // L = (1/N) Σ_i ||o_SA_i - gs * correction_i - target_i||^2
        // ∂L/∂gs = -(2/N) Σ_i (o_SA_i - gs*corr_i - target_i) · corr_i
        //        = -(2/N) Σ_i residual_i · corr_i
        // where residual_i = o_PLX_i - target_i

        // Compute correction (without gate_scale applied)
        let rho = {
            let mut rho = vec![0.0f32; d];
            compute_rho(&w_r, &x, &mut rho);
            rho
        };
        // Need sigma_kv for both activations
        let mut scores_buf = vec![0.0f32; n];
        let mut col_sums = vec![0.0f32; n];
        let mut sigma_kv_sm = vec![0.0f32; d * d];
        let mut sigma_kv_sig = vec![0.0f32; d * d];
        compute_sigma_kv(
            &q,
            &k,
            &v,
            n,
            d,
            scale,
            ParallaxActivation::Softmax,
            &mut scores_buf,
            &mut col_sums,
            &mut sigma_kv_sm,
        );
        compute_sigma_kv(
            &q,
            &k,
            &v,
            n,
            d,
            scale,
            ParallaxActivation::Sigmoid,
            &mut scores_buf,
            &mut col_sums,
            &mut sigma_kv_sig,
        );
        let mut corr_sm_vec = vec![0.0f32; d];
        let mut corr_sig_vec = vec![0.0f32; d];
        parallax_correction(&sigma_kv_sm, &rho, &mut corr_sm_vec);
        parallax_correction(&sigma_kv_sig, &rho, &mut corr_sig_vec);

        // ∂L/∂gs for softmax
        let mut grad_gs_sm = 0.0f32;
        let mut grad_gs_sig = 0.0f32;
        for i in 0..n {
            let off = i * d;
            for j in 0..d {
                let res_sm = out_sm[off + j] - target[off + j];
                let res_sig = out_sig[off + j] - target[off + j];
                grad_gs_sm += -2.0 * res_sm * (-corr_sm_vec[j]); // correction is subtracted
                grad_gs_sig += -2.0 * res_sig * (-corr_sig_vec[j]);
            }
        }
        grad_gs_sm /= n as f32;
        grad_gs_sig /= n as f32;

        // AdamW update for scalar gate
        if step < STEPS {
            t_adam += 1;
            let bc1 = 1.0 - BETA1.powi(t_adam as i32);
            let bc2 = 1.0 - BETA2.powi(t_adam as i32);

            // Softmax gate
            m_sm = BETA1 * m_sm + (1.0 - BETA1) * grad_gs_sm;
            v_sm = BETA2 * v_sm + (1.0 - BETA2) * grad_gs_sm * grad_gs_sm;
            let m_hat = m_sm / bc1;
            let v_hat = v_sm / bc2;
            gate_sm -= LR * (m_hat / (v_hat.sqrt() + EPS) + WEIGHT_DECAY * gate_sm);

            // Sigmoid gate
            m_sig = BETA1 * m_sig + (1.0 - BETA1) * grad_gs_sig;
            v_sig = BETA2 * v_sig + (1.0 - BETA2) * grad_gs_sig * grad_gs_sig;
            let m_hat = m_sig / bc1;
            let v_hat = v_sig / bc2;
            gate_sig -= LR * (m_hat / (v_hat.sqrt() + EPS) + WEIGHT_DECAY * gate_sig);
        }
    }

    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");

    eprintln!("\n── Gate Dynamics Verdict ──");
    eprintln!("  Softmax final gate: {gate_sm:.6}");
    eprintln!("  Sigmoid final gate: {gate_sig:.6}");

    if gate_sig > gate_sm * 1.5 {
        eprintln!(
            "  → Sigmoid gate stays {:.1}× more open than softmax ({:.4} vs {:.4})",
            gate_sig / gate_sm.max(1e-12),
            gate_sig,
            gate_sm
        );
        eprintln!("  → CONFIRMED: AdamW collapses softmax gate but sigmoid resists");
    } else {
        eprintln!(
            "  → Both gates converge to similar values ({:.4} vs {:.4})",
            gate_sm, gate_sig
        );
        eprintln!("  → Gate dynamics are similar on random data");
    }
}

// ── T2: Synthetic Sink Injection ─────────────────────────────────

/// Compute Σ_KV with sink bias injected into score computation.
///
/// scores[j] = q_i · k_j * scale + sink_bias[j]
/// where sink_bias[j] = sink_strength * exp(-j / decay_rate)
#[allow(clippy::too_many_arguments)] // test helper: fixed Σ_KV I/O shape
fn compute_sigma_kv_sink(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    activation: ParallaxActivation,
    sink_bias: &[f32],
    scores_buf: &mut [f32],
    col_sums: &mut [f32],
    sigma_kv: &mut [f32],
) {
    use katgpt_core::simd;
    let d = head_dim;
    let n = seq_len;
    col_sums[..n].fill(0.0);
    sigma_kv[..d * d].fill(0.0);

    for i in 0..n {
        let q_off = i * d;
        for j in 0..n {
            let k_off = j * d;
            scores_buf[j] = simd::simd_dot_f32(&q[q_off..q_off + d], &k[k_off..k_off + d], d)
                * scale
                + sink_bias[j];
        }
        let row = &mut scores_buf[..n];
        normalize_weights(row, activation);
        simd::simd_add_inplace(&mut col_sums[..n], &row[..n]);
    }

    let mut pv_buf = vec![0.0f32; d];
    // j needed for stride v_off/k_off = j*d
    #[allow(clippy::needless_range_loop)]
    for j in 0..n {
        let c_j = col_sums[j];
        if c_j == 0.0 {
            continue;
        }
        let v_off = j * d;
        let k_off = j * d;
        simd::simd_fused_decay_write(&mut pv_buf, 0.0, &v[v_off..v_off + d], c_j);
        simd::simd_outer_product_acc(sigma_kv, &pv_buf, &k[k_off..k_off + d], d, d);
    }
}

/// Forward pass with sink bias injected into scores.
/// Returns (output, raw_correction, sigma_kv).
#[allow(clippy::too_many_arguments)] // test helper: fixed parallax forward I/O shape
fn forward_sink(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    x: &[f32],
    w_r: &[f32],
    gate_scale: f32,
    activation: ParallaxActivation,
    sink_bias: &[f32],
    n: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    use katgpt_core::simd;
    let scale = 1.0 / (d as f32).sqrt();

    // Compute rho = W_R · x
    let mut rho = vec![0.0f32; d];
    compute_rho(w_r, x, &mut rho);

    // Compute sigma_kv with sink bias
    let mut scores_buf = vec![0.0f32; n];
    let mut col_sums = vec![0.0f32; n];
    let mut sigma_kv = vec![0.0f32; d * d];
    compute_sigma_kv_sink(
        q,
        k,
        v,
        n,
        d,
        scale,
        activation,
        sink_bias,
        &mut scores_buf,
        &mut col_sums,
        &mut sigma_kv,
    );

    // Compute base attention output with sink bias
    let mut output = vec![0.0f32; n * d];
    for i in 0..n {
        let q_off = i * d;
        let out_off = i * d;
        for j in 0..n {
            let k_off = j * d;
            scores_buf[j] = simd::simd_dot_f32(&q[q_off..q_off + d], &k[k_off..k_off + d], d)
                * scale
                + sink_bias[j];
        }
        normalize_weights(&mut scores_buf[..n], activation);
        // j needed for stride v_off = j*d
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            let p = scores_buf[j];
            let v_off = j * d;
            simd::simd_fused_scale_acc(
                &mut output[out_off..out_off + d],
                &v[v_off..v_off + d],
                p,
                d,
            );
        }
    }

    // Compute correction = Σ_KV · ρ
    let mut correction = vec![0.0f32; d];
    parallax_correction(&sigma_kv, &rho, &mut correction);

    // Apply correction: output -= gate_scale * correction (broadcast over rows)
    let mut scaled_corr = correction.clone();
    simd::simd_scale_inplace(&mut scaled_corr, -gate_scale);
    for i in 0..n {
        let off = i * d;
        simd::simd_add_inplace(&mut output[off..off + d], &scaled_corr[..d]);
    }

    (output, correction, sigma_kv)
}

/// Compute gradient of L = ||o_PLX - target||^2 w.r.t. W_R (sink-aware version).
fn compute_grad_w_r_sink(
    sigma_kv: &[f32],
    x: &[f32],
    output: &[f32],
    target: &[f32],
    gate_scale: f32,
    n: usize,
    d: usize,
) -> Vec<f32> {
    let mut grad = vec![0.0f32; d * d];

    // Mean residual
    let mut mean_residual = vec![0.0f32; d];
    for i in 0..n {
        let off = i * d;
        for j in 0..d {
            mean_residual[j] += output[off + j] - target[off + j];
        }
    }
    let s = 2.0 / (n as f32);
    for r in mean_residual.iter_mut() {
        *r *= s;
    }

    // d_L/d_correction = -gs · mean_residual
    let mut dl_dc = vec![0.0f32; d];
    for j in 0..d {
        dl_dc[j] = -gate_scale * mean_residual[j];
    }

    // d_L/d_rho = Σ_KV^T · d_L/d_correction
    let mut dl_drho = vec![0.0f32; d];
    for a in 0..d {
        let mut sum = 0.0f32;
        for b in 0..d {
            sum += sigma_kv[b * d + a] * dl_dc[b];
        }
        dl_drho[a] = sum;
    }

    // d_L/d_W_R = outer(dl_drho, x)
    for a in 0..d {
        for b in 0..d {
            grad[a * d + b] = dl_drho[a] * x[b];
        }
    }

    grad
}

#[test]
fn experiment_sink_injection() {
    eprintln!(
        "\n╔══════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    eprintln!(
        "║  Experiment T2: Synthetic Sink Injection — Sigmoid vs Softmax under AdamW             ║"
    );
    eprintln!(
        "║  Sink bias: sink_bias[j] = strength * exp(-j / decay_rate)                            ║"
    );
    eprintln!(
        "╚══════════════════════════════════════════════════════════════════════════════════════════╝"
    );

    let d = DIM;
    let n = SEQ_LEN;
    let gate_scale = 1.0f32;
    let steps = STEPS;

    // Generate frozen backbone
    let mut rng = fastrand::Rng::with_seed(SEED + 200);
    let q = rand_vec(n * d, &mut rng);
    let k = rand_vec(n * d, &mut rng);
    let v = rand_vec(n * d, &mut rng);
    let x = rand_vec(d, &mut rng);

    // Target: noisy base attention (no correction, no sinks)
    let target_config = ParallaxConfig {
        gate_scale: 0.0,
        zero_init: true,
        activation: ParallaxActivation::Softmax,
    };
    let mut target = vec![0.0f32; n * d];
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut target,
        n,
        d,
        1.0 / (d as f32).sqrt(),
        &vec![0.0f32; d * d],
        &x,
        &target_config,
        None,
    );
    let mut target_rng = fastrand::Rng::with_seed(SEED + 2999);
    for t in target.iter_mut() {
        *t += (target_rng.f32() - 0.5) * 0.1;
    }

    // Initial W_R
    let w_r_init = rand_vec(d * d, &mut fastrand::Rng::with_seed(SEED + 142));

    let sink_strengths: &[f32] = &[0.0, 0.5, 1.0, 2.0, 5.0];
    let decay_rates: &[usize] = &[2, 4, 8];

    // Results matrix: for each (strength, decay) → (sm_cor, sig_cor, sm_loss, sig_loss, sm_gate, sig_gate)
    // We'll also run a gate_scale learning variant for each config
    let mut results: Vec<(f32, usize, f32, f32, f32, f32)> = Vec::new();

    for &strength in sink_strengths {
        for &decay in decay_rates {
            // Build sink bias vector
            let sink_bias: Vec<f32> = (0..n)
                .map(|j| strength * (-(j as f32) / decay as f32).exp())
                .collect();

            eprintln!("\n── sink_strength={strength}, decay_rate={decay} ──");
            eprintln!("  sink_bias: {:?}", &sink_bias[..n.min(8)]);

            // ── Experiment A: W_R training ──
            let mut w_r_sm = w_r_init.clone();
            let mut adam_sm = AdamWState::new(d * d);
            let mut w_r_sig = w_r_init.clone();
            let mut adam_sig = AdamWState::new(d * d);

            let mut final_sm_cor = 0.0f32;
            let mut final_sig_cor = 0.0f32;
            let mut final_sm_loss = 0.0f32;
            let mut final_sig_loss = 0.0f32;

            for step in 0..=steps {
                let (out_sm, corr_sm, sigkv_sm) = forward_sink(
                    &q,
                    &k,
                    &v,
                    &x,
                    &w_r_sm,
                    gate_scale,
                    ParallaxActivation::Softmax,
                    &sink_bias,
                    n,
                    d,
                );
                let (out_sig, corr_sig, sigkv_sig) = forward_sink(
                    &q,
                    &k,
                    &v,
                    &x,
                    &w_r_sig,
                    gate_scale,
                    ParallaxActivation::Sigmoid,
                    &sink_bias,
                    n,
                    d,
                );

                // Loss
                let loss_sm: f32 = out_sm
                    .chunks(d)
                    .zip(target.chunks(d))
                    .map(|(o, t)| {
                        o.iter()
                            .zip(t.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                    })
                    .sum::<f32>()
                    / (n as f32);
                let loss_sig: f32 = out_sig
                    .chunks(d)
                    .zip(target.chunks(d))
                    .map(|(o, t)| {
                        o.iter()
                            .zip(t.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                    })
                    .sum::<f32>()
                    / (n as f32);

                // COR
                let mut out_norm_sm = 0.0f32;
                let mut out_norm_sig = 0.0f32;
                for i in 0..n {
                    out_norm_sm += vec_norm(&out_sm[i * d..(i + 1) * d]);
                    out_norm_sig += vec_norm(&out_sig[i * d..(i + 1) * d]);
                }
                let cor_sm = if out_norm_sm > 1e-12 {
                    vec_norm(&corr_sm) * gate_scale / out_norm_sm
                } else {
                    0.0
                };
                let cor_sig = if out_norm_sig > 1e-12 {
                    vec_norm(&corr_sig) * gate_scale / out_norm_sig
                } else {
                    0.0
                };

                if step % 50 == 0 || step == steps {
                    eprintln!(
                        "  step={:>3} SM: loss={:>10.4} COR={:.4} ‖W_R‖={:.4}  |  Sig: loss={:>10.4} COR={:.4} ‖W_R‖={:.4}",
                        step,
                        loss_sm,
                        cor_sm,
                        vec_norm(&w_r_sm),
                        loss_sig,
                        cor_sig,
                        vec_norm(&w_r_sig),
                    );
                }

                if step == steps {
                    final_sm_cor = cor_sm;
                    final_sig_cor = cor_sig;
                    final_sm_loss = loss_sm;
                    final_sig_loss = loss_sig;
                }

                // Gradient + AdamW update
                if step < steps {
                    let grad_sm =
                        compute_grad_w_r_sink(&sigkv_sm, &x, &out_sm, &target, gate_scale, n, d);
                    let grad_sig =
                        compute_grad_w_r_sink(&sigkv_sig, &x, &out_sig, &target, gate_scale, n, d);
                    adam_sm.step(&mut w_r_sm, &grad_sm, LR, WEIGHT_DECAY);
                    adam_sig.step(&mut w_r_sig, &grad_sig, LR, WEIGHT_DECAY);
                }
            }

            eprintln!(
                "  → COR ratio (sig/sm): {:.4}  |  loss ratio: {:.4}",
                if final_sm_cor > 1e-12 {
                    final_sig_cor / final_sm_cor
                } else {
                    f32::NAN
                },
                if final_sm_loss > 1e-12 {
                    final_sig_loss / final_sm_loss
                } else {
                    f32::NAN
                },
            );

            results.push((
                strength,
                decay,
                final_sm_cor,
                final_sig_cor,
                final_sm_loss,
                final_sig_loss,
            ));
        }
    }

    // ── Summary Table ──
    eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  T2 Summary: COR by (sink_strength, decay_rate)                             ║");
    eprintln!("╠══════════╦═════════╦═════════════╦═════════════╦══════════╦═══════════════╣");
    eprintln!("║ strength ║ decay   ║ SM COR      ║ Sig COR     ║ COR ratio║ Verdict       ║");
    eprintln!("╟──────────╫─────────╫─────────────╫─────────────╫──────────╫───────────────╢");

    for (strength, decay, sm_cor, sig_cor, _sm_loss, _sig_loss) in &results {
        let ratio = if *sm_cor > 1e-12 {
            *sig_cor / *sm_cor
        } else {
            0.0
        };
        let verdict = if sig_cor > sm_cor && ratio > 1.2 {
            "SIG>A"
        } else if sm_cor > sig_cor && (1.0 / ratio) > 1.2 {
            "SM>SIG"
        } else {
            "SIMILAR"
        };
        eprintln!(
            "║ {:>6.1}   ║ {:>5}   ║ {:>11.4} ║ {:>11.4} ║ {:>8.4} ║ {:<13} ║",
            strength, decay, sm_cor, sig_cor, ratio, verdict,
        );
    }

    eprintln!("╚══════════╩═════════╩═════════════╩═════════════╩══════════╩═══════════════╝");

    // ── T2b: Learnable gate with sink injection ──
    eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  T2b: Learnable gate_scale under AdamW with sink injection                  ║");
    eprintln!("╠══════════╦═════════╦═════════════╦═════════════╦══════════════════════════╣");
    eprintln!("║ strength ║ decay   ║ SM gate     ║ Sig gate    ║ Gate ratio              ║");
    eprintln!("╟──────────╫─────────╫─────────────╫─────────────╫──────────────────────────╢");

    let w_r_fixed = w_r_init.clone();

    for &strength in sink_strengths {
        for &decay in decay_rates {
            let sink_bias: Vec<f32> = (0..n)
                .map(|j| strength * (-(j as f32) / decay as f32).exp())
                .collect();

            let mut gate_sm: f32 = 0.95;
            let mut gate_sig: f32 = 0.95;
            let mut m_sm = 0.0f32;
            let mut v_sm = 0.0f32;
            let mut m_sig = 0.0f32;
            let mut v_sig = 0.0f32;
            let mut t_adam = 0usize;

            for _step in 0..steps {
                t_adam += 1;

                // Forward softmax
                let (out_sm, corr_sm, _) = forward_sink(
                    &q,
                    &k,
                    &v,
                    &x,
                    &w_r_fixed,
                    gate_sm,
                    ParallaxActivation::Softmax,
                    &sink_bias,
                    n,
                    d,
                );
                // Forward sigmoid
                let (out_sig, corr_sig, _) = forward_sink(
                    &q,
                    &k,
                    &v,
                    &x,
                    &w_r_fixed,
                    gate_sig,
                    ParallaxActivation::Sigmoid,
                    &sink_bias,
                    n,
                    d,
                );

                // Gradient of loss w.r.t. gate_scale
                // ∂L/∂gs = -(2/N) Σ_i residual_i · (-correction_i) = (2/N) Σ_i residual_i · correction_i
                let mut grad_gs_sm = 0.0f32;
                let mut grad_gs_sig = 0.0f32;
                for i in 0..n {
                    let off = i * d;
                    for j in 0..d {
                        let res_sm = out_sm[off + j] - target[off + j];
                        let res_sig = out_sig[off + j] - target[off + j];
                        grad_gs_sm += -2.0 * res_sm * (-corr_sm[j]);
                        grad_gs_sig += -2.0 * res_sig * (-corr_sig[j]);
                    }
                }
                grad_gs_sm /= n as f32;
                grad_gs_sig /= n as f32;

                // AdamW scalar update
                let bc1 = 1.0 - BETA1.powi(t_adam as i32);
                let bc2 = 1.0 - BETA2.powi(t_adam as i32);

                m_sm = BETA1 * m_sm + (1.0 - BETA1) * grad_gs_sm;
                v_sm = BETA2 * v_sm + (1.0 - BETA2) * grad_gs_sm * grad_gs_sm;
                let m_hat = m_sm / bc1;
                let v_hat = v_sm / bc2;
                gate_sm -= LR * (m_hat / (v_hat.sqrt() + EPS) + WEIGHT_DECAY * gate_sm);

                m_sig = BETA1 * m_sig + (1.0 - BETA1) * grad_gs_sig;
                v_sig = BETA2 * v_sig + (1.0 - BETA2) * grad_gs_sig * grad_gs_sig;
                let m_hat = m_sig / bc1;
                let v_hat = v_sig / bc2;
                gate_sig -= LR * (m_hat / (v_hat.sqrt() + EPS) + WEIGHT_DECAY * gate_sig);
            }

            let gate_ratio = if gate_sm.abs() > 1e-12 {
                gate_sig / gate_sm
            } else {
                f32::NAN
            };
            eprintln!(
                "║ {:>6.1}   ║ {:>5}   ║ {:>11.4} ║ {:>11.4} ║ {:>24.4} ║",
                strength, decay, gate_sm, gate_sig, gate_ratio,
            );
        }
    }

    eprintln!("╚══════════╩═════════╩═════════════╩═════════════╩══════════════════════════╝");

    // ── Final Verdict ──
    eprintln!("\n── T2 Final Verdict ──");

    // Check if any configuration shows sigmoid resisting collapse while softmax doesn't
    let sig_wins: Vec<_> = results
        .iter()
        .filter(|(_, _, sm_cor, sig_cor, _, _)| *sig_cor > *sm_cor * 1.2)
        .collect();
    let sm_wins: Vec<_> = results
        .iter()
        .filter(|(_, _, sm_cor, sig_cor, _, _)| *sm_cor > *sig_cor * 1.2)
        .collect();

    if !sig_wins.is_empty() {
        eprintln!(
            "  → Sigmoid outperforms softmax in {} config(s):",
            sig_wins.len()
        );
        for (s, d, sm_c, sig_c, _, _) in &sig_wins {
            eprintln!(
                "    strength={s}, decay={d}: COR sig={sig_c:.4} > sm={sm_c:.4} (ratio={:.2})",
                if *sm_c > 1e-12 {
                    sig_c / sm_c
                } else {
                    f32::INFINITY
                }
            );
        }
        eprintln!(
            "  → EVIDENCE: Sigmoid's sink-free property provides COR advantage under AdamW with synthetic sinks"
        );
    } else if !sm_wins.is_empty() {
        eprintln!(
            "  → Softmax outperforms sigmoid in {} config(s)",
            sm_wins.len()
        );
        eprintln!(
            "  → Hypothesis not supported: synthetic sinks don't create the expected COR divergence"
        );
    } else {
        eprintln!("  → No significant COR divergence in any configuration");
        eprintln!(
            "  → Sink injection alone may not reproduce the language-model-specific collapse mechanism"
        );
        eprintln!("  → Need real data (T3) or stronger sink simulation");
    }
}

// ══════════════════════════════════════════════════════════════
// T2c: Structured COR-Boosting — close the COR gap
// ══════════════════════════════════════════════════════════════
//
// Problem: Random Q/K/V → COR ≈ 0.06. Real LM → COR 4–12.
// The correction branch is barely active on random data, so
// AdamW has nothing to collapse.
//
// Strategy: Engineer structured Q/K/V that produce high COR:
//   1. "Sink tokens" (positions 0..sink_count) share a common K direction
//      → natural attention concentration (not additive bias)
//   2. V has position-dependent structure → Σ_KV carries real signal
//   3. Target deliberately differs from base attention → correction is needed
//
// This should push COR into the 4–12 range where the collapse
// mechanism operates in real language models.

/// Generate structured Q/K/V with natural attention sinks.
///
/// Design:
/// - First `sink_count` tokens: K vectors biased toward a shared "sink direction"
/// - Remaining tokens: random K vectors
/// - All Q vectors partially align with the sink direction
/// - V vectors carry position-dependent signal (not pure random)
fn generate_structured_qkv(
    n: usize,
    d: usize,
    sink_count: usize,
    sink_alignment: f32, // how much K vectors align with sink direction [0..1]
    q_alignment: f32,    // how much Q vectors align with sink direction [0..1]
    rng: &mut fastrand::Rng,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Create a sink direction (unit vector)
    let sink_dir: Vec<f32> = {
        let mut v = (0..d).map(|_| rng.f32() * 2.0 - 1.0).collect::<Vec<f32>>();
        let norm = vec_norm(&v);
        v.iter_mut().for_each(|x| *x /= norm);
        v
    };

    let mut q = vec![0.0f32; n * d];
    let mut k = vec![0.0f32; n * d];
    let mut v = vec![0.0f32; n * d];

    for i in 0..n {
        let off = i * d;

        // Q: random + alignment toward sink direction
        for j in 0..d {
            q[off + j] = (rng.f32() * 2.0 - 1.0) * (1.0 - q_alignment) + sink_dir[j] * q_alignment;
        }

        // K: sink tokens get strong alignment, others get random
        let alignment = if i < sink_count {
            sink_alignment
        } else {
            0.05 // slight alignment even for non-sink tokens
        };
        for j in 0..d {
            k[off + j] = (rng.f32() * 2.0 - 1.0) * (1.0 - alignment) + sink_dir[j] * alignment;
        }

        // V: position-dependent structure (sinusoidal + random)
        // Real LMs have structured V due to learned embeddings
        let pos_signal = (i as f32 * 0.5).sin();
        for j in 0..d {
            let basis = (j as f32 * 0.3).cos(); // per-dim basis function
            v[off + j] = (rng.f32() * 2.0 - 1.0) * 0.3 + pos_signal * basis * 0.7;
        }
    }

    (q, k, v)
}

#[test]
fn experiment_structured_cor_boosting() {
    eprintln!(
        "\n\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2557}"
    );
    eprintln!(
        "\u{2551}  Experiment T2c: Structured COR-Boosting \u{2014} Sigmoid vs Softmax under AdamW              \u{2551}"
    );
    eprintln!(
        "\u{2551}  Goal: Push COR into real-model range (4\u{2013}12) with structured Q/K/V                     \u{2551}"
    );
    eprintln!(
        "\u{255a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255d}"
    );

    let d = DIM;
    let n = SEQ_LEN;
    let gate_scale = 1.0f32;
    let steps = STEPS;

    // Configurations to sweep
    // sink_count: how many tokens form the attention sink
    // sink_alignment: how strongly sink K vectors align [0..1]
    // q_alignment: how strongly Q vectors point toward sinks [0..1]
    let configs: &[(&str, usize, f32, f32)] = &[
        ("baseline", 0, 0.0, 0.0),    // Random (T1 baseline, for comparison)
        ("weak_sink", 2, 0.5, 0.3),   // Mild sink structure
        ("strong_sink", 2, 0.9, 0.6), // Strong sink concentration
        ("mega_sink", 2, 0.95, 0.8),  // Very strong sink (extreme)
        ("broad_sink", 4, 0.7, 0.5),  // More sink tokens, moderate alignment
        ("full_sink", 4, 0.95, 0.9),  // Many tokens, very strong alignment
    ];

    let mut all_results: Vec<(&str, f32, f32, f32, f32, f32, f32)> = Vec::new();

    for &(label, sink_count, sink_alignment, q_alignment) in configs {
        eprintln!(
            "\n══ {} (sink_count={}, sink_align={:.2}, q_align={:.2}) ══",
            label, sink_count, sink_alignment, q_alignment
        );

        let mut rng = fastrand::Rng::with_seed(SEED + 300);
        let (q, k, v) =
            generate_structured_qkv(n, d, sink_count, sink_alignment, q_alignment, &mut rng);
        let x = rand_vec(d, &mut fastrand::Rng::with_seed(SEED + 301));

        // Target: base attention output (no correction) + structured perturbation
        // This simulates "the base attention is close but needs correction"
        // We make the target = base_attention + alpha * low_rank_signal
        // where the low_rank_signal is a known pattern the correction should learn
        let base_config = ParallaxConfig {
            gate_scale: 0.0, // no correction
            zero_init: true,
            activation: ParallaxActivation::Softmax,
        };
        let mut base_output = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut base_output,
            n,
            d,
            1.0 / (d as f32).sqrt(),
            &vec![0.0f32; d * d],
            &x,
            &base_config,
            None,
        );

        // Create a structured target signal: sum of sinusoidal basis functions
        // This is what the correction should learn to produce
        let alpha = 2.0f32; // signal strength — controls how much correction is needed
        let mut target = base_output.clone();
        for i in 0..n {
            let off = i * d;
            for j in 0..d {
                let freq1 = ((i * 3 + j * 7) as f32 * 0.1).sin();
                let freq2 = ((i * 5 + j * 11) as f32 * 0.07).cos();
                target[off + j] += alpha * (freq1 + freq2) * 0.5;
            }
        }

        // Initial W_R (shared between SM and Sig)
        let w_r_init = rand_vec(d * d, &mut fastrand::Rng::with_seed(SEED + 342));

        // Train W_R for both activations
        let mut w_r_sm = w_r_init.clone();
        let mut adam_sm = AdamWState::new(d * d);
        let mut w_r_sig = w_r_init.clone();
        let mut adam_sig = AdamWState::new(d * d);

        let mut final_sm_cor = 0.0f32;
        let mut final_sig_cor = 0.0f32;
        let mut final_sm_loss = 0.0f32;
        let mut final_sig_loss = 0.0f32;

        for step in 0..=steps {
            let sink_bias_zero = vec![0.0f32; n];

            let (out_sm, corr_sm, sigkv_sm) = forward_sink(
                &q,
                &k,
                &v,
                &x,
                &w_r_sm,
                gate_scale,
                ParallaxActivation::Softmax,
                &sink_bias_zero,
                n,
                d,
            );
            let (out_sig, corr_sig, sigkv_sig) = forward_sink(
                &q,
                &k,
                &v,
                &x,
                &w_r_sig,
                gate_scale,
                ParallaxActivation::Sigmoid,
                &sink_bias_zero,
                n,
                d,
            );

            // Loss
            let loss_sm: f32 = out_sm
                .chunks(d)
                .zip(target.chunks(d))
                .map(|(o, t)| {
                    o.iter()
                        .zip(t.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                })
                .sum::<f32>()
                / (n as f32);
            let loss_sig: f32 = out_sig
                .chunks(d)
                .zip(target.chunks(d))
                .map(|(o, t)| {
                    o.iter()
                        .zip(t.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                })
                .sum::<f32>()
                / (n as f32);

            // COR
            let mut out_norm_sm = 0.0f32;
            let mut out_norm_sig = 0.0f32;
            for i in 0..n {
                out_norm_sm += vec_norm(&out_sm[i * d..(i + 1) * d]);
                out_norm_sig += vec_norm(&out_sig[i * d..(i + 1) * d]);
            }
            let cor_sm = if out_norm_sm > 1e-12 {
                vec_norm(&corr_sm) * gate_scale / out_norm_sm
            } else {
                0.0
            };
            let cor_sig = if out_norm_sig > 1e-12 {
                vec_norm(&corr_sig) * gate_scale / out_norm_sig
            } else {
                0.0
            };

            if step % 50 == 0 || step == steps {
                eprintln!(
                    "  step={:>3} SM: loss={:>10.4} COR={:.4} \u{2016}W_R\u{2016}={:.4}  |  Sig: loss={:>10.4} COR={:.4} \u{2016}W_R\u{2016}={:.4}",
                    step,
                    loss_sm,
                    cor_sm,
                    vec_norm(&w_r_sm),
                    loss_sig,
                    cor_sig,
                    vec_norm(&w_r_sig),
                );
            }

            if step == steps {
                final_sm_cor = cor_sm;
                final_sig_cor = cor_sig;
                final_sm_loss = loss_sm;
                final_sig_loss = loss_sig;
            }

            // Gradient + AdamW update
            if step < steps {
                let grad_sm =
                    compute_grad_w_r_sink(&sigkv_sm, &x, &out_sm, &target, gate_scale, n, d);
                let grad_sig =
                    compute_grad_w_r_sink(&sigkv_sig, &x, &out_sig, &target, gate_scale, n, d);
                adam_sm.step(&mut w_r_sm, &grad_sm, LR, WEIGHT_DECAY);
                adam_sig.step(&mut w_r_sig, &grad_sig, LR, WEIGHT_DECAY);
            }
        }

        let cor_ratio = if final_sm_cor > 1e-12 {
            final_sig_cor / final_sm_cor
        } else {
            f32::NAN
        };
        let loss_ratio = if final_sm_loss > 1e-12 {
            final_sig_loss / final_sm_loss
        } else {
            f32::NAN
        };

        eprintln!(
            "  \u{2192} COR: SM={:.4} Sig={:.4} ratio={:.4}  |  Loss: SM={:.4} Sig={:.4} ratio={:.4}",
            final_sm_cor, final_sig_cor, cor_ratio, final_sm_loss, final_sig_loss, loss_ratio,
        );

        all_results.push((
            label,
            final_sm_cor,
            final_sig_cor,
            cor_ratio,
            final_sm_loss,
            final_sig_loss,
            loss_ratio,
        ));
    }

    // Summary Table
    eprintln!(
        "\n\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2557}"
    );
    eprintln!(
        "\u{2551}  T2c Summary: Structured COR-Boosting                                                    \u{2551}"
    );
    eprintln!(
        "\u{2560}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256c}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256c}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256c}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256c}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256c}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2563}"
    );
    eprintln!(
        "\u{2551} Config         \u{2551} SM COR      \u{2551} Sig COR     \u{2551} COR ratio\u{2551} SM loss     \u{2551} Sig loss    \u{2551}"
    );
    eprintln!(
        "\u{2560}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2563}"
    );

    for (label, sm_cor, sig_cor, cor_ratio, sm_loss, sig_loss, _) in &all_results {
        eprintln!(
            "\u{2551} {:<14} \u{2551} {:>11.4} \u{2551} {:>11.4} \u{2551} {:>8.4} \u{2551} {:>11.4} \u{2551} {:>11.4} \u{2551}",
            label, sm_cor, sig_cor, cor_ratio, sm_loss, sig_loss,
        );
    }

    eprintln!(
        "\u{255a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2569}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2569}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2569}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2569}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2569}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2568}"
    );

    // COR Gap Analysis
    eprintln!("\n\u{2500}\u{2500} T2c COR Gap Analysis \u{2500}\u{2500}");
    let max_cor = all_results
        .iter()
        .map(|(_, sm, sig, _, _, _, _)| sm.max(*sig))
        .fold(0.0f32, f32::max);
    eprintln!("  Max COR achieved: {:.4}", max_cor);
    eprintln!("  Real model COR range: 4\u{2013}12 (Research 135)");
    if max_cor < 1.0 {
        eprintln!(
            "  \u{2192} COR still below real-model range. Structured Q/K/V alone may not be sufficient."
        );
        eprintln!("  \u{2192} Consider: stronger signal target, larger alpha, or longer training.");
    } else if max_cor < 4.0 {
        eprintln!("  \u{2192} COR approaching real-model range but not yet there.");
        eprintln!(
            "  \u{2192} The correction is becoming active. Stronger sinks may push it further."
        );
    } else {
        eprintln!(
            "  \u{2192} COR in real-model range! The structured data activates the correction branch."
        );
    }

    // Sigmoid vs Softmax Divergence
    eprintln!("\n\u{2500}\u{2500} T2c Sigmoid vs Softmax Divergence \u{2500}\u{2500}");
    let diverged: Vec<_> = all_results
        .iter()
        .filter(|(_, _, _, ratio, _, _, _)| *ratio > 1.2 || *ratio < 0.8)
        .collect();

    if diverged.is_empty() {
        eprintln!("  \u{2192} No significant COR divergence in any configuration");
        eprintln!("  \u{2192} Even with structured sinks, both activations maintain similar COR");
    } else {
        eprintln!(
            "  \u{2192} COR divergence detected in {} config(s):",
            diverged.len()
        );
        for (label, sm_cor, sig_cor, ratio, _, _, _) in &diverged {
            let winner = if *ratio > 1.0 { "SIGMOID" } else { "SOFTMAX" };
            eprintln!(
                "    {}: COR sig={:.4} vs sm={:.4} (ratio={:.4}) \u{2192} {} wins",
                label, sig_cor, sm_cor, ratio, winner
            );
        }
    }

    // T2c Final Verdict
    eprintln!("\n\u{2500}\u{2500} T2c Final Verdict \u{2500}\u{2500}");
    if max_cor >= 4.0 && !diverged.is_empty() {
        eprintln!(
            "  \u{2192} HIGH-COR + DIVERGENCE: Sigmoid shows different behavior when correction is active"
        );
        eprintln!(
            "  \u{2192} This supports the hypothesis that activation function matters under high COR"
        );
    } else if max_cor >= 4.0 && diverged.is_empty() {
        eprintln!(
            "  \u{2192} HIGH-COR + NO DIVERGENCE: Correction is active but both activations behave similarly"
        );
        eprintln!(
            "  \u{2192} Evidence against the hypothesis (sink-free property doesn't help COR)"
        );
    } else {
        eprintln!(
            "  \u{2192} LOW-COR: Could not reproduce real-model COR range with synthetic data"
        );
        eprintln!("  \u{2192} The collapse mechanism requires COR in the 4\u{2013}12 range");
        eprintln!("  \u{2192} T3 (real data) remains essential for hypothesis testing");
    }
}

// ══════════════════════════════════════════════════════════════
// T2d: Reverse-Engineering High COR — start from high COR,
// observe whether AdamW collapses it differently for SM vs Sig
// ══════════════════════════════════════════════════════════════
//
// T2c showed structured Q/K/V can't push COR above ~0.06.
// The issue is scale: at dim=32, seq_len=16, the correction
// is always a tiny fraction of the output.
//
// New approach: INVERT the problem.
// 1. Use a large gate_scale (e.g., 10.0) to amplify the correction
// 2. Use a target that deliberately REQUIRES the correction
//    (target = base_attention + large_structured_residual)
// 3. Start from random W_R and train
//
// With gate_scale=10, even a modest correction magnitude gets
// multiplied to produce a significant COR. AdamW then decides
// whether to keep or suppress the amplified correction.
//
// Hypothesis: Under strong sinks, softmax's correction becomes
// noisy → AdamW suppresses gate. Sigmoid's correction stays
// clean → AdamW keeps gate high.

#[test]
fn experiment_reverse_cor() {
    eprintln!(
        "\n\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2557}"
    );
    eprintln!(
        "\u{2551}  Experiment T2d: Reverse-Engineering High COR                                         \u{2551}"
    );
    eprintln!(
        "\u{2551}  Start with large gate_scale, train W_R, track COR dynamics                                \u{2551}"
    );
    eprintln!(
        "\u{255a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255d}"
    );

    let d = DIM;
    let n = SEQ_LEN;
    let steps = STEPS;

    // Gate scale values to sweep — higher = correction is amplified more
    let gate_scales: &[f32] = &[1.0, 5.0, 10.0, 20.0];

    // Sink configurations
    let sink_configs: &[(&str, usize, f32, f32)] = &[
        ("no_sink", 0, 0.0, 0.0),
        ("strong_sink", 2, 0.9, 0.6),
        ("full_sink", 4, 0.95, 0.9),
    ];

    let mut results: Vec<(&str, f32, f32, f32, f32, f32, f32)> = Vec::new();

    for &gate_scale in gate_scales {
        for &(sink_label, sink_count, sink_alignment, q_alignment) in sink_configs {
            let label = format!("gs={:.0}_{}", gate_scale, sink_label);
            eprintln!(
                "\n\u{2550}\u{2550} gate_scale={}, {} (sink_count={}, align={:.2}) \u{2550}\u{2550}",
                gate_scale, sink_label, sink_count, sink_alignment
            );

            let mut rng = fastrand::Rng::with_seed(SEED + 400);
            let (q, k, v) =
                generate_structured_qkv(n, d, sink_count, sink_alignment, q_alignment, &mut rng);
            let x = rand_vec(d, &mut fastrand::Rng::with_seed(SEED + 401));

            // Target: base attention output + large structured residual
            // The larger the residual, the more the correction is needed
            let base_config = ParallaxConfig {
                gate_scale: 0.0,
                zero_init: true,
                activation: ParallaxActivation::Softmax,
            };
            let mut base_output = vec![0.0f32; n * d];
            tiled_attention_parallax_forward(
                &q,
                &k,
                &v,
                &mut base_output,
                n,
                d,
                1.0 / (d as f32).sqrt(),
                &vec![0.0f32; d * d],
                &x,
                &base_config,
                None,
            );

            // Target = base + large structured signal (scale proportional to gate_scale)
            // With high gate_scale, the correction should be able to fit this
            let signal_strength = gate_scale * 0.5; // scale signal with gate
            let mut target = base_output.clone();
            for i in 0..n {
                let off = i * d;
                for j in 0..d {
                    let freq = ((i * 3 + j * 7) as f32 * 0.1).sin();
                    target[off + j] += signal_strength * freq;
                }
            }

            let w_r_init = rand_vec(d * d, &mut fastrand::Rng::with_seed(SEED + 442));

            // Train both SM and Sig
            let mut w_r_sm = w_r_init.clone();
            let mut adam_sm = AdamWState::new(d * d);
            let mut w_r_sig = w_r_init.clone();
            let mut adam_sig = AdamWState::new(d * d);

            let mut final_sm_cor = 0.0f32;
            let mut final_sig_cor = 0.0f32;
            let mut final_sm_loss = 0.0f32;
            let mut final_sig_loss = 0.0f32;
            // Track COR trajectory (sample at key steps)
            let mut cor_trajectory_sm: Vec<(usize, f32)> = Vec::new();
            let mut cor_trajectory_sig: Vec<(usize, f32)> = Vec::new();

            for step in 0..=steps {
                let sink_bias_zero = vec![0.0f32; n];

                let (out_sm, corr_sm, sigkv_sm) = forward_sink(
                    &q,
                    &k,
                    &v,
                    &x,
                    &w_r_sm,
                    gate_scale,
                    ParallaxActivation::Softmax,
                    &sink_bias_zero,
                    n,
                    d,
                );
                let (out_sig, corr_sig, sigkv_sig) = forward_sink(
                    &q,
                    &k,
                    &v,
                    &x,
                    &w_r_sig,
                    gate_scale,
                    ParallaxActivation::Sigmoid,
                    &sink_bias_zero,
                    n,
                    d,
                );

                let loss_sm: f32 = out_sm
                    .chunks(d)
                    .zip(target.chunks(d))
                    .map(|(o, t)| {
                        o.iter()
                            .zip(t.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                    })
                    .sum::<f32>()
                    / (n as f32);
                let loss_sig: f32 = out_sig
                    .chunks(d)
                    .zip(target.chunks(d))
                    .map(|(o, t)| {
                        o.iter()
                            .zip(t.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                    })
                    .sum::<f32>()
                    / (n as f32);

                let mut out_norm_sm = 0.0f32;
                let mut out_norm_sig = 0.0f32;
                for i in 0..n {
                    out_norm_sm += vec_norm(&out_sm[i * d..(i + 1) * d]);
                    out_norm_sig += vec_norm(&out_sig[i * d..(i + 1) * d]);
                }
                let cor_sm = if out_norm_sm > 1e-12 {
                    vec_norm(&corr_sm) * gate_scale / out_norm_sm
                } else {
                    0.0
                };
                let cor_sig = if out_norm_sig > 1e-12 {
                    vec_norm(&corr_sig) * gate_scale / out_norm_sig
                } else {
                    0.0
                };

                if step % 50 == 0 || step == steps {
                    eprintln!(
                        "  step={:>3} SM: loss={:>10.4} COR={:.4}  |  Sig: loss={:>10.4} COR={:.4}",
                        step, loss_sm, cor_sm, loss_sig, cor_sig,
                    );
                    cor_trajectory_sm.push((step, cor_sm));
                    cor_trajectory_sig.push((step, cor_sig));
                }

                if step == steps {
                    final_sm_cor = cor_sm;
                    final_sig_cor = cor_sig;
                    final_sm_loss = loss_sm;
                    final_sig_loss = loss_sig;
                }

                if step < steps {
                    let grad_sm =
                        compute_grad_w_r_sink(&sigkv_sm, &x, &out_sm, &target, gate_scale, n, d);
                    let grad_sig =
                        compute_grad_w_r_sink(&sigkv_sig, &x, &out_sig, &target, gate_scale, n, d);
                    adam_sm.step(&mut w_r_sm, &grad_sm, LR, WEIGHT_DECAY);
                    adam_sig.step(&mut w_r_sig, &grad_sig, LR, WEIGHT_DECAY);
                }
            }

            let cor_ratio = if final_sm_cor > 1e-12 {
                final_sig_cor / final_sm_cor
            } else {
                f32::NAN
            };
            let loss_ratio = if final_sm_loss > 1e-12 {
                final_sig_loss / final_sm_loss
            } else {
                f32::NAN
            };

            // Detect COR trend: is COR growing or shrinking during training?
            let initial_sm_cor = cor_trajectory_sm.first().map(|(_, c)| *c).unwrap_or(0.0);
            let initial_sig_cor = cor_trajectory_sig.first().map(|(_, c)| *c).unwrap_or(0.0);
            let sm_cor_change = final_sm_cor - initial_sm_cor;
            let sig_cor_change = final_sig_cor - initial_sig_cor;

            eprintln!(
                "  \u{2192} Final COR: SM={:.4} Sig={:.4} ratio={:.4}  |  COR \u{394}: SM={:+.4} Sig={:+.4}",
                final_sm_cor, final_sig_cor, cor_ratio, sm_cor_change, sig_cor_change,
            );

            results.push((
                Box::leak(label.into_boxed_str()),
                final_sm_cor,
                final_sig_cor,
                cor_ratio,
                sm_cor_change,
                sig_cor_change,
                loss_ratio,
            ));
        }
    }

    // Summary Table
    eprintln!(
        "\n\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2557}"
    );
    eprintln!(
        "\u{2551}  T2d Summary: COR by (gate_scale, sink_config)                                            \u{2551}"
    );
    eprintln!(
        "\u{2560}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256c}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256c}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256c}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256c}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256c}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2563}"
    );
    eprintln!(
        "\u{2551} Config               \u{2551} SM COR      \u{2551} Sig COR     \u{2551} COR ratio\u{2551} SM COR \u{394}   \u{2551} Sig COR \u{394}   \u{2551}"
    );
    eprintln!(
        "\u{2560}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{256a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2563}"
    );

    for (label, sm_cor, sig_cor, cor_ratio, sm_delta, sig_delta, _) in &results {
        eprintln!(
            "\u{2551} {:<20} \u{2551} {:>11.4} \u{2551} {:>11.4} \u{2551} {:>8.4} \u{2551} {:>+11.4} \u{2551} {:>+11.4} \u{2551}",
            label, sm_cor, sig_cor, cor_ratio, sm_delta, sig_delta,
        );
    }

    eprintln!(
        "\u{255a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2569}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2569}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2569}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2569}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2569}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2568}"
    );

    // COR range achieved
    let max_cor = results
        .iter()
        .map(|(_, sm, sig, _, _, _, _)| sm.max(*sig))
        .fold(0.0f32, f32::max);
    eprintln!("\n\u{2500}\u{2500} T2d COR Range Achieved \u{2500}\u{2500}");
    eprintln!("  Max COR: {:.4} (target: 4\u{2013}12)", max_cor);

    // Check for differential COR collapse
    eprintln!("\n\u{2500}\u{2500} T2d Differential Collapse Analysis \u{2500}\u{2500}");
    let collapse_cases: Vec<_> = results
        .iter()
        .filter(|(_, _, _, ratio, sm_delta, sig_delta, _)| {
            // Sigmoid's COR grows more (or shrinks less) than softmax's
            *sig_delta > *sm_delta && (*ratio > 1.1 || *sig_delta - *sm_delta > 0.001)
        })
        .collect();

    if collapse_cases.is_empty() {
        eprintln!("  \u{2192} No differential COR dynamics between activations");
        eprintln!(
            "  \u{2192} Both activations' COR change identically under all gate_scale/sink configs"
        );
        eprintln!("  \u{2192} The collapse mechanism is not reproducible with synthetic data");
    } else {
        eprintln!(
            "  \u{2192} Differential COR dynamics in {} config(s):",
            collapse_cases.len()
        );
        for (label, sm_cor, sig_cor, ratio, sm_delta, sig_delta, _) in &collapse_cases {
            eprintln!(
                "    {}: COR sig={:.4} vs sm={:.4} (ratio={:.4}), \u{394} sig={:+.4} vs sm={:+.4}",
                label, sig_cor, sm_cor, ratio, sig_delta, sm_delta,
            );
        }
    }

    // T2d Final Verdict
    eprintln!("\n\u{2500}\u{2500} T2d Final Verdict \u{2500}\u{2500}");
    if max_cor >= 4.0 && !collapse_cases.is_empty() {
        eprintln!(
            "  \u{2192} HIGH-COR + DIFFERENTIAL: Sigmoid maintains higher COR than softmax under sinks"
        );
        eprintln!("  \u{2192} This supports the hypothesis: sigmoid's sink-free property helps");
    } else if max_cor >= 4.0 {
        eprintln!(
            "  \u{2192} HIGH-COR + NO DIFFERENTIAL: Both reach similar COR regardless of sinks"
        );
        eprintln!("  \u{2192} Evidence against the hypothesis");
    } else {
        eprintln!(
            "  \u{2192} COR still below real-model range ({:.4} vs 4\u{2013}12)",
            max_cor
        );
        eprintln!(
            "  \u{2192} Synthetic experiments cannot reproduce the COR magnitude of trained LMs"
        );
        eprintln!(
            "  \u{2192} The correction-to-output ratio is a property of the trained weight structure,"
        );
        eprintln!("     not something easily engineered from random initialization.");
        eprintln!(
            "  \u{2192} T3 (real model activations) remains the only path to test this hypothesis."
        );
    }
}
