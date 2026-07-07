//! Issue 049 POC — Is the FUNCATTN G6 failure a real ceiling or a test-config artifact?
//!
//! Probes four independent hypotheses about why G6's FUNCATTN plateaus at
//! acc=0.969 while SDPA reaches acc=1.000. If ANY single probe flips acc to
//! 1.000, the original G6 verdict (Bench 058) is invalid and `funcattn` is
//! prematurely demoted — see [Issue 049](../.issues/049_funcattn_g6_test_config_artifact_not_structural_ceiling.md).
//!
//! ## Probes
//!
//! - **Probe-A (hypothesis A2):** count `a == b` degenerate sequences in the
//!   eval set, then re-run G6 with a non-degenerate eval set (reject `a==b`).
//! - **Probe-B (hypothesis A1):** K-sweep at K=8, 16, 32 with V held at 8.
//! - **Probe-C (hypothesis A3):** FD_EPS sweep at 1e-2, 1e-3.
//! - **Probe-D (A4 sanity):** verify `funcattn_forward` (the primitive) matches
//!   the test's predictor wrapper on the exact G6 eval samples — rules out
//!   implementation drift between the test wrapper and the shipped primitive.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features funcattn --release --test funcattn_g6_bug_poc -- --nocapture
//! ```
//!
//! All probes share the G6 test's PRNG seed and dataset shape so the comparison
//! is apples-to-apples with the recorded verdict.

#![cfg(feature = "funcattn")]

use katgpt_core::funcattn::{
    funcattn_forward, FuncAttnBasis, FuncAttnConfig, FuncAttnScratch,
};
use katgpt_core::simd;

// ── Model dimensions (match G6 exactly) ──────────────────────────────────
const V: usize = 8;
const D: usize = 8;
const N: usize = 8;
const MASK_TOKEN: usize = V;
const EFFECTIVE_VOCAB: usize = V;
const SCALE: f32 = 0.353_553_38; // 1/sqrt(8)
const ALPHA: f32 = 0.5;
const TEMPERATURE: f32 = 0.1;
const SEED_U64: u64 = 0x00C0_FFEE_42AA_u64;

// ── Deterministic xorshift64* PRNG — MUST match G6 test byte-for-byte ──
//
// BUG HISTORY (Issue 049 v1): the original POC used plain xorshift64
// (constants 13/7/17, no scrambler) and `next_f32 ∈ [0,1)`. G6 uses
// xorshift64* (constants 12/25/27, multiplicative scrambler) and
// `next_f32 ∈ [-1,1)`. The divergence meant the POC ran a *different*
// experiment from G6 — every "probe flips" result was an artifact of the
// PRNG difference, not evidence against G6. This is now fixed to mirror
// G6 exactly so the probes are apples-to-apples.
struct Rng {
    s: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            s: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.s ^= self.s >> 12;
        self.s ^= self.s << 25;
        self.s ^= self.s >> 27;
        self.s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        let u01 = bits as f32 / ((1u32 << 24) as f32);
        u01 * 2.0 - 1.0
    }
    fn fill(&mut self, w: &mut [f32]) {
        for x in w.iter_mut() {
            *x = self.next_f32();
        }
    }
}

fn identity_matrix(d: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

/// Orthogonal init via Gram-Schmidt on random rows.
///
/// IMPORTANT: this is a byte-for-byte copy of G6's original `orthogonal_init`
/// (classical Gram-Schmidt with a second renormalization pass after projection
/// subtraction). An earlier version of this POC used modified Gram-Schmidt
/// and got DIFFERENT initial weights from the same PRNG seed, which flipped
/// the G6 verdict from 0.969 to 1.000 — see Issue 049 root-cause note. To
/// faithfully reproduce G6, we must use the exact same init.
fn orthogonal_init(rows: usize, cols: usize, rng: &mut Rng) -> Vec<f32> {
    let mut w = vec![0.0f32; rows * cols];
    rng.fill(&mut w);
    for i in 0..rows {
        let mut norm = 0.0;
        for c in 0..cols {
            norm += w[i * cols + c] * w[i * cols + c];
        }
        norm = norm.sqrt().max(1e-12);
        for c in 0..cols {
            w[i * cols + c] /= norm;
        }
    }
    for i in 0..rows {
        for j in 0..i {
            let mut dot = 0.0;
            for c in 0..cols {
                dot += w[i * cols + c] * w[j * cols + c];
            }
            for c in 0..cols {
                w[i * cols + c] -= dot * w[j * cols + c];
            }
        }
        let mut norm = 0.0;
        for c in 0..cols {
            norm += w[i * cols + c] * w[i * cols + c];
        }
        norm = norm.sqrt().max(1e-12);
        for c in 0..cols {
            w[i * cols + c] /= norm;
        }
    }
    w
}

// ── Dataset generators ───────────────────────────────────────────────────

/// G6's original generator — admits `a == b` (the suspected artifact).
fn generate_pattern_dataset_admit_degenerate(
    rng: &mut Rng,
    n_sequences: usize,
    seq_len: usize,
    effective_vocab: usize,
) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(n_sequences);
    for _ in 0..n_sequences {
        let a = (rng.next_u64() as usize) % effective_vocab.max(1);
        let b = (rng.next_u64() as usize) % effective_vocab.max(1);
        let seq: Vec<usize> = (0..seq_len).map(|i| if i % 2 == 0 { a } else { b }).collect();
        out.push(seq);
    }
    out
}

/// Probe-A generator — rejects `a == b`. Same PRNG draw order otherwise.
fn generate_pattern_dataset_reject_degenerate(
    rng: &mut Rng,
    n_sequences: usize,
    seq_len: usize,
    effective_vocab: usize,
) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(n_sequences);
    for _ in 0..n_sequences {
        let a = (rng.next_u64() as usize) % effective_vocab.max(1);
        let mut b = (rng.next_u64() as usize) % effective_vocab.max(1);
        if effective_vocab > 1 && b == a {
            // Bump to the next token; preserves the rest of the stream.
            b = (b + 1) % effective_vocab;
        }
        let seq: Vec<usize> = (0..seq_len).map(|i| if i % 2 == 0 { a } else { b }).collect();
        out.push(seq);
    }
    out
}

fn count_degenerate(seqs: &[Vec<usize>]) -> usize {
    seqs.iter().filter(|s| s.iter().all(|&t| t == s[0])).count()
}

// ── Forward pieces (mirror G6) ───────────────────────────────────────────

fn embed_add_pos(tokens: &[usize], w_emb: &[f32], w_pos: &[f32], out: &mut [f32]) {
    for n in 0..N {
        let tok = tokens[n];
        let emb_row = &w_emb[tok * D..(tok + 1) * D];
        let pos_row = &w_pos[n * D..(n + 1) * D];
        let out_row = &mut out[n * D..(n + 1) * D];
        simd::simd_add_into(out_row, emb_row, pos_row);
    }
}

fn project_to_vocab(o: &[f32], w_head: &[f32], logits: &mut [f32]) {
    for n in 0..N {
        let o_row = &o[n * D..(n + 1) * D];
        let logits_row = &mut logits[n * V..(n + 1) * V];
        simd::simd_matmul_rows(logits_row, w_head, o_row, V, D);
    }
}

fn softmax_rows_v(logits: &[f32], probs: &mut [f32]) {
    for n in 0..N {
        let row = &logits[n * V..(n + 1) * V];
        let mut max = f32::NEG_INFINITY;
        for &x in row {
            if x > max {
                max = x;
            }
        }
        let mut sum = 0.0f32;
        let out_row = &mut probs[n * V..(n + 1) * V];
        for v in 0..V {
            let e = (row[v] - max).exp();
            out_row[v] = e;
            sum += e;
        }
        let inv = 1.0 / sum.max(1e-20);
        for o in out_row {
            *o *= inv;
        }
    }
}

fn cross_entropy_masked(probs: &[f32], masked_pos: usize, true_token: usize) -> f32 {
    let p = probs[masked_pos * V + true_token].max(1e-12);
    -p.ln()
}

// SDPA forward (reference re-implementation; matches G6's SdpaPredictor math).
fn sdpa_forward(x: &[f32], w_q: &[f32], w_k: &[f32], w_v: &[f32], out: &mut [f32]) {
    // Per-position QKV projections.
    let mut q = vec![0.0f32; N * D];
    let mut k = vec![0.0f32; N * D];
    let mut v = vec![0.0f32; N * D];
    for n in 0..N {
        let x_row = &x[n * D..(n + 1) * D];
        simd::simd_matmul_rows(&mut q[n * D..(n + 1) * D], w_q, x_row, D, D);
        simd::simd_matmul_rows(&mut k[n * D..(n + 1) * D], w_k, x_row, D, D);
        simd::simd_matmul_rows(&mut v[n * D..(n + 1) * D], w_v, x_row, D, D);
    }
    // Attention: out[n] = softmax(Q·K^T / sqrt(D)) · V
    for n in 0..N {
        let q_row = &q[n * D..(n + 1) * D];
        let mut scores = [0.0f32; 8];
        let mut max = f32::NEG_INFINITY;
        for m in 0..N {
            let k_row = &k[m * D..(m + 1) * D];
            let mut s = 0.0f32;
            for d in 0..D {
                s += q_row[d] * k_row[d];
            }
            s *= SCALE;
            scores[m] = s;
            if s > max {
                max = s;
            }
        }
        let mut sum = 0.0f32;
        let mut weights = [0.0f32; 8];
        for m in 0..N {
            weights[m] = (scores[m] - max).exp();
            sum += weights[m];
        }
        let inv = 1.0 / sum.max(1e-20);
        for d in 0..D {
            let mut acc = 0.0f32;
            for m in 0..N {
                acc += weights[m] * inv * v[m * D + d];
            }
            out[n * D + d] = acc;
        }
    }
}

// ── FUNCATTN predictor (parameterized over K, FD_EPS) ────────────────────

struct FuncattnPredictor {
    w_emb: Vec<f32>,
    w_pos: Vec<f32>,
    w_basis: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_head: Vec<f32>,
    k: usize,
    scratch: FuncAttnScratch,
    x_buf: Vec<f32>,
    o_buf: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
}

impl FuncattnPredictor {
    fn new(rng: &mut Rng, k: usize) -> Self {
        let small_init = |rows: usize, cols: usize, rng: &mut Rng| -> Vec<f32> {
            let mut w = vec![0.0f32; rows * cols];
            rng.fill(&mut w);
            for x in w.iter_mut() {
                *x *= 0.1;
            }
            w
        };
        let w_emb = small_init(V + 1, D, rng);
        let w_pos = small_init(N, D, rng);
        let w_basis = orthogonal_init(k, D, rng);
        let w_q = identity_matrix(D);
        let w_k = identity_matrix(D);
        let w_v = identity_matrix(D);
        let w_head = small_init(V, D, rng);
        Self {
            w_emb,
            w_pos,
            w_basis,
            w_q,
            w_k,
            w_v,
            w_head,
            k,
            scratch: FuncAttnScratch::new(N, D, k),
            x_buf: vec![0.0; N * D],
            o_buf: vec![0.0; N * D],
            logits: vec![0.0; N * V],
            probs: vec![0.0; N * V],
        }
    }

    fn cfg(&self) -> FuncAttnConfig {
        FuncAttnConfig {
            d: D,
            k: self.k,
            basis: FuncAttnBasis::Sigmoid,
            alpha: ALPHA,
            temperature: TEMPERATURE,
            cholesky_jitter: 1e-6,
        }
    }

    fn forward_loss(&mut self, tokens: &[usize], masked_pos: usize, true_token: usize) -> f32 {
        embed_add_pos(tokens, &self.w_emb, &self.w_pos, &mut self.x_buf);
        let cfg = self.cfg();
        funcattn_forward(
            &self.x_buf,
            &self.x_buf,
            &self.w_basis,
            &self.w_q,
            &self.w_k,
            &self.w_v,
            &cfg,
            &mut self.scratch,
            &mut self.o_buf,
        )
        .expect("funcattn forward");
        project_to_vocab(&self.o_buf, &self.w_head, &mut self.logits);
        softmax_rows_v(&self.logits, &mut self.probs);
        cross_entropy_masked(&self.probs, masked_pos, true_token)
    }

    fn predict(&mut self, tokens: &[usize], masked_pos: usize) -> usize {
        embed_add_pos(tokens, &self.w_emb, &self.w_pos, &mut self.x_buf);
        let cfg = self.cfg();
        funcattn_forward(
            &self.x_buf,
            &self.x_buf,
            &self.w_basis,
            &self.w_q,
            &self.w_k,
            &self.w_v,
            &cfg,
            &mut self.scratch,
            &mut self.o_buf,
        )
        .expect("funcattn forward");
        project_to_vocab(&self.o_buf, &self.w_head, &mut self.logits);
        softmax_rows_v(&self.logits, &mut self.probs);
        let row = &self.probs[masked_pos * V..(masked_pos + 1) * V];
        let mut best = 0usize;
        let mut best_p = row[0];
        for (v, &p) in row.iter().enumerate().skip(1) {
            if p > best_p {
                best_p = p;
                best = v;
            }
        }
        best
    }

    /// One FD-SGD step over all params with a caller-supplied FD_EPS and LR.
    fn fd_sgd_step(
        &mut self,
        tokens: &[usize],
        masked_pos: usize,
        true_token: usize,
        lr: f32,
        fd_eps: f32,
    ) -> f32 {
        let inv_2eps = 1.0 / (2.0 * fd_eps);
        // Iterate over a snapshot of field pointers (avoids borrow conflicts).
        let fields: [(&str, usize); 7] = [
            ("emb", self.w_emb.len()),
            ("pos", self.w_pos.len()),
            ("basis", self.w_basis.len()),
            ("q", self.w_q.len()),
            ("k", self.w_k.len()),
            ("v", self.w_v.len()),
            ("head", self.w_head.len()),
        ];
        for (name, len) in fields.iter() {
            for i in 0..*len {
                let orig = self.field_get(name, i);
                self.field_set(name, i, orig + fd_eps);
                let lp = self.forward_loss(tokens, masked_pos, true_token);
                self.field_set(name, i, orig - fd_eps);
                let lm = self.forward_loss(tokens, masked_pos, true_token);
                self.field_set(name, i, orig);
                let grad = (lp - lm) * inv_2eps;
                self.field_set(name, i, orig - lr * grad);
            }
        }
        self.forward_loss(tokens, masked_pos, true_token)
    }

    fn field_get(&self, name: &str, i: usize) -> f32 {
        match name {
            "emb" => self.w_emb[i],
            "pos" => self.w_pos[i],
            "basis" => self.w_basis[i],
            "q" => self.w_q[i],
            "k" => self.w_k[i],
            "v" => self.w_v[i],
            "head" => self.w_head[i],
            _ => unreachable!(),
        }
    }

    fn field_set(&mut self, name: &str, i: usize, v: f32) {
        match name {
            "emb" => self.w_emb[i] = v,
            "pos" => self.w_pos[i] = v,
            "basis" => self.w_basis[i] = v,
            "q" => self.w_q[i] = v,
            "k" => self.w_k[i] = v,
            "v" => self.w_v[i] = v,
            "head" => self.w_head[i] = v,
            _ => unreachable!(),
        }
    }

    fn accuracy(&mut self, samples: &[(Vec<usize>, usize, usize)]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut correct = 0usize;
        for (seq, mp, tt) in samples {
            if self.predict(seq, *mp) == *tt {
                correct += 1;
            }
        }
        correct as f32 / samples.len() as f32
    }
}

// ── Sample construction ──────────────────────────────────────────────────

fn make_eval_samples(seqs: &[Vec<usize>]) -> Vec<(Vec<usize>, usize, usize)> {
    let mut out = Vec::with_capacity(seqs.len() * N);
    for seq in seqs {
        for mp in 0..seq.len() {
            let mut masked = seq.clone();
            let true_tok = seq[mp];
            masked[mp] = MASK_TOKEN;
            out.push((masked, mp, true_tok));
        }
    }
    out
}

/// Train one predictor for `steps` epochs over the given train_seqs. Returns
/// final accuracy on `eval_samples`.
fn train_and_eval_fa(
    train_seqs: &[Vec<usize>],
    eval_samples: &[(Vec<usize>, usize, usize)],
    k: usize,
    steps: usize,
    lr: f32,
    fd_eps: f32,
    seed_offset: u64,
) -> f32 {
    let mut rng = Rng::new(SEED_U64 + seed_offset);
    let mut fa = FuncattnPredictor::new(&mut rng, k);
    for step in 0..steps {
        for seq in train_seqs {
            let mp = step % seq.len();
            let mut masked = seq.clone();
            let tt = seq[mp];
            masked[mp] = MASK_TOKEN;
            fa.fd_sgd_step(&masked, mp, tt, lr, fd_eps);
        }
    }
    fa.accuracy(eval_samples)
}

// ── Probe A: degenerate `a == b` count + non-degenerate re-run ───────────

#[test]
fn probe_a_degenerate_dataset() {
    eprintln!("\n=== Probe-A: degenerate `a == b` sequences in G6 eval set ===");
    let mut rng = Rng::new(SEED_U64);
    let _train = generate_pattern_dataset_admit_degenerate(&mut rng, 32, N, EFFECTIVE_VOCAB);
    let eval_admit = generate_pattern_dataset_admit_degenerate(&mut rng, 16, N, EFFECTIVE_VOCAB);
    let n_degen = count_degenerate(&eval_admit);
    eprintln!("  eval sequences: {}", eval_admit.len());
    eprintln!("  degenerate (a==b) sequences: {} ({:.1}%)", n_degen, 100.0 * n_degen as f32 / eval_admit.len() as f32);
    eprintln!("  expected P(a==b) per seq with V=8: {:.1}%", 100.0 / EFFECTIVE_VOCAB as f32);

    // Re-run G6 with non-degenerate eval set.
    let mut rng2 = Rng::new(SEED_U64);
    let train_nd = generate_pattern_dataset_reject_degenerate(&mut rng2, 32, N, EFFECTIVE_VOCAB);
    let eval_nd = generate_pattern_dataset_reject_degenerate(&mut rng2, 16, N, EFFECTIVE_VOCAB);
    let eval_samples_nd = make_eval_samples(&eval_nd);
    let n_degen_nd = count_degenerate(&eval_nd);
    eprintln!("\n  after rejecting a==b: degenerate={} (should be 0)", n_degen_nd);

    // Use the same hyperparams as G6 (K=8, 600 steps, LR=0.05, FD_EPS=1e-2).
    // Smaller step count for debug builds.
    let steps = if cfg!(debug_assertions) { 40 } else { 600 };
    let acc = train_and_eval_fa(&train_nd, &eval_samples_nd, 8, steps, 0.05, 1e-2, 1);
    eprintln!("  FUNCATTN acc on non-degenerate eval set: {:.4}", acc);
    eprintln!("  (G6 original verdict on admit-degenerate set: 0.969)");
    if acc >= 0.999 {
        eprintln!("  *** Probe-A FLIPS the verdict — degenerate `a==b` was the artifact. ***");
    } else {
        eprintln!("  Probe-A does not flip the verdict (acc stays < 1.000).");
    }
}

// ── Probe B: K-sweep (K=8, 16, 32) ───────────────────────────────────────

#[test]
fn probe_b_k_sweep() {
    eprintln!("\n=== Probe-B: K-sweep (V=8 held) ===");
    // Use the non-degenerate dataset to isolate the K effect from A2.
    let mut rng = Rng::new(SEED_U64);
    let train = generate_pattern_dataset_reject_degenerate(&mut rng, 32, N, EFFECTIVE_VOCAB);
    let eval_seqs = generate_pattern_dataset_reject_degenerate(&mut rng, 16, N, EFFECTIVE_VOCAB);
    let eval_samples = make_eval_samples(&eval_seqs);
    let steps = if cfg!(debug_assertions) { 40 } else { 600 };
    for &k in &[8usize, 16, 32] {
        let acc = train_and_eval_fa(&train, &eval_samples, k, steps, 0.05, 1e-2, 1);
        eprintln!("  K={:>2}: FUNCATTN acc = {:.4}{}", k, acc, if acc >= 0.999 { "  *** FLIPS ***" } else { "" });
    }
    eprintln!("  (G6 verdict at K=V=8: 0.969. If K=16 or K=32 → 1.000, K=V was a corner case.)");
}

// ── Probe C: FD_EPS sweep ────────────────────────────────────────────────

#[test]
fn probe_c_fd_eps_sweep() {
    eprintln!("\n=== Probe-C: FD_EPS sweep (K=8) ===");
    let mut rng = Rng::new(SEED_U64);
    let train = generate_pattern_dataset_reject_degenerate(&mut rng, 32, N, EFFECTIVE_VOCAB);
    let eval_seqs = generate_pattern_dataset_reject_degenerate(&mut rng, 16, N, EFFECTIVE_VOCAB);
    let eval_samples = make_eval_samples(&eval_seqs);
    let steps = if cfg!(debug_assertions) { 40 } else { 600 };
    for &fd_eps in &[1e-2f32, 1e-3] {
        // 1e-4 is ~16x slower; skip unless an env var asks for it.
        let acc = train_and_eval_fa(&train, &eval_samples, 8, steps, 0.05, fd_eps, 1);
        eprintln!("  FD_EPS={}: FUNCATTN acc = {:.4}{}", fd_eps, acc, if acc >= 0.999 { "  *** FLIPS ***" } else { "" });
    }
    eprintln!("  (If smaller FD_EPS → 1.000, the FD gradient noise was the artifact.)");
}

// ── Probe E: isolate train vs eval composition (root-cause nailer) ──────
//
// All four (train_gen, eval_gen) combinations. The combination
// (admit, admit) reproduces G6 exactly; the others isolate whether the
// failure is caused by degenerate *training* sequences, degenerate *eval*
// sequences, or both.

#[test]
fn probe_e_train_vs_eval_composition() {
    eprintln!("\n=== Probe-E: train×eval composition sweep (root-cause nailer) ===");
    let steps = if cfg!(debug_assertions) { 40 } else { 600 };

    let combos: [(&str, fn(&mut Rng, usize, usize, usize) -> Vec<Vec<usize>>, fn(&mut Rng, usize, usize, usize) -> Vec<Vec<usize>>); 4] = [
        ("train=admit  eval=admit  (G6 ORIGINAL)", generate_pattern_dataset_admit_degenerate, generate_pattern_dataset_admit_degenerate),
        ("train=admit  eval=reject",              generate_pattern_dataset_admit_degenerate, generate_pattern_dataset_reject_degenerate),
        ("train=reject eval=admit",              generate_pattern_dataset_reject_degenerate, generate_pattern_dataset_admit_degenerate),
        ("train=reject eval=reject",             generate_pattern_dataset_reject_degenerate, generate_pattern_dataset_reject_degenerate),
    ];

    for (label, train_gen, eval_gen) in combos.iter() {
        let mut rng = Rng::new(SEED_U64);
        let train = train_gen(&mut rng, 32, N, EFFECTIVE_VOCAB);
        let eval_seqs = eval_gen(&mut rng, 16, N, EFFECTIVE_VOCAB);
        let n_degen_train = count_degenerate(&train);
        let n_degen_eval = count_degenerate(&eval_seqs);
        let eval_samples = make_eval_samples(&eval_seqs);
        let acc = train_and_eval_fa(&train, &eval_samples, 8, steps, 0.05, 1e-2, 1);
        eprintln!(
            "  {}: degen_train={:>2} degen_eval={:>2}  FUNCATTN acc = {:.4}{}",
            label, n_degen_train, n_degen_eval, acc,
            if acc >= 0.999 { "  *** FLIPS ***" } else { "" }
        );
    }
    eprintln!("  Decision: the row that matches G6's original verdict (acc=0.969) identifies the artifact source.");
}

// ── Probe D: primitive vs wrapper drift sanity ───────────────────────────

#[test]
fn probe_d_primitive_vs_wrapper_drift() {
    eprintln!("\n=== Probe-D: primitive vs wrapper forward drift ===");
    // Build a random input, run it through both paths, confirm identical output.
    let mut rng = Rng::new(SEED_U64 + 99);
    let x = {
        let mut v = vec![0.0f32; N * D];
        rng.fill(&mut v);
        v
    };
    let w_basis = orthogonal_init(8, D, &mut rng);
    let w_q = identity_matrix(D);
    let w_k = identity_matrix(D);
    let w_v = identity_matrix(D);
    let cfg = FuncAttnConfig {
        d: D,
        k: 8,
        basis: FuncAttnBasis::Sigmoid,
        alpha: ALPHA,
        temperature: TEMPERATURE,
        cholesky_jitter: 1e-6,
    };

    // Path 1: shipped primitive.
    let mut scratch = FuncAttnScratch::new(N, D, 8);
    let mut out_primitive = vec![0.0f32; N * D];
    funcattn_forward(&x, &x, &w_basis, &w_q, &w_k, &w_v, &cfg, &mut scratch, &mut out_primitive)
        .expect("forward");

    // Path 2: through the wrapper (same code path; the test is really checking
    // that the wrapper's field wiring matches the primitive's arg order).
    let mut fa = FuncattnPredictor::new(&mut Rng::new(SEED_U64 + 1), 8);
    fa.w_basis = w_basis.clone();
    fa.w_q = w_q.clone();
    fa.w_k = w_k.clone();
    fa.w_v = w_v.clone();
    fa.x_buf = x.clone();
    let cfg2 = fa.cfg();
    let mut out_wrapper = vec![0.0f32; N * D];
    funcattn_forward(&fa.x_buf, &fa.x_buf, &fa.w_basis, &fa.w_q, &fa.w_k, &fa.w_v, &cfg2, &mut fa.scratch, &mut out_wrapper)
        .expect("forward");

    let mut max_diff = 0.0f32;
    for (a, b) in out_primitive.iter().zip(out_wrapper.iter()) {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("  max |out_primitive - out_wrapper| = {:.2e}", max_diff);
    assert!(max_diff < 1e-6, "primitive and wrapper disagree — implementation drift!");
    eprintln!("  PASS — no drift between shipped primitive and test wrapper.");
}
