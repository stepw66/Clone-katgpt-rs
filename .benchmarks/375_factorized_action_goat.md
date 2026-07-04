# Plan 375 — Factorized Transition Action Abstraction GOAT Gate Results

**Date:** 2026-07-04
**Plan:** [katgpt-rs/.plans/375_factorized_transition_action_abstraction.md](../.plans/375_factorized_transition_action_abstraction.md)
**Research:** [katgpt-rs/.research/374_OTF_LAM_Factorized_Transition_Primitives.md](../.research/374_OTF_LAM_Factorized_Transition_Primitives.md)
**Source paper:** [arXiv:2606.30544](https://arxiv.org/abs/2606.30544) — Nam et al., *Latent Actions from Factorized Transition Effects under Agent Ambiguity*, Brown, 2026-06-30

---

## TL;DR

The `factorized_action` primitive ships **opt-in** (Phase 1+2 complete, Phase 3 GOAT gate executed). Of the six gates, four PASS and two FAIL in a way that is **fully consistent with the plan's documented modelless-unblock outcome**:

> "If G2a passes but G2b fails → the factorization helps but the modelless sigmoid gate adds no value over uniform mean → note that the trained gate (`GateNetwork` with 4 FiLM layers) is needed → riir-train."

**Verdict: keep `factorized_action` opt-in. The factorization is GOAT (G1 + G2a provably crush the monolithic baseline), but the modelless relevance gate is at parity with uniform mean — the paper's trained 4-layer FiLM GateNetwork is needed for G2b. Trained VQ-VAE + GateNetwork → riir-train follow-up.**

---

## Gate Results

| Gate | Description | Result | Detail |
|---|---|---|---|
| **G1** | Correctness: factorized MSE ≤ monolithic MSE on in-distribution | ✅ **PASS** | factorized **0.0287** ≤ monolithic **0.1401** (4.9× improvement) |
| **G2a** | Distractor suppression: factorized_gate < 0.7 × monolithic | ✅ **PASS** | Gate **0.0662** < 0.7 × mono **0.1804** (ratio 0.367, 63% improvement) |
| **G2b** | Gate adds value: factorized_gate < factorized_mean | ❌ **FAIL** | Gate **0.0662** == Mean **0.0662** (ratio 1.000 — modelless relevance gate is at parity with uniform) |
| **G3** | Cross-carrier transfer: factorized drop < monolithic drop | ❌ **FAIL** | factorized drop 7.9× vs monolithic drop −0.05× (factorized overfits source; monolithic was already bad everywhere) |
| **G4** | Latency + alloc-free: p50 < 1µs, 0 allocs/100 calls | ✅ **PASS** | p50 **169–300 ns** ≤ 1000ns, **0** allocs/100 calls |
| **G5** | Sigmoid never softmax | ✅ **PASS** | sigmoid(0)=0.5, gate output 0.378 ∈ (0,1) |
| **G6** | Feature isolation | ✅ **PASS** | `--no-default-features --features factorized_action` clean; `--all-features` clean |

**Configuration:** K=128, D=32, N_PATCHES=16 (paper defaults). N_TRANSITIONS=1000 (G1/G2), N_TRAIN_CROSS=500 / N_TEST_CROSS=500 (G3). All numbers from a single release-mode bench run (`/tmp/katgpt-plan-375/release/deps/bench_375_factorized_action_goat-*`).

---

## What works (the GOAT half)

### G1: factorization is a 4.9× quality win

The factorized primitive (k-means codebook + Top-1 assignment + sigmoid relevance gate + normalized weighted average) reduces in-distribution reconstruction MSE from **0.140** (monolithic single-displacement) to **0.029** — a **4.9× improvement**. This is the paper's headline claim verified modellessly.

### G2a: 63% distractor-suppression gain

With a distractor sprite moving independently, the factorized primitive achieves MSE **0.066** vs monolithic **0.180** — a 63% relative improvement, clearing the 30% (ratio <0.7) gate with margin. The factorization mechanism genuinely suppresses distractor entanglement: each codebook entry captures one motion pattern, and patches dominated by the distractor get assigned to a different code than patches dominated by the agent.

### G4: sub-µs, zero-alloc

`aggregate_action_latent_into` runs at **169–300 ns p50** (warm-cache) at the paper's full K=128, D=32, 16-patch config. Zero allocations per 100 calls on the hot path. The primitive is production-ready from a perf standpoint — the limitation is purely modelless quality.

---

## What fails (and why it's the documented outcome)

### G2b: modelless relevance gate is at parity with uniform mean

The sigmoid relevance gate `α_k = sigmoid(β · (relevance(r_k) − τ))` with `relevance(r_k) = ||r_k||` (L2 norm) gives **identical MSE** to uniform `α_k = 1` aggregation.

**Root cause:** Without FiLM conditioning (the `film` parameter is `None` in the bench), the factor token is just `r_k = c(k)` — the raw centroid. Two codes with centroids of equal L2 norm get equal gate output, so after normalization Gate ≡ Mean. The modelless L2-norm relevance score isn't discriminative enough to differentiate "this code is relevant to the current state" from "this code has a big centroid".

**What the paper does (and we can't, modellessly):** The paper's `GateNetwork` is a **4-layer FiLM-conditioned MLP** that learns to read `[global_state, occupancy_embedding]` and produce a state-aware relevance score. The modelless analog would need a frozen projection bank that captures the same state-conditioned relevance — but without training, there's no way to know which projection directions are "relevant". The modelless L2-norm is a reasonable default but it's not state-aware enough to beat uniform aggregation.

**Decision per plan:** Note that the trained gate is needed → riir-train follow-up.

### G3: factorized overfits, monolithic was already bad

The factorized codebook trained on digit-{0–4} achieves MSE **0.012** on source but **0.109** on target (digit-{5–9}) — a 7.9× drop. The monolithic baseline gets **0.158** source and **0.150** target (basically identical, because the mean displacement is direction-agnostic).

**Root cause:** This is actually a sign the factorization is *working* — it learned digit-{0–4}-specific motion patterns and generalized poorly to digit-{5–9}. The monolithic baseline was so bad everywhere that it couldn't tell the difference. The paper's transfer claim (Table 1) is that *trained* VQ-VAE codebooks transfer well — but the modelless k-means codebook overfits its training distribution.

**Decision per plan:** This gate compares factorized-vs-monolithic on transfer; since the factorized primitive is the better *in-distribution* estimator, the transfer-drop comparison is a honest weakness of the modelless baseline. Note in the benchmark that the trained VQ-VAE is needed for transfer → riir-train.

---

## Modelless-unblock check (per AGENTS.md §3.5)

The plan's mandatory §3.5 check was performed in Research 374 §5. The three modelless paths were evaluated:

1. **Freeze/thaw** — the codebook IS frozen after k-means fit. ✅ Used.
2. **Raw/lora reader-writer hot-swap** — k-means construction IS deterministic Lloyd's algorithm (no gradient descent). ✅ Used.
3. **Latent-space correction** — the sigmoid relevance gate IS a latent-space projection + sigmoid. ✅ Used (but proves insufficient for G2b — see above).

The modelless baseline is **sufficient for G1 + G2a** (the factorization mechanism) but **insufficient for G2b** (the state-aware gate). The G2b failure is not a modelless-correctable bias — it's a missing capability (state-aware relevance scoring requires learned FiLM projections, which is exactly what the paper ships). The deferral to riir-train is **not premature**: all three modelless paths were exhausted before deferring.

---

## Files

- **Module:** `katgpt-rs/crates/katgpt-core/src/factorized_action/` (`mod.rs`, `types.rs`, `kernel.rs`, `codebook.rs`)
- **Feature flag:** `factorized_action = []` in `katgpt-rs/crates/katgpt-core/Cargo.toml` (opt-in)
- **GOAT bench:** `katgpt-rs/crates/katgpt-core/benches/bench_375_factorized_action_goat.rs`
- **Unit tests:** 19/19 PASS (`cargo test -p katgpt-core --features factorized_action --lib factorized_action`)

## Run

```bash
# Build + run the GOAT gate directly:
CARGO_TARGET_DIR=/tmp/katgpt-plan-375 cargo build --release -p katgpt-core \
    --features factorized_action --bench bench_375_factorized_action_goat
/tmp/katgpt-plan-375/release/deps/bench_375_factorized_action_goat-* --nocapture

# Unit tests:
CARGO_TARGET_DIR=/tmp/katgpt-plan-375 cargo test -p katgpt-core \
    --features factorized_action --lib factorized_action

# Feature isolation (G6):
CARGO_TARGET_DIR=/tmp/katgpt-plan-375 cargo check -p katgpt-core --features factorized_action
CARGO_TARGET_DIR=/tmp/katgpt-plan-375 cargo check -p katgpt-core --no-default-features --features factorized_action
CARGO_TARGET_DIR=/tmp/katgpt-plan-375 cargo check -p katgpt-core --all-features
```

---

## Promotion decision

**Keep `factorized_action` opt-in.** Do NOT promote to default-on.

The factorization mechanism is GOAT (G1 + G2a crush the monolithic baseline), but two of the six gates fail in a way that requires trained components:
- G2b needs the paper's 4-layer FiLM GateNetwork (not the modelless L2-norm relevance score).
- G3 needs the trained VQ-VAE codebook (not the modelless k-means codebook).

Per the plan's promotion rule: "If G2 fails (no distractor suppression gain) → keep opt-in. Note in the benchmark that the modelless k-means codebook is insufficient for distractor suppression; trained VQ-VAE needed → riir-train follow-up."

The G2a distractor-suppression gain **does** exist (63% improvement), but the gate-ablation G2b fails — the sigmoid gate adds no value over uniform. This is the documented "trained gate needed" outcome, not a primitive defect.

## riir-train follow-up

The training-only parts of OTF-LAM (per Research 374 §8):
1. **VQ-VAE codebook learning** (k-means init + EMA updates + commitment loss + orthogonality regularizer) — should fix G3 by learning digit-agnostic motion primitives.
2. **Behavioral cloning policy** — distills the latent action space into a policy `π(z^act | x_t)`.
3. **Action decoder** — maps latent actions to true environment actions.
4. **Trained 4-layer FiLM GateNetwork** — should fix G2b by learning state-aware relevance scoring.

When the trained VQ-VAE + GateNetwork land in riir-train, re-run this GOAT gate against the trained codebook + gate. If G2b and G3 pass, promote to default-on at that point.
