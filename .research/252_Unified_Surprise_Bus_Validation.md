# Research 252: Unified Surprise Bus — Super-GOAT Validation

> **Source:** Plan 277 (Temporal Derivative Kernel), Research 243 §2.5, Issue 026
> **Date:** 2026-06-16
> **Status:** Active — controlled α-sweep validation
> **Related Research:** 243 (Temporal Derivative Kernel), 244 (Self-Evolver Faithfulness)
> **Related Plans:** 277 (Temporal Derivative Kernel)
> **Related Issues:** 026 (Super-GOAT escalation)
> **Classification:** Public

---

## TL;DR

Plan 277 shipped a single `TemporalDerivativeKernel<N>` primitive driving four independent consumers (HLA companion, δ-Mem gate, collapse detector, derivative curiosity) with the same paper-default α-pair (0.3, 0.03). All four GOAT gates passed. **This note validates whether the unified surprise bus is a real universal property** (Super-GOAT) or whether each consumer just happened to work with the paper default (GOAT only).

**Distilled for katgpt-rs (modelless, inference-time):**
The neocortical prediction-error signal (O'Reilly 2026) has a ~10× fast/slow time-constant ratio. If this ratio is universal across consumer domains (HLA reconstruction, memory consolidation, collapse detection, curiosity), then a single α-schedule suffices — no per-consumer tuning. This would make the derivative kernel a **zero-config surprise primitive**.

---

## 1. The Super-GOAT Claim

> A single `TemporalDerivativeKernel` with one paper-default α-pair (0.3, 0.03) drives all four consumers without per-consumer tuning, AND (0.3, 0.03) is in the Pareto-optimal region for ALL four simultaneously.

### What "Pareto-optimal" means here

For each consumer, we sweep α_fast ∈ {0.1, 0.2, 0.3, 0.5, 0.8} × α_slow ∈ {0.01, 0.03, 0.05, 0.1} (only valid combos where α_fast > α_slow) and measure the consumer's primary metric. The Pareto frontier is the set of α-pairs not dominated by any other. (0.3, 0.03) is "Pareto-optimal" if no other α-pair is strictly better on the consumer's target metric.

We use a relaxed criterion: (0.3, 0.03) is "in the Pareto-optimal region" if it is within 10% of the best observed metric value. This accounts for measurement noise and the flat-peak structure near the optimum.

---

## 2. Validation Protocol

### 2.1 Per-consumer sweep

| Consumer | N | α-setter API | Primary metric | Target |
|----------|---|-------------|----------------|--------|
| F1: HLA companion | 8 | `ReconstructionConfig { temporal_deriv_alpha_fast, temporal_deriv_alpha_slow }` | Event detection F1 (recall × precision on 1000-tick emotional-event trace) | recall ≥ 0.80, FPR ≤ 0.10 |
| F2: δ-Mem gate | 8 | `enable_surprise_gate_with_alphas(af, as)` | Write suppression % at recall ≥ baseline | suppression ≥ 30% |
| F3: Collapse detector | 1 | `with_temporal_deriv_alphas(af, as)` | FN reduction on 24 gradual-convergence traces | reduction ≥ 20% |
| F4: Derivative curiosity | 64 | `with_alphas(af, as)` | Recovery cycles from one-hot collapse | recovery ≤ 2× CGSP baseline |

For each consumer, we run the existing GOAT-gate test setup (from `.benchmarks/277_temporal_deriv_goat.md`) across the α-grid and record the metric.

### 2.2 α-grid

```
α_fast ∈ {0.1, 0.2, 0.3, 0.5, 0.8}
α_slow ∈ {0.01, 0.03, 0.05, 0.1}
Constraint: α_fast > α_slow (kernel validation)
Valid combos: 12 out of 20
```

### 2.3 Cross-consumer interference test

A single `TemporalDerivativeKernel<8>` instance (shared α-pair) drives all four consumers concurrently on a synthetic trace. We verify that no consumer's metric degrades more than 5% from its standalone measurement. This tests whether the consumers interfere when sharing the same surprise signal.

### 2.4 Honest failure mode

We actively seek a scenario where the unified α-pair fails:
- Very high α_fast (0.8) → fast EMA too reactive → noise amplification
- Very low α_slow (0.01) → slow EMA too sluggish → derivative never settles
- α_fast ≈ α_slow → no frequency separation → derivative ≈ 0 always

If (0.3, 0.03) is NOT Pareto-optimal for any consumer, we document the per-consumer recommended α-pair and downgrade from Super-GOAT to GOAT.

---

## 3. Sweep Results

**Benchmark:** `tests/bench_277_unified_surprise_bus.rs`
**Run:** `cargo test --features 'temporal_deriv sense_composition delta_mem collapse_aware_thinking cgsp' --test bench_277_unified_surprise_bus -- --nocapture --test-threads=1`

### F1: HLA companion (N=8)

Metric: `recall · (1 − FPR)` on 1000-tick emotional-event trace (3 events at ticks 200, 500, 800).

| α_fast ＼ α_slow | 0.01 | 0.03 | 0.05 | 0.1 |
|-----------------|------|------|------|-----|
| 0.1 | 0.00 | 1.00 | 1.00 | — |
| 0.2 | 1.00 | 1.00 | 1.00 | 1.00 |
| **0.3** | 1.00 | **1.00** ◄ | 1.00 | 1.00 |
| 0.5 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.8 | 1.00 | 1.00 | 1.00 | 1.00 |

**Paper (0.3, 0.03) = 1.00 | Best = 1.00 | Within ±10%: ✅ YES**

Note: the synthetic trace has very clear events (additive deltas on distinct dimensions), so almost any valid α-pair achieves perfect recall. A noisier trace would differentiate α-pairs more.

### F2: δ-Mem gate (N=8)

Metric: write suppression % on 1250-write stream (80% background cluster, 20% well-separated events).

| α_fast ＼ α_slow | 0.01 | 0.03 | 0.05 | 0.1 |
|-----------------|------|------|------|-----|
| 0.1 | 8.8% | 50.5% | 62.8% | — |
| 0.2 | 7.9% | 49.4% | 60.5% | **81.0%** |
| **0.3** | 7.7% | **49.4%** ◄ | 60.5% | 80.3% |
| 0.5 | 7.4% | 49.0% | 60.5% | 80.3% |
| 0.8 | 7.4% | 49.0% | 60.5% | 80.3% |

**Paper (0.3, 0.03) = 49.4% | Best = 81.0% (at 0.2, 0.1) | Within ±10%: ❌ NO**

**This is the outlier.** Suppression is dominated by `α_slow`: at `α_slow=0.1` the slow EMA tracks background writes faster, so the derivative decays to near-zero for repetitive writes → more aggressive gating. The paper-default's conservative `α_slow=0.03` leaves the slow EMA too sluggish to distinguish repetitive background from novel events.

**Recommended α-pair for F2:** `(0.3, 0.1)` — same `α_fast` as paper, but 3.3× faster slow EMA. Suppression 80.3% (vs paper 49.4%), well above the 30% target.

### F3: Collapse detector (N=1)

Metric: FN reduction on 24 gradual-convergence traces (hesitation-only baseline = 100% FN).

| α_fast ＼ α_slow | 0.01 | 0.03 | 0.05 | 0.1 |
|-----------------|------|------|------|-----|
| 0.1 | 0% | 100% | 100% | — |
| 0.2 | 0% | 100% | 100% | 100% |
| **0.3** | 0% | **100%** ◄ | 100% | 100% |
| 0.5 | 0% | 100% | 100% | 100% |
| 0.8 | 0% | 100% | 100% | 100% |

**Paper (0.3, 0.03) = 100% | Best = 100% | Within ±10%: ✅ YES**

Note: `α_slow=0.01` fails across the board — the slow EMA is too sluggish to track the convergence, so the derivative never drops below `τ_deriv`. Everything `α_slow ≥ 0.03` works.

### F4: Derivative curiosity (N=64)

Metric: recovery cycles from one-hot collapse (lower = better).

| α_fast ＼ α_slow | 0.01 | 0.03 | 0.05 | 0.1 |
|-----------------|------|------|------|-----|
| 0.1 | 1 | 1 | 1 | — |
| 0.2 | 1 | 1 | 1 | 1 |
| **0.3** | 1 | **1** ◄ | 1 | 1 |
| 0.5 | 1 | 1 | 1 | 1 |
| 0.8 | 1 | 1 | 1 | 1 |

**Paper (0.3, 0.03) = 1 | Best = 1 | Within ±10%: ✅ YES**

Note: all valid α-pairs recover in exactly 1 cycle. The metric is too coarse to differentiate α-pairs — the curiosity reward spikes on the first cycle after one-hot injection regardless of EMA coefficients (the preference vector shifts dramatically, producing a large derivative on any reasonable timescale).

---

## 4. Verdict

### **GOAT only (3/4) — NOT Super-GOAT**

The paper-default (0.3, 0.03) is within ±10% of the best metric for **3 of 4** consumers. The δ-Mem gate (F2) is the outlier.

| Consumer | Paper (0.3, 0.03) | Grid Best | Within ±10%? | Verdict |
|----------|:-:|:-:|:-:|:-:|
| F1: HLA companion | 1.000 | 1.000 | ✅ | Pareto-optimal |
| F2: δ-Mem gate | 49.4% | 81.0% (at 0.2, 0.1) | ❌ | **Outlier** |
| F3: Collapse detector | 100% | 100% | ✅ | Pareto-optimal |
| F4: Derivative curiosity | 1 cycle | 1 cycle | ✅ | Pareto-optimal |

Per the decision tree (3/4 → GOAT, not Super-GOAT): the unified surprise bus is **not a universal property**. The δ-Mem consolidation use case needs a faster slow EMA (`α_slow=0.1`) than the paper-default's `0.03`.

### Why F2 diverges

The δ-Mem gate suppresses writes when `surprise_norm() < θ`. The surprise is `‖fast − slow‖₂`. For repetitive background writes (tight cluster), both EMAs converge to the cluster centroid. The **rate** at which the slow EMA converges determines how quickly the derivative decays to zero (triggering suppression):

- `α_slow=0.03` (paper): slow EMA time constant ≈ 33 writes. After ~100 background writes, derivative is still ~30% of peak → many background writes pass the gate.
- `α_slow=0.1` (F2-optimal): slow EMA time constant ≈ 10 writes. After ~30 background writes, derivative ≈ 5% of peak → aggressive suppression.

The HLA companion (F1) and collapse detector (F3) don't have this issue because their signals are event-driven (sparse, well-separated events), not stream-driven (continuous background).

### Per-consumer recommended α-pairs

| Consumer | Recommended α_fast | Recommended α_slow | Rationale |
|----------|:-:|:-:|----------|
| F1: HLA companion | 0.3 | 0.03 | Paper-default; event-driven, insensitive to α_slow |
| F2: δ-Mem gate | 0.3 | **0.1** | Stream-driven; faster slow EMA needed for background suppression |
| F3: Collapse detector | 0.3 | 0.03 | Paper-default; `α_slow < 0.03` fails (too sluggish) |
| F4: Derivative curiosity | 0.3 | 0.03 | Paper-default; all valid α-pairs equivalent |

### Honest failure mode identified

The δ-Mem gate is the failure mode: the paper-default's conservative slow EMA (`α_slow=0.03`) under-suppresses repetitive background writes. The fix is consumer-specific: `α_slow=0.1` for the δ-Mem gate, keeping `α_fast=0.3` unchanged. This breaks the "single α-pair for all consumers" claim.

### Cross-consumer interference

Not run separately — the per-consumer sweep already shows that the δ-Mem gate needs a different α_slow. Running all 4 concurrently with a shared kernel would inherit F2's suboptimal suppression. The interference test is moot: F2 fails standalone, so it would fail concurrently too.

---

## 5. Implications

### If Super-GOAT confirmed

The neocortical prediction-error signal is a **zero-config primitive**: one α-schedule, four consumers, no tuning. This means:
- The ~10× ratio (0.3 / 0.03) captures the essential biological timescale separation.
- Future consumers can adopt the derivative kernel without α-tuning — just instantiate with defaults.
- The "unified surprise bus" architecture pattern (Plan 277) is validated as a reusable design.

### If GOAT only (not Super-GOAT)

Each consumer documents its recommended α-pair. The derivative kernel is still valuable (4/4 individual gates passed), but the "universal α" claim doesn't hold. This is honest — the biological mechanism may be domain-specific.

---

## Cross-References

- **Research 243:** Temporal Derivative Kernel (the primitive being validated)
- **Plan 277:** The implementation that shipped the 4 fusions
- **Issue 026:** The escalation issue tracking this validation
- **`.benchmarks/277_temporal_deriv_goat.md`:** The individual GOAT gate results

---

**TL;DR:** Plan 277's unified surprise bus is **GOAT, not Super-GOAT**. The paper-default α-pair (0.3, 0.03) is Pareto-optimal for 3/4 consumers (HLA, collapse, curiosity) but NOT for the δ-Mem gate, which needs `α_slow=0.1` for adequate background-write suppression (81% vs 49%). The ~10× ratio is universal for event-driven consumers but too conservative for stream-driven consolidation. Per-consumer α-tuning documented. Issue 026 closed as "GOAT but not Super-GOAT."
