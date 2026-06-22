# Research 286: Attention Drift — Depth-Invariance Diagnostic & Magnitude-Regularized Recursive Latent State

> **Source:** [Attention Drift: What Autoregressive Speculative Decoding Models Learn](https://arxiv.org/abs/2605.09992) — Eldenk, Mohapatra, Comlek, Oktay, Zhang, Xia (Northwestern / GE Aerospace / fal / Waterloo), arXiv:2605.09992v1, 2026-05-11
> **Date:** 2026-06-22
> **Status:** Active — **Super-GOAT** (all four novelty-gate questions pass; private guide created this session at `riir-ai/.research/151_Recursive_Latent_State_Magnitude_Hygiene_Guide.md`)
> **Related Research:** 217 (NextLat BeliefDrafter — has the bug), 258 (Sink-Aware — different mechanism), 242 (Topological State Tracking — recurrent belief), 276 (MicroBelief — already clamps), 244 (Self-Evolver Cognitive Integrity), 270 (ICT Distributional Branching), 277 (Temporal Derivative Kernel), 284 (Simplicity Bias Sampler), 232 (Task-Relevant Identifiability)
> **Related Plans:** 306 (this — open `DepthInvarianceDiagnostic` primitive), 217 (BeliefDrafter — diagnostic target), 276 (MicroBelief attractor — already-shipped fix pattern), 304 (GainCostLoopHalter — root-cause signal extension)
> **Cross-ref (riir-ai):** Research 151 (`Recursive_Latent_State_Magnitude_Hygiene_Guide` — private selling-point guide), Plan 331 (kernel audits + runtime fixes)
> **Classification:** Public (katgpt-rs engine slot)

---

## TL;DR

The paper names a failure mode we **already ship in production** but had not diagnosed: in any recursive latent-state kernel `h_{t+1} = h_t + Δ(h_t)` where the residual path is unnormalized, hidden-state magnitude `‖h_t‖` grows monotonically with chain depth. The kernel then implicitly learns a **depth-specific refinement** (acts as N+1, N+2... extra transformer layers stacked on the verifier) instead of a **depth-invariant autoregressive predictor**. The visible symptom — attention migrating from prompt-sink to recently-generated tokens ("attention drift") — is a *downstream* effect; the *root cause* is magnitude accumulation in the residual stream. Post-norm on the drafter output (and per-target-hidden RMSNorm) prevents the accumulation and lets short train-time-test depths generalize to longer inference chains (TTT 8→2 with no acceptance loss beyond training horizon).

**Distilled for katgpt-rs (modelless, inference-time):**

Two transferable primitives, both inference-time, both modelless:

1. **`DepthInvarianceDiagnostic`** — given a chain of recursive latent states `h_0, h_1, …, h_k` from *any* of our kernels (BeliefDrafter, micro_belief attractor, leaky integrator, HLA `evolve_hla`, functor composition), classify the kernel as `{DepthInvariant, DepthSpecificRefinement, Collapsed}` via three scalar signals computed in O(k·d) with zero allocation:
   - magnitude growth rate `d‖h_t‖/dt` (root-cause signal — paper's primary insight),
   - cosine-step `cos(h_t, h_{t-1})` (drift direction signal),
   - effective-rank trend `d effective_rank(h_0..h_t)/dt` (collapse signal — bridges to `BeliefRankPruner`).

   **This is the root-cause counterpart to `BeliefRankPruner`'s symptom-only `effective_rank` and `GainCostLoopHalter`'s coherence-decay signal.** Those detect *that* something is wrong; this names *why* (magnitude accumulation) and *what the kernel is actually doing* (depth-specific refinement vs stable autoregression).

2. **`MagnitudeRegularizedResidual`** wrapper — for our own kernels (NOT frozen pretrained drafters), wrap the residual update `h_{t+1} = h_t + Δ(h_t)` in an optional post-norm or scalar-pinch:
   ```
   h_raw  = h_t + Δ(h_t)
   h_{t+1} = post_norm ? rmsnorm(h_raw) : h_raw * pinch(h_raw)
   ```
   This is the *upstream* fix. The *downstream* fix (re-derive when drift exceeds τ) is `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` ("coherence-driven re-estimation scheduler") — the two compose as defense-in-depth.

**Critical finding from codebase audit:** our existing `BeliefDrafter` (Plan 217) at `katgpt-rs/src/speculative/belief_drafter.rs:80-81, 193-197` implements exactly the paper's failure mode: `h_{t+1} = h_t + FC3(...)` with the residual path **unnormalized** (LayerNorm is on the *input* only, line 35-50). This drafter IS subject to attention/magnitude drift and the paper's diagnosis applies verbatim. For BeliefDrafter specifically, only the **diagnostic** is modelless — the post-norm *fix* requires MLP retraining (paper §4.4 Table 4: inference-time magnitude pin drops acceptance 56% on pre-norm models). For our *own* kernels (HLA, latent_functor, micro_belief, engram retrieval chains, Raven consolidation) the fix is modelless because we own the kernel — we can post-norm at runtime without retraining anything.

**Verdict: Super-GOAT.** Latent-to-latent reframing (mandatory per skill §1.5): the mechanism is "magnitude accumulation in recursive latent state". Re-cast on the six Super-GOAT factory modules — every one of HLA / latent_functor / cgsp_runtime / sense / neuron-shard / LatCal has a recursive latent-state kernel susceptible to this failure mode. This is not adapter routing (GOAT-tier fallback framing per R269); the primary framing is latent-state hygiene across the entire runtime substrate.

---

## 1. Paper Core Findings

### 1.1 Attention drift (the symptom)

As a drafter generates successive tokens during a speculation chain, its attention progressively migrates from the prompt's attention sink onto its own recently-generated tokens. Observed across EAGLE-3 drafters and MTP heads (Qwen3.5 9B, Llama3.1 8B, GPT-oss 120B), suggesting drift is a property of autoregressive drafter *designs*, not of a specific architecture.

### 1.2 Hidden-state magnitude mismatch (the mechanism)

The paper traces drift to the **unnormalized residual path** between speculation steps:

| Target | `‖h_low‖` | `‖h_mid‖` | `‖h_high‖` | `‖h_FC‖` | `‖h*_1‖` | `‖h*_2‖` | `‖h*_3‖` | `‖h*_8‖` |
|---|---|---|---|---|---|---|---|---|
| Llama 3.1 8B | 0.56 | 0.58 | 0.78 | 12.46 | 3.92 | 4.87 | 5.86 | **14.02** |
| Qwen 3.5 35B | 0.03 | 0.12 | 0.21 | 0.89 | 3.47 | 4.03 | 4.42 | **5.92** |
| GPT-oss 120B | 3 | 163 | 1455 | 583 | 497 | 511 | 537 | **647** |

Three observations:
1. Drafter hidden states are at different magnitudes than the verifier's (the drafter's RMSNorm placement doesn't bridge this).
2. The verifier-fused `h_FC` is itself imbalanced — pre-norm dilution means `‖h_high‖ > ‖h_mid‖ > ‖h_low‖`, so `h_high` dominates the fused signal.
3. **`‖h*_k‖` grows monotonically with speculation depth k** — the drafter is not operating on a depth-invariant distribution.

### 1.3 Depth-specific refinement vs autoregression (the classification)

The monotonic magnitude growth reveals *what the drafter learns*: a pre-norm drafter behaves like **additional pre-norm transformer layers stacked on top of the verifier** (N+1, N+2, ..., N+k), not like an independent autoregressive predictor. Each speculation step changes the scale of the representation consumed by the next step. A *depth-invariant* drafter would have stable `‖h*_k‖` across k.

**Experimental proof (paper Figure 10):** a pre-norm drafter trained with TTT=2 collapses beyond its training horizon (acceptance drops to ~0 at k=8). A post-norm drafter trained with the same TTT=2 stays stable to k=16+. The pre-norm model *cannot* generalize beyond TTT because it has learned depth-specific transformations; the post-norm model *can* because it has been regularized toward a stable autoregressive function.

### 1.4 Two independent failure modes (the key disentanglement)

Paper §4.3 Table 2 is the load-bearing experiment. They compare pre-norm, gated-attention (eliminates sink), post-norm (elimiminates magnitude growth), and gated+post-norm on the same target:

| Drafter | `‖h*_1‖ → ‖h*_8‖` | Sink attn `1→8` | Recent attn `1→8` |
|---|---|---|---|
| Pre-norm | 3.92 → **14.02** | 0.46 → 0.08 | 0.14 → 0.31 |
| Gated-Attn | 8.41 → **39.07** | 0.03 → 0.02 | 0.29 → 0.32 |
| Post-norm | 1.21 → **1.20** | 0.11 → 0.10 | 0.50 → 0.53 |
| Gated + post-norm | 0.87 → 0.88 | 0.00 → 0.00 | 0.50 → 0.51 |

**Reading:**
- **Gated-attention fixes the visible sink drift but magnitude still grows 5.0×** → drift and magnitude are *independent* failure modes.
- **Post-norm pins magnitude and stabilizes attention** → addresses the root cause.
- **Gated + post-norm over-regularizes** (entropy collapses to H≈0.62, effective support ≈2 tokens) → applying both is *worse* than either alone.

This is the deepest insight in the paper. Fixing the visible symptom (sink drift) is *not* sufficient; the magnitude accumulation is the underlying disease.

### 1.5 Noise tolerance (mechanistic explanation for deep-chain behavior)

Paper §4.4 Table 3: post-norm tolerates an order of magnitude more hidden-state perturbation than pre-norm (58% vs 5% of baseline acceptance at α=0.5 noise). This is the mechanism for post-norm's better deep-chain behavior: it accumulates less error per speculation step. The drafter consumes its own predictions as inputs to subsequent steps; cleaner per-step updates compound gracefully.

### 1.6 Inference-time magnitude pin (Table 4 — the modelless control)

Pin `‖h_out‖` to `‖h_FC‖` at inference, no retraining:
- Pre-norm: acceptance 3.06 → 1.33 (-56%). Drift significantly lessened but still present weakly.
- Post-norm: acceptance 3.15 → 2.09 (-34%). Drift unchanged (already flat).

**Conclusion:** magnitude accumulation is *one contributor* to drift, not the only one (sink-collapse training-window effects also contribute). Inference-time pin reduces drift but hurts acceptance because the model wasn't trained for the new scale. **For our codebase: inference-time magnitude pin is a diagnostic demonstration, not a fix — the fix requires retraining for frozen-MLP drafters, but is modelless for our own kernels.**

### 1.7 Performance impact (paper §5)

Post-norm improvements over pre-norm EAGLE-3 across four target models (Llama 3.1 8B, Qwen 3 8B, Qwen 3.5 9B, GPT-OSS 20B):
- **Template perturbation:** up to **2×** acceptance length (pre-norm drops 52%, post-norm drops ≤5%).
- **Long context (LongBench, SWA):** **1.20–1.25×** across summarization/few-shot/coding.
- **Standard 7-benchmark average:** **1.10×** (math/coding/multi-turn chat).
- **TTT reduction:** 8→4 with no performance impact (≈1/3 training cost saved).

---

## 2. Distillation

### 2.1 The transferable primitive — `DepthInvarianceDiagnostic`

The distilled primitive is a **classifier**, not an attention modification. Given a chain `h_0, h_1, …, h_k ∈ ℝ^d` from any recursive latent-state kernel, plus a config of thresholds, classify the kernel's behavior:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DepthInvarianceKind {
    /// ‖h_t‖ flat, cos(h_t, h_{t-1}) ≈ const, effective rank flat.
    /// Kernel is a stable depth-invariant autoregressive predictor.
    DepthInvariant,
    /// ‖h_t‖ monotonically growing. Kernel is doing depth-specific refinement
    /// (acting as N+1, N+2, ... extra layers on whatever feeds it).
    DepthSpecificRefinement,
    /// Effective rank collapsing toward 1, magnitude may be flat OR growing.
    /// Kernel has collapsed onto a rank-1 attractor.
    Collapsed,
    /// Insufficient samples (k < min_samples).
    Insufficient,
}

pub struct DepthInvarianceDiagnostic {
    /// Magnitude growth rate: slope of ‖h_t‖ vs t, fitted via least-squares.
    /// Root-cause signal (paper's primary insight).
    pub magnitude_slope: f32,
    /// Mean cosine between consecutive states: low → oscillation, high → drift lock.
    pub mean_cos_step: f32,
    /// Effective rank trend: slope of effective_rank(h_0..h_t) vs t.
    pub effective_rank_slope: f32,
    /// Classification.
    pub kind: DepthInvarianceKind,
}

pub struct DepthInvarianceConfig {
    pub min_samples: usize,             // default 4 — need ≥4 to fit a slope
    pub magnitude_slope_drift: f32,     // default 0.05 — |slope| > this → DepthSpecific
    pub magnitude_slope_collapse: f32,  // default -0.05 — slope < this → Collapsed
    pub effective_rank_collapse: f32,   // default -0.05 — rank slope < this → Collapsed
    pub cos_step_drift_min: f32,        // default 0.95 — cos > this AND magnitude grows → locked drift
}
```

**Implementation constraints (per AGENTS.md):**
- Zero allocation in hot path: caller passes `&[f32]` slice of flattened `[k+1][d]` states + caller-owned scratch.
- O(k·d) — three passes (magnitude, cosine, effective-rank-via-`simd_dot_f32`). Effective rank uses our existing `BeliefRankPruner::flatness` math (Paper PR = `(Σh²)² / (n·Σh⁴)`).
- `DepthInvarianceKind` is `#[repr(u8)]` per AGENTS.md.
- No `manifold_power_iter_router` dependency (the rank computation here is per-timestep, not per-matrix — cheaper).

### 2.2 Vocabulary crosswalk (mandatory per skill §Workflow step 2)

| Paper term | Codebase-equivalent term (≥2 each) |
|---|---|
| attention drift | "latent state drift", "magnitude accumulation", "recursive state divergence" |
| depth-invariant | "stable recurrent kernel", "fixed-magnitude attractor", "bounded leaky integrator" |
| depth-specific refinement | "depth-specific transformation", "extra-layer emulation", "depth-conditioned functor" |
| drafter hidden state | "HLA state", "belief state", "sense projection", "latent subspace" |
| speculation step | "decision stage", "functor application", "cgsp cycle", "consolidation tick" |
| pre-norm / post-norm | "leaky integrator" (no output bound), "bounded attractor" (clamp/rmsnorm on output) |
| layer-stacking interpretation | "depth-conditioned functor composition", "stage-gated subspace activation" |

**Three-layer grep results (notes + code + vocabulary-translated):**

| Closest cousin | Repo | What it ships | Relation to this paper |
|---|---|---|---|
| **`BeliefDrafter::LatentDynamicsMLP`** (Plan 217) | katgpt-rs | Recursive `h_{t+1} = h_t + FC3(...)` with LayerNorm on input only | **HAS THE BUG** — unnormalized residual. This paper's diagnosis applies verbatim. Diagnostic-only fix (frozen MLP); retrain for post-norm. |
| **`micro_belief/attractor.rs`** (Plan 276) | katgpt-rs | Attractor + leaky belief kernels with `clamp(-1, 1)` | **Already ships the fix pattern.** This is the bounded-magnitude recursive latent state the paper prescribes. |
| **`latent_functor/reestimation.rs`** (riir-ai, Plan 303/317) | riir-ai | Coherence-driven re-estimation scheduler (re-derive functor when coherence < τ_reest) | **Downstream fix** (re-derive on drift). This paper provides the *upstream* fix (prevent drift at source via magnitude regularization). The two compose as defense-in-depth. |
| **`BeliefRankPruner`** (katgpt-rs/src/pruners/) | katgpt-rs | `effective_rank` / `flatness` of hidden states as a quality signal | **Symptom detector.** This paper's magnitude-growth-rate is the *root-cause* counterpart. |
| **`gain_cost_halt.rs::GainCostLoopHalter`** (Plan 304) | katgpt-rs | Halts loops based on gain/cost/cos_theta; "cost = coherence decay or staleness" | **Symptom-based halter.** This paper adds *root-cause* signal (magnitude slope > τ → halt, we're in depth-specific refinement). |
| **Sink-Aware Attention** (R258, Plan 287) | katgpt-rs | NOP/Broadcast sink classifier + dual-policy attention | **DIFFERENT mechanism.** That paper (2606.08105) is about *target-side* sink mechanisms. This paper (2605.09992) is about *drafter-side* magnitude accumulation. The two papers are frequently confused but address different layers of the stack. |
| **`Config.post_norm: bool`** (Gemma2) | katgpt-rs | Base transformer post-attention norm | **Same primitive, different scope.** Already shipped for the base transformer attention; this paper asks us to apply it to the *recursive residual path* of drafters and our own recursive kernels. |
| **`HLA evolve_hla`** (katgpt-rs/crates/katgpt-core/src/sense/) | katgpt-rs | Per-NPC 8-dim recurrent latent state | **Audit target (private guide).** If HLA's leaky integrator doesn't bound magnitude, NPC emotion state drifts over long tick horizons. |
| **Raven/δ-Mem consolidation** (riir-neuron-db) | riir-neuron-db | Offline shard consolidation cycles | **Audit target.** Each consolidation cycle is a "speculation step" on `style_weights[64]`; if unnormalized, shards drift in magnitude over cycles. |

### 2.3 Fusion — the novel combination

The single strongest fusion (this paper × existing shipped primitives × latent-functor reframing):

**Attention-Drift × latent_functor/reestimation.rs × GainCostLoopHalter × MicroBelief attractor** → **Defense-in-Depth Recursive Latent State Hygiene**

Three layers, each modelless:

```
Layer 1 (prevent): MagnitudeRegularizedResidual wrapper
   - Applied at kernel write: h_{t+1} = post_norm ? rmsnorm(h_t + Δ) : h_t + Δ
   - For OUR kernels (HLA, latent_functor, micro_belief, engram, Raven): free, no retraining
   - For BeliefDrafter (frozen MLP): diagnostic-only — pin hurts acceptance without retrain

Layer 2 (detect): DepthInvarianceDiagnostic
   - Applied per-tick on the last K state samples
   - Three-signal classifier (magnitude_slope, cos_step, effective_rank_slope)
   - Distinguishes {DepthInvariant, DepthSpecificRefinement, Collapsed}
   - Bridges to BeliefRankPruner (effective_rank) and GainCostLoopHalter (cos_theta oscillation)

Layer 3 (recover): latent_functor/reestimation.rs coherence-driven re-derivation
   - When diagnostic says DepthSpecificRefinement or Collapsed AND magnitude regularization
     is unavailable (frozen kernel) OR insufficient (distribution shift pushed us out of training support)
   - Re-derive the latent state from upstream signal (the existing reestimation.rs pattern)
   - This is the existing Plan 303/317 mechanism, now invoked with a *root-cause* trigger
```

**Why this fusion is Super-GOAT (not just GOAT):**

- **New capability class:** "provably depth-invariant recursive latent state at MMORPG scale" is a property no competitor offers. Current alternatives are (a) bound recursion depth (artificial cap, doesn't compose), (b) periodic reset (loses learned state), or (c) re-estimation thrash (what we currently do). Defense-in-depth gives us O(1) magnitude hygiene + drift-triggered fallback, no caps, no resets, minimal thrash.
- **Force multiplier:** the fusion ties together six existing pillars — BeliefDrafter (217), MicroBelief attractor (276), GainCostLoopHalter (304), latent_functor reestimation (303/317), BeliefRankPruner (pruners/), HLA evolve_hla (sense/). None of them alone provides this; the combination does.
- **Selling point:** "Our NPCs run unbounded-tick cognition with O(1) magnitude-regularized recursive latent state — no coherence-collapse re-estimation thrash at 20Hz × thousands of NPCs × hours of gameplay. The first MMORPG-scale runtime with provably depth-invariant per-NPC latent state." See private guide for the full commercial framing.

### 2.4 What does NOT transfer

| Paper element | Why it stays out |
|---|---|
| Drafter retraining with post-norm (§4–5) | Training-side. → riir-train. The paper's TTT reduction result (8→4) is interesting but requires retraining; we cannot apply it modellessly to BeliefDrafter. |
| Template perturbation benchmark construction | Eval recipe; we can borrow the four-condition protocol (template+BoS, no-BoS, no-template, no-template-no-BoS) for our own validation, but no code to ship. |
| The GPT-oss per-head learnable softmax bias | Architectural feature of GPT-oss's *target*; not relevant to our inference stack. |
| The Llama / Qwen / GPT-oss specific TTT and training hyperparameters | Training-only. |

---

## 3. Verdict

**🟢 Super-GOAT — open primitive + private guide + plans, all created this session.**

### One-line reasoning

The paper names a failure mode we **already ship in production** (BeliefDrafter's unnormalized residual), generalizes it to a universal property of recursive latent-state kernels (depth-specific refinement vs depth-invariant autoregression), and provides a clean three-signal diagnostic primitive that is the *root-cause* counterpart to four of our existing *symptom*-only detectors. Fusion with our existing `latent_functor/reestimation.rs` (downstream recovery) and `micro_belief/attractor.rs` (already-shipped prevention pattern) produces a defense-in-depth latent-state hygiene system that no competitor offers — a new capability class with a concrete MMORPG-scale selling point.

### Why Super-GOAT (novelty gate Q1–Q4)

| Q | Answer |
|---|---|
| **Q1: No prior art?** | **YES.** Three-layer grep (notes + code + vocabulary-translated) across all five repos confirms: (a) `BeliefDrafter` HAS the bug but no notes framing names it as magnitude drift; (b) `micro_belief/attractor.rs` ships the *fix pattern* (clamp) but doesn't generalize it to a diagnostic; (c) `BeliefRankPruner` uses `effective_rank` as a *quality* signal, not a *depth-drift* signal; (d) `gain_cost_halt.rs` uses coherence decay (symptom), not magnitude growth rate (root cause); (e) `Config.post_norm` exists for the base transformer but not for recursive residual paths; (f) Sink-Aware (R258) is a *different paper* about *target-side* sink mechanisms, frequently confused with this one. The specific primitive — "classify recursive latent-state kernel as depth-invariant vs depth-specific-refinement via magnitude-slope + cosine-step + effective-rank-slope" — is genuinely missing. |
| **Q2: New class of behavior?** | **YES.** "Provably depth-invariant recursive latent state at MMORPG scale" is a new *property class*, not an optimization. Current alternatives are caps (artificial), resets (state loss), or re-estimation thrash (current). Defense-in-depth gives O(1) magnitude hygiene with drift-triggered fallback. |
| **Q3: Product selling point?** | **YES.** "Our NPCs run unbounded-tick cognition with O(1) magnitude-regularized recursive latent state — no coherence-collapse re-estimation thrash at 20Hz × thousands of NPCs × hours. The first MMORPG-scale runtime with provably depth-invariant per-NPC latent state." Finishes the selling-point sentence strongly. See private guide §1 for the full commercial framing. |
| **Q4: Force multiplier?** | **YES** (≥2 pillars). Connects to: BeliefDrafter (Plan 217), micro_belief attractor (Plan 276), GainCostLoopHalter (Plan 304), latent_functor reestimation (Plan 303/317), BeliefRankPruner, HLA evolve_hla (sense/), Raven/δ-Mem consolidation (riir-neuron-db), Sink-Aware Attention (R258 — *complementary*, not duplicative). Eight pillars. |

### Avoidance of documented failure modes

- **R269 failure mode (defaulting to adapter routing when latent-space reframing is stronger):** AVOIDED. The primary framing is recursive latent-state hygiene across the six Super-GOAT factory modules (HLA / latent_functor / cgsp_runtime / sense / neuron-shard / LatCal). Adapter routing is not even a secondary framing here — the paper isn't about adapter composition.
- **`evolve_hla` failure mode (no notes framing at all → false Super-GOAT claim):** ADDRESSED. The private guide (riir-ai/.research/151) explicitly audits `evolve_hla` and the other recursive kernels for the bug before claiming the selling point.
- **DiPOD / `latent_functor/reestimation.rs` failure mode (notes framing under different vocabulary → missed by paper-vocabulary grep):** ADDRESSED. The vocabulary crosswalk (§2.2) explicitly maps the paper's "depth-specific refinement / pre-norm magnitude growth" to the codebase's "depth-conditioned functor composition / leaky integrator without output bound", and the grep hit `reestimation.rs` on the first pass via this translation.

### Mandatory outputs (created this session)

| Artifact | Repo | Path | Status |
|---|---|---|---|
| Open primitive (math, no game semantics) | katgpt-rs | `crates/katgpt-core/src/depth_invariance.rs` (new, behind `depth_invariance` feature) | **Plan 306 — to ship** |
| Open plan | katgpt-rs | `.plans/306_depth_invariance_diagnostic.md` | **Created this session** |
| Private selling-point guide | riir-ai | `.research/151_Recursive_Latent_State_Magnitude_Hygiene_Guide.md` | **Created this session** |
| Private runtime audit + fix plan | riir-ai | `.plans/331_recursive_latent_state_magnitude_hygiene_runtime.md` | **Created this session** |

### Validation protocol (from private guide §5)

- **G1 (correctness):** `DepthInvarianceDiagnostic` labels match hand-built ground truth on synthetic chains: flat-magnitude → `DepthInvariant`; linearly-growing-magnitude → `DepthSpecificRefinement`; rank-1 collapse → `Collapsed`; <min_samples → `Insufficient`. 8 tests.
- **G2 (BeliefDrafter audit):** run the diagnostic on `LatentDynamicsMLP::forward_into` chain outputs at TTT=2 and TTT=8. Pre-paper-fix BeliefDrafter should classify as `DepthSpecificRefinement` at k>TTT (matching paper Figure 10 left panel). This *reproduces* the paper's finding on our own drafter — if it doesn't, our drafter is somehow immune (worth understanding why).
- **G3 (micro_belief control):** run the diagnostic on `micro_belief/attractor.rs` chain outputs. Should classify as `DepthInvariant` (clamp bounds magnitude). This is the negative control — confirms the diagnostic correctly distinguishes healthy kernels.
- **G4 (latency):** `DepthInvarianceDiagnostic::classify_chain` overhead ≤ 5% of one forward pass of the audited kernel, on d=8..1024, k=4..64.
- **G5 (private guide Gs, in riir-ai/.research/151):** kernel-by-kernel audit of HLA / latent_functor / cgsp_runtime / engram / Raven — each classified and (where the kernel is ours) magnitude-regularized. Benchmarked before/after on the crowd-scale coherence benchmark.

If G2 reproduces (our BeliefDrafter has the bug) AND G3 confirms (micro_belief is healthy) AND G4 meets latency → promote `depth_invariance` to default-on diagnostic. If any kernel audit in G5 reveals a bug fixable by `MagnitudeRegularizedResidual` that produces a measurable crowd-coherence gain at 20Hz × 1000 NPCs → Super-GOAT confirmed, plan the kernel fix in riir-ai.

### Routing

| Artifact | Repo | Path |
|---|---|---|
| `DepthInvarianceDiagnostic` + `MagnitudeRegularizedResidual` primitives | katgpt-rs (public, MIT) | `crates/katgpt-core/src/depth_invariance.rs` (new) |
| `DepthInvarianceConfig` extension to existing Config | katgpt-rs | `crates/katgpt-core/src/types/config.rs` |
| Plan | katgpt-rs | `.plans/306_depth_invariance_diagnostic.md` |
| BeliefDrafter audit (diagnostic-only, no fix without retrain) | katgpt-rs | `.plans/306_*.md` Phase 3 |
| Private guide | riir-ai | `.research/151_Recursive_Latent_State_Magnitude_Hygiene_Guide.md` |
| Runtime audits + MagnitudeRegularizedResidual wiring | riir-ai | `.plans/331_recursive_latent_state_magnitude_hygiene_runtime.md` |
| Raven/δ-Mem consolidation audit | riir-neuron-db | Issue (file in `riir-neuron-db/.issues/` if G5 reveals shard drift) |
| LatCal bridge (raw ↔ latent boundary) | riir-chain | Not required — LatCal already enforces deterministic scalar commitment; magnitude hygiene is on the latent side of the bridge |

---

## References

- Eldenk, D. et al. (2026). *Attention Drift: What Autoregressive Speculative Decoding Models Learn*. arXiv:2605.09992.
- Li, Y. et al. (2026). *EAGLE-3: Scaling up inference acceleration of large language models via training-time test*. NeurIPS 2026. (The drafter architecture this paper diagnoses.)
- Fesser, L. et al. (2026). *A Unifying View of Attention Sinks: Two Algorithms, Two Solutions*. arXiv:2606.08105. (Complementary paper — *target-side* sink mechanisms. Frequently confused with this one; we keep both as separate research notes R258 and R286.)
- Xiong, R. et al. (2020). *On layer normalization in the transformer architecture*. ICML. (Pre-norm vs post-norm in standard transformers.)
- Qiu, Z. et al. (2025). *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*. NeurIPS. (The gated-attention variant the paper compares against.)
- Katgpt-rs Plan 217 — NextLat BeliefDrafter (has the bug; diagnostic target).
- Katgpt-rs Plan 276 — MicroBelief attractor (ships the fix pattern; negative control).
- Katgpt-rs Plan 304 — GainCostLoopHalter (symptom-based halter; root-cause signal extension).
- Riir-ai Plan 303/317 — latent_functor runtime (downstream re-derivation fix; composes as defense-in-depth).
