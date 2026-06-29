# Research 345: Chain-of-Embedding (CoE) — Output-Free LLM Self-Evaluation

> **Source:** Wang, Zhang, Yang, Wong, Wang, *Latent Space Chain-of-Embedding Enables Output-Free LLM Self-Evaluation* — [arXiv:2410.13640](https://arxiv.org/abs/2410.13640) (ICLR 2025, v2 Mar 2025)
> **Date:** 2026-06-29
> **Status:** Done
> **Related Research:** 324 (Trajectory Geometry — **direct prior art**, ships the metrics), 286 (Depth-Invariance Diagnostic), 053 (CNA), 325 §7.2 G3 (survey gap framing), 322 (conformal floor rule), 284 (CLR — UQ cousin), 255 (VibeThinker CLR), 165 (Per-NPC Conformal UQ Guide)
> **Related Plans:** 342 (latent_trajectory_geometry — **COMPLETE**), 340 (conformal overlay — floor, Phase 1 pending), 284 (CLR — shipped, DEFAULT-ON), 306 (depth-invariance)
> **Classification:** Public

---

## TL;DR

The paper proposes **Chain-of-Embedding (CoE)** — the layer-by-layer trajectory of mean-pooled hidden states `(h⁰, h¹, …, hᴸ)` — and shows that two geometric features of this trajectory (per-step magnitude change `M`, per-step angle change `A`) differ systematically between correct and incorrect LLM responses. It combines the two features via either a real-space linear sum (CoE-R) or a complex-plane magnitude+argument sum (CoE-C) into a single scalar score, used as a label-free, output-free self-evaluation / correctness-classification signal. AUROC 60–85% across 7 LLMs × 4 domains, **millisecond-scale** compute (no softmax over vocab, no sampling), beats 10 baselines including perplexity, entropy, energy, MC-Dropout, LN-Entropy, and Eigenscore.

**Verdict: Gain.** Every geometric primitive the paper uses **already ships** as `katgpt-rs/crates/katgpt-core/src/latent_trajectory_geometry.rs` (Plan 342 / Research 324): `length` = paper's `Σ M`, `mean_curvature` = paper's `κ̄`, `min_adjacent_cosine` = paper's cosine stability. The shipped module **deliberately refused** to claim UQ (its docstring invokes the "Report the Floor" rule to say it is NOT a confidence primitive). The paper's only genuinely novel contributions are: (a) **endpoint normalization** `Z_Mag = ‖hᴸ − h⁰‖₂`, `Z_Ang = arccos(h⁰·hᴸ / …)` that converts absolute changes to relative-to-endpoint, (b) the **complex-plane combination metric CoE-C** producing a single scalar, and (c) the **UQ / correctness-classifier application**. Contribution (c) is UQ-bearing → MUST beat the conformal-naive floor (Plan 340, not yet shipped) before any GOAT claim. Until that gate runs, this is Gain at best.

**Distilled for katgpt-rs (modelless, inference-time):**
Two small additions to the existing `latent_trajectory_geometry` module that turn its raw diagnostic fields into a single calibrated self-eval scalar: a `relative_to_endpoints` normalization fold and a `complex_plane_combine` reduction. Both are zero-alloc, O(L·d), pure math — no game IP, no chain IP. The UQ application is gated behind the conformal floor rule and cannot be claimed until Plan 340 ships; the open primitive value is "an alternative CLR vote arm, conformal-gated, sigmoid-projected" — a fusion with CLR (284), not a new capability class.

---

## 1. Paper Core Findings

### 1.1 Definition — Chain-of-Embedding (CoE)

For an `L`-layer transformer producing sentence hidden states `hˡ = (1/T) Σₜ zₜˡ`, the CoE is the ordered chain `H = h⁰ → h¹ → … → hᴸ`. Two adjacent-pair features:

- **Magnitude change** `M(hˡ, hˡ⁺¹) = ‖hˡ⁺¹ − hˡ‖₂`
- **Angle change** `A(hˡ, hˡ⁺¹) = arccos(hˡ⁺¹·hˡ / (‖hˡ⁺¹‖·‖hˡ‖))`

Whole-trajectory features are the per-pair means, **normalized by the endpoint-pair features** to suppress sample-position bias:

```
Mag(H) = (1/L) Σₗ M(hˡ, hˡ⁺¹) / M(h⁰, hᴸ)
Ang(H) = (1/L) Σₗ A(hˡ, hˡ⁺¹) / A(h⁰, hᴸ)
```

`Z_Mag = M(h⁰, hᴸ)`, `Z_Ang = A(h⁰, hᴸ)` are the scaling factors — *relative* changes w.r.t. the input↔output distance.

### 1.2 The two combination metrics

- **CoE-R** (real-space, eq. 5): linear sum, `CoE-R(H) = (1/L) Σₗ ( M/Z_Mag − A/Z_Ang )`. Magnitude *positively* monotone with correctness; angle *negatively* monotone (correct answers have *less* semantic instability per step). Linear, simple, sensitive to outliers.
- **CoE-C** (complex-plane, eq. 7): treat each pair as complex point `C = M·exp(i·A)` with magnitude `M` and argument `A`; average the real and imaginary parts separately, take the magnitude of the result. Provably ≤ CoE-R under feature increments (paper §B.2.2 — `∆F_C(Lᵢ) ≤ ∆F_R(Lᵢ)`), hence **more robust to outliers**. Authors' recommended variant.

### 1.3 Findings (correct vs incorrect CoE)

- **Correct** samples have *more significant* `Mag` (more convoluted, detour-through-semantic-space trajectory) and *less significant* `Ang` (more semantically stable per step).
- **Incorrect** samples have *less* `Mag` and *more* `Ang` (intermediate states drift near the origin → larger angles between adjacent pairs).
- Visualization (PCA-projected): correct CoEs detour *away* from the straight input→output line; incorrect CoEs hug the origin.

### 1.4 Empirical results (Table 1)

- 7B–72B models, 4 domains (Math, Reasoning, Knowledge, Understanding), 6 datasets (GSM8K, MATH, CommonsenseQA, TheoremQA, MMLU, Belebele).
- **AUROC** average gain over best baseline: +9.83% (Math) / +8.36% (Reasoning) / +7.78% (Knowledge) / +7.23% (Understanding).
- More discriminative on **high-difficulty** tasks (MATH vs GSM8K, TheoremQA vs CommonsenseQA) — Figure 5.
- **10× model-scale robust** — gain holds (often widens) from 7B to 70B+.
- **Multilingual** — gain holds across 11 MGSM languages including low-resource (bn).

### 1.5 Efficiency (Table 3)

Excluding base inference: SoftMax computation = 10.32 s ± 3.51 s; **CoE computation = 1.12e-03 s** (millisecond-scale). Only addition, multiplication, and trig. No vocab-sized softmax. Massive advantage over sampling-based methods (which need ≥1 extra full forward pass).

### 1.6 Theoretical analysis (§5)

- Monotonicity: in practice (>98% of cases) `Aᵢ ∈ [0, π/2]`, so both `∆F_C(Lᵢ) > 0` and `∆F_C(αᵢ) < 0` — CoE-C monotonicity matches CoE-R's analytic monotonicity.
- Robustness proof: `∆F_C(Lᵢ) ≤ ∆F_R(Lᵢ) = ∆L/n` — CoE-C increment is bounded above by CoE-R's, so it is *strictly more outlier-robust*. (Eq. 12 + B.2.2 derivation.)

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (mandatory per skill §Workflow step 2)

| Paper vocabulary | Codebase vocabulary |
|---|---|
| "chain of embedding" / "CoE" / "trajectory of hidden states" | `LatentTrajectoryGeometry`, `from_states`, `latent_trajectory_geometry` (Plan 342 / Research 324); `classify_chain` (depth-invariance, 286) |
| "magnitude change" `M` / `Mag(H)` | `LatentTrajectoryGeometry::length` (Σ ‖Δ‖₂ — the *unnormalized* total; paper normalizes per-step and sums) |
| "angle change" `A` / `Ang(H)` / turning angle | `LatentTrajectoryGeometry::mean_curvature` (mean arccos of consecutive displacement vectors) |
| "cosine similarity between adjacent states" | `LatentTrajectoryGeometry::min_adjacent_cosine` |
| "endpoint scaling factor" `Z_Mag`, `Z_Ang` | **MISSING** — the ship uses absolute length, not relative-to-endpoints |
| "CoE-R" / "CoE-C" combination | **MISSING** — the ship returns raw fields; no single-scalar combination |
| "self-evaluation" / "output-free" / "label-free correctness" | CLR `clr_vote` (Plan 284 — `(mean_m v_k,m)^5` reliability gate, sigmoid-style); **NOT** trajectory-based |
| "confidence" / "calibration" / "AUROC" / "FPR95" / "AUPR" | CLR bench (`.benchmarks/284_clr_goat.md` — ECE 0.0087, +78pp over majority); conformal floor (Plan 340, pending) |
| "trajectory geometry distinguishes correct from incorrect" | "coherence-as-correctness-signal" / "claim verifier" / "Salience Tri-Gate"; depth-invariance `DepthInvarianceKind` (286) |
| "intermediate state near origin → larger angle" | "attractor basin ping-pong" (324 §2.1, oscillation signature — `mean_curvature ≈ π`); "magnitude hygiene" (286 / `MagnitudeRegularizedResidual`) |

### 2.2 Three-layer prior-art audit (notes + code + vocabulary)

**Notes layer (`.research/` across all 5 repos):**
- **Research 324** (`katgpt-rs/.research/324_Trajectory_Geometry_Transformer_Layers.md`) — Pandey 2026 paper, ships the *exact* three metrics (length, mean_curvature, min_adjacent_cosine) as `LatentTrajectoryGeometry`. Verdict: Gain. Explicitly punted the UQ application: "the only transferable piece is a small reusable `LatentTrajectoryGeometry` diagnostic struct".
- **Research 286** (`katgpt-rs/.research/286_Attention_Drift_Depth_Invariance_Diagnostic.md`) — `DepthInvarianceDiagnostic::classify_chain`. Super-GOAT (the *root-cause* counterpart — magnitude slope, cosine step, effective rank slope). The diagnostic's `mean_cos_step` is the structural twin of Wang 2024's `min_adjacent_cosine`.
- **Research 053** (CNA) — neuron-level attribution. Cited by Research 325 §2.5 as the cousin of "chain-of-embedding trajectory".
- **Research 325 §7.2 G3** — *explicitly* flagged Wang 2024 as a modelless-candidate gap with closest cousins 286 and 053. **This is the canonical failure-mode prophylactic: the survey already pre-classified the paper.**
- **Research 322** / **Plan 340** / **Issue 010** — conformal floor rule. Any UQ-bearing primitive MUST beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with m=1 on CRPS / coverage / Winkler. **The floor does not ship yet** (Plan 340 Phase 1 pending) — so no UQ claim from CoE can be validated today.

**Code layer (`katgpt-rs/crates/katgpt-core/src/latent_trajectory_geometry.rs`):**
- `LatentTrajectoryGeometry { length, mean_curvature, min_adjacent_cosine, n_steps }` — **COMPLETE**, opt-in feature `latent_trajectory_geometry`, GOAT gate passed (Plan 342 Phase 1–3, `.benchmarks/342_latent_trajectory_geometry_gate.md`).
- `from_states(&[&[f32]])` — streaming fold, **zero allocation**, O(L·d), 3.04 µs at HLA scale (100×8).
- `bifurcation_ratio(a, b) -> BifurcationResult { separation_ratio, onset_step, final_separation }` — paper's Finding 3 analog.
- `fast_acos` — Nvidia polynomial, ~3 ns/call vs stdlib ~80 ns/call.
- **Module docstring explicitly says** (lines 27–33): *"NOT probabilities / confidence scores / predictive intervals. The 'Report the Floor' conformal-naive rule does NOT apply."* — Research 324 deliberately refused the UQ framing.

**Conclusion:** The three geometric primitives are identical to the shipped module. The only genuinely missing pieces are: (a) endpoint normalization, (b) CoE-C combination metric, (c) UQ framing + classifier application.

### 2.3 The transferable primitive (open, generic, modelless)

Three small additions to `latent_trajectory_geometry.rs`, all pure math, zero-alloc:

```text
// 1. Endpoint-normalized variant of the existing struct.
LatentTrajectoryGeometryRelative {
    rel_length: f32,         // (Σₗ ‖Δ‖) / ‖hᴸ − h⁰‖     — paper Mag(H)
    rel_mean_curvature: f32, // (Σₗ A) / A(h⁰, hᴸ)        — paper Ang(H)
    n_steps: u16,
}

// 2. Single-scalar self-eval score (complex-plane combination).
fn coe_c_score(states: &[&[f32]]) -> f32
//   = (1/L) · | Σₗ (M/Z_Mag)·(cos(A) + i·sin(A)) |
// O(L·d), zero alloc, single streaming pass.

fn coe_r_score(states: &[&[f32]]) -> f32
//   = (1/L) · Σₗ ( M/Z_Mag − A/Z_Ang )
// Provided for completeness / ablation; CoE-C is the recommended variant.
```

**Crucial non-claim:** `coe_c_score` returns a *raw geometric score*, **not** a calibrated probability. To convert it into a correctness probability (the paper's UQ application), it MUST be wrapped in a conformal calibrator (Plan 340) — and the calibrated output MUST beat the conformal-naive floor on CRPS / coverage / Winkler before any UQ selling-point claim. Until Plan 340 ships, the score is a *ranking signal*, not a UQ distribution.

### 2.4 Latent-space reframing (mandatory per skill §Workflow step 3)

The paper operates on per-layer transformer hidden states. Reframings onto the seven Super-GOAT factory modules:

**(a) HLA per-NPC latent state (8-dim, `evolve_hla`):** Apply `coe_c_score` to the HLA evolution history across `K` ticks → a per-NPC "cognitive-arc-curvature score". The paper's "correct CoEs detour through semantic space; incorrect hug the origin" maps to: *committed NPC personality produces detour-rich HLA trajectories; degenerate/looping NPCs produce low-curvature trajectories hugging the zero vector*. This is the *same* oscillation-vs-commitment signal Research 324 §2.3 already uses `mean_curvature ≈ π` to detect — but CoE-C folds in the magnitude normalization, which is more robust on real HLA trajectories.

**(b) `latent_functor/` applications:** Apply `coe_c_score` to a single NPC's functor-application sequence within one tick → a "functor-arc-coherence score". The paper's monotonicity result (`∆F_C(Lᵢ) ≤ ∆F_R(Lᵢ)`) means CoE-C is *strictly more robust* than the linear-combination fallback when one functor application produces an anomalous displacement. This is a marginal robustness improvement over the existing `mean_curvature` signal.

**(c) `cgsp_runtime/` curiosity:** CoE-C is *not* a natural fit — curiosity is already driven by `belief_mass_divergence` (Plan 314) → `divergence_to_curiosity` sigmoid gate, which is a divergence/boundary-flux operator, not a trajectory-length/angle operator. **DEC codifferential δ is the right tool there, not CoE.**

**(d) DEC Stokes-calculus reframing (the interesting one):** Can the CoE trajectory be framed as a *cochain* whose exterior derivative `d` detects "reasoning inconsistency"?

- Treat the per-layer state sequence `(h⁰, …, hᴸ)` as a **rank-0 cochain** (one value per layer-node).
- The exterior derivative `d` applied rank-0 → rank-1 produces the per-edge displacement `Δˡ = hˡ⁺¹ − hˡ`. This *is* the paper's magnitude-change vector. `‖dω‖` is the paper's `Mag(H)` (up to normalization).
- The codifferential `δ` applied rank-1 → rank-0 produces the discrete *divergence* — `δΔˡ = Δˡ⁺¹ − Δˡ` — which measures the *acceleration* of the trajectory, not the angle. **This is NOT what the paper computes.** The paper's angle feature `A(hˡ, hˡ⁺¹) = arccos(Δˡ · Δˡ⁺¹ / …)` is the *discrete turning angle*, which is a rank-1→rank-1 measure on consecutive displacement vectors, not a DEC operator.
- The Hodge decomposition `ω = dα ⊕ β ⊕ δγ` would split the trajectory into exact (gradient-like, "straight") + harmonic (oscillatory) + coexact (solenoidal, "circling") components. The paper's "correct CoEs detour; incorrect hug origin" maps loosely onto the harmonic-component magnitude — but this is a *spectral* claim, not the geometric-trio the paper actually computes.

**DEC reframing verdict:** *Partially maps but does not strengthen the primitive.* The magnitude feature is exactly `‖dω‖`. The angle feature is *not* a DEC operator and does not gain anything from being re-cast as one. The Hodge decomposition would be a *richer* diagnostic than CoE-C (three components vs one scalar), but it is also strictly more expensive (Plan 314 wrappers + `hodge_decompose`) and is the subject of Research 296, not this paper. **Conclusion: ship CoE-C as a thin combination metric on top of the existing `LatentTrajectoryGeometry` fields; do NOT route through DEC.**

**(e) LatCal / NeuronShard:** not applicable — CoE operates on per-layer transformer hidden states, which are neither committed-to-chain (they are transient per-token) nor stored as shards. The UQ output, IF a future conformal wrapper makes it calibrated, could be a per-NPC CLR vote arm (see §2.5) — that crosses no sync boundary (CLR is local).

### 2.5 Fusion — the novel combination

**Strongest fusion — CoE-C as a CLR vote arm, conformal-gated:**

CLR (Plan 284, shipped, DEFAULT-ON, ECE 0.0087) ranks trajectories by `(mean_m v_k,m)^5` — a sigmoid-style reliability gate on claim embeddings projected onto direction vectors. The current CLR vote takes K claims, projects each onto M direction vectors, sharpens with a 5th-power gate. **CoE-C is a *trajectory-geometry* signal orthogonal to CLR's *static-embedding-magnitude* signal:** CLR sees "how strongly does this claim align with verified directions"; CoE-C sees "how committed vs degenerate is the latent trajectory that produced this claim". A fusion that adds `coe_c_score` as a **second vote arm**, normalized and conformal-gated alongside the existing CLR arm, could catch the failure mode where a claim embedding looks strong (CLR high) but the underlying trajectory was degenerate (CoE-C low) — and vice versa.

**This fusion is Gain-tier today and GOAT-tier only IF:**
1. Plan 340 ships the conformal floor.
2. CoE-C (conformal-wrapped) beats the conformal-naive floor on a self-eval benchmark (CRPS / coverage / Winkler on the LLM correctness target).
3. CLR+CoE-C fusion beats CLR-alone on the same benchmark.

If all three hold → CLR gains a second orthogonal arm, the GOAT gate re-runs, and the winner is promoted. This is the only fusion worth a plan, and only behind a feature flag with explicit conformal-floor comparison. It is NOT a Super-GOAT — CLR is the capability class; CoE-C is an additional signal.

**Why not a Super-GOAT fusion:**
- The geometric primitives ship (324).
- The UQ application is exactly what 324 punted on — picking it up is *completing the existing primitive's roadmap*, not creating a new class.
- The CLR capability class is the selling point; CoE-C is a marginal signal inside it.
- No new moat: "we use trajectory geometry as a self-eval signal" is a recognizable ML technique (the paper has 7 LLMs × 4 domains × public code at github.com/Alsace08/Chain-of-Embedding), not a defensible product feature.

### 2.6 What does NOT transfer

- **Closed-source models** — paper's CoE needs white-box access to hidden states. We have white-box access to our own inference stack, so this is fine for per-NPC use, but it limits direct adoption in cloud-LLM pipelines.
- **Per-token vs sentence-level pooling** — paper averages over T tokens per layer to get `hˡ`. For per-NPC use, the analogous "trajectory" is across ticks (not tokens within a tick) or across functor applications within a tick. The 22%-of-depth bifurcation onset (paper Finding 3, transformer-specific) does not transfer.
- **The CoE-R variant** — strictly dominated by CoE-C (paper §B.2.2 proof). Ship CoE-R only as ablation baseline.
- **OOD-vs-TruthfulQA trained-classifier comparison** (Table 4) — paper beats ITI/MIND on OOD by 20–30 points. This is a paper-side argument against label-based probing classifiers, not a modelless primitive we ship.

---

## 3. Verdict

| Question | Answer |
|---|---|
| Training-only? | NO — pure inference-time geometric computation on hidden states. |
| No prior art? | **NO.** Research 324 (`LatentTrajectoryGeometry`) ships the exact three metrics (length, mean_curvature, min_adjacent_cosine). Research 286 ships the structural twin (`classify_chain`). Research 325 §7.2 G3 *pre-classified* this paper as a gap with cousins 286 and 053. |
| New capability class? | **NO.** The paper's only novel pieces are (a) endpoint normalization, (b) the complex-plane CoE-C combination metric, (c) the UQ application. (a) and (b) are thin additions to an existing primitive; (c) is exactly the UQ framing 324 punted on, and is *blocked* by the conformal floor rule. |
| Product selling point? | **Weak.** "Our NPCs use trajectory geometry as a self-eval signal" is recognizable ML, not a defensible moat. CLR (284) is the selling point; CoE-C would be a marginal second arm inside it. |
| Force multiplier? | **Loose** — connects to CLR (284), conformal overlay (340), latent_trajectory_geometry (342), depth-invariance (306). All are existing primitives; this paper adds a thin combination layer. |

**Tier: Gain.**

**One-line reasoning:** Every geometric primitive the paper uses already ships as `latent_trajectory_geometry.rs` (Plan 342 / Research 324); the only novel pieces — endpoint normalization, the complex-plane CoE-C combination, and the UQ application — are either thin additions to an existing primitive or UQ-bearing claims blocked by the conformal floor rule (Plan 340, not yet shipped); no new capability class, no new moat, no Super-GOAT guide.

**Routing:** Open primitive additions → `katgpt-rs` (extend `latent_trajectory_geometry.rs` with `coe_c_score`, `coe_r_score`, `LatentTrajectoryGeometryRelative` — pure math, no game IP). Plan only, opt-in feature flag, no private guide, no Super-GOAT.

### Honest verdict on the UQ angle (per the floor rule)

The paper's headline claim — AUROC 60–85% on LLM correctness classification — is a UQ-bearing claim. Under the "Report the Floor" policy (Research 322 / Plan 340 / Issue 010), any primitive claiming calibrated correctness probabilities MUST beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1) on CRPS / coverage / Winkler. **The floor does not ship yet** (Plan 340 Phase 1 pending). Therefore:

- **Today:** CoE-C cannot be claimed as a UQ primitive. The raw `coe_c_score` is a *ranking signal*, not a calibrated distribution.
- **After Plan 340 ships:** a follow-up plan can wrap `coe_c_score` in `ConformalIntervalCalibrator` and benchmark against the floor. If it beats the floor → UQ claim is valid, primitive is a CLR-orthogonal vote arm (Gain promoted toward GOAT only if CLR+CoE-C fusion beats CLR-alone). If it cannot beat the floor → CoE-C is not adding UQ value and the selling point must be reframed (Issue 010 §failure-mode).
- This dependency is recorded but **not enforceable** until Plan 340 lands. Track in `.issues/010` alongside the other grandfathered UQ primitives.

---

## 4. Caveats and explicit non-claims

1. **The geometric primitives are NOT novel here.** Research 324 + Plan 342 ship `LatentTrajectoryGeometry` with the exact three fields (length / mean_curvature / min_adjacent_cosine). The ship's GOAT gate (`.benchmarks/342_latent_trajectory_geometry_gate.md`) already validates the primitive at HLA scale (3.04 µs at 100×8, oscillation-vs-commitment curvature gap +2.986 rad). Re-distilling the metrics under "CoE" vocabulary would be a duplicate note.
2. **The UQ claim is gated by the conformal floor rule.** Per Issue 010, no UQ selling-point claim is valid until Plan 340 Phase 1 ships `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` AND a benchmark proves CoE-C (conformal-wrapped) beats the floor. This note does NOT make any UQ claim. The `coe_c_score` open primitive returns a raw geometric score, not a probability.
3. **The DEC reframing does not strengthen the primitive.** The magnitude feature is exactly `‖dω‖` (DEC exterior derivative on a rank-0 cochain), but the angle feature is *not* a DEC operator. The Hodge decomposition (Research 296) would be a richer but more expensive diagnostic — orthogonal to this paper, not a fusion target here.
4. **Paper is observational for the headline claim.** The paper empirically shows AUROC 60–85% on real LLMs; it does not prove *why* correct CoEs detour and incorrect ones hug the origin. Any per-NPC transfer is a modelless bet, not a paper-endorsed guarantee — same caveat as Research 324 §4.1.
5. **Per-token trajectory is transformer-specific.** The paper's sentence-pooling over T tokens has no direct per-NPC analog. The closest analog is HLA evolution across K ticks (intra-entity) or functor-application sequence within one tick (intra-stage). Both need their own calibration — the paper's 0.5-rad magnitude / π-angle thresholds do not transfer verbatim.
6. **Not Super-GOAT.** Per the workflow's "no candidate escape hatch" rule, no Super-GOAT guide is created in riir-ai/riir-chain/riir-neuron-db. If a future CLR+CoE-C fusion gate proves a measurable CLR win AND the conformal floor is beaten, this can be re-evaluated at that time — file an issue, do not pre-claim.

---

## TL;DR

Gain-tier paper. The geometric trio (length / curvature / cosine) already ships as `latent_trajectory_geometry.rs` (Plan 342 / Research 324). The paper's only novel pieces are endpoint normalization + the complex-plane CoE-C combination metric + the UQ application — the first two are thin additions to the existing primitive, the third is UQ-bearing and blocked by the conformal floor rule (Plan 340, pending). Fusion candidate: CoE-C as a CLR (284) second vote arm, conformal-gated — but only worth a plan after Plan 340 ships AND the floor is beaten. Not Super-GOAT — no new capability class, no new moat, no private guide.
