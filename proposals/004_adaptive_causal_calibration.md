# Proposal 004 — Adaptive Causal Calibration: cheap-proxy-escalate for RTPurbo head-importance scoring

Status: **proposed** (not yet implemented — deferred to a plan only after G1/G2 validation)
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Fusion of: Plan 353 (`HeadSubstitutionGate` cheap-proxy→expensive-validation pattern) × Plan 358
(`CausalHeadImportance` causal scorer) × R086 (RTPurbo calibration slot)
Related: [Research 362](../.research/362_HydraHead_Causal_Head_Importance_Hybrid_Attention.md),
[Plan 358](../.plans/358_causal_head_importance_calibration.md),
[Plan 353](../.plans/353_Program_Synthesized_Head_Primitive.md)

## TL;DR

Ship a third `CalibrationMode` variant — **`AdaptiveCausal`** — that uses a cheap
observational proxy to detect *bystander suspects* and escalates to the expensive
causal patching (Plan 358) only on those `k` suspects, instead of on all `n_heads`.
When no suspects exist the mode degenerates to `AttentionMass` cost; when suspects
exist it pays causal cost only on `k ≈ 4–8` heads, not `n_heads × n_samples`.

**This is a `katgpt-rs` invention, not a distillation of HydraHead (arXiv:2606.20097).**
HydraHead supplies the causal scorer (Plan 358, already shipped). The adaptive
escalation scheme and the OV-circuit cheap proxy proposed here are **our design**.
The proxy is an **unvalidated mech-interp hypothesis** — promotion to default is
blocked on empirical validation that can only run in `riir-engine` (the patched
forward pass is explicitly out of scope for katgpt-rs per Plan 358 Risk #1/#4).

What lands now: the open decision-rule primitive (pure fn over caller-supplied
`(attention_mass, output_norm)` pairs — leaf-clean, same pattern as the existing
causal scorer). What is deferred: G1/G2 validation (needs a real transformer
forward), and therefore any promotion to default.

## The problem this solves

Plan 358 shipped `CausalHeadImportance` opt-in. It is strictly stronger than
RTPurbo's `AttentionMass` calibration — G2 proved it filters correlated-bystander
heads that attention-mass wrongly promotes (causal Jaccard 1.0 vs attention-mass
Jaccard 0.0 once bystanders exist; bystanders attend 0.92 > load-bearing 0.78 and
displace all load-bearing heads from the top-K). The partition step itself is even
*cheaper* than attention-mass at high head counts (n=144: 1527 ns causal vs slower
attention-mass, because causal does sort+split where attention-mass builds full
`HeadClassification` structs).

But the T4.4 promote/demote decision **kept `AttentionMass` as default** for one
reason: causal score *production* is `O(n_heads × n_calibration_samples)` patched
forward passes — ~10–100× more expensive than RTPurbo's single forward + per-head
mass scan. And real-world bystander prevalence is unknown (G1/G2 were synthetic).
When no bystanders exist, both modes agree (G2 @ 0 bystanders: both Jaccard 1.0),
so paying 10–100× for identical output is pure waste.

**The gap:** there is no mode that gets causal quality *when bystanders exist*
without paying causal cost *even when they don't*. This proposal fills that gap.

## The proposed design — the "adaptive thermal path"

The shape mirrors Plan 353's `HeadSubstitutionGate` exactly: a cheap proxy decides,
an expensive validation runs only when the cheap proxy is borderline.

```
For each head h:
  1. Compute cheap observables:  attention_mass[h] ,  ||OV_out[h]||
  2. Compute bystander-suspect score:  s[h] = attention_mass[h] / ||OV_out[h]||
  3. If s[h] > tau_suspect  →  head h is a SUSPECT (attends to needle, low OV output)
                                 → escalate: run causal patching on h only
     Else                     →  head h is NOT a suspect
                                 → keep its attention-mass ranking (cheap)

If #suspects == 0:
  → partition == attention-mass partition (cost == attention-mass cost)
If #suspects == k (k ≈ 4–8):
  → run causal patching on the k suspects only, re-rank them by IE,
    merge with the non-suspects' attention-mass ranking
  → cost ≈ k × n_samples patched forwards  (vs n_heads × n_samples today)
```

**The win:** when there are no bystanders, the mode pays **0** patched forwards →
same cost as `AttentionMass` → **default-viable**. When bystanders exist, it pays
`k × n_samples` (k≈4–8) instead of `n_heads × n_samples` (n_heads ≈ 16–144).

## The OV-circuit cheap proxy — and why it is principled (but unvalidated)

A *correlated bystander* (HydraHead's term, Research 362 §2.1) is definitionally:
*attends to the needle* (high attention-mass) **but** *projects to zero in the
readout direction* (low output contribution at the answer position). So the ratio

```
s[h] = attention_mass[h] / ||OV_out[h]||
```

is a direct observational proxy for the bystander property: high `s[h]` means
"attends a lot, contributes little" — exactly the bystander signature.

This is standard OV-circuit mech-interp (Elhage et al., "A Mathematical Framework
for Transformer Circuits"): a head's contribution to the residual stream at
position `t` is `OV · attn(·, t)`. A bystander has high `attn(·, needle)` but low
`||OV · attn||` along the readout direction. The proxy reads both terms off a
single forward pass — no patched forwards needed to *detect* suspects, only to
*confirm* them causally.

**Why this is principled, not a hack:** the bystander property *is* the
high-attention-mass / low-OV-output property. The proxy isn't an approximation of
something else — it's a direct measurement of the defining feature. The only
question is whether it has enough precision to avoid flagging non-bystanders
(false positives → unnecessary patched forwards → cost goes back up).

## Honest caveats — READ BEFORE IMPLEMENTING

These are mandatory and non-negotiable. Anyone picking up this proposal must
treat all four as live constraints.

1. **The adaptive scheme is our invention.** HydraHead (arXiv:2606.20097) proposes
   causal head-importance scoring and scale-normalized fusion. It does **not**
   propose an OV-circuit cheap proxy, an adaptive escalate-on-suspects mode, or
   any two-stage cheap-then-expensive calibration. We are designing this. The
   `.docs/` and `.research/` writeups must say so explicitly. Do not attribute
   this to the paper.

2. **The OV-circuit proxy is an unvalidated hypothesis.** It is a *reasonable*
   mech-interp argument (see above), but it has never been measured against
   ground-truth causal IE on a real model. If the proxy has low precision — i.e.
   it flags heads that are *not* actually bystanders — the escalation fires too
   often and the cost win evaporates. **The proposal is worthless without G1.**

3. **Validation requires `riir-engine`.** Computing `||OV_out[h]||` at the
   readout position requires a real transformer forward that exposes per-head OV
   outputs. Computing ground-truth causal IE for proxy validation requires the
   patched-forward pass (selective head-output substitution + downstream-attention
   freezing). Both are explicitly **out of scope for katgpt-rs** (Plan 358 Risk
   #1, Risk #4) — katgpt-rs ships the *scorer*; the *patched forward pass* is the
   caller's responsibility and lives in riir-engine. **G1 cannot run in katgpt-rs.**

4. **Promotion to default is blocked on G1 + G2.** The open primitive can land
   now (it's a pure fn over caller-supplied pairs — leaf-clean). But
   `AdaptiveCausal` MUST NOT become the default `CalibrationMode` until both G1
   (proxy precision on a real model) and G2 (cost reduction holds at production
   head counts) pass empirically. Until then it stays opt-in alongside
   `CausalNecessity`, and `AttentionMass` remains default.

## Fusion lineage (where this comes from)

Research 362 §2.3 already documents the broader fusion
(`R086 × R244 × R353 × HydraHead`) and lists a unified head-importance diagnostic
as the target. But that note describes the *combination of shipped primitives into
a unified diagnostic*, not the *adaptive escalate-on-suspects cost-reduction mode*
proposed here. The adaptive mode is a novel combination of:

| Source | What it contributes |
|---|---|
| Plan 353 (`HeadSubstitutionGate`) | The cheap-proxy → expensive-validation cadence pattern. IoU is the cheap proxy there; the OV-circuit ratio is the cheap proxy here. Same shape, different proxy. |
| Plan 358 (`CausalHeadImportance`) | The expensive validation (causal patching) that the cheap proxy gates. |
| R086 (RTPurbo) | The calibration slot the new mode competes for. The `CalibrationMode` enum (Plan 358 Phase 4) already has `AttentionMass` and `CausalNecessity`; this adds `AdaptiveCausal`. |

**This is not a new capability class.** Research 362 §2.3 is explicit that the
broader fusion is "a measurement-quality upgrade that makes the existing
head-routing decisions more reliable", not a new capability. The adaptive mode is
a **cost-reduction upgrade** on top of that — it makes the causal-quality option
default-viable by avoiding its cost when there's nothing to measure. Verdict
ceiling: **GOAT** (provable cost gain at parity quality), not Super-GOAT.

## GOAT gate

| Gate | Criterion | Where it runs |
|---|---|---|
| **G1 — proxy precision** | On a real transformer, does `attention_mass[h] / \|\|OV_out[h]\|\|` predict causal IE = 0 with precision ≥ 0.8 at recall ≥ 0.9? (i.e. when the proxy flags a suspect, is it actually a bystander ≥ 80% of the time, and does it catch ≥ 90% of real bystanders?) | **riir-engine** — needs real transformer forward for OV outputs + patched forward for ground-truth IE. **Cannot run in katgpt-rs.** |
| **G2 — cost reduction** | At production head counts (n_heads ∈ {16, 64, 144}) and typical bystander fractions (0%, 10%, 25%, 50%), does escalating on k suspects bring total calibration cost within ~2× of pure `AttentionMass`? (Target: `k × n_samples + 1 forward ≪ n_heads × n_samples`.) | riir-engine (real forward cost) + katgpt-rs (partition-step microbench, mirroring Plan 358 G3). |
| **G3 — no-regression** | When bystanders = 0, `AdaptiveCausal` partition must equal `AttentionMass` partition bit-for-bit (no suspects → no escalation → identical ranking). | katgpt-rs unit test. |
| **G4 — alloc-free hot path** | The suspect-detection fn is `#[inline]`, takes `&[(f32, f32)]` (or two `&[f32]`), returns a small fixed-size suspect index set, allocates nothing. | katgpt-rs (by inspection + criterion, mirroring Plan 358 G4). |

**Promotion rule:**
- **G1 + G2 + G3 + G4 all pass** → promote `AdaptiveCausal` to default
  `CalibrationMode`. Demote pure `AttentionMass` to the "no-OV-observable-available"
  fallback (for callers that can't supply per-head OV norms). Pure `CausalNecessity`
  stays opt-in for the "I want full causal patching regardless of cost" regime.
- **G1 fails** (proxy imprecise) → `AdaptiveCausal` stays opt-in or is dropped.
  Do NOT promote. The cost win is contingent on the proxy being right often enough.
- **G2 fails** (suspect count too high in practice) → same: stays opt-in. The win
  requires k ≪ n_heads in production; if real models have ~50% bystanders the
  escalation fires on half the heads and there's no savings.

## What ships now (katgpt-rs) vs deferred (riir-engine)

### Ships now — open primitive, leaf-clean (katgpt-rs)

A pure decision-rule module under
`katgpt-rs/crates/katgpt-core/src/causal_head_importance/` (or a sibling
`adaptive_calibration/`), feature-gated `adaptive_causal_calibration` (opt-in).
Contents:

- `suspect_indices(attention_mass: &[f32], ov_output_norm: &[f32], tau_suspect: f32) -> impl Iterator<Item=usize>` — the cheap proxy. Zero-alloc, `#[inline]`.
- `adaptive_partition(...)` — merges the suspect causal ranking with the non-suspect attention-mass ranking into a single `HeadClassification`. Takes caller-supplied causal scores for the suspects only (caller runs the patched forwards on `suspect_indices` output).
- Unit tests for G3 (no-suspect → identical to attention-mass) and G4 (alloc-free).

**Why this can land before G1:** it's a pure function over caller-supplied pairs,
exactly like the existing `partition_by_causal_score` (Plan 358 T1.6). The caller
(riir-engine) supplies `(attention_mass, ov_output_norm)` from a real forward;
katgpt-rs doesn't need to know how they were computed. This matches the leaf-clean
pattern: the primitive is the *rule*, the caller supplies the *measurement*.

### Deferred — validation (riir-engine / riir-ai)

- The patched-forward-pass implementation to extract per-head OV outputs at the
  readout position (Plan 358 Risk #1 — explicitly out of scope for katgpt-rs).
- The ground-truth causal IE computation for proxy validation (G1).
- The real-model bystander-prevalence measurement (G2's "typical bystander
  fractions" needs real models, not synthetic).
- Track in `riir-ai/.issues/` (or wherever riir-engine tracks follow-ups) —
  **not** in `katgpt-rs/.issues/`, since the work doesn't live in this repo.

### Explicitly NOT shipped by this proposal

- Promotion of `AdaptiveCausal` to default `CalibrationMode`. Blocked on G1+G2.
- Any change to the default behavior of RTPurbo. `AttentionMass` stays default.
- The OV-output extraction itself. That's riir-engine.

## Phased rollout (sketch — a plan would expand this)

### Phase 1 — Open primitive skeleton (katgpt-rs, ships now)
- Add `adaptive_causal_calibration` feature to `katgpt-core` Cargo.toml (opt-in, default-off).
- Implement `suspect_indices` + `adaptive_partition` under `causal_head_importance/` (or new `adaptive_calibration/` mod).
- Unit tests: G3 (no-suspect degenerates to attention-mass), G4 (alloc-free by inspection).
- `cargo test -p katgpt-core --features adaptive_causal_calibration --lib` green.

### Phase 2 — Wire `AdaptiveCausal` into `CalibrationMode` (katgpt-rs, ships now)
- Add `AdaptiveCausal = 2` to `CalibrationMode` enum (`crates/katgpt-types/src/enums.rs`).
- Wire into RTPurbo calibration path: when `AdaptiveCausal` is selected, call `suspect_indices` then escalate via the caller-supplied causal closure on suspects only.
- Document in the enum docstring that `AdaptiveCausal` requires the caller to supply per-head OV norms (unlike the other two modes).
- **Do NOT change `CalibrationMode::default()`** — stays `AttentionMass`.

### Phase 3 — Deferred: G1 + G2 validation (riir-engine)
- Blocked. Not in this proposal's scope. A riir-ai/riir-engine plan owns this.
- Outcome of Phase 3 decides Phase 4.

### Phase 4 — Promotion decision (katgpt-rs, only after Phase 3 passes)
- If G1+G2 pass: flip `CalibrationMode::default()` to `AdaptiveCausal`, demote `AttentionMass` to opt-in fallback, record in `.benchmarks/004_*.md`.
- If G1 or G2 fails: keep `AdaptiveCausal` opt-in, document why, close the proposal as "shipped open primitive, promotion rejected".

## Risks

1. **Proxy precision unknown.** The whole cost win hinges on the OV-circuit ratio
   being a *precise* bystander detector. If it flags many non-bystanders, the
   escalation fires too often and we're back to near-full causal cost. G1 is the
   gate that kills this proposal if it fails. **This is the #1 risk and it cannot
   be evaluated in katgpt-rs.**

2. **Real-model bystander prevalence unknown.** G2 in Plan 358 was synthetic. If
   real game-AI transformers (the ones riir-ai actually runs) have very few
   bystanders, `AdaptiveCausal` degenerates to `AttentionMass` (fine — no harm,
   no gain). If they have ~50%, the escalation fires on half the heads and the
   savings vanish. Either way, the *value* of the proposal is unmeasurable until
   Phase 3.

3. **OV-output extraction cost.** Computing `||OV_out[h]||` at the readout
   position requires materializing per-head OV outputs from a single forward —
   cheap relative to patched forwards, but not free. If it requires modifying the
   forward pass to expose per-head outputs, that's a non-trivial riir-engine
   change. The proposal assumes the caller can supply it; whether that's easy in
   riir-engine is out of scope here.

4. **Tau-suspect tuning.** The threshold `tau_suspect` on the ratio is a
   hyperparameter. Picking it requires the G1 precision/recall curve — you can't
   set it blind. Ship with a documented default but mark it "tune after G1".

5. **Name collision / scope creep.** "Adaptive" is overloaded in this codebase
   (adaptive CoT, adaptive budget, adaptive speculation...). `AdaptiveCausal` is
   precise enough but the `.docs/` writeup must be clear about *what* is adapting
   (the calibration cost, in response to detected bystander suspects).

## Out of scope

- **→ riir-engine**: the patched-forward pass, OV-output extraction, G1/G2
  validation harness on a real transformer.
- **→ riir-ai**: applying the adaptive mode to HLA direction-vector importance
  (Research 362 §2.5(a)) — the *primitive* is open in katgpt-rs; the *application*
  to HLA's 8-dim affect space is a riir-ai follow-up, same as Plan 358's deferral.
- **→ riir-neuron-db**: applying the adaptive mode to NeuronShard dendritic-branch
  importance (Research 362 §2.5(e)).
- **→ riir-train**: nothing. This proposal is modelless end-to-end (the primitive
  is a pure fn; the validation is forward-pass-only). No training dependency.

## References

- [Research 362 — HydraHead distillation](../.research/362_HydraHead_Causal_Head_Importance_Hybrid_Attention.md) (§2.1 bystander definition, §2.3 fusion lineage, §3.3 GOAT gate table)
- [Plan 358 — CausalHeadImportance (shipped, opt-in)](../.plans/358_causal_head_importance_calibration.md) (the expensive validation this proposal gates)
- [Plan 353 — HeadSubstitutionGate](../.plans/353_Program_Synthesized_Head_Primitive.md) (the cheap-proxy→expensive-validation cadence pattern this fuses with)
- [R086 — RTPurbo](../.research/086_RTPurbo_Retrieval_Head_Sparse_Attention.md) (the calibration slot)
- Elhage et al., "A Mathematical Framework for Transformer Circuits" (the OV-circuit mech-interp the proxy is grounded in — not in the corpus, standard reference)
- HydraHead: [arXiv:2606.20097](https://arxiv.org/abs/2606.20097) (the causal scorer source — does NOT propose the adaptive mode)

## TL;DR

Ship a third `CalibrationMode::AdaptiveCausal` that uses a cheap OV-circuit proxy
(`attention_mass / ||OV_out||`) to detect bystander suspects and escalates to
Plan 358's causal patching only on those k suspects. When no suspects exist it
costs the same as `AttentionMass`; when k ≈ 4–8 suspects exist it costs `k ×
n_samples` instead of `n_heads × n_samples`. **The adaptive scheme and the proxy
are our invention, not HydraHead's; the proxy is an unvalidated hypothesis; G1
(precision) and G2 (cost) can only be measured in riir-engine on a real
transformer.** The open primitive (pure fn over caller-supplied pairs) can land
in katgpt-rs now behind `adaptive_causal_calibration` feature (opt-in); promotion
to default is blocked on G1+G2. Verdict ceiling: GOAT (cost-reduction at quality
parity), not Super-GOAT (no new capability class per Research 362 §2.3).
