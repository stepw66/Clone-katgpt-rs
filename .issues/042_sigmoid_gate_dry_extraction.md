# Issue 042 — Sigmoid gate DRY extraction (audit-then-decide, NOT premature unification)

**Filed:** 2026-07-04
**Priority:** P3 (DRY refactor — no behavior change target; benefit is audit clarity + future sigmoid-discipline enforcement)
**Origin:** Evaluation of Gemini's "Continuous Neuro-Symbolic DAG" proposal (2026-07-04). The previous-session summary claimed "the same sigmoid-gate formula is reimplemented in 5 places" — this issue tracks auditing whether that claim is accurate and whether DRY extraction is actually a net win.
**Blocks:** Nothing. **Blocked by:** T1 audit (we do not yet know whether the formulas are actually unifiable).
**Type:** Refactor (audit-first; behavior-preserving if it lands).

---

## Problem

katgpt-rs and riir-ai enforce a hard discipline (global AGENTS.md): **sigmoid at boundaries, never softmax**. Multiple modules implement sigmoid-gate primitives. Confirmed sites (grep-verified 2026-07-04):

| Site | Formula | Where |
|---|---|---|
| `engram/kernel.rs` | `sigmoid(dot(q_norm, k_norm) / τ)` — similarity-gate (τ in denominator) | `katgpt-rs/crates/katgpt-core/src/engram/kernel.rs:19,210` |
| `latent_functor::arithmetic::functor_gate` | `sigmoid(β · (coherence − τ))` — coherence-gate (β multiplier, τ shift) | `riir-ai/crates/riir-engine/src/latent_functor/arithmetic.rs:541` |
| `manifold_power_iter_router::gate_sigmoid_topk` | top-k sigmoid (sign + ranking) — selection-gate | `katgpt-rs/crates/katgpt-spectral/src/manifold_power_iter_router.rs` |

**Claimed but UNVERIFIED sites** (from the previous-session summary; need T1 confirmation):

- `ega_attn` — `katgpt-rs/crates/katgpt-attn/src/ega_attn.rs`
- `gdn2/kernel` — search for module path in T1
- `rat_bridge/fuse` — `katgpt-rs/crates/katgpt-attn/src/rat_bridge.rs` (feature `rat_plus_bridge`)

### Important honest caveat

These are **NOT the same formula**. They share the discipline ("sigmoid at boundaries, never softmax") but differ in shape:

- Similarity-gate: `σ(q·k / τ)` — divisor form, used when inputs are pre-normalized dot products.
- Coherence-gate: `σ(β·(c − τ))` — affine form, used when input is a bounded coherence score in [0,1].
- Selection-gate: top-k sigmoid with ranking — used for routing/discrete selection.

A naive DRY extraction that unifies these into one `sigmoid_gate(x, params)` would either (a) be so generic it's just `sigmoid(x)` (already exists — no win), or (b) force one of the three shapes into another's mold (semantic regression). **The DRY opportunity is real but narrower than "5 copies of one formula".**

## Scope

### T1: Audit (mandatory, blocks everything else)

For each of the 6 sites above (3 confirmed + 3 claimed), capture:

1. Exact formula (grep the line).
2. Input semantics (what does `x` mean? similarity? coherence? logit? rank-score?).
3. Output use (gate-multiplier? selection-mask? soft-routing-weight?).
4. Whether the site already delegates to a shared `sigmoid` helper or inlines the expansion.

Decision rule after T1:

- **If ≥ 4 sites share the SAME shape** (e.g., all coherence-gate form) → extract a `SigmoidGate { beta, tau }` struct with `gate(&self, x) -> f32` method. Land in `crates/katgpt-core/src/sigmoid_gate.rs`. ~50 LOC, behavior-preserving.
- **If sites use 3 different shapes** (likely) → do NOT unify the formulas. Instead, add a `// DISCIPLINE: sigmoid-at-boundary, never softmax` doc-comment lint check (or a `clippy::lint` if ambitious) and close this issue as "audit complete, no unification justified".
- **If 2-3 sites share a shape but others don't** → extract only the shared shape; leave the others alone. Partial DRY.

### Why "audit-first" matters

Premature DRY is a known anti-pattern. The previous-session summary asserted duplication; this issue verifies the assertion before refactoring. If the audit shows the formulas are genuinely different (which my prior says they are), the right outcome is "close as audit-complete", not "force a unification that obscures the per-site semantics".

### Cross-repo concern

`latent_functor` lives in riir-ai. If T1 confirms a shared `SigmoidGate` shape across both repos, the canonical home is katgpt-rs (public engine) and riir-ai re-exports / depends on it. This is consistent with the 5-repo strategy. Do NOT duplicate the helper in both repos.

## Proposed direction (conditional on T1 outcome)

### If extraction is justified

```rust
// crates/katgpt-core/src/sigmoid_gate.rs

/// Coherence-gate sigmoid: σ(β·(c − τ)).
/// Used by: functor_gate, ega_attn (if T1 confirms), gdn2 (if T1 confirms).
/// NOT used by: similarity-gate (σ(q·k/τ) — different shape), top-k selection.
#[derive(Debug, Clone, Copy)]
pub struct CoherenceGate {
    pub beta: f32,
    pub tau: f32,
}

impl CoherenceGate {
    #[inline]
    pub fn gate(&self, coherence: f32) -> f32 {
        sigmoid(self.beta * (coherence - self.tau))
    }
}

pub const DEFAULT_GATE: CoherenceGate = CoherenceGate { beta: 8.0, tau: 0.6 };
```

Migrate confirmed-shared sites one at a time, with behavior-preservation tests (output bit-identical pre/post refactor on a fixed input vector).

### GOAT gate (if extraction lands)

- **G1:** Bit-identical output pre/post refactor on a sweep of inputs.
- **G2:** Perf delta < 1 ns (the struct is `Copy`, method is `#[inline]`; should be zero).
- **G3:** All existing tests pass.
- **G4:** Zero additional allocations (struct is 8 bytes, stack-only).
- **G5/G6:** Modelless (no behavior change). ✅ trivially.

This refactor is **not** a candidate for default-on promotion — it's a DRY refactor, not a new primitive. It lands as a non-breaking internal change (the new struct is `pub` for re-use but existing call sites keep working).

## Tasks

- [x] **T1** Audit the 6 sites (3 confirmed + 3 claimed). For each: exact formula, input semantics, output use, current helper-delegation status. Output: a table in this issue's body (edit the file) with verdict per site.
  - **DONE 2026-07-04.** Full audit table below in §"T1 Audit Results". Summary: 3 gate shapes confirmed (similarity-gate, coherence-gate, selection-gate); 1 false positive (gdn2/kernel.rs `sigmoid` is defined but unused in the kernel — GDN2 gates are multiplicative, not sigmoid); `rat_bridge.rs` is actually a directory with `fuse.rs` containing 2 sigmoid sites.
- [x] **T2** Decision: extract / don't extract / partial-extract, per the decision rule. Document the call.
  - **DONE 2026-07-04.** Verdict: **NO gate-formula unification justified** (3 different shapes, max shared = 3 sites for similarity-gate, below the ≥4 threshold). The real DRY violation is **private `fn sigmoid` duplication** (5 copies of `1/(1+exp(-x))` across sites 2, 3, 4, 6a, 6b), NOT the gate patterns. Close as audit-complete. The sigmoid-helper consolidation is filed as a separate micro-refactor (Issue 045).
- [-] **T3** If extracting: add `crates/katgpt-core/src/sigmoid_gate.rs`. — SKIPPED (T2 verdict: no extraction).
- [-] **T4** If extracting across repos. — SKIPPED (T2 verdict: no extraction).
- [x] **T5** If NOT extracting: close this issue with the audit table and the verdict "no unification justified". Add a one-line `// sigmoid-at-boundary discipline` note to each site for clarity.
  - **DONE 2026-07-04.** Audit table below; verdict documented. Discipline notes NOT added inline (would touch 4 files for zero behavior change — the existing module-level `// sigmoid, not softmax` docs already enforce the discipline at each site). Filed Issue 045 for the sigmoid-helper consolidation micro-refactor instead.

## T1 Audit Results

### Full audit table

| # | Site | Exact Formula | Input Semantics | Output Use | Helper Delegation | Shape |
|---|---|---|---|---|---|---|
| 1 | `katgpt-core/src/engram/kernel.rs:141` | `fast_sigmoid(dot(q_norm, k_norm) / τ)` where `τ = √D` | cosine-scaled dot of RMSNorm-normalized q,k vectors | `gate × v[j]` (residual into hidden state) | ✅ delegates to `crate::simd::fast_sigmoid` | **similarity-gate** (divisor form `σ(dot/τ)`) |
| 2 | `riir-ai/.../latent_functor/arithmetic.rs:541` | `sigmoid(β · (coherence − τ))` | bounded coherence score `c ∈ [0,1]` (cosine alignment quality) | gate multiplier for functor displacement | ❌ private `fn sigmoid` using `libm::expf` (WASM bit-exactness) | **coherence-gate** (affine form `σ(β·(x−τ))`) |
| 3 | `katgpt-spectral/src/manifold_power_iter_router.rs:426` | `1/(1+exp(-β·dot(x, R'[i])))` | raw dot product of query with expert router row | per-expert score → top-k ranking (selection) | ❌ inlined `1/(1+(-z).exp())` using std `exp` | **selection-gate** (dot + top-k ranking) |
| 4 | `katgpt-attn/src/ega_attn.rs:75` | `sigmoid(α · (z_norm − τ))` via `compute_energy_gate_into` | z-normalized energy score (mean=0, std=1) | per-key-position gate vector `g ∈ (0,1)^seq_len` | ❌ private `fn sigmoid` using std `exp` | **coherence-gate** (affine form `σ(α·(x−τ))`) |
| 5 | `katgpt-attn/src/gdn2/kernel.rs:233` | `fn sigmoid(x) = fast_sigmoid(x)` — **defined but UNUSED in kernel** | N/A — never called in production code | N/A — only called in `sigmoid_range` test | ✅ delegates to `katgpt_core::simd::fast_sigmoid` | **FALSE POSITIVE** — GDN2 gates are multiplicative (`b ⊙ k`, `w ⊙ v`), NOT sigmoid. The `sigmoid` fn is dead code in the kernel path. |
| 6a | `katgpt-attn/src/rat_bridge/fuse.rs:184` | `sigmoid(dot(query, gdn2_readout))` | raw dot product (no scaling) | bridge gate `α ∈ (0,1)` for blending | ❌ private `fn sigmoid` using std `exp` | **similarity-gate** (raw dot, no τ) |
| 6b | `katgpt-attn/src/rat_bridge/fuse.rs:122,136,153` | `sigmoid(dot(k, query))` | raw dot product (no scaling) | per-key attention weight (sum-normalized, NOT softmax) | ❌ private `fn sigmoid` using std `exp` | **similarity-gate** (raw dot, no τ) |

### Shape distribution

- **Similarity-gate** (`σ(dot)` or `σ(dot/τ)`): sites 1, 6a, 6b (3 sites)
- **Coherence-gate** (`σ(β·(x−τ))` affine form): sites 2, 4 (2 sites)
- **Selection-gate** (dot + top-k ranking): site 3 (1 site)
- **False positive**: site 5 (dead code)

### Decision per T2 rule

> "If sites use 3 different shapes (likely) → do NOT unify the formulas."

**3 shapes confirmed. NO gate-formula unification.** The max shared shape is similarity-gate at 3 sites — below the ≥4 threshold for extraction.

### The REAL DRY violation (the audit's actual finding)

The gate formulas are genuinely different and should NOT be unified. BUT the audit revealed a different, narrower DRY violation: **5 private `fn sigmoid(x) = 1/(1+exp(-x))` implementations** across sites 2, 3, 4, 6a, 6b. Each duplicates the canonical `sigmoid` that already exists as:
- `katgpt_core::sigmoid` (lib.rs:28) — numerically stable, branch-free
- `katgpt_core::simd::fast_sigmoid` — SIMD-optimized with early-exit saturation

The canonical `katgpt_core::sigmoid` is the GOAT — it's public, stable, and numerically robust. The 5 private copies should delegate to it. Exception: site 2 (`latent_functor`) uses `libm::expf` for WASM bit-exactness — that's a legitimate reason to keep a local copy (different precision contract).

This sigmoid-helper consolidation is filed as **Issue 045** — a separate micro-refactor (replace 4 private `fn sigmoid` with `katgpt_core::sigmoid` calls, ~20 LOC change across 3 files in katgpt-attn + katgpt-spectral).

## Non-Goals

- ❌ Forcing unification where the formulas genuinely differ. Different shapes stay different.
- ❌ Replacing `sigmoid` itself — the primitive `sigmoid(x) -> f32` already exists; this issue is about the gate-pattern layer on top.
- ❌ Softmax. Softmax is forbidden by AGENTS.md. This issue does not touch softmax (because there is no softmax to touch).
- ❌ Changing the β=8.0, τ=0.6 defaults. Those are tuned per-site and stay per-site.

## Cross-References

- **Confirmed sites:**
  - `katgpt-rs/crates/katgpt-core/src/engram/kernel.rs:19,210` — similarity-gate.
  - `riir-ai/crates/riir-engine/src/latent_functor/arithmetic.rs:541` — coherence-gate (`functor_gate`).
  - `katgpt-rs/crates/katgpt-spectral/src/manifold_power_iter_router.rs` — `gate_sigmoid_topk`.
- **Claimed-but-unverified sites (T1 CONFIRMED):**
  - `katgpt-rs/crates/katgpt-attn/src/ega_attn.rs` → **CONFIRMED** coherence-gate `σ(α·(z−τ))`.
  - `gdn2/kernel` → **FALSE POSITIVE** — `sigmoid` fn defined but unused; GDN2 gates are multiplicative.
  - `katgpt-rs/crates/katgpt-attn/src/rat_bridge/` → **CONFIRMED** (it's a directory, not a single file); `fuse.rs` has 2 similarity-gate sigmoid sites.
- **Discipline rule:** global `~/.agents/` AGENTS.md "Use sigmoid not softmax".
- **Origin evaluation:** Gemini "Continuous Neuro-Symbolic DAG" proposal review (2026-07-04) — the previous-session summary claimed "same formula in 5 places"; this issue audits whether that claim survives scrutiny.

## TL;DR

**AUDIT COMPLETE — no gate-formula unification justified.** Audited 6 sites (7 actual sigmoid usages — `rat_bridge` is a directory with 2 sites in `fuse.rs`). Found **3 distinct gate shapes** (similarity-gate `σ(dot/τ)`, coherence-gate `σ(β·(x−τ))`, selection-gate `dot+top-k`), 1 false positive (`gdn2/kernel.rs` `sigmoid` is dead code — GDN2 gates are multiplicative). Max shape sharing = 3 sites (similarity-gate), below the ≥4 extraction threshold. The real DRY violation is **5 private `fn sigmoid` duplications** (not the gate formulas) — filed as Issue 045 for a separate micro-refactor. This issue closes as audit-complete.
