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

- [ ] **T1** Audit the 6 sites (3 confirmed + 3 claimed). For each: exact formula, input semantics, output use, current helper-delegation status. Output: a table in this issue's body (edit the file) with verdict per site.
- [ ] **T2** Decision: extract / don't extract / partial-extract, per the decision rule. Document the call.
- [ ] **T3** If extracting: add `crates/katgpt-core/src/sigmoid_gate.rs` with the chosen shape(s). Migrate confirmed-shared sites one at a time with bit-identical-output tests.
- [ ] **T4** If extracting across repos: add the helper to katgpt-rs, update riir-ai's `latent_functor/arithmetic.rs::functor_gate` to delegate. Verify no behavior change.
- [ ] **T5** If NOT extracting: close this issue with the audit table and the verdict "no unification justified". Add a one-line `// sigmoid-at-boundary discipline` note to each site for clarity.

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
- **Claimed-but-unverified sites (T1 to confirm):**
  - `katgpt-rs/crates/katgpt-attn/src/ega_attn.rs`.
  - `gdn2/kernel` (location TBD).
  - `katgpt-rs/crates/katgpt-attn/src/rat_bridge.rs` (feature `rat_plus_bridge`).
- **Discipline rule:** global `~/.agents/` AGENTS.md "Use sigmoid not softmax".
- **Origin evaluation:** Gemini "Continuous Neuro-Symbolic DAG" proposal review (2026-07-04) — the previous-session summary claimed "same formula in 5 places"; this issue audits whether that claim survives scrutiny.

## TL;DR

Multiple modules implement sigmoid-gate primitives under the "sigmoid at boundaries, never softmax" discipline. Three sites are grep-confirmed (`engram/kernel.rs`, `latent_functor/arithmetic.rs::functor_gate`, `manifold_power_iter_router::gate_sigmoid_topk`); three more are claimed but unverified (`ega_attn`, `gdn2/kernel`, `rat_bridge/fuse`). **Important: these are NOT the same formula** — similarity-gate `σ(q·k/τ)`, coherence-gate `σ(β·(c−τ))`, and top-k selection-gate are three different shapes. This issue tracks auditing whether a DRY extraction is justified (≥ 4 sites share a shape → yes; 3 different shapes → no, close as audit-complete). P3, audit-first — premature DRY is the known anti-pattern we're avoiding.
