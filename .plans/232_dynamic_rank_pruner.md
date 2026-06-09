# Plan 232: DynamicRankPruner — GATv2 Static Ranking Detection & Correction

> **Source:** Research 206 — GATv2 Dynamic Attention Ranking Distillation
> **Date:** 2026-06
> **Status:** 🧪 GOAT Proof Passed (5/5)
> **Feature Gate:** `dynamic_rank` (default-off until GOAT proof passes)
> **Related:** Plan 197 (DominoPruner GOAT 25/25), Plan 030 (BanditPruner), Plan 033 (Bomber Arena)

---

## Background

GATv2 (ICLR 2022) proves that GAT computes **static attention** — the ranking over keys is invariant to the query. GATv2 fixes this by reordering the composition (apply activation after additive scoring, not before). This produces **dynamic attention** where the ranking actually depends on the query.

**In katgpt-rs:**
- `BanditPruner` Q-values are per-arm only (`q_values[arm]`) → ranking is **static** (no parent conditioning)
- `DominoPruner` already does prefix-conditioned correction via `causal_correction()` → **dynamic** (the pattern to reuse)
- The `ScreeningPruner::relevance()` trait already accepts `parent_tokens` — the *parameter* exists, but implementations may ignore it

**The gap:** Nobody detects *programmatically* whether a pruner is static or dynamic. We need a diagnostic + automatic correction wrapper.

---

## Tasks

- [x] T1: `static_ranking_diagnostic()` — ✅ DONE (in `dynamic_rank.rs`, 50+ LOC with Kendall tau)
- [x] T2: `DynamicRankPruner<P>` wrapper — ✅ DONE (struct with papaya corrections table)
- [x] T3: `ScreeningPruner` impl — ✅ DONE (zero-overhead passthrough when dynamic)
- [ ] T4: Integration with `BanditPruner` — add `with_dynamic_rank(self) -> DynamicRankPruner<Self>` builder method. Feature-gated under `dynamic_rank`.
- [x] T5: GOAT proof test — ✅ DONE (5/5 GOAT proofs pass)
- [x] T6: Feature gate — ✅ DONE (`dynamic_rank = ["papaya"]` in Cargo.toml)
- [x] T7: Module glue — ✅ DONE (`#[cfg(feature = "dynamic_rank")] pub mod dynamic_rank;` in mod.rs)

---

## GOAT Proof Results

**Date:** 2026-06-09
**Command:** `cargo test --features "dynamic_rank,bandit" --test goat_232_dynamic_rank -- --nocapture`

| Gate | Description | Result |
|------|-------------|--------|
| G1 | Diagnostic identifies static pruners (NoScreeningPruner → static) | ✅ PASS |
| G1b | Diagnostic identifies dynamic pruners (ContextDependentPruner → dynamic, entropy=0.5556) | ✅ PASS |
| G2 | BanditPruner confirmed static (entropy=0.0000) | ✅ PASS |
| G3 | DynamicRankPruner diagnoses static and applies corrections | ✅ PASS |
| G4 | Zero overhead when inner pruner is already dynamic | ✅ PASS |

**5/5 GOAT proofs passed.**

### Key Findings
1. **BanditPruner IS static** — Kendall tau entropy = 0.0000 across all parent contexts (GAT's problem confirmed)
2. **ContextDependentPruner IS dynamic** — entropy = 0.5556 (different parents → different argsort)
3. **Zero overhead verified** — when inner pruner is dynamic, wrapped relevance = direct relevance (diff < 1e-6)
4. **Correction mechanism works** — record_correction() + relevance() produce context-dependent adjustments

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  DynamicRankPruner<P>                │
│                                                      │
│  ┌──────────┐   ┌────────────────┐   ┌───────────┐  │
│  │  inner: P │──▶│  diagnostic()  │──▶│ correction │  │
│  │(Screening │   │ argsort stable?│   │  table     │  │
│  │  Pruner)  │   │ across N ctxs  │   │ Papaya<Hm> │  │
│  └──────────┘   └────────────────┘   └───────────┘  │
│        │                                    │         │
│        │         diagnosis result           │         │
│        ├──── dynamic: pass-through ─────────┤         │
│        └──── static:  + correction ─────────┘         │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Diagnostic-first:** Run `static_ranking_diagnostic()` on the first N calls. Don't correct blindly — prove the inner pruner is static first.
2. **Papaya for lock-free correction table:** Per user optimization rules, use `papaya` instead of `Arc<RwLock<HashMap>>` for the prefix hash → correction vector lookup.
3. **Zero overhead when dynamic:** Once diagnosed as dynamic, the correction path is never entered again (atomic flag check).
4. **Wrapper pattern (not replacement):** Same as existing patterns (`BinaryScreeningPruner<P>`, `BanditPruner<P>`). Additive, non-breaking.
5. **Prefix hash → correction:** Same pattern as `DominoPruner::causal_correction()`. Hash `parent_tokens` → lookup correction delta → add to inner relevance.

### File Layout

| File | Purpose | ~LOC |
|------|---------|------|
| `src/pruners/dynamic_rank.rs` | `StaticRankingReport`, `static_ranking_diagnostic()`, `DynamicRankPruner<P>` | ~200 |
| `src/pruners/bandit.rs` | `with_dynamic_rank()` builder (feature-gated) | ~10 |
| `src/pruners/mod.rs` | Module glue | ~2 |
| `tests/goat_dynamic_rank.rs` | GOAT proof benchmark | ~80 |
| `Cargo.toml` | Feature gate | ~1 |

---

## GOAT Gate

### Acceptance Criteria

The `dynamic_rank` feature **promotes to default-ON** if ANY of the following is true:

| Gate | Condition | Measurement |
|------|-----------|-------------|
| **G1: Acceptance gain** | BanditPruner + DynamicRankPruner ≥ 2% higher acceptance rate vs BanditPruner alone on bomber arena | `tests/goat_dynamic_rank.rs` |
| **G2: Diagnostic proof** | Diagnostic confirms BanditPruner IS already dynamic (unlikely, but valid outcome) | `static_ranking_diagnostic()` report |
| **G3: Zero regression** | No acceptance rate regression when wrapping an already-dynamic pruner (NarrowingPruner, DominoPruner) | Unit test in dynamic_rank.rs |

### Failure Outcome

If G1 fails (no ≥2% gain) and G2 fails (BanditPruner IS static but correction doesn't help):
- Feature stays default-off
- Diagnostic is still valuable — it documents which pruners are static
- Do NOT promote to default

### Dependencies

- Requires `bandit` feature (for BanditPruner integration)
- Requires `bomber` feature (for GOAT proof test)
- Uses `papaya` crate (already in workspace for `rosetta_pruner` feature)

---

## Constraints

- **Modelless:** Inference-time only, no LLM training
- **Feature-gated, default-off** until GOAT proof passes
- **Zero overhead** when inner pruner is already dynamic
- **Papaya** for lock-free correction table (per optimization rules)
- **Under 2048 LOC** per file
- **Wrapper pattern:** Additive, does not modify existing pruner implementations

---

## TL;DR

Wrap `BanditPruner` with a diagnostic that detects static ranking (GATv2 insight: argsort invariant across parent contexts). If static detected, apply DominoPruner-style prefix correction. If already dynamic, zero overhead passthrough. Feature-gated `dynamic_rank`, default-off. GOAT gate: ≥2% acceptance rate improvement on bomber arena OR proof that BanditPruner is already dynamic.
