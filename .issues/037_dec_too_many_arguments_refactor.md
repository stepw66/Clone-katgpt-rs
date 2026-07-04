# Issue 037 — `katgpt-dec` DEC operator functions: refactor `too_many_arguments`

**Filed:** 2026-07-04
**Priority:** P3 (refactor / API hygiene — not blocking, no perf regression observed)
**Origin:** Clippy cleanup pass on 2026-07-04 (commit `8b8b5335`). Four `too_many_arguments` lints were intentionally left in the DEC crate because they require API signature changes, not mechanical fixes.
**Blocks:** Nothing. **Blocked by:** Nothing.
**Type:** Refactor (no behavior change target).

---

## Problem

`cargo clippy --workspace --all-features --tests` reports 4 `clippy::too_many_arguments` lints in `katgpt-dec`, all in the Stokes-calculus / DEC operator surface:

| File | Line | Function | Approx. arg count |
|---|---|---|---|
| `crates/katgpt-dec/src/heat_kernel.rs` | 573 | heat-kernel step | 9+ |
| `crates/katgpt-dec/src/motor_gated.rs` | 228 | motor-gated operator | 8+ |
| `crates/katgpt-dec/src/nonlinear_heat_kernel.rs` | 474 | nonlinear step | 8+ |
| `crates/katgpt-dec/src/nonlinear_heat_kernel.rs` | 532 | nonlinear step (variant) | 8+ |

These were left intact during the clippy pass because the suggested fix (collapse parameters into a builder/struct) is an **API change**, not a mechanical refactor. The clippy pass explicitly avoided touching signatures that could affect callers (per the "behavior-preserving only" constraint on that pass).

## Why this is not a perf bug today

DEC operators are not the hottest path — they're invoked per-cochain-step, not per-tick-per-entity. There is **no observed perf regression** from the wide signatures. The current argument order is also load-bearing for some callers (positional reads), so a naive collapse could regress readability.

## Why it's worth fixing anyway

1. **Call-site fragility** — positional `f32` / `&[f32]` parameters in 8+ arg lists are easy to swap silently (this already bit the clippy pass itself: a zip rewrite in `heat_kernel.rs:704` initially swapped operands `v_k - λ·Lv` vs `Lv - λ·v_k`; the eigenvector test caught it). Named fields in a struct make the intent self-documenting.
2. **Feature-flag growth pressure** — DEC operators gain parameters as new features land (`motor_gated`, `nonlinear_*`). The signature will keep growing.
3. **Dovetails with the manifold-geometry strategy** — DEC operators are the canonical substrate (per global `AGENTS.md` "Manifold Geometry"). The API should be the cleanest surface in the engine.

## Proposed direction (not committed)

Group related parameters into a `DecStepCtx` (or similar) struct passed by `&`:

```rust
pub struct DecStepCtx<'a> {
    pub field: &'a DecFlowField,
    pub hodge_star: &'a HodgeStar,
    pub lambda: f32,
    pub dt: f32,
    // ...motor / nonlinear extensions grouped under one Option<Box<...>>
}

pub fn heat_kernel_step(ctx: &DecStepCtx<'_>, out: &mut DecFlowField) { ... }
```

Hot-path fields stay inline; rarely-used extensions can go behind `Option<Box<Extensions>>` (same pattern recommended in Issue 036 for `BanditPruner`).

## Tasks

- [ ] **T1** Audit all call sites of the 4 functions (count + ergonomics). Decide whether the refactor pays for itself.
- [ ] **T2** If yes: introduce `DecStepCtx`, migrate one function as a pilot (prefer `heat_kernel_step` — most callers).
- [ ] **T3** Migrate the remaining 3 functions.
- [ ] **T4** Verify: `cargo test -p katgpt-dec` 160/160 still pass. Add a positional-arg-swap regression test if not already present (mirror the eigenvector test that caught the operand-swap during the clippy pass).
- [ ] **T5** Confirm clippy clean: `cargo clippy -p katgpt-dec --all-features --tests` reports 0 `too_many_arguments`.

## Non-Goals

- ❌ Changing DEC math semantics (operators must remain bit-identical post-refactor).
- ❌ Rewriting the entire DEC crate — only the 4 flagged signatures.
- ❌ Adding new operators. This is purely an ergonomics refactor.

## Cross-References

- Clippy cleanup commit: `8b8b5335` (2026-07-04) — the pass that surfaced these lints.
- Related pattern: Issue 036 (`BanditPruner` field-bloat — same `Box<Extensions>` idea, different crate).
- Global `AGENTS.md` §"Manifold Geometry (Stokes Calculus)" — names the DEC operators as the canonical substrate.

## TL;DR

Four `too_many_arguments` lints in `katgpt-dec` (heat_kernel, motor_gated, nonlinear_heat_kernel ×2) were left during the 2026-07-04 clippy pass because they need API changes, not mechanical fixes. P3 — no perf bug today, but call-site fragility (positional `f32` args already caused one operand-swap bug caught only by the eigenvector test). Proposed fix: collapse into `DecStepCtx` struct with hot-path fields inline and extensions behind `Option<Box<>>`.
