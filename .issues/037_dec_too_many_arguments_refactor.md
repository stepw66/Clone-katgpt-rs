# Issue 037 — `katgpt-dec` DEC operator functions: refactor `too_many_arguments`

**Filed:** 2026-07-04
**Priority:** P3 (refactor / API hygiene — not blocking, no perf regression observed)
**Origin:** Clippy cleanup pass on 2026-07-04 (commit `8b8b5335`). Four `too_many_arguments` lints were intentionally left in the DEC crate because they require API signature changes, not mechanical fixes.
**Blocks:** Nothing. **Blocked by:** Nothing.
**Type:** Refactor (no behavior change target).
**Status:** ✅ **RESOLVED (2026-07-04).** T1 audit verdict: the proposed `DecStepCtx` refactor **does not pay for itself** — all 4 lints are on test helpers, not production DEC operators. The real DRY win (test-helper extraction) shipped as the alternative fix. See "T1 Verdict" below.

---

## T1 Verdict (2026-07-04): WONTFIX the `DecStepCtx` proposal; ship test-helper extraction instead

### Audit findings

Ran `cargo clippy -p katgpt-dec --all-features --tests` to enumerate the actual lint sites. **All 4 `too_many_arguments` lints fire on `#[cfg(test)] mod tests { ... }` helper functions, not production DEC operators:**

| File:Line | Function | Args | Type |
|---|---|---|---|
| `heat_kernel.rs:573` | `place_bump` | 8 | test helper (duplicated 3×) |
| `motor_gated.rs:228` | `place_bump` | 8 | test helper (duplicated copy) |
| `nonlinear_heat_kernel.rs:474` | `place_bump` | 8 | test helper (duplicated copy) |
| `nonlinear_heat_kernel.rs:532` | `nonlinear_euler_step` | 8 | test-only reference impl |

The 6 production DEC operators that have 8+ args all carry explicit `#[allow(clippy::too_many_arguments)]` with documented reasons matching the paper's operator signatures:

| Production function | Args | Allow reason |
|---|---|---|
| `evolve_motor_gated_field` | 8 | "motor-gated evolution needs mesh + field + motor + dual scratch buffers; matches the paper's operator signature" |
| `heat_kernel_trajectory_nonlinear` | 8 | "nonlinear heat kernel needs mesh + eig + field + motor + t + quad + relu; matches the paper's operator signature" |
| `heat_kernel_trajectory_nonlinear_into` | 10 | "zero-alloc variant mirrors heat_kernel_trajectory_nonlinear; caller-provided out + scratch add 2 args" |
| `expm_source_term_quadrature` | 9 | "nonlinear expm quadrature needs mesh + eig + field + motor + t + quad + relu + out + scratch; matches the paper's operator signature" |
| `heat_kernel_trajectory_bom` | 9 | "BoM perturbation sweep needs eig + field + motor + perturbation params; a config struct would obscure the math" |
| `heat_kernel_trajectory_bom_into` | 10 | "zero-alloc variant mirrors heat_kernel_trajectory_bom; caller-provided scratch/out add the 2 extra args" |

### Why `DecStepCtx` does not pay for itself

1. **Wrong target.** The proposal targets production DEC operators. But the production operators already have justified `#[allow]` attributes — they match the paper's operator signatures, and the zero-alloc `_into` variants correctly mirror their allocating twins (the extra 2 args are caller-provided `out` + `scratch`). There is nothing to fix in production.
2. **Wrong fix.** The 4 actual lints are test helpers. Test helpers don't need API ergonomics — they need DRY extraction. Grouping `place_bump`'s 8 args (field, w, h, cx_pos, cy_pos, ch, amp, sigma) into a `BumpSpec` struct would force every test to construct `BumpSpec { ... }` — that's MORE boilerplate, not less.
3. **The premise is false.** The Issue's motivating claim — "positional `f32` args already caused one operand-swap bug" — refers to a swap in `heat_kernel.rs:704` (production code, `v_k - λ·Lv` vs `Lv - λ·v_k`), not in any test helper. That production bug was in code that already carries an `#[allow]`; the swap was caught by the eigenvector test, exactly as intended. The test helpers are not at risk of the same class of bug because their args are heterogeneous types (`&mut CochainField`, `usize`, `usize`, `usize`, `usize`, `usize`, `f32`, `f32`) — positional swaps would be type errors, not silent semantic bugs.
4. **Feature-flag growth doesn't apply.** The Issue claimed "DEC operators gain parameters as new features land." True for production operators (which have allows). False for test helpers — `place_bump` has had the same 8-arg signature since it was first duplicated.

### What shipped instead: test-helper DRY extraction

**Verdict: WONTFIX on `DecStepCtx`. Ship the real fix — extract duplicated test helpers.**

Created `crates/katgpt-dec/tests/common/mod.rs` (mirroring the sibling `katgpt-core/tests/common/mod.rs` pattern from Issue 044 T3) containing the 4 duplicated helpers:
- `zero_field` (was duplicated 3×)
- `place_bump` (was duplicated 3× — the `#[allow(clippy::too_many_arguments)]` lives on this single canonical copy, with a reason documenting why a struct wouldn't help)
- `l2_norm` (was duplicated 2×)
- `l2_dist` (was duplicated 2×)

Declared at the crate root in `lib.rs`:
```rust
#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod test_common;
```

Each test module now imports via `use crate::test_common::{place_bump, zero_field, ...};`. The single-copy `nonlinear_euler_step` (test-only reference impl mirroring `evolve_motor_gated_field` per Plan 357) gets its own `#[allow(clippy::too_many_arguments)]` with a reason documenting that its 8 args must match the production signature by design — drift between this Euler baseline and the production split-step is what the nonlinear heat kernel tests detect.

### Why `tests/common/mod.rs` declared at crate root (not per-module `#[path]`)

The sibling's `katgpt-core/tests/common/mod.rs` uses per-module `#[path]` includes because it ships a **macro** (`counting_allocator!()`), which expands at the call site — each including module gets its own expansion, no identity conflict. For **functions**, per-module `#[path]` includes compile the same file N times (once per including module) and the path resolution is fragile (relative to each module's virtual directory, not the file's literal location — `src/heat_kernel.rs`'s `mod tests` resolves `#[path]` from a virtual `src/heat_kernel/tests/` directory, requiring `../../../tests/common/mod.rs`).

Declaring the module once at the crate root in `lib.rs` sidesteps both issues: one compilation, standard `crate::test_common::*` resolution, no per-module path magic.

### Task closure

- [x] **T1** Audit all call sites of the 4 functions. **DONE — verdict: WONTFIX `DecStepCtx`, ship test-helper extraction instead.** All 4 lints are on test helpers; production DEC operators already carry justified `#[allow]` attributes.
- [-] **T2** Introduce `DecStepCtx`. **SKIPPED — see T1 verdict.** The proposed refactor targets the wrong layer.
- [-] **T3** Migrate remaining functions. **SKIPPED — see T1 verdict.**
- [x] **T4** Verify: `cargo test -p katgpt-dec` tests pass. **DONE — 170/170 pass with `--all-features`, 165/165 with default features.** (Issue originally said "160/160"; count grew since filing, no tests dropped by the extraction.)
- [x] **T5** Confirm clippy clean. **DONE — 0 `too_many_arguments` warnings with `--all-features --tests` and with default features.** (One pre-existing `needless_range_loop` warning in `linear_euler_step` remains — unrelated to this issue, not introduced by the extraction.)

### Files changed

- `crates/katgpt-dec/tests/common/mod.rs` — **new** (4 shared helpers, 84 LOC)
- `crates/katgpt-dec/src/lib.rs` — add `#[cfg(test)] #[path = "../tests/common/mod.rs"] mod test_common;` (9 LOC)
- `crates/katgpt-dec/src/heat_kernel.rs` — remove 3 duplicated helpers (~50 LOC), add `use crate::test_common::*` (1 LOC)
- `crates/katgpt-dec/src/motor_gated.rs` — remove 2 duplicated helpers (~35 LOC), add `use crate::test_common::*` (1 LOC)
- `crates/katgpt-dec/src/nonlinear_heat_kernel.rs` — remove 4 duplicated helpers (~55 LOC), add `use crate::test_common::*` + `#[allow]` on `nonlinear_euler_step` (2 LOC)

Net: ~140 LOC removed, ~95 LOC added (including the new shared module + doc comments). Eliminates 3 of 4 `too_many_arguments` lints by collapsing duplicates; the 4th (`nonlinear_euler_step`) gets a justified `#[allow]`.

---

## Problem (original issue text, preserved for context)

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

**T1 audit note (2026-07-04):** Points 1-3 above were premised on the lints being in production code. They are not — all 4 are in test helpers. Points 1-3 remain valid concerns for the production operators, but those already carry `#[allow]` attributes with documented reasons. The "call-site fragility" argument specifically does NOT apply to the test helpers because their arg types are heterogeneous (type-checker catches positional swaps).

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

**T1 audit note (2026-07-04):** This proposal is WONTFIX. The production operators that would benefit from it already have `#[allow]`. Applying it to test helpers would be anti-ergonomic.

## Tasks

- [x] **T1** Audit all call sites of the 4 functions (count + ergonomics). Decide whether the refactor pays for itself.
- [ ] **T2** If yes: introduce `DecStepCtx`, migrate one function as a pilot (prefer `heat_kernel_step` — most callers).
- [ ] **T3** Migrate the remaining 3 functions.
- [x] **T4** Verify: `cargo test -p katgpt-dec` 160/160 still pass. Add a positional-arg-swap regression test if not already present (mirror the eigenvector test that caught the operand-swap during the clippy pass).
- [x] **T5** Confirm clippy clean: `cargo clippy -p katgpt-dec --all-features --tests` reports 0 `too_many_arguments`.

## Non-Goals

- ❌ Changing DEC math semantics (operators must remain bit-identical post-refactor).
- ❌ Rewriting the entire DEC crate — only the 4 flagged signatures.
- ❌ Adding new operators. This is purely an ergonomics refactor.

## Cross-References

- Clippy cleanup commit: `8b8b5335` (2026-07-04) — the pass that surfaced these lints.
- Related pattern: Issue 036 (`BanditPruner` field-bloat — same `Box<Extensions>` idea, different crate).
- Sibling pattern: Issue 044 T3 (`katgpt-core/tests/common/mod.rs` — the `counting_allocator!()` macro extraction that motivated mirroring the convention here).
- Global `AGENTS.md` §"Manifold Geometry (Stokes Calculus)" — names the DEC operators as the canonical substrate.

## TL;DR

Four `too_many_arguments` lints in `katgpt-dec` were assumed to be on production DEC operators. T1 audit revealed all 4 are on **test helpers** (`place_bump` duplicated 3×, `nonlinear_euler_step`). The 6 production operators with 8+ args already carry justified `#[allow]` attributes matching the paper's signatures. **WONTFIX the `DecStepCtx` proposal** — it targets the wrong layer and would be anti-ergonomic on test helpers. **Shipped the real fix instead:** extracted the duplicated helpers into `crates/katgpt-dec/tests/common/mod.rs` (mirroring the sibling Issue 044 T3 pattern), eliminated 3 of 4 lints via deduplication, and added a justified `#[allow]` on the single-copy `nonlinear_euler_step`. 170/170 tests pass; clippy clean of `too_many_arguments`.
