# Issue 045 — Private `fn sigmoid` helper consolidation (DRY micro-refactor)

**Filed:** 2026-07-04
**Priority:** P3 (DRY duplication of a numerically-stable sigmoid across 4 sites; behavior-preserving if `katgpt_core::sigmoid` is bit-identical to the private copies)
**Origin:** Issue 042 T1 audit (2026-07-04) — the audit found that the gate *formulas* use 3 different shapes and should NOT be unified, but the underlying `sigmoid` primitive is duplicated 5 times.
**Blocks:** Nothing. **Blocked by:** Nothing.
**Type:** Refactor (behavior-preserving; target is bit-identical output pre/post).

---

## Problem

The canonical sigmoid `σ(x) = 1/(1+e^{-x})` is implemented **5 times** across the katgpt-rs workspace:

| # | Site | Implementation | Delegates? |
|---|---|---|---|
| 1 | `katgpt-core/src/lib.rs:28` | `pub fn sigmoid` — branch-free, numerically stable | **CANONICAL** |
| 2 | `riir-ai/.../latent_functor/arithmetic.rs:44` | `fn sigmoid` using `libm::expf` | ❌ private (WASM bit-exact contract) |
| 3 | `katgpt-spectral/src/manifold_power_iter_router.rs:426` | inlined `1/(1+(-z).exp())` | ❌ inlined |
| 4 | `katgpt-attn/src/ega_attn.rs:38` | `pub fn sigmoid` using `(-x).exp()` | ❌ private |
| 5 | `katgpt-attn/src/rat_bridge/fuse.rs:173` | `fn sigmoid` using `(-x).exp()` | ❌ private |

The canonical `katgpt_core::sigmoid` (site 1) is the GOAT:
- Public, documented, numerically stable (branches on sign of `x` to avoid `e^{-x}` overflow for large negatives).
- Already used by `closure/bridge.rs`, `data_probe.rs`, `latent_steering.rs`, `set_attention.rs`, etc.
- The branch-free form is strictly safer than the naive `1/(1+(-x).exp())` used at sites 3, 4, 5 (which overflows to `+inf` for `x < -88.7` in f32, yielding `NaN`).

## Scope

Replace the 4 private sigmoid implementations (sites 3, 4, 5; site 2 is exempt — see below) with `katgpt_core::sigmoid` calls.

### Why site 2 is exempt

Site 2 (`riir-ai/.../latent_functor/arithmetic.rs:44`) uses `libm::expf` specifically for **WASM bit-exactness** — the module doc says: "We use `libm::expf` for the sigmoid so the math is WASM-safe and bit-identical across targets (matches the engine's existing convention)." This is a legitimate precision contract that differs from `katgpt_core::sigmoid` (which uses `f32::exp`, a std/libm call that may differ on WASM targets). **Do NOT change site 2.**

### Files to change

1. **`katgpt-spectral/src/manifold_power_iter_router.rs:426`** — replace `1/(1+(-z).exp())` with `katgpt_core::sigmoid(z)`.
2. **`katgpt-attn/src/ega_attn.rs:38`** — remove private `pub fn sigmoid`, replace calls with `katgpt_core::sigmoid`.
3. **`katgpt-attn/src/rat_bridge/fuse.rs:173`** — remove private `fn sigmoid`, replace calls with `katgpt_core::sigmoid`.

Estimated: ~20 LOC changed across 3 files.

## Proposed direction

### Step 1: Verify `katgpt_core::sigmoid` is bit-identical

Before changing anything, verify that `katgpt_core::sigmoid` produces bit-identical output to the private copies on a sweep of inputs. The canonical impl branches on sign:

```rust
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}
```

The private copies use the naive form `1/(1+(-x).exp())`. For `x >= 0`, these are algebraically identical. For `x < 0`, the canonical form computes `e^x / (1 + e^x)` instead of `1 / (1 + e^{-x})` — these are mathematically equal but may differ in f32 rounding. **This is the key verification.**

If they differ on any input, the refactor is NOT behavior-preserving and must be re-scoped or abandoned.

### Step 2: If bit-identical, replace the 3 private copies

Simple find-and-replace: remove the private `fn sigmoid`, add `use katgpt_core::sigmoid` (or qualify the call).

### Step 3: Also remove dead code at gdn2/kernel.rs:233

The audit found `gdn2/kernel.rs:233` has a `pub fn sigmoid(x) = fast_sigmoid(x)` that is **never called** in the kernel (only in `sigmoid_range` test). This is dead code. Either:
- Remove the function and its test, OR
- Leave it (it correctly delegates to `fast_sigmoid` — low priority).

## Severity

**P3.** The private sigmoid copies are a real DRY violation with a correctness dimension (the naive form can NaN on extreme negatives). But:
- Not P2: the current code works for the input ranges encountered in practice (attention scores, coherence scores are bounded).
- Not P1: no active bug.
- The refactor is small (~20 LOC) and behavior-preserving (if step 1 verifies bit-identical output).

## GOAT gate

- **G1:** Bit-identical output pre/post on a sweep of `x ∈ [-100, 100]` at 0.01 resolution. The canonical impl's branch-free form must produce the same f32 bits as the naive form for all tested inputs.
- **G2:** Perf delta < 1 ns (the canonical impl has a branch; the naive form doesn't — but the branch is predictable and the canonical form avoids the overflow path).
- **G3:** All existing tests pass.
- **G4:** Zero additional allocations (sigmoid is stack-only).
- **G5/G6:** Modelless (no behavior change). ✅ trivially.

## Tasks

- [ ] **T1** Verify `katgpt_core::sigmoid` is bit-identical to the naive `1/(1+(-x).exp())` on a sweep. If NOT bit-identical, close this issue with the finding.
- [ ] **T2** Replace private sigmoid at `manifold_power_iter_router.rs:426`.
- [ ] **T3** Replace private sigmoid at `ega_attn.rs:38`.
- [ ] **T4** Replace private sigmoid at `rat_bridge/fuse.rs:173`.
- [ ] **T5** Optionally remove dead code at `gdn2/kernel.rs:233`.
- [ ] **T6** Run tests for all 3 affected crates: `cargo test -p katgpt-spectral -p katgpt-attn --lib`.

## Non-Goals

- ❌ Touching site 2 (`latent_functor/arithmetic.rs`) — it has a WASM bit-exact contract.
- ❌ Unifying gate formulas — Issue 042 closed that as "3 different shapes, no unification".
- ❌ Touching `katgpt_core::simd::fast_sigmoid` — it's a different primitive (SIMD-optimized, early-exit saturation) used by `engram/kernel.rs` and `gdn2/kernel.rs`.

## Cross-References

- **Origin audit:** Issue 042 (sigmoid gate DRY extraction — closed as audit-complete).
- **Canonical sigmoid:** `katgpt-rs/crates/katgpt-core/src/lib.rs:28`.
- ** SIMD variant:** `katgpt_core::simd::fast_sigmoid` (used by `engram/kernel.rs:141`, `curator.rs:167`, `dendritic_gate.rs:87`, `faithfulness/gate.rs:31`, `traits.rs:694`).
- **Exempt site:** `riir-ai/crates/riir-engine/src/latent_functor/arithmetic.rs:44` (WASM bit-exact `libm::expf`).

## TL;DR

Issue 042's audit found 5 private `fn sigmoid` duplications (not gate-formula duplication). The canonical `katgpt_core::sigmoid` (`lib.rs:28`, branch-free, numerically stable) is the GOAT. Replace the 3 private copies in `katgpt-spectral` and `katgpt-attn` with `katgpt_core::sigmoid` calls (~20 LOC). Site 2 (`latent_functor`) is exempt — it uses `libm::expf` for WASM bit-exactness. P3, behavior-preserving, gated on T1 bit-identical verification.
