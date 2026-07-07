# KatgptProof — Lean 4 formal verification for the sigmoid ranking-preservation property

Second Lean 4 formal-verification instance in the 5-repo quintet (katgpt-rs / riir-ai / riir-chain / riir-neuron-db / riir-train), and the **first in the public MIT repo** (`katgpt-rs`). The first instance is `riir-chain/.proofs/RiirChainProof` (Plan 004 — LatCal fixed-point round-trip).

## What this proves

| File | Theorem | Statement |
|---|---|---|
| `Bridge/Basic.lean` | (spec only) | `dot` product over `ℝ` mirroring `ActionBridge::select_action`'s `mul_add` loop; sigmoid = Mathlib's `Real.sigmoid` |
| `Bridge/RankingPreserved.lean` | `action_bridge_ranking_preserved` | `∀ (q d₁ d₂ : ι → ℝ), dot q d₁ > dot q d₂ → sigmoid (dot q d₁) > sigmoid (dot q d₂)` |
| `Bridge/RankingPreserved.lean` | `action_bridge_argmax_preserved` | If `d₁` has the strictly largest dot product, it also has the strictly largest sigmoid score |
| `Ssmax/Basic.lean` | (spec only) | `alphaGold N c = 1 / (1 + (N−1)·N^(−c))` — the paper's dilution bound; `alphaGold_bounded` proves `(0,1)` |
| `Ssmax/DilutionBound.lean` | `alphaGold_strictMono_in_c` | For `N > 1`, `alphaGold` is strictly increasing in `c` — the monotonicity that makes SSMax work |
| `Ssmax/DilutionBound.lean` | `ssmax_dominates_base` | For `s_L · log(N) ≥ 1` and `c_base > 0`: SSMax does not decrease gold mass (threshold is `s_L·log(N)≥1`, NOT `N≥2`) |

The headline theorem is `action_bridge_ranking_preserved`: it proves that `ActionBridge::select_action`'s sigmoid projection preserves dot-product ordering. This is the ∀-form of the empirical `g1_3_bridge_ranking_preservation` test in `crates/katgpt-core/src/micro_belief/tests.rs` (Plan 281 G1.3), which samples only 1000 random triples. The Lean theorem holds for **every** triple.

## Why this exists

The bridge projects latent Q-values to raw action scores via `sigmoid(dot(q, direction))`. The whole point — per `AGENTS.md`'s latent-vs-raw rules — is that downstream consumers can rank entities by the *scalar* projection without needing the latent vector. This is only sound if sigmoid preserves the dot-product ordering, i.e. if sigmoid is strictly monotone.

Before this proof, that property was enforced by:
1. A doc comment ("never softmax").
2. The empirical G1.3 test (1000 random triples).

After this proof, it is enforced by a Lean theorem (over `ℝ`) plus a Rust spec-match test that fails CI if the Rust `fast_sigmoid` drifts from the Mathlib `Real.sigmoid` spec.

## Why Mathlib (and why the toolchain differs from riir-chain)

`RiirChainProof` (riir-chain Plan 004) deliberately avoids Mathlib: its theorem reduces to integer linear arithmetic, decided by Lean core's `omega` tactic, keeping `lake build` under 5 seconds with a pinned `leanprover/lean4:v4.31.0`.

`KatgptProof` cannot avoid Mathlib: sigmoid's strict monotonicity depends on the transcendental analysis of `exp` (`Real.exp`), which is not in Lean core. Mathlib ships `Real.sigmoid_strictMono` (in `Mathlib.Analysis.SpecialFunctions.Sigmoid`) — the exact lemma this proof needs. Adding Mathlib forces the toolchain to `leanprover/lean4:v4.32.0-rc1` (Mathlib's current requirement), which is *higher* than riir-chain's pinned version. This is an unavoidable consequence of depending on Mathlib for transcendental analysis. The first `lake build` downloads Mathlib's precompiled cache (8592 files from the lake cache server), so build time stays reasonable.

## How to run

```bash
# 1. Install Lean 4 toolchain (one-time, no root needed)
curl https://elan-init.lean-lang.org/elan-init.sh -sSf | sh

# 2. Build the proofs
cd katgpt-rs/.proofs
lake build

# 3. Verify the spec-match test on the Rust side
cd katgpt-rs
cargo test --features action_bridge --test bridge_spec_match
```

Both must pass for the proof to be valid:
- `lake build` proves the math (`Real.sigmoid_strictMono` ⟹ ranking preserved).
- `cargo test --features action_bridge --test bridge_spec_match` proves the Rust `fast_sigmoid` / `select_action` match the Lean spec.

If either fails, the proof is invalid.

## Axioms

All three theorems depend only on Lean's three standard foundational axioms:
- `propext` (propositional extensionality)
- `Classical.choice` (axiom of choice)
- `Quot.sound` (quotient soundness)

No `sorry`. No `sorryAx`. Verified by `#print axioms`. These are the same axioms Mathlib itself is built on.

## Layout

```
.proofs/
├── lakefile.toml              # Lean 4 build manifest (requires Mathlib)
├── lean-toolchain             # Pins Lean version (v4.32.0-rc1, Mathlib's requirement)
├── .gitignore                 # .lake/, lake-manifest.json
├── README.md                  # this file
└── KatgptProof/
    ├── Bridge/
    │   ├── Basic.lean                  # Spec: dot product + sigmoid (Mathlib's Real.sigmoid)
    │   └── RankingPreserved.lean       # Theorems: ranking + argmax preservation (Plan 293)
    └── Ssmax/
        ├── Basic.lean                  # Spec: alphaGold dilution bound + monotonicity of N^c (Plan 411 S3)
        └── DilutionBound.lean           # Theorems: alphaGold strictMono in c + ssmax_dominates_base (Plan 411 S3)
```

## The f32 caveat (and why it doesn't break the theorem)

The Lean theorem is stated over `ℝ` (infinite precision). The Rust `fast_sigmoid` is an `f32` approximation:
- For `|x| > 40`: saturates to exactly `0.0` or `1.0`.
- Near `±18`: f32's ~6e-8 spacing near 1.0 causes distinct dot products to map to the *same* f32 sigmoid value (a tie).

Neither affects the theorem's validity:
- **Saturation ties** are consistent — the bridge breaks them by first-wins insertion order. No action can *outrank* another via sigmoid that didn't already win via dot product.
- A genuine **flip** (larger dot → strictly smaller sigmoid) would violate the theorem and is caught by the `empirical_ranking_preserved_within_f32_precision` spec-match test.

## Regenerating after bridge changes

If `katgpt-rs/crates/katgpt-core/src/bridge/mod.rs::select_action` or `simd/activations.rs::fast_sigmoid` changes:
1. If the projection is no longer `sigmoid` (e.g. swapped to softmax), the Lean theorem is invalid — the bridge must keep using a strictly-monotone function.
2. If `fast_sigmoid`'s mathematical definition changes, update `Bridge/Basic.lean`'s doc to match.
3. Run `lake build` — the theorem should still hold (it depends only on `Real.sigmoid`'s monotonicity, not the Rust implementation).
4. Run `cargo test --features action_bridge --test bridge_spec_match` — the spec-match tests must still pass.

## Cross-references

- Plan: `.plans/293_action_bridge_lean4_monotonicity_proof.md`
- Research: `.research/292_Bridge_Neuro_Symbolic_Formal_Verification_Gap.md`
- Plan: `.plans/411_ssmax_goldshare.md` (Stretch S3 — SSMax dilution-bound)
- Research: `.research/392_*` (SSMax / GoldShare distillation)
- Sibling instance (Tier 1): `riir-chain/.proofs/RiirChainProof` (Plan 004 — LatCal round-trip, Mathlib-free)
- Empirical test (complementary ∃-check): `crates/katgpt-core/src/micro_belief/tests.rs::g1_3_bridge_ranking_preservation`
- Rust implementation: `crates/katgpt-core/src/bridge/mod.rs::ActionBridge::select_action`, `crates/katgpt-core/src/simd/activations.rs::fast_sigmoid`
- Rust implementation (SSMax): `crates/katgpt-core/src/ssmax.rs::apply_ssmax_inplace`, `SsmaxMode`

## Status

**Phase 1–3 of Plan 293: COMPLETE.** All gates pass:

- **G1** (toolchain bootstraps): ✅ `lake build` succeeds, Lean `v4.32.0-rc1` + Mathlib.
- **G2** (theorem type-checks): ✅ `action_bridge_ranking_preserved` + `action_bridge_argmax_preserved` + `action_bridge_ranking_preserved'` all compile, no `sorry`, axioms = `{propext, Classical.choice, Quot.sound}`.
- **G3** (Rust spec matches Lean): ✅ `cargo test --features action_bridge --test bridge_spec_match` — 6/6 tests pass.

**Plan 411 S3 (SSMax dilution-bound theorems): COMPLETE.** Added 2026-07-07.

- **G1** (Lean builds): ✅ `lake build` succeeds — `Ssmax/Basic.lean` + `Ssmax/DilutionBound.lean`.
- **G2** (theorem type-checks): ✅ `alphaGold_strictMono_in_c` + `alphaGold_lt_of_c_lt` + `ssmax_dominates_base` + `alphaGold_bounded` all compile, no `sorry`, axioms = `{propext, Classical.choice, Quot.sound}`.
- **G3** (Rust spec matches Lean): ✅ `cargo test --features ssmax_temperature --test ssmax_spec_match` — 8/8 tests pass (now runs by default since `ssmax_temperature` is Phase 13 DEFAULT-ON).

**The formal-verification value-add:** the Lean proof *sharpened the plan's
threshold*. Plan 411 S3 sketched "`s_L = 1, N ≥ 2 ⇒ SSMax ≥ base`", but the
correct condition is `s_L · log(N) ≥ 1` (i.e. `N ≥ 3` for `s_L = 1`, since
`log(2) ≈ 0.693 < 1`). The `ssmax_dominates_base` theorem establishes the
tight threshold; the `spec_threshold_is_s_l_times_log_n_geq_one_not_n_geq_two`
Rust test guards both the dominance (for `N ≥ 3`) and the reversal at `N = 2`.

Verified by:
```bash
cd katgpt-rs/.proofs && lake build    # → Build completed successfully (2276 jobs)
cd katgpt-rs && cargo test -p katgpt-core --test ssmax_spec_match  # → 8 passed
cd katgpt-rs && cargo test -p katgpt-core --test bridge_spec_match --features action_bridge  # → 6 passed
```

## License

Same as the rest of `katgpt-rs` — MIT (public).
