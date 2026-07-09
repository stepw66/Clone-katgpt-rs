# Sigmoid-not-Softmax: The Universality-Class Escape

> Theoretical grounding: **Liu & Gore, *Neural Scaling Universality: If Exponents Are Fixed, Time to Understand Coefficients*** (arXiv:2606.25008, Jun 2026). Distilled in [`katgpt-rs/.research/315_Neural_Scaling_Universality_Exponents_Fixed_Coefficients_Actionable.md`](../../.research/315_Neural_Scaling_Universality_Exponents_Fixed_Coefficients_Actionable.md).

## The rule and its grounding

The `AGENTS.md` rule **"Use sigmoid not softmax"** is enforced pervasively in this codebase — `fast_sigmoid` / `simd_sigmoid_inplace` in `katgpt-core/src/simd/`, `direction_vector_decode` in the analytic lattice decoder, `select_action` in the bridge layer, the CUCG G6 gate (`benches/cucg_goat.rs`, statically asserts 0 softmax hits), `CommittedFieldBlend` and `PersonalityWeightedComposition` blend weights, and the LM head's `log-softmax` over vocabulary (the one place softmax is correct, since the LM head must produce a proper distribution). Liu & Gore argue that scaling-law exponents are universal and fixed by three generic mechanisms: the **1/3 time exponent** is fixed by softmax's partition-of-unity nonlinearity; the **inverse-width** exponent by superposition; the **inverse-depth** exponent by ensembling. Coefficients, not exponents, are the actionable lever. Per their argument, our pervasive use of sigmoid gates (gentler, per-coordinate, no partition constraint) places our runtime blending gates in a **different universality class** than softmax-gated blending — not better, just structurally distinct, with different effective exponent behavior. This is the missing theoretical justification that Research 295 deferred for the sigmoid-not-softmax rule. The runtime canary in CUCG G6 (`benches/cucg_goat.rs`) demonstrates the structural difference quantitatively: under large scale `T`, softmax output entropy collapses to 0 (one-hot) while sigmoid normalized entropy plateaus at `log(n_positive)` — a nonzero floor set by input sign structure, not by partition dynamics.

## Coefficient-space navigation is the actionable lever

Liu & Gore's complementary claim — **coefficients, not exponents, are where performance is won** — endorses the codebase's existing committed-blend stack as the correct inference-time instantiation of their recommendation. Four primitives navigate coefficient space without touching the (universal) exponent structure: [`PersonalityWeightedComposition`](../../.plans/297_personality_weighted_composition.md) (Plan 297, per-tick sigmoid-gated personality drift), [`CommittedFieldBlend`](../../.plans/321_sampling_invariant_per_entity_moe_primitive.md) (Plan 321, BLAKE3-committed archetype blend logits), the [Spectral Budget Router](../../.plans/254_spectral_budget_router.md) (Plan 254, NS-depth layer allocation), and the `MerkleFrozenEnvelope` freeze/thaw swap in `riir-neuron-db`. Resources spent improving blend granularity, commit frequency, or drift stability are spending on the actionable axis the paper identifies — not on a workaround around scaling limits.

## What this is NOT

- **Not a claim that sigmoid is *better* than softmax.** The paper makes no such claim and neither do we. The claim is structural: sigmoid gates live in a different universality class. Which class is better for which inference task is empirical (riir-train territory).
- **Not a Chinchilla-style compute recipe.** The paper argues coefficients matter but does not prescribe how to navigate them at training time. Our coefficient navigation is inference-time (blend, route, freeze) and already shipped.
- **Not a constraint on the LM head.** The LM head softmax is exactly where the 1/3 exponent *should* be fixed (it is the canonical softmax nonlinearity). The escape applies to *blending gates*, not to the output distribution.

## Cross-references

- [Research 315 — full distillation](../../.research/315_Neural_Scaling_Universality_Exponents_Fixed_Coefficients_Actionable.md)
- [Research 295 — AC-GPT Arbitrary Conditionals (deferred this grounding)](../../.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md)
- [Research 222 — Spectral Scaling Laws (empirical predecessor)](../../.research/222_Spectral_Scaling_Laws_Muon_Adaptive_Inference.md)
- [CUCG G6 canary — runtime demonstration](../benches/cucg_goat.rs)
- [bench_313 comment — LM head exception explained](../../crates/katgpt-core/benches/bench_313_ac_prefix_goat.rs)
