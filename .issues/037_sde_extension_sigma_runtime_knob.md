# Issue 037: SDE Extension σ as Runtime Determinism/Exploration Knob

**Opened:** 2026-06-20
**Closed:** 2026-06-20
**Origin:** Research 271 §5 (MIT 6.S184 textbook vocabulary crosswalk)
**Status:** ❌ **CLOSED — NOT NOVEL (Q1 fails: prior art ships).** σ-as-runtime-knob already exists as `TrdConfig::elf_noise_scale` (default 0.1) + `inject_sde_noise` (ELF Plan 079). Gradient-guided Langevin (the "real" SDE extension) was tested by PTRM and gave **zero improvement** over plain Gaussian — explicit negative result. No plan, no Super-GOAT.
**Parent skill rule invoked:** "If you are NOT confident enough to commit all 4 YES right now, do not write 'Super-GOAT candidate'. Write 'fusion idea — novelty TBD, needs Q1–Q4 check before verdict' and create an issue."

---

## The fusion idea

MIT 6.S184 Theorem 17 (SDE Extension Trick): given a trained flow model with vector field `u_θ_t(x)`, you can sample via either

```
ODE (deterministic):   dX = u_θ_t(X) dt
SDE (stochastic):      dX = [u_θ_t(X) + (σ²_t/2) ∇ log p_t(X)] dt + σ_t dW_t
```

for **any** `σ_t ≥ 0`, chosen **at inference time**, no retraining.

**Proposed fusion:** expose `σ_t` as a per-NPC, per-zone, or per-context runtime knob that switches between:
- **σ = 0** (sync-critical NPCs, quorum commit, deterministic replay, anti-cheat)
- **σ > 0** (exploring NPCs, curiosity-driven, generates novel latent trajectories)

This connects:
- **Freeze/thaw over fine-tuning** (don't retrain to add stochasticity, just turn the σ knob)
- **Sync boundary** (raw values still commit deterministically; only latent exploration is stochastic)
- **Curiosity/exploration** (`cgsp_runtime` — currently uses decayed-absorb bandits, NOT SDE noise)
- **Per-NPC HLA divergence** (different σ per NPC → emergent behavioral diversity)

## Why it might be novel

Reading `cgsp_runtime/runtime.rs`: curiosity is currently modeled as a **decayed-absorb priority bandit** (`p ← p·decay + reward`, decay=0.7). This is **NOT** Langevin dynamics or SDE-driven exploration. The textbook's `σ_t` is a genuinely different mechanism for the same goal (intrinsic exploration).

`bench_elf_modelless.rs::bench_sde_noise_injection_overhead` benchmarks SDE noise injection **cost** but does not expose it as a runtime determinism/exploration knob.

## Why it might NOT be novel (Q1–Q4 to run before any verdict)

- **Q1 (no prior art?):** Must grep `katgpt-rs/crates/`, `riir-ai/crates/`, `riir-armageddon/crates/` for `sigma`, `noise_inject`, `langevin`, `stochastic_explore`, `brownian`, `sde_extension`, AND codebase-vocabulary alternatives (`rng_inject`, `explore_noise`, `curiosity_noise`, `tick_jitter`). The MIT 6.S184 grep (Research 271) only confirmed it's not in the **diffusion-inference** path; it might be in `cgsp_runtime`, `npc/`, or `plasma/` under a different name.
- **Q2 (new capability class?):** "Per-NPC stochastic exploration via SDE noise injection on latent state, gated by sync-tier" — does this enable behavior no incumbent (bandits, MCTS collapse bridge) can?
- **Q3 (product selling point?):** "Our NPCs have intrinsic curiosity-driven stochasticity that's bit-identical reproducible when needed and exploratory when wanted — all from one runtime knob, no retraining." Finish this sentence or downgrade.
- **Q4 (force multiplier ≥2 pillars?):** Connects to freeze/thaw, sync boundary, cgsp_runtime, HLA divergence. Plausible ≥2. Needs explicit verification.

## Q1–Q4 Gate Result (2026-06-20) — ❌ CLOSED NOT NOVEL

Vocabulary translation + grep across both repos, both layers revealed **prior art that the initial Research 271 grep missed**. This is a textbook R269-class failure mode: the original Issue 037 used paper vocabulary ("σ", "Langevin", "Brownian", "SDE extension") and missed the codebase vocabulary ("noise_scale", "inject_sde_noise", "elf_noise_scale").

### Q1: No prior art? — ❌ FAILS

**Three pieces of prior art, in increasing severity:**

1. **`TrdConfig::elf_noise_scale: f32` (default 0.1)** in `katgpt-rs/src/distill/trd.rs` L108 — exactly the runtime σ knob Issue 037 hypothesized. Configurable per-call. Shipped.

2. **`inject_sde_noise`** — the kernel that injects Gaussian noise into the latent state. Shipped, benchmarked in `katgpt-rs/tests/bench_elf_modelless.rs::bench_sde_noise_injection_overhead`. Cross-referenced from Research 049 §8.4: *"ELF's SDE noise injection (Plan 079) IS PTRM's noise injection. Our `inject_sde_noise` was distilled from ELF; PTRM validates it from a completely different angle."*

3. **`katgpt-rs/.research/049_PTRM_Probabilistic_Tiny_Recursive_Model.md`** explicitly tested the *stronger* version of Issue 037's hypothesis (gradient-guided Langevin, not just isotropic Gaussian) and got a **negative result**:
   - L76-77: *"Langevin sampling with Q-head gradients adds nothing over pure noise. Using ∇Q to guide noise direction gave zero improvement. Pure isotropic Gaussian noise is sufficient."*
   - §6.1: *"PTRM explicitly tested Langevin sampling with Q-head gradients and found zero improvement over pure Gaussian noise. Our `inject_sde_noise` uses simple Gaussian — no changes needed."*
   - §7.4: *"Gradient-guided noise: PTRM's own negative result. Langevin sampling adds nothing."*
   - `katgpt-rs/.plans/083_ptrm_width_scaling_goat.md` §"Why not Langevin / gradient-guided noise": *"PTRM's own negative result (Appendix C): Langevin sampling with Q-head gradients contributes zero measurable improvement over pure Gaussian noise. Our `inject_sde_noise` already uses simple Gaussian. No changes needed."*

The exact mechanism Issue 037 proposed (per-NPC σ as runtime determinism/exploration knob) is the **already-shipped** `elf_noise_scale`. The *stronger* version (score-guided Langevin) was tested and rejected with explicit evidence.

### Q2: New capability class? — ❌ FAILS

No. The capability ("inject noise at runtime for exploration, knob is per-call") already exists. The hypothesized "per-NPC σ scheduling" angle is a configuration pattern, not a new mechanism.

### Q3: Product selling point? — ❌ FAILS

Cannot finish the sentence in a way that isn't already true: *"Our NPCs already use runtime-configurable Gaussian noise injection for exploration (ELF Plan 079) and we already proved gradient-guided Langevin adds nothing (PTRM)."* No new selling point.

### Q4: Force multiplier ≥2 pillars? — ❌ FAILS

Already multiplied: ELF (044), PTRM (049), TRD (217), Plan 083 all use `inject_sde_noise` / `elf_noise_scale`. The connection is made; the mechanism is shipped.

### Verdict

**CLOSED NOT NOVEL.** No plan, no Super-GOAT, no guide. Update Research 271 §2 to point the "Diffusion coefficient σ_t" row at `elf_noise_scale` / `inject_sde_noise` (ELF Plan 079) instead of "gap → Issue 037".

### Lesson (vocabulary translation failure)

The original Issue 037 was created from a paper-vocabulary-only grep ("σ", "Langevin", "Brownian", "SDE extension"). The codebase-vocabulary grep ("noise_scale", "inject_sde_noise", "elf_noise_scale") was not run. This is **exactly** the failure mode documented in workflow §1.5 step 1: *"paper-vocabulary-only is the #2 cause of false Super-GOAT claims"*. The prophylactic (Research 271 §2 crosswalk table) should have included `elf_noise_scale` and `inject_sde_noise` from the start. Fixing Research 271 in the same commit that closes this issue.

## Reading list (post-mortem, no action needed)

- MIT 6.S184 lecture notes §2.2 (Diffusion Models), §4.2 (Theorem 17 SDE Extension Trick), §4.3 Remark 20 (Langevin dynamics)
- `katgpt-rs/.research/271_MIT_6S184_Diffusion_Flow_Textbook_Vocabulary_Crosswalk.md` (vocabulary crosswalk)
- `riir-ai/crates/riir-engine/src/cgsp_runtime/runtime.rs` (current curiosity implementation — bandits, NOT SDE)
- `katgpt-rs/tests/bench_elf_modelless.rs::bench_sde_noise_injection_overhead` (existing cost benchmark)
- `katgpt-rs/.research/215_ECHO_Environment_Prediction_Inference_Time.md` (related inference-time prediction)
- `katgpt-rs/.research/236_QGF_Test_Time_Q_Guided_Flow.md` (test-time gradient guidance, adjacent framing)
