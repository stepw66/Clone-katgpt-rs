# Plan 164: GEPA-D Reflective Config Evolution — Modelless Distillation

**Date:** 2026-05-31
**Research:** `.research/146_RLM_GEPA_Reflective_Prompt_Evolution.md`
**Status:** Planned
**Feature Gate:** `gepa_reflective = ["bandit", "memo_reflections"]` — **off by default** until GOAT proof

---

## Goal

Distill GEPA's reflective prompt evolution into our modelless stack: evolve system-level configuration (rubric weights, template hints, bandit params) from MeMo trajectory reflection using Pareto-frontier bandit selection.

**No gradient updates. No LoRA. No model-based path.** Config variants = bandit arms, reflection quality = reward.

---

## Architecture

```text
Episode → TrialLog → MeMo Reflection
                           │
                    ReflectionScore
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ParetoConfigFrontier      ReflectiveBanditPruner
    (Pareto-optimal configs    (arm = config variant,
     by reward × cost)          reward = reflection score)
              │                         │
              └────────────┬────────────┘
                           ▼
                  Next episode config
```

### Components

| Component | Type | Location |
|-----------|------|----------|
| `ReflectionScore` | Struct | `src/pruners/gepa_reflective.rs` |
| `ParetoConfigFrontier` | Struct | `src/pruners/gepa_reflective.rs` |
| `ReflectiveBanditPruner<P>` | Generic wrapper | `src/pruners/gepa_reflective.rs` |
| `ConfigVariant` | Enum | `src/pruners/gepa_reflective.rs` |

---

## Tasks

### Phase 1: Core Types & Reflection Score

- [ ] Define `ConfigVariant` enum with our configurable knobs (rubric weights, bandit ε, template hint index, absorb threshold)
- [ ] Define `ReflectionScore` struct — maps MeMo `ReflectionResult` to a scalar config-evaluation score
- [ ] Implement `ReflectionScore::from_reflection(result: &ReflectionResult) -> f32`
- [ ] Unit test: known reflection → expected score

### Phase 2: Pareto Config Frontier

- [ ] Define `ParetoConfigFrontier` — fixed-size array of Pareto-optimal `(ConfigVariant, reward, cost)` triples
- [ ] Implement `insert()` with Pareto dominance check (reward ↑, cost ↓)
- [ ] Implement `best()` — returns highest-reward config from current frontier
- [ ] Unit test: insert dominated variant → dominated variant not in frontier
- [ ] Unit test: insert non-dominated variant → frontier expands correctly

### Phase 3: Reflective Bandit Pruner

- [ ] Define `ReflectiveBanditPruner<P: ScreeningPruner>` wrapping `BanditPruner<P>`
- [ ] Each arm maps to a `ConfigVariant`
- [ ] `observe_reflection(arm, reflection_result)` — compute `ReflectionScore`, feed as bandit reward
- [ ] `best_config()` — returns config from `ParetoConfigFrontier` for next episode
- [ ] Unit test: observe good reflection for arm 0, bad for arm 1 → arm 0 config preferred

### Phase 4: Template Hint Evolution

- [ ] Extend `TemplateProposer` with a hint variant pool (instead of static hints)
- [ ] `propose_with_variant(variant: &ConfigVariant)` — select hint based on config
- [ ] `observe_hint_delta(variant_idx, delta)` — track which hint variants work best
- [ ] Unit test: hint variants evolve toward high-δ templates

### Phase 5: GOAT Proof

- [ ] Benchmark: `ReflectiveBanditPruner` throughput (target: ≥ BanditPruner baseline)
- [ ] Benchmark: `ParetoConfigFrontier::insert()` overhead (target: ≤1μs)
- [ ] Integration test: Bomber arena with reflective config evolution vs static config
- [ ] Verify zero hot-path overhead — config evolution between episodes only
- [ ] GOAT proof checklist: all 11/11 checks pass with `gepa_reflective` enabled

### Phase 6: Feature Gate & Default Decision

- [ ] Feature gate: `gepa_reflective = ["bandit", "memo_reflections"]`
- [ ] Add to `Cargo.toml` features — **off by default**
- [ ] If GOAT proof shows gain with no perf hurt → switch to default-on
- [ ] Update README with GEPA-D section

---

## Optimization Constraints

Per `optimization.md`:

1. **Zero hot-path overhead** — all config evolution happens between episodes, not during decode
2. **Fixed-size arrays** — `ParetoConfigFrontier` uses `[T; MAX_CONFIGS]`, not `Vec`
3. **O(1) per arm** — reflection score computation is arithmetic, no allocation
4. **No rayon** — config space is tiny (~10 variants), parallelism overhead dominates
5. **Pre-compute reflection scores** — compute once from `ReflectionResult`, cache per variant

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Config space too small for meaningful evolution | Start with 6 template hints + 4 rubric weight presets = 24 arms — same as WASM batch |
| Reflection quality insufficient for config scoring | Use existing `ReviewMetrics` benefit_ratio as fallback signal |
| Pareto frontier degenerates to single point | Prune frontier to keep K best variants, force exploration via ε-greedy |
| Feature bloat from new feature gate | Single file `gepa_reflective.rs`, no changes to existing modules |

---

## Dependencies

- Plan 036: ReviewMetrics (helpful/harmful tracking) ✅
- Plan 049: G-Zero Hint-δ (dense reward signal) ✅
- Plan 071: ROPD Rubric (per-criterion scoring) ✅
- Plan 094: MeMo Reflection QA pipeline ✅
- Plan 112: SR²AM Configurator Bandit ✅
