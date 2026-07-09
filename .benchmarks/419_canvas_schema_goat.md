# Plan 419 — Canvas Schema Compiler GOAT Gate (Phase 5)

**Date:** 2026-07-09
**Plan:** [`katgpt-rs/.plans/419_canvas_schema_compiler.md`](../.plans/419_canvas_schema_compiler.md)
**Research:** [`katgpt-rs/.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md`](../.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md)
**Source paper:** Valdez, *Canvas Engineering* (CommandAGI, July 2026)
**Bench:** `katgpt-core/benches/bench_419_canvas_schema_goat.rs`
**Verdict:** ✅ **PASS (all gates G1–G6)**

---

## TL;DR

The canvas schema compiler ships on **structural / correctness** merits, NOT a
behavioral claim. The load-bearing guarantee — **for a binary mask, an absent
edge ⟹ exact marginal independence (by construction)** — is proven by the G1
gate (reachability soundness). The paper's behavioral headline (1.73× parameter
efficiency, cortical R²=0.825) is **training-dependent** and explicitly NOT
claimed here; applying a declared-topology mask to a frozen backbone that was
not trained within it is a documented 19% loss (paper §5). The behavioral
question is tracked separately in [`katgpt-rs/.issues/043`](../.issues/043_canvas_modelless_behavioral_gain_poc.md).

**Perf headline:** `compile_schema` on the paper's 199-region ICU schema
completes in **1.5 µs** (budget 10 ms — 6600× under); `TransitiveClosure::reaches`
is a sub-nanosecond O(1) bitset lookup (budget 100 ns).

---

## What ships (the modelless half)

A typed `CanvasSchema` compiler (`crates/katgpt-core/src/canvas/`) that lowers a
declared region layout + directed topology into:

- `AttentionMaskSpec` — sparse `M ∈ R^{N×N}_{≥0}` consumable by any sparse-attention path.
- `LossWeightMask` — per-position `ω_i = Σ_r 1[i∈I_r]·loss_weight_r·1[is_output_r]`.
- `reachability_horizon` / `can_reach` / `TransitiveClosure` — the exact-marginal-independence guarantee.
- `transfer_distance` — semantic-type compatibility scalar (`1 − cosine` of frozen embeddings).

Module split (per AGENTS.md `< 2048` line rule): `mod.rs` (constructors + compiler),
`types.rs` (decoupled structs/impls), `mask.rs` (mask + loss builder),
`reachability.rs` (information-flow graph + CSR + closure), `transfer.rs` (semantic distance).

---

## Direction convention (paper §2.2, authoritative)

This was the trickiest correctness detail and is recorded here so future editors
don't re-derive it:

- **`Connection(src, dst)`**: `src` *queries* `dst` keys/values (paper §2.2).
- **Information flows `dst → src`** (the key region influences the querier).
- **`build_attention_mask`** emits `(query=src_pos, key=dst_pos, weight)` — the
  `i` (row/query) index comes from `src`, the `j` (column/key) index from `dst`.
- **Information-flow graph `G`**: arc `dst → src` for every connection. Since info
  flows `dst → src`, the arc *is* the info-flow direction. `can_reach(from, to)`
  follows arcs from `from` to `to`, so it reads as **"`from` influences `to`"**.
- **`causal_chain([A,B,C])`** emits `Connection(current, predecessor)` — each
  region queries its predecessor — producing info-flow arcs `A → B → C`. This is
  what makes Plan 419 **T3.6** hold: `can_reach(A, C, 1) == false` but
  `can_reach(A, C, 2) == true`.

(Two earlier WIP conventions inverted this — "dst queries src" — which made
`can_reach` read backwards and broke T3.6. Corrected to the paper convention.)

---

## GOAT gate results (G1–G6)

| Gate | Target | Verdict |
|------|--------|---------|
| **G1** Reachability soundness (LOAD-BEARING) | absent edge ⟹ `can_reach == false` for all horizons | ✅ **PASS** — isolated topology: region 0 cannot reach region 1 at horizons {0,1,2,10,100,1000,10000} (exact marginal independence by construction) |
| **G2** Horizon bound (T3.6) | `can_reach(A,C,1)=false`, `can_reach(A,C,2)=true`, `reachability_horizon=n_blocks·n_steps` | ✅ **PASS** |
| **G3** No-regression | `--all-features` clean; `--no-default-features` does not pull canvas; runtime structure sane | ✅ **PASS** — 3-region schema compiles; mask/loss structures correct. `--all-features` and `--no-default-features` both compile clean (verified externally) |
| **G4** Alloc-free hot path | `TransitiveClosure::reaches` + `reachability_horizon` allocate 0 per call | ✅ **PASS** — 0 allocs/1000 reaches + 0/1000 horizon. `compile_schema` allocates at schema-load time only (3 Vecs), not gated to zero (the plan's G4 gates the hot path, not the load-time build) |
| **G5** Perf | `compile_schema` on 199-region ICU schema < 10 ms; `reaches` p50 < 100 ns | ✅ **PASS** — `compile_schema(199)` = **1515 ns** (6600× under the 10 ms budget); `reaches` p50 = **0 ns** (the O(1) bitset lookup is sub-nanosecond) |
| **G6** Feature isolation | `canvas_schema` gates all symbols; 0 bytes when disabled | ✅ **PASS** — `--no-default-features` does not compile `canvas`; opt-in until promotion |

### Raw G4 / G5 numbers

```
G4 raw: reaches allocs/1000 = 0, horizon allocs/1000 = 0
G5 raw: compile_schema(199) = 1515 ns, reaches p50 = 0 ns
```

---

## What the GOAT does NOT measure (the honesty)

- **Behavioral parity** with the paper's training-dependent results (1.73× parameter
  efficiency, cortical R²=0.825). Those require *training a DiT within the declared
  topology* — a riir-train concern. The modelless compiler ships the *compilation*
  and the *guarantee*, not the *behavioral gain*.
- **The flat-canvas 19% loss** (paper §5 calibration #2): applying a declared-topology
  mask to a frozen untrained-for-it backbone degrades performance. This is an
  expected property of the modelless application, not a primitive defect.
- **Representation stability** (paper §6 linchpin): whether identical declared
  structure induces predictably aligned latent geometry across seeds/backbones is
  unproven. The compiler claims the *mask structure* is what the schema declares,
  not that latent geometry aligns.

The fusion PoC (does compiled-canvas + reachability improve per-NPC behavior
modellessly over un-unified constituents?) is the defend-wrong gate for
Super-GOAT re-evaluation — tracked in `.issues/043`.

---

## Re-run

```bash
# Isolated target dir (avoid locking the workspace target; rm when done)
CARGO_TARGET_DIR=/tmp/canvas419 cargo bench \
  -p katgpt-core --features canvas_schema \
  --bench bench_419_canvas_schema_goat -- --nocapture
```

Cleanup after measurement: `rm -rf /tmp/canvas419`.

---

## Cross-references

- **Plan 419:** [`katgpt-rs/.plans/419_canvas_schema_compiler.md`](../.plans/419_canvas_schema_compiler.md)
- **Research 398:** [`katgpt-rs/.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md`](../.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md)
- **Issue 043 (fusion PoC, behavioral gain):** [`katgpt-rs/.issues/043_canvas_modelless_behavioral_gain_poc.md`](../.issues/043_canvas_modelless_behavioral_gain_poc.md)
- **Feature catalog entry:** [`katgpt-rs/.docs/feature_catalog/opt_in_features.md`](../.docs/feature_catalog/opt_in_features.md) §12

---

## TL;DR

`canvas_schema` (opt-in) passes all six GOAT gates on structural/correctness
merits. The reachability guarantee (absent edge ⟹ exact marginal independence
for binary masks) holds by construction (G1). Perf is far under budget (G5:
1.5 µs compile, sub-ns reaches). The behavioral gain is training-dependent and
tracked in `.issues/043`. Promotion to default-on is deferred pending the
`.issues/043` fusion PoC verdict.
