# Issue 040 — PTG × latent_functor edge composition (continuous functor as PTG edge operator)

**Filed:** 2026-07-04
**Priority:** P2 (high product payoff: closes the neuro-symbolic gap — symbolic graph edges become differentiable transitions, enabling per-NPC cognitive-style fingerprinting and cross-task transfer of execution motifs)
**Origin:** Evaluation of Gemini's "Continuous Neuro-Symbolic DAG" proposal (2026-07-04). PTG already ships as the symbolic DAG (Plan 290, default-on); `latent_functor` ships in riir-ai (Plan 273) as the continuous transition operator. They don't compose at the edge level.
**Blocks:** Nothing. **Blocked by:** Nothing — both substrates ship.
**Type:** New primitive (modelless, ~150 LOC estimate). Note: this is closer to a new feature than a refactor; if pursued it likely warrants a Plan, but per AGENTS.md it can land as an Issue first with the Plan triggered when T1 audit confirms the gain.

---

## Problem

`PrimitiveTransitionGraph` (PTG, Plan 290, `closure/mod.rs`) ships as the symbolic execution DAG:

```rust
pub enum OperatorKind {
    Sequence = 0,       // A → B
    Branch = 1,         // A → B | skip
    Recurse = 2,        // A calls A at smaller scale
    ParallelJoin = 3,   // (A ∥ B) → C
}

pub struct PtgEdge {
    pub op: OperatorKind,
    pub from: u32,
    pub to: u32,
}
```

These edges are **purely symbolic** — they record *that* primitive B followed primitive A, with no continuous operator semantics. The graph answers "what ran in what order" but not "what continuous latent transition connects A's output state to B's input state".

Meanwhile, `latent_functor` (riir-ai Plan 273, `riir-engine/src/latent_functor/arithmetic.rs`) ships the continuous transition operator:

```rust
pub fn apply_functor(state: &[f32], direction: &[f32], bias: f32) -> Vec<f32>
pub fn extract_functor_into(source: &[f32], target: &[f32], out: &mut [f32])
pub fn functor_gate(coherence: f32, beta: f32, tau: f32) -> f32  // sigmoid(β·(c−τ))
```

The two have ONE existing bridge: `closure/bridge.rs::ptg_to_motif_embedding` projects a PTG's primitive-frequency feature vector through a `MotifDirections` table → a K-dim **sigmoid embedding** (used for motif mining, PRI scoring, TAR — not for execution). The PTG's *edges* never carry a continuous operator.

**This is the gap Gemini's proposal accidentally named correctly**, even though the proposal mis-named every existing component. The composition "edge = latent_functor operator" does not exist.

### Why this matters

1. **Per-NPC cognitive-style fingerprinting.** Two NPCs with identical PTGs but different functor edge operators (different direction sets, different beta/tau gates) have *different* cognitive styles. Today the PTG cannot represent this — style lives in the shard set, disconnected from execution.
2. **Cross-task transfer of execution motifs.** A learned functor on edge `(A→B)` in task family F1 could transfer to task family F2 if the edge semantics are continuous (direction vectors can be reused). Symbolic-only edges offer no transfer signal.
3. **Bridge pattern completion.** Per global AGENTS.md §"Bridge Pattern": raw → latent projection onto learned direction vectors is the canonical bridge. PTG → latent is a missing bridge instance.
4. **Deterministic replay enrichment.** Replay currently re-executes symbolic steps. With functor edges, replay could *also* re-project latent state through the recorded functors, giving post-hoc analysis of "what the NPC was thinking" not just "what the NPC did".

## Scope

A new PTG edge variant that carries a continuous functor operator:

### Option A (preferred): extend `OperatorKind`

```rust
pub enum OperatorKind {
    Sequence = 0,
    Branch = 1,
    Recurse = 2,
    ParallelJoin = 3,
    /// Continuous latent transition: state' = apply_functor(state, direction, bias),
    /// gated by functor_gate(coherence, β, τ). The direction set is referenced
    /// by EngramTableId (BLAKE3 root of a table of direction vectors) so the
    /// edge stays 32 bytes + scalar params, not a variable-length blob.
    Functor = 4,
}

pub struct PtgEdge {
    pub op: OperatorKind,
    pub from: u32,
    pub to: u32,
    /// Only populated when `op == Functor`. Carries:
    ///   - direction_set: EngramTableId (32 bytes — content-addressed ref into engram table)
    ///   - direction_index: u16 (which row of the table — supports K-direction sets)
    ///   - beta: f32 (gate steepness — baked at extraction time)
    ///   - tau: f32 (gate threshold — baked at extraction time)
    /// For symbolic ops, this is `None`.
    pub functor: Option<FunctorEdgeParams>,
}

pub struct FunctorEdgeParams {
    pub direction_set: EngramTableId,
    pub direction_index: u16,
    pub beta: f32,
    pub tau: f32,
}
```

### Wire-format compatibility note

`PtgEdge` is postcard-serialized (`closure::serialize_postcard`). Adding `Option<FunctorEdgeParams>` is **backward-compatible** if (a) the variant is feature-gated and (b) `Option<T>` postcard-encodes as a discriminant byte (which it does). Existing PTGs (all `op ∈ {Sequence, Branch, Recurse, ParallelJoin}` and `functor: None`) serialize identically. **Verify** with a round-trip test on a pre-existing PTG fixture (T3).

### Option B (not preferred): parallel `ContinuousPtg` structure

Rejected because it duplicates the topology and complicates motif mining (which would need to walk both graphs). Option A keeps one graph with two edge types.

## Proposed direction

### File locations

- `katgpt-rs/crates/katgpt-core/src/closure/mod.rs` — extend `OperatorKind` and `PtgEdge`.
- `katgpt-rs/crates/katgpt-core/src/closure/functor_edge.rs` (new) — `FunctorEdgeParams`, `apply_functor_edge(state, params, engram_table) -> Vec<f32>` (delegates to a small local `functor_apply` helper — NOT a re-port of riir-ai's `latent_functor`; just the numerics needed for edge application).
- Feature flag: `ptg_functor_edges` (opt-in). Promotion to default requires GOAT gate.

### Why katgpt-rs, not riir-ai

PTG lives in katgpt-rs (public engine). Per the 5-repo strategy, the public engine ships the substrate; the private runtime consumes it. The edge primitive belongs with PTG. riir-ai's full `latent_functor` (Plan 273) has a richer surface (HLA-aware extraction, tropical variants, KARC consumers) — that stays in riir-ai. katgpt-rs only needs the edge-operator concept and the apply path.

The apply path in katgpt-rs can be ~30 LOC: `state' = gate * (state + direction * scale) + (1 - gate) * state` where `gate = sigmoid(beta * (coherence - tau))`. riir-ai can later supply richer `apply_functor` if a benchmark shows the simple form is insufficient (TBD in T6).

### GOAT gate

- **G1 (correctness):** Spec-match test — given a fixed direction set and known state, the functor edge produces a deterministic output. Round-trip: encode → decode → apply produces the same output as direct apply.
- **G2 (perf):** Per-edge apply target < 200 ns (one dot product of D≤64, one sigmoid, one scaled add; 0-alloc mandatory). Bench in `benches/bench_040_ptg_functor_edge.rs`.
- **G3 (no-regression):** Existing PTG tests (closure/admit.rs, closure/bridge.rs, closure/metrics.rs) pass unchanged. The feature is opt-in.
- **G4 (alloc-free):** `apply_functor_into(state, params, table, &mut out)` signature — no `Vec` in the hot path.
- **G5/G6 (modelless):** No training. The direction set is supplied by the caller (extracted offline via `latent_functor::extract_functor` in riir-ai, or supplied by hand). ✅ trivially.

### Wire-compat regression test (mandatory)

T3 must include a fixture PTG serialized BEFORE this feature (capture bytes from `develop` HEAD pre-change) and verify the SAME bytes deserialize + re-serialize identically AFTER the feature lands (with feature OFF and feature ON, both with `functor: None` on all edges).

## Tasks

- [x] **T1** Audit: confirm `latent_functor::apply_functor` signature in riir-ai and decide whether katgpt-rs needs a local lite-version or can depend on a published `katgpt-core::functor_apply` helper (probably the latter, extracted to `crates/katgpt-core/src/functor/apply.rs`).
  - **DONE 2026-07-04 (T1 audit by parallel agent).** Findings below; **a critical wire-format discovery changes the design** — see "T1 Wire-Format Finding".

### T1.1 — riir-ai `latent_functor::apply_functor` signature (confirmed)

- `apply_functor(source, functor, dim, out)` at `riir-ai/crates/riir-engine/src/latent_functor/arithmetic.rs:459` — trivially additive: `out[i] = source[i] + functor[i]` for `i ∈ 0..dim`. ~7 LOC.
- `functor_gate(coherence, beta, tau) -> f32` at line 541 — `sigmoid(beta * (coherence - tau))`. 1 LOC.
- `extract_functor_into(sources, targets, dim, f_out) -> f32` at line 132 — returns coherence f32 (the alignment quality).
- **Decision:** katgpt-rs needs only the edge-apply numerics (cosine coherence + sigmoid gate + SAXPY). No riir-ai dependency; the math is ~20 LOC. The full HLA-aware `latent_functor` (rank-k variants, tropical, KARC consumers) stays in riir-ai.

### T1 Wire-Format Finding — `#[serde(skip_serializing_if, default)]` on `PtgEdge.functor` is BROKEN

**The issue's original wire-compat claim (line 93) is incorrect.** Adding `Option<FunctorEdgeParams>` to `PtgEdge` is NOT backward-compatible, regardless of `skip_serializing_if` / `default` annotations. Verified empirically (2026-07-04):

| Approach | Serialize(None) byte-identical to old? | Deserialize those bytes back? |
|---|---|---|
| Plain `Option<T>` | NO (+1 byte None discriminant) | YES (but wire changed) |
| `#[serde(skip_serializing_if, default)]` | YES (bytes identical) | **NO — "Hit end of buffer"** |

The `skip_serializing_if` approach makes serialization byte-identical BUT **the round-trip fails**: `NewSkip(None) → bytes → NewSkip` errors "Hit end of buffer, expected more data". Postcard is positional — `#[serde(default)]` cannot kick in on EOF because the deserializer doesn't know the field is "missing" vs "next byte is the field". This is a fundamental postcard property, not a serde bug.

**Implication:** ANY design that adds a field to `PtgEdge` (Option A in the issue) changes the wire format and breaks round-trip for existing PTGs. The issue's claim "Existing PTGs serialize identically" is wrong.

### T1 Recommended Design — `FunctorPtg` composite (zero wire impact)

Instead of modifying `PtgEdge`, wrap the unchanged PTG with a parallel functor-params array:

```rust
#[cfg(feature = "ptg_functor_edges")]
pub struct FunctorPtg {
    pub ptg: PrimitiveTransitionGraph,  // byte-identical wire format + commitment
    pub edge_functors: Vec<Option<FunctorEdgeParams>>,  // indexed by edge position
}
```

This preserves wire format 100% (no field added to `PtgEdge`), preserves commitment 100% (PTG bytes unchanged → BLAKE3 unchanged), and follows the existing indirection pattern (`direction_set: [u8;32]` — same as Issue 039's `functor_sig_root`, decoupled from the `engram` feature). The feature flag becomes `ptg_functor_edges = ["closure_instrument"]` (NOT `["closure_instrument", "engram"]`).

**If the current `functor_edge.rs` implementation modifies `PtgEdge` (adds `functor: Option<FunctorEdgeParams>` with `skip_serializing_if`), it MUST be redesigned to the `FunctorPtg` composite before promotion. The wire-format round-trip test (`bare_ptg_bytes_identical_to_inner_ptg_bytes`) will fail otherwise.**

- [ ] **T2** Extend `OperatorKind` with `Functor = 4` variant (gated `ptg_functor_edges`). Extend `PtgEdge` with `pub functor: Option<FunctorEdgeParams>`. Update `PtgRecorder` accordingly.
  - **BLOCKED on T1 design pivot.** With the `FunctorPtg` composite design, `OperatorKind::Functor` is optional (semantic marker only — a functor edge can be `op = Sequence` + `edge_functors[i] = Some(params)`). The `PtgEdge.functor` field MUST NOT be added (wire-format break per T1 finding above). `PtgRecorder` stays unchanged; `FunctorPtg::set_edge_functor(i, params)` replaces `PtgRecorder::exit_functor`.
- [ ] **T3** Wire-format regression: capture pre-change serialization of a 5-node PTG fixture, verify post-change round-trip is byte-identical (with and without feature flag, on `functor: None` edges).
  - **With `FunctorPtg` composite (T1 recommendation):** wire format is byte-identical by construction (no field added to `PtgEdge`). Test: `bare_ptg_bytes == FunctorPtg.ptg` serialized bytes. Trivially passes.
  - **With `PtgEdge.functor` field (old Option A design):** this test WILL FAIL ("Hit end of buffer" per T1 finding). Do not ship.
- [ ] **T4** Implement `apply_functor_edge_into(state, params, engram_table, &mut out)` in `closure/functor_edge.rs`. Zero-alloc.
- [ ] **T5** Spec-match + GOAT bench. Record in `.benchmarks/040_ptg_functor_edge_goat.md`.
- [ ] **T6** If G2 perf target missed (likely — D=64 dot product is ~50ns, but the EngramTable lookup to resolve `direction_index` may dominate): profile and decide whether to cache the direction vector in `FunctorEdgeParams` directly (32 bytes inline, no lookup) vs reference-by-id. Inline wins perf, ref-by-id wins tamper-evidence. Default: inline, with a `FunctorEdgeParamsRef` alternative for commitment use cases.
- [ ] **T7** If G1–G6 pass → promote `ptg_functor_edges` to default. Update closure/mod.rs module docs.

## Non-Goals

- ❌ Porting riir-ai's `latent_functor` to katgpt-rs. Only the edge-apply numerics.
- ❌ Replacing symbolic edges. `Sequence/Branch/Recurse/ParallelJoin` stay. `Functor` is additive.
- ❌ Chain sync of functor edges. That's `riir-chain`'s job once this ships.
- ❌ Making `latent_functor` direction extraction modelless-trainable. That stays in riir-train (Issue 004 split). The edge operator consumes pre-extracted directions.

## Cross-References

- **PTG substrate:** `katgpt-rs/crates/katgpt-core/src/closure/mod.rs` (Plan 290, default-on).
- **Existing bridge (projection only):** `katgpt-rs/crates/katgpt-core/src/closure/bridge.rs::ptg_to_motif_embedding`. Projects PTG → embedding for motif mining; does NOT carry edge-level operators.
- **latent_functor (riir-ai):** `riir-engine/src/latent_functor/arithmetic.rs` (Plan 273). `apply_functor`, `extract_functor_into`, `functor_gate`.
- **Sibling primitive (related, parallel):** Issue 039 (whole-architecture commitment). Once this issue ships, a `FunctorEdgeParams.direction_set` reference can be included in the architecture root.
- **Origin evaluation:** Gemini "Continuous Neuro-Symbolic DAG" proposal review (2026-07-04). The proposal's core insight — "edges should be continuous functors" — is correct; the proposal's structural names (`HlaNode`, `FunctionalEdge`, `NeuroSymbolicDag`) are duplicates of existing atoms (PTG already IS the DAG).

## TL;DR

PTG (Plan 290) is the symbolic execution DAG; `latent_functor` (riir-ai Plan 273) is the continuous transition operator. They don't compose at the edge level — PTG edges are purely symbolic (`OperatorKind ∈ {Sequence, Branch, Recurse, ParallelJoin}`), never functors. This issue tracks adding `OperatorKind::Functor` + `PtgEdge.functor: Option<FunctorEdgeParams>` (direction set ref + β/τ gate scalars) so a PTG edge can carry a continuous latent transition. ~150 LOC, modelless (directions supplied pre-extracted), behind feature flag `ptg_functor_edges`. Wire-format compatible (Option<T> postcard encoding). GOAT-gated; default-on promotion requires G2 perf target < 200ns/edge. High product payoff: per-NPC cognitive-style fingerprinting, cross-task motif transfer, "what was the NPC thinking" replay.
