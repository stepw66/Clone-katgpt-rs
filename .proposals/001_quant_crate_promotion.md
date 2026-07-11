# Proposal 001 — Promote the rotation/codebook quant family to `katgpt-quant`

Status: **SUPERSEDED** by Proposal 003 (`003_src_consolidation_master.md`), Phase 1.
The quant promotion is absorbed unchanged into the master plan.
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned

## TL;DR

Five modelless KV-cache quantization modules marooned in `katgpt-rs/src/`
share one cohesive domain (rotation + codebook compression), an identical
module shape, and a clean dependency closure. Promote them into a single
new workspace crate **`katgpt-quant`**, benchmark the family under one
roof, and promote the GOAT variant to default-on.

The five modules:

| Module | Files | Method | Perf signature |
|---|---|---|---|
| `turboquant/` | 6 | Random rotation (O(d²) WHT) + Lloyd-Max codebook | baseline; 8–16× KV compression |
| `planar_quant/` | 4 | 2D Givens rotation, O(d) vs TurboQuant's O(d²) | 256 FMAs for d=128 (vs 16,384) |
| `iso_quant/` | 4 | 4D quaternion rotation | 512–1024 FMAs for d=128 |
| `hybrid_oct_pq/` | 3 | planar rotation + octopus encoding hybrid | near-OCTOPUS MSE at planar speed |
| `octopus/` | 8 | octahedral triplet encoding | pulled in by `hybrid_oct_pq` (leaf, zero katgpt deps) |

**Total closure: 5 modules, 25 files.**

## Why this is the GOAT next promotion

1. **`katgpt-spectral` sets the template and pre-declares intent.** The
   spectral crate was spun out of `katgpt-rs/src/spectralquant/` (Issue 015
   Phase 2) for the same reason. Its `Cargo.toml` `[features]` block
   literally lists `turboquant = []` as a tracking flag, and its `lib.rs`
   line 54 already does `#[cfg(feature = "turboquant")] pub use
   spectral_rotation::RandomRotation;` — TurboQuant's random rotation is
   *already partially absorbed* into the spectral crate. The intent to
   consolidate quant under named crates is on record.

2. **Modelless mandate compliance — clean.** All five modules are
   deterministic: random rotation matrices, Givens angles, quaternion
   Hamilton products, Lloyd-Max codebooks, octahedral triplet packing.
   No training, no backprop, no gradient descent. Promotion requires
   modelless gain (per `AGENTS.md` feature-flag discipline) — this family
   is modelless by construction, so the GOAT gate's quality bar (G1) is
   satisfiable without riir-train.

3. **Clean dependency closure.** Internal cross-module deps form a DAG:
   ```
   octopus (leaf, zero katgpt deps)
     ↑
   hybrid_oct_pq ──uses──> planar_quant::rotation
     ↑                       │
     │                       └──uses──> turboquant::codebook
     │                                       ↑
   iso_quant ─────────────────────────────────┘ (reuses codebook + types)
   planar_quant ───────────────────────────────┘ (reuses codebook)
   ```
   External deps are only `katgpt_core::simd` (`simd_scale_inplace`,
   `simd_sum_sq`) and `crate::types::Rng`. No transformer-runtime coupling,
   no attention coupling, no KV-cache-trait coupling beyond the local
   `*KVCache` structs each module defines.

4. **Identical module shape** (`rotation` + `kv_cache` + `types` +
   optional `codebook`/`forward`) → trivial to unify under one crate with
   per-variant feature gates mirroring the existing root flags.

5. **They are a comparison family.** planar vs iso vs hybrid vs turbo vs
   octopus trade off rotation cost vs MSE. They belong in one crate so
   the benchmark suite can A/B them under a shared harness — exactly the
   GOAT-gate workflow (G2 perf, G3 no-regression).

## Naming decision: new `katgpt-quant`, not extend `katgpt-spectral`

`katgpt-spectral`'s `Cargo.toml` description is specifically *"calibrated
eigenbasis KV cache compression ... Lloyd-Max / water-fill bit allocation,
outlier-aware guard."* The rotation family is structurally different
(Givens / quaternion / octahedral, not eigenbasis). Folding them in would
muddy the spectral crate's identity and force its non-KV consumers
(`funcattn_compose`, `chiaroscuro`, `benchmark`) to pull rotation code
they don't use.

**Decision:** new crate `katgpt-quant`. `katgpt-spectral` keeps the
eigenbasis path; `katgpt-quant` owns the rotation/codebook path. Both
re-export through `katgpt-rs` for back-compat (mirroring Issue 015 Phase 5).

## Proposed crate shape

```
crates/katgpt-quant/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs                  # re-exports per feature
    ├── turboquant/             # mod.rs, codebook.rs, forward.rs, kv_cache.rs, rotation.rs, types.rs
    ├── planar_quant/           # mod.rs, kv_cache.rs, rotation.rs, types.rs
    ├── iso_quant/              # mod.rs, kv_cache.rs, rotation.rs, types.rs
    ├── hybrid_oct_pq/          # mod.rs, kv_cache.rs, types.rs
    └── octopus/                # mod.rs, codebook.rs, encode.rs, forward.rs, kv_cache.rs, octahedral.rs, triplet.rs, types.rs
```

### `Cargo.toml` skeleton

```toml
[package]
name = "katgpt-quant"
version = "0.1.0"
edition = "2024"
license = "MIT"
description = "Rotation/codebook KV-cache quantization family — TurboQuant (random rotation + Lloyd-Max), PlanarQuant (2D Givens, O(d)), IsoQuant (4D quaternion), OCTOPUS (octahedral triplet), HybridOctPq (planar + octopus). All modelless. Spun out of katgpt-rs/src/{turboquant,planar_quant,iso_quant,hybrid_oct_pq,octopus}/."
repository = "https://github.com/katopz/katgpt-rs"
keywords = ["quantization", "kv-cache", "rotation", "lloyd-max", "modelless"]
categories = ["algorithms", "science"]
publish = false  # mirror katgpt-attn-match — only katgpt-core ships to crates.io

[dependencies]
katgpt-core = { path = "../katgpt-core" }   # simd kernels (simd_scale_inplace, simd_sum_sq)
katgpt-types = { path = "../katgpt-types" } # Rng, shared types

[features]
default = []

# Per-variant gates mirroring the historical root feature surface.
turboquant      = []
planar_quant    = []
iso_quant       = []
hybrid_oct_pq   = []
octopus         = []
```

### Root `katgpt-rs` wiring (back-compat re-export)

Mirror Issue 015 Phase 5: the root crate re-exports `katgpt-quant` under
the existing module paths (`katgpt_rs::turboquant`, `katgpt_rs::planar_quant`,
etc.) so no external consumer breaks.

## GOAT gate (must pass before default-on promotion)

Per `AGENTS.md` feature-flag discipline:

- [ ] **G1 correctness** — each variant's dequantized output matches the
      f32 reference within its documented MSE bound on the standard KV
      fixture. No silent correctness regression vs the in-tree baseline.
- [ ] **G2 perf** — benchmark each variant's encode/decode FMAs and wall
      time against TurboQuant (the baseline). planar/iso/hybrid must beat
      TurboQuant on rotation cost; octopus must beat on MSE at equal bits.
- [ ] **G3 no-regression** — `cargo check --all-features` clean; the
      existing `turboquant`/`planar_quant`/etc root tests pass unchanged
      against the re-exported crate.
- [ ] **G4 alloc-free hot path** — encode/decode inner loops stay
      allocation-free (scratch buffers passed as `&mut [T]`, per the
      hot-loop rules). No new `Vec` allocations inside the quant kernels.

Only after G1–G4 pass AND the gain is modelless → promote the winning
variant to `default = [...]`. If the gate ties (e.g. planar and iso both
win on different d), keep both opt-in and document the decision matrix.

## Phased rollout

- [ ] **Phase 0 — verify closure.** Grep for any *other* consumer of
      these five modules outside the closure (e.g. `benchmark/`, `examples/`).
      Update call sites to the new crate path. Confirm no surprise dep.
- [ ] **Phase 1 — scaffold crate.** Create `crates/katgpt-quant/`, copy
      the five modules verbatim, fix `use crate::` → `use katgpt_quant::`
      + adjust the inter-module paths. `cargo check -p katgpt-quant
      --all-features` clean.
- [ ] **Phase 2 — root re-export.** Add `katgpt-quant` to root
      `Cargo.toml` deps; replace `mod turboquant;` etc in `src/lib.rs`
      with `pub use katgpt_quant::turboquant;`. All existing root tests
      and examples compile unchanged.
- [ ] **Phase 3 — delete in-tree copies.** Remove `src/turboquant/`,
      `src/planar_quant/`, `src/iso_quant/`, `src/hybrid_oct_pq/`,
      `src/octopus/`. `cargo check --workspace --all-features` clean.
- [ ] **Phase 4 — GOAT benchmark.** Add a bench comparing all five
      variants on the standard KV fixture (rotation FMAs, encode/decode
      wall time, dequant MSE). Run G1–G4.
- [ ] **Phase 5 — promote GOAT to default.** If a variant wins on perf
      with no MSE regression, promote to `default`. Demote losers to
      opt-in with a one-line rationale in `Cargo.toml`.
- [ ] **Phase 6 — commit + record.** Commit on `develop` with
      `feat:` prefix. Update this proposal status to **done**. Cross-link
      from `katgpt-spectral`'s README (the `turboquant = []` tracking flag
      in its Cargo.toml can now point here).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Hidden consumer of `crate::turboquant` outside the 5-module closure | Phase 0 grep; if found, either bring it into the crate or keep a thin root shim |
| `katgpt-spectral` already re-exports `RandomRotation` under `turboquant` feature | Keep that re-export working — `katgpt-quant` re-exports the same type; spectral's flag becomes a passthrough |
| Planar/iso reuse `turboquant::codebook` — moving them must preserve that path | Internal `use` paths inside the new crate; no external surface change |
| Benchmark shows no clear winner (planar vs iso trade off by d) | Keep both opt-in; document the decision matrix in the crate README rather than forcing a default |

## Out of scope (tracked in issues)

- `mux_latent/` promotion (12 files) — fuzzy MUX dep boundary, deferred.
  See `issues/001_deferred_promotion_candidates.md`.
- `proof_cert/` promotion (7 files) — cross-cuts chain/WASM runtime,
  deferred. Same issue.
- `dash_attn/` promotion (16 files) — separate, larger lift; deserves its
  own proposal (002) if pursued.

## References

- `katgpt-spectral` precedent: `crates/katgpt-spectral/Cargo.toml` +
  `src/lib.rs` (Issue 015 Phase 2/5).
- Modelless mandate: `katgpt-rs/AGENTS.md` §"Modelless-first mandate".
- GOAT gate: `katgpt-rs/AGENTS.md` §"Feature Flag Discipline".
- Feature-flag discipline + promotion rule: same section.
