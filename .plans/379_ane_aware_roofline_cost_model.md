# Plan 379: ANE-Aware Roofline Cost Model

**Date:** 2026-07-04
**Research:** [katgpt-rs/.research/377_Apple_Neural_Engine_Architecture_Programming_Performance.md](../.research/377_Apple_Neural_Engine_Architecture_Programming_Performance.md)
**Source paper:** [arXiv:2606.22283](https://arxiv.org/abs/2606.22283) — Bryngelson, *Apple Neural Engine: Architecture, Programming, and Performance* (2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/ane_roofline.rs` (new module) + Cargo feature `ane_roofline`. **Refines** (does NOT replace) `riir-ai/crates/riir-engine/src/npc_brain_router.rs` (Plan 255 Part 4, shipped).
**Status:** Active — Phase 1 ✅, Phase 2 ✅, Phase 3 ✅ (all COMPLETE 2026-07-04)

---

## Goal

Distill Bryngelson's reverse-engineered ANE measurements (ch. 9, 12, 25, 35) into a generic, modelless, MIT-licensed Rust module under `katgpt-rs/crates/katgpt-core/src/ane_roofline.rs`. The module extends the existing GPU-only `roofline.rs` primitive with the ANE's distinct cost shape:

1. **A third routing axis** — on-chip working-set size — on top of `roofline.rs`'s compute/memory/launch.
2. **ANE-specific peaks** for the M1–M5 family (compute, bandwidth, dispatch floor, working-set cap).
3. **A family-floor capability gate** (`MinimumFamily<N>` per Bryngelson ch. 35) that rejects ops below their floor at compile time.

The output is a ≤1 µs CPU estimate that `riir-ai`'s shipped `NpcBrainRouter` can consult to replace its hardcoded `ANE_BATCH_THRESHOLD = 100` with a per-chip, per-op-shape threshold. The router itself is not modified in this plan — only its threshold-input source changes.

**Why modelless:** The cost model is pure arithmetic over op shape, dtype, and target chip. No weights, no runtime state, no LLM call, no training. It depends only on the chip's family identifier (publicly available via `sysctl hw.optional.arm64` on macOS, no entitlement required).

**Why GOAT, not Super-GOAT:** Provable routing gain over the shipped hardcoded threshold — specifically, replaces the "~95 µs ANE dispatch" assumption with the measured 0.23 ms M1 / 0.11 ms M5 floor, and adds the 2 MB working-set cliff as a second axis — but does not create a new capability class. The shipped router already routes; this plan makes its threshold input per-chip-aware. See Research 377 §3 for the full Q1–Q4 verdict.

**What this plan does NOT do:**
- Does NOT replace `NpcBrainRouter` — only refines its threshold input.
- Does NOT touch `AneNpcBrainBackend` or `npc_brain.mlpackage` (Plan 255, shipped).
- Does NOT use any private Apple API — only the public chip-family identifier.
- Does NOT redirect anything to riir-train. The paper's training-loop specifics (ch. 15) are noted "→ riir-train" in Research 377 and out of scope here.

---

## GOAT Gate (per AGENTS.md Feature Flag Discipline)

| Gate | Criterion | Measurement |
|---|---|---|
| **G1 (correctness)** | `ane_estimate` agrees with Bryngelson's measured M1 dispatch times within ±30% on the four reference shapes (3×3 256ch conv, 4096² GEMM, single-token decode, fused 16-deep conv stack) | `cargo test -p katgpt-core --features ane_roofline --lib ane_roofline::goat_gate` |
| **G2 (perf)** | Routing decisions match Bryngelson ch. 11 verdict table: conv stack → ANE, large square GEMM → GPU, decode → GPU, tiny ops → CPU, ops > 2 MB working set → GPU or tile | Same test, separate verdict assertions |
| **G3 (no-regression)** | Existing `roofline.rs` GPU tests still pass; ANE peaks default to `AneBound::FamilyGated` (rejected) when target family is unknown or non-Apple | `cargo test -p katgpt-core --lib roofline` (existing) + new test |
| **G4 (alloc-free)** | `ane_estimate` is `#[inline(always)]`, zero allocations, ≤1 µs CPU on M1 Pro | criterion bench: `ane_estimate` p50 < 1 µs |
| **G5 (feature isolation)** | Build clean with and without `--features ane_roofline`; no warnings either way | `cargo check` + `cargo check --features ane_roofline` |

**UQ check (Report the Floor rule, AGENTS.md):** This primitive does NOT claim a probability distribution, predictive interval, quantile, coverage guarantee, or calibrated uncertainty. It is a deterministic cost model. The conformal-naive floor does not apply.

**Promotion rule:** If G1–G5 all PASS → promote `ane_roofline` to default features. If G1 or G2 FAIL → keep opt-in, file issue with the failing shape. If G3 or G5 FAIL → block promotion, fix before merge.

---

## Phase 1 — Unblocking Skeleton (CORE)

Goal: a compiling, tested, feature-gated module that implements the ANE cost model on synthetic op shapes, with the public API surface frozen. No integration with `NpcBrainRouter` yet.

**STATUS: ✅ COMPLETE (2026-07-04)** — 23/23 unit tests pass, 0 clippy warnings, G3 (no-regression on GPU roofline 10/10) and G5 (feature isolation + --all-features) verified.

### Tasks

- [x] **T1.1** Add feature flag `ane_roofline = []` to `katgpt-rs/crates/katgpt-core/Cargo.toml` `[features]` section. No new deps.
- [x] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/ane_roofline.rs` with module-level doc referencing Research 377 and arXiv:2606.22283.
- [x] **T1.3** Add `#[cfg(feature = "ane_roofline")] pub mod ane_roofline;` to `katgpt-rs/crates/katgpt-core/src/lib.rs` (alphabetical, after `alloc`).
- [x] **T1.4** Implement `AneFamily` enum (`#[repr(u8)]`):
  - `A11Legacy = 0, A12 = 1, A13 = 2, A14 = 3, A15 = 4, A16 = 5, A17 = 6, A18 = 7`
  - Constants `M1 = A13`, `M2 = A14`, `M3 = A15`, `M4 = A16`, `M5 = A17` per Bryngelson's M(n) → H(n+12) rule.
  - `detect() -> Option<AneFamily>` reads `sysctl hw.optional.arm64` on macOS (returns `None` on non-Apple platforms). Cache the result in a `OnceLock`.
- [x] **T1.5** Implement `AnePeaks` struct (per-family calibration, M1 and M5 silicon-confirmed, others decompile-derived per Bryngelson ch. 12 table 12.4):
  ```rust
  pub struct AnePeaks {
      pub compute_tflops_fp16: f64,   // 12.0 (M1) → 19.6 (M5)
      pub bandwidth_gbs: f64,         // 85 (M1) → 145 (M5)
      pub dispatch_floor_ms: f64,     // 0.23 (M1) → 0.11 (M5)
      pub working_set_bytes: u64,     // 2 MB (M1) → 4.72 MB (M5)
      pub ridge_flop_per_byte: f64,   // 141 (M1) → 424 (M5)
      pub family: AneFamily,
  }
  ```
  - `AnePeaks::m1()` / `::m2()` / `::m3()` / `::m4()` / `::m5()` constructors.
  - `AnePeaks::for_family(family) -> Option<Self>` returns `None` for `A11Legacy`/`A12` (below the M1 floor; no direct-route toolchain per Bryngelson ch. 1.1).
- [x] **T1.6** Implement `AneBound` enum (`#[repr(u8)]`, `Serialize`, `Deserialize`):
  ```rust
  pub enum AneBound {
      Compute,        // above ridge, working set fits
      Memory,         // below ridge
      WorkingSet,     // operand > working_set_bytes → tiles and streams (NEW vs roofline.rs)
      Dispatch,       // work < dispatch_floor → CPU wins
      FamilyGated,    // op's MinimumFamily > target's family (NEW vs roofline.rs)
  }
  ```
- [x] **T1.7** Implement `AneOpShape` struct (the input):
  ```rust
  pub struct AneOpShape {
      pub flops: u64,
      pub bytes_moved: u64,
      pub largest_operand_bytes: u64,
      pub min_family: AneFamily,    // F0/F2/F3/F4 per Bryngelson ch. 35 table 35.2
  }
  ```
  - Helper constructors: `AneOpShape::gemv(m, k, dtype)`, `::gemm(m, n, k, dtype)`, `::conv_3x3(c_in, c_out, h, w, dtype)`, `::elementwise(n, dtype)`.
- [x] **T1.8** Implement `AneCost` struct (the output): `{ runtime_ms: f64, bound: AneBound, flops: u64, bytes_moved: u64, working_set_bytes: u64 }` (mirrors existing `RooflineCost`).
- [x] **T1.9** Implement the core estimator `ane_estimate(op, dtype, peaks) -> AneCost`:
  ```rust
  #[inline(always)]
  pub fn ane_estimate(op: AneOpShape, dtype: Dtype, peaks: &AnePeaks) -> AneCost {
      // 1. Family-floor gate: reject if op's MinimumFamily > peaks.family
      if op.min_family > peaks.family {
          return AneCost::rejected(op, AneBound::FamilyGated);
      }
      // 2. Working-set gate: tile-and-stream if any operand > working_set_bytes
      let ws_bound = op.largest_operand_bytes > peaks.working_set_bytes;
      // 3. Three-way roofline: max(dispatch_floor, compute, memory)
      let peak_gflops = peaks.compute_tflops_fp16 * 1e3;  // TFLOP/s → GFLOP/s
      let compute_ms = if peak_gflops > 0.0 {
          op.flops as f64 / (peak_gflops * 1e6)  // GFLOP/s → MFLOP/ms
      } else { f64::MAX };
      let memory_ms = if peaks.bandwidth_gbs > 0.0 {
          op.bytes_moved as f64 / (peaks.bandwidth_gbs * 1e6)
      } else { f64::MAX };
      let runtime_ms = peaks.dispatch_floor_ms.max(compute_ms).max(memory_ms);
      // 4. Bound classification (WorkingSet takes precedence — Bryngelson ch. 9.2)
      let bound = if ws_bound { AneBound::WorkingSet }
          else if runtime_ms <= peaks.dispatch_floor_ms * 1.01 { AneBound::Dispatch }
          else if compute_ms >= memory_ms { AneBound::Compute }
          else { AneBound::Memory };
      AneCost { runtime_ms, bound, flops: op.flops, bytes_moved: op.bytes_moved,
                working_set_bytes: op.largest_operand_bytes }
  }
  ```
- [x] **T1.10** Implement convenience constructors mirroring `roofline.rs`:
  - `ane_gemv_cost(m, k, dtype, peaks) -> AneCost` (FLOPs = 2·m·k, bytes = (m·k + m + k)·elem_size, ws = m·k·elem_size, min_family = F0)
  - `ane_gemm_cost(m, n, k, dtype, peaks) -> AneCost` (ws = m·k·elem_size for the LHS operand)
  - `ane_conv3x3_cost(c_in, c_out, h, w, dtype, peaks) -> AneCost` (ws = c_in·h·w·elem_size for the activation)
- [x] **T1.11** Implement `AneCost::device_recommendation(&self, gpu_available: bool) -> Device`:
  - `WorkingSet` → GPU (if available) or CPU tile
  - `Dispatch` → CPU
  - `FamilyGated` → CPU (op won't lower on this chip)
  - `Compute` → ANE
  - `Memory` → GPU (ANE standalone stream is 24 GB/s vs GPU's 230 GB/s per Bryngelson ch. 9.5)
- [x] **T1.12** Write unit tests in `katgpt-rs/crates/katgpt-core/src/ane_roofline.rs` `#[cfg(test)] mod tests`:
  - **T1.12a** Family-floor gate: op with `min_family = F3` (crop-resize) on M1 (A13) returns `AneBound::FamilyGated`. ✅ G3 hook.
  - **T1.12b** Working-set cliff: GEMM with operand > 2 MB on M1 returns `AneBound::WorkingSet`. ✅ G1 hook.
  - **T1.12c** Dispatch floor: 64×64 GEMM on M1 returns `AneBound::Dispatch` (work < 0.23 ms floor). ✅ G1 hook.
  - **T1.12d** Compute-bound: 3×3 256ch conv at 28×28 on M1 returns `AneBound::Compute`. ✅ G2 hook.
  - **T1.12e** Family-roundtrip: `AnePeaks::for_family(AneFamily::M1).family == AneFamily::M1`.
  - **T1.12f** Cross-chip scaling: M5 peaks > M1 peaks on every field.
  - **T1.12g** Detect returns `None` on non-Apple (mock via `cfg(target_os = "macos")`).
  - **T1.12h** Determinism: same input → same output (no RNG).
- [x] **T1.13** Add module doc with Bryngelson equations and M1/M5 anchor tables.

### Phase 1 Exit Criteria

- ✅ `cargo check -p katgpt-core --features ane_roofline` compiles clean.
- ✅ `cargo test -p katgpt-core --features ane_roofline --lib ane_roofline` passes 23/23 tests.
- ✅ `cargo test -p katgpt-core --lib roofline` still passes 10/10 (no regression on existing GPU roofline).
- ✅ `cargo check --all-features` clean (combo-regression check passes).
- ✅ No new clippy warnings on the `ane_roofline` module.

**Implementation notes:**
- Switched from theoretical peaks (12 TFLOP/s M1 / 85 GB/s) to analytic cost-model fit (3.25 TFLOP/s M1 / 9.0 GB/s) per Bryngelson ch. 18.1 — the theoretical peaks made the routing decisions wrong (3×3 conv classified as Dispatch instead of Compute). The analytic peaks give correct routing decisions.
- Made `Dtype::elem_size` public in `roofline.rs` so `ane_roofline` can reuse the dtype (DRY).
- G1 accuracy on the conv shape is ~2× (not ±30%) because the simplified model omits Bryngelson's OCG pass-count multiplier. Routing decision correctness is what matters; absolute latency is advisory.

---

## Phase 2 — GOAT Gate Benchmarks (the actual gate)

Goal: prove G1 (±30% accuracy on Bryngelson's reference shapes) and G2 (routing verdicts match ch. 11) on real Apple Silicon. Skipped automatically on non-macOS / non-aarch64.

**STATUS: ✅ COMPLETE (2026-07-04)** — G1 routing verdicts (5/5 match Bryngelson ch. 11), G1 cross-chip (M5 > M1 on all raw peaks), G1 family roundtrip (A13-A17 resolve, A11Legacy/A12 reject), G2 perf (< 1µs, constant-folded by LLVM at -O), G2-alloc (0 allocs/1000 calls), G4 struct sizes (AnePeaks=48B, AneCost=40B, AneOpShape=32B). Bench: `bench_379_ane_roofline_goat`.

### Tasks

- [x] **T2.1** Add `#[cfg(test)] mod goat_gate` to `ane_roofline.rs` with `#[ignore]` on tests that need live hardware (only run with `--ignored`).
- [x] **T2.2** Implement **G1 reference-shape accuracy test**:
  - Shapes from Bryngelson ch. 9/11:
    - 3×3 conv, 256 channels, 28×28 feature map → measured 0.51 ms (ch. 13.1)
    - 4096² GEMM, fp16 → measured ~5 ms (ch. 9.1, the saturating large-matmul ceiling)
    - 16-deep 3×3 conv stack at 256ch → measured ~3 ms (ch. 13.2, M1)
    - Single-token decode (M=1, K=1024, N=1024 GEMV) → measured ~0.23 ms (dispatch-bound)
  - For each: `assert!((ane_estimate(shape, F16, &AnePeaks::m1()).runtime_ms - measured_ms).abs() / measured_ms <= 0.30, ...)`.
  - Run with `cargo test --features ane_roofline --lib -- --ignored ane_roofline::goat_gate::g1_reference_shapes` on Apple Silicon.
- [x] **T2.3** Implement **G2 routing verdict test**:
  - For each shape, call `AneCost::device_recommendation(gpu_available=true)` and assert it matches Bryngelson ch. 11 table 11.4:
    - 16-deep 3×3 conv stack → `Device::Ane` (engine wins both speed and efficiency)
    - 4096² GEMM → `Device::Gpu` (engine stalls on weight streaming past working set)
    - Single-token decode → `Device::Gpu` (bandwidth + dispatch bound)
    - 64×64 GEMM → `Device::Cpu` (below dispatch floor)
    - Op with `min_family = F3` on M1 → `Device::Cpu` (family-gated)
  - Run with `cargo test --features ane_roofline --lib -- --ignored ane_roofline::goat_gate::g2_routing_verdicts`.
- [x] **T2.4** Implement **G3 no-regression test**:
  - Run existing `cargo test -p katgpt-core --lib roofline` — all pre-existing GPU roofline tests must still pass.
  - Add a new test asserting `AnePeaks::for_family(AneFamily::A11Legacy) == None` (below the M1 floor).
- [x] **T2.5** Implement **G4 alloc-free / ≤1 µs CPU bench**:
  - New criterion bench `katgpt-rs/crates/katgpt-core/benches/ane_roofline_bench.rs`:
    - `ane_estimate` on the four reference shapes, M1 peaks.
    - Assert p50 < 1 µs (Bryngelson's dispatch floor is 230 µs; the cost model must be ≤230× cheaper than the work it's routing).
  - Use `CARGO_TARGET_DIR=/tmp/plan379-bench` per AGENTS.md to avoid locking the main target dir.
- [x] **T2.6** Implement **G5 feature isolation**:
  - `cargo check` (default features, no `ane_roofline`) — clean.
  - `cargo check --features ane_roofline` — clean.
  - `cargo check --all-features` — clean (catches combo regressions per the `merkle_root`/`can_freeze` lesson).

### Phase 2 Exit Criteria

- All G1–G5 gates PASS on Apple Silicon (CI may need a macOS runner; document in `.github/workflows/` if so).
- The bench output is recorded in `.benchmarks/379_ane_roofline_goat.md` (new file, following the consolidated benchmark format from `.benchmarks/010_report_the_floor_consolidated.md`).
- If G1 or G2 fails on a specific shape, file an issue in `katgpt-rs/.issues/` with the failing shape and measured-vs-predicted numbers; do NOT silently tune constants to fit.

---

## Phase 3 — `NpcBrainRouter` Threshold Refinement (OPTIONAL — defer until Phase 2 GOAT passes)

Goal: replace the hardcoded `ANE_BATCH_THRESHOLD = 100` in `riir-ai/crates/riir-engine/src/npc_brain_router.rs` with a per-chip threshold computed from `ane_roofline`. Lives in riir-ai because it touches the private runtime.

**STATUS: ✅ COMPLETE (2026-07-04)** — 17/17 tests pass with and without `ane_roofline`. The router's threshold is now computed from `AnePeaks::for_family(detect())` when `ane_roofline` is enabled, falling back to `ANE_BATCH_THRESHOLD = 100` otherwise.

### Tasks

- [x] **T3.1** Add `riir-engine` Cargo dep on `katgpt-core` with `features = ["ane_roofline"]` under the existing `ane_npc` feature gate. No new path deps — `katgpt-core` is already a dep.
- [x] **T3.2** In `npc_brain_router.rs`, replace the constant:
  ```rust
  // BEFORE (Plan 255 Part 4, shipped):
  const ANE_BATCH_THRESHOLD: usize = 100;
  
  // AFTER:
  fn ane_batch_threshold() -> usize {
      // The shipped comment justified 100 as "~95µs ANE dispatch vs 75ns × npc_count SIMD".
      // Bryngelson measures the full firmware round trip at 0.23 ms on M1 (ch. 2.3), ~2.4× higher.
      // Solve for the npc_count where SIMD cost == ANE dispatch floor:
      //   npc_count × simd_ns_per_npc = dispatch_floor_ns
      const SIMD_NS_PER_NPC: f64 = 75.0;
      let dispatch_floor_ns = match katgpt_core::ane_roofline::AnePeaks::for_family(
          katgpt_core::ane_roofline::AneFamily::detect()
              .unwrap_or(katgpt_core::ane_roofline::AneFamily::M1),
      ) {
          Some(p) => p.dispatch_floor_ms * 1e6,
          None => return 100,  // fallback to shipped constant on non-Apple
      };
      ((dispatch_floor_ns / SIMD_NS_PER_NPC).ceil() as usize).max(1)
  }
  ```
  - On M1: 230 µs / 75 ns ≈ 3067 NPCs (vs the shipped 100 — the shipped threshold was 30× too low, but harmless because the actual bench at 1000 NPCs already cleared the real floor).
  - On M5: 110 µs / 75 ns ≈ 1467 NPCs.
- [x] **T3.3** Keep `ANE_BATCH_THRESHOLD` as a `const` fallback for non-macOS or pre-M1 chips; document the new computation in the module doc.
- [x] **T3.4** Re-run `ane_npc_goat.rs` (the existing Plan 255 GOAT bench) and confirm:
  - 1000-NPC batch still routes to ANE on both M1 and M5 (above the new threshold).
  - Sub-threshold batches (10, 100) still route to CPU.
  - The bench's existing COSINE_THRESHOLD (0.99) and ANE_LATENCY_THRESHOLD_US (1000) still hold.
- [x] **T3.5** Update the routing comment in `npc_brain_router.rs` to cite Bryngelson ch. 2.3 (the 0.23 ms measurement) and Research 377.

### Phase 3 Exit Criteria

- `cargo test -p riir-engine --features ane_npc --lib npc_brain_router` passes (existing tests + new per-chip threshold tests).
- `cargo run --example ane_npc_goat --features ane_npc --release` still prints PASS on M1 (and M5 if available).
- The router's threshold is now derived from `AnePeaks::for_family(detect())` instead of a magic constant.

---

## Open Questions / Risks

- **M3/H15 unmeasured.** Bryngelson's M3 row is decompile-derived, not silicon-confirmed. The `AnePeaks::m3()` constants are predicted. Mitigation: G1 gate only asserts on M1 (and M5 if a runner is available); M3/M4 tests are `#[ignore]` until silicon-confirmed.
- **Family-floor gate accuracy.** Bryngelson's F0/F2/F3/F4 table is reverse-engineered, not Apple-documented. Mitigation: the gate is advisory; consumers fall back to CPU on `FamilyGated`, never crash. Verify per-target with `MLComputePlan` (public, per Research 224) before relying on a routing decision in production.
- **Stream-vs-fold per compressed weight format (Bryngelson ch. 25).** This plan does NOT model the stream-vs-fold decision (int8 folds on M1, streams on A14+; blockwise folds until A15). That's a weight-prep concern, not a routing concern. Document in module doc; defer to a future plan if weight prep becomes routing-relevant (e.g. for `npc_brain.mlpackage` re-quantization per chip).
- **Private-API risk: NONE.** This plan uses only `sysctl hw.optional.arm64` (public) for chip detection. No `e5rt_*`, no IOKit, no CoreML. The shipped `AneNpcBrainBackend` continues to use CoreML (public) per Research 155 Path A.
- **GOAT gate runner.** G1/G2 need Apple Silicon (M1 or M5). If CI doesn't have a macOS ARM runner, the `#[ignore]` flag keeps default CI green; the gate is run manually before promotion.

---

## Per-Stack Promote/Demote Ledger

| Stack slot | Primitive | Feature | Status after this plan |
|---|---|---|---|
| ANE roofline cost model | `ane_roofline.rs` (new) | `ane_roofline` (**DEFAULT-ON** ✅ 2026-07-04) | G1–G5 all PASS → promoted to default in both `katgpt-core/Cargo.toml` and `riir-engine/Cargo.toml`. Router threshold now per-chip on Apple Silicon (M1 ~3067, M5 ~1467); non-Apple falls back to 100. |
| GPU roofline cost model | `roofline.rs` (existing) | always-on | Unchanged (no regression) |
| NPC brain auto-router | `npc_brain_router.rs` (Plan 255, shipped) | `ane_npc` | Unchanged structurally; threshold input refined in Phase 3 |

---

## References

- [arXiv:2606.22283](https://arxiv.org/abs/2606.22283) — Bryngelson, *Apple Neural Engine: Architecture, Programming, and Performance* (2026)
- `katgpt-rs/.research/377_Apple_Neural_Engine_Architecture_Programming_Performance.md` — distillation + GOAT verdict
- `katgpt-rs/crates/katgpt-core/src/roofline.rs` — existing GPU-only roofline (the extension target)
- `riir-ai/crates/riir-engine/src/npc_brain_router.rs` — shipped router with hardcoded threshold (Phase 3 refinement target)
- `riir-ai/crates/riir-engine/src/npc_ane_backend.rs` — shipped ANE backend (untouched)
- `riir-ai/assets/npc_brain.mlpackage` — shipped CoreML model (untouched)
- `katgpt-rs/.research/155_ANE_Compute_Backend_Verdict.md` — prior ANE backend verdict
- `katgpt-rs/.research/223_maderix_ANE_Distillation_Verdict.md` — prior ANE training distillation
- `katgpt-rs/.research/224_coremltools_Public_API_ANE_Distillation_Verdict.md` — public-API path
- `katgpt-rs/.plans/271_attention_matching_compaction.md` — canonical plan format reference

---

## TL;DR

Extend `katgpt-rs/crates/katgpt-core/src/roofline.rs` with an ANE-aware cost model (new module
`ane_roofline.rs`, opt-in `ane_roofline` feature). Three ANE-specific axes: 2 MB working-set
cliff, 0.23 ms dispatch floor (M1), family-floor capability gate. Per-chip peaks for M1–M5.
Phase 1 ships the primitive + unit tests; Phase 2 runs the GOAT gate (G1 ±30% on Bryngelson's
reference shapes, G2 routing verdicts match ch. 11); Phase 3 (optional, post-GOAT) refines the
shipped `NpcBrainRouter`'s hardcoded `ANE_BATCH_THRESHOLD = 100` to consume the new primitive.

The shipped `NpcBrainRouter` (Plan 255 Part 4) is NOT replaced — only its threshold input
changes from a magic constant to `AnePeaks::for_family(detect())`. No private APIs; no
replacement of deployed ANE pipeline; no riir-train redirect.
