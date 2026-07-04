# Research 377: Apple Neural Engine — Architecture, Programming, and Performance

> **Source:** Spencer H. Bryngelson, *Apple Neural Engine: Architecture, Programming, and Performance*,
> arXiv:2606.22283 [cs.AR], 21 Jun 2026 (302-page reverse-engineering reference).
> **Date:** 2026-07-04
> **Status:** Active
> **Related Research:** katgpt-rs 155 (ANE Backend Verdict), 223 (maderix/ANE Distillation),
> 224 (coremltools Public API), 276 (Personality-Weighted Composition, the
> `InferenceBackend` trait pattern); riir-ai 045 (ANE Compute Backend Verdict),
> 044 (frame-sampling), 121 (Attention Matching Game AI Fusion → Plan 297 AM core)
> **Related Plans:** katgpt-rs 175 (ANE-Inspired DDTree), 176 (GPU/ANE Trigger Gate),
> 255 (ANE-Latent NPC Brain Compute — **shipped**, includes the `npc_brain.mlpackage`
> model and `NpcBrainRouter`); riir-ai 297 (AM Game AI Fusion)
> **Shipped code (DO NOT duplicate):** `riir-ai/assets/npc_brain.mlpackage` (deployed
> CoreML ML Program), `riir-ai/crates/riir-engine/src/{ane_backend.rs,
> npc_ane_backend.rs, npc_brain_router.rs}` (load + dispatch + auto-route),
> `riir-ai/crates/riir-engine/examples/ane_npc_{arena,goat,power}.rs` (GOAT bench),
> `katgpt-rs/scripts/generate_npc_brain_model.py` (model generator)
> **Classification:** Public

---

## TL;DR

This paper is a 302-page hardware reference work (not an ML paper) that reverse-engineers the
Apple Neural Engine (ANE) from the silicon datapath up to the system interface, covering A11–A18
and M1–M5. It is the most thorough public account of the ANE to date.

For our codebase it is **incremental**, not paradigm-shifting. We already ship an ANE
pipeline end-to-end: `riir-ai/assets/npc_brain.mlpackage` is a deployed CoreML ML Program
(three fused ops: sense projection, emotion projection, zone projection) loaded by
`AneNpcBrainBackend`, dispatched via CoreML `ComputeUnits::All`, and auto-routed by
`NpcBrainRouter` against a hardcoded `ANE_BATCH_THRESHOLD = 100` (Plan 255 Part 4,
shipped, with GOAT bench in `ane_npc_goat.rs`). On top of that we have three prior ANE
verdicts (155/223/224) and a GPU-only `roofline.rs` primitive.

What this paper adds that we do **not** already cover:

1. **The 2 MB on-chip working-set threshold as a hard cliff** (HAL field `0x1b8`) — operands above
   this tile and stream from DRAM, collapsing arithmetic intensity. Our `roofline.rs` has no
   working-set axis; it only models compute/memory/launch.
2. **The 0.23 ms dispatch floor** (M1/H13) — ~4–5× higher than our GPU `launch_overhead_us = 50µs`.
   The shape of the cost curve is fundamentally different from GPU.
3. **The family-floor capability table** (F0/F2/F3/F4) — operation legality declared as a
   `MinimumFamily<N>` trait; nothing floors above F4 (A15+).
4. **Stream-vs-fold per compressed-weight format** — M1 streams only int4-LUT (2.37×) and sparse
   (1.55–1.64×); int8 and blockwise *fold* (no bandwidth gain). A14+ adds int8 stream, A15+ adds
   blockwise stream.
5. **Wide accumulator numerics** — fp32-class, radix-4 first-stage input tile. Slice with nonzero
   width-axis offset × 16 saturates above 4094 on M1/M2. MAC output port saturates at 2^15 = 32768.
6. **Resident state via output-to-input buffer aliasing** (M1 `share_buffer`) — the hardware
   substrate for KV cache and freeze/thaw state on M1, pending native counter-and-event engine
   (A15+).
7. **The M(n) → H(n+12) naming rule** and the 28-target compiler family.

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is an **ANE-aware roofline cost model** that refines the existing
shipped `NpcBrainRouter`'s hardcoded `ANE_BATCH_THRESHOLD = 100` (Plan 255) into a per-chip,
per-op-shape threshold. The shipped router's comment justifies the constant as "~95µs ANE
dispatch overhead vs 75ns × npc_count SIMD cost". Bryngelson measures the *full* firmware round
trip at **0.23 ms** on M1 — ~2.4× higher than the shipped router assumes — and identifies the
**2 MB working-set cliff** and **family-floor capability gate** as additional axes the shipped
router does not model. Extending `roofline.rs` with ANE peaks + these three axes produces the
input `NpcBrainRouter` needs to make per-chip routing decisions instead of one global constant.

This is a **GOAT**, not Super-GOAT: it produces a provable perf gain (more accurate routing,
fewer misplacements on small/dispatch-bound ops or ops near the 2 MB cliff) over the current
hardcoded threshold, but does not create a new capability class. The product selling point is
"our router now knows the difference between M1 (0.23 ms floor, 2 MB working set) and M5
(0.11 ms floor, 4.72 MB working set)" — useful, not moat-defining.

---

## 1. Paper Core Findings

### 1.1 The machine (Part I, chs. 1–4)

- **Fixed-function fp16 MAC array with wide accumulator.** Inputs/weights round to fp16 going in,
  output rounds to fp16 going out, but the running sum is held in a register of fp32 class. The
  first reduction stage groups input lanes into tiles of four (radix-4), each rounded to fp16,
  which then feed the wide accumulator. A `[4096] + [1]×1024` reduction returns 5116 (between
  naive-fp16 4096 and exact 5120), proving the accumulator is wider than fp16 but inputs are
  pre-rounded.
- **MAC output port saturates at 2^15 = 32768**, not the fp16 ceiling 65504. Half the storage
  range. A `linear` with output magnitude > 32768 silently returns inf regardless of K.
- **Width-axis slice × 16 saturation.** A nonzero begin offset on the width axis routes through a
  crop DMA with a fixed Q.4 gain of 16. Values above `65504/16 = 4094` overflow to ±inf on M1/H13
  and A14/H14. Clean route arrives A15+. This is a finite-to-infinity cliff, not a ULP drift.
- **NaN coerced to +∞ at input boundary.** Engine never produces a NaN anywhere.
- **Activation LUT: 33-knot piecewise-linear.** Worst abs error: sigmoid 0.0034, tanh 0.0017,
  gelu 0.0059. Origin biases: gelu(0) = −0.000543, swish(0) = −0.001259 (below fp16 floor).
- **Bit-deterministic** for fixed graph+input across reruns, recompiles, and fresh processes.
  Cross-generation divergence is at most one ULP, set by tiling-boundary alignment.

### 1.2 Reaching the engine (Part II, chs. 5–8)

- **Direct route via Espresso runtime (`e5rt_*`)** is reachable from ordinary user space below
  Core ML with no entitlement for accepted operations.
- **Compile-once-dispatch-many.** Compile phase is costly and out of the hot loop; dispatch binds
  operands, posts one mailbox command, waits. Programs are content-addressed on disk.
- **Resident state via output-to-input buffer aliasing** (M1):
  - Allocate one `e5rt_buffer_object_t`, bind it to BOTH the output port of step N and the input
    port of step N+1. The tensor persists in the engine's working set across dispatches with no
    host re-supply.
  - This is the M1 substrate for KV cache, optimizer state, and any resident accumulator.
  - Native counter-and-event DMA engine is stubbed in the M1 descriptor; arrives on later
    generations.
- **Loaded-program cap per process ≈ 128**; in-flight cap 127 per program.
- **Gated features on direct path** (none run on M1 via either route): 3D conv (no backend
  lowering), native stateful types (counter-event engine stubbed), bf16 I/O (not in 11 dtype
  codes), symbolic shapes (parse then fail to lower).

### 1.3 Performance (Part III, chs. 9–12)

**M1/H13 roofline anchors:**

| Quantity | Symbol | M1 value |
|---|---|---|
| Compute roof (overhead-isolated matmul slope) | P | 12 fp16 TFLOP/s |
| Compute roof (saturating large matmul) | — | 4.8 fp16 TFLOP/s |
| Conv end-to-end ceiling | — | 1.8 TFLOP/s |
| DRAM bandwidth ceiling | B | 85 GB/s |
| Standalone activation stream rate | — | 24 GB/s |
| Saturating weight-stream wall clock | — | 51 GB/s |
| **Roofline ridge point** | I* | **141 FLOP/byte** |
| **On-chip working-set threshold** | HAL `0x1b8` | **2 MB** |
| **Per-dispatch floor** | t0 | **0.23 ms** |
| Fused per-dispatch cost (1 MB operands) | — | 0.76 ms |
| Energy per FLOP | — | 0.5 pJ sustained, 0.37 pJ optimum |
| NE cores (HAL `0x238`) | — | 4 (base), 8 (Pro/Max) |
| Marketing core count | — | 16 (different quantity) |

**Critical shape difference from GPU:**

- GPU ridge (M5/H17s) ≈ 134 FLOP/byte; ANE ridge ≈ 424 FLOP/byte (3× higher).
- A standalone elementwise stream on ANE reaches only 24 GB/s; GPU reaches 230 GB/s.
- A standalone `layer_norm` reaches 18% of ANE's bandwidth roof; `softmax` 63% — dispatching them
  standalone leaves the engine mostly idle.
- **Fusion moves the operating point past the ridge.** A 3-conv fused stack reaches 20718 GFLOP/s
  (110% of single-conv saturation peak 18771 GFLOP/s).

**Cross-generation scaling:**

- Naming rule: **M(n) → H(n+12)**. M1 = H13, M5 = H17s.
- Operation surface stops expanding at A15. A16/A17/A18 add cores, not operations.
- Working-set threshold scales: 2 MB (M1) → 4.72 MB (M5).
- Per-dispatch floor: 0.23 ms (M1) → 0.11 ms (M5).
- Weight-stream bandwidth: 51 GB/s (M1, one channel) → 145 GB/s (M5, two channels).
- Cross-generation training parity: 0.9080 (M1) vs 0.9070 (M5) — one sample in a thousand.

### 1.4 Workloads (Part IV, chs. 13–16)

- **Conv stack (16× 3×3, 256ch)**: 2× faster + 14.5× more efficient than GPU on M1; 4.2× faster +
  13× more efficient on M5.
- **Single-sentence encoder**: 4.4× faster than GPU. Throughput crossover to GPU near batch 23;
  self-attention crossover near batch 6. Energy crossover never appears for vision conv.
- **Decode is NOT a compute problem** — bandwidth-bound + dispatch-bound. At batch 16: GPU 2.7×
  faster, 4.6× more efficient. Per-token step issues 40–50 small dispatches, each paying 0.23 ms
  floor.
- **Hybrid decoder placement**: Q/K/gate/up/output-embedding projections fit ANE in fp16; value,
  output projection, and down-projection need wider precision off-engine (cancellation amplifies
  fp16 input rounding).
- **Training on the engine**: no backward op — backward built from forward ops as a second graph,
  optimizer state resident via buffer aliasing. Conv weight-gradient saturates above 4094 on M1
  when loss scale × input magnitude crosses the threshold.

### 1.5 Practice (Part V, chs. 17–19)

- **Validator limits** (compile-time rejects):
  - Width/Height ≤ 16384 (M1) or 65536 (M5)
  - Channel ≤ 65536
  - Conv fp16 kernel width ≤ 13 (M1) or 16 (M5)
  - Arg-min/arg-max reduction axis ≤ 2048
  - Matmul operand depth = 1 on both operands
- **Working-set rule**: keep largest live operand ≤ 2 MB or it tiles and streams.
- **Channel-interleave rule**: channel counts should be multiples of interleave factor or lanes
  are padded and wasted.
- **Five direct-path pitfalls**: (1) M1 slice saturation above 4094 on width offset, (2)
  dynamic-weight conv above batch one crashes compiler service, (3) saturating slice kernel is
  target-keyed (M1 template specialization), (4) failed compiles spaced < 15s stall shared service,
  (5) true 4CC image input doesn't lower on direct path.

### 1.6 Family floors (Part IX, chs. 34–36)

| Floor | Native on | Operations |
|---|---|---|
| F0 | all families | conv, matmul, pooling, elementwise, reshape, transpose, concat, ReLU/sigmoid/tanh/gelu/swish |
| F2 (A13+) | A13 → A18 | softmax, layer/instance/batch norm, reductions, fused attention, erf, sqrt |
| F3 (A14+) | A14 → A18 | crop-resize, resample (texture engine) |
| F4 (A15+) | A15 → A18 | native sin, cos, global arg-min/arg-max |

**No compute operation floors above F4.** A16/A17/A18 add cores, not operations.

**Stream-vs-fold per compressed format (ch. 25):**

| Format | M1/H13 | A14/M2 | A15/M3+ | M5/H17s |
|---|---|---|---|---|
| int4 lookup-table | stream (2.37×) | stream | stream | stream (1.6–1.8×) |
| structured sparsity | stream (1.55–1.64×) | stream | stream | stream |
| int8 affine | **fold** (no gain) | stream (0.52× at 8K weight) | stream | stream |
| blockwise affine | **fold** | **fold** | stream | stream |

---

## 2. Distillation

### 2.1 What's new vs our prior ANE work

| Aspect | Already covered (155/223/224 + shipped code) | NEW in this paper |
|---|---|---|
| ANE as a compute backend | ✅ Path A/B/C (coreml-native, rane, rustane) | — |
| Dispatch overhead | ✅ ~5ms CoreML, ~0.24ms rane, ~0.095ms XPC | Quantified: 0.23 ms floor (M1) is the *standalone* floor; 98% of tiny-op wall time is dispatch overhead |
| INT8 quantization | ✅ coremltools.optimize.coreml | Stream-vs-fold decision per family — int8 *folds* on M1 (no bandwidth gain) |
| MLComputePlan | ✅ Public ANE placement introspection | Family-floor trait `MinimumFamily<N>` as compile-time gate |
| Stateful NPC accumulators | ✅ StateTensorSpec (iOS 18+) | The M1 *substrate* is output-to-input buffer aliasing; counter-event engine arrives later |
| Concat-tap pattern | ✅ Research 223 | Resident-state chaining primitive (`CSNE_CMD_PROCEDURE_CALL_CACHE_REQUEST`) is the firmware mechanism |
| Wide accumulator | ❌ Not covered | **NEW**: fp32-class, radix-4 first stage, MAC output saturates at 2^15 |
| 2 MB working-set cliff | ❌ Not covered | **NEW**: hard design limit, not a soft slope |
| Family floor table | ❌ Not covered | **NEW**: F0/F2/F3/F4, no op floors above F4 |
| Stream-vs-fold per format | ❌ Not covered | **NEW**: int8/blockwise fold on M1, stream on later families |
| M(n) → H(n+12) rule | ❌ Not covered | **NEW**: 28-target compiler family, generation-tag byte |
| Cross-generation training parity | ❌ Not covered | **NEW**: 0.9080 (M1) vs 0.9070 (M5), deterministic |

### 2.2 Fusion — the closest cousins across the 5-repo quintet

| Cousin | Repo | What it ships | Fusion with this paper produces |
|---|---|---|---|
| `roofline.rs` | katgpt-rs | GPU-only roofline (M1/M2/M3/M4 Pro peaks, 50µs launch) | **ANE-aware roofline** with 0.23ms dispatch floor + 2MB working-set axis + ANE peaks |
| `npc_brain_router.rs` + `AneNpcBrainBackend` (Plan 255, **shipped**) | riir-engine | Hardcoded `ANE_BATCH_THRESHOLD = 100`, justifies as "~95µs ANE dispatch vs 75ns × npc_count SIMD" | **Per-chip roofline-driven threshold** that replaces the constant with `chip_family + op_shape → threshold`, accounting for the actual 0.23 ms M1 floor and the 2 MB working-set cliff |
| `npc_brain.mlpackage` (Plan 255, **shipped**) | riir-ai/assets | Three-fused-op CoreML ML Program (sense/emotion/zone projection, FP16) | **Stream-vs-fold aware weight prep**: int8/blockwise *fold* on M1 → no bandwidth gain → keep weights as fp16 or use int4-LUT / sparse |
| `MerkleFrozenEnvelope` | riir-neuron-db | Freeze/thaw integrity envelope | **Resident-state buffer aliasing** as the M1 hardware substrate for frozen snapshot hot-swap |
| Raven RSM (Research 006) | katgpt-rs | O(1) routing-slot memory | **ANE KV-cache substrate** via `share_buffer` (M1) / native state (A15+) |
| Plan 175 / Plan 176 | katgpt-rs | ANE-Inspired DDTree + GPU/ANE trigger gate | **Family-floor capability gate** that rejects ops below their MinimumFamily at compile time |
| Plasma → Hot → Warm → Cold tiering (constraint #8/#9) | all repos | Tier naming convention | **Quantitative tier boundaries**: Plasma = sub-0.23ms-floor CPU SIMD; Hot = batched ANE/GPU; Warm/Cold unaffected |

### 2.3 The distilled primitive (modelless)

The transferable insight, stripped of the paper's hardware-internal detail:

> **The ANE is not a faster GPU. It is a fixed-function fp16 MAC array with a 2 MB on-chip
> working set, a 0.23 ms dispatch floor (M1), and a 141 FLOP/byte ridge point that is ~3× higher
> than the GPU's. Auto-routing between CPU SIMD, ANE, and GPU must therefore consider three
> independent axes — arithmetic intensity, working-set size, and dispatch overhead — not the
> single FLOP/byte ratio the GPU roofline uses.**

This is a modelless inference primitive: a cost model that runs in ≤1 µs on CPU, takes an op
shape + dtype + target chip, and returns `(device, runtime_ms, bound)`. It does not depend on
training, weights, or runtime state — it is pure arithmetic.

### 2.4 Latent-space reframing (mandatory per workflow §1 step 3)

This paper is hardware, not latent-space, so the latent reframing is thin. The closest angle:

- **The wide accumulator is a modelless correctness primitive.** A 16000-element fp16 reduction
  returns bit-exact, where a naive fp16 running sum stalls near 2000. This means our HLA
  per-NPC belief-state reductions (which currently use sigmoid-bounded dot-products) can be
  faithfully delegated to the ANE without precision loss up to 16000 elements — well above the
  per-NPC scale (8-dim HLA × ~64 atoms = 512 elements).
- **The buffer-aliasing resident state is a freeze/thaw substrate.** The M1's `share_buffer`
  primitive (bind one buffer to output of step N and input of step N+1) is exactly the mechanism
  by which a frozen snapshot can be hot-swapped atomically without a host round-trip. This
  connects to `MerkleFrozenEnvelope` (riir-neuron-db) and `LoRAHotSwap` (riir-ai) as the hardware
  layer beneath them on Apple Silicon.
- **The family-floor trait is a compile-time capability gate.** Mirrors our
  `phase_transition_subspace_phase_gate` pattern (riir-neuron-db): a deterministic, named
  threshold that decides whether an op runs. Same shape, different domain.

None of these rise to a Super-GOAT — they are extensions of existing primitives onto a new
substrate, not new capability classes.

---

## 3. Verdict

**GOAT.**

**One-line reasoning:** A modelless ANE-aware roofline cost model produces a provable routing
gain over the current shipped `NpcBrainRouter`'s hardcoded `ANE_BATCH_THRESHOLD = 100`
(Plan 255 Part 4) — specifically, it replaces the assumed "~95µs ANE dispatch" with the measured
0.23 ms M1 / 0.11 ms M5 floor and adds the 2 MB working-set cliff as a second routing axis,
which the shipped router cannot do — but it does not create a new capability class.

### Why not Super-GOAT

- **Q1 (No prior art?):** Mixed. The OVERALL topic (ANE backend + auto-router) is heavily
  covered: Plan 255 Part 4 shipped `NpcBrainRouter` + `npc_brain.mlpackage` + `AneNpcBrainBackend`;
  Research 155/223/224 cover the public-API path; `roofline.rs` covers the GPU cost model. The
  specific NEW contributions (working-set cliff, family floor, stream-vs-fold, M(n)→H(n+12)
  naming rule) are not in our code, but they are *refinements* of an existing, shipped area.
- **Q2 (New class of behavior?):** No. Extends existing `roofline.rs` and `device_selector.rs`;
  does not introduce a capability we lack.
- **Q3 (Product selling point?):** "Our auto-router never misplaces a 64×64 matmul on ANE" — yes,
  but incremental.
- **Q4 (Force multiplier?):** Touches ≥2 systems (roofline + device_selector + Plasma/Hot
  tiering), but not pillar-level.

Failing Q2 and Q3 ⇒ GOAT, not Super-GOAT.

### MOAT gate (per domain)

- **katgpt-rs (public engine):** IN SCOPE. The ANE-aware roofline is a fundamental/primitive
  that the adoption funnel depends on. Ship behind a feature flag, benchmark, GOAT gate
  per-stack (transformer-stack slot: roofline/ANE-routing).
- **riir-ai (private runtime):** IN SCOPE for the resident-state substrate mapping (M1
  `share_buffer` ↔ `MerkleFrozenEnvelope`). Private HOW, not public.

### Per-stack promote/demote tracking

| Stack slot | Current primitive | This paper's contribution | Outcome |
|---|---|---|---|
| Roofline cost model | `roofline.rs` (GPU-only, M1/M2/M3/M4 peaks) | ANE peaks + 2 MB working-set axis + 0.23 ms floor + family-floor gate | **Add as opt-in `ane_roofline` feature**; promote to default if GOAT gate passes |
| NPC brain auto-router | `NpcBrainRouter` with hardcoded `ANE_BATCH_THRESHOLD = 100` (Plan 255 Part 4, shipped) | Per-chip roofline-driven threshold that accounts for actual 0.23 ms M1 floor (vs the shipped 95µs assumption) and the 2 MB working-set cliff | **Add as opt-in `roofline_router` feature**; demote the hardcoded constant to fallback if GOAT passes. **Does NOT replace `NpcBrainRouter` — refines its threshold input.** |

---

## 4. Plan sketch (NOT yet opened — GOAT only)

A plan would target `katgpt-rs/crates/katgpt-core/src/roofline.rs` (extend) and a new
`katgpt-rs/crates/katgpt-core/src/ane_roofline.rs` (new module). Sketch:

```rust
/// ANE-specific roofline cost model (Plan TBD, Research 377).
///
/// Extends `roofline.rs` with the ANE's distinct cost shape:
/// - 0.23 ms dispatch floor (M1) vs 50 µs GPU launch overhead
/// - 2 MB on-chip working-set cliff (HAL field 0x1b8)
/// - 141 FLOP/byte ridge point (~3× GPU ridge)
/// - Family-floor capability gate (F0/F2/F3/F4)

#[repr(u8)]
pub enum AneBound {
    Compute,        // above ridge, working set fits
    Memory,         // below ridge
    WorkingSet,     // operand > 2 MB → tiles and streams
    Dispatch,       // work < 0.23 ms floor → CPU wins
    FamilyGated,    // op requires family > target's family
}

pub struct AnePeaks {
    pub compute_tfidf16: f64,        // 12.0 (M1) → 19.6 (M5)
    pub bandwidth_gbs: f64,          // 85 (M1) → 145 (M5)
    pub dispatch_floor_ms: f64,      // 0.23 (M1) → 0.11 (M5)
    pub working_set_bytes: u64,      // 2 MB (M1) → 4.72 MB (M5)
    pub ridge_flop_per_byte: f64,    // 141 (M1) → 424 (M5)
    pub family: AneFamily,           // A13/A14/A15/A16/A17/A18
}

pub fn ane_estimate(op: OpShape, dtype: Dtype, peaks: &AnePeaks) -> AneCost {
    // 1. Family-floor gate: reject if op's MinimumFamily > peaks.family
    if op.min_family > peaks.family { return AneCost::rejected(); }
    // 2. Working-set gate: tile-and-stream if any operand > working_set_bytes
    let ws_bound = op.largest_operand_bytes(dtype) > peaks.working_set_bytes;
    // 3. Three-way roofline: max(dispatch_floor, compute, memory)
    let compute_ms = op.flops / (peaks.compute_tfidf16 * 1e6);
    let memory_ms = op.bytes / (peaks.bandwidth_gbs * 1e6);
    let runtime_ms = peaks.dispatch_floor_ms.max(compute_ms).max(memory_ms);
    // 4. Bound classification
    let bound = if ws_bound { AneBound::WorkingSet }
        else if runtime_ms <= peaks.dispatch_floor_ms * 1.01 { AneBound::Dispatch }
        else if compute_ms >= memory_ms { AneBound::Compute }
        else { AneBound::Memory };
    AneCost { runtime_ms, bound, .. }
}
```

**GOAT gate (per AGENTS.md):**

- **G1 (correctness):** ANE cost model agrees with measured M1 dispatch times within ±30% on
  the paper's reference shapes (3×3 256ch conv, 4096² GEMM, single-token decode).
- **G2 (perf):** Routing decisions match the paper's per-workload verdict table (ch. 11): conv
  stack → ANE, decode → GPU, tiny ops → CPU.
- **G3 (no-regression):** Existing GPU roofline tests still pass; ANE peaks default to "off"
  when target family unknown.
- **G4 (alloc-free):** `ane_estimate` is `#[inline(always)]`, no allocations, ≤1 µs CPU.

**UQ check (Report the Floor rule, AGENTS.md):** This primitive does NOT claim a probability
distribution, predictive interval, or coverage guarantee. It is a deterministic cost model.
Floor rule does not apply.

---

## 5. What stays out (riir-train redirect)

The paper's chapter 15 ("Training on the engine") documents that a full forward+backward+
optimizer loop runs on ANE as ordinary inference-style graph operations with optimizer state
resident via buffer aliasing, achieving 0.9080 (M1) / 0.9070 (M5) cross-generation training
parity on a seeded CNN.

**This is interesting but training-method research → riir-train.** The modelless-unblock
protocol (§3.5) does not apply because we are not blocked on a gate; the paper's training loop
is a research observation, not a modelless primitive we need to ship. Note "→ riir-train" for
the training-loop specifics and stop.

The *inference-time* angle (graph-of-forward-ops as the substrate for any compute, including
gradient computation expressed as forward ops) is a modelless insight that lands here, not in
riir-train. It reinforces the existing modelless-first mandate: even "training" on the ANE is
just inference over a graph that includes backward ops.

---

## 6. Open questions / risks

- **Private-API risk**: this paper documents private `e5rt_*` and IOKit interfaces that are
  "undocumented, unsupported, and version-fragile across operating-system updates". Our existing
  ANE backend uses CoreML (public) per Research 155's Path A verdict. The distilled primitive
  here (ANE-aware roofline) does NOT require private APIs — it only needs the chip's family
  identifier (publicly available via `sysctl hw.optional.arm64`).
- **Family-floor gate accuracy**: the paper's F0/F2/F3/F4 table is reverse-engineered, not
  Apple-documented. Treat as advisory; verify per-target with `MLComputePlan` (public, per
  Research 224) before relying on a routing decision.
- **M3/H15 unmeasured**: the paper's M3 row is decompile-derived, not silicon-confirmed. The
  cost-model constants for M3 are predicted; the routing primitive should fall back to "GPU if
  available, else CPU" when target family is uncertain.

---

## TL;DR

The Bryngelson ANE reference is the most thorough public account of Apple's Neural Engine, but
for our codebase it is **incremental** — Plan 255 Part 4 already shipped an end-to-end ANE
pipeline (`npc_brain.mlpackage` + `AneNpcBrainBackend` + `NpcBrainRouter` with hardcoded
`ANE_BATCH_THRESHOLD = 100`), Research 155/223/224 already covered the public-API path, and
`roofline.rs` already covers the GPU cost model.

The transferable modelless primitive is an **ANE-aware roofline cost model** that extends our
GPU-only `roofline.rs` with three ANE-specific axes (2 MB working-set cliff, 0.23 ms dispatch
floor, family-floor capability gate) and an ANE peak table for M1–M5. The shipped
`NpcBrainRouter`'s comment justifies its threshold as "~95µs ANE dispatch overhead"; Bryngelson
measures the full firmware round trip at 0.23 ms on M1, ~2.4× higher than the shipped
assumption, plus identifies the working-set cliff as a second routing axis.

**Verdict: GOAT.** Provable routing gain over the shipped hardcoded threshold, but not a new
capability class. Plan into `katgpt-rs/crates/katgpt-core/src/ane_roofline.rs` (opt-in
`ane_roofline` feature), benchmark against the paper's reference shapes, and refine
`NpcBrainRouter`'s threshold to consume the new primitive (do NOT replace the router — only its
threshold input). Promote to default if GOAT gate passes. The training-loop specifics
(ch. 15) → riir-train, not here.
