# Salience Tri-Gate Primitive — Per-Tick `Speak` / `Silent` / `Delegate`

**Plan:** [303](../.plans/303_salience_tri_gate_primitive.md)
**Research:** [281](../.research/281_Per_Tick_Salience_Tri_Gate_Speak_Silent_Delegate.md)
**Source paper:** [arxiv 2606.14777](https://arxiv.org/abs/2606.14777) — JoyAI-VL-Interaction (Yao et al., JD.com, Jun 2026)
**Status:** Phase 1 + Phase 3 + Phase 4 shipped. Phase 2 (latency bench, T2.2) deferred.
**Feature flag:** `salience_tri_gate` (opt-in, **not** default — pending Phase 5 GOAT promotion decision).

---

## What it is

The open, modelless primitive that distills JoyAI-VL-Interaction's per-second
emit decision into a generic 3-way gate. It consumes any latent activation
vector `a` plus two context scalars (zone-attention `z`, curiosity `c`) and
produces one of three **first-class** decisions:

| Variant     | Meaning                                                  |
|-------------|----------------------------------------------------------|
| `Silent`    | Agent actively chose silence this tick (not a default). |
| `Speak`     | Speak inline (no async escalation).                      |
| `Delegate`  | Hand off to an async backend via `DelegateToken<A>`.     |

**Zero game semantics in this crate** — NPC wiring lives in `riir-ai` Plan 330.
This crate is math-only, MIT, no game IP.

---

## API surface

All under `katgpt_rs::salience::`, gated behind `feature = "salience_tri_gate"`.

### `SalienceTriGate<A, const D: usize>`

The gate. Generic over activation dimension `D` (const generic) and delegate
payload type `A: Clone`. Zero-allocation on the hot path.

| Method                                                          | Purpose                                              |
|-----------------------------------------------------------------|------------------------------------------------------|
| `new(d_speak, d_delegate, w_z, w_c, beta_*, tau_*, floor, ceil)`| Construct with validated params (panics on bad).     |
| `decide(&a, z, c, payload, tick) -> SalienceDecision<A>`        | Per-tick decision. Hot path.                         |
| `decide_batch(activations, z, c, payloads, tick, out)`          | Batched form; caller owns output buffer.             |
| `build_delegate_token(payload, tick, idx, foldback) -> DelegateToken<A2>` | Phase 3 convenience constructor.          |

### `SalienceDecision<A>`

```rust
pub enum SalienceDecision<A> {
    Silent,
    Speak,
    Delegate(A),
}
```

`Silent` is a variant, not a threshold-suppression default. Bound: `Clone +
Debug + PartialEq` (no `Eq` — payloads may carry floats).

### `DelegateToken<A>` + `FoldbackTarget`

Typed handoff for the `Delegate` branch. The caller spawns the async task;
this crate only provides the payload.

```rust
pub struct DelegateToken<A: Clone> {
    pub payload: A,
    pub issued_tick: u64,
    pub holding_reply_idx: u8,   // caller's template table
    pub foldback_target: FoldbackTarget,
}

#[repr(u8)]
pub enum FoldbackTarget {
    ActivationState = 0,
    PatternMemory   = 1,
    ExternalJudge   = 2,
    ColdTier        = 3,
}
```

### `PendingDelegateQueue<A, const CAP: usize = 2>` (Phase 3)

Fixed-capacity ring buffer of pending `DelegateToken`s. Zero-allocation
handoff between the per-tick decision and the caller-owned async spawn.

```rust
let mut q: PendingDelegateQueue<MyPayload> = PendingDelegateQueue::new();
q.push(token)?;            // Err(token) if full — caller decides policy
let oldest = q.pop();      // FIFO; None if empty
```

**Contract (Plan 303 T3.3):** this crate does **not** spawn async tasks. The
caller (riir-ai runtime, Plan 330) owns the spawn. This queue is just the
typed handoff.

### `SilenceToken`

Newtype signaling "this NPC actively chose silence this tick". Emitted with
`Silent` from Phase 3 onward.

---

## Design rationale

### Two stacked sigmoids, never softmax

Per `AGENTS.md`: the gate uses two independent logistic sigmoids
(`score_speak`, `score_delegate`) chained with simple threshold comparisons,
**not** a 3-way softmax over `(Speak, Silent, Delegate)`.

Why:
- **Separability.** With `ceil_delegate = +∞`, the delegate sigmoid provably
  contributes nothing to the output stream — the gate collapses to a clean
  speak/silent binary. This is the G2 ablation-parity GOAT gate (implemented
  as a test). A softmax over 3 logits cannot make this claim: the delegate
  logit always renormalizes the other two.
- **No temperature coupling.** `beta_speak` and `beta_delegate` are
  independent inverse temperatures. Softmax couples all classes through a
  single denominator.
- **Determinism.** Bit-identical output for bit-identical input — no RNG, no
  thread-local state, no allocation. The G1 determinism test asserts this.

### `Silent` as a first-class variant

In the paper, "chose not to speak" is an explicit decision with its own
training signal (`w_first_silence = 1.0`, `w_repeated_silence = 0.4`). It is
**not** "speak-score below threshold → default to silent". Subscribers
observe `Silent` through the same channels as `Speak` / `Delegate`, so they
can distinguish "nothing to say" from "explicitly chose silence".

### Generic over `D` and `A`

- `D` (const generic): activation dimension. Caller picks the latent space
  (HLA, CGSP embedding, etc.). This crate is modelless — direction vectors
  are caller-supplied.
- `A: Clone`: delegate payload type. The gate never inspects `A`; it just
  moves it into the `Delegate(A)` variant. `build_delegate_token` is generic
  over a *separate* `A2` so callers can use a richer handoff payload than the
  lightweight payload passed to `decide()`.

### Zero-allocation hot path

`decide` and `decide_batch` perform no heap allocation. All temporaries are
stack scalars. `decide_batch` takes a caller-owned output buffer. The dot
products use `f32::mul_add` for single-rounding FMA contraction (matches the
`bridge/mod.rs` / `cumprodsum.rs` convention).

### `PendingDelegateQueue` is a fixed-size ring

`slots: [Option<DelegateToken<A>>; CAP]` with `head: u8, len: u8`. Default
`CAP = 2` (one in-flight + one queued). Push returns `Err(token)` when full
— the caller decides policy (drop oldest, drop newest, refuse). `CAP <= 255`
because head/len are `u8`.

Initialized with `[const { None }; CAP]` (inline const blocks, stable since
Rust 1.79) — works for non-`Copy` `DelegateToken<A>` without a `Copy` bound
on `A`.

---

## Usage

### Basic (per-tick decision)

```text
cargo run --example salience_tri_gate_basic --features salience_tri_gate
```

```rust
use katgpt_rs::salience::{SalienceDecision, SalienceTriGate};

const D: usize = 8;
let gate: SalienceTriGate<u32, D> = SalienceTriGate::new(
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // d_speak
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // d_delegate
    0.3, 0.2, 2.0, 2.0, 0.5, 0.5, 0.15, 0.4,
);
let a = [0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
match gate.decide(&a, 0.5, 0.5, 42, 0) {
    SalienceDecision::Silent => { /* observe channel */ }
    SalienceDecision::Speak => { /* emit inline */ }
    SalienceDecision::Delegate(p) => { /* spawn async with payload p */ }
}
```

### Batched (throughput)

```text
cargo run --example salience_tri_gate_batch --features salience_tri_gate --release
```

On the dev machine (release build, N=10000, D=8): **~50M decisions/sec** via
single-shot `Instant` timing. See the [GOAT gate status](#goat-gate-status)
caveat — this is a smoke test, not the authoritative bench.

### Delegate queue + async spawn (caller pattern)

See `PendingDelegateQueue` doc example (`cargo doc --features
salience_tri_gate`). Summary: build a `DelegateToken` on `Delegate`, push to
the queue, the caller's runtime pops and spawns the async task, and on
completion applies foldback per `token.foldback_target`.

---

## GOAT gate status

Honest assessment of what is measured vs not.

### G1 — determinism + monotonicity ✅ (implemented as tests)

- **G1 determinism** (`test_g1_determinism`): same inputs ⇒ same decision.
  Run `decide` twice, assert `==`.
- **G1 monotonicity in salience** (`test_g1_monotonicity_in_salience`): sweep
  `a[0]` along `d_speak`, verify exactly one Silent→Speak threshold crossing.
- **G1 monotonicity in delegate_dot** (`test_g1_monotonicity_in_delegate_dot`):
  sweep `a[1]` along `d_delegate`, verify exactly one Speak→Delegate crossing.

**Status: PASS** (3 tests).

### G2 — two-sigmoid ablation parity ✅ (implemented as test)

- **G2** (`test_g2_ablation_parity`): with `ceil_delegate = +∞`, the gate's
  Silent/Speak sequence is bit-identical to a speak/silent-only reference
  over 100 deterministic inputs. Proves the delegate sigmoid is separable.

**Status: PASS** (1 test).

### Latency / throughput ⚠️ (Phase 2 deferred — T2.2)

- The **authoritative** Criterion bench (`benches/salience_tri_gate_bench.rs`,
  Plan 303 T2.2) is **not yet implemented**. Target: `< 50ns` for a single
  `decide()` call at `D=8` (cf. `evolve_hla` ~14ns for D=8).
- The `salience_tri_gate_batch` example reports ~50M decisions/sec via
  single-shot `Instant` timing on the dev machine (release build). This is a
  **smoke test of the API shape, not a GOAT-gate-quality number** —
  single-shot timing includes setup noise and lacks Criterion's statistical
  rigor (warmup, outlier detection, confidence intervals).

**Status: NOT MEASURED to GOAT-gate standard.** The Phase 5 promotion
decision is blocked on T2.2.

---

## Private boundary (what is NOT in this crate)

- **NPC wiring** → `riir-ai` Plan 330. This crate is game-agnostic.
- **HLA / CGSP binding** (activation `a` source) → `riir-ai` Plan 330. The
  activation `a` is generic here.
- **R133 mind-reading `ca` scalar** → `riir-ai` Plan 311.
- **cgsp curiosity scalar** → `riir-ai` Plan 299 (curiosity runtime).
- **Async delegate backends** (AnyRAG gateway, Engram, Cold-tier) →
  `riir-neuron-db` (`gateway.rs`) + `riir-ai` Plan 330 routing layer.
- **Training recipe** (GRPO + role-weighted SFT, the `w_first_silence=1.0`,
  `w_repeated_silence=0.4`, `w_response=1.5` role-token weights) →
  `riir-train`.
- **AdaCodec streaming visual codec** (paper §3.1) → orthogonal; separate
  paper (2606.02569).
- **Long-horizon three-tier memory** (paper §4.3) → Plan 312 (Dual-Pool CGSP)
  + research 007 (Four-Tier Memory).

---

## References

- Plan: `katgpt-rs/.plans/303_salience_tri_gate_primitive.md`
- Research: `katgpt-rs/.research/281_Per_Tick_Salience_Tri_Gate_Speak_Silent_Delegate.md`
- Private NPC guide: `riir-ai/.research/148_Per_Tick_Emit_Salience_NPC_Guide.md`
- Runtime plan: `riir-ai/.plans/330_proactive_npc_salience_gate_runtime.md`
- Source paper: [arxiv 2606.14777](https://arxiv.org/abs/2606.14777)
