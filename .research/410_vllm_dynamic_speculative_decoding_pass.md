# Research 410: vLLM Dynamic Speculative Decoding — PASS (Already Ships, Superior Form)

> **Source:** [vLLM Docs — Dynamic Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/) (vLLM project, retrieved 2026-07-11)
> **Date:** 2026-07-11
> **Status:** Done
> **Related Research:** 002 (Speculative Decoding — Leviathan), 162 (TRAS), 194 (CaDDTree), 218 (Breakeven Complexity), 316 (DSpark — Hardware-Aware Prefix Scheduler)
> **Related Plans:** 182 (TRAS), 219 (CaDDTree budget), 250 (Breakeven), 339 (Hardware-Aware Prefix Scheduler)
> **Classification:** Public

---

## TL;DR

vLLM's "Dynamic Speculative Decoding" is a **statically configured lookup table** that selects the draft token count `K` from the current batch size via `num_speculative_tokens_per_batch_size: [[start_bs, end_bs, optimal_K], ...]`. It is NOT adaptive or learned — "dynamic" means "runtime table lookup by observed concurrency." katgpt-rs already ships a **strictly superior** mechanism: `HardwareAwarePrefixScheduler` + `SpsCurve` (Plan 339, `crates/katgpt-speculative/src/prefix_scheduler.rs`), distilled from DSpark (Research 316). The vLLM table is a degenerate case of the shipped scheduler.

**Verdict: PASS.** No new files, no plan, no primitive. The mechanism already ships in a more capable form.

---

## 1. What vLLM Dynamic SD Does

The mechanism is a single config field on the speculative config:

```json
"num_speculative_tokens_per_batch_size": [
  [1, 64, 3],
  [65, 128, 1],
  [129, 512, 0]
]
```

At runtime, the scheduler observes the current batch size (concurrency), finds the matching `[start_bs, end_bs]` range, and uses `optimal_K` as the number of draft tokens for that step. When `K=0`, speculation is disabled entirely (pure autoregressive decode). The motivation: as BS grows, verification compute scales as `BS × K`, so at high concurrency speculation hurts TPOT (time-per-output-token) — the table lets the operator cap K where it stops paying.

**Key properties:**
- **Static / operator-configured.** The table is fixed at serve time. No runtime learning, no per-request signal.
- **Global K, not per-request.** Every request in the batch gets the same K.
- **No correctness proof.** Any K is "correct" for lossless SD; the table just tunes throughput.
- **Step-function cost model.** The `[start_bs, end_bs]` ranges are a piecewise-constant approximation of the engine's batch-size-vs-throughput curve.
- **No per-position survival probabilities.** K is a single integer for the whole draft, not a per-position budget.

**Limitations vLLM documents:**
- Only tested with Eagle, Eagle-3, DFlash.
- Full Cudagraph requires Model Runner V2.
- Incompatible with data parallelism (ranks would pick different K → DP collective divergence).

---

## 2. What katgpt-rs Already Ships (Strictly Superior)

### 2.1 `HardwareAwarePrefixScheduler` + `SpsCurve` (Plan 339, Research 316)

Location: `katgpt-rs/crates/katgpt-speculative/src/prefix_scheduler.rs`

This is the DSpark (DeepSeek, arXiv via Research 316) **Hardware-Aware Prefix Scheduler** (Algorithm 1), shipped behind the `hardware_aware_scheduler` feature flag. It is a **global, multi-request, load-aware greedy** verification-budget allocator:

- **Input:** R active requests, each with a per-position survival probability `a_{r,j} = Π_{i≤j} c_{r,i}` (monotone non-increasing in j), where `c_{r,i}` comes from Bebop's `AcceptanceForecast` (entropy-bounded `α ≈ a − b·H(p)`, Plan 243).
- **Cost model:** a profiled `SpsCurve` — `steps_per_second(batch_size)` with **linear interpolation** between samples (not a step function). `SpsCurve::from_profile(&[(B, sps), ...])`.
- **Objective:** maximize `Θ = τ · SPS(B)` where `B = Σ_r (1 + ℓ_r)` (total verification batch) and `τ = Σ_r (1 + Σ_{j≤ℓ_r} a_{r,j})` (expected accepted tokens).
- **Algorithm:** globally sort all `(a_{r,j}, r, j)` candidates descending, greedily admit, O(1) `SpsCurve` lookup per step, **non-anticipating early-stop when `Θ ≤ Θ_best`**.
- **Correctness:** the early-stop is a **correctness theorem** (DSpark Appendix A), not just an optimization — without it, retrospective global search leaks future-token info into the current admission decision, breaking the lossless distribution-preservation guarantee. Shipped bit-identically to the paper's proof.
- **Complexity:** O(R·γ log(R·γ)), zero-allocation (`schedule_with_scratch` takes a reusable `&mut [..]` buffer).

GOAT gate (Plan 339): G1–G5 ALL PASS on synthetic multi-request workload (11.55× throughput gain on a cliff SPS curve). Stays opt-in because katgpt-rs default is single-request — the primitive has no leverage without a real multi-request batch caller (riir-ai crowd-NPC cognition or a batch server).

### 2.2 The degenerate case = vLLM Dynamic SD

The vLLM table is a **strict subset** of the shipped scheduler:

| vLLM Dynamic SD | `HardwareAwarePrefixScheduler` degenerate mode |
|---|---|
| `[[1,64,3],[65,128,1],[129,512,0]]` step table | `SpsCurve::from_profile` with step-function samples (or any profiled curve) |
| Global K per batch-size range | Per-request `ℓ_r` from greedy admission (degenerates to uniform K if all `a_{r,j}` are equal) |
| No per-position signal | Consumes Bebop `AcceptanceForecast` per-position `c_k` |
| No correctness proof | Non-anticipating early-stop correctness theorem (Appendix A) |
| Operator-configured at serve time | Runtime-adaptive per step |

Concretely: if you give `HardwareAwarePrefixScheduler` a step-function `SpsCurve` and uniform survival probabilities (no Bebop forecast), the greedy degenerates to "admit K candidates per request where K is the batch-size-tier from the SPS cliff" — i.e., the vLLM table behavior, but derived from the cost curve rather than configured as a table.

### 2.3 Additional shipped adaptive-budget primitives (also superior to a static table)

- **`AcceptanceSurrogate::expected_accepted_length_at_budget`** (`caddtree_budget.rs`, Plan 219 / Research 194) — per-request budget-vs-acceptance tradeoff via `τ = Σ sigmoid(k·(Π top1 − t))`. CaDDTree proves throughput is **unimodal in budget** under convex verification cost → greedy stopping is provably optimal.
- **`PositionWeightedBudget`** (`speculative/types.rs`) — position-weighted budget allocation with `gamma` decay and `min_budget_per_depth`.
- **`BudgetAdaption` enum** (`Off | Compression | Entropy | EchoConsistency`) — adaptive budget by entropy or compression signal.
- **TRAS** (`trust_region.rs`, Plan 182 / Research 162) — trust-region adaptive speculation window: high acceptance → expand window, low → window=1.
- **Breakeven Complexity Router** (Plan 250 / Research 218) — cost-amortization-aware tier routing.

---

## 3. Verdict

**PASS.** The vLLM Dynamic SD mechanism is a static, operator-configured lookup table for K-by-batch-size. katgpt-rs already ships `HardwareAwarePrefixScheduler` + `SpsCurve` (Plan 339), which is strictly more capable: per-request per-position survival probabilities (via Bebop `AcceptanceForecast`), a continuously-interpolated profiled cost curve (not a step table), a global multi-request greedy objective (`Θ = τ · SPS(B)`), and a non-anticipating early-stop correctness theorem (DSpark Appendix A) that preserves lossless speculative decoding. The vLLM table is the degenerate case of the shipped scheduler with a step-function SPS curve and uniform survival probabilities.

**One-line reasoning:** No new primitive, no plan, no files. The mechanism already ships in a superior, correctness-proven form. The only thing vLLM offers that the shipped scheduler lacks is **operational simplicity** (a one-line config table vs. profiling + Bebop forecast) — but `HardwareAwarePrefixScheduler::default()` already degenerates to "admit everything" (constant SPS), and `SpsCurve::from_profile` can represent a step function, so even the simple fallback is covered. If a "no-profiling, no-forecast, just-lookup-K-by-BS" convenience constructor is ever wanted, it is a trivial `SpsCurve::step_table(&[(bs_range, k)])` helper — not a research note.

**MOAT gate (katgpt-rs domain):** Neutral. The shipped scheduler is the moat-relevant primitive; the vLLM table adds nothing on top of it. No reroute needed.

---

## 4. Cross-References

- **Research 316** (DSpark) — the paper the shipped scheduler was distilled from. Notes that the scheduler is Gain-tier (not Super-GOAT) because it's a serving-system optimization, not a new capability class.
- **Plan 339** — the shipped `HardwareAwarePrefixScheduler` + `SpsCurve` implementation. GOAT G1–G5 PASS. Opt-in (`hardware_aware_scheduler`) until a real multi-request caller exercises it.
- **Research 194 / Plan 219** (CaDDTree) — per-request adaptive budget via unimodal greedy search (provably optimal under convex verification cost).
- **Research 162 / Plan 182** (TRAS) — trust-region adaptive speculation window.
- **Research 243** (Bebop) — the `AcceptanceForecast` that feeds per-position `c_k` into the scheduler.
- **Issue 003** (`katgpt-rs/.issues/`) — the original tracker for the Hardware-Aware Prefix Scheduler (resolved-and-removed; benchmark evidence in `.benchmarks/339_hardware_aware_prefix_scheduler_goat.md`).

## TL;DR

vLLM Dynamic SD = static K-by-batch-size lookup table. Already ships as `HardwareAwarePrefixScheduler` (Plan 339) in a strictly superior, correctness-proven form. PASS — no new files, no plan, no primitive.
