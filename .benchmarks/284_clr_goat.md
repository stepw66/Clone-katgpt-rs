# Benchmark 284: CLR GOAT Gate — G1–G5

**Plan:** [katgpt-rs/.plans/284_runtime_clr_self_adaptive_loop.md](../.plans/284_runtime_clr_self_adaptive_loop.md)
**Research:** [katgpt-rs/.research/255_VibeThinker_CLR_Test_Time_Reliability.md](../.research/255_VibeThinker_CLR_Test_Time_Reliability.md)
**Feature:** `clr` (opt-in until G1–G5 pass)
**Status:** Phase 4 stubbed — gate not yet run. Stubs live in `tests/bench_284_clr_goat.rs`.

---

## GOAT Criteria

| Gate | Criterion | Target | Status |
|------|-----------|--------|--------|
| **G1** | CLR-vote accuracy vs best-of-N majority on synthetic suite | CLR ≥ +3pp over majority | ⏳ Not run |
| **G2** | `SigmoidProjectionVerifier` Expected Calibration Error | ECE ≤ 0.10 over 10K samples | ⏳ Not run |
| **G3** | `clr_vote_minimal()` latency at K=32, M=5, 8-dim directions | ≤200µs (stretch ≤50µs) | ⏳ Not run |
| **G4** | Heap allocations on vote path after `ClrScratch::new()` warmup | 0 net allocations | ⏳ Not run |
| **G5** | Feature isolation | compiles with/without `clr`; zero overhead when off | ✅ Verified at build level (Phase 1) |

---

## G1 — CLR beats best-of-N majority

**Suite:** 50 trajectory-groups × 5 clusters × 10 trajectories each. In each
cluster, exactly 1 trajectory has a ground-truth-flawed claim (embedding
orthogonal to the relevant direction vector, forcing `v < 0.5`).

**Metric:** accuracy of winner-cluster matching the ground-truth flawless
cluster, CLR-vote vs best-of-N majority, over 100 random seeds.

**Result:**

```
| Method           | Mean accuracy | Stddev | Δ vs majority |
|------------------|---------------|--------|---------------|
| Best-of-N majority | TBD         | TBD    | —             |
| CLR-vote         | TBD           | TBD    | TBD           |
```

**Verdict:** TBD (must be ≥ +3pp to pass).

---

## G2 — Verifier calibration (ECE)

**Suite:** 10K random claim embeddings + direction vectors. True verdict is
`Bernoulli(sigmoid(dot))`. Compute ECE of `SigmoidProjectionVerifier::verify`
outputs in 10 equal-width bins.

**Result:**

```
| Bins | ECE    | Pass? |
|------|--------|-------|
| 10   | TBD    | TBD   |
```

**Verdict:** TBD (must be ≤ 0.10 to pass).

---

## G3 — Hot-path latency

**Bench:** `cargo bench` criterion group. K=32 trajectories, M=5 claims each,
8-dim direction vectors. Time per `clr_vote_minimal()` call.

**Result:**

```
| K  | M | dim | Mean (µs) | P50 (µs) | P99 (µs) | Pass? |
|----|---|-----|-----------|----------|----------|-------|
| 8  | 5 | 8   | TBD       | TBD      | TBD      | TBD   |
| 16 | 5 | 8   | TBD       | TBD      | TBD      | TBD   |
| 32 | 5 | 8   | TBD       | TBD      | TBD      | TBD   |
```

**Verdict:** TBD (must be ≤200µs at K=32; stretch ≤50µs).

---

## G4 — Zero allocation

**Bench:** counting global allocator. Warm up `ClrScratch::new(32, 5)` once,
then call `clr_vote_minimal()` 1000×. Assert 0 net allocations after warmup.

**Result:**

```
| Warmup allocs | Steady-state allocs/call | Pass? |
|---------------|--------------------------|-------|
| TBD           | TBD                      | TBD   |
```

**Verdict:** TBD (must be exactly 0 steady-state to pass).

**Note:** the extractor (`FnClaimExtractor` / domain extractor) allocates per
call — the zero-alloc contract covers the vote arithmetic, clustering, and
tiebreak only, NOT the caller-supplied extractor. A hot-path variant taking
pre-extracted claims can eliminate extractor allocations if a caller needs it.

---

## G5 — Feature isolation

**Verified at build level (Phase 1):**

- ✅ `cargo build --no-default-features --features clr` compiles cleanly.
- ✅ `cargo build --no-default-features` compiles cleanly (clr symbols absent).
- ✅ `clr = []` declared with no dependencies; not in `default` or `full`.

---

## Promotion decision (Phase 5 T5.6)

After G1–G4 run:

- **All pass** → promote `clr` to default-on in root `Cargo.toml`. Mark
  "GOAT-proved" in README.
- **G1 fails** → keep opt-in. The mechanism is correct but the synthetic suite
  may be too easy; revisit with a harder suite.
- **G3 fails** (>200µs) → keep opt-in. Profile to find the bottleneck (likely
  `powf` or the `outcome_eq` callback). Demote hot-path callers to
  `clr_vote_minimal` only.
- **G4 fails** (allocates) → critical bug in scratch discipline; fix before any
  promotion.
