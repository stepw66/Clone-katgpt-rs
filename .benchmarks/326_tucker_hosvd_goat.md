# Plan 326 — Tucker / HOSVD GOAT Bench Record

**Date:** 2026-06-25
**Primitive:** `tucker_factorization` (N-mode HOSVD in `crates/katgpt-core/src/linalg/tucker.rs`)
**Bench:** `crates/katgpt-core/benches/bench_326_tucker_hosvd_goat.rs`
**Status:** **PARTIAL** — G1/G3 PASS via unit tests; G2/G4 PENDING (bench could not run — system-level Rust binary execution failure under heavy parallel cargo load).

---

## Gate Definitions

| Gate | Target | How Verified |
|------|--------|--------------|
| **G1** reconstruction quality | rank-`(2,2,2)` tensor recovered with ranks `(2,2,2)`, rel Frob error `< 1e-4` | bench `g1_reconstruction_quality` + unit test `hosvd_low_rank_recovers_exact_low_rank_tensor` |
| **G2** perf | `(8,8,8)` decompose + reconstruct, mean latency `≤ 500µs` over 1000 iters | bench `g2_perf` (cold-tier archival budget) |
| **G3** no-regression | full-rank `(4,4,4)` reconstruction max abs error `< 1e-4` | bench `g3_no_regression` + unit test `hosvd_full_rank_is_near_lossless_3mode` |
| **G4** alloc-free hot path | `tucker_decompose_into` with pre-warmed scratch → 0 allocs / 100 calls | bench `g4_alloc_free` (CountingAllocator) |

---

## Results

### G1 — reconstruction quality: **PASS** (unit test)

```
test linalg::tucker::tests::hosvd_low_rank_recovers_exact_low_rank_tensor ... ok
```

Construction: `X[i,j,k] = a[i]·b[j]·c[k]` with `a = [1.0, 0.5, 0, 0]`, `b = [0.7, -0.3, 0, 0]`, `c = [0.4, 0.9, 0, 0]`. Every mode-n unfolding has rank ≤ 2. HOSVD with ranks `(2,2,2)` reconstructs with rel Frob error `< 1e-4` (the discarded singular values are ~0 up to f32 round-off). This is the strongest correctness claim — it proves HOSVD recovers the exact low-rank structure when the rank budget matches the true rank.

### G2 — perf: **PENDING** (bench could not run)

The bench binary hangs at startup — a system-level Rust binary execution failure that affects ALL Rust binaries (including the known-good Plan 325 bench and a trivial `println!("hello")` test program). Shell commands execute normally; only Rust binaries hang. Cause is external to this code — likely related to heavy parallel cargo activity (multiple clippy + test + check runs from sibling sessions competing for process slots or hitting a dyld/mach deadlock under load).

**Analytical estimate:** for `(8,8,8)` with ranks `(4,4,4)`, the cost is 3 SVDs of `(8, 64)` unfoldings (each: 60 Jacobi sweeps max, ~8² / 2 pairs per sweep × 64 inner ≈ 2k FMA/sweep × 60 = 120k FMA per SVD, ~24µs at 5 GFLOP/s) + 3 n-mode contractions (each: `8 × 64 × 8` ≈ 4k FMA, negligible). Total ~75µs. Well under the 500µs budget. To be confirmed empirically once the environment recovers.

### G3 — no-regression (full-rank lossless): **PASS** (unit test)

```
test linalg::tucker::tests::hosvd_full_rank_is_near_lossless_3mode ... ok
```

Full-rank HOSVD of a `(4,4,4)` tensor (ranks = shape) reconstructs with max abs error `< 1e-4`. The decomposition is lossless up to f32 round-off when no truncation occurs — the only error source is the one-sided-Jacobi SVD residual accumulated over 3 modes + reconstruction.

### G4 — alloc-free hot path: **PENDING** (bench could not run)

Same system-level execution failure as G2. **Design verification (static):** the hot path `tucker_decompose_into` borrows `&mut scratch` and `&mut result`, both pre-allocated via `with_capacity`. All internal buffers (`unfold_buf`, `y_buf`, `contract_buf`, `transpose_buf`, `svd_work`, `svd_result`) are sized once at construction and never grown on the hot path. The `SvdScratch` / `SvdResultScratch` are sized in `TuckerScratch::with_capacity` for the worst-case mode-n SVD dimensions, so their auto-grow path (which would allocate) is never triggered. To be confirmed empirically via CountingAllocator once the environment recovers.

---

## Verdict

**2/4 gates empirically PASS (G1, G3 via unit tests); 2/4 gates PENDING (G2, G4 — bench blocked by system-level execution failure).**

**Promotion decision: KEEP OPT-IN.** Per the katgpt-rs AGENTS.md GOAT gate rule, promotion to default-on requires the full gate (G1-G4) to pass empirically. G2 (perf) and G4 (alloc-free) are not yet verified. The feature stays behind `tucker_factorization = ["subspace_phase_gate"]` until the bench can be run in a clean environment.

**Re-validation protocol:** re-run `cargo bench -p katgpt-core --features tucker_factorization --bench bench_326_tucker_hosvd_goat -- --nocapture` in a clean environment (no parallel cargo activity). If all 4 gates pass, promote `tucker_factorization` to the `default` feature list with a comment citing this record.

---

## Unit test coverage (25 tests, all PASS)

These exercise the algorithm exhaustively and are the basis for the G1/G3 PASS verdicts:

- `config_rejects_*` (7 tests): TuckerConfig validation (zero modes, too many modes, zero shape, rank > shape, rank > unfolding bound, shape > SVD limit, mismatched lengths).
- `config_accepts_valid_3mode`: valid config accepted.
- `unfold_fold_round_trip_3mode` / `unfold_produces_i_n_rows`: unfolding/folding correctness.
- `hosvd_full_rank_is_near_lossless_3mode` / `hosvd_full_rank_is_near_lossless_wide_matrix_modes`: **G3 equivalent** — full-rank lossless reconstruction.
- `hosvd_low_rank_recovers_exact_low_rank_tensor`: **G1 equivalent** — rank-`(2,2,2)` exact recovery.
- `hosvd_truncated_error_bounded_by_discarded_energy`: reconstruction energy does not exceed original.
- `factor_columns_are_orthonormal`: factor matrix columns are unit-norm + mutually orthogonal (SVD property).
- `reconstruction_error_decreases_with_higher_ranks`: error is monotonic non-increasing in ranks.
- `two_mode_tucker_matches_truncated_svd_energy`: 2-mode Tucker ≈ truncated SVD (energy match).
- `owned_decompose_matches_into_path`: convenience API matches hot-path API.
- `compression_ratio_is_correct` / `full_rank_compression_ratio_is_above_one`: compression accounting.
- `decompose_rejects_wrong_input_size` / `reconstruct_rejects_wrong_out_shape`: error paths.
- `decompose_is_deterministic_across_calls`: same input → bit-identical output (sync-boundary safety).
- `shard_batch_shape_8_8_8_smoke` / `shard_batch_shape_16_8_8_smoke`: integration-target shapes.

---

## Design notes (for future reference)

### The wide-matrix transpose trick

Mode-n unfoldings of `(N, R, C)` tensors can be wide: e.g. mode-1 of `(16, 8, 8)` is `(8, 128)`. The one-sided-Jacobi SVD's working matrix `V` has shape `(n_cols, n_cols)` — factoring `(8, 128)` directly would allocate a `128² = 16384`-float `V`. The transpose trick: if `I_n < prod_others`, transpose the unfolding to `(prod_others, I_n) = (128, 8)`, factor that (V is `8² = 64` floats), and read the right singular vectors of the transpose as the factor (they are the left singular vectors of the original). This keeps scratch memory proportional to `min(I_n, prod_others)²` not `max(...)²`.

### The SVD_MAX_RANK = 16 constraint

The underlying `one_sided_jacobi_svd_into` uses a stack-allocated `[f32; 16]` raw-sigma buffer (`subspace_phase_gate.rs:687`). Tucker configs are rejected at construction (`TuckerConfig::new`) if any mode-n unfolding's smaller dimension `min(shape[n], prod_others)` exceeds 16. For shard-batch `(N, 8, 8)`: mode 0 needs `min(N, 64) ≤ 16` → `N ≤ 16`. Larger batches must be chunked into groups of ≤ 16 shards per Tucker call.

This is NOT a fundamental limit — lifting it would require either (a) heap-allocating the raw-sigma buffer in `one_sided_jacobi_svd_into`, or (b) adding a chunked/block-Jacobi SVD path. Both are out of scope for Plan 326; the cold-tier archival use case naturally batches at zone granularity (≤ 16 shards/zone is typical).
