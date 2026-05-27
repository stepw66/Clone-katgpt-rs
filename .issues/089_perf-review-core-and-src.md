# Performance Review: katgpt-core + src/

## Status: ✅ Fixed
#1 `sample_token_into` already exists with pre-allocated CDF. #2 softmax already uses `simd_sum_f32`. #3 softmax already uses `simd_add_scalar_inplace`. #7 added `simd_fused_sub_scale_inplace` for fused sub+scale in `softmax_scaled`. #8 removed redundant rmsnorm in `forward_coda`. Issues #4, #5, #6, #9, #10 are low priority or already addressed.

Audited against `.contexts/optimization.md` guidelines.

**Scope**: `crates/katgpt-core/src/` and `src/` (transformer, weights, types re-exports, simd re-exports).

---

## Issue 1: `sample_token` allocates `Vec` on every call (HOT PATH)

**File**: `crates/katgpt-core/src/types.rs:1617–1642`

`sample_token()` builds a `Vec::with_capacity(n)` CDF every call. For a 256K vocab this is a ~1MB allocation per token — on the single hottest decode path.

**Guideline violated**: _Don't: Allocate inside hot loops_ ("Pre-allocate output arrays upfront, write in-place instead of collecting per-iteration").

**Fix**: Pre-allocate a `cdf: Vec<f32>` in `ForwardContext` (already holds `logits: Vec<f32>`), reuse it via `cdf.clear(); cdf.extend(...)` or pass as `&mut [f32]`:

```rust
pub fn sample_token_into(probs: &[f32], rng: &mut Rng, cdf: &mut Vec<f32>) -> usize {
    cdf.clear();
    cdf.reserve(probs.len());
    let mut sum = 0.0f32;
    for &p in probs {
        sum += p;
        cdf.push(sum);
    }
    // ... binary search unchanged ...
}
```

**Expected impact**: ~1–3 μs/token at 256K vocab (eliminates alloc + dealloc). High-impact for decode throughput.

---

## Issue 2: `softmax` and `softmax_scaled` use scalar `.iter().sum()` instead of SIMD

**File**: `crates/katgpt-core/src/types.rs:1320, 1349`

Both functions compute `let sum: f32 = x.iter().copied().sum()` for the normalization pass. The project already has `simd_sum_f32` which does this with NEON/AVX2.

**Guideline violated**: _Don't: Recompute unchanged values_ (using scalar when SIMD equivalent exists in same crate).

**Fix**: Replace both occurrences:

```rust
// Before
let sum: f32 = x.iter().copied().sum();

// After
let sum: f32 = crate::simd::simd_sum_f32(x);
```

**Expected impact**: For softmax sizes > 16 elements, ~2–4× speedup on the sum pass. Especially significant for `attention_head` which calls `softmax_scaled`-equivalent logic inline (and correctly uses `simd_sum_f32` there — but `softmax_scaled` itself doesn't).

---

## Issue 3: `softmax` uses scalar subtraction loop instead of SIMD `add_scalar_inplace`

**File**: `crates/katopz/git/katgpt-core/src/types.rs:1312–1314`

```rust
for val in x.iter_mut() {
    *val -= max_val;
}
```

The project already has `simd_add_scalar_inplace`. Compare with `softmax_scaled` (L1341) which correctly does `*val = (*val - max_val) * inv_temp` (a fused multiply that can't easily use the existing SIMD helper — but the plain subtract can).

**Fix**:

```rust
// Before
for val in x.iter_mut() {
    *val -= max_val;
}

// After
crate::simd::simd_add_scalar_inplace(x, -max_val);
```

**Expected impact**: Minor but consistent — eliminates scalar loop for arrays > 8 elements.

---

## Issue 4: `gegelu` / `silu` / `swiglu` use scalar copy loops for negation

**File**: `crates/katgpt-core/src/types.rs:1383, 1428, 1457`

All three activations do:
```rust
for j in 0..CHUNK {
    buf[j] = -gate[i + j];  // or -x[i + j]
}
```

The project has `simd_scale_inplace` which could do `simd_scale_inplace(&mut buf, -1.0)` after copying, but more importantly has `simd_fused_scale_acc` and friends. The copy-negate pattern could use `simd_scale_inplace` after a copy, or better yet, a `simd_neg_into(dst, src, len)` helper. Alternatively, copy + negate is so cheap the existing approach is fine — but the inconsistency with the rest of the codebase (using SIMD everywhere else) is notable.

**Severity**: Low. The copy is L1-friendly at CHUNK=64 (256 bytes). Only worth fixing if profiling shows these activations as bottleneck.

---

## Issue 5: `tiled_attention_batched` allocates `scores_buf` per parallel task

**File**: `crates/katgpt-core/src/attention.rs:328`

```rust
output.par_chunks_mut(head_size).enumerate().for_each(|(idx, out_chunk)| {
    let mut scores_buf = vec![0.0f32; scores_buf_size];  // alloc per task!
    ...
});
```

Each rayon task allocates a `scores_buf`. For batch × heads = 32 and seq_len = 128, that's 32 × 64KB = 2MB of allocations.

**Guideline violated**: _Cache allocations: `Vec::with_capacity()` once, `clear()` + reuse across calls_.

**Fix options**:
1. **Best**: Use `rayon::ThreadPool` + thread-local scratch buffers (one per thread, not per task).
2. **Simple**: Accept a pre-allocated `&mut [Vec<f32>]` scratch pool from the caller (one per thread).

```rust
let pool = vec![vec![0.0f32; scores_buf_size]; rayon::current_num_threads().min(total)];
let pool = std::sync::Mutex::new(pool);  // or use scope + split

output.par_chunks_mut(head_size).enumerate().for_each(|(idx, out_chunk)| {
    let mut buf = vec![0.0f32; scores_buf_size]; // TODO: reuse from pool
    ...
});
```

Note: The `Mutex` approach partially violates the _Don't: Use Mutex in Rayon closures_ guideline. The thread-local or scope-based approach is preferred.

**Expected impact**: ~5–10 μs saved per batch call (eliminates N allocs). Important for prefill path.

---

## Issue 6: `tiled_attention_inner` — score tile inner loops not SIMD-ized

**File**: `crates/katgpt-core/src/attention.rs:153–163`

The Q×K score computation and the P̃×V accumulation loops (L190–200) are scalar:

```rust
for j in 0..actual_bc {
    let k_off = (k_start + j) * head_dim;
    s_tile[i * BC + j] = crate::simd::simd_dot_f32(
        &q[q_off..q_off + head_dim],
        &k[k_off..k_off + head_dim],
        head_dim,
    );
}
```

Each Q row is dotted against each K row — this is correct but the outer loops (i, j) iterate over the full BR × BC tile. For BR=8, BC=128 this is 1024 dot products. The dot products themselves use SIMD, but the orchestration overhead (slice creation per iteration) is non-trivial.

**Optimization opportunity**: The `attention_head` function in `transformer.rs` (L525–580) does the same pattern but avoids the tile abstraction for small N. The tiled path should only be used when N > 128 (which it already is — `TILED_ATTENTION_THRESHOLD = 128`). At those sizes, the current approach is reasonable. **Low priority** — the SIMD is in the right place (dot products).

---

## Issue 7: `softmax_scaled` subtraction could fuse with `inv_temp` multiply

**File**: `crates/katopz/git/katgpt-rs/crates/katgpt-core/src/types.rs:1341–1343`

```rust
for val in x.iter_mut() {
    *val = (*val - max_val) * inv_temp;
}
```

This is already a fused multiply-subtract, but done in scalar. The project has no `simd_fused_sub_scale_inplace(dst, max_val, scale)` helper. Could add one to SIMD-ize this:

```rust
// dst[i] = (dst[i] - sub) * scale
pub fn simd_fused_sub_scale_inplace(dst: &mut [f32], sub: f32, scale: f32) {
    // NEON: vmlaq_n_f32(vsubq_n, ...) or similar
}
```

**Expected impact**: ~30–40 ns saved per softmax call (single pass vs two passes). Medium priority since softmax is called per-head per-layer.

---

## Issue 8: `forward_base` / `forward_coda` — `rmsnorm` called twice in `forward_coda`

**File**: `src/transformer.rs:1966–1968`

```rust
rmsnorm(&mut ctx.x);          // pre-attention norm
ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
rmsnorm(&mut ctx.x);          // pre-MLP norm (applied immediately after copy)
```

In `forward_base` (L1672), there's only one `rmsnorm` before attention, then another before MLP at L1745 — with a full layer of computation between them. But in `forward_coda`, the second `rmsnorm` is applied immediately after the copy, meaning `x` is normalized twice in a row with no intervening computation. This looks like it matches the CODA paper's fused kernel pattern where the second norm is "delayed" and applied by the fused kernel — but if the fused kernel (`simd_matmul_rmsnorm_activation`) already applies the rstd scaling, the second standalone `rmsnorm` is redundant and wastes ~100ns.

**Question**: Is the second `rmsnorm` at L1968 intentional for CODA, or is it redundant because `simd_matmul_rmsnorm_activation` already applies the delayed RMS?

**If redundant**, removing it saves one full pass over `n_embd` elements (~50–100ns).

---

## Issue 9: `gegelu_tanh` is fully scalar — no SIMD at all

**File**: `crates/katopz-git/katgpt-core/src/types.rs:1407–1415`

```rust
pub fn gegelu_tanh(hidden: &mut [f32], gate: &[f32], up: &[f32]) {
    let sqrt_2_over_pi = ...;
    for i in 0..hidden.len() {
        let g = gate[i];
        let inner = sqrt_2_over_pi * (g + 0.044715 * g * g * g);
        let gelu_val = 0.5 * g * (1.0 + inner.tanh());
        hidden[i] = gelu_val * up[i];
    }
}
```

Compare with `gegelu` (L1376) which uses SIMD exp via chunked buffers. `gegelu_tanh` (used in Gemma 2) is fully scalar with per-element `tanh()` calls. This is the MLP activation and runs over `mlp_hidden` elements per token.

**Fix**: Chunk it like the other activations, using `simd_exp_inplace` for the tanh approximation, or implement a SIMD `tanh` approximation (or use the existing exp-based sigmoid path with tanh = 2·sigmoid(2x) - 1).

**Expected impact**: At `mlp_hidden=2304`, this is ~2–5 μs per layer. Medium priority.

---

## Issue 10: `lora_apply` inner loop could use `simd_dot_f32` already does, but rank may be small

**File**: `crates/katopz-git/katgpt-core/src/types.rs:1898–1913`

`lora_apply` calls `simd_dot_f32` for each output row — this is correct. But for typical `rank=4` or `rank=8`, the SIMD overhead (dispatch, accumulator setup) may exceed the benefit. The function already dispatches to scalar for small lengths in `simd_dot_f32`, so this is handled correctly. **No issue** — noting for completeness.

---

## Summary: Priority-ordered action items

| # | Issue | Impact | Effort | File |
|---|-------|--------|--------|------|
| 1 | `sample_token` heap-allocs CDF every call | **High** (1–3 μs/token) | Low | `types.rs:1617` |
| 2 | `softmax`/`softmax_scaled` use scalar `.sum()` | **Medium** (2–4× sum pass) | Trivial | `types.rs:1320,1349` |
| 7 | `softmax_scaled` scalar sub+mul not fused SIMD | **Medium** (~30–40 ns/call) | Medium | `types.rs:1341` |
| 3 | `softmax` scalar subtract vs SIMD | **Low–Medium** | Trivial | `types.rs:1312` |
| 8 | `forward_coda` double `rmsnorm` — may be redundant | **Medium** (50–100 ns/layer) | Verify | `transformer.rs:1966–1968` |
| 9 | `gegelu_tanh` fully scalar | **Medium** (2–5 μs/layer) | Medium | `types.rs:1407` |
| 5 | `tiled_attention_batched` allocs per rayon task | **Low–Medium** (5–10 μs/batch) | Medium | `attention.rs:328` |
| 4 | Activation negation loops scalar | **Low** | Low | `types.rs:1383,1428,1457` |
| 6 | Tiled attention score loops overhead | **Low** (already SIMD dots) | High | `attention.rs:153` |

**Recommended first actions**: Fix issues #1 and #2 immediately (trivial changes, measurable impact). Then verify #8 and fix #9.
