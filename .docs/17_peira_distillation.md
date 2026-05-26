# PEIRA Modelless Distillation

**Feature flag:** `peira_distill` (not default-on)
**Plan:** 153 · **Research:** 115 · **Source:** arXiv:2605.17671

## Overview

PEIRA (Predictive Encoders through Inter-View Regressor Alignment) provides a theoretically grounded, collapse-free distillation loss. Instead of backpropagating through a matrix inverse, it:

1. Maintains EMA estimates of k×k covariance matrices Σ (cross-view) and N (within-view)
2. Computes closed-form P\* = Σ(N + λI)⁻¹ and Q\* = (N + λI)⁻¹
3. Evaluates the auxiliary loss L_aux without differentiating through the inverse
4. Uses L_aux gradients to update encoder parameters

All k×k operations run on CPU — no GPU/WGSL needed since k is typically 128–512.

## Architecture

```mermaid
graph TD
    A[Student Repr u] --> C[PeiraCovariance]
    B[Teacher Repr v] --> C
    C -->|EMA update| D[Σ cross-view]
    C -->|EMA update| E[N within-view]
    D --> F[predictor]
    E --> F
    F -->|P* Q*| G[peira_aux_loss]
    A --> G
    B --> G
    G -->|loss f64| H[PeiraDistiller]
    C -->|σ N| I[peira_alignment_score]
    I -->|α ∈ 0 1| H
```

## Key Types

### `PeiraConfig` — `crates/katgpt-core/src/peira.rs`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lambda` | `f64` | 0.1 | Regularization λ > 0. Larger → fewer canonical directions |
| `ema_rate` | `f64` | 0.9 | EMA momentum α ∈ (0,1). Higher = more stable |
| `dim` | `usize` | 8 | Representation dimension k |

```rust
let config = PeiraConfig::new(128)
    .with_lambda(0.1)
    .with_ema_rate(0.95);
```

### `PeiraCovariance` — `crates/katgpt-core/src/peira.rs`

Tracks running EMA estimates of Σ and N:

- `update(student, teacher)` — updates EMA with one (u, v) pair
- `predictor() -> (P*, Q*)` — closed-form optimal predictor matrices
- `sigma()`, `n_matrix()` — read current covariance estimates
- `reset()` — clear for new episode

### `PeiraDistiller` — `src/distill/peira.rs`

Wraps the full SC-PEIRA Algorithm 1 loop:

```rust
let mut distiller = PeiraDistiller::new(config);
for (student, teacher) in pairs {
    let (loss, alignment) = distiller.step(&student, &teacher);
}
```

- `step()` returns `(auxiliary_loss, alignment_score)`
- `loss_history()`, `alignment_history()` — training curves
- `predictor()` — current (P\*, Q\*)

### `peira_alignment_score` — `src/distill/peira.rs`

Spectral alignment metric α ∈ [0, 1]:

- **1.0** = perfect canonical structure recovered
- **0.0** = random alignment (early training)

Uses power iteration to find top eigenvectors of Σ and N, then computes their cosine similarity.

## Auxiliary Loss

L_aux = -½ Tr(Σ\_sample · P\*^T) + ¼ Tr(P\* · (N\_sample + λI) · P\*^T) + λ/2 (‖u‖² + ‖v‖²)

Key property: no backpropagation through the matrix inverse. The inverse is computed once from EMA statistics, then the loss is evaluated against the current sample.

## GOAT Proof Results

All gates passed via `core_06_peira` example (k=8, 500 steps):

| Gate | Result | Evidence |
|------|--------|----------|
| T1: Compiles under `peira_distill` | ✅ | 1697 tests passed |
| T2: EMA covariance tracks identity | ✅ | Q\* diagonal all positive |
| T3: Auxiliary loss finite | ✅ | loss = -1.354, finite |
| T4: SC-PEIRA loop completes | ✅ | 500 steps processed |
| T8: Collapse-free | ✅ | min norm = 0.723 > 0 |
| T9: CCA alignment ≥ 0.9 | ✅ | final α = 0.987 |

## Feature Dependencies

```toml
# Root Cargo.toml
peira_distill = ["katgpt-core/peira_distill", "bandit"]
```

Interacts with: `bandit` (required), `sr2am_configurator` (optional, future T11).

## Running

```sh
# Tests
cargo test --features peira_distill --lib peira --quiet

# GOAT example
cargo run --example core_06_peira --features peira_distill --release

# Clippy
cargo clippy --features peira_distill --examples --quiet
```

## Files

| File | Content |
|------|---------|
| `crates/katgpt-core/src/peira.rs` | `PeiraConfig`, `PeiraCovariance`, `peira_aux_loss`, matrix ops |
| `crates/katgpt-core/src/lib.rs` | Feature-gated re-exports |
| `crates/katgpt-core/Cargo.toml` | `peira_distill` feature gate |
| `src/distill/peira.rs` | `PeiraDistiller`, `peira_alignment_score`, `synthetic_cca_sample` |
| `src/distill/mod.rs` | Feature-gated module |
| `examples/core_06_peira.rs` | GOAT proof demo |
