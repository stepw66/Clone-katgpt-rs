//! Shared DEC test helpers (Issue 037 T1 verdict: test-helper DRY extraction).
//!
//! Provides the duplicated helpers that previously appeared in 2-3 test modules
//! across `heat_kernel.rs`, `motor_gated.rs`, and `nonlinear_heat_kernel.rs`.
//! Eliminates ~80 LOC of copy-paste drift risk and removes 3 of the 4
//! `clippy::too_many_arguments` lints that motivated Issue 037.
//!
//! # Why `#[path]` and not `#[macro_export]` or a dev-dep?
//!
//! Same reasoning as `katgpt-core/tests/common/mod.rs` (sibling pattern,
//! Issue 044 T3): each test module is its own compilation unit, and
//! `#[macro_export]` would pollute the public API. The `#[path]` include
//! keeps the helpers internal to whichever test module pulls them in.
//!
//! # Usage from a `src/*.rs` test module
//!
//! ```ignore
//! #[cfg(test)]
//! mod tests {
//!     #[path = "../tests/common/mod.rs"]
//! mod common;
//!     // common::place_bump(...), common::zero_field(...), etc.
//! }
//! ```
//!
//! `crate::types::{CellComplex, CochainField}` resolves correctly here because
//! `#[path]` includes this file as a child module of whatever `mod tests`
//! pulls it in — so `crate::` still refers to the `katgpt_dec` crate root.

use crate::types::{CellComplex, CochainField};

/// Build a rank-0 `w×h` grid cochain with `dim` channels, zeroed.
///
/// Used by every DEC operator test to set up the initial field state.
pub(crate) fn zero_field(cx: &CellComplex, dim: usize) -> CochainField {
    CochainField::zeros(0, cx.n_vertices(), dim)
}

/// Place a Gaussian bump of amplitude `amp` at grid cell `(cx_pos, cy_pos)`
/// into channel `ch` of a 2D-grid cochain.
///
/// Non-negative everywhere (so ReLU is identity — used for linear-vs-Euler
/// comparison tests in `heat_kernel.rs` and `nonlinear_heat_kernel.rs`).
#[allow(clippy::too_many_arguments, reason = "test helper: 8 named positional params (field, w, h, cx, cy, ch, amp, sigma) are the natural signature; grouping into a BumpSpec struct would add boilerplate at every call site without reducing the operand-swap risk that motivated Issue 037")]
pub(crate) fn place_bump(
    field: &mut CochainField,
    w: usize,
    h: usize,
    cx_pos: usize,
    cy_pos: usize,
    ch: usize,
    amp: f32,
    sigma: f32,
) {
    let dim = field.dim;
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx_pos as f32;
            let dy = y as f32 - cy_pos as f32;
            let r2 = dx * dx + dy * dy;
            let v = amp * (-r2 / (2.0 * sigma * sigma)).exp();
            field.data[(y * w + x) * dim + ch] = v;
        }
    }
}

/// L2 norm of a cochain field (all channels).
pub(crate) fn l2_norm(field: &CochainField) -> f32 {
    field.data.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// L2 distance between two cochain fields (all channels).
pub(crate) fn l2_dist(a: &CochainField, b: &CochainField) -> f32 {
    debug_assert_eq!(a.data.len(), b.data.len());
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}
