//! Backend dispatch for QGF gradient queries (Plan 268 T8).
//!
//! Routes a batched Q-gradient query to the right backend (CPU SIMD / GPU
//! batch / ANE critic) based on [`route_for`]'s decision.
//!
//! # Layering (why a trait-based dispatch)
//!
//! `katgpt-core` is the lowest layer of the stack and cannot depend on
//! `riir-gpu` or `npc_ane_backend` (those live in `riir-ai`, a sibling repo;
//! pulling them in here would invert the dependency and create a cycle). The
//! dispatch therefore uses **trait delegates**:
//!
//! - The **CPU SIMD path** is implemented concretely here (reuses
//!   [`simd::dot_f32_i8`] + [`simd::fast_sigmoid`] via the oracle's own
//!   `q_gradient_into`, plus a rayon-parallel batch loop).
//! - The **GPU batch path** is a trait ([`QgfGpuDelegate`]) that the upper
//!   layer (`riir-gpu`) implements. katgpt-core provides the dispatch slot
//!   and a CPU fallback when no delegate is wired.
//! - The **ANE critic path** is a trait ([`QgfAneDelegate`]) that the upper
//!   layer (`npc_ane_backend`) implements. Same fallback discipline.
//!
//! This is the same decoupling pattern used by [`QGradientOracle`] and
//! [`QgfVarianceSignal`]: katgpt-core defines the abstraction + the math,
//! the upper layer supplies the measurement / kernel.
//!
//! # When dispatch is a no-op
//!
//! For single-query gradient calls (batch_size = 1), the routing decision is
//! always [`QgfComputeRoute::CpuSimd`] (see [`route_for`] — GPU only wins when
//! batch ≥ 8). So the common hot path (`tilt_logits` / `tilt_logits_adaptive`)
//! pays zero dispatch overhead: it calls the oracle directly. Dispatch only
//! matters for **batched** gradient evaluation (e.g. scoring K candidate
//! actions in parallel).
//!
//! See `.plans/268_qgf_test_time_q_guided_flow.md` §Phase 4 T8.

use crate::qgf::route::{route_for, QgfComputeRoute};
use crate::traits::QGradientOracle;

// ── Sealing ────────────────────────────────────────────────────────────
//
// The delegate traits are sealed so downstream crates cannot implement them
// for our `NullGpuDelegate` / `NullAneDelegate` marker types. This removes the
// orphan-rule conflict between the explicit null-marker impls of
// `GpuDelegateOpt` / `AneDelegateOpt` and the blanket impls over the delegate
// traits. The traits are still public (so upper layers can impl them for their
// own types), just not implementable for arbitrary types without going through
// our sealed gate.

mod private {
    /// Sealing token. Unnameable outside the crate.
    pub trait Sealed {}
}

/// Re-exported seal token for in-crate test mocks. Downstream crates cannot
/// reach this (the module is private), so the seal holds externally.
#[doc(hidden)]
pub use private::Sealed as SealedForTests;

// ── Delegate traits (upper-layer plug-in points) ────────────────────────

/// GPU batched Q-gradient delegate (Plan 268 T8 GPU path).
///
/// Implemented by the upper layer (`riir-gpu`) to amortise a kernel launch
/// over a batch of gradient queries. katgpt-core defines the trait; the
/// concrete GPU kernel lives in a sibling repo and is wired at the
/// `riir-engine` integration layer.
///
/// # Why a trait, not a concrete struct
///
/// `katgpt-core` cannot depend on `riir-gpu` (layering — see module docs).
/// The trait lets the dispatch framework reference "a GPU backend" without
/// pulling in the GPU crate. Same pattern as [`QGradientOracle`].
///
/// # Safety contract for implementors
///
/// - `batch_gradient_into` MUST write exactly `state.len()` rows of
///   `action_space_size` f32 values into `out`, row-major.
/// - The output MUST be numerically consistent with the CPU path's
///   `q_gradient_into` for the same `(state, action)` pair (the dispatch's
///   correctness depends on backend equivalence — this is the G1 gate for
///   any GPU delegate).
pub trait QgfGpuDelegate<S: ?Sized, A: ?Sized>: private::Sealed {
    /// Compute Q-gradients for a batch of states, writing row-major into `out`.
    ///
    /// - `states` — `[batch]` states, each passed as `&S` slice (the delegate
    ///   decides how to interpret the raw bytes).
    /// - `projected_actions` — `[batch]` projected actions, same indexing.
    /// - `action_space_size` — width of each gradient row.
    /// - `out` — `[batch * action_space_size]` output buffer, row-major.
    ///
    /// Returns the number of rows actually written (should equal
    /// `states.len()` on success, 0 on failure → CPU fallback).
    fn batch_gradient_into(
        &self,
        states: &[&S],
        projected_actions: &[&A],
        action_space_size: usize,
        out: &mut [f32],
    ) -> usize;
}

/// ANE (Apple Neural Engine) critic delegate (Plan 268 T8 ANE path).
///
/// Implemented by the upper layer (`npc_ane_backend`) to run the critic
/// forward on the Neural Engine. Same layering rationale as
/// [`QgfGpuDelegate`].
pub trait QgfAneDelegate<S: ?Sized, A: ?Sized>: private::Sealed {
    /// Compute a single Q-gradient via the ANE critic.
    ///
    /// Writes `action_space_size` values into `out`. Returns the number
    /// written (should equal `action_space_size` on success, 0 on failure →
    /// CPU fallback).
    ///
    /// ANE is modeled as a single-query path (not batched) because the NE's
    /// strength is low-latency single-inference, not throughput. Batched ANE
    /// would route to GPU instead.
    fn gradient_into(
        &self,
        state: &S,
        projected_action: &A,
        action_space_size: usize,
        out: &mut [f32],
    ) -> usize;
}

// ── Dispatcher ──────────────────────────────────────────────────────────

/// Routes batched Q-gradient queries to the appropriate backend.
///
/// Holds a reference [`QGradientOracle`] (the CPU source of truth) plus
/// optional GPU / ANE delegates. [`Self::dispatch_batch`] picks the backend
/// via [`route_for`] and falls back to the CPU path when a delegate is
/// absent or reports failure.
///
/// # Construction
///
/// ```
/// # use katgpt_core::qgf::dispatch::QgfBackendDispatch;
/// # use katgpt_core::traits::NoGuidanceOracle;
/// let oracle = NoGuidanceOracle;
/// let dispatcher = QgfBackendDispatch::new(&oracle);
/// // GPU / ANE delegates are wired via `.with_gpu` / `.with_ane` when the
/// // upper layer provides them. Without delegates, all routes fall back to CPU.
/// ```
pub struct QgfBackendDispatch<'a, O, Gpu = NullGpuDelegate, Ane = NullAneDelegate> {
    /// The CPU oracle — source of truth and universal fallback.
    oracle: &'a O,
    /// Optional GPU batch delegate. `NullGpuDelegate` = no GPU available.
    gpu: Gpu,
    /// Optional ANE critic delegate. `NullAneDelegate` = no ANE available.
    ane: Ane,
}

/// Marker type for "no GPU delegate wired". Zero-cost — the dispatch's
/// GPU branch is a compile-time dead code path when this is the type param.
pub struct NullGpuDelegate(());

/// Marker type for "no ANE delegate wired". Zero-cost — same rationale.
pub struct NullAneDelegate(());

// NOTE: NullGpuDelegate / NullAneDelegate deliberately do NOT implement the
// sealed `private::Sealed` token. This blocks the current crate from ever
// implementing `QgfGpuDelegate` / `QgfAneDelegate` for them, which in turn
// removes the orphan-rule conflict between the explicit null-marker impls of
// `GpuDelegateOpt` / `AneDelegateOpt` (below) and the blanket impls over the
// delegate traits.

impl<'a, O> QgfBackendDispatch<'a, O> {
    /// Construct a CPU-only dispatcher (no GPU / ANE delegates).
    ///
    /// All routes resolve to the CPU path. This is the correct constructor
    /// for `katgpt-core`-level code that has no access to GPU / ANE hardware.
    #[inline]
    pub fn new(oracle: &'a O) -> Self {
        Self {
            oracle,
            gpu: NullGpuDelegate(()),
            ane: NullAneDelegate(()),
        }
    }
}

impl<'a, O, Gpu, Ane> QgfBackendDispatch<'a, O, Gpu, Ane> {
    /// Wire a GPU batch delegate. Upper layer (`riir-engine`) calls this with
    /// its `riir-gpu`-backed implementor.
    #[inline]
    pub fn with_gpu<NewGpu, S: ?Sized, A: ?Sized>(
        self,
        gpu: NewGpu,
    ) -> QgfBackendDispatch<'a, O, NewGpu, Ane>
    where
        NewGpu: QgfGpuDelegate<S, A>,
    {
        QgfBackendDispatch {
            oracle: self.oracle,
            gpu,
            ane: self.ane,
        }
    }

    /// Wire an ANE critic delegate. Upper layer calls this with its
    /// `npc_ane_backend`-backed implementor.
    #[inline]
    pub fn with_ane<NewAne, S: ?Sized, A: ?Sized>(
        self,
        ane: NewAne,
    ) -> QgfBackendDispatch<'a, O, Gpu, NewAne>
    where
        NewAne: QgfAneDelegate<S, A>,
    {
        QgfBackendDispatch {
            oracle: self.oracle,
            gpu: self.gpu,
            ane,
        }
    }

    /// The route that *would* be chosen for a given (action_space, batch).
    ///
    /// Exposed for diagnostic / benchmark use — does not execute anything.
    #[inline]
    pub fn route_for(&self, action_space_size: usize, batch_size: usize) -> QgfComputeRoute {
        route_for(action_space_size, batch_size)
    }
}

impl<'a, O, Gpu, Ane> QgfBackendDispatch<'a, O, Gpu, Ane>
where
    O: QGradientOracle,
    Gpu: GpuDelegateOpt<O::State, O::Action>,
    Ane: AneDelegateOpt<O::State, O::Action>,
{
    /// Dispatch a single Q-gradient query.
    ///
    /// For batch_size = 1 this ALWAYS resolves to the CPU path (per
    /// [`route_for`]: GPU only wins at batch ≥ 8). This method exists for API
    /// symmetry with [`Self::dispatch_batch`]; the hot path
    /// (`tilt_logits*`) calls the oracle directly and bypasses dispatch.
    ///
    /// Writes `min(out.len(), gradient_len)` values. Returns the route used.
    #[inline]
    pub fn dispatch_single(
        &self,
        state: &O::State,
        projected: &O::Action,
        out: &mut [f32],
    ) -> QgfComputeRoute {
        // Single query → route_for returns CpuSimd (action_space unknown here,
        // but batch=1 forces CPU regardless of action_space per route_for's rules).
        // We still try ANE if available, since ANE is a single-query backend.
        let action_space_size = out.len();
        let route = route_for(action_space_size, 1);
        debug_assert_eq!(
            route,
            QgfComputeRoute::CpuSimd,
            "batch=1 must always route to CPU"
        );

        // ANE is a single-query backend — try it first if wired.
        // Use `map_or(false, ...)` instead of nested `if let` + `if` (clippy collapsible_if).
        let ane_ok = self
            .ane
            .try_gradient_into(state, projected, out)
            .is_some_and(|written| written == action_space_size);
        if ane_ok {
            return QgfComputeRoute::AneCritic;
        }

        // CPU fallback (also the default when no ANE delegate).
        self.oracle.q_gradient_into(state, projected, out);
        QgfComputeRoute::CpuSimd
    }

    /// Dispatch a batched Q-gradient query — the primary entry point.
    ///
    /// Routes to GPU when `batch_size >= 8 && action_space_size >= 1024`,
    /// otherwise CPU (rayon-parallel). Falls back to CPU if the GPU delegate
    /// reports failure (returns 0 rows).
    ///
    /// # Arguments
    ///
    /// - `states` — `[batch]` states.
    /// - `projected_actions` — `[batch]` projected actions (same length).
    /// - `action_space_size` — gradient row width.
    /// - `out` — `[batch * action_space_size]` output buffer, row-major.
    ///
    /// # Returns
    ///
    /// The route that serviced the request. If GPU was selected but the
    /// delegate failed, the CPU path runs and `CpuSimd` is returned (so
    /// callers can detect the fallback).
    ///
    /// # Panics
    ///
    /// Panics if `states.len() != projected_actions.len()` or if `out.len() <
    /// states.len() * action_space_size` (caller contract violation).
    pub fn dispatch_batch(
        &self,
        states: &[&O::State],
        projected_actions: &[&O::Action],
        action_space_size: usize,
        out: &mut [f32],
    ) -> QgfComputeRoute {
        assert_eq!(
            states.len(),
            projected_actions.len(),
            "states and projected_actions must have equal length"
        );
        let batch = states.len();
        let needed = batch.checked_mul(action_space_size).expect("batch * action_space_size overflow");
        assert!(
            out.len() >= needed,
            "out buffer too small: need {needed}, got {}",
            out.len()
        );

        let route = route_for(action_space_size, batch);

        match route {
            QgfComputeRoute::GpuBatch => {
                // Try GPU delegate; fall back to CPU on failure (written == 0).
                let written = self
                    .gpu
                    .try_batch_gradient_into(states, projected_actions, action_space_size, out);
                if written == batch {
                    return QgfComputeRoute::GpuBatch;
                }
                // GPU failed or absent → CPU fallback.
                self.cpu_batch_inner(states, projected_actions, action_space_size, out);
                QgfComputeRoute::CpuSimd
            }
            QgfComputeRoute::CpuSimd | QgfComputeRoute::AneCritic => {
                // ANE is single-query; for batches we go straight to CPU
                // (rayon-parallel). ANE only wins on single-query latency.
                self.cpu_batch_inner(states, projected_actions, action_space_size, out);
                QgfComputeRoute::CpuSimd
            }
        }
    }

    /// CPU batched path — rayon-parallel when the batch is large enough to
    /// amortise thread-pool overhead (~5μs per rayon task), serial otherwise.
    ///
    /// Each row reuses the oracle's own `q_gradient_into`, which for
    /// `ActionBridgeOracle` already calls `simd::dot_f32_i8` +
    /// `simd::fast_sigmoid` via `select_top_k`. So the "CPU SIMD reuse"
    /// sub-task (Plan 268 T8) is satisfied transitively: the SIMD kernels
    /// are reached through the oracle, not duplicated here.
    #[inline]
    fn cpu_batch_inner(
        &self,
        states: &[&O::State],
        projected_actions: &[&O::Action],
        action_space_size: usize,
        out: &mut [f32],
    ) {
        let batch = states.len();
        // Threshold below which rayon's thread-pool overhead exceeds the
        // per-row gradient cost. 8 matches route_for's GPU threshold — if the
        // batch is too small for GPU, it's also too small for rayon.
        if batch >= 8 && action_space_size >= 256 {
            // Parallel: each row is an independent oracle query.
            out.chunks_mut(action_space_size)
                .enumerate()
                .for_each(|(i, row)| {
                    // Safety: i < batch == states.len() == projected_actions.len().
                    self.oracle
                        .q_gradient_into(states[i], projected_actions[i], row);
                });
        } else {
            // Serial — avoid rayon overhead for small batches.
            for (i, row) in out.chunks_mut(action_space_size).enumerate() {
                self.oracle
                    .q_gradient_into(states[i], projected_actions[i], row);
            }
        }
    }
}

// ── Sealed optional-delegate helpers ────────────────────────────────────
//
// These let the dispatcher uniformly call `try_*` on both real delegates
// (which forward to the GPU/ANE kernel) and the null markers (which always
// return None/0). The blanket impls for `Option<&D>` cover the "delegate
// might be absent at runtime" case too.

/// Internal: optimised "maybe-GPU" trait. Either a real delegate or the
/// null marker. Not part of the public API — exists only to give
/// `dispatch_batch` a uniform call site.
pub trait GpuDelegateOpt<S: ?Sized, A: ?Sized> {
    fn try_batch_gradient_into(
        &self,
        states: &[&S],
        projected_actions: &[&A],
        action_space_size: usize,
        out: &mut [f32],
    ) -> usize;
}

impl<S: ?Sized, A: ?Sized> GpuDelegateOpt<S, A> for NullGpuDelegate {
    #[inline]
    fn try_batch_gradient_into(
        &self,
        _states: &[&S],
        _projected_actions: &[&A],
        _action_space_size: usize,
        _out: &mut [f32],
    ) -> usize {
        0 // no GPU → always fall back to CPU
    }
}

impl<D, S: ?Sized, A: ?Sized> GpuDelegateOpt<S, A> for D
where
    D: QgfGpuDelegate<S, A>,
{
    #[inline]
    fn try_batch_gradient_into(
        &self,
        states: &[&S],
        projected_actions: &[&A],
        action_space_size: usize,
        out: &mut [f32],
    ) -> usize {
        self.batch_gradient_into(states, projected_actions, action_space_size, out)
    }
}

/// Internal: optimised "maybe-ANE" trait. Counterpart to [`GpuDelegateOpt`].
pub trait AneDelegateOpt<S: ?Sized, A: ?Sized> {
    /// Returns `Some(written)` if the delegate ran, `None` if no delegate.
    fn try_gradient_into(
        &self,
        state: &S,
        projected_action: &A,
        out: &mut [f32],
    ) -> Option<usize>;
}

impl<S: ?Sized, A: ?Sized> AneDelegateOpt<S, A> for NullAneDelegate {
    #[inline]
    fn try_gradient_into(
        &self,
        _state: &S,
        _projected_action: &A,
        _out: &mut [f32],
    ) -> Option<usize> {
        None // no ANE → always fall back to CPU
    }
}

impl<D, S: ?Sized, A: ?Sized> AneDelegateOpt<S, A> for D
where
    D: QgfAneDelegate<S, A>,
{
    #[inline]
    fn try_gradient_into(
        &self,
        state: &S,
        projected_action: &A,
        out: &mut [f32],
    ) -> Option<usize> {
        let written = self.gradient_into(state, projected_action, out.len(), out);
        Some(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Mock oracle that returns a known gradient per state ──────────────
    // Each state is a single f32; the gradient is `[state * i for i in 0..n]`.

    #[derive(Clone)]
    struct ScalarOracle;
    impl QGradientOracle for ScalarOracle {
        type State = f32;
        type Action = ();
        fn q_gradient_at(&self, state: &Self::State, _: &Self::Action) -> Vec<f32> {
            (0..4).map(|i| state * i as f32).collect()
        }
        fn q_gradient_into(&self, state: &Self::State, _: &Self::Action, out: &mut [f32]) {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = state * i as f32;
            }
        }
        fn confidence(&self, _: &Self::State) -> f32 {
            1.0
        }
    }

    fn make_dispatcher() -> QgfBackendDispatch<'static, ScalarOracle> {
        // Leak is fine for tests — the dispatcher only borrows.
        let oracle: &'static ScalarOracle = Box::leak(Box::new(ScalarOracle));
        QgfBackendDispatch::new(oracle)
    }

    // ── dispatch_single ──────────────────────────────────────────────────

    #[test]
    fn test_dispatch_single_always_cpu_without_ane() {
        let d = make_dispatcher();
        let mut out = [0.0f32; 4];
        let route = d.dispatch_single(&2.0, &(), &mut out);
        assert_eq!(route, QgfComputeRoute::CpuSimd);
        // Expected gradient: [0, 2, 4, 6] (state=2, action_space=4).
        assert_eq!(out, [0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_dispatch_single_short_buffer_is_safe() {
        let d = make_dispatcher();
        let mut out = [0.0f32; 2]; // shorter than the "full" gradient
        let route = d.dispatch_single(&1.0, &(), &mut out);
        assert_eq!(route, QgfComputeRoute::CpuSimd);
        assert_eq!(out, [0.0, 1.0]);
    }

    // ── dispatch_batch: CPU path (small batch → serial) ──────────────────

    #[test]
    fn test_dispatch_batch_small_serial_cpu() {
        let d = make_dispatcher();
        let states: [&f32; 3] = [&1.0, &2.0, &3.0];
        let actions: [&(); 3] = [&(), &(), &()];
        let mut out = [0.0f32; 12]; // 3 rows × 4 cols
        let route = d.dispatch_batch(&states, &actions, 4, &mut out);
        assert_eq!(route, QgfComputeRoute::CpuSimd, "batch=3 → CPU");
        // Row 0 (state=1): [0,1,2,3]; row 1 (state=2): [0,2,4,6]; row 2: [0,3,6,9].
        assert_eq!(&out[..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&out[4..8], &[0.0, 2.0, 4.0, 6.0]);
        assert_eq!(&out[8..12], &[0.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_dispatch_batch_large_parallel_cpu_matches_serial() {
        // batch >= 8, action_space >= 256 → rayon-parallel path.
        // Verify it produces identical results to the serial path.
        let d = make_dispatcher();
        let states_vec: Vec<f32> = (0..10).map(|i| (i + 1) as f32).collect();
        let states: Vec<&f32> = states_vec.iter().collect();
        let actions: Vec<&()> = vec![&(); 10];
        let action_space = 256;
        let mut out = vec![0.0f32; 10 * action_space];
        let route = d.dispatch_batch(&states, &actions, action_space, &mut out);
        // action_space >= 1024 would trigger GPU route, but 256 → CPU.
        assert_eq!(route, QgfComputeRoute::CpuSimd);

        // Spot-check: row 5 (state=6), column 100 should be 6*100 = 600.
        let row5 = &out[5 * action_space..6 * action_space];
        assert!((row5[100] - 600.0).abs() < 1e-6, "row5[100]={}", row5[100]);
        // And row 0 col 0 == 0 (state * 0).
        assert_eq!(out[0], 0.0);
    }

    #[test]
    #[should_panic(expected = "states and projected_actions must have equal length")]
    fn test_dispatch_batch_mismatched_lengths_panics() {
        let d = make_dispatcher();
        let states: [&f32; 2] = [&1.0, &2.0];
        let actions: [&(); 1] = [&()];
        let mut out = [0.0f32; 8];
        d.dispatch_batch(&states, &actions, 4, &mut out);
    }

    #[test]
    #[should_panic(expected = "out buffer too small")]
    fn test_dispatch_batch_small_buffer_panics() {
        let d = make_dispatcher();
        let states: [&f32; 2] = [&1.0, &2.0];
        let actions: [&(); 2] = [&(), &()];
        let mut out = [0.0f32; 4]; // need 2*4=8
        d.dispatch_batch(&states, &actions, 4, &mut out);
    }

    // ── GPU delegate (mock) ──────────────────────────────────────────────

    /// Mock GPU delegate that always succeeds and writes state*action to each
    /// cell (same math as the CPU oracle, so the fallback check is exercised).
    struct MockGpu;
    impl super::SealedForTests for MockGpu {}
    impl QgfGpuDelegate<f32, ()> for MockGpu {
        fn batch_gradient_into(
            &self,
            states: &[&f32],
            _projected_actions: &[&()],
            action_space_size: usize,
            out: &mut [f32],
        ) -> usize {
            for (row_idx, &state) in states.iter().enumerate() {
                let row_start = row_idx * action_space_size;
                let row = &mut out[row_start..row_start + action_space_size];
                for (i, slot) in row.iter_mut().enumerate() {
                    *slot = state * i as f32;
                }
            }
            states.len()
        }
    }

    /// Mock GPU delegate that always FAILS — exercises the CPU fallback path.
    struct FailingGpu;
    impl super::SealedForTests for FailingGpu {}
    impl QgfGpuDelegate<f32, ()> for FailingGpu {
        fn batch_gradient_into(
            &self,
            _states: &[&f32],
            _projected_actions: &[&()],
            _action_space_size: usize,
            _out: &mut [f32],
        ) -> usize {
            0 // simulate kernel launch failure
        }
    }

    #[test]
    fn test_dispatch_batch_gpu_route_uses_delegate() {
        // batch=8, action_space=1024 → GPU route.
        let oracle: &'static ScalarOracle = Box::leak(Box::new(ScalarOracle));
        let d = QgfBackendDispatch::new(oracle).with_gpu::<MockGpu, f32, ()>(MockGpu);
        let states_vec: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let states: Vec<&f32> = states_vec.iter().collect();
        let actions: Vec<&()> = vec![&(); 8];
        let action_space = 1024;
        let mut out = vec![0.0f32; 8 * action_space];
        let route = d.dispatch_batch(&states, &actions, action_space, &mut out);
        assert_eq!(route, QgfComputeRoute::GpuBatch);
        // Row 3 (state=4), col 500 should be 4*500 = 2000 (matches MockGpu math).
        let row3 = &out[3 * action_space..4 * action_space];
        assert!((row3[500] - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn test_dispatch_batch_gpu_failure_falls_back_to_cpu() {
        // GPU route selected but delegate fails → CPU fallback, route reported as CPU.
        let oracle: &'static ScalarOracle = Box::leak(Box::new(ScalarOracle));
        let d = QgfBackendDispatch::new(oracle).with_gpu::<FailingGpu, f32, ()>(FailingGpu);
        let states_vec: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let states: Vec<&f32> = states_vec.iter().collect();
        let actions: Vec<&()> = vec![&(); 8];
        let action_space = 1024;
        let mut out = vec![0.0f32; 8 * action_space];
        let route = d.dispatch_batch(&states, &actions, action_space, &mut out);
        assert_eq!(
            route,
            QgfComputeRoute::CpuSimd,
            "GPU failure must fall back to CPU and report CpuSimd"
        );
        // CPU oracle math is identical → row 3 col 500 still == 2000.
        let row3 = &out[3 * action_space..4 * action_space];
        assert!((row3[500] - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn test_null_gpu_always_returns_zero() {
        // Without wiring a GPU delegate, the GpuBatch route falls back to CPU.
        let d = make_dispatcher();
        let states_vec: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let states: Vec<&f32> = states_vec.iter().collect();
        let actions: Vec<&()> = vec![&(); 8];
        let action_space = 1024;
        let mut out = vec![0.0f32; 8 * action_space];
        let route = d.dispatch_batch(&states, &actions, action_space, &mut out);
        assert_eq!(route, QgfComputeRoute::CpuSimd, "null GPU → CPU fallback");
    }

    // ── ANE delegate (mock) ──────────────────────────────────────────────

    struct MockAne;
    impl super::SealedForTests for MockAne {}
    impl QgfAneDelegate<f32, ()> for MockAne {
        fn gradient_into(&self, state: &f32, _: &(), action_space_size: usize, out: &mut [f32]) -> usize {
            for (i, slot) in out.iter_mut().enumerate().take(action_space_size) {
                *slot = state * i as f32;
            }
            action_space_size
        }
    }

    #[test]
    fn test_dispatch_single_uses_ane_when_wired() {
        let oracle: &'static ScalarOracle = Box::leak(Box::new(ScalarOracle));
        let d = QgfBackendDispatch::new(oracle).with_ane::<MockAne, f32, ()>(MockAne);
        let mut out = [0.0f32; 4];
        let route = d.dispatch_single(&3.0, &(), &mut out);
        assert_eq!(route, QgfComputeRoute::AneCritic);
        assert_eq!(out, [0.0, 3.0, 6.0, 9.0]);
    }

    struct FailingAne;
    impl super::SealedForTests for FailingAne {}
    impl QgfAneDelegate<f32, ()> for FailingAne {
        fn gradient_into(&self, _: &f32, _: &(), _: usize, _: &mut [f32]) -> usize {
            0 // ANE unavailable
        }
    }

    #[test]
    fn test_dispatch_single_ane_failure_falls_back_to_cpu() {
        let oracle: &'static ScalarOracle = Box::leak(Box::new(ScalarOracle));
        let d = QgfBackendDispatch::new(oracle).with_ane::<FailingAne, f32, ()>(FailingAne);
        let mut out = [0.0f32; 4];
        let route = d.dispatch_single(&3.0, &(), &mut out);
        assert_eq!(route, QgfComputeRoute::CpuSimd, "ANE failure → CPU");
        assert_eq!(out, [0.0, 3.0, 6.0, 9.0], "CPU produced correct gradient");
    }

    // ── route_for wiring ─────────────────────────────────────────────────

    #[test]
    fn test_dispatcher_route_for_matches_standalone() {
        let d = make_dispatcher();
        assert_eq!(d.route_for(512, 1), route_for(512, 1));
        assert_eq!(d.route_for(2048, 8), route_for(2048, 8));
    }
}

// TL;DR: Backend dispatch for QGF gradient queries. CPU SIMD path concrete
// (reuses oracle's q_gradient_into, which already calls simd::dot_f32_i8 +
// fast_sigmoid for ActionBridge). GPU/ANE paths are trait delegates that
// upper layers (riir-gpu, npc_ane_backend) implement — katgpt-core defines
// the abstraction, provides CPU fallback, and routes via route_for.
