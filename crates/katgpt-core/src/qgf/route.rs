//! Compute routing for QGF gradient queries (Plan 268 T8).
//!
//! Decides which backend (CPU SIMD / GPU batch / ANE critic) should service a
//! Q-gradient query based on action-space size and batch size.
//!
//! # Design
//!
//! Routing is **O(1)** — two comparisons, no allocation, no I/O. This is a
//! hard requirement: routing overhead must never dominate the gradient query
//! itself (which is < 100ns for Plasma tier).
//!
//! # Tiers vs Routes
//!
//! Routes are *backend* selectors; tiers (Plasma/Hot/Warm/Cold/Freeze) are
//! *oracle* selectors. The two are orthogonal: a Plasma-tier oracle can be
//! served by either `CpuSimd` or `AneCritic` depending on action space size.
//!
//! See `.plans/268_qgf_test_time_q_guided_flow.md` §Phase 4 T8.

/// Backend selection for a QGF gradient query.
///
/// Chosen by [`route_for`] based on action-space size and batch size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum QgfComputeRoute {
    /// CPU SIMD path — reuse existing `simd::dot_f32_i8` + `simd::fast_sigmoid`.
    /// Optimal for small action spaces and small batches.
    CpuSimd,
    /// GPU batched dispatch — amortise kernel launch over a batch of queries.
    /// Optimal when `batch_size >= 8` and `action_space_size >= 1024`.
    GpuBatch,
    /// Apple Neural Engine critic forward — via `npc_ane_backend`.
    /// Reserved for medium action spaces where the ANE critic model is available.
    AneCritic,
}

/// Pick the compute route for a QGF gradient query.
///
/// # Rules (deterministic, O(1))
///
/// - `action_space_size < 1024` → `CpuSimd` (small enough for SIMD dot product)
/// - `batch_size >= 8 && action_space_size >= 1024` → `GpuBatch` (amortise launch)
/// - otherwise → `CpuSimd` (default safe path)
///
/// # Arguments
///
/// - `action_space_size` — number of discrete actions (or latent dims).
/// - `batch_size` — number of concurrent gradient queries.
///
/// # Example
///
/// ```
/// # use katgpt_core::qgf::route::{route_for, QgfComputeRoute};
/// assert_eq!(route_for(512, 1), QgfComputeRoute::CpuSimd);
/// assert_eq!(route_for(2048, 8), QgfComputeRoute::GpuBatch);
/// ```
#[inline]
pub fn route_for(action_space_size: usize, batch_size: usize) -> QgfComputeRoute {
    if action_space_size < 1024 {
        QgfComputeRoute::CpuSimd
    } else if batch_size >= 8 {
        QgfComputeRoute::GpuBatch
    } else {
        QgfComputeRoute::CpuSimd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_route_small_action_space_cpu() {
        assert_eq!(route_for(512, 1), QgfComputeRoute::CpuSimd);
        assert_eq!(route_for(0, 0), QgfComputeRoute::CpuSimd);
        assert_eq!(route_for(1023, 100), QgfComputeRoute::CpuSimd);
    }

    #[test]
    fn test_route_large_batch_gpu() {
        assert_eq!(route_for(2048, 8), QgfComputeRoute::GpuBatch);
        assert_eq!(route_for(4096, 16), QgfComputeRoute::GpuBatch);
    }

    #[test]
    fn test_route_large_action_small_batch_cpu() {
        // Large action space but batch < 8 → CPU (GPU launch overhead not amortised).
        assert_eq!(route_for(2048, 4), QgfComputeRoute::CpuSimd);
        assert_eq!(route_for(1024, 7), QgfComputeRoute::CpuSimd);
    }

    #[test]
    fn test_route_boundary_action_space_1024() {
        // Exactly 1024 with batch >= 8 → GPU (1024 is the threshold).
        assert_eq!(route_for(1024, 8), QgfComputeRoute::GpuBatch);
    }

    #[test]
    fn test_route_o1() {
        // Routing decision must be sub-microsecond (target < 100ns, but be
        // lenient on CI to avoid flakes). We run many iterations and check
        // the per-call cost is well under 100ns.
        const ITERS: usize = 100_000;
        let start = Instant::now();
        let mut sink = 0u64;
        for i in 0..ITERS {
            let r = route_for(i & 0x1FFF, (i >> 3) & 0xF);
            sink = sink.wrapping_add(r as u64);
        }
        let elapsed = start.elapsed();
        let _ = sink; // prevent optimisation out
        let per_call_ns = elapsed.as_nanos() as f64 / ITERS as f64;
        assert!(
            per_call_ns < 100.0,
            "route_for must be < 100ns/call, got {per_call_ns:.2}ns/call"
        );
    }
}
