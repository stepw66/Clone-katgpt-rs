//! Plan 264 Phase 4 — Module-Energy Compute Routing (Fusion D).
//!
//! Extracted from `inference_router.rs` (Issue 018) — pure mechanical move,
//! no logic change. All items remain gated by `#[cfg(feature = "module_energy_route")]`
//! and re-exported from `inference_router` to preserve the public API path
//! `katgpt::inference_router::{ComputeTarget, ModuleEnergyProfile, route_by_module_energy}`.
//!
//! Distilled from arXiv 2606.13657 §4.2 Table 3: OPD/RLVR weight deltas are
//! heavily skewed toward FFN layers (62–86% of total delta energy). This
//! module exposes a modelless `route_by_module_energy` that picks the most
//! efficient compute target for the observed module-energy profile and QPS.
//!
//! # Plasma / Hot / Warm / Cold / Freeze tier mapping
//!
//! The router maps each `ComputeTarget` to a Plasma-tier execution mode:
//!
//! | ComputeTarget | Plasma Tier | Use Case                                  |
//! |---------------|-------------|-------------------------------------------|
//! | Plasma        | Plasma      | FFN-heavy, low QPS — ternary hot path     |
//! | Simd          | Hot         | Balanced profile, moderate QPS — SIMD f32 |
//! | Gpu           | Warm        | Attention-heavy, high QPS — batched GPU   |
//! | Ane           | Cold        | Very low QPS — ANE for cold-start savings |
//!
//! (Freeze tier is reserved for `pruners::freeze` — orthogonal to this router.)
//!
//! All additive: existing `InferenceRouter` fields and methods are untouched.
//! The `ModuleEnergyProfile` struct is intentionally standalone — callers feed
//! its fields into `route_by_module_energy` rather than mutating `TriggerGate`.

/// Compute target selected by [`route_by_module_energy`].
///
/// Each variant maps to a distinct execution backend and Plasma tier (see the
/// tier table in the module-level docs above).
#[cfg(feature = "module_energy_route")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ComputeTarget {
    /// Ternary hot path — FFN-dominated workloads at low QPS. Uses the
    /// `plasma_path` ternary kernels for ~3–5× throughput on Apple Silicon.
    Plasma,
    /// CPU SIMD f32 — balanced module profile at moderate QPS. The default
    /// fallback when no specific profile dominates.
    Simd,
    /// Batched GPU forward — attention-dominated workloads at high QPS where
    /// the matmul intensity justifies the GPU dispatch overhead.
    Gpu,
    /// Apple Neural Engine — very low QPS / cold-start, where the ANE's
    /// power efficiency wins over keeping the CPU/GPU warm.
    Ane,
}

/// Per-module energy profile of a weight delta or active adapter.
///
/// Each field is the fraction of total delta Frobenius energy attributed to
/// that module family. The four fields should sum to 1.0 (callers are
/// responsible for normalization — see [`ModuleEnergyProfile::is_valid`]).
///
/// # Paper grounding
///
/// arXiv 2606.13657 §4.2 Table 3 reports the following average profile for
/// OPD-trained adapters:
///
/// | Module | Fraction |
/// |--------|----------|
/// | FFN    | 0.62–0.86 (mean ≈ 0.78) |
/// | Attn   | 0.10–0.30 |
/// | Embed  | 0.02–0.08 |
/// | Other  | < 0.04   |
///
/// The [`route_by_module_energy`] function uses `ffn_frac` and `attn_frac`
/// to pick the most efficient backend for the observed profile.
#[cfg(feature = "module_energy_route")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModuleEnergyProfile {
    /// Feed-forward network (MLP/FFN) energy fraction.
    pub ffn: f32,
    /// Attention (Q/K/V/O) energy fraction.
    pub attn: f32,
    /// Embedding / LM-head energy fraction.
    pub embed: f32,
    /// Other (layernorm, bias, etc.) energy fraction.
    pub other: f32,
}

#[cfg(feature = "module_energy_route")]
impl ModuleEnergyProfile {
    /// Paper-average OPD profile: FFN=0.78, Attn=0.16, Embed=0.04, Other=0.02.
    ///
    /// Use as the default when no per-adapter profile has been measured.
    pub const PAPER_AVERAGE: Self = Self {
        ffn: 0.78,
        attn: 0.16,
        embed: 0.04,
        other: 0.02,
    };

    /// Sum of all four fractions. Should be ≈ 1.0 for a well-formed profile.
    #[inline]
    pub fn total(&self) -> f32 {
        self.ffn + self.attn + self.embed + self.other
    }

    /// Returns `true` iff the profile sums to ≈ 1.0 (within `1e-3`).
    #[inline]
    pub fn is_valid(&self) -> bool {
        (self.total() - 1.0).abs() < 1e-3
    }
}

/// Pick the most efficient [`ComputeTarget`] for the given module profile and QPS.
///
/// Routing rules (paper §4.2 Table 3), equivalently structured as a `match`
/// on QPS band with an inner ffn/attn branch per band:
///
/// 1. **FFN-dominated + low QPS** (`ffn_frac > 0.70 && qps < 1000`) →
///    [`ComputeTarget::Plasma`]. The ternary hot path is ~3–5× faster than
///    dense f32 on FFN matmuls, but only pays off when the batch is small
///    enough to fit the ternary gather kernel's sweet spot.
/// 2. **Attention-dominated + high QPS** (`attn_frac > 0.40 && qps >= 1000`)
///    → [`ComputeTarget::Gpu`]. Attention matmuls are batch-friendly — the
///    GPU dispatch overhead amortizes at QPS ≥ 1000.
/// 3. **Very low QPS** (`qps < 100`) → [`ComputeTarget::Ane`]. At cold-start
///    QPS, the ANE's static power dominates and beats keeping CPU/GPU warm.
/// 4. **Otherwise** → [`ComputeTarget::Simd`]. The safe f32 SIMD fallback.
///
/// The rule order is significant: rule 1 is checked before rule 3, so an
/// FFN-heavy adapter at very low QPS still routes to Plasma (where ternary
/// kernels shine) rather than ANE.
///
/// # Paper grounding
///
/// GOAT G7 verifies rule 1 fires at the paper's measured FFN fraction
/// (0.78) and QPS 500. GOAT G8 verifies the routing is monotone in QPS —
/// no flapping back and forth as QPS increases.
#[cfg(feature = "module_energy_route")]
#[inline]
pub fn route_by_module_energy(ffn_frac: f32, attn_frac: f32, qps: u32) -> ComputeTarget {
    // Branch on QPS band first: groups all qps-dependent rules so each path
    // evaluates at most one ffn/attn comparison instead of up to three.
    match qps {
        0..=99 => match ffn_frac > 0.70 {
            // Rule 1 (low QPS half): FFN-heavy → Plasma.
            true => ComputeTarget::Plasma,
            // Rule 3: cold-start QPS → ANE.
            false => ComputeTarget::Ane,
        },
        100..=999 => match ffn_frac > 0.70 {
            // Rule 1 (mid QPS half): FFN-heavy → Plasma.
            true => ComputeTarget::Plasma,
            // Rule 4: neither rule applies → Simd.
            false => ComputeTarget::Simd,
        },
        // qps >= 1000: rule 2 applies, else Simd.
        _ => match attn_frac > 0.40 {
            true => ComputeTarget::Gpu,
            false => ComputeTarget::Simd,
        },
    }
}

#[cfg(all(test, feature = "module_energy_route"))]
mod module_energy_route_tests {
    use super::*;

    #[test]
    fn g7_route_matches_paper_ffn_profile() {
        // GOAT G7: FFN frac = 0.78 (paper average), qps = 500 → Plasma.
        let target = route_by_module_energy(0.78, 0.16, 500);
        assert_eq!(
            target,
            ComputeTarget::Plasma,
            "GOAT G7 FAIL: expected Plasma for FFN=0.78 qps=500, got {target:?}"
        );
    }

    #[test]
    fn g8_route_monotone_in_qps() {
        // GOAT G8: route transitions are monotone in QPS — no flapping.
        //
        // We sweep qps from 10 to 10000 with the paper-average profile and
        // verify the ordinal position of ComputeTarget is non-decreasing
        // along the (arbitrary but fixed) ordering Plasma < Ane < Simd < Gpu.
        //
        // "Monotone" here means: as QPS increases, the route never goes back
        // to an earlier tier in this ordering. This catches flapping where,
        // e.g., the router flips Plasma → Ane → Plasma as QPS rises.
        //
        // The ordering is chosen so the typical low→high QPS progression
        // (Plasma at low, Simd at mid, Gpu at high) is monotone. ANE is
        // placed between Plasma and Simd because it fires at very low QPS
        // when the FFN-Plasma rule does not match.
        fn ordinal(t: ComputeTarget) -> u8 {
            match t {
                ComputeTarget::Plasma => 0,
                ComputeTarget::Ane => 1,
                ComputeTarget::Simd => 2,
                ComputeTarget::Gpu => 3,
            }
        }

        // Paper-average profile: FFN=0.78 (so rule 1 fires whenever qps<1000).
        let ffn = 0.78;
        let attn = 0.16;

        let mut prev_ord = 0_u8;
        let mut transitions = Vec::new();
        for qps_log in 0..=400 {
            let qps = (10.0_f32 * 10.0_f32.powf(qps_log as f32 / 100.0)) as u32;
            let target = route_by_module_energy(ffn, attn, qps);
            let ord = ordinal(target);
            assert!(
                ord >= prev_ord,
                "GOAT G8 FAIL: non-monotone at qps={qps} — ord {ord} < prev {prev_ord} (target={target:?})"
            );
            if ord != prev_ord {
                transitions.push((qps, target));
                prev_ord = ord;
            }
        }
        // Sanity: we should have seen at least one transition (Plasma → Simd
        // around qps=1000 given the paper profile).
        assert!(
            !transitions.is_empty(),
            "GOAT G8: expected at least one QPS-driven transition, saw none"
        );
    }

    // ── Unit tests for individual routing rules ─────────────────────────

    #[test]
    fn route_ffn_heavy_low_qps_is_plasma() {
        assert_eq!(route_by_module_energy(0.75, 0.10, 100), ComputeTarget::Plasma);
        assert_eq!(route_by_module_energy(0.85, 0.05, 500), ComputeTarget::Plasma);
        assert_eq!(route_by_module_energy(0.95, 0.02, 999), ComputeTarget::Plasma);
    }

    #[test]
    fn route_attn_heavy_high_qps_is_gpu() {
        assert_eq!(route_by_module_energy(0.30, 0.50, 1000), ComputeTarget::Gpu);
        assert_eq!(route_by_module_energy(0.20, 0.60, 5000), ComputeTarget::Gpu);
        assert_eq!(route_by_module_energy(0.10, 0.80, 10000), ComputeTarget::Gpu);
    }

    #[test]
    fn route_very_low_qps_is_ane() {
        // FFN fraction below the 0.70 threshold so rule 1 doesn't fire.
        assert_eq!(route_by_module_energy(0.50, 0.30, 10), ComputeTarget::Ane);
        assert_eq!(route_by_module_energy(0.60, 0.20, 50), ComputeTarget::Ane);
        assert_eq!(route_by_module_energy(0.10, 0.10, 99), ComputeTarget::Ane);
    }

    #[test]
    fn route_default_is_simd() {
        // Neither FFN-dominated nor attention-dominated, moderate QPS.
        assert_eq!(route_by_module_energy(0.50, 0.30, 500), ComputeTarget::Simd);
        assert_eq!(route_by_module_energy(0.40, 0.35, 1500), ComputeTarget::Simd);
        // FFN fraction high but QPS >= 1000 so rule 1 doesn't fire, and
        // attention is below 0.40 so rule 2 doesn't fire either.
        assert_eq!(route_by_module_energy(0.75, 0.15, 1000), ComputeTarget::Simd);
    }

    #[test]
    fn route_boundary_qps_1000() {
        // At exactly qps=1000, rule 1 requires qps < 1000 (false), rule 2
        // requires qps >= 1000 (true). So an attention-heavy profile at qps=1000
        // goes GPU; an FFN-heavy profile at qps=1000 does NOT go Plasma.
        assert_eq!(route_by_module_energy(0.80, 0.10, 1000), ComputeTarget::Simd);
        assert_eq!(route_by_module_energy(0.30, 0.50, 1000), ComputeTarget::Gpu);
        // Just below 1000: FFN-heavy → Plasma.
        assert_eq!(route_by_module_energy(0.80, 0.10, 999), ComputeTarget::Plasma);
    }

    #[test]
    fn route_boundary_qps_100() {
        // At qps=100, rule 3 requires qps < 100 (false).
        // So an FFN-light profile at qps=100 goes Simd, not ANE.
        assert_eq!(route_by_module_energy(0.50, 0.30, 100), ComputeTarget::Simd);
        assert_eq!(route_by_module_energy(0.50, 0.30, 99), ComputeTarget::Ane);
    }

    #[test]
    fn module_energy_profile_paper_average_sums_to_one() {
        let p = ModuleEnergyProfile::PAPER_AVERAGE;
        assert!(p.is_valid(), "PAPER_AVERAGE total = {}", p.total());
    }

    #[test]
    fn module_energy_profile_total() {
        let p = ModuleEnergyProfile {
            ffn: 0.5,
            attn: 0.3,
            embed: 0.15,
            other: 0.05,
        };
        assert!((p.total() - 1.0).abs() < 1e-6);
        assert!(p.is_valid());
    }

    #[test]
    fn module_energy_profile_invalid_when_not_normalized() {
        let p = ModuleEnergyProfile {
            ffn: 0.5,
            attn: 0.3,
            embed: 0.1,
            other: 0.05,
        };
        // total = 0.95, not 1.0
        assert!(!p.is_valid());
    }
}
