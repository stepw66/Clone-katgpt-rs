//! Compute router — picks CPU / GPU / ANE per layer based on topology width.
//!
//! Per optimisation.md: GPU launch overhead is ~50us — only worth it when
//! width >= 4 (amortise across parallel branches). ANE wins on fixed-shape
//! final decode (per Research 155 / 223).

use super::types::{ComputeTarget, LayerRole};

/// Decide where a layer's forward pass should run.
///
/// - Width 1 hidden -> CPU (no GPU launch overhead amortisation).
/// - Width >= threshold hidden -> GPU (data-parallel branches).
/// - Output layer -> ANE (final decode, latency-sensitive).
///
/// `gpu_threshold` default 4 per optimisation.md (~50us GPU launch cost).
/// Threshold-governed switching satisfies constraint 9.
pub fn pick_compute(width: usize, role: LayerRole, gpu_threshold: usize, prefer_ane: bool) -> ComputeTarget {
    if role == LayerRole::Output && prefer_ane {
        return ComputeTarget::Ane;
    }
    if width >= gpu_threshold {
        ComputeTarget::Gpu
    } else {
        ComputeTarget::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_width_1_hidden_is_cpu() {
        assert_eq!(pick_compute(1, LayerRole::Hidden, 4, true), ComputeTarget::Cpu);
    }

    #[test]
    fn test_width_4_hidden_is_gpu() {
        assert_eq!(pick_compute(4, LayerRole::Hidden, 4, true), ComputeTarget::Gpu);
    }

    #[test]
    fn test_output_layer_is_ane_when_preferred() {
        assert_eq!(pick_compute(1, LayerRole::Output, 4, true), ComputeTarget::Ane);
    }

    #[test]
    fn test_output_layer_is_cpu_when_ane_disabled() {
        assert_eq!(pick_compute(1, LayerRole::Output, 4, false), ComputeTarget::Cpu);
    }

    #[test]
    fn test_input_layer_width_1_is_cpu() {
        assert_eq!(pick_compute(1, LayerRole::Input, 4, true), ComputeTarget::Cpu);
    }

    #[test]
    fn test_custom_threshold_respected() {
        // If threshold is 2, width-2 hidden goes GPU.
        assert_eq!(pick_compute(2, LayerRole::Hidden, 2, true), ComputeTarget::Gpu);
    }
}
