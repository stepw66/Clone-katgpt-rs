//! [`AttributionProbe`] — finite-difference sensitivity surrogate for
//! Integrated Gradients.
//!
//! Modelless: zero backprop. Uses central differences in an ε-ball around the
//! memory to approximate ‖∇_M C(x; M)‖. Based on paper App D.7, Research 244
//! §2.2. The paper's ablation shows embedding-gradient L2 norm correlates
//! strongly with attention-level IG, so this surrogate is a valid ranking
//! signal without requiring a gradient graph.

use super::types::{ConsumerContext, MemorySlice};

/// Finite-difference sensitivity surrogate for Integrated Gradients.
///
/// Reports ‖∇_M C(x; M)‖ approximated by central differences in an ε-ball.
/// Zero backprop — just forward-pass probing.
///
/// Based on Research 244 §2.2 / paper App D.7.
pub trait AttributionProbe {
    /// Memory representation under attribution.
    type Memory;

    /// Compute the L2 norm of the finite-difference gradient of the consumer's
    /// behavior with respect to the memory, at perturbation scale `epsilon`.
    ///
    /// Per-element central difference: `(f(M + ε·e_i) − f(M − ε·e_i)) / (2ε)`,
    /// then L2-norm the resulting gradient vector.
    fn attribution_norm(&mut self, memory: &Self::Memory, epsilon: f32) -> f32;
}

/// Central-difference attribution probe.
///
/// Computes `(f(M + εδ) − f(M − εδ)) / (2ε)` per memory axis, then L2-norms
/// the gradient vector. Two forward passes per element (one `+ε`, one `−ε`).
///
/// Constraints: `C::Behavior = f32` and `C::Memory: MemorySlice<Elem = f32>`
/// so arithmetic on behavior values is well-defined. For non-scalar behaviors,
/// implement [`AttributionProbe`] directly on your probe type.
///
/// Reuses two scratch clones (`plus`, `minus`) across all axes — O(1) extra
/// allocations per `attribution_norm` call regardless of memory length.
pub struct FiniteDifferenceAttributionProbe<C>
where
    C: ConsumerContext,
{
    pub consumer: C,
}

impl<C> FiniteDifferenceAttributionProbe<C>
where
    C: ConsumerContext,
{
    /// Create a probe wrapping the given consumer.
    pub fn new(consumer: C) -> Self {
        Self { consumer }
    }
}

impl<C> AttributionProbe for FiniteDifferenceAttributionProbe<C>
where
    C: ConsumerContext<Behavior = f32>,
    C::Memory: MemorySlice<Elem = f32> + Clone,
{
    type Memory = C::Memory;

    fn attribution_norm(&mut self, memory: &Self::Memory, epsilon: f32) -> f32 {
        let orig = memory.mem_as_slice();
        let n = orig.len();
        if n == 0 || epsilon == 0.0 {
            return 0.0;
        }

        // Two scratch clones, reused across all axes (O(1) allocations).
        let mut plus = memory.clone();
        let mut minus = memory.clone();
        let inv_2eps = 1.0 / (2.0 * epsilon);

        let mut l2_sq = 0.0_f32;

        for i in 0..n {
            let orig_i = orig[i];

            // f(M + ε·e_i)
            {
                let s = plus.mem_as_mut_slice();
                s[i] = orig_i + epsilon;
            }
            let f_plus = self.consumer.behavior_with_memory(&plus);

            // f(M - ε·e_i)
            {
                let s = minus.mem_as_mut_slice();
                s[i] = orig_i - epsilon;
            }
            let f_minus = self.consumer.behavior_with_memory(&minus);

            // Restore both scratches to original for next axis.
            {
                let s = plus.mem_as_mut_slice();
                s[i] = orig_i;
                let s = minus.mem_as_mut_slice();
                s[i] = orig_i;
            }

            // Central difference (signed — not behavior_delta which is |·|).
            let grad_i = (f_plus - f_minus) * inv_2eps;
            l2_sq += grad_i * grad_i;
        }

        l2_sq.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Unit tests — Plan 278 T2.3 (G2 reference IG consistency, simplified)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faithfulness::types::ConsumerContext;

    /// Linear consumer: behavior = dot(memory, weights). The exact gradient is
    /// the weight vector itself, so attribution_norm should ≈ ‖weights‖₂.
    struct LinearConsumer {
        weights: Vec<f32>,
    }

    impl ConsumerContext for LinearConsumer {
        type Behavior = f32;
        type Delta = f32;
        type Memory = Vec<f32>;

        fn baseline_behavior(&self) -> f32 {
            0.0
        }

        fn behavior_with_memory(&self, memory: &Vec<f32>) -> f32 {
            memory
                .iter()
                .zip(self.weights.iter())
                .map(|(&v, &w)| v * w)
                .sum()
        }

        fn behavior_delta(&self, a: &f32, b: &f32) -> f32 {
            (a - b).abs()
        }
    }

    #[test]
    fn test_attribution_norm_matches_exact_gradient_for_linear_consumer() {
        let weights = vec![3.0_f32, 4.0]; // ‖weights‖₂ = 5.0
        let consumer = LinearConsumer { weights };
        let mut probe = FiniteDifferenceAttributionProbe::new(consumer);

        let memory = vec![1.0_f32, 1.0]; // value doesn't matter for linear gradient
        let norm = probe.attribution_norm(&memory, 1e-3);

        // Exact gradient is weights = [3, 4], L2 norm = 5.0.
        assert!(
            (norm - 5.0).abs() < 1e-3,
            "attribution_norm should be ~5.0 (‖[3,4]‖₂), got {}",
            norm
        );
    }

    #[test]
    fn test_attribution_norm_zero_for_empty_memory() {
        let consumer = LinearConsumer {
            weights: vec![1.0, 2.0],
        };
        let mut probe = FiniteDifferenceAttributionProbe::new(consumer);
        let memory: Vec<f32> = vec![];
        assert_eq!(probe.attribution_norm(&memory, 1e-3), 0.0);
    }

    #[test]
    fn test_attribution_norm_zero_for_zero_epsilon() {
        let consumer = LinearConsumer {
            weights: vec![1.0, 2.0],
        };
        let mut probe = FiniteDifferenceAttributionProbe::new(consumer);
        let memory = vec![1.0_f32, 2.0];
        assert_eq!(probe.attribution_norm(&memory, 0.0), 0.0);
    }

    #[test]
    fn test_attribution_ranks_segments_consistently() {
        // G2: attribution should rank segments consistently with exact IG.
        // Segment A has larger weights (higher gradient norm) than segment B.
        let weights_a = vec![5.0_f32, 5.0]; // ‖·‖₂ ≈ 7.07
        let weights_b = vec![1.0_f32, 1.0]; // ‖·‖₂ ≈ 1.41

        let mem_a = vec![1.0_f32, 1.0];
        let mem_b = vec![1.0_f32, 1.0];

        let mut probe_a = FiniteDifferenceAttributionProbe::new(LinearConsumer {
            weights: weights_a,
        });
        let mut probe_b = FiniteDifferenceAttributionProbe::new(LinearConsumer {
            weights: weights_b,
        });

        let norm_a = probe_a.attribution_norm(&mem_a, 1e-4);
        let norm_b = probe_b.attribution_norm(&mem_b, 1e-4);

        // Ranking: A > B (consistent with exact gradient norms).
        assert!(
            norm_a > norm_b,
            "segment A (‖[5,5]‖₂≈7.07) should rank higher than B (‖[1,1]‖₂≈1.41): {} vs {}",
            norm_a,
            norm_b
        );
    }
}
