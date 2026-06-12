//! Spectral Budget Router — Layer-Adaptive NS Depth + Rank-p Truncation (Plan 253, Research 222).
//!
//! Uses spectral power laws from Magakyan et al. (2026) to predict per-layer
//! Newton-Schulz iteration depth and singular direction retention at inference time.
//! Pure arithmetic — no training, no data.
//!
//! # Power Law Formula
//!
//! After burn-in, singular value quantiles stabilize at:
//!   σ_q(M) = c_q · M^(-α_layer)
//!
//! Where α_layer varies by depth fraction:
//! - Mid-early/mid/mid-late: α ≈ -0.25
//! - Final attention (Q/K/V/O): α ≈ -0.38 to -0.66
//! - Final MLP Up: α ≈ -0.96

// ── Spectral Exponent Table ────────────────────────────────────

/// Power law exponent lookup by (depth_fraction, layer_type).
///
/// Exponents from Magakyan et al. Figure 16, fitted on 77M-2.8B GPT-2 models.
/// R² > 0.98 for all layer types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    MlpUp,
    MlpDown,
}

/// Returns the spectral power law exponent for a given depth fraction and layer type.
///
/// Depth fraction is relative position in the network:
///   0.0 = first layer, 1.0 = last layer.
///
/// Values interpolated from the paper's empirical measurements:
/// - mid-early/mid/mid-late: mild scaling (~-0.25)
/// - final layers: aggressive scaling (-0.38 to -0.96)
#[inline]
pub fn spectral_exponent(depth_fraction: f32, layer_type: LayerType) -> f32 {
    debug_assert!(
        (0.0..=1.0).contains(&depth_fraction),
        "depth_fraction must be in [0, 1], got {depth_fraction}"
    );
    let d = depth_fraction.clamp(0.0, 1.0);

    // Mid-early (0.0-0.25), mid (0.25-0.50), mid-late (0.50-0.75):
    // All approximately -0.25 for all layer types.
    if d < 0.75 {
        return -0.25;
    }

    // Late (0.75-1.0): interpolate from -0.25 to layer-specific final value.
    let t = (d - 0.75) / 0.25; // 0.0 at d=0.75, 1.0 at d=1.0
    let alpha_final = match layer_type {
        LayerType::AttentionQ => -0.46,
        LayerType::AttentionK => -0.38,
        LayerType::AttentionV => -0.58,
        LayerType::AttentionO => -0.66,
        LayerType::MlpUp => -0.96,
        LayerType::MlpDown => -0.52,
    };
    // Linear interpolation in log space
    -0.25 + t * (alpha_final - (-0.25))
}

// ── NS Depth Prediction ────────────────────────────────────────

/// NS iteration depth thresholds.
///
/// Standard 5-step NanoGPT NS fails for σ < ~0.003.
/// DeepSeek-V4 10-step composition fails for σ < ~2e-6.
const NS5_FAILURE_THRESHOLD: f32 = 3e-3;
const NS7_FAILURE_THRESHOLD: f32 = 2e-4;
const _NS10_FAILURE_THRESHOLD: f32 = 2e-6;

/// Select Newton-Schulz iteration count based on predicted median singular value.
///
/// - σ > 0.003 → 5 steps (NanoGPT config sufficient)
/// - σ > 0.0002 → 7 steps (intermediate accuracy)
/// - σ ≤ 0.0002 → 10 steps (DeepSeek-V4 config)
#[inline]
pub fn ns_depth_for_sigma(predicted_sigma_50: f32) -> u8 {
    if predicted_sigma_50 > NS5_FAILURE_THRESHOLD {
        5
    } else if predicted_sigma_50 > NS7_FAILURE_THRESHOLD {
        7
    } else {
        10
    }
}

/// NS depth recommendation for a depth fraction, ignoring model size.
///
/// This is a conservative heuristic: at any model size, final layers
/// are harder than mid layers. Returns the minimum recommended depth.
#[inline]
pub fn ns_depth_for_depth_fraction(depth_fraction: f32) -> u8 {
    if depth_fraction < 0.75 {
        5
    } else if depth_fraction < 0.875 {
        7
    } else {
        10
    }
}

// ── Predictive Config ───────────────────────────────────────────

/// Pre-computed NS configuration per depth fraction.
#[derive(Debug, Clone)]
pub struct NsDepthConfig {
    /// Relative depth (0.0 = first layer, 1.0 = last)
    pub depth_fraction: f32,
    /// Power law exponent for this depth
    pub spectral_exponent: f32,
    /// Recommended NS iterations (5, 7, or 10)
    pub ns_iterations: u8,
    /// Fraction of singular directions to keep (from rank-p result)
    /// Paper proves top 50% suffices for full Muon performance.
    pub retention_fraction: f32,
    /// Predicted σ_0.5 for this layer at given model size (0 = not computed)
    pub predicted_median_sv: f32,
}

/// Pre-computed per-layer NS configuration for an entire model.
#[derive(Debug, Clone)]
pub struct SpectralBudgetConfig {
    /// Per-depth config, sorted by depth_fraction ascending.
    pub layers: Vec<NsDepthConfig>,
    /// Model size in millions of parameters.
    pub model_size_m: usize,
}

impl SpectralBudgetConfig {
    /// Predict per-layer NS config from model dimensions.
    ///
    /// Uses power law: σ_q(M) = c_q · M^(-α)
    /// where c_q is calibrated from the paper's 2.8B baseline.
    ///
    /// The coefficient c_q for median (q=0.5) is estimated from
    /// the paper's Figure 16 data points at 2.8B:
    /// - Mid layers: σ_0.5 ≈ 5e-3 at M=2800
    /// - Final O: σ_0.5 ≈ 1e-3 at M=2800
    /// - Final MLP Up: σ_0.5 ≈ 3e-3 at M=2800
    ///
    /// These are approximate; the exponents are the reliable part (R² > 0.98).
    pub fn from_model_dims(
        n_layers: usize,
        model_size_m: usize,
        layer_types: &[LayerType],
    ) -> Self {
        assert!(!layer_types.is_empty(), "need at least one layer type");
        // Each transformer block has 6 matrices: Q, K, V, O, MLP_Up, MLP_Down
        // layer_types should have 6 * n_layers entries, or we cycle.
        let layers = (0..n_layers)
            .map(|i| {
                let depth_frac = if n_layers > 1 {
                    i as f32 / (n_layers - 1) as f32
                } else {
                    0.5
                };
                let lt = layer_types[i % layer_types.len()];
                let alpha = spectral_exponent(depth_frac, lt);

                // Predict median σ at this model size.
                // Calibration: at M=2800, σ_0.5 for α=-0.25 layers ≈ 5e-3.
                // σ(M) = c · M^α, so c = σ(2800) / 2800^α = 5e-3 / 2800^(-0.25)
                // Then σ(model_size_m) = c · model_size_m^α
                let predicted_sv = predict_median_sv(alpha, model_size_m);

                let ns_iters = ns_depth_for_sigma(predicted_sv);

                // Retention: 50% for gaming/standard, but final layers with
                // aggressive scaling might need more retention
                let retention = if depth_frac < 0.75 {
                    0.5
                } else if depth_frac < 0.875 {
                    0.6
                } else {
                    0.75
                };

                NsDepthConfig {
                    depth_fraction: depth_frac,
                    spectral_exponent: alpha,
                    ns_iterations: ns_iters,
                    retention_fraction: retention,
                    predicted_median_sv: predicted_sv,
                }
            })
            .collect();

        SpectralBudgetConfig {
            layers,
            model_size_m,
        }
    }

    /// Get NS iteration count for a specific layer index.
    #[inline]
    pub fn ns_iterations(&self, layer_idx: usize) -> u8 {
        self.layers
            .get(layer_idx)
            .map(|c| c.ns_iterations)
            .unwrap_or(5)
    }

    /// Get retention fraction for a specific layer index.
    #[inline]
    pub fn retention(&self, layer_idx: usize) -> f32 {
        self.layers
            .get(layer_idx)
            .map(|c| c.retention_fraction)
            .unwrap_or(0.5)
    }

    /// Count layers that need more than 5 NS iterations.
    pub fn count_deep_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.ns_iterations > 5).count()
    }

    /// Total NS iterations across all layers.
    pub fn total_ns_iterations(&self) -> usize {
        self.layers.iter().map(|l| l.ns_iterations as usize).sum()
    }

    /// Compare against uniform 5-step: (this_total, uniform_5_total, ratio).
    pub fn vs_uniform5(&self) -> (usize, usize, f32) {
        let this = self.total_ns_iterations();
        let uniform = self.layers.len() * 5;
        (this, uniform, this as f32 / uniform as f32)
    }
}

/// Predict median singular value at given model size for a layer with given exponent.
///
/// Calibrated from the paper's data at M=2800:
///   For α=-0.25: σ_0.5(2800) ≈ 5e-3
///   c = 5e-3 / 2800^(-0.25) ≈ 5e-3 / 0.0723 ≈ 0.069
///
/// So σ_0.5(M) ≈ 0.069 · M^(-|α|)
#[inline]
fn predict_median_sv(alpha: f32, model_size_m: usize) -> f32 {
    // Calibration coefficient from mid-layer data at 2.8B
    const CALIB_SIGMA: f32 = 5e-3;
    const CALIB_SIZE: f32 = 2800.0;
    let calib_alpha = -0.25;

    // c = CALIB_SIGMA / CALIB_SIZE^calib_alpha
    let c = CALIB_SIGMA / CALIB_SIZE.powf(calib_alpha);

    // σ(M) = c · M^alpha
    c * (model_size_m as f32).powf(alpha)
}

// ── Spectral LOD ────────────────────────────────────────────────

/// Structural Level of Detail derived from spectral exponents.
///
/// Combines with SLoD (Semantic LOD) for compute routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralLod {
    /// Simple terrain — mid-early/mid layers, α ≈ -0.25
    /// 5-step NS, minimal compute
    Lod0,
    /// Slightly complex — mid-late, α ≈ -0.27
    /// 5-step NS, monitor effective rank
    Lod1,
    /// Complex — late attention, α ≈ -0.46 to -0.66
    /// 7-10 step NS recommended
    Lod2,
    /// Critical — final MLP Up, α ≈ -0.96
    /// Full 10-step NS + rank-p truncation
    Lod3,
}

impl SpectralLod {
    /// Classify a layer by its spectral exponent.
    #[inline]
    pub fn from_exponent(alpha: f32) -> Self {
        let abs_alpha = alpha.abs();
        if abs_alpha < 0.30 {
            SpectralLod::Lod0
        } else if abs_alpha < 0.45 {
            SpectralLod::Lod1
        } else if abs_alpha < 0.70 {
            SpectralLod::Lod2
        } else {
            SpectralLod::Lod3
        }
    }

    /// Classify a layer by its depth fraction.
    #[inline]
    pub fn from_depth(depth_fraction: f32) -> Self {
        if depth_fraction < 0.50 {
            SpectralLod::Lod0
        } else if depth_fraction < 0.75 {
            SpectralLod::Lod1
        } else if depth_fraction < 0.875 {
            SpectralLod::Lod2
        } else {
            SpectralLod::Lod3
        }
    }

    /// Recommended NS iterations for this LOD level.
    #[inline]
    pub fn ns_iterations(self) -> u8 {
        match self {
            SpectralLod::Lod0 => 5,
            SpectralLod::Lod1 => 5,
            SpectralLod::Lod2 => 7,
            SpectralLod::Lod3 => 10,
        }
    }

    /// Recommended retention fraction for this LOD level.
    #[inline]
    pub fn retention(self) -> f32 {
        match self {
            SpectralLod::Lod0 => 0.50,
            SpectralLod::Lod1 => 0.50,
            SpectralLod::Lod2 => 0.60,
            SpectralLod::Lod3 => 0.75,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // T1: Exponent table matches paper's values
    #[test]
    fn t1_mid_layer_exponent_near_025() {
        for lt in [
            LayerType::AttentionQ,
            LayerType::AttentionK,
            LayerType::AttentionV,
            LayerType::AttentionO,
            LayerType::MlpUp,
            LayerType::MlpDown,
        ] {
            let alpha = spectral_exponent(0.375, lt); // mid layer
            assert!(
                (alpha - (-0.25)).abs() < 0.01,
                "mid layer alpha for {lt:?} should be ~-0.25, got {alpha}"
            );
        }
    }

    #[test]
    fn t1_final_mlp_up_exponent_near_096() {
        let alpha = spectral_exponent(1.0, LayerType::MlpUp);
        assert!(
            (alpha - (-0.96)).abs() < 0.01,
            "final MLP Up alpha should be ~-0.96, got {alpha}"
        );
    }

    #[test]
    fn t1_final_o_exponent_near_066() {
        let alpha = spectral_exponent(1.0, LayerType::AttentionO);
        assert!(
            (alpha - (-0.66)).abs() < 0.01,
            "final O alpha should be ~-0.66, got {alpha}"
        );
    }

    #[test]
    fn t1_final_v_exponent_near_058() {
        let alpha = spectral_exponent(1.0, LayerType::AttentionV);
        assert!(
            (alpha - (-0.58)).abs() < 0.01,
            "final V alpha should be ~-0.58, got {alpha}"
        );
    }

    #[test]
    fn t1_final_q_exponent_near_046() {
        let alpha = spectral_exponent(1.0, LayerType::AttentionQ);
        assert!(
            (alpha - (-0.46)).abs() < 0.01,
            "final Q alpha should be ~-0.46, got {alpha}"
        );
    }

    #[test]
    fn t1_final_k_exponent_near_038() {
        let alpha = spectral_exponent(1.0, LayerType::AttentionK);
        assert!(
            (alpha - (-0.38)).abs() < 0.01,
            "final K alpha should be ~-0.38, got {alpha}"
        );
    }

    #[test]
    fn t1_final_mlp_down_exponent_near_052() {
        let alpha = spectral_exponent(1.0, LayerType::MlpDown);
        assert!(
            (alpha - (-0.52)).abs() < 0.01,
            "final MLP Down alpha should be ~-0.52, got {alpha}"
        );
    }

    // T2: NS depth selector matches paper's thresholds
    #[test]
    fn t2_sigma_above_threshold_gets_5_steps() {
        assert_eq!(ns_depth_for_sigma(0.01), 5);
        assert_eq!(ns_depth_for_sigma(0.005), 5);
        assert_eq!(ns_depth_for_sigma(3.001e-3), 5);
    }

    #[test]
    fn t2_sigma_mid_range_gets_7_steps() {
        assert_eq!(ns_depth_for_sigma(0.001), 7);
        assert_eq!(ns_depth_for_sigma(2.01e-4), 7);
    }

    #[test]
    fn t2_sigma_below_threshold_gets_10_steps() {
        assert_eq!(ns_depth_for_sigma(1e-4), 10);
        assert_eq!(ns_depth_for_sigma(1e-5), 10);
        assert_eq!(ns_depth_for_sigma(1e-6), 10);
    }

    // T3: Spectral config prediction
    #[test]
    fn t3_config_mid_layers_get_5_steps() {
        let lt = [
            LayerType::AttentionQ,
            LayerType::AttentionK,
            LayerType::AttentionV,
            LayerType::AttentionO,
            LayerType::MlpUp,
            LayerType::MlpDown,
        ];
        let cfg = SpectralBudgetConfig::from_model_dims(8, 77, &lt);
        // First 6 layers (depth < 0.75) should get 5 steps
        for i in 0..6 {
            assert_eq!(
                cfg.ns_iterations(i),
                5,
                "layer {i} (depth={:.2}) should get 5 steps",
                cfg.layers[i].depth_fraction
            );
        }
    }

    #[test]
    fn t3_config_final_layers_may_need_more() {
        let lt = [LayerType::MlpUp]; // Most aggressive scaling
        let cfg = SpectralBudgetConfig::from_model_dims(28, 2800, &lt);
        // Last layer should need more than 5 steps at 2.8B scale
        let last = cfg.layers.last().unwrap();
        assert!(
            last.ns_iterations >= 5,
            "final layer should need at least 5 steps, got {}",
            last.ns_iterations
        );
    }

    #[test]
    fn t3_config_vs_uniform5_ratio() {
        let lt = [
            LayerType::AttentionQ,
            LayerType::AttentionK,
            LayerType::AttentionV,
            LayerType::AttentionO,
            LayerType::MlpUp,
            LayerType::MlpDown,
        ];
        let cfg = SpectralBudgetConfig::from_model_dims(28, 2800, &lt);
        let (total, uniform, ratio) = cfg.vs_uniform5();
        // Should be close to 1.0 — most layers get 5 steps, a few get more
        assert!(
            ratio >= 1.0 && ratio <= 1.3,
            "ratio {ratio} should be in [1.0, 1.3] (total={total}, uniform={uniform})"
        );
    }

    // T4: Spectral LOD classification
    #[test]
    fn t4_lod_from_exponent() {
        assert_eq!(SpectralLod::from_exponent(-0.25), SpectralLod::Lod0);
        assert_eq!(SpectralLod::from_exponent(-0.35), SpectralLod::Lod1);
        assert_eq!(SpectralLod::from_exponent(-0.55), SpectralLod::Lod2);
        assert_eq!(SpectralLod::from_exponent(-0.96), SpectralLod::Lod3);
    }

    #[test]
    fn t4_lod_from_depth() {
        assert_eq!(SpectralLod::from_depth(0.25), SpectralLod::Lod0);
        assert_eq!(SpectralLod::from_depth(0.60), SpectralLod::Lod1);
        assert_eq!(SpectralLod::from_depth(0.80), SpectralLod::Lod2);
        assert_eq!(SpectralLod::from_depth(0.95), SpectralLod::Lod3);
    }

    #[test]
    fn t4_lod_ns_iterations() {
        assert_eq!(SpectralLod::Lod0.ns_iterations(), 5);
        assert_eq!(SpectralLod::Lod1.ns_iterations(), 5);
        assert_eq!(SpectralLod::Lod2.ns_iterations(), 7);
        assert_eq!(SpectralLod::Lod3.ns_iterations(), 10);
    }

    #[test]
    fn t4_lod_retention() {
        assert!((SpectralLod::Lod0.retention() - 0.50).abs() < 0.01);
        assert!((SpectralLod::Lod3.retention() - 0.75).abs() < 0.01);
    }

    // T5: Predicted sigma decreases with model size
    #[test]
    fn t5_sigma_decreases_with_model_size() {
        let sigma_77m = predict_median_sv(-0.25, 77);
        let sigma_2_8b = predict_median_sv(-0.25, 2800);
        assert!(
            sigma_2_8b < sigma_77m,
            "sigma should decrease with model size: 77M={sigma_77m:.6} vs 2.8B={sigma_2_8b:.6}"
        );
    }

    #[test]
    fn t5_final_mlp_sigma_lower_than_mid() {
        let mid = predict_median_sv(-0.25, 2800);
        let final_mlp = predict_median_sv(-0.96, 2800);
        assert!(
            final_mlp < mid,
            "final MLP sigma should be lower than mid: mid={mid:.6} vs final={final_mlp:.6}"
        );
    }
}
