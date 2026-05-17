//! Core types for SP-KV: Self-Pruned Key-Value Attention.
//!
//! Based on "Self-Pruned Key-Value Attention: Learning When to Write by Predicting Future Utility"
//! (arXiv:2605.14037, Meta FAIR). Learns which KV pairs to retain by predicting future utility.

/// Gate mode: controls how utility predictions map to attention bias.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpKvGateMode {
    /// Soft gating: bias = log(u + ε), differentiable for training.
    Soft,
    /// Hard gating: bias = 0 if u ≥ τ, else -∞. For inference.
    Hard,
    /// TAHG annealing: blend soft and hard over `tahg_anneal_steps`.
    /// ũ = (1-α)·u + α·1[u≥τ], α ramps 0→1.
    Tahg { step: usize, total_steps: usize },
}

/// Configuration for SP-KV self-pruned attention.
///
/// Controls utility predictor architecture, gating behavior, and local window size.
/// Feature-gated behind `sp_kv` in `Cargo.toml`.
#[derive(Debug, Clone)]
pub struct SpKvConfig {
    /// Local sliding window always retained (default: 128).
    /// Positions within `window` of the current token are never gated out.
    pub window: usize,
    /// Gate threshold τ for hard gating at inference (default: 0.5).
    /// Higher τ = more aggressive sparsity (fewer KV retained).
    pub threshold: f32,
    /// Utility predictor hidden dimension (default: d_model / 4).
    pub predictor_hidden: usize,
    /// Utility predictor learning rate multiplier (default: 5.0).
    /// Paper ablation: 1× → more density, 0.1× → 82% density (barely sparsifies).
    pub predictor_lr_mult: f32,
    /// Initial bias for utility predictor (default: 5.0).
    /// σ(5) ≈ 0.993 → gates start nearly fully open.
    pub predictor_init_bias: f32,
    /// TAHG annealing steps (default: 500).
    pub tahg_anneal_steps: usize,
    /// TAHG starts at this fraction of training (default: 0.75).
    pub tahg_start_fraction: f32,
}

impl Default for SpKvConfig {
    fn default() -> Self {
        Self {
            window: 128,
            threshold: 0.5,
            predictor_hidden: 0, // resolved from config.n_embd / 4 at init
            predictor_lr_mult: 5.0,
            predictor_init_bias: 5.0,
            tahg_anneal_steps: 500,
            tahg_start_fraction: 0.75,
        }
    }
}

impl SpKvConfig {
    /// Resolve `predictor_hidden` from model embedding dimension if not set.
    pub fn resolve_hidden(&mut self, n_embd: usize) {
        if self.predictor_hidden == 0 {
            self.predictor_hidden = n_embd / 4;
        }
        // Ensure at least 16 to avoid degenerate MLP
        self.predictor_hidden = self.predictor_hidden.max(16);
    }

    /// Current gate mode for training at a given step.
    ///
    /// Before `tahg_start_fraction`: soft gating.
    /// After: TAHG annealing over `tahg_anneal_steps`.
    pub fn gate_mode_at_step(&self, step: usize, total_steps: usize) -> SpKvGateMode {
        let tahg_start = ((total_steps as f32) * self.tahg_start_fraction) as usize;
        if step < tahg_start {
            SpKvGateMode::Soft
        } else {
            let anneal_step = step - tahg_start;
            SpKvGateMode::Tahg {
                step: anneal_step.min(self.tahg_anneal_steps),
                total_steps: self.tahg_anneal_steps,
            }
        }
    }

    /// Gate mode for inference (always hard).
    pub fn inference_gate_mode() -> SpKvGateMode {
        SpKvGateMode::Hard
    }
}

/// Per-layer sparse-write KV cache with gate tracking.
///
/// Unlike `KVCache` which unconditionally writes every position,
/// `SpKvLayerCache` conditionally writes based on utility prediction.
/// Non-retained positions stay zeroed.
pub struct SpKvLayerCache {
    /// Standard key cache [block_size, kv_dim]. Sparse — only retained positions filled.
    pub key: Vec<f32>,
    /// Standard value cache [block_size, kv_dim]. Sparse — only retained positions filled.
    pub value: Vec<f32>,
    /// Per-position gate utility scores (training gradient flow).
    /// One value per position, shared across KV heads (max or mean).
    pub utilities: Vec<f32>,
    /// Bitfield: which positions have retained KV entries.
    pub retained: Vec<bool>,
    /// Number of retained positions (for density computation).
    pub retained_count: usize,
}

impl SpKvLayerCache {
    /// Create a new empty sparse KV cache for one layer.
    pub fn new(block_size: usize, kv_dim: usize) -> Self {
        Self {
            key: vec![0.0; block_size * kv_dim],
            value: vec![0.0; block_size * kv_dim],
            utilities: vec![0.0; block_size],
            retained: vec![false; block_size],
            retained_count: 0,
        }
    }

    /// Reset cache to empty state.
    pub fn reset(&mut self) {
        self.key.fill(0.0);
        self.value.fill(0.0);
        self.utilities.fill(0.0);
        self.retained.fill(false);
        self.retained_count = 0;
    }

    /// Density ratio: fraction of positions retained.
    pub fn density(&self, pos: usize) -> f32 {
        if pos == 0 {
            1.0
        } else {
            self.retained_count as f32 / pos as f32
        }
    }

    /// Conditionally write KV pair at position.
    /// Returns true if written (retained), false if skipped (pruned).
    #[allow(clippy::too_many_arguments)]
    pub fn write_gated(
        &mut self,
        k: &[f32],
        v: &[f32],
        utility: f32,
        pos: usize,
        pos_is_in_window: bool,
        threshold: f32,
        kv_dim: usize,
    ) -> bool {
        let retain = pos_is_in_window || utility >= threshold;
        self.utilities[pos] = utility;

        if retain {
            let off = pos * kv_dim;
            self.key[off..off + kv_dim].copy_from_slice(k);
            self.value[off..off + kv_dim].copy_from_slice(v);
            if !self.retained[pos] {
                self.retained[pos] = true;
                self.retained_count += 1;
            }
        }
        retain
    }

    /// Unconditional write (e.g., during prefill or warmup before predictor is trained).
    pub fn write_unconditional(&mut self, k: &[f32], v: &[f32], pos: usize, kv_dim: usize) {
        let off = pos * kv_dim;
        self.key[off..off + kv_dim].copy_from_slice(k);
        self.value[off..off + kv_dim].copy_from_slice(v);
        if !self.retained[pos] {
            self.retained[pos] = true;
            self.retained_count += 1;
        }
        self.utilities[pos] = 1.0;
    }
}

/// Multi-layer SP-KV cache: one `SpKvLayerCache` per transformer layer.
pub struct SpKvCache {
    /// Per-layer sparse KV caches.
    pub layers: Vec<SpKvLayerCache>,
    /// SP-KV configuration.
    pub config: SpKvConfig,
}

impl SpKvCache {
    /// Create a new multi-layer SP-KV cache.
    pub fn new(config: &SpKvConfig, n_layer: usize, block_size: usize, kv_dim: usize) -> Self {
        Self {
            layers: (0..n_layer)
                .map(|_| SpKvLayerCache::new(block_size, kv_dim))
                .collect(),
            config: config.clone(),
        }
    }

    /// Reset all layers.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Average density across all layers up to position `pos`.
    pub fn avg_density(&self, pos: usize) -> f32 {
        if pos == 0 || self.layers.is_empty() {
            return 1.0;
        }
        let total: f32 = self.layers.iter().map(|l| l.density(pos)).sum();
        total / self.layers.len() as f32
    }
}

/// Weights for the utility predictor: 2-layer MLP per layer.
///
/// Architecture: `h ∈ R^{d_model} → SiLU(W1·h + b1) → sigmoid(W2·hidden + b2) → u ∈ (0,1)^{n_kv_heads}`
///
/// One set of weights per transformer layer (different heads learn different sparsity).
/// Bias `b2` initialized to `predictor_init_bias` (default 5.0) so gates start open.
#[derive(Debug, Clone)]
pub struct UtilityPredictorWeights {
    /// First layer weights [hidden, d_model], row-major.
    pub w1: Vec<f32>,
    /// First layer bias [hidden]. Initialized to 0.
    pub b1: Vec<f32>,
    /// Second layer weights [n_kv_heads, hidden], row-major.
    pub w2: Vec<f32>,
    /// Second layer bias [n_kv_heads]. Initialized to +5.0 for open gates.
    pub b2: Vec<f32>,
}

impl UtilityPredictorWeights {
    /// Create new predictor weights with Xavier-like initialization.
    ///
    /// `b2` is set to `init_bias` (default 5.0) so sigmoid(b2) ≈ 0.993 → gates open.
    pub fn new(d_model: usize, hidden: usize, n_kv_heads: usize, init_bias: f32) -> Self {
        let w1_scale = (2.0 / d_model as f32).sqrt();
        let w2_scale = (2.0 / hidden as f32).sqrt();

        let mut rng = crate::types::Rng::new(42);

        let w1: Vec<f32> = (0..hidden * d_model)
            .map(|_| rng.normal() * w1_scale)
            .collect();
        let b1 = vec![0.0; hidden];
        let w2: Vec<f32> = (0..n_kv_heads * hidden)
            .map(|_| rng.normal() * w2_scale)
            .collect();
        let b2 = vec![init_bias; n_kv_heads];

        Self { w1, b1, w2, b2 }
    }

    /// Parameter count for this predictor.
    pub fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}

/// All utility predictor weights: one per transformer layer.
#[derive(Debug, Clone)]
pub struct SpKvPredictors {
    /// Per-layer utility predictor weights.
    pub layers: Vec<UtilityPredictorWeights>,
    /// Whether the predictors are frozen (TAHG phase).
    pub frozen: bool,
}

impl SpKvPredictors {
    /// Create predictors for all layers.
    pub fn new(
        n_layer: usize,
        d_model: usize,
        hidden: usize,
        n_kv_heads: usize,
        init_bias: f32,
    ) -> Self {
        Self {
            layers: (0..n_layer)
                .map(|_| UtilityPredictorWeights::new(d_model, hidden, n_kv_heads, init_bias))
                .collect(),
            frozen: false,
        }
    }

    /// Total parameter count across all layers.
    pub fn total_param_count(&self) -> usize {
        self.layers.iter().map(|p| p.param_count()).sum()
    }

    /// Freeze all predictor weights (start TAHG phase).
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Unfreeze all predictor weights (resume soft gating).
    pub fn unfreeze(&mut self) {
        self.frozen = false;
    }
}

/// Precomputed gate biases for one attention pass.
///
/// Built from utility predictions before the attention loop.
/// Avoids recomputing log(u) or threshold decisions per head per position.
pub struct GateBiasBuffer {
    /// Gate bias per position [block_size].
    /// Soft: log(u + ε), Hard: 0.0 or -∞, Tahg: blended.
    pub bias: Vec<f32>,
}

impl GateBiasBuffer {
    /// Create a new buffer for the given block size.
    pub fn new(block_size: usize) -> Self {
        Self {
            bias: vec![0.0; block_size],
        }
    }

    /// Build gate biases for soft gating mode (training phase 1).
    ///
    /// bias[s] = log(utility[s] + ε) for positions outside window.
    /// Inside window: bias = 0.0 (always attend).
    #[allow(clippy::needless_range_loop)]
    pub fn build_soft(&mut self, utilities: &[f32], pos: usize, window: usize) {
        let eps = 1e-8f32;
        for s in 0..=pos {
            let in_window = pos.saturating_sub(s) < window;
            self.bias[s] = if in_window {
                0.0
            } else {
                (utilities[s] + eps).ln()
            };
        }
    }

    /// Build gate biases for hard gating mode (inference).
    ///
    /// bias[s] = 0.0 if retained (utility ≥ τ or in window), else -∞.
    pub fn build_hard(
        &mut self,
        utilities: &[f32],
        retained: &[bool],
        pos: usize,
        window: usize,
        threshold: f32,
    ) {
        for s in 0..=pos {
            let in_window = pos.saturating_sub(s) < window;
            let retained_by_utility = utilities[s] >= threshold;
            self.bias[s] = if in_window || retained_by_utility || retained[s] {
                0.0
            } else {
                f32::NEG_INFINITY
            };
        }
    }

    /// Build gate biases for TAHG annealing (training phase 2).
    ///
    /// ũ = (1-α)·u + α·1[u≥τ], then bias = log(ũ + ε).
    /// α ramps linearly from 0→1 over `total_steps`.
    #[allow(clippy::needless_range_loop)]
    pub fn build_tahg(
        &mut self,
        utilities: &[f32],
        pos: usize,
        window: usize,
        threshold: f32,
        anneal_step: usize,
        total_steps: usize,
    ) {
        let eps = 1e-8f32;
        let alpha = if total_steps == 0 {
            1.0f32
        } else {
            (anneal_step as f32 / total_steps as f32).min(1.0)
        };

        for s in 0..=pos {
            let in_window = pos.saturating_sub(s) < window;
            if in_window {
                self.bias[s] = 0.0;
            } else {
                let u = utilities[s];
                let hard_indicator = if u >= threshold { 1.0f32 } else { 0.0f32 };
                let blended = (1.0 - alpha) * u + alpha * hard_indicator;
                self.bias[s] = (blended + eps).ln();
            }
        }
    }
}
