//! Inference backend abstraction for the transformer forward pass.
//!
//! Defines the [`InferenceBackend`] trait that decouples the high-level generate
//! loop from the concrete compute backend (CPU, Apple Neural Engine, etc.).
//!
//! The default [`CpuBackend`] delegates to [`crate::transformer::forward`].

use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::Config;

/// Backend for transformer forward pass inference.
///
/// Implementations: [`CpuBackend`] (default), AneBackend (Apple Neural Engine, feature-gated).
pub trait InferenceBackend {
    /// Run one forward pass: token + position → logits.
    ///
    /// Returns a logits slice of length `config.vocab_size`.
    /// The returned slice borrows from `ctx`.
    fn forward<'a>(
        &'a mut self,
        ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'a mut [f32];

    /// Human-readable device name for logging.
    fn device_name(&self) -> &'static str;

    /// Whether this backend supports stateful KV caching (e.g. CoreML MLState).
    fn supports_stateful(&self) -> bool {
        false
    }

    /// Reset any backend-specific state for a new sequence.
    fn reset(&mut self) {}
}

// ---------------------------------------------------------------------------
// CpuBackend
// ---------------------------------------------------------------------------

/// CPU backend using the standard Rust transformer forward pass.
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl InferenceBackend for CpuBackend {
    fn forward<'a>(
        &'a mut self,
        ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'a mut [f32] {
        crate::transformer::forward(ctx, weights, cache, token, pos, config)
    }

    fn device_name(&self) -> &'static str {
        "CPU"
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Auto-route selection
// ---------------------------------------------------------------------------

/// Backend selection for CLI flag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendKind {
    /// Automatically select best available backend.
    #[default]
    Auto,
    /// Force CPU backend.
    Cpu,
    /// Force ANE backend (error if unavailable).
    Ane,
}

/// Select the best available inference backend.
///
/// Logic:
/// 1. If `kind` is `Cpu`, always return `CpuBackend`.
/// 2. If `kind` is `Ane`, try to load `.mlmodelc` and create `AneBackend`.
/// 3. If `kind` is `Auto`:
///    - On macOS with `ane` feature: try ANE, fall back to CPU on failure.
///    - Otherwise: use CPU.
///
/// Logs which backend was selected via `log::info!`.
pub fn auto_backend(
    kind: BackendKind,
    model_path: Option<&std::path::Path>,
) -> Box<dyn InferenceBackend> {
    match kind {
        BackendKind::Cpu => {
            log::info!("Backend: CPU (forced)");
            Box::new(CpuBackend::new())
        }
        BackendKind::Ane => {
            // Will be caught below if ANE is not available
            try_ane_backend(model_path).expect("ANE backend requested but unavailable")
        }
        BackendKind::Auto => {
            let backend = try_ane_backend(model_path);
            match backend {
                Ok(b) => b,
                Err(reason) => {
                    log::info!("Backend: CPU (ANE unavailable: {reason})");
                    Box::new(CpuBackend::new())
                }
            }
        }
    }
}

/// Attempt to create an ANE backend. Returns error message on failure.
fn try_ane_backend(
    model_path: Option<&std::path::Path>,
) -> Result<Box<dyn InferenceBackend>, String> {
    #[cfg(all(target_os = "macos", feature = "ane"))]
    {
        use crate::ane_backend::{AneBackend, AneError};

        let path = match model_path {
            Some(p) => p,
            None => std::path::Path::new("model.mlmodelc"),
        };

        if !path.exists() {
            return Err(format!("model not found: {}", path.display()));
        }

        match AneBackend::new(path) {
            Ok(backend) => {
                // Residency check: verify ANE placement (<1ms threshold)
                match backend.check_residency(1000) {
                    Ok(latency_us) => {
                        log::info!("Backend: ANE (residency OK, {}μs)", latency_us);
                        Ok(Box::new(backend))
                    }
                    Err(AneError::ResidencyFailed {
                        latency_us,
                        threshold_us,
                    }) => Err(format!(
                        "residency check failed: {latency_us}μs > {threshold_us}μs"
                    )),
                    Err(e) => Err(e.to_string()),
                }
            }
            Err(e) => Err(e.to_string()),
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "ane")))]
    {
        let _ = model_path;
        Err("ANE not available (requires macOS + 'ane' feature)".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer;
    use crate::types::Rng;

    fn micro_fixtures() -> (
        Config,
        TransformerWeights,
        ForwardContext,
        MultiLayerKVCache,
    ) {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let ctx = ForwardContext::new(&config);
        let cache = MultiLayerKVCache::new(&config);
        (config, weights, ctx, cache)
    }

    #[test]
    fn test_cpu_backend_matches_direct_forward() {
        let (config, weights, _, _) = micro_fixtures();

        // Direct call
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let direct = transformer::forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config).to_vec();

        // Through CpuBackend
        let mut backend = CpuBackend::new();
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let via_backend = backend
            .forward(&mut ctx2, &weights, &mut cache2, 0, 0, &config)
            .to_vec();

        assert_eq!(
            direct, via_backend,
            "CpuBackend should produce identical logits to direct forward"
        );
    }

    #[test]
    fn test_cpu_backend_device_name() {
        let backend = CpuBackend::new();
        assert_eq!(backend.device_name(), "CPU");
    }

    #[test]
    fn test_cpu_backend_supports_stateful() {
        let backend = CpuBackend::new();
        assert!(!backend.supports_stateful());
    }

    #[test]
    fn test_cpu_backend_deterministic() {
        let (config, weights, _, _) = micro_fixtures();

        let mut backend = CpuBackend::new();

        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let run1 = backend
            .forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config)
            .to_vec();

        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let run2 = backend
            .forward(&mut ctx2, &weights, &mut cache2, 0, 0, &config)
            .to_vec();

        assert_eq!(run1, run2, "same input must produce same logits");
    }

    #[test]
    fn test_auto_backend_cpu_forced() {
        let backend = auto_backend(BackendKind::Cpu, None);
        assert_eq!(backend.device_name(), "CPU");
    }

    #[test]
    fn test_auto_backend_auto_falls_back_to_cpu() {
        // No model file available, so auto should fall back to CPU
        let backend = auto_backend(BackendKind::Auto, None);
        assert_eq!(backend.device_name(), "CPU");
    }

    #[test]
    fn test_backend_kind_default() {
        assert_eq!(BackendKind::default(), BackendKind::Auto);
    }
}
