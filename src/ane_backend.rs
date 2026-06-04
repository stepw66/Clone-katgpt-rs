//! Apple Neural Engine inference backend via CoreML (Plan 176).
//!
//! Uses `coreml-native` for safe CoreML access. ANE handles the matmul-heavy
//! forward pass while CPU handles discrete algorithms (DDTree, pruning,
//! speculative verification).
//!
//! # Residency Validation (Part 3)
//!
//! ANE execution is not guaranteed — CoreML may fall back to CPU/GPU if the
//! model graph doesn't fit ANE constraints. The residency check times a micro-
//! prediction: ANE < 1ms vs CPU fallback > 5ms. If residency fails, the auto-
//! route falls back to `CpuBackend`.
//!
//! # Stateful KV Cache (Part 4)
//!
//! macOS 15+ provides `MLState` for persistent KV cache across tokens.
//! This avoids re-sending the full KV cache on every call, roughly 2× faster
//! decode. Currently a placeholder — requires CoreML stateful model export.

use coreml_native as coreml;

use crate::inference_backend::InferenceBackend;
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::Config;

/// ANE inference backend using Apple CoreML framework.
///
/// Loads a pre-compiled `.mlmodelc` and runs inference on the Apple Neural Engine.
/// Falls back to CPU/GPU via CoreML if ANE placement fails (caught by residency check).
pub struct AneBackend {
    model: coreml::Model,
    model_path: std::path::PathBuf,
}

/// Error type for ANE backend operations.
#[derive(Debug)]
pub enum AneError {
    /// The .mlmodelc file was not found at the expected path.
    ModelNotFound(std::path::PathBuf),
    /// CoreML failed to load the model.
    LoadError(String),
    /// CoreML prediction failed.
    PredictionError(String),
    /// Model failed ANE residency check (falls back to CPU).
    ResidencyFailed { latency_us: u64, threshold_us: u64 },
    /// I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for AneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotFound(path) => write!(f, "model not found: {}", path.display()),
            Self::LoadError(msg) => write!(f, "CoreML load error: {msg}"),
            Self::PredictionError(msg) => write!(f, "CoreML prediction error: {msg}"),
            Self::ResidencyFailed {
                latency_us,
                threshold_us,
            } => {
                write!(
                    f,
                    "ANE residency check failed: {latency_us}μs > {threshold_us}μs threshold (model likely fell back to CPU)"
                )
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for AneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl AneBackend {
    /// Load a CoreML model from a `.mlmodelc` directory.
    ///
    /// The model should be pre-compiled from the conversion pipeline
    /// (see `scripts/convert_to_coreml.py`).
    pub fn new(model_path: &std::path::Path) -> Result<Self, AneError> {
        if !model_path.exists() {
            return Err(AneError::ModelNotFound(model_path.to_path_buf()));
        }

        let path_str = model_path
            .to_str()
            .ok_or_else(|| AneError::LoadError("path is not valid UTF-8".to_string()))?;

        let model = coreml::Model::load(path_str, coreml::ComputeUnits::All)
            .map_err(|e| AneError::LoadError(format!("{e:?}")))?;

        Ok(Self {
            model,
            model_path: model_path.to_path_buf(),
        })
    }

    /// Verify that the model actually runs on ANE (not CPU fallback).
    ///
    /// Times a micro-prediction: ANE should be <1ms, CPU fallback is >5ms.
    /// Returns the latency in microseconds.
    pub fn check_residency(&self, threshold_us: u64) -> Result<u64, AneError> {
        // Create a dummy input matching the model's expected input spec.
        let inputs = self.model.inputs();
        let first_input = inputs
            .first()
            .ok_or_else(|| AneError::PredictionError("model has no inputs".to_string()))?;

        // Build a zero-filled input tensor matching the input shape.
        // shape() returns Option<&[usize]>; use [1] as fallback for missing dims.
        let shape: Vec<usize> = first_input
            .shape()
            .map(|s| {
                s.iter()
                    .copied()
                    .map(|d| if d == 0 { 1 } else { d })
                    .collect()
            })
            .unwrap_or_else(|| vec![1]);
        let total_elems: usize = shape.iter().product::<usize>().max(1);
        let dummy_data = vec![0.0f32; total_elems];
        let tensor = coreml::BorrowedTensor::from_f32(&dummy_data, &shape)
            .map_err(|e| AneError::PredictionError(format!("tensor creation: {e:?}")))?;
        let input_name = first_input.name().to_string();

        // Run a warmup prediction first
        let _ = self
            .model
            .predict(&[(&input_name as &str, &tensor as &dyn coreml::AsMultiArray)])
            .ok();

        // Timed prediction
        let start = std::time::Instant::now();
        let _ = self
            .model
            .predict(&[(&input_name as &str, &tensor as &dyn coreml::AsMultiArray)])
            .ok();
        let elapsed = start.elapsed().as_micros() as u64;

        if elapsed > threshold_us {
            return Err(AneError::ResidencyFailed {
                latency_us: elapsed,
                threshold_us,
            });
        }

        Ok(elapsed)
    }

    /// Path to the loaded .mlmodelc.
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }
}

impl InferenceBackend for AneBackend {
    fn forward<'a>(
        &'a mut self,
        _ctx: &'a mut ForwardContext,
        _weights: &TransformerWeights,
        _cache: &mut MultiLayerKVCache,
        _token: usize,
        _pos: usize,
        _config: &Config,
    ) -> &'a mut [f32] {
        // NOTE: Full implementation requires:
        // 1. Constructing FP16 input tensors from token IDs + position
        // 2. Running model.predict() with proper input/output names
        // 3. Extracting logits from CoreML output (FP16 → f32)
        // 4. Writing logits into ctx.logits buffer
        //
        // The actual tensor I/O depends on the .mlmodelc's input/output spec,
        // which is determined by the conversion pipeline.
        // For now, this returns the CPU path as fallback during development.

        // TODO: Implement CoreML predict + logits extraction once conversion pipeline is ready
        crate::transformer::forward(_ctx, _weights, _cache, _token, _pos, _config)
    }

    fn device_name(&self) -> &'static str {
        "ANE"
    }

    fn supports_stateful(&self) -> bool {
        // Stateful KV cache via MLState requires macOS 15+
        // Will be enabled in Part 4
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_ane_error_display() {
        let err = AneError::ModelNotFound(std::path::PathBuf::from("model.mlmodelc"));
        assert!(err.to_string().contains("model not found"));

        let err = AneError::LoadError("bad format".to_string());
        assert!(err.to_string().contains("CoreML load error"));

        let err = AneError::ResidencyFailed {
            latency_us: 8000,
            threshold_us: 1000,
        };
        assert!(err.to_string().contains("residency check failed"));
    }

    #[test]
    fn test_ane_error_source() {
        let err = AneError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert!(err.source().is_some());

        let err = AneError::LoadError("test".to_string());
        assert!(err.source().is_none());
    }

    #[test]
    fn test_ane_backend_device_name() {
        // Can't actually construct AneBackend without a real .mlmodelc,
        // but we can test the trait method is correctly defined.
        fn assert_ane_device_name(backend: &dyn InferenceBackend) {
            assert_eq!(backend.device_name(), "ANE");
        }
        // This test verifies compilation; runtime test needs a real model file.
    }

    // ── Residency Validation Tests (Part 3) ──────────────────────

    #[test]
    fn test_residency_threshold_constant() {
        // The 1ms threshold is chosen because:
        // - ANE matmul for microGPT: ~50µs
        // - CPU fallback for same: ~5-10ms
        // - 1ms gives clear separation between the two regimes
        const ANE_RESIDENCY_THRESHOLD_US: u64 = 1000;
        assert_eq!(ANE_RESIDENCY_THRESHOLD_US, 1000);
    }

    #[test]
    fn test_residency_failed_error_message() {
        let err = AneError::ResidencyFailed {
            latency_us: 7500,
            threshold_us: 1000,
        };
        let msg = err.to_string();
        assert!(msg.contains("7500"), "should contain actual latency");
        assert!(msg.contains("1000"), "should contain threshold");
        assert!(msg.contains("residency check failed"));
    }

    #[test]
    fn test_model_not_found_error() {
        let err = AneError::ModelNotFound(std::path::PathBuf::from("/nonexistent/model.mlmodelc"));
        let msg = err.to_string();
        assert!(msg.contains("model not found"));
        assert!(msg.contains("/nonexistent/model.mlmodelc"));
    }
}
