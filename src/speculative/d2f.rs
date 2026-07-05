//! D2F (Discrete Diffusion Forcing) Inference Pipeline — root re-export shim.
//!
//! Plan 399 (2026-07-05): the bulk of this module (production code + 20
//! inference-only integration tests) moved to `katgpt_forward::d2f`. This
//! file is a thin re-export shim that preserves every historical
//! `crate::speculative::d2f::*` import path, plus a slim `#[cfg(test)]`
//! block for the 2 training-dependent tests that stayed in root because
//! they call `crate::dllm::{train_mini_dllm, generate_pattern_dataset}`.

pub use katgpt_forward::d2f::*;

#[cfg(test)]
mod tests {
    // These 2 tests stayed in root because they exercise the train+infer
    // interaction: they call `crate::dllm::{train_mini_dllm, generate_pattern_dataset}`
    // (root-only training code) to produce weights, then run the D2F
    // inference pipeline (now living in `katgpt_forward::d2f`) on them.
    use super::*;
    use crate::dllm::{generate_pattern_dataset, train_mini_dllm};
    use crate::types::{Config, Rng};
    use katgpt_core::traits::{NoPruner, NoScreeningPruner};

    #[test]
    fn test_decode_with_trained_model() {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(123);

        let train_data =
            generate_pattern_dataset(&mut rng, 20, config.block_size, config.vocab_size - 1);
        let test_data =
            generate_pattern_dataset(&mut rng, 5, config.block_size, config.vocab_size - 1);

        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);

        let decode_config = D2fDecodeConfig {
            denoise_steps: 16,
            confidence_threshold: 0.3,
            block_size: config.block_size,
            temperature: 0.8,
            ..D2fDecodeConfig::default()
        };

        let result = d2f_decode_block(
            &weights,
            &config,
            &decode_config,
            &NoPruner,
            &NoScreeningPruner,
            &mut rng,
        );

        let n_unmasked = result
            .tokens
            .iter()
            .filter(|&&t| t != config.mask_token)
            .count();
        assert!(
            n_unmasked > 0,
            "Expected at least 1 unmasked token, got {n_unmasked}"
        );
    }

    #[test]
    fn test_multistep_with_trained_model() {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);

        let train_data =
            generate_pattern_dataset(&mut rng, 20, config.block_size, config.vocab_size - 1);
        let test_data =
            generate_pattern_dataset(&mut rng, 5, config.block_size, config.vocab_size - 1);

        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);

        // Multistep with 4 steps should produce comparable results to standard 16 steps
        let multistep_config = D2fDecodeConfig {
            denoise_steps: 4,
            multistep: true,
            confidence_threshold: 0.3,
            block_size: config.block_size,
            temperature: 0.8,
            ..D2fDecodeConfig::default()
        };

        let result = d2f_decode_block(
            &weights,
            &config,
            &multistep_config,
            &NoPruner,
            &NoScreeningPruner,
            &mut rng,
        );

        let n_unmasked = result
            .tokens
            .iter()
            .filter(|&&t| t != config.mask_token)
            .count();
        assert!(
            n_unmasked > 0,
            "Multistep should unmask at least 1 token, got {n_unmasked}"
        );
    }
}
