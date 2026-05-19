//! SpectralQuant forward pass helpers.
//!
//! Provides dequantization and attention scoring functions for the
//! SpectralQuant KV cache path. The main forward function
//! (`forward_quantized`) is generic and lives in [`crate::transformer`].

use super::spectral_kv_cache::SpectralQuantKVCache;

/// Dequantize SpectralQuant key vectors for positions `[0..=pos]` into a flat buffer.
///
/// Layout: `[block_size * kv_dim]` row-major, compatible with the
/// attention kernel's expected `key_cache` layout.
pub fn dequantize_spectral_keys_flat(
    cache: &mut SpectralQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut flat = vec![0.0f32; (pos + 1) * kv_dim];
    for t in 0..=pos {
        cache.dequantize_key_into(layer, t, &mut flat[t * kv_dim..(t + 1) * kv_dim]);
    }
    flat
}

/// Dequantize SpectralQuant value vectors for positions `[0..=pos]` into a flat buffer.
///
/// Layout: `[block_size * kv_dim]` row-major, compatible with the
/// attention kernel's expected `value_cache` layout.
pub fn dequantize_spectral_values_flat(
    cache: &mut SpectralQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut flat = vec![0.0f32; (pos + 1) * kv_dim];
    for t in 0..=pos {
        cache.dequantize_value_into(layer, t, &mut flat[t * kv_dim..(t + 1) * kv_dim]);
    }
    flat
}

/// Compute per-head attention scores using dequantized SpectralQuant KV cache.
///
/// Self-contained attention scoring: Q·K → softmax → weighted V accumulation.
/// Accepts flat buffers produced by [`dequantize_spectral_keys_flat`] / [`dequantize_spectral_values_flat`].
#[allow(clippy::too_many_arguments)]
pub fn attention_spectralquant(
    q: &[f32],
    flat_keys: &[f32],
    flat_values: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    head_dim: usize,
    pos: usize,
    scale: f32,
) {
    let t_n = pos + 1;

    // Pass 1: Q·K scores + find max
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        let dot = unsafe {
            let q_slice = std::slice::from_raw_parts(q.as_ptr().add(q_head_offset), head_dim);
            let k_slice = std::slice::from_raw_parts(flat_keys.as_ptr().add(k_off), head_dim);
            crate::simd::simd_dot_f32(q_slice, k_slice, head_dim)
        };
        let score = dot * scale;
        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        if score > max_score {
            max_score = score;
        }
    }

    // Pass 2: exp(scores - max) + sum
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }

    // Pass 3: normalize + weighted value accumulation
    let inv_sum = 1.0 / sum;
    for d in 0..head_dim {
        let mut val = 0.0f32;
        for t in 0..t_n {
            unsafe {
                val += *scores_buf.get_unchecked(t)
                    * inv_sum
                    * *flat_values.get_unchecked(t * kv_dim + kv_group_offset + d);
            }
        }
        unsafe {
            *attn_out.get_unchecked_mut(q_head_offset + d) = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectralquant::spectral::participation_ratio;
    use crate::spectralquant::types::{SpectralQuantCalibration, SpectralQuantKVCacheConfig};
    use crate::types::{Config, Rng};

    #[test]
    fn test_spectralquant_forward_produces_finite() {
        let config = Config::micro();
        let kv_dim = crate::types::kv_dim(&config);
        let head_dim = config.head_dim;
        let n_embd = config.n_embd;

        // Build calibration with identity eigenvectors
        let mut eigenvectors = vec![0.0f32; kv_dim * kv_dim];
        for i in 0..kv_dim {
            eigenvectors[i * kv_dim + i] = 1.0;
        }
        let eigenvalues: Vec<f32> = (0..kv_dim).map(|i| 10.0 * 0.8f32.powi(i as i32)).collect();
        let d_eff = participation_ratio(&eigenvalues);
        let cal = SpectralQuantCalibration {
            eigenvectors,
            eigenvalues,
            d_eff,
            spectral_gap: None,
            var_95: 10,
            var_99: 20,
            n_samples: 100,
            head_dim: kv_dim,
        };

        let sq_config = SpectralQuantKVCacheConfig {
            avg_bits: 3.0,
            min_tail_bits: 1,
            max_bits: 8,
            qjl_dim: 16,
            lloyd_max_iter: 30,
            calibration_samples: 100,
            seed: 42,
            use_water_fill: false,
            wf_min_bits: 1,
            wf_max_bits: 6,
            n_layers: config.n_layer,
            kv_dim,
            max_seq_len: config.block_size,
        };

        let mut sq_cache = SpectralQuantKVCache::from_calibration(
            &sq_config,
            &vec![cal.clone(); config.n_layer],
            &vec![cal; config.n_layer],
        );

        // Store synthetic KV entries
        let kv: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.05).sin()).collect();
        for pos in 0..4 {
            sq_cache.store_key(0, pos, &kv);
            sq_cache.store_value(0, pos, &kv);
        }

        let flat_keys = dequantize_spectral_keys_flat(&mut sq_cache, 0, 3, kv_dim);
        let flat_values = dequantize_spectral_values_flat(&mut sq_cache, 0, 3, kv_dim);

        let mut rng = Rng::new(99);
        let q: Vec<f32> = (0..n_embd).map(|_| rng.normal()).collect();
        let mut attn_out = vec![0.0f32; n_embd];
        let mut scores = vec![0.0f32; config.block_size];

        attention_spectralquant(
            &q,
            &flat_keys,
            &flat_values,
            &mut attn_out,
            &mut scores,
            0,
            0,
            kv_dim,
            head_dim,
            3,
            1.0 / (head_dim as f32).sqrt(),
        );

        for (i, &v) in attn_out.iter().enumerate() {
            assert!(v.is_finite(), "attn_out[{i}] = {v} is not finite");
        }
    }
}
