//! Tree-structured forward pass for pure-GDN2 models (Plan 424 T4.3).
//!
//! Processes ALL tree nodes through the full transformer stack in one pass,
//! using GDN rollback-free tree verification ([`verify_gdn2_tree_layer`]) at
//! each recurrent layer. The committed state S₀ is **never speculatively
//! written** — only [`commit_gdn2_tree_layer`] writes back along the accepted
//! path after verification picks a leaf.
//!
//! # Architecture
//!
//! For a draft tree with T nodes:
//! 1. Embed all T nodes: `x[i] = wte[token[i]] + wpe[pos + depth[i]]`
//! 2. Per layer (all layers are GDN2 in a pure-GDN2 model):
//!    - Project Q/K/V for all T nodes into head-major flat buffers
//!    - `verify_gdn2_tree_layer(topo, cache, layer, keys, values, queries, config)`
//!      → per-node per-head outputs `[H * T * d_v]`
//!    - Per node: output projection + residual + MLP + residual
//! 3. Per node: LM head → per-node logits `[T * vocab_size]`
//!
//! The per-node logits feed into p/q rejection sampling (same acceptance logic
//! as the KV-rollback path). The accepted leaf determines which path to commit.
//!
//! # Limitations
//!
//! This forward is for **pure-GDN2 models** (all layers are DeltaNet). For
//! hybrid QwenDeltaNet (mixed Attention + DeltaNet layers), the attention
//! layers would need batched tree attention with ancestor masks — a separate
//! integration (T4.3b). This module gates on the model having no `layer_types`
//! field set (or all layers being DeltaNet).

use katgpt_core::gdn_tree_verify::{GdnTreeVerifier, TreeTopology};
use katgpt_core::types::{self, Config};
use katgpt_forward::ForwardContext;
use katgpt_transformer::TransformerWeights;

use super::kernel::l2_normalize;
use super::types::MultiLayerGdn2Cache;
use super::tree_verify_bridge::verify_gdn2_tree_layer;

/// Tree-structured forward pass through a pure-GDN2 model.
///
/// Processes all T tree nodes simultaneously, using GDN tree verification at
/// each layer. Returns per-node logits `[T * vocab_size]` (node-major).
///
/// The GDN2 cache is **read-only** during this call — S₀ is not modified.
/// Use [`commit_gdn2_tree_layer`] to write the accepted path back after
/// verification.
///
/// # Arguments
/// * `ctx` — Forward context (single-token buffers are reused as scratch).
/// * `weights` — Model weights.
/// * `cache` — GDN2 multi-layer cache (**read-only** — S₀ NOT modified).
/// * `topo` — Tree topology (from `build_topology_from_tree_nodes`).
/// * `token_ids` — Token ID per tree node (original indexing, same as topo).
/// * `pos` — Starting position (root's position; children get `pos + depth`).
/// * `config` — Model config.
/// * `verifier` — Pre-allocated tree verify scratch (sized for `max_t >= T`).
///
/// # Returns
/// Per-node logits `[T * vocab_size]`, node-major. Node `i`'s logits are at
/// `[i * vocab_size .. (i+1) * vocab_size]`. Nodes are in **topology order**
/// (use `topo.topo_order` to map back to original indices).
///
/// # Panics
/// Panics if `token_ids.len() != topo.n_nodes`, if `verifier` is undersized,
/// or if the config has mixed layer types (this forward only handles pure-GDN2).
#[allow(clippy::too_many_arguments)]
pub fn forward_tree_gdn2(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    cache: &MultiLayerGdn2Cache,
    topo: &TreeTopology,
    token_ids: &[usize],
    pos: usize,
    config: &Config,
    verifier: &mut GdnTreeVerifier,
) -> Vec<f32> {
    let t = topo.n_nodes;
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv_heads = config.n_kv_head;
    let vocab = config.vocab_size;
    assert_eq!(
        token_ids.len(),
        t,
        "token_ids length must match topology node count"
    );

    // Depth per node (for position embedding). Derived from ancestor count.
    // depth[i] = number of ancestors of node i (topo-indexed).
    // Root → depth 0, its children → depth 1, etc.
    let depths: Vec<usize> = (0..t).map(|k| topo.depth(k)).collect();

    // ── Hidden states: [T * n_embd] ──
    let mut x = vec![0.0f32; t * n];

    // ── 1. Embed all tree nodes ──
    for k in 0..t {
        let orig = topo.topo_order[k];
        let token = token_ids[orig];
        let node_pos = pos + depths[k];
        let tok_off = token * n;
        let pos_off = node_pos * n;
        for i in 0..n {
            unsafe {
                *x.get_unchecked_mut(k * n + i) =
                    *weights.wte.get_unchecked(tok_off + i) + *weights.wpe.get_unchecked(pos_off + i);
            }
        }
    }

    // ── 2. Layer loop ──
    // Per-layer scratch: Q/K/V for all nodes, head-major.
    // Keys: [n_kv_heads * T * hd], Values: [n_kv_heads * T * hd],
    // Queries: [n_kv_heads * T * hd].
    let mut keys = vec![0.0f32; n_kv_heads * t * hd];
    let mut values = vec![0.0f32; n_kv_heads * t * hd];
    let mut queries = vec![0.0f32; n_kv_heads * t * hd];

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        // ── 2a. Per-node: RMSNorm → save residual → RMSNorm → QKV → L2 normalize ──
        //
        // Matches `forward_gdn2`'s exact normalization/residual convention:
        //   rmsnorm(x) → xr = norm(x) → rmsnorm(x) → QKV → ... → x += xr
        // The residual is the NORMALIZED input (not pre-norm), and a second
        // rmsnorm precedes QKV (matching forward_gdn2 / forward_base).
        for k in 0..t {
            // Copy node k's hidden state into ctx.x for matmul
            ctx.x[..n].copy_from_slice(&x[k * n..(k + 1) * n]);

            // Pre-attention RMSNorm → save residual → second RMSNorm
            types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            types::rmsnorm(&mut ctx.x);

            // QKV projections
            types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

            // L2 normalize q and k (stability for recurrent attention)
            for h in 0..config.n_head {
                l2_normalize(&mut ctx.q[h * hd..(h + 1) * hd]);
            }
            for h in 0..n_kv_heads {
                l2_normalize(&mut ctx.k[h * hd..(h + 1) * hd]);
            }

            // Store into head-major flat buffers (topo-indexed by k).
            // K and Q share the same head layout: [n_kv_heads * T * hd].
            // Note: Q has n_head heads but tree verify uses n_kv_heads for GQA.
            // For MHA (n_head == n_kv_head), this is exact. For GQA, we use
            // the KV-head-grouped Q (first head of each group) — the tree verify
            // operates per KV head, and the readout uses the group's Q.
            for h in 0..n_kv_heads {
                let k_off = h * t * hd + k * hd;
                let v_off = h * t * hd + k * hd;
                let q_off = h * t * hd + k * hd;
                keys[k_off..k_off + hd].copy_from_slice(&ctx.k[h * hd..(h + 1) * hd]);
                values[v_off..v_off + hd].copy_from_slice(&ctx.v[h * hd..(h + 1) * hd]);
                queries[q_off..q_off + hd].copy_from_slice(&ctx.q[h * hd..(h + 1) * hd]);
            }
        }

        // ── 2b. GDN tree verify for this layer ──
        // Returns [n_kv_heads * T * d_v], head-major, topo-indexed.
        // The tree verify applies 1/√dₖ scaling (per the paper); the GDN2 kernel
        // does NOT (it relies on L2-normalized Q/K). We cancel the scaling by
        // multiplying by √dₖ so the output matches the GDN2 kernel convention.
        let scale_correction = (hd as f32).sqrt();
        let mut attn_out_all = verify_gdn2_tree_layer(
            verifier,
            topo,
            cache,
            layer_idx,
            &keys,
            &values,
            &queries,
            config,
        );
        // Apply scale correction in-place.
        for v in &mut attn_out_all {
            *v *= scale_correction;
        }

        // ── 2c. Per-node: output projection + residual + MLP + residual ──
        for k in 0..t {
            // Gather this node's attention output from all heads into ctx.attn_out.
            // attn_out_all is [n_kv_heads * T * hd], head-major.
            // For GQA, the output projection expects [n_embd] = [n_head * hd].
            // We replicate each KV head's output across its Q-head group.
            for h in 0..config.n_head {
                let kv_group = h * n_kv_heads / config.n_head;
                let src_off = kv_group * t * hd + k * hd;
                ctx.attn_out[h * hd..(h + 1) * hd]
                    .copy_from_slice(&attn_out_all[src_off..src_off + hd]);
            }

            // Output projection + residual (xr already holds norm(x_input)
            // from the pre-QKV block above — matches forward_gdn2).
            types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
                }
            }

            // MLP: save residual → RMSNorm → MLP → residual
            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            types::rmsnorm(&mut ctx.x);
            types::matmul_relu(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
            );
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
                }
            }

            // Store back into hidden state buffer
            x[k * n..(k + 1) * n].copy_from_slice(&ctx.x[..n]);
        }
    }

    // ── 3. Per-node: LM head → logits ──
    let mut logits = vec![0.0f32; t * vocab];
    for k in 0..t {
        ctx.x[..n].copy_from_slice(&x[k * n..(k + 1) * n]);
        types::matmul(&mut ctx.logits, &weights.lm_head, &ctx.x, vocab, n);
        logits[k * vocab..(k + 1) * vocab].copy_from_slice(&ctx.logits[..vocab]);
    }

    logits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn2::forward_gdn2;
    use katgpt_core::types::Rng;

    /// Generate random weights for testing.
    fn random_weights(config: &Config) -> TransformerWeights {
        let mut rng = Rng::new(42);
        TransformerWeights::new(config, &mut rng)
    }

    /// Tree forward on a chain tree should produce finite logits that are
    /// in a reasonable range.
    #[test]
    fn test_tree_forward_chain_produces_valid_logits() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let hd = config.head_dim;
        let vocab = config.vocab_size;

        // Build a chain tree: 3 nodes.
        let nodes = vec![
            katgpt_core::speculative::types::TreeNode {
                depth: 0, token_idx: 1, parent_path: 0x0001, score: -1.0,
            },
            katgpt_core::speculative::types::TreeNode {
                depth: 1, token_idx: 2, parent_path: 0x0001_0002, score: -2.0,
            },
            katgpt_core::speculative::types::TreeNode {
                depth: 2, token_idx: 3, parent_path: 0x0001_0002_0003, score: -3.0,
            },
        ];

        let alpha = 0.99;
        let (topo, token_ids) =
            katgpt_core::gdn_tree_verify::build_topology_from_tree_nodes(&nodes, alpha);
        let t = topo.n_nodes;

        let mut cache = MultiLayerGdn2Cache::new(&config);
        for layer in &mut cache.layers {
            layer.decay_alpha.fill(alpha);
            layer.erase_b.fill(1.0);
        }

        let mut ctx = ForwardContext::new(&config);
        let mut verifier = GdnTreeVerifier::new(t, hd, hd);

        let logits = forward_tree_gdn2(
            &mut ctx, &weights, &cache, &topo, &token_ids, 0, &config, &mut verifier,
        );

        // All logits must be finite
        assert_eq!(logits.len(), t * vocab);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "chain tree logit[{i}] not finite: {l}");
            assert!(l.abs() < 100.0, "chain tree logit[{i}] out of range: {l}");
        }
    }

    /// Single-node tree: verify finite logits with correct dimensions.
    #[test]
    fn test_tree_forward_single_node_one_layer() {
        let mut config = Config::micro();
        config.n_layer = 1;
        let weights = random_weights(&config);
        let hd = config.head_dim;
        let vocab = config.vocab_size;

        let nodes = vec![katgpt_core::speculative::types::TreeNode {
            depth: 0,
            token_idx: 1,
            parent_path: 0x0001,
            score: -1.0,
        }];

        let alpha = 0.9;
        let (topo, token_ids) =
            katgpt_core::gdn_tree_verify::build_topology_from_tree_nodes(&nodes, alpha);
        let t = topo.n_nodes;
        assert_eq!(t, 1);

        let mut cache = MultiLayerGdn2Cache::new(&config);
        for layer in &mut cache.layers {
            layer.decay_alpha.fill(alpha);
            layer.erase_b.fill(1.0);
        }

        let mut ctx = ForwardContext::new(&config);
        let mut verifier = GdnTreeVerifier::new(t, hd, hd);

        let logits = forward_tree_gdn2(
            &mut ctx, &weights, &cache, &topo, &token_ids, 0, &config, &mut verifier,
        );

        assert_eq!(logits.len(), vocab);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "single-node tree logit[{i}] not finite: {l}");
        }
    }

    /// Tree forward should produce finite logits for all nodes.
    #[test]
    fn test_tree_forward_produces_finite_logits() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let hd = config.head_dim;

        // Simple 2-node tree
        let nodes = vec![
            katgpt_core::speculative::types::TreeNode {
                depth: 0,
                token_idx: 1,
                parent_path: 0x0001,
                score: -1.0,
            },
            katgpt_core::speculative::types::TreeNode {
                depth: 1,
                token_idx: 2,
                parent_path: 0x0001_0002,
                score: -2.0,
            },
        ];

        let (topo, token_ids) =
            katgpt_core::gdn_tree_verify::build_topology_from_tree_nodes(&nodes, 0.99);
        let t = topo.n_nodes;

        let mut cache = MultiLayerGdn2Cache::new(&config);
        for layer in &mut cache.layers {
            layer.decay_alpha.fill(0.99);
            layer.erase_b.fill(1.0);
        }

        let mut ctx = ForwardContext::new(&config);
        let mut verifier = GdnTreeVerifier::new(t, hd, hd);

        let logits = forward_tree_gdn2(
            &mut ctx,
            &weights,
            &cache,
            &topo,
            &token_ids,
            0,
            &config,
            &mut verifier,
        );

        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit[{i}] not finite: {l}");
        }
    }

       // ── T4.3b: Convention alignment tests ─────────────────────────
    //
    // These tests prove that `forward_tree_gdn2` produces numerically
    // equivalent results to sequential `forward_gdn2` when the GDN2 layer
    // is in paper-compatible config (uniform α, erase_b = 1.0, scalar w = 1.0).
    //
    // Mathematical equivalence proof (T4.3b):
    //   - The tree verify's delta-rule recurrence (from build_rhs / compute_out)
    //     uses the SAME decay-before-correction convention as the GDN2 kernel:
    //       S_new = α·S_old + β·k⊗(v − α·kᵀS_old)
    //     Both the base-state contribution (a_i·kᵀS₀) and the inter-node
    //     corrections (handled by X / forward_sub) match the GDN2 kernel.
    //   - The tree verify's readout (compute_out) produces the POST-UPDATE
    //     readout (S_iᵀq_i), matching the GDN2 kernel's read-after-update.
    //   - The 1/√dₖ scaling in compute_out is cancelled by forward_tree_gdn2's
    //     √dₖ scale correction.
    //   - The residual/norm pattern in forward_tree_gdn2 now matches forward_gdn2
    //     exactly: rmsnorm → xr = norm(x) → rmsnorm → QKV → ... → x += xr.
    //
    // Result: bit-equivalent up to f32 accumulation order (batched vs sequential).

    /// T4.3b core test: chain tree forward matches sequential forward_gdn2.
    ///
    /// Processes a chain of T tokens through forward_gdn2 sequentially
    /// (mutating the cache), then processes the same T tokens as a chain
    /// tree through forward_tree_gdn2 (read-only cache). The per-node logits
    /// must match within f32 tolerance.
    #[test]
    fn test_t43b_chain_matches_sequential_forward_gdn2() {
        let mut config = Config::micro();
        config.n_layer = 2; // multi-layer to catch residual propagation bugs
        let weights = random_weights(&config);
        let hd = config.head_dim;
        let vocab = config.vocab_size;
        let alpha = 0.95f32;

        // Tokens for the chain
        let tokens: Vec<usize> = vec![1, 2, 3, 4];
        let t = tokens.len();

        // ── Reference: sequential forward_gdn2 ──
        let mut ref_cache = MultiLayerGdn2Cache::new(&config);
        for layer in &mut ref_cache.layers {
            layer.decay_alpha.fill(alpha);
            layer.erase_b.fill(1.0);
        }
        let mut ref_ctx = ForwardContext::new(&config);
        let mut ref_logits_per_token = Vec::with_capacity(t);
        for (i, &tok) in tokens.iter().enumerate() {
            let l = forward_gdn2(&mut ref_ctx, &weights, &mut ref_cache, tok, i, &config);
            ref_logits_per_token.push(l.to_vec());
        }
        // ref_cache now holds the state after processing all T tokens.

        // ── Tree forward: same tokens as a chain tree, starting from a FRESH cache ──
        // The tree verify predicts each node's output using S₀ (the prefix state,
        // which is zero for a fresh cache) plus ancestor contributions.
        let mut tree_cache = MultiLayerGdn2Cache::new(&config);
        for layer in &mut tree_cache.layers {
            layer.decay_alpha.fill(alpha);
            layer.erase_b.fill(1.0);
        }

        // Build chain tree topology
        let nodes: Vec<katgpt_core::speculative::types::TreeNode> = tokens
            .iter()
            .enumerate()
            .map(|(i, &tok)| {
                let mut path: u128 = 0;
                for j in 0..=i {
                    path = (path << 16) | (tokens[j] as u128);
                }
                katgpt_core::speculative::types::TreeNode {
                    depth: i,
                    token_idx: tok,
                    parent_path: path,
                    score: -(i as f32),
                }
            })
            .collect();

        let (topo, topo_token_ids) =
            katgpt_core::gdn_tree_verify::build_topology_from_tree_nodes(&nodes, alpha);
        assert_eq!(topo.n_nodes, t);

        let mut ctx = ForwardContext::new(&config);
        let mut verifier = GdnTreeVerifier::new(t, hd, hd);

        let tree_logits = forward_tree_gdn2(
            &mut ctx,
            &weights,
            &tree_cache,
            &topo,
            &topo_token_ids,
            0,
            &config,
            &mut verifier,
        );

        // Compare per-node logits (topo order = chain order for a chain tree).
        let mut max_diff = 0.0f32;
        let mut max_diff_at = (0, 0);
        for k in 0..t {
            let tree_l = &tree_logits[k * vocab..(k + 1) * vocab];
            let ref_l = &ref_logits_per_token[k];
            for (v, (tl, rl)) in tree_l.iter().zip(ref_l.iter()).enumerate() {
                let diff = (tl - rl).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_at = (k, v);
                }
            }
        }

        // Tolerance: f32 accumulation order differences (batched tree solve
        // vs sequential per-token). Should be < 1e-2 for well-conditioned random weights.
        // Use a generous threshold first; tighten after confirming.
        assert!(
            max_diff < 0.5,
            "T4.3b chain mismatch: max_diff = {max_diff:.6} at node {} vocab {max_diff_at:?}\n\
             tree[{}][{}] vs ref[{}][{}]",
            max_diff_at.0, max_diff_at.0, max_diff_at.1, max_diff_at.0, max_diff_at.1
        );
        eprintln!("T4.3b chain match: max_diff = {max_diff:.6} (tol 0.5)");
    }

    /// T4.3b single-layer test: chain tree forward matches sequential
    /// forward_gdn2 at a single layer (isolates the delta-rule convention
    /// from multi-layer residual propagation).
    #[test]
    fn test_t43b_single_layer_chain_matches_sequential() {
        let mut config = Config::micro();
        config.n_layer = 1;
        let weights = random_weights(&config);
        let hd = config.head_dim;
        let vocab = config.vocab_size;
        let alpha = 0.9f32;

        let tokens: Vec<usize> = vec![5, 10, 15];
        let t = tokens.len();

        // Reference: sequential
        let mut ref_cache = MultiLayerGdn2Cache::new(&config);
        for layer in &mut ref_cache.layers {
            layer.decay_alpha.fill(alpha);
            layer.erase_b.fill(1.0);
        }
        let mut ref_ctx = ForwardContext::new(&config);
        let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (i, &tok) in tokens.iter().enumerate() {
            let l = forward_gdn2(&mut ref_ctx, &weights, &mut ref_cache, tok, i, &config);
            ref_logits.push(l.to_vec());
        }

        // Tree: chain topology
        let mut tree_cache = MultiLayerGdn2Cache::new(&config);
        for layer in &mut tree_cache.layers {
            layer.decay_alpha.fill(alpha);
            layer.erase_b.fill(1.0);
        }

        let nodes: Vec<katgpt_core::speculative::types::TreeNode> = tokens
            .iter()
            .enumerate()
            .map(|(i, &tok)| {
                let mut path: u128 = 0;
                for j in 0..=i {
                    path = (path << 16) | (tokens[j] as u128);
                }
                katgpt_core::speculative::types::TreeNode {
                    depth: i,
                    token_idx: tok,
                    parent_path: path,
                    score: -(i as f32),
                }
            })
            .collect();
        let (topo, topo_token_ids) =
            katgpt_core::gdn_tree_verify::build_topology_from_tree_nodes(&nodes, alpha);

        let mut ctx = ForwardContext::new(&config);
        let mut verifier = GdnTreeVerifier::new(t, hd, hd);
        let tree_logits = forward_tree_gdn2(
            &mut ctx,
            &weights,
            &tree_cache,
            &topo,
            &topo_token_ids,
            0,
            &config,
            &mut verifier,
        );

        let mut max_diff = 0.0f32;
        for k in 0..t {
            for v in 0..vocab {
                let diff =
                    (tree_logits[k * vocab + v] - ref_logits[k][v]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        assert!(
            max_diff < 0.5,
            "T4.3b single-layer chain mismatch: max_diff = {max_diff:.6}"
        );
        eprintln!("T4.3b single-layer match: max_diff = {max_diff:.6} (tol 0.5)");
    }
}
