//! Speculative step using GDN tree verification (Plan 424 T4.3).
//!
//! Routes GDN layers through [`forward_tree_gdn2`] instead of the KV-rollback
//! path. For pure-GDN2 models (all layers are DeltaNet), this uses the tree
//! verify primitive to process all draft tree nodes in one forward pass —
//! no state rollback needed.
//!
//! # Convention note
//!
//! The tree verify primitive uses the paper's convention (decay → read →
//! update with 1/√dₖ scaling), while the GDN2 kernel uses update-then-read.
//! The [`forward_tree_gdn2`] function applies a √dₖ scale correction to
//! bridge this gap. Full numerical equivalence with the GDN2 kernel requires
//! aligning the read/update order — tracked as T4.3b.

#![allow(clippy::needless_range_loop)]

use katgpt_attn::gdn2::forward::forward_gdn2;
use katgpt_attn::gdn2::tree_forward::forward_tree_gdn2;
use katgpt_attn::gdn2::MultiLayerGdn2Cache;
use katgpt_core::gdn_tree_verify::{GdnTreeVerifier, build_topology_from_tree_nodes};
use katgpt_core::speculative::sampling::{sample_from_distribution, sample_residual_distribution_into};
use katgpt_core::traits::NoPruner;
use katgpt_forward::{ForwardContext, SpeculativeContext};
use katgpt_forward::dflash::dflash_predict_with;
use katgpt_speculative::dd_tree::TreeBuilder;
use katgpt_transformer::TransformerWeights;
use crate::types::{Config, Rng, softmax_scaled};

/// Speculative step with GDN tree verification for pure-GDN2 models.
///
/// Uses [`forward_tree_gdn2`] to verify all draft tree nodes in one pass,
/// then applies p/q rejection sampling along the best path. The accepted path
/// is committed to the GDN2 cache via [`commit_gdn2_tree_layer`].
///
/// # Arguments
/// * `draft_sctx` — Draft speculative context (marginals buffer + scratch).
/// * `tree_builder` — Pre-allocated DDTree builder.
/// * `draft_weights` / `draft_config` — Draft model (for marginal prediction).
/// * `target_weights` / `target_config` — Target model (GDN2, for verification).
/// * `target_ctx` — Target forward context.
/// * `target_cache` — Target GDN2 multi-layer cache.
/// * `verifier` — Pre-allocated tree verify scratch.
/// * `token` / `pos` — Current token and position.
/// * `rng` — Random number generator.
///
/// # Returns
/// `(accepted_tokens, num_accepted)` — same format as `speculative_step_rollback_with`.
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_gdn_tree(
    draft_sctx: &mut SpeculativeContext,
    tree_builder: &mut TreeBuilder,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    target_ctx: &mut ForwardContext,
    target_cache: &mut MultiLayerGdn2Cache,
    verifier: &mut GdnTreeVerifier,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    let vocab_size = draft_config.vocab_size;

    // 1. Draft marginals via DFlash
    let _num_steps = dflash_predict_with(draft_sctx, draft_weights, draft_config, token, pos);
    let steps_populated = draft_sctx.steps_populated;
    let marginals_flat = &draft_sctx.marginals_flat;

    // Build marginals view (same as speculative_step_rollback_with)
    let mut marginals_buf: [&[f32]; 64] = [&[]; 64];
    let count = steps_populated.min(64);
    for (i, slot) in marginals_buf.iter_mut().enumerate().take(count) {
        let start = i * vocab_size;
        let end = start + vocab_size;
        *slot = if end <= marginals_flat.len() && i < steps_populated {
            &marginals_flat[start..end]
        } else {
            &[]
        };
    }
    let marginals = &marginals_buf[..count];

    // 2. Build DDTree
    let tree = tree_builder.build(marginals, draft_config, &NoPruner, false);

    if tree.is_empty() {
        let fallback = sample_from_distribution(
            marginals.first().copied().unwrap_or(&[1.0]),
            rng,
        );
        return (vec![fallback], 1);
    }

    // 3. Build tree topology from DDTree nodes
    // Use the GDN2 cache's first layer alpha as the scalar decay.
    let alpha = target_cache
        .layers
        .first()
        .and_then(|l| l.decay_alpha.first().copied())
        .unwrap_or(0.99);

    let (topo, token_ids) = build_topology_from_tree_nodes(tree, alpha);
    let t = topo.n_nodes;

    // 4. Forward all tree nodes through the target GDN2 model (read-only verify)
    let tree_logits = forward_tree_gdn2(
        target_ctx,
        target_weights,
        target_cache, // read-only — S₀ not modified
        &topo,
        &token_ids,
        pos,
        target_config,
        verifier,
    );

    // 5. Find the best path (highest-scoring root → leaf) and apply p/q rejection
    // For simplicity, we extract paths and try them in order (same as rollback_with).
    let paths = katgpt_forward::step::extract_ddtree_paths(tree);

    if paths.is_empty() {
        // Fallback: sample from the root node's logits
        let root_logits = &tree_logits[0..vocab_size];
        let mut probs = root_logits.to_vec();
        softmax_scaled(&mut probs, 1.0 / target_config.temperature);
        let fallback = sample_from_distribution(&probs, rng);
        return (vec![fallback], 1);
    }

    // 6. Try each candidate path with p/q rejection
    // We need the target logits per position. The tree_logits are in topo order.
    // Build a map: for each depth d, the target logits at that depth.
    // The tree topology's topo_order maps topo index → original index.
    // We need: depth d → topo index → logits.
    //
    // For the rejection sampling, we use the target logits at each depth
    // along the accepted path. The tree logits are already computed for ALL
    // nodes — we just need to pick the right ones for each path.

    let mut residual_buf: Vec<f32> = Vec::new();

    for path in &paths {
        let mut accepted = Vec::with_capacity(path.len());
        let mut all_accepted = true;

        // For each depth, find the topo node that matches this path and use
        // its logits for rejection sampling.
        // The path is a sequence of tokens. We need to find the tree node
        // at each depth that matches the path's prefix.
        let mut current_path_prefix: u128 = 0;

        for (depth, &draft_tok) in path.iter().enumerate() {
            current_path_prefix = if depth == 0 {
                draft_tok as u128
            } else {
                (current_path_prefix << 16) | (draft_tok as u128)
            };

            // Find the topo node matching (depth, current_path_prefix)
            // The tree_logits are in topo order: tree_logits[k * vocab..]
            // corresponds to topo node k.
            // We need to find the original node index, then its topo index.
            let target_logits: Option<Vec<f32>> = (0..t).find_map(|k| {
                let orig = topo.topo_order[k];
                let node = &tree[orig];
                if node.depth == depth && node.parent_path == current_path_prefix {
                    // Found the matching node at topo index k
                    let logits = &tree_logits[k * vocab_size..(k + 1) * vocab_size];
                    Some(logits.to_vec())
                } else {
                    None
                }
            });

            let Some(node_logits) = target_logits else {
                // No matching tree node — can't verify this token
                all_accepted = false;
                break;
            };

            let mut probs = node_logits;
            softmax_scaled(&mut probs, 1.0 / target_config.temperature);

            let q_dist = marginals.get(depth).copied().unwrap_or(&[]);
            let q_i = q_dist.get(draft_tok).copied().unwrap_or(0.0);
            let p_i = probs.get(draft_tok).copied().unwrap_or(0.0);

            let acceptance_prob = if q_i > 0.0 { (p_i / q_i).min(1.0) } else { 1.0 };

            if rng.uniform() <= acceptance_prob {
                accepted.push(draft_tok);
            } else {
                residual_buf.clear();
                residual_buf.resize(probs.len(), 0.0);
                let replacement =
                    sample_residual_distribution_into(&probs, q_dist, &mut residual_buf, rng);
                accepted.push(replacement);
                all_accepted = false;
                break;
            }
        }

        // Bonus token if all accepted
        if all_accepted && !accepted.is_empty() {
            // Use the last node's logits for the bonus
            let last_depth = path.len() - 1;
            let last_prefix = path.iter().take(last_depth + 1).enumerate().fold(0u128, |acc, (d, &tok)| {
                if d == 0 { tok as u128 } else { (acc << 16) | (tok as u128) }
            });
            let bonus_logits: Option<Vec<f32>> = (0..t).find_map(|k| {
                let orig = topo.topo_order[k];
                let node = &tree[orig];
                if node.depth == last_depth && node.parent_path == last_prefix {
                    let logits = &tree_logits[k * vocab_size..(k + 1) * vocab_size];
                    Some(logits.to_vec())
                } else {
                    None
                }
            });

            if let Some(mut bl) = bonus_logits {
                softmax_scaled(&mut bl, 1.0 / target_config.temperature);
                let bonus = sample_from_distribution(&bl, rng);
                accepted.push(bonus);
            }
        }

        if !accepted.is_empty() {
            // 7. Commit the accepted path to the GDN2 cache
            // Find the accepted leaf in the tree topology and commit along
            // the path root → leaf.
            let accepted_len = accepted.len().saturating_sub(1); // exclude bonus
            let mut commit_prefix: u128 = 0;
            let mut commit_leaf_topo: Option<usize> = None;

            for (depth, &tok) in accepted.iter().take(accepted_len).enumerate() {
                commit_prefix = if depth == 0 {
                    tok as u128
                } else {
                    (commit_prefix << 16) | (tok as u128)
                };
                // Find topo index of this node
                for k in 0..t {
                    let orig = topo.topo_order[k];
                    let node = &tree[orig];
                    if node.depth == depth && node.parent_path == commit_prefix {
                        commit_leaf_topo = Some(k);
                        break;
                    }
                }
            }

            if commit_leaf_topo.is_some() {
                // Build per-node Q/K/V for commit (same as forward_tree_gdn2).
                // The commit function needs the same keys/values/queries.
                // For now, we commit by replaying the accepted path through
                // the standard forward_gdn2 — this is simpler and correct
                // (the commit is a single sequential update along the path).
                commit_accepted_path_sequential(
                    target_ctx,
                    target_weights,
                    target_cache,
                    &accepted[..accepted_len],
                    token,
                    pos,
                    target_config,
                );
            }

            let len = accepted.len();
            return (accepted, len);
        }
    }

    // All paths exhausted: forward from current token, sample
    let logits = forward_gdn2(
        target_ctx,
        target_weights,
        target_cache,
        token,
        pos,
        target_config,
    );
    let mut probs = logits.to_vec();
    softmax_scaled(&mut probs, 1.0 / target_config.temperature);
    let fallback = sample_from_distribution(&probs, rng);
    (vec![fallback], 1)
}

/// Commit the accepted path by replaying it through `forward_gdn2` sequentially.
///
/// This is the simplest correct commit: it advances the GDN2 state along the
/// accepted path using the standard kernel (update-then-read). The tree verify
/// state (read-before-update) is not used for the commit — only for verification.
///
/// This means the committed state uses the GDN2 kernel's convention, which is
/// the convention the model uses for subsequent decode steps.
fn commit_accepted_path_sequential(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerGdn2Cache,
    accepted: &[usize],
    initial_token: usize,
    pos: usize,
    config: &Config,
) {
    let mut current_token = initial_token;
    for (i, &tok) in accepted.iter().enumerate() {
        // Forward processes the current token, updates state, produces logits.
        // The logits are discarded — we only care about the state update.
        let _logits = forward_gdn2(ctx, weights, cache, current_token, pos + i, config);
        current_token = tok;
    }
    // Process the last accepted token to update the state for it
    if let Some(&last) = accepted.last() {
        let _logits = forward_gdn2(ctx, weights, cache, last, pos + accepted.len(), config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_weights(config: &Config) -> TransformerWeights {
        let mut rng = Rng::new(42);
        TransformerWeights::new(config, &mut rng)
    }

    /// The GDN tree speculative step should return at least one token.
    #[test]
    fn test_speculative_step_gdn_tree_returns_tokens() {
        let draft_config = Config::micro();
        let target_config = Config::micro();
        let draft_weights = random_weights(&draft_config);
        let target_weights = random_weights(&target_config);

        let mut draft_sctx = SpeculativeContext::new(&draft_config);
        let mut tree_builder = TreeBuilder::new(&draft_config);
        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerGdn2Cache::new(&target_config);

        // Set paper-compatible alpha
        for layer in &mut target_cache.layers {
            layer.decay_alpha.fill(0.99);
            layer.erase_b.fill(1.0);
        }

        let hd = target_config.head_dim;
        let max_tree = 64; // generous for the DDTree
        let mut verifier = GdnTreeVerifier::new(max_tree, hd, hd);

        let mut rng = Rng::new(42);

        let (accepted, len) = speculative_step_gdn_tree(
            &mut draft_sctx,
            &mut tree_builder,
            &draft_weights,
            &draft_config,
            &target_weights,
            &target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut verifier,
            target_config.bos_token,
            0,
            &mut rng,
        );

        assert!(!accepted.is_empty(), "must accept at least one token");
        assert_eq!(len, accepted.len());
    }

    /// The GDN tree speculative step should be deterministic for the same seed.
    #[test]
    fn test_speculative_step_gdn_tree_deterministic() {
        let config = Config::micro();
        let weights = random_weights(&config);

        let run = || {
            let mut draft_sctx = SpeculativeContext::new(&config);
            let mut tree_builder = TreeBuilder::new(&config);
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerGdn2Cache::new(&config);
            for layer in &mut cache.layers {
                layer.decay_alpha.fill(0.99);
                layer.erase_b.fill(1.0);
            }
            let hd = config.head_dim;
            let mut verifier = GdnTreeVerifier::new(64, hd, hd);
            let mut rng = Rng::new(42);

            speculative_step_gdn_tree(
                &mut draft_sctx, &mut tree_builder,
                &weights, &config, &weights, &config,
                &mut ctx, &mut cache, &mut verifier,
                config.bos_token, 0, &mut rng,
            )
        };

        let (a1, _) = run();
        let (a2, _) = run();
        assert_eq!(a1, a2, "same seed must produce same accepted tokens");
    }
}
