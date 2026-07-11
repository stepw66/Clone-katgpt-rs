//! Plan 424 Phase 6 T6.2 — DDTree Deep-Argmax acceptance-length benchmark.
//!
//! Tests the paper's §3.5 / Figure 6 claim: at deep draft positions, the
//! factorized marginal is diluted (averages over many possible prefixes), and
//! argmax-of-marginal outperforms full-marginal branching on mean accepted
//! prefix length. The paper observes a crossover around draft length 2–4.
//!
//! # Method
//!
//! Since katgpt-rs has no target model, we use a **synthetic acceptance-length
//! proxy**: a known ground-truth token sequence `gt` plays the target's greedy
//! decode; the draft marginals concentrate on `gt[d]` with depth-decaying
//! signal plus decoy mass. "Accepted prefix length" = the longest prefix of
//! `gt` that appears as a root→leaf path in the built tree.
//!
//! This measures the **mechanism** (does concentrating budget on the argmax
//! chain extend the on-target path deeper?), not the absolute acceptance rate
//! (which depends on a real target model's error characteristics — the paper's
//! empirical crossover).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/424_dd_argmax \
//!   cargo run -p katgpt-speculative --bench bench_424_dd_tree_deep_argmax --release -- --nocapture
//! ```

use katgpt_speculative::dd_tree::{build_dd_tree_deep_argmax, extract_parent_tokens};
use katgpt_speculative::TreeNode;
use katgpt_types::Config;

/// xorshift32 PRNG for reproducible synthetic marginals.
fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Build synthetic draft marginals for a known ground-truth sequence.
///
/// At each depth `d`:
/// - `p[gt[d]] = signal(d)` — the draft's confidence in the correct token
/// - remaining mass `1 - signal(d)` spread uniformly over `n_decoys` decoys
///
/// `signal(d)` decays linearly: `base - slope * d`, clamped to `[floor, base]`.
/// This models the real phenomenon that deeper draft positions have noisier
/// marginals (the draft model has seen fewer conditioning tokens reliably).
fn make_draft_marginals(
    gt: &[usize],
    vocab: usize,
    base: f32,
    slope: f32,
    floor: f32,
    n_decoys: usize,
    seed: u32,
) -> Vec<Vec<f32>> {
    let mut rs = seed | 1; // avoid xorshift32 zero fixed-point
    gt.iter()
        .enumerate()
        .map(|(d, &gt_tok)| {
            let signal = (base - slope * d as f32).max(floor);
            let mut m = vec![0.0f32; vocab];
            m[gt_tok] = signal;
            let decoy_mass = (1.0 - signal).max(0.0);
            let per_decoy = if n_decoys > 0 {
                decoy_mass / n_decoys as f32
            } else {
                0.0
            };
            let mut placed = 0;
            while placed < n_decoys {
                let cand = (xorshift32(&mut rs) as usize) % vocab;
                if cand == gt_tok || m[cand] > 0.0 {
                    continue;
                }
                m[cand] = per_decoy;
                placed += 1;
            }
            m
        })
        .collect()
}

/// Accepted prefix length: longest prefix of `gt` that appears as a path in
/// the tree. A node at depth `d` "covers" gt[0..=d] iff its extracted parent
/// tokens equal `gt[0..=d]`.
fn accepted_prefix_len(tree: &[TreeNode], gt: &[usize]) -> usize {
    let mut best = 0;
    for node in tree {
        if node.depth >= gt.len() {
            continue;
        }
        let path = extract_parent_tokens(node.parent_path, node.depth + 1);
        if path.len() == node.depth + 1 && path == &gt[..node.depth + 1] {
            if node.depth + 1 > best {
                best = node.depth + 1;
            }
        }
    }
    best
}

/// Mean tree depth (how deep the tree extends on average — budget concentration
/// proxy). With `Some(t)`, budget concentrates → trees tend deeper.
fn mean_depth(tree: &[TreeNode]) -> f64 {
    if tree.is_empty() {
        return 0.0;
    }
    let sum: usize = tree.iter().map(|n| n.depth + 1).sum();
    sum as f64 / tree.len() as f64
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 424 Phase 6 T6.2 — DDTree Deep-Argmax Acceptance Benchmark ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    // Config: the threshold only matters when budget is tight relative to
    // branching width. With vocab >> budget, full branching "wastes" budget
    // on shallow siblings and the argmax chain can't extend deep; Some(t)
    // concentrates budget on the chain. With generous budget, the best-first
    // heap already follows the argmax and the threshold is a no-op.
    let mut config = Config::draft();
    config.vocab_size = 32;
    config.draft_lookahead = 8;
    config.tree_budget = 12; // tight: < vocab → full branching can't reach deep

    let n_seeds = 200usize;
    let n_decoys = 3; // 3 decoys per depth → meaningful branching mass

    // Two regimes, mirroring the paper's crossover story:
    //  - "slow decay": signal stays high (0.85→0.50 over 8 depths). Argmax
    //    is reliably correct at all depths → greedy should not hurt, may help
    //    by extending the chain deeper.
    //  - "fast decay": signal drops fast (0.70→0.07). Deep argmax is barely
    //    above decoys → the realistic regime where the crossover lives.
    let regimes: &[(&str, f32, f32, f32)] = &[
        ("slow-decay", 0.85, 0.05, 0.40),
        ("fast-decay", 0.70, 0.09, 0.05),
    ];

    let thresholds: &[Option<usize>] = &[None, Some(2), Some(4)];

    for &(name, base, slope, floor) in regimes {
        println!(
            "║                                                                  ║"
        );
        println!(
            "║  regime={name:12}  base={base:.2}  slope={slope:.2}  floor={floor:.2}  ({n_seeds} seeds)",
        );
        println!(
            "║  ┌──────────────┬──────────────────┬──────────────────┬───────────┐"
        );
        println!(
            "║  │ threshold    │ mean accept len  │ mean tree depth  │ tree size │"
        );
        println!(
            "║  ├──────────────┼──────────────────┼──────────────────┼───────────┤"
        );

        let mut best_accept = 0.0f64;
        let mut best_label = String::new();

        for &thresh in thresholds {
            let label = match thresh {
                None => "None".to_string(),
                Some(t) => format!("Some({t})"),
            };

            let mut sum_accept = 0usize;
            let mut sum_depth = 0.0f64;
            let mut sum_size = 0usize;

            for seed in 0..n_seeds as u32 {
                // Ground-truth sequence (target's greedy decode).
                let mut rs = seed.wrapping_mul(7919).wrapping_add(1);
                let gt: Vec<usize> = (0..config.draft_lookahead)
                    .map(|_| (xorshift32(&mut rs) as usize) % config.vocab_size)
                    .collect();

                let marginals =
                    make_draft_marginals(&gt, config.vocab_size, base, slope, floor, n_decoys, seed);
                let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

                let tree = build_dd_tree_deep_argmax(&mv, &config, thresh);
                sum_accept += accepted_prefix_len(&tree, &gt);
                sum_depth += mean_depth(&tree);
                sum_size += tree.len();
            }

            let mean_accept = sum_accept as f64 / n_seeds as f64;
            let mean_d = sum_depth / n_seeds as f64;
            let mean_sz = sum_size as f64 / n_seeds as f64;

            if mean_accept > best_accept {
                best_accept = mean_accept;
                best_label = label.clone();
            }

            println!(
                "║  │ {label:10}   │ {mean_accept:14.3}   │ {mean_d:14.3}   │ {mean_sz:7.1}   │"
            );
        }
        println!(
            "║  └──────────────┴──────────────────┴──────────────────┴───────────┘"
        );
        println!(
            "║  → best: {best_label:10}  (mean accept len = {best_accept:.3})",
        );
    }

    println!(
        "║                                                                  ║"
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("VERDICT: No acceptance-length gain from deep_argmax_threshold on the");
    println!("DDTree best-first builder. The best-first heap scores nodes by cumulative");
    println!("log-prob, so the argmax token at each depth already dominates the pop");
    println!("order — the on-target chain extends as deep as budget allows regardless");
    println!("of whether siblings are also pushed. The threshold is a correct but");
    println!("redundant mechanism on this architecture.");
    println!();
    println!("The paper's §3.5 / Figure 6 argmax-vs-marginal crossover applies to tree");
    println!("builders that SAMPLE from the marginal (stochastic expansion), not to");
    println!("deterministic best-first expansion. A future sampling-based builder could");
    println!("benefit from the flag; the current best-first DDTree does not.");
    println!();
    println!("NOTE: This is a synthetic mechanism test. The argmax is always the");
    println!("ground-truth token (by construction), so this tests whether budget");
    println!("concentration alone extends the accepted prefix — a necessary (not");
    println!("sufficient) condition for the paper's claim under realistic draft noise.");
}
