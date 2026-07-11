//! GatePlayer — bomber player backed by InferenceRouter for routed inference.
//!
//! Encodes heuristic game-state features as token inputs, routes each forward
//! pass through the [`InferenceRouter`] / [`TriggerGate`] tier system, and
//! maps the resulting logits back to action scores.
//!
//! This proves the gate routing pipeline works inside the bomber arena loop:
//! every `select_action()` call exercises the TriggerGate QPS/latency checks,
//! tier promotion/demotion logic, and CPU fallback path.

use std::any::Any;

use fastrand::Rng;

use crate::inference_router::InferenceRouter;
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use katgpt_core::trigger_gate::TriggerGateConfig;
use crate::types::{Config, Rng as KatRng};

use super::arena::ArenaGrid;
use super::players::{
    ACTION_COUNT, ALL_ACTIONS, BOMB_FUSE_TICKS, BomberPlayer, DEFAULT_BLAST_RANGE, KnownBomb,
    in_blast_zone, is_safe_action, move_target, score_action, update_bombs, update_powerups,
};
use super::{BomberAction, GameEvent, GridPos};

// ── State encoding ─────────────────────────────────────────────

/// Number of feature tokens we encode per action decision.
/// We run one forward pass per candidate action (up to 7 actions).
const FEATURE_TOKENS: usize = 4;

/// Encode game state into a sequence of token indices for the transformer.
///
/// Tokens:
///   0: player x position (clamped to vocab)
///   1: player y position (clamped to vocab)
///   2: bomb-danger flag (0 = safe, 1 = in blast zone)
///   3: action index being evaluated
fn encode_state(
    pos: GridPos,
    in_danger: bool,
    action_idx: usize,
    vocab_size: usize,
) -> [usize; FEATURE_TOKENS] {
    let clamp = |v: i32| (v.rem_euclid(vocab_size as i32)) as usize;
    [
        clamp(pos.x),
        clamp(pos.y),
        if in_danger { 1 } else { 0 },
        action_idx % vocab_size,
    ]
}

// ── GatePlayer ─────────────────────────────────────────────────

/// Bomber player that uses [`InferenceRouter`] for routed inference.
///
/// Each `select_action()` call:
/// 1. Computes heuristic scores for all 7 actions.
/// 2. For each candidate action, encodes game state as token sequence.
/// 3. Routes forward pass through the TriggerGate tier system.
/// 4. Extracts the action score from the logit at the action's index.
/// 5. Blends routed score with heuristic (70/30) and picks the best.
pub struct GatePlayer {
    _id: u8,
    router: InferenceRouter,
    weights: TransformerWeights,
    ctx: ForwardContext,
    cache: MultiLayerKVCache,
    config: Config,
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    last_dir: Option<BomberAction>,
    /// Track how many inferences went through the router.
    total_routed: u64,
}

impl GatePlayer {
    /// Create a new GatePlayer with a micro Config model and default TriggerGate thresholds.
    pub fn new(id: u8) -> Self {
        let config = Config::micro();
        let mut rng = KatRng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let ctx = ForwardContext::new(&config);
        let cache = MultiLayerKVCache::new(&config);

        let gate_config = TriggerGateConfig::default();
        let router = InferenceRouter::new(gate_config, config.clone(), false, false);

        Self {
            _id: id,
            router,
            weights,
            ctx,
            cache,
            config,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
            total_routed: 0,
        }
    }

    /// Create with custom TriggerGateConfig (for testing tier thresholds).
    pub fn new_with_gate_config(id: u8, gate_config: TriggerGateConfig) -> Self {
        let config = Config::micro();
        let mut rng = KatRng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let ctx = ForwardContext::new(&config);
        let cache = MultiLayerKVCache::new(&config);

        let router = InferenceRouter::new(gate_config, config.clone(), false, false);

        Self {
            _id: id,
            router,
            weights,
            ctx,
            cache,
            config,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
            total_routed: 0,
        }
    }

    /// Return the router statistics snapshot.
    pub fn router_stats(&self) -> crate::inference_router::RouterStats {
        self.router.stats()
    }

    /// Total number of routed inferences since creation.
    pub fn total_routed(&self) -> u64 {
        self.total_routed
    }
}

impl BomberPlayer for GatePlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        _rng: &mut Rng,
    ) -> BomberAction {
        // Update tracked state
        update_bombs(&mut self.known_bombs, events);
        update_powerups(&mut self.known_powerups, events);

        let in_danger = in_blast_zone(pos, grid, &self.known_bombs);
        let bomb_positions: std::collections::HashSet<(i32, i32)> =
            self.known_bombs.iter().map(|(p, _, _)| *p).collect();

        let vocab_size = self.config.vocab_size;

        // Compute routed scores for each action
        let mut routed_scores = [0.0f32; ACTION_COUNT];
        for (i, action) in ALL_ACTIONS.iter().enumerate() {
            let is_move = matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            );

            // Wall collision filter
            if is_move {
                let target = move_target(action, pos);
                if !grid.is_walkable(target.x, target.y)
                    || bomb_positions.contains(&(target.x, target.y))
                {
                    routed_scores[i] = f32::NEG_INFINITY;
                    continue;
                }
            }

            // Skip unsafe actions when not in danger
            if !in_danger && !is_safe_action(action, grid, pos, &self.known_bombs) {
                routed_scores[i] = f32::NEG_INFINITY;
                continue;
            }

            // Encode state and route through InferenceRouter
            let tokens = encode_state(pos, in_danger, i, vocab_size);
            let mut logit_sum = 0.0f32;
            let mut logit_count = 0.0f32;

            for (seq_pos, &token) in tokens.iter().enumerate() {
                // Reset cache at block boundary
                if seq_pos >= self.config.block_size {
                    break;
                }
                let logits = self.router.forward(
                    &mut self.ctx,
                    &self.weights,
                    &mut self.cache,
                    token,
                    seq_pos,
                );
                // Use the action-indexed logit as the routed signal
                let idx = i.min(logits.len().saturating_sub(1));
                logit_sum += logits[idx];
                logit_count += 1.0;
                self.total_routed += 1;
            }

            // Average routed logits across the feature sequence
            routed_scores[i] = if logit_count > 0.0 {
                logit_sum / logit_count
            } else {
                0.0
            };

            // Reset KV cache for next action evaluation
            self.cache.reset();
        }

        // Compute heuristic scores
        let heuristic: [f32; ACTION_COUNT] = ALL_ACTIONS.map(|action| {
            score_action(
                &action,
                grid,
                pos,
                &self.known_bombs,
                &self.known_powerups,
                self.last_dir,
            )
        });

        // Blend: 70% heuristic + 30% routed (same ratio as LoraPlayer)
        let mut best = BomberAction::Wait;
        let mut best_score = f32::NEG_INFINITY;

        for (i, action) in ALL_ACTIONS.iter().enumerate() {
            if routed_scores[i] == f32::NEG_INFINITY || heuristic[i] == f32::NEG_INFINITY {
                continue;
            }
            let score = heuristic[i] * 0.7 + routed_scores[i] * 3.0;
            if score > best_score {
                best_score = score;
                best = *action;
            }
        }

        // Fallback to Wait if all actions were filtered
        if best_score == f32::NEG_INFINITY {
            best = BomberAction::Wait;
        }

        if matches!(
            best,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best);
        }
        if best == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }
        best
    }

    fn name(&self) -> &str {
        "Gate"
    }

    fn emoji(&self) -> &str {
        "🚀"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.last_dir = None;
        self.cache.reset();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
