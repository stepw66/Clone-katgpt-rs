# Plan 049: G-Zero Self-Play Distillation

> **Source:** [G-Zero: Self-Play for Open-Ended Generation from Zero Data](https://arxiv.org/pdf/2605.09959) — Huang et al., 2026
> **Status:** Planning
> **Depends on:** Plan 048 (Research Audit), Plan 032 (HL Infrastructure), Plan 036 (Review Metrics)

## Tasks

### Phase 1: Modelless (δ → existing HL infrastructure)

- [ ] T1: Implement `HintDelta` computation (log-prob shift between assisted/unassisted responses)
- [ ] T2: Implement `DeltaGatedAbsorbCompress` (use δ to gate heuristic promotion)
- [ ] T3: Implement `DeltaBanditPruner` (use δ as reward signal for bandit arms)
- [ ] T4: Implement `TemplateProposer` (rule-based query-hint generation, no neural model)
- [ ] T5: Benchmark modelless G-Zero vs existing HL AbsorbCompress

### Phase 2: Model-Based (δ → DPO/GRPO weight updates)

- [ ] T6: Implement `Proposer` trait + `GRPO` optimizer (gradient-based query-hint training)
- [ ] T7: Implement `LengthNormalizedDPO` loss for Generator training
- [ ] T8: Implement `DeltaFilter` + reward hacking defenses (lower-half retention, penalties)
- [ ] T9: Implement model-based `GZeroLoop` + wire into `SelfImprovingCycle`
- [ ] T10: Update README, .docs, references

---

## Paper Summary

G-Zero enables **verifier-free self-evolution** for open-ended (non-verifiable) domains by replacing external LLM judges with an **intrinsic signal** derived entirely from the model's own predictive distribution.

### Core Innovation: Hint-δ

Measures how much a hint shifts the Generator's distribution over its own unassisted response:

```text
δ(q, h, a_hard) = (1/T) Σ [log πG(at | q, h, a<t) − log πG(at | q, a<t)]
```

**Key property:** δ is large only when (1) the query is challenging AND (2) the hint carries information the Generator lacks. Two objectives compressed into one scalar — no external oracle needed.

### Why Two Paths?

Hint-δ is **architecture-agnostic** — it's a scalar like `ScreeningPruner::relevance()`. The paper uses it for gradient-based training (DPO/GRPO), but it fits equally well into our existing **gradient-free HL infrastructure**:

| Path | Mechanism | Updates | Cost | Strength |
|------|-----------|---------|------|----------|
| **Modelless** (Phase 1) | δ → AbsorbCompress + BanditPruner | Heuristics/rules | Low | Safe, fast, proven HL loop |
| **Model-based** (Phase 2) | δ → GRPO + DPO | LoRA weights | High | Stronger for open-ended domains |

Modelless makes the existing HL smarter with a better reward signal. Model-based adds neural self-play on top.

---

## Phase 1: Modelless G-Zero

### Design Principle

Don't train weights — use δ as a **smarter reward signal** for the existing AbsorbCompress and BanditPruner. The model generates log-probs (inference), but nothing is gradient-updated. All learning happens through heuristic promotion and bandit Q-values.

```text
┌─────────────────────────────────────────────────────────────┐
│  Modelless G-Zero Loop                                       │
│                                                              │
│  TemplateProposer ──(query, hint)──▸ Generator (frozen)      │
│        │                                │                     │
│        │                         log-probs with/without hint  │
│        │                                │                     │
│        │                           HintDelta                  │
│        │                                │                     │
│        │                    ┌───────────┴────────────┐       │
│        │                    ▼                        ▼       │
│        │          DeltaGatedAbsorbCompress   DeltaBanditPruner│
│        │          (promote high-δ arms       (δ as reward     │
│        │           to hard constraints)       for arm selection│
│        │                    │                        │       │
│        │                    └──────────┬─────────────┘       │
│        │                               ▼                     │
│        │                     TrialLog (JSONL)                 │
│        │                               │                     │
│        │                     RegressionSuite                  │
│        │                               │                     │
│        └─── next episode ◂─────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### T1: HintDelta Computation

**Shared foundation for both paths.** Requires per-token log-prob access from `transformer.rs` forward pass.

Currently `generate()` returns token indices only. Options:

- **Option A:** Add `generate_with_logprobs()` returning `Vec<(usize, f32)>`
- **Option B:** Add `logprobs()` method that recomputes for a given token sequence

```rust
/// Intrinsic reward: log-prob shift from hint conditioning
pub struct HintDelta {
    pub value: f32,       // per-token mean log-prob difference
    pub query: String,
    pub hint: String,
    pub unassisted_len: usize,
    pub assisted_len: usize,
}

impl HintDelta {
    /// δ(q, h, a_hard) = (1/T) Σ [log πG(at|q,h,a<t) − log πG(at|q,a<t)]
    pub fn compute(
        logprobs_unassisted: &[f32],
        logprobs_assisted: &[f32],
        query: &str,
        hint: &str,
    ) -> Self {
        let t = logprobs_unassisted.len().min(logprobs_assisted.len());
        let sum: f32 = (0..t)
            .map(|i| logprobs_assisted[i] - logprobs_unassisted[i])
            .sum();
        Self {
            value: sum / t as f32,
            query: query.to_string(),
            hint: hint.to_string(),
            unassisted_len: logprobs_unassisted.len(),
            assisted_len: logprobs_assisted.len(),
        }
    }
}
```

### T2: DeltaGatedAbsorbCompress

Use δ as the **absorb gate** — only promote arms where the hint made a meaningful difference. This replaces the current `ReviewMetrics` benefit-ratio gate with a signal derived from the model's own distributional dynamics.

```rust
/// AbsorbCompress gated by Hint-δ instead of ReviewMetrics
pub struct DeltaGatedAbsorbCompress<P: ScreeningPruner> {
    inner: AbsorbCompressLayer<P>,
    delta_threshold: f32,    // minimum δ to absorb (default: 0.02)
    delta_history: Vec<f32>, // rolling δ for each arm
}

impl<P: ScreeningPruner> AbsorbCompress for DeltaGatedAbsorbCompress<P> {
    fn absorb(&mut self, arm: usize, reward: f32) {
        // Only absorb if hint made meaningful difference
        let delta = self.delta_history.get(arm).copied().unwrap_or(0.0);
        if delta >= self.delta_threshold {
            self.inner.absorb(arm, reward);
        }
    }

    fn should_compress_gated(&self, metrics: Option<&ReviewMetrics>) -> bool {
        // Dual gate: δ must be meaningful AND reviewer must be net-positive
        let delta_ok = self.delta_history.iter().any(|&d| d >= self.delta_threshold);
        let review_ok = metrics
            .map(|m| m.benefit_ratio() >= 2.0)
            .unwrap_or(true);
        delta_ok && review_ok && self.inner.should_compress()
    }
}
```

**Why this is smarter than current AbsorbCompress:**
- Current: absorbs based on raw reward (did the environment say "good"?)
- New: absorbs based on δ (did the hint reveal a blind spot?)
- Blind spots = high-δ = the model doesn't already know this → promote to constraint

### T3: DeltaBanditPruner

Use δ as the **reward signal** for bandit arm selection. Instead of waiting for environment reward, use the model's own predictive shift as an immediate, dense reward.

```rust
/// BanditPruner using Hint-δ as reward signal
pub struct DeltaBanditPruner<P: ScreeningPruner> {
    inner: BanditPruner<P>,
    delta_weights: Vec<f32>,  // per-arm accumulated δ
}

impl<P: ScreeningPruner> ScreeningPruner for DeltaBanditPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

impl<P: ScreeningPruner> DeltaBanditPruner<P> {
    /// Feed δ signal as reward to bandit
    pub fn observe_delta(&mut self, arm: usize, delta: f32) {
        // Blend δ with environment reward: high δ = "this arm has blind spots"
        let reward = delta.max(0.0); // negative δ = hint hurt, ignore
        self.inner.observe(arm, reward);
        self.delta_weights[arm] += delta;
    }

    /// Which arms have highest accumulated blind-spot density?
    pub fn blind_spot_arms(&self, top_k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self.delta_weights.iter()
            .copied()
            .enumerate()
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed.into_iter().take(top_k).map(|(i, _)| i).collect()
    }
}
```

**Why this is smarter than current BanditPruner:**
- Current: reward comes from environment (compile success, game score)
- New: reward comes from the model itself (δ = how much the hint helped)
- Dense, immediate, no need to wait for episode completion
- Maps directly to `ScreeningPruner::relevance()` philosophy

### T4: TemplateProposer

Rule-based query-hint generator — no neural model, no LoRA, no gradient updates. Uses templates, bandit history, and TrialLog patterns to generate (query, hint) pairs.

```rust
/// Modelless proposer: template + bandit-driven query-hint generation
pub struct TemplateProposer {
    templates: Vec<QueryTemplate>,
    bandit_history: Arc<Mutex<Vec<BanditTrial>>>,
    delta_history: Arc<Mutex<Vec<HintDelta>>>,
    rng: SmallRng,
}

pub enum QueryTemplate {
    /// Task type templates from G-Zero paper Appendix A
    Writing { subtypes: Vec<&'static str> },     // email, story, essay, pitch, review
    Explanation { audiences: Vec<&'static str> }, // engineer, student, executive
    Advice { domains: Vec<&'static str> },        // career, travel, project
    Analysis { types: Vec<&'static str> },        // argument, text, product
    Coding { languages: Vec<&'static str> },      // function, debug, design
    Reasoning { difficulty: Vec<&'static str> },  // logic, math (≤1/6 of output)
}

pub struct GeneratedPair {
    pub query: String,
    pub hint: String,
    pub template_id: usize,
    pub blind_spot_target: Option<usize>, // arm with highest δ history
}

impl TemplateProposer {
    /// Generate a query-hint pair targeting the Generator's blind spots
    pub fn propose(&mut self) -> GeneratedPair {
        // Strategy: bias toward arms with highest accumulated δ
        let blind_arms = self.blind_spot_arms_from_history();

        if let Some(arm) = blind_arms.first() {
            // Target a known blind spot
            self.generate_targeted(*arm)
        } else {
            // Explore: random template with bandit-weighted selection
            let template_id = self.bandit_weighted_template();
            self.generate_from_template(template_id)
        }
    }

    /// Weight template selection toward categories with high historical δ
    fn bandit_weighted_template(&mut self) -> usize {
        // Use UCB1-style selection over template categories
        let n = self.templates.len() as f32;
        let total: f32 = self.templates.iter()
            .map(|t| t.pull_count() as f32)
            .sum();
        self.templates.iter()
            .map(|t| {
                let q = t.mean_delta();
                let explore = (2.0 * total.ln() / t.pull_count() as f32).sqrt();
                q + explore
            })
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
```

**Why template-based works:**
- G-Zero paper's Proposer prompt (Appendix A) is essentially a template with category sampling
- Our bandit + TrialLog already tracks which categories have blind spots
- TemplateProposer targets those blind spots without needing a neural model
- 0 GPU cost, instant generation, fully deterministic

### T5: Modelless Benchmark

Compare modelless G-Zero vs existing HL on Bomberman/Monopoly arenas:

```text
Benchmark Design:
  1. Baseline: Current HL (AbsorbCompress + BanditPruner + ReviewMetrics)
  2. Modelless G-Zero: DeltaGatedAbsorbCompress + DeltaBanditPruner + TemplateProposer
  3. Metrics: win rate, score, survival, episodes to convergence, blind-spot discovery rate

Hypothesis: Modelless G-Zero should converge faster because δ is a denser,
more informative signal than raw environment reward.
```

### Modelless Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| δ absorb threshold | 0.02 | Minimum δ to promote heuristic |
| δ reward floor | 0.0 | Negative δ = hint hurt, ignore |
| Template categories | 6 | Writing, Explanation, Advice, Analysis, Coding, Reasoning |
| Reasoning ratio | ≤1/6 | Cap math/logic queries (paper heuristic) |
| Bandit strategy | UCB1 | Same as existing BanditPruner |
| Exploration bonus | sqrt(2·ln(N)/n) | Standard UCB1 formula |

---

## Phase 2: Model-Based G-Zero

### Design Principle

Add gradient-based self-play on top of the modelless foundation. The modelless path already computes δ — now use it to train LoRA weights via DPO and GRPO.

```text
┌─────────────────────────────────────────────────────────────┐
│  Model-Based G-Zero Loop (builds on Phase 1)                 │
│                                                              │
│  Phase 1: Proposer Training (GRPO)                           │
│    NeuralProposer πP generates {(qi, hi)}                    │
│    → Generator answers unassisted                             │
│    → HintDelta reward                                        │
│    → GRPO update πP (gradient-based)                         │
│                                                              │
│  Phase 2: Generator Training (Length-Normalized DPO)          │
│    Frozen πP generates query-hints                            │
│    → Generator answers with/without hint                      │
│    → DeltaFilter (lower-half retention)                       │
│    → DPO update πG (hint-assisted=chosen, unassisted=rejected)│
│                                                              │
│  Phase 3: Deploy                                             │
│    → HotSwapPruner (zero-downtime adapter reload)             │
│    → Phase 1 modelless loop continues with improved model     │
└─────────────────────────────────────────────────────────────┘
```

### T6: Proposer Trait + GRPO

Replace TemplateProposer with a neural Proposer trained via GRPO.

```rust
/// Neural proposer trained via GRPO to target Generator blind spots
pub trait Proposer: Send + Sync {
    fn propose(&self, round: usize) -> Vec<ProposerOutput>;
    fn update_grpo(&mut self, rollouts: &[(String, String)], rewards: &[f32]);
}

pub struct ProposerOutput {
    pub query: String,
    pub hint: String,
}

/// GRPO: Group Relative Policy Optimization
/// No external value model — group baseline replaces it
pub struct GrpoConfig {
    pub group_size: usize,      // K rollouts per context (default: 16)
    pub clip_epsilon: f32,      // PPO-style clip (default: 0.2)
    pub learning_rate: f32,     // (default: 4e-5)
    pub batch_size: usize,      // (default: 128)
    pub max_steps: usize,       // Phase 1 steps (default: 6)
}

/// Advantage = (reward - μ) / σ within group
fn group_advantage(rewards: &[f32]) -> Vec<f32> {
    let mu: f32 = rewards.iter().copied().sum::<f32>() / rewards.len() as f32;
    let sigma: f32 = {
        let variance: f32 = rewards.iter()
            .map(|r| (r - mu).powi(2))
            .sum::<f32>() / rewards.len() as f32;
        variance.sqrt().max(1e-8)
    };
    rewards.iter().map(|r| (r - mu) / sigma).collect()
}
```

### T7: Length-Normalized DPO

Per-token mean log-ratio DPO loss — the key to avoiding length collapse.

```rust
/// L = -E[log σ(β·(r̄_θ(x,yw) - r̄_θ(x,yl)))]
/// where r̄_θ(x,y) = (1/|y|) · log(πθ(y|x) / πref(y|x))
pub struct LengthNormalizedDpo {
    pub beta: f32,  // KL penalty (default: 2.0)
}

pub struct PreferencePair {
    pub query: String,
    pub chosen: String,    // hint-assisted response
    pub rejected: String,  // unassisted response
    pub delta: f32,        // must be in lower half after filtering
}

impl LengthNormalizedDpo {
    pub fn loss(
        &self,
        policy_chosen: f32,    // mean log πθ(yw|x)
        policy_rejected: f32,  // mean log πθ(yl|x)
        ref_chosen: f32,       // mean log πref(yw|x)
        ref_rejected: f32,     // mean log πref(yl|x)
    ) -> f32 {
        let r_chosen = policy_chosen - ref_chosen;
        let r_rejected = policy_rejected - ref_rejected;
        -log_sigmoid(self.beta * (r_chosen - r_rejected))
    }
}
```

### T8: DeltaFilter + Reward Hacking Defenses

Multi-stage filtering pipeline for preference dataset curation:

```rust
/// Filter preference pairs for DPO training quality
pub struct DeltaFilter {
    pub delta_percentile: (f32, f32),  // (low, high) default: (0.0, 0.5)
    pub chosen_max_chars: usize,       // default: 10_000
    pub chosen_min_chars: usize,       // default: 100
    pub length_ratio_max: f32,         // default: 2.5
    pub zlib_threshold: f32,           // default: 0.15
    pub echo_prefix_len: usize,        // default: 30
}

impl DeltaFilter {
    pub fn filter(&self, pairs: &mut Vec<PreferencePair>, all_deltas: &[f32]) {
        // 1. Delta percentile filter
        let (p_low, p_high) = self.delta_percentile;
        let d_low = percentile(all_deltas, p_low);
        let d_high = percentile(all_deltas, p_high);
        pairs.retain(|p| p.delta >= d_low && p.delta <= d_high);

        // 2. Length quality heuristics
        pairs.retain(|p| {
            let len = p.chosen.len();
            len >= self.chosen_min_chars
                && len <= self.chosen_max_chars
                && (len as f32 / p.rejected.len().max(1) as f32) <= self.length_ratio_max
        });

        // 3. Repetition filter (zlib compression ratio)
        pairs.retain(|p| zlib_ratio(&p.chosen) >= self.zlib_threshold);

        // 4. Prompt echo filter
        pairs.retain(|p| !p.chosen.starts_with(&p.query[..self.echo_prefix_len.min(p.query.len())]));

        // 5. Template leakage filter
        pairs.retain(|p| !contains_role_markers(&p.chosen));
    }
}

/// Penalties for GRPO reward
fn length_penalty(hint: &str, target: usize, lambda: f32) -> f32 {
    let excess = hint.len() as f32 - target as f32;
    if excess > 0.0 { lambda * excess / 100.0 } else { 0.0 }
}

fn bleu_duplication_penalty(rollout_question: &str, batch: &[String]) -> f32 {
    let cluster_size = batch.iter()
        .filter(|q| sentence_bleu(rollout_question, q) > 0.5)
        .count();
    cluster_size as f32 / batch.len() as f32
}

/// Total reward: r(q,h) = δ − P_length − P_BLEU
fn grpo_reward(delta: f32, hint: &str, question: &str, batch: &[String]) -> f32 {
    delta - length_penalty(hint, 200, 0.03) - bleu_duplication_penalty(question, batch)
}
```

### T9: Model-Based GZeroLoop + SelfImprovingCycle

```rust
/// One round of model-based G-Zero co-evolutionary loop
pub struct GZeroRound {
    pub round: usize,
    pub proposer_steps: usize,      // Phase 1 GRPO steps (default: 6)
    pub proposer_batch: usize,      // (default: 128)
    pub proposer_group: usize,      // GRPO group size (default: 16)
    pub dpo_max_steps: usize,       // Phase 2 DPO steps (default: 50)
    pub dpo_batch: usize,           // (default: 8)
    pub delta_cutoff: (f32, f32),   // percentile range (default: [0.0, 0.5])
    pub questions_per_round: usize, // (default: 2000)
}
```

Wire into anyrag's `SelfImprovingCycle`:

```text
SelfImprovingCycle {
  Collecting → ReadyToSynthesize → ...
    ├── Path A (existing):  Export JSONL → riir-burner LoRA SFT      (modelless HL)
    ├── Path B (Phase 1):   DeltaGatedAbsorbCompress + DeltaBanditPruner (smarter modelless)
    └── Path C (Phase 2):   Proposer↔Generator self-play → DPO LoRA  (model-based G-Zero)
}
```

All three paths feed into `HotSwapPruner` for zero-downtime model updates.

---

## Mapping to Existing Infrastructure

### Direct Maps (exists, reuse)

| G-Zero Concept | Our Component | Path |
|----------------|---------------|------|
| Episode history | `TrialLog` (JSONL) | Both |
| Reward hacking defense | `ReviewMetrics` benefit-ratio | Both |
| Hot-swap updated model | `HotSwapPruner` | Both |
| Regression safety | `RegressionSuite` | Both |
| LoRA training | `riir-burner` (rank 32) | Model-based |
| Domain inference budget | `InferenceBudget` (β) | Both |
| δ reward signal | `ScreeningPruner::relevance()` | Both (needs log-prob access) |
| Bandit exploration | `BanditPruner` (UCB1/Thompson) | Modelless (enhanced with δ) |
| Absorb-compress learning | `AbsorbCompressLayer` | Modelless (gated by δ) |

### New Components

| Component | Phase | Description |
|-----------|-------|-------------|
| `HintDelta` | 1 | Log-prob difference computation |
| `DeltaGatedAbsorbCompress` | 1 | Absorb only when δ reveals blind spot |
| `DeltaBanditPruner` | 1 | δ as dense reward signal |
| `TemplateProposer` | 1 | Rule-based query-hint generation |
| `Proposer` trait | 2 | Neural proposer with GRPO |
| `GrpoConfig` | 2 | Group-relative policy optimization |
| `LengthNormalizedDpo` | 2 | Per-token mean log-ratio DPO loss |
| `DeltaFilter` | 2 | Lower-half δ retention + quality heuristics |
| `GZeroRound` | 2 | Round orchestration |

---

## Model-Based Hyperparameters (from paper)

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank | 32 | Match existing riir-burner config |
| Proposer batch size | 128 | |
| Proposer group size (GRPO K) | 16 | |
| Proposer steps | 6 | Phase 1 |
| Proposer learning rate | 4e-5 | |
| Proposer max_tokens | 8,192 | |
| Hint length target | 200 chars | Penalty threshold |
| Hint length penalty λ | 0.03 | |
| BLEU cluster merge threshold | 0.5 | Average linkage |
| Questions per round | 2,000 | |
| Generator sampling temperature | 0.7 | |
| Generator max_tokens | 16,384 | |
| δ cutoff | [0, 50] percentile | Lower half retention |
| chosen_max_chars | 10,000 | Quality heuristic |
| chosen_min_chars | 100 | Quality heuristic |
| chosen/rejected ratio max | 2.5 | Length inflation filter |
| zlib compression threshold | 0.15 | Repetition filter |
| DPO β | 2.0 | KL penalty |
| DPO learning rate | 1e-5 | |
| DPO max steps | 50 | |
| DPO batch size | 8 | |
| DPO log-ratio normalization | length-normalized | Critical for stability |

---

## Relationship to Existing Work

| Paper | Our Status | G-Zero Relation |
|-------|-----------|-----------------|
| R-Zero (arXiv:2508.05004) | Referenced in README | Predecessor; R-Zero = verifiable only, G-Zero = open-ended |
| DPO (Rafailov et al., 2023) | Not implemented | T7 implements length-normalized variant |
| GRPO (DeepSeekMath, 2024) | Not implemented | T6 implements for Proposer training |
| HL (Learning Beyond Gradients) | ✅ Fully implemented | Phase 1 makes HL smarter with δ; Phase 2 adds gradient-based on top |
| Self-evolving agents (Xiang et al.) | Partial via SelfImprovingCycle | G-Zero provides concrete self-play mechanism |
| Model collapse (Shumailov et al.) | Mitigated via TrialLog diversity | G-Zero's BLEU penalty + δ filter address same concern |
| Plan 025 Bandit results | ✅ Model-based +12.1% reward | δ could improve both model-based and modelless bandits |

---

## Risk Assessment

| Risk | Mitigation | Phase |
|------|------------|-------|
| Log-prob access in transformer | Need forward pass modification — non-trivial but shared | 1 |
| TemplateProposer too simplistic | Bandit-weighted templates + TrialLog patterns; upgrade to neural later | 1 |
| δ threshold too aggressive/lenient | Make configurable; benchmark sweep | 1 |
| R3 round collapse (paper reports this) | Monitor response lengths; min_chars circuit breaker | 2 |
| Proposer reward hacking (verbose hints) | Length penalty + BLEU penalty (paper-proven) | 2 |
| Length collapse under DPO | Length-normalized loss + chosen_min_chars filter | 2 |
| DPO off-manifold drift | Lower-half δ filter + KL penalty (β=2.0) | 2 |
| Compute cost (~$2000 for paper runs) | Phase 1 is free; Phase 2 start with small models | 2 |
| No GRPO/DPO in Rust ecosystem | Greenfield implementation; derive from paper equations | 2 |

---

## Success Criteria

### Phase 1 (Modelless)

1. **Hint-δ computation** produces meaningful signal (positive for informative hints, near-zero for useless)
2. **DeltaGatedAbsorbCompress** converges faster than ReviewMetrics-gated AbsorbCompress
3. **DeltaBanditPruner** discovers blind spots that raw-reward BanditPruner misses
4. **TemplateProposer** generates non-trivial queries across ≥4 categories
5. **Benchmark**: modelless G-Zero ≥ existing HL on Bomberman/Monopoly arenas

### Phase 2 (Model-Based)

6. **Co-evolutionary loop** completes ≥2 rounds without collapse
7. **Preference dataset** passes all DeltaFilter quality heuristics
8. **DPO-trained model** shows improvement on at least 1 metric (chat OR reasoning)
9. **Structural transfer** confirmed: non-verifiable training → verifiable domain improvement
10. **Comparison**: model-based G-Zero ≥ modelless G-Zero ≥ existing HL