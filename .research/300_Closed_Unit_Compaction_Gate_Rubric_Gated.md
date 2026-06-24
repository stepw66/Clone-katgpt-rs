# Research 300: Closed-Unit Compaction Gate — Rubric-Gated Trajectory Compaction (SelfCompact)

> **Source:** *Self-Compacting Language Model Agents* (SelfCompact), Li et al., Johns Hopkins + Apple, arXiv:2606.23525, 22 Jun 2026
> **Date:** 2026-06-25
> **Status:** Active
> **Related Research:** 175 (ThoughtFold — HOW to fold, complementary), 233 (Attention Matching — KV compaction mechanism), 255 (CLR — skip-if-correct suppression fuse), 270 (ICT Branching — C1 closed-unit signal), 281 (Salience Tri-Gate — per-tick emit, complementary granularity), 287 (Claim Rubric Runtime — host for rubric), 296 (Stokes DEC — C3 progress divergence)
> **Related Plans:** 195 (chain_fold), 238 (MUX-Latent wire patch — AdaptiveTraceCompactor), 271 (attention_matching), 284 (CLR), 294 (ict_branching), 307 (claim_rubric_runtime), 320 (this note's plan), 251/314 (DEC wrappers)
> **Cross-ref (riir-ai):** Research 155 (per-NPC sub-goal compaction guide — the selling point)
> **Cross-ref (riir-neuron-db):** Research 007 (`can_freeze` as CUCG instance — cross-domain isomorphism)
> **Classification:** Public

---

## TL;DR

SelfCompact introduces a **rubric-gated, inference-time, training-free context compaction** scaffold for long agent trajectories. It pairs (i) a summarization tool the model invokes itself with (ii) a lightweight **cite-verbatim rubric** that decides *when* compaction is structurally safe (closed reasoning unit, summarizable, progressed, not stuck). The rubric fires `COMPRESS` iff `C1 ∧ C2 ∧ C3 ∧ ¬N1`; otherwise it reverts and the trajectory continues unchanged. KV cache is reused across probe and summarizer (append-only, no re-prefill). Across 6 benchmarks / 7 models it matches or exceeds fixed-interval summarization at 30–70% lower token cost.

**Distilled for katgpt-rs (modelless, inference-time):** a generic **Closed-Unit Compaction Gate (CUCG)** primitive — a Boolean-over-cite-able-rubric compaction trigger with (a) pluggable predicates, (b) append-only cache-reuse probe protocol, (c) token-pct backstop, (d) skip-if-reliable suppression fuse, (e) probe-revert on `CONTINUE`. The rubric's predicates are sigmoid gates on latent trajectory features (coherence stability, intrinsic-rank collapse, positive divergence, novelty rate) — **never softmax**, per AGENTS.md.

**The Super-GOAT punchline:** the paper's `C1/C2/C3/N1` rubric and our already-shipped `can_freeze` shard gate (`riir-neuron-db/src/phase_gate.rs`, Plan 002) are **structurally isomorphic closed-unit compaction gates** — one for trajectories, one for shards. They converged independently; unifying them as instances of one primitive (CUCG) is the force multiplier. We have the shard-side freeze gate shipped and GOAT-validated; the paper gives us the trajectory-side counterpart for free.

---

## 1. Paper Core Findings (verified by full read)

### 1.1 The two-element scaffold (§3)

Two elements are needed *together*:

1. **Inline summarization tool `S : (x, y_{1:t}) → ỹ`** — the model invokes it; the summarizer is the same model π; no external verifier.
2. **Lightweight rubric `P_R`** appended at probe intervals — enumerates a small set of conditions, each requiring **verbatim evidence quoted from the trajectory**, so π can check them locally. Returns `r_t ∈ {COMPRESS, CONTINUE}`.

**Ablation (§4, Table 5/6):** tool-alone (no rubric) collapses to ~fixed-interval performance (41.0% vs 46.4% full SelfCompact on search; 40.9% vs 45.5% on math). The rubric is what makes the retained context reliable.

### 1.2 The math rubric (Appendix A) — `Q1 ∨ (Q2 ∧ Q3)`

- **Q1 ANSWER**: latest round states a specific final answer (`\boxed{}` or "Final Answer:") — quote verbatim.
- **Q2 STUCK**: last 2 rounds added no non-trivial fact (only paraphrases / abandoned attempts) — name the rounds.
- **Q3 HAS-NEXT**: can write the exact next step (case split, substitution, lemma to prove) — write the step.
- **Fire rule:** `COMPRESS iff Q1=Y ∨ (Q2=Y ∧ Q3=Y)`.
  - Branch A (`Q1=Y`): lock-in re-prompt preserving the boxed answer.
  - Branch B (`Q2=Y ∧ Q3=Y`): reset preserving the named next step.

### 1.3 The search rubric (Appendix B) — `C1 ∧ C2 ∧ C3 ∧ ¬N1`

- **C1 CLOSED-UNIT**: most recent assistant message is a closed unit (completed tool call whose result is visible, OR completed sub-analysis with clear stopping point). NOT mid-sentence. Quote closing/open fragment.
- **C2 SUMMARIZABLE**: can write 3–5 essential facts (each a single concrete statement: name, date, URL, quoted claim) with verbatim citations. Answer N if value is dispersed across many small inferences (dead-end queries, negative results) that would be lost.
- **C3 PROGRESS**: since last compression, obtained a new concrete fact OR refined the sub-question. Name it.
- **N1 STUCK**: at least 3 of last 4 search queries returned no new URL/fact. If yes, name 1 strategy not yet tried.
- **Fire rule:** `COMPRESS iff C1 ∧ C2 ∧ C3 ∧ ¬N1`.
- **Token-pct backstop:** forces `COMPRESS` once prompt crosses `0.30 · ctx_window` regardless of rubric (this is the only fixed-threshold escape hatch — kept as a safety net).

### 1.4 KV-cache reuse — the perf claim (§3, Appendix C/D)

Both probe and summarizer **append** to `(x, y_{1:t})`, so the running KV cache is preserved across the call:

- **Probe** contributes only its own ~60-token verdict to `N_out`. The `O(L²)` re-prefill that a naive re-encode would incur is avoided.
- **Summarizer** pays `O(L·ℓ)` to generate `ỹ` plus a one-time `O(ℓ²)` prefill of `(x, ỹ)`.
- **Post-compaction:** every later call attends to `ỹ` (length `ℓ ≈ 1–3k`) instead of `y_{1:t}` (length `L ≈ 50–100k`) → **20–80× shrinkage** of the post-compact prompt.
- **Break-even:** compaction wins iff `L/ℓ > 10`. Search summarizer reaches 20–80×, so the trade is comfortably favorable.

### 1.5 Empirical headline (§4)

| Domain | Metric | SelfCompact vs no-compaction | SelfCompact vs fixed-interval |
|---|---|---|---|
| Competition math (IMO / HMMT) | accuracy | +16.4 / +10.0 / +18.1 pp (Qwen3.5-9B) | wins 11/12 cells at matched token budget |
| Agentic search (BrowseComp(+)/DeepSearchQA) | accuracy | +5 to +9 pp | +up to 6.3 pp over fixed-interval |
| Agentic search | per-question cost | — | **−30% to −70%** vs no-compaction |

### 1.6 Skip-if-correct oracle (§4.1, Table 3)

An oracle that suppresses summarization whenever the current answer is correct, but otherwise follows the same fixed schedule, achieves **52.9%** on IMO-Answerbench — **+11.5 over fixed-interval** and +14.0 over baseline. This is a strict subset of a fully adaptive policy. **Headroom exists for a "skip-if-reliable" suppression fuse** layered on top of the rubric.

### 1.7 What's NOT in the paper (and matters for us)

- **No latent-space reframing.** The paper stays entirely in token space. The C1/C2/C3/N1 predicates are LLM-judged from verbatim quotes — not computed from latent features.
- **No crowd-scale variant.** Single agent, single trajectory. No notion of thousands of concurrent agents each with their own compaction gate.
- **No commitment / audit trail.** The rubric verdict is ephemeral; nothing is bit-committed across nodes.
- **No cross-domain unification.** The paper doesn't notice that its rubric is structurally the same gate as a "is this shard done consolidating?" freeze gate.

These four gaps are exactly where our codebase's Super-GOAT reframing lands.

---

## 2. Distillation

### 2.1 What's already in katgpt-rs (verified by code + notes grep in this session)

| Paper mechanism | Our shipped equivalent | Status |
|---|---|---|
| Fixed-interval compaction trigger | `OnlineCompactor::trigger_threshold()` (`src/attn_match/online.rs:86`) — fires when `current_pos >= phys_budget` | ✅ shipped — this is the **baseline the paper beats** |
| Entropy-EMA-gated compaction trigger | `AdaptiveTraceCompactor` (`src/attn_match/adaptive_cot.rs`, Plan 238 wire-patch) — compacts when `ema_entropy < theta_low`; preserves when `> theta_high`; `FrequencyBandit` tunes thresholds | ⚠️ **scalar entropy signal, not rubric**; KV-level not trajectory-summary-level; fires on LOW entropy (compressible stretches) — **orthogonal axis** to closed-unit detection |
| Discrete step pruning (the HOW) | `chain_fold` feature (Plan 195 ThoughtFold) — attention-importance ranking + binary search + KV rollback-replay | ⚠️ **different mechanism** — discrete pruning vs LLM summarization; attention scores vs verbatim-citation rubric |
| Per-tick emit gate | `SalienceTriGate` (`src/salience/gate.rs`, Plan 303) — `Speak/Silent/Delegate` per tick | ✅ **different granularity** — per-tick emit, not per-interval compaction; `Delegate` is async handoff, not context compaction |
| Per-completion reliability | CLR `(mean_m v_k,m)^M` (Plan 284, default-on) — picks most reliable completion from K | ✅ **different axis** — reliability scoring, not compaction trigger; per-completion not per-trajectory |
| Claim evidence ladder | `claim_rubric::checklist` (`src/claim_rubric/checklist.rs`, Plan 307) — L1/L2/L3 evidence items as code | ✅ **host for the rubric** — provides the cite-verbatim vocabulary SelfCompact's C1/C2/C3/N1 needs |
| Branching-point detector | ICT `BranchingDetector` (Plan 294, R270) — JS-divergence-to-group-mean top-k% | ✅ **C1 signal source** — "branching moment passed" = "closed-unit reached" |
| Entropy-driven verification tier | `llmexec_guard` (Plan 223) — `sigmoid(-steepness·(H1−0.5)+depth_bonus)` | ✅ **different signal** — scalar entropy gate, not multi-predicate rubric |
| Rubric-vector gated absorb-compress | `RubricGatedAbsorbCompress` (Plan 071) — per-criterion gap targeting for DDTree absorb | ✅ **different domain** — arena-player pattern learning, not trajectory context compaction |

**Key finding from the grep:** there is **no shipped primitive that combines** (a) multi-predicate cite-verbatim rubric + (b) Boolean fire rule + (c) trajectory-summary compaction + (d) cache-reuse probe protocol + (e) token-pct backstop + (f) skip-if-reliable suppression. Each individual piece has a cousin; the combination is novel.

### 2.2 What's NOT in katgpt-rs (the gaps — what CUCG adds)

1. **`ClosedUnitCompactionGate<R>`** — a generic gate parameterized over a rubric `R: Rubric`. Zero-allocation hot path. Decides `COMPRESS | CONTINUE` per probe.
2. **`Rubric` trait** — `fn evaluate(&self, trajectory: &[Token], scratch: &mut Scratch) -> RubricVerdict` returning `Y(quote)/N(reason)` per predicate. Hosts C1/C2/C3/N1 (search), Q1/Q2/Q3 (math), or domain-specific predicates (NPC sub-goal, shard freeze).
3. **Boolean fire rule** — `COMPRESS iff C1 ∧ C2 ∧ C3 ∧ ¬N1` (conjunction + negation). Composable: `FireRule::And(...)`, `FireRule::Or(...)`, `FireRule::Not(...)`.
4. **Append-only probe protocol** — `probe(trajectory, rubric_prompt) -> Verdict` that **preserves the KV cache** of `y_{1:t}` and only pays prefill on the appended instruction. The probe verdict is **reverted from the rolling cache on `CONTINUE`** — does not pollute subsequent generation.
5. **Hard-reset summarizer protocol** — `summarize(trajectory) -> Summary` that replaces `y_{1:t}` with `ỹ`, then resumes decoding from `(x, ỹ)`.
6. **Token-pct backstop** — `Backstop::TokenPct(0.30)` forces `COMPRESS` once prompt length crosses the threshold regardless of rubric. Pluggable: `Backstop::None`, `Backstop::TokenPct(p)`, `Backstop::Never` (rubric-only).
7. **Skip-if-reliable suppression fuse** — composes with CLR (Plan 284): if `clr_vote(recent_completions) > τ_reliable`, suppress `COMPRESS` even if the rubric fires. This is the paper's §4.1 oracle, made real and modelless.
8. **`CompactionAuditRecord`** — the rubric verdict, evidence quotes, fire rule evaluation, and resulting decision as a single deterministic record. **Crosses the sync boundary as raw** (per AGENTS.md latent-vs-raw rule) — two honest nodes must agree on whether compaction fired and why.

### 2.3 The modelless primitive (sketch)

```rust
/// Closed-Unit Compaction Gate — rubric-gated, training-free context compaction.
///
/// Source: SelfCompact (arXiv:2606.23525). Generic over the rubric `R` so the
/// same gate hosts trajectory compaction (paper's C1/C2/C3/N1), NPC sub-goal
/// memory compaction (riir-ai runtime), and shard consolidation freeze
/// (riir-neuron-db `can_freeze` — see Research 007 cross-ref).
///
/// **Zero-allocation on the hot path**: all state is fixed-size; `evaluate`
/// and `decide` perform no heap allocation.
///
/// **Sigmoid, never softmax** (per AGENTS.md): each predicate's confidence is
/// a scalar in `[0,1]` derived from a sigmoid projection; the fire rule is a
/// Boolean combination, not a 3-way softmax over {COMPRESS, CONTINUE, DEFER}.
pub struct ClosedUnitCompactionGate<R: Rubric> {
    rubric: R,
    fire_rule: FireRule,
    backstop: Backstop,
    /// Suppression fuse: if `Some(clr_handle)`, suppress COMPRESS when
    /// CLR reliability vote on recent completions exceeds `reliable_threshold`.
    /// Implements the paper's §4.1 skip-if-correct oracle, modellessly.
    skip_if_reliable: Option<f32>,
    /// Probe interval in tokens (paper's `N`). Probes at `t ∈ {N, 2N, 3N, ...}`.
    probe_interval_tokens: usize,
}

pub trait Rubric {
    /// Evaluate the rubric against the trajectory. Each predicate returns
    /// `Y(quote)` with verbatim evidence, or `N(reason)` explaining the miss.
    /// Quotes are required — they make the verdict auditable.
    fn evaluate(
        &self,
        trajectory_prefix: &[u8],   // y_{1:t}, borrowed
        scratch: &mut RubricScratch,
    ) -> RubricVerdict;
}

pub struct RubricVerdict {
    /// Per-predicate results. Order is the rubric's canonical order.
    /// Stored as a fixed-size array (rubric arity is compile-time known).
    pub predicates: [PredicateResult; R::ARITY],
}

pub enum PredicateResult {
    Yes { quote_start: u32, quote_len: u16 },  // cite-verbatim, into trajectory
    No { reason: PredicateReason },
}

pub enum FireRule {
    /// COMPRESS iff all named predicates are Yes.
    And(u8),                                    // bitmask over predicate indices
    /// COMPRESS iff any named predicate is Yes.
    Or(u8),
    /// COMPRESS iff the named predicate is No.
    Not(u8),
    /// Compose: paper's math rule is `Or(Q1, And(Q2, Q3))`.
    Box(Box<FireRule>, Box<FireRule>),
}

pub enum CompactionDecision {
    /// Compaction is structurally safe. Caller should run summarizer,
    /// then hard-reset to `(x, ỹ)`.
    Compress { audit: CompactionAuditRecord },
    /// Continue from `(x, y_{1:t})` unchanged. Probe verdict is reverted
    /// from the rolling cache — does not pollute subsequent generation.
    Continue { audit: CompactionAuditRecord },
    /// Token-pct backstop forced the decision. Rubric verdict may disagree.
    Forced { audit: CompactionAuditRecord },
}

/// Deterministic, bit-identical audit record. Crosses sync boundary as raw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactionAuditRecord {
    pub trajectory_len: u32,
    pub predicates: [PredicateResult; R::ARITY],
    pub fire_rule_eval: FireRuleEval,
    pub backstop_triggered: bool,
    pub skip_if_reliable_triggered: bool,
    pub decision: u8,  // 0=Continue, 1=Compress, 2=Forced
}
```

### 2.4 Latent reframing (mandatory per skill — this is a latent-to-latent op)

The C1/C2/C3/N1 predicates are **scalars derived from latent trajectory features** via dot-product + sigmoid projections (per AGENTS.md — **never softmax**). The fire rule is a Boolean combination of these scalar gates:

| Paper predicate | Latent feature (codebase vocabulary) | Sigmoid projection |
|---|---|---|
| C1 closed-unit | coherence stability (from `latent_functor/quality_gate.rs`) | `σ(β_c1 · (coherence − τ_c1))` |
| C2 summarizable | negative intrinsic-rank (from `subspace_phase_gate`) | `σ(β_c2 · (d_eff − τ_c2))` inverted |
| C3 progress | positive divergence since last summary (DEC `codifferential` on belief cochain) | `σ(β_c3 · (div_since_last − τ_c3))` |
| N1 stuck | negative novelty rate (from `cgsp_runtime` derivative curiosity, or ICT `collision_purity`) | `σ(β_n1 · (novelty − τ_n1))` inverted |
| Fire rule | Boolean conjunction | `C1 ∧ C2 ∧ C3 ∧ ¬N1` |

The verbatim-quote requirement (paper's anti-rationalization device) is **preserved as an audit-trail obligation**: even when the predicate is computed from latent features, the gate records the trajectory span `[quote_start, quote_start+quote_len]` that grounded the decision. This is what makes the gate auditable — and what lets the audit cross the sync boundary as raw without leaking latent embeddings.

**This reframing is what makes CUCG a Super-GOAT rather than a prompt-engineering trick.** The paper's contribution is empirical (a scaffold that works); our contribution is the primitive that makes the scaffold's mechanism generic, latent-aware, and cross-domain.

### 2.5 Connection map (force multiplier — 10+ existing pillars)

| Pillar | Connection | Multiplier effect |
|---|---|---|
| `AdaptiveTraceCompactor` (Plan 238) | C2 signal source (entropy-EMA) | CUCG adds structural safety to entropy-driven compaction |
| `chain_fold` (Plan 195) | The HOW after CUCG's WHEN | CUCG decides WHEN to compact; chain_fold could be the HOW (or LLM summarizer) |
| `SalienceTriGate` (Plan 303) | Compose per-tick emit with per-interval compact | Per-tick speak/silent + per-interval context-compact = complete meta-cognitive budget |
| CLR (Plan 284) | Skip-if-reliable suppression fuse | The paper's §4.1 oracle becomes a one-line CLR composition |
| Claim Rubric Runtime (Plan 307) | Host for the rubric predicates | C1/C2/C3/N1 land as `EvidenceItem`s in the L2 evidence ladder |
| ICT Branching (Plan 294) | C1 closed-unit signal | "branching moment passed" = "decision made" = closed-unit |
| `latent_functor/quality_gate` | C1 coherence-stability signal | Coherence-driven compaction trigger (re-estimation peer) |
| DEC `codifferential` (Plan 251/314) | C3 progress divergence signal | "positive divergence since last summary" = mass creation on belief cochain |
| `cgsp_runtime` curiosity | N1 novelty-rate signal | Curiosity-driven stuck detection |
| `can_freeze` (riir-neuron-db Plan 002) | **Cross-domain isomorphic instance** | Trajectory compaction and shard freeze are the same primitive |

---

## 3. Verdict

### Super-GOAT.

**One-line reasoning:** CUCG is a novel mechanism (no shipped primitive combines rubric + Boolean fire rule + cache-reuse probe + token-pct backstop + skip-if-reliable), it creates a new capability class (compaction-as-meta-cognition at structurally-safe moments, not at fixed thresholds), it is a product selling point ("agents compact context without wiping verified facts — emergent social behavior persists across thousands of NPCs"), and it force-multiplies ≥10 existing pillars — most strikingly by recognizing that `can_freeze` (shipped) and SelfCompact's rubric (paper) are isomorphic instances of the same primitive.

### Tiers check

- **Super-GOAT criteria:** ✅ Novel mechanism (4-predicate rubric + Boolean fire rule + cache-reuse protocol combination is not shipped), ✅ new capability class (structural-safety-gated compaction vs resource-pressure-gated), ✅ product selling point (game AI: per-NPC sub-goal memory compaction; LLM agents: no mid-derivation fact wipes), ✅ force multiplier (10+ pillars, including the cross-domain `can_freeze` isomorphism).

### Why not GOAT

GOAT would require CUCG to be a provable gain over an existing approach but not a new class. CUCG **is** a new class: existing compaction triggers (entropy, token-count, attention-importance) all fire on **resource pressure or local compressibility**; CUCG fires on **structural safety** (closed-unit + summarizable + progress + not-stuck). The paper's Figure 1 demonstrates the capability difference concretely (4 verified facts preserved vs wiped). For our codebase, the cross-domain unification with `can_freeze` is itself a new-class claim — we did not previously recognize shard freeze and trajectory compaction as the same operation.

### Why not Gain

Gain would be an incremental improvement. The skip-if-reliable fuse alone is a Gain; the full CUCG with its cross-domain unification and latent reframing is a Super-GOAT.

---

## 4. Novelty gate — all 4 YES

### Q1: No prior art? — **YES**

Three-layer check (notes + code + vocabulary translation) performed in this session:

**Paper-vocabulary grep** (`compaction|compact|summariz|rubric|closed.unit|mid.derivation|verbatim|self.compact|context.rot`):
- Hits in skill SKILL.md (workflow vocabulary — not a shipped primitive).
- Hits in `.benchmarks/007_ropd_rubric_modelless.md` — `RubricGatedAbsorbCompress` (Plan 071). **Different domain** — arena-player pattern absorb, not trajectory context compaction. Confirmed by reading Plan 071 T5: gates DDTree absorb on rubric-vector gaps, not on trajectory summary decision.
- Hits in `.benchmarks/059_sink_aware_goat.md` — `summarize_layer_sinks` (Sink-Aware Attention, Plan 287). **Different domain** — per-layer sink classification summary, not trajectory compaction.
- No hits for `closed.unit.compaction`, `cucg`, `ClosedUnit`, `RubricGate`, `when_to_compact`, `compaction_gate`.

**Codebase-vocabulary grep** (`AdaptiveTraceCompactor|adaptive_trace|observe_entropy|threshold_trigger|fixed.interval` + `salience_tri_gate|chain_fold|thought_fold|llmexec_guard`):
- `AdaptiveTraceCompactor` (Plan 238 wire-patch) — closest cousin. **Scalar entropy gate**, KV-level, fires on LOW entropy. Orthogonal axis (compressibility vs structural safety).
- `OnlineCompactor::trigger_threshold()` — fixed-position trigger. This is the **baseline the paper beats**.
- `chain_fold` (Plan 195) — discrete step pruning, attention-importance-ranked. **Different mechanism** (pruning vs summarization).
- `SalienceTriGate` (Plan 303) — per-tick emit, not per-interval compact.
- `RubricGatedAbsorbCompress` (Plan 071) — rubric-gated absorb for arena players, not trajectory compaction.

**Super-GOAT factory modules explicitly listed:**
- `katgpt-rs/crates/katgpt-core/src/sense/` — HLA belief state (latent substrate for C1/C3, not a compaction gate).
- `riir-ai/crates/riir-engine/src/latent_functor/` — `quality_gate.rs` is a coherence-driven gate but for functor application, not trajectory compaction; `reestimation.rs` is a coherence-decay trigger, structurally similar but for functor cycles not context compaction.
- `riir-ai/crates/riir-engine/src/hla/` — per-NPC latent state (substrate, not gate).
- `riir-ai/crates/riir-engine/src/cgsp_runtime/` — curiosity signals (C3/N1 source, not gate).
- `riir-neuron-db/src/phase_gate.rs` — **`can_freeze` is the cross-domain isomorphic instance** (see Research 007 cross-ref). Recognizing it as CUCG-for-shards is the novelty; the freeze gate itself is shipped, but its unification with trajectory compaction is not.
- `riir-chain/src/encoding/latcal*.rs` — LatCal commitment (sync-boundary bridge — relevant for the audit trail, not the gate itself).
- `katgpt-rs/crates/katgpt-core/src/dec/` — DEC operators (C3 divergence signal source, not gate).

**Honest prior-art accounting:** The mechanism "multi-predicate rubric + Boolean fire rule + trajectory-summary compaction + cache-reuse probe + token-pct backstop + skip-if-reliable" is **not shipped in any combination**. Each individual piece has a cousin; the combination and its cross-domain unification are novel. → **YES**.

### Q2: New class of behavior? — **YES**

Existing compaction triggers fire on:
- **Resource pressure** (`OnlineCompactor` — position ≥ budget)
- **Local compressibility** (`AdaptiveTraceCompactor` — low entropy EMA)
- **Discrete pruning targets** (`chain_fold` — attention-importance rank)

CUCG fires on **structural safety** (closed-unit + summarizable + progress + not-stuck). This is a new axis: "is the trajectory at a safe summary point?" rather than "is the cache under pressure?" or "is this stretch compressible?". The paper's Figure 1 shows the capability difference: fixed-interval wipes 4 verified facts → model falls back to guess; CUCG preserves them → model continues correctly.

For our codebase, the new class is also: **"compaction-as-meta-cognition"** — the agent decides when its own context is rotting, supplied as scaffolding rather than baked into weights. None of our existing primitives supplies this.

### Q3: Product selling point? — **YES**

**LLM-agent selling point (open):** "Our agent scaffold compacts context at structurally-safe moments — never mid-derivation, never wiping verified facts. Matches fixed-interval accuracy at 30–70% lower token cost."

**Game-AI selling point (private, riir-ai):** "Each NPC compacts its working memory at sub-goal resolution (combat won, dialogue done, resource gathered), not on a fixed tick budget. Verified facts (KG triples, committed shard state) persist across thousands of concurrent NPCs at 20Hz without context rot — emergent social behavior survives long horizons."

Both are concrete, user-visible, and finish the sentence "our agents do X that no competitor can".

### Q4: Force multiplier (≥2 pillars)? — **YES (10+ pillars)**

See §2.5 connection map. Most importantly: **the cross-domain isomorphism with `can_freeze`** (riir-neuron-db Plan 002) means CUCG simultaneously (a) gives us a trajectory-side compaction gate (new), (b) gives us a unifying abstraction for the shard-side freeze gate we already shipped (reframes existing IP), and (c) gives us a deterministic audit-trail format that works for both domains (anti-cheat contract for shards; explainability contract for agents).

→ **All 4 YES → Super-GOAT.**

---

## 5. Mandatory outputs (created in this session per anti-deferral rule)

1. **Open primitive** — this research note + `katgpt-rs/.plans/320_closed_unit_compaction_gate.md`. The generic `ClosedUnitCompactionGate<R>` + `Rubric` trait + `FireRule` enum + `Backstop` enum + `CompactionAuditRecord` land in `katgpt-rs/src/compaction/` behind feature `closed_unit_compaction`.
2. **Private guide** — `riir-ai/.research/155_Per_NPC_Sub_Goal_Compaction_Guide.md`. The game-runtime selling point: per-NPC sub-goal-triggered memory compaction at MMO scale.
3. **Cross-ref** — `riir-neuron-db/.research/007_Can_Freeze_As_Cucg_Instance_Crossref.md`. Recognizes `can_freeze` as the shard-side CUCG instance; documents the isomorphism.
4. **Plan(s)** — `katgpt-rs/.plans/320_closed_unit_compaction_gate.md` (open primitive). Runtime plan for the per-NPC variant deferred to riir-ai (TBD on guide acceptance).

---

## 6. Validation protocol (GOAT gate for CUCG)

The guide (riir-ai Research 155) contains the full gate; the open-primitive gate lives here.

- **G1 — Rubric beats fixed-interval on structural safety.** Synthetic trajectory with hand-marked "safe-to-compact" and "mid-derivation" points. Measure: compaction recall at safe points ≥ 80%, false-positive rate at mid-derivation ≤ 20%. (Paper's Figure 1 is the existence proof.)
- **G2 — Skip-if-reliable suppression.** When CLR vote on recent completions > `τ_reliable`, compaction is suppressed. Measure: suppression rate ≥ 50% on reliable prefixes; quality maintained (paper's §4.1 oracle gives the headroom target).
- **G3 — Cache-reuse probe overhead independent of L.** Probe latency measured at L = 1k, 10k, 100k tokens. Target: probe latency within ±10% across L (only the appended instruction pays prefill).
- **G4 — Zero-alloc hot path.** `evaluate()` and `decide()` perform no heap allocation. Same gate as Salience Tri-Gate (Bench 303): alloc-count via `#[track_caller]` allocator or `dhat`.
- **G5 — Feature isolation.** Compiles ±`closed_unit_compaction`. `nm target/release/libkatgpt_rs.dylib | grep -ic compaction` → 0 when feature off.
- **G6 — Sigmoid, never softmax.** Each predicate is a scalar in `[0,1]` from a sigmoid projection; fire rule is Boolean. Static check: no `softmax` call in the module.
- **G7 — Cross-domain isomorphism (riir-neuron-db).** `can_freeze` and CUCG share the same `FireRule::And` structure and the same `CompactionAuditRecord` shape. Test: construct a CUCG with a shard-freeze rubric and verify it produces bit-identical decisions to `can_freeze` on the same inputs.
- **G8 — Runtime fusion (riir-ai).** Per-NPC sub-goal compaction integrated into the tick loop at 20Hz × 1000 NPCs. Target: compaction fires at sub-goal boundaries (combat won, dialogue done) ≥ 80% of the time; never fires mid-action. (Paper does not validate this — it's our contribution.)

G1–G7 are the open-primitive gate. G8 is the runtime gate (riir-ai Plan TBD).

---

## 7. Latent vs raw boundary (per AGENTS.md)

- **Latent / local** (never synced raw): the trajectory `y_{1:t}` itself, the rubric's predicate confidence scalars before audit serialization, the verbatim quotes (they reference trajectory spans, not raw coordinates).
- **Raw / deterministic** (crosses sync boundary): `CompactionAuditRecord` — the bit-identical decision record (decision byte, fire-rule evaluation, backstop/skip flags). Two honest nodes must agree on whether compaction fired and why. This is the **anti-cheat / explainability contract**.
- **Bridge** (latent → raw): the audit record is constructed by projecting the latent predicate confidences through fixed thresholds (`Yes/No`) and recording the trajectory span that grounded each `Yes`. Zero-allocation, gateable by feature flag, no sync dependency introduced.

KG triple emission: a `Compress` decision on a semantic trajectory may emit a KG triple ("entity resolved sub-goal G at tick T") from latent similarity; the `Compress` decision itself is raw (auditable). Physical events (position change) are unaffected.

---

## 8. What stays public vs private

| Component | Repo | License |
|---|---|---|
| `ClosedUnitCompactionGate<R>` + `Rubric` trait + `FireRule` + `Backstop` + `CompactionAuditRecord` | katgpt-rs | MIT |
| Search rubric (C1/C2/C3/N1) + math rubric (Q1/Q2/Q3) as `Rubric` impls | katgpt-rs | MIT |
| Token-pct backstop + skip-if-reliable fuse + probe-revert protocol | katgpt-rs | MIT |
| **NPC sub-goal rubric** (combat-won, dialogue-done, resource-gathered predicates) | riir-ai | Private |
| **Per-NPC tick-loop wiring** (gate → memory compaction → KG emission) | riir-ai | Private |
| **HLA-based predicate recipes** (valence/arousal convergence = closed-unit for emotional sub-goals) | riir-ai | Private |
| **Shard-freeze rubric** (`can_freeze` as CUCG instance) | riir-neuron-db | Private (already shipped; the cross-ref recognizes it) |
| **LatCal-committed audit trail** (chain bridge for `CompactionAuditRecord`) | riir-chain | Private |

---

## 9. Implementation priority

| Priority | Task | Owner | Gate |
|---|---|---|---|
| **P0** | `Rubric` trait + `FireRule` + `Backstop` + `CompactionAuditRecord` types | katgpt-rs | — |
| **P0** | `ClosedUnitCompactionGate<R>::evaluate() / decide()` zero-alloc kernel | katgpt-rs | G4, G6 |
| **P0** | Search rubric (C1/C2/C3/N1) impl + paper Figure 1 reproduction test | katgpt-rs | G1 |
| **P1** | Skip-if-reliable CLR fuse | katgpt-rs | G2 |
| **P1** | Cache-reuse probe protocol + probe-revert on CONTINUE | katgpt-rs | G3 |
| **P1** | Token-pct backstop | katgpt-rs | G1 (backstop arm) |
| **P2** | Cross-domain isomorphism test: shard-freeze rubric vs `can_freeze` | katgpt-rs + riir-neuron-db | G7 |
| **P2** | Math rubric (Q1/Q2/Q3) impl | katgpt-rs | G1 (math arm) |
| **P3** | Per-NPC sub-goal rubric + tick-loop wiring | riir-ai | G8 |
| **P3** | LatCal-committed audit trail bridge | riir-chain | — |

---

## 10. Risk and validation

- **Paper-LLM-judge dependence.** The paper's predicates are LLM-judged from verbatim quotes. Our latent reframing (§2.4) replaces this with sigmoid projections on coherence/intrinsic-rank/divergence/novelty — but we must validate that the latent predicates track the LLM-judged predicates closely enough to preserve the paper's gains. **Mitigation:** G1 uses the paper's own Figure 1 setup as the existence proof; our latent version must reproduce the "4 verified facts preserved" outcome on a synthetic BrowseComp-like trajectory.
- **Probe-revert correctness.** The paper's CONTINUE path requires reverting the rubric judgement from the rolling cache. Implementation must guarantee no cache pollution. **Mitigation:** unit test that generates k CONTINUE probes then verifies subsequent generation matches a no-probe baseline byte-for-byte (modulo KV-cache indexing).
- **Cross-domain isomorphism is a claim, not yet a proof.** `can_freeze` uses `output_converged` (spectral flatness < 0.3) and `input_sufficient` (N ≥ d); CUCG uses C1/C2/C3/N1. The isomorphism is structural (both are multi-predicate Boolean fire rules over audit-able predicates with deterministic records), not semantic (the predicates measure different things). G7 must construct a shard-freeze `Rubric` impl whose decisions match `can_freeze` bit-for-bit. **Mitigation:** Plan 320 P2 task is exactly this construction.
- **Skip-if-reliable threshold.** CLR's `(mean_m v_k,m)^M` reliability vote is per-completion; the suppression fuse must decide "is the current answer reliable enough to suppress compaction?". The threshold `τ_reliable` needs calibration. **Mitigation:** start with paper's oracle bound (skip-if-correct gives +11.5 over fixed-interval); tune `τ_reliable` to recover ≥80% of that headroom.

---

## 11. References

- Li, Zhang, Jurayj, Wang, Jin, Farajtabar, Nalisnick, Khashabi. *Self-Compacting Language Model Agents.* arXiv:2606.23525, 22 Jun 2026.
- Existing closest cousins (this session): Plans 195 (chain_fold), 238 (AdaptiveTraceCompactor), 271 (attention_matching), 284 (CLR), 294 (ict_branching), 303 (salience_tri_gate), 307 (claim_rubric_runtime); riir-neuron-db Plan 002 (`can_freeze`).
- DEC substrate for C3 divergence signal: Plans 251 (DEC operators), 314 (Stokes wrappers).

---

## TL;DR

SelfCompact (arXiv:2606.23525) gives us a rubric-gated, training-free trajectory compaction scaffold: a 4-predicate cite-verbatim rubric (C1 closed-unit ∧ C2 summarizable ∧ C3 progress ∧ ¬N1 stuck) with a Boolean fire rule, an append-only cache-reuse probe protocol, a token-pct backstop, and a skip-if-correct oracle (+11.5 headroom). **Verdict: Super-GOAT.** All 4 novelty-gate questions pass: no shipped primitive combines these pieces; structural-safety-gated compaction is a new capability class vs resource-pressure-gated; both LLM-agent and game-AI selling points are concrete; force-multiplies 10+ existing pillars — most importantly by recognizing that our already-shipped `can_freeze` shard gate (riir-neuron-db Plan 002) and SelfCompact's rubric are **isomorphic closed-unit compaction gates**, one for shards and one for trajectories, converged independently. **Mandatory outputs created in this session:** research note (this file), open plan (`katgpt-rs/.plans/320_*`), private guide (`riir-ai/.research/155_*`), cross-ref (`riir-neuron-db/.research/007_*`). Latent reframing: predicates are sigmoid projections on coherence / intrinsic-rank / divergence / novelty — never softmax; fire rule is Boolean conjunction. Open primitive (MIT): `ClosedUnitCompactionGate<R>` + `Rubric` trait + `FireRule` + `Backstop` + `CompactionAuditRecord` in `katgpt-rs/src/compaction/`. Private selling point (riir-ai): per-NPC sub-goal memory compaction at MMO scale. GOAT gates G1–G8 defined; G1–G7 open, G8 runtime.
