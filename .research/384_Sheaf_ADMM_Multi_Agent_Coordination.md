# Research 384: Sheaf-ADMM Multi-Agent Coordination → Modelless Primal/Consensus/Dual Triple

> **Source:** Seely, Cupiał, Jones — "Learning Multi-Agent Coordination via Sheaf-ADMM" (ICML 2026), [arXiv:2605.31005](https://arxiv.org/abs/2605.31005)
> **Date:** 2026-07-06
> **Status:** Active — Super-GOAT (private guide in `riir-ai/.research/314`)
> **Related Research:** katgpt-rs 219 (DEC substrate), 296 (Stokes vocabulary crosswalk), 354 (cross-datapoint set attention — the closest cousin); riir-ai 143 (Latent CCE Crowd), 167 (crowd joint inference — single-state prior art), 314 (private Super-GOAT guide)
> **Related Plans:** katgpt-rs 407 (open primitive), riir-ai 394 (private runtime)
> **Cross-ref (riir-ai):** Research 314 (NPC Crowd Sheaf Coordination Guide — the selling-point doc)
> **Classification:** Public (math + open primitive). Selling-point fusion stays in riir-ai/314.

---

## TL;DR

Sheaf-ADMM coordinates N agents with **individually insufficient local views** by giving each agent **three state variables** — a primal proposal `x_i`, a consensus projection `z_i`, and a dual disagreement accumulator `u_i` — and iterating ADMM on a **cellular sheaf** that specifies which aspects of neighboring states must agree. The novel transferable piece (independent of the paper's backprop training) is the **three-state primal/consensus/dual decomposition** plus the **sheaf-structured heterogeneous consensus** (agree only on projections into shared edge stalks). Both map cleanly onto substrate we already ship: the sheaf Laplacian `L_F = F^T F` IS our `hodge_laplacian` Δ, the coboundary operator F IS our `exterior_derivative` d, and "project onto the harmonic subspace" IS `hodge_decompose`. The dual `u_i` is the genuinely new capability class — a per-agent **accumulated disagreement signature** with no analog in any shipped crowd system.

**Distilled for katgpt-rs (modelless, inference-time):**
A generic `sheaf_admm_step` operator on `CellComplex` that, given per-vertex primal/consensus/dual cochains and per-edge restriction maps, performs one ADMM iteration (x-update proximal solve → z-update sheaf diffusion → u-update dual accumulation). Zero training, zero backprop — the restriction maps are constructed deterministically from the cell complex incidence structure (or loaded as a frozen artifact). The operator is the open adoption hook; the per-NPC / per-zone wiring and the selling-point fusion live in `riir-ai/.research/314`.

---

## 1. Paper Core Findings

### 1.1 The three-state decomposition (the headline)

Standard MPNNs and our shipped R167 crowd attention carry **one hidden state** `h_i` per agent. Sheaf-ADMM splits this into three:

| Variable | Name | Role | Update |
|---|---|---|---|
| `x_i ∈ R^{d_v}` | **primal** | Agent's local proposal / decision | `x^{k+1}_i = prox_f(z^k_i − u^k_i)` — locally greedy solve |
| `z_i ∈ R^{d_v}` | **consensus** | Nearest globally consistent state | `z^{k+1} = Π_{ker F}(x^{k+1} + u^k)` — sheaf diffusion |
| `u_i ∈ R^{d_v}` | **dual** | Accumulated disagreement integral | `u^{k+1} = u^k + x^{k+1} − z^{k+1}` |

The dual `u_i` is the genuinely new variable. It is the **running integral of how often agent i's local proposal diverged from the zone consensus** — a per-agent *disagreement fingerprint* that no single-state architecture produces.

### 1.2 The cellular sheaf (the math we already ship)

A cellular sheaf `F` over a graph `G = (V, E)` assigns:
- A vertex stalk `R^{d_v}` to each vertex (agent state space)
- An edge stalk `R^{d_e}` to each edge (the agreement space, `d_e ≤ d_v`)
- Restriction maps `F_{i→e} ∈ R^{d_e × d_v}` projecting vertex state into edge stalk

**Heterogeneous consensus:** agents need only agree on **projections** of their states. Two agents solving adjacent maze regions need only coordinate on whether paths connect at their boundary, not on entire internal structures. The paper's exact words: "A more flexible notion of consensus allows neighboring agents to agree only on low-dimensional projections."

The **sheaf Laplacian** `L_F = F^T F` measures total disagreement:
```
x^T L_F x = Σ_{e=(i,j)} ‖F_{i→e} x_i − F_{j→e} x_j‖²
```
Gradient descent on `x^T L_F x` converges to the projection onto `ker(L_F) = ker(F)` — the harmonic subspace.

### 1.3 The z-update is literally `hodge_decompose`

The paper's "sheaf diffusion" z-update is gradient descent on the sheaf energy:
```
z^{(t+1)} = z^{(t)} − η L_F z^{(t)} = z^{(t)} − η F^T F z^{(t)}
```
In our shipped vocabulary (`katgpt-dec/src/operators.rs`):
- The coboundary operator `F` is `exterior_derivative` (d)
- The adjoint `F^T` is `codifferential` (δ)
- The sheaf Laplacian `L_F = F^T F` is `hodge_laplacian` (Δ = δd + dδ)
- The target `ker(L_F)` is the **harmonic component** of `hodge_decompose`

The "consensus projection" target IS the harmonic subspace. The "disagreement" IS the exact + coexact components. **The math is shipped; we just never framed it as multi-agent consensus.**

### 1.4 The d_e < d_v capacity rule (critical design constraint)

Appendix E Table 5: when `d_v ≤ d_e`, performance collapses (e.g. `d_v=32, d_e=64` → 0.2% solved). The restriction maps MUST compress: vertex stalk must be richer than edge stalk, forcing agents to agree on a low-dim summary rather than passing high-dim noise. This is exactly our **latent-to-raw sync rule** (sync the 5 affect scalars, never the full 64-dim HLA vector) — the paper provides the formal justification.

### 1.5 Empirical evidence for the structural claim

- Sudoku 92.6% solve vs MPNN 10.7% (parameter-matched) — the primal/consensus/dual split + sheaf constraint beats generic message passing
- MNIST +16px padding: CNN drops to 11.4%, Sheaf-ADMM retains 86.3% — local-view decomposition + heterogeneous consensus generalizes OOD
- Maze size generalization: trained on 19×19, near-saturated through 39×39 (2× linear) — the sheaf structure transfers to larger graphs

---

## 2. Distillation

### 2.1 What ships here (open primitive)

A `sheaf_admm` module in `katgpt-dec` exposing:

```rust
/// One ADMM iteration on a cellular sheaf. Modelless: restriction maps are
/// either identity (homogeneous consensus) or constructed deterministically
/// from incidence structure / loaded as a frozen artifact. No training.
pub fn sheaf_admm_step(
    cx: &CellComplex,
    restriction_maps: &SheafMaps,        // F_{i→e} per edge, d_e × d_v
    primal_x: &mut CochainField,         // rank-0, dim=d_v per vertex
    consensus_z: &mut CochainField,      // rank-0, dim=d_v per vertex
    dual_u: &mut CochainField,           // rank-0, dim=d_v per vertex
    local_objective: &LocalObjective,    // f_i params (quadratic / diagonal+ℓ1)
    rho: f32,                            // ADMM penalty
    diffusion_steps: usize,              // T sheaf-diffusion steps for z-update
    scratch: &mut AdmmScratch,           // pre-allocated buffers
)
```

The z-update is the critical reuse: it IS sheaf diffusion, which IS gradient descent on the Hodge energy. We can implement it as `T` steps of `consensus_z -= eta * hodge_laplacian(cx, consensus_z)` — already-shipped operator.

### 2.2 What is training-only (→ riir-train)

- Learning the restriction maps `F_{i→e}` via backprop through unrolled K iterations
- Learning the local objective encoder (Q_i, q_i, λ_i per agent)
- Learning the LoRA modulation `ΔF_i = U_i V_i^T` on base restriction maps
- End-to-end training of encoder + ADMM + decoder

These all stay in riir-train. The runtime analog (riir-ai) constructs the restriction maps modellessly — see §2.3.

### 2.3 Modelless unblock paths (per workflow §3.5 — checked before any riir-train deferral)

For runtime use (not training), the restriction maps can be constructed deterministically:

1. **Identity restriction maps** (homogeneous consensus) — `F_{i→e} = [I_{d_e}; 0]` for all edges. Agents agree on the first `d_e` components of their state. This is the modelless floor; the paper's "Learned Shared Maps" ablation (Table 3) shows this still works on Sudoku (92.5%).
2. **Selector restriction maps** (heterogeneous consensus) — `F_{i→e}` selects a specific `d_e`-subset of `d_v` based on edge direction / zone topology. E.g. north-south edges select position components; east-west edges select affect components. Deterministic, derived from `CellComplex` incidence.
3. **CS-ranking-derived restriction maps** (the fusion with Mind-Reading R133) — use the existing offline CS-KV-Importance Probe ranking to pick which `d_e` dims to project onto per task family. This is the modelless analog of the paper's learned LoRA modulation.

None of these require gradient descent. All three are modelless-validable.

### 2.4 Fusion (the Super-GOAT angle)

**Sheaf-ADMM × Research 167 (cross-NPC set attention) × DEC substrate × CCE Crowd Batch (P328) × Latent Functor (R123) × Committed Personality (P336):**

R167 already does cross-NPC peer attention but with **single state** per NPC. Sheaf-ADMM adds:
1. **The primal/consensus/dual triple** — three analyzable variables per NPC instead of one fused hidden state
2. **Sheaf-structured heterogeneous consensus** — different NPC pairs agree on different state projections (formalizes the latent-to-raw sync rule)
3. **The dual `u_i` as disagreement fingerprint** — a NEW per-NPC signal that R167 cannot produce. Persistent disagreement → personality divergence marker (feeds Committed Personality P336). Sudden `u_i` spike → coordinated anomaly (feeds chain forensic).

This is the Super-GOAT: the primal/consensus/dual triple on a cellular sheaf is a new capability class, not an optimization. See `riir-ai/.research/314` for the full selling-point guide.

---

## 3. Verdict

**Tiers:**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | Open primitive → katgpt-rs. Architectural guide → riir-ai/.research/. Plans → both repos as needed. |
| **GOAT** | Provable gain over existing approach, but not a new class. | Plan + implement. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline. | Plan only, behind feature flag. |
| **Pass** | Not relevant, OR training-only. | One-line note. |

**Verdict: Super-GOAT.**

**One-line reasoning:** The primal/consensus/dual triple per agent is a new capability class with no prior art in any of the 5 repos (R167 ships single-state; DEC ships the math but not the multi-agent framing); the dual `u_i` disagreement accumulator is a genuinely new per-NPC signal that no incumbent produces; the mechanism is a force multiplier connecting DEC substrate + Crowd MCGS + CCE Crowd + HLA + Committed Personality + chain forensic (≥6 systems); and the selling point — "NPCs carry three coordinated state variables and run sheaf-diffusion consensus over zone topology" — is a defensible moat.

**Novelty gate (§1.5):**
- **Q1 No prior art?** YES. Grepped all 5 repos for `primal.*consensus.*dual|ADMM|sheaf|dual accumulator` — zero ADMM-triple implementations. R167 is closest (peer attention, single-state). DEC substrate ships the math but not the multi-agent framing. The `consensus_state` field in `VibeTelemetry` is a u8 status enum, not a continuous consensus variable.
- **Q2 New class of behavior?** YES. Three-state per-NPC decomposition with disagreement accumulation is qualitatively different from single-state attention. The dual `u_i` enables persistent-disagreement detection and forensic signatures — impossible with single-state architectures.
- **Q3 Product selling point?** YES. "Our NPCs carry three coordinated state variables (local decision, zone consensus, accumulated disagreement integral) and run sheaf-diffusion consensus over the zone topology, producing emergent collective behavior with analyzable coordination dynamics — and a per-NPC disagreement fingerprint that no competitor's single-state NPC can produce."
- **Q4 Force multiplier?** YES. Connects DEC substrate (katgpt-dec) + Crowd MCGS (pillar extension) + CCE Crowd Batch (R143) + HLA belief state (R242) + Committed Personality (P336) + chain forensic (R268 LatCal fingerprinting) + two-brain model. ≥6 systems, ≥2 pillars.

**All 4 YES → Super-GOAT confirmed.** Per workflow §1.5 mandatory outputs: open primitive in katgpt-rs (Plan 407), private guide in riir-ai/.research/314, runtime plan in riir-ai/.plans/394.

**MOAT gate per domain (§1.6):**
- **katgpt-rs (this note + Plan 407):** the `sheaf_admm_step` operator on `CellComplex` is a generic math primitive — paper-derived fundamental mechanism. **Strengthens the engine adoption funnel.** Correct repo.
- **riir-ai (Research 314 + Plan 394):** the per-NPC primal/consensus/dual wiring on HLA, fusion with Mind-Reading / Latent Functor / Crowd MCGS, the disagreement-fingerprint selling point. **Pillar-level amplifier** — touches Pillar 5 (NPC Dialog) and the crowd MCGS extension. Correct repo.

---

## 4. Latent vs raw boundary (per AGENTS.md §Latent vs Raw)

| Data | Space | Synced? | Rule |
|---|---|---|---|
| Per-NPC primal `x_i` (HLA-derived) | Latent (`d_v`-dim) | NO | Local subjective proposal — fog-of-war gated, never synced |
| Per-NPC consensus `z_i` (sheaf projection) | Latent (`d_v`-dim) | NO | Local projection onto harmonic subspace — per-NPC subjective |
| Per-NPC dual `u_i` (disagreement integral) | Latent (`d_v`-dim) | NO (but **committable**) | Local accumulator — never synced raw, but BLAKE3-committable as forensic fingerprint (see chain bridge below) |
| Restriction maps `F_{i→e}` (frozen) | Latent | NO (BLAKE3-committed in NeuronShard) | Constructed modellessly or loaded at zone-init |
| Z-update harmonic projection output (5 affect scalars) | Raw | **YES (existing bridge)** | `compute_animal_emotions()` projects to valence/arousal/desperation/calm/fear — unchanged sync path |
| `u_i` BLAKE3 hash (forensic signature) | Raw (committed) | **YES (new, optional)** | The disagreement fingerprint commits to chain as a tamper-evident per-NPC signature. Bridge: `u_i → BLAKE3 → LatCal fixed-point → chain commit` |

**Critical invariant:** the ADMM iterations are purely *local latent-state refinement*. They introduce no new raw sync data. The existing 5-scalar bridge is the only thing that crosses, exactly as before. The optional `u_i` BLAKE3 hash is a *commitment* (deterministic, replayable), not a sync — it's forensic evidence, not gameplay state.

---

## 5. What stays public vs private

| Piece | Location | Why |
|---|---|---|
| `sheaf_admm_step` operator on `CellComplex` (the math) | **katgpt-rs** (`katgpt-dec`) | Generic math, no game semantics. Adoptable by anyone for any sheaf-ADMM use. |
| `SheafMaps` struct + identity/selector constructors | **katgpt-rs** | Generic construction primitives. |
| Local-objective trait + quadratic / diagonal+ℓ1 impls | **katgpt-rs** | Generic optimization substrate. |
| HLA-wiring convention (`d_v=8` HLA affect as primal) | **riir-ai** | Game-specific. |
| Mind-Reading CS-ranking as restriction-map source | **riir-ai** | Reuses R133 — our secret sauce. |
| Latent Functor direction set as consensus-key source | **riir-ai** | Reuses R123 — also our sauce. |
| Per-zone `ρ` / `η` / `T` ADMM hyperparameter derivation from crowd density | **riir-ai** | Game-specific tuning. |
| Crowd MCGS / CCE integration points | **riir-ai** | The fusion architecture. |
| `u_i` disagreement-fingerprint → chain forensic bridge | **riir-chain** (bridge) + **riir-neuron-db** (commitment Pod) | Cross-boundary commitment. |
| Bevy demo (guard patrol collective inference with three-state NPCs) | **riir-ai** | The visible selling point. |

**No fuel leaks:** the open primitive in katgpt-rs is generic sheaf-ADMM math on cell complexes. A competitor reading it learns "you can run ADMM on a cellular sheaf over a cell complex" — they don't learn "wire the primal/consensus/dual triple onto 8-dim HLA affect with Mind-Reading-derived restriction maps for crowd-scale NPC collective intelligence". That fusion is the moat, and it stays in riir-ai/314.

---

## 6. Performance considerations

- The x-update is `O(N · d_v²)` for diagonal quadratic (elementwise division), `O(N · d_v³)` for full QP — for `d_v=8` (HLA), this is trivial (~512 flops per agent).
- The z-update is `T` sheaf-diffusion steps; each step is a Hodge Laplacian matvec, which is `O(nnz(F)) = O(|E| · d_e)` for sparse restriction maps. For a zone with K NPCs and 4-way connectivity, `|E| ≈ 2K`, so one diffusion step is `O(K · d_e)`. With `T=5` and `d_e=5`, that's `O(50K)` flops per ADMM iteration.
- The u-update is `O(N · d_v)` — a simple vector add.
- Total per ADMM iteration at K=100 NPCs, d_v=8, d_e=5, T=5: ~50Kflops → ~500ns at 100 GFLOP/s SIMD throughput. Well within the 20Hz tick (50ms) budget — leaves 100,000× headroom.
- **Hot-loop rules respected:** pre-allocated `AdmmScratch`, no per-iteration allocation, all matvecs are sparse, the `CellComplex` is built once at zone-init.

**CPU/SIMD/GPU routing:**

| Zone size | Backend | Why |
|---|---|---|
| K ≤ 100 NPCs | CPU SIMD | Fits in L1, latency-critical (plasma tier) |
| 100 < K ≤ 1000 | SIMD batch | Hot tier, batched matvec |
| K > 1000 (server-scale) | GPU | Warm tier, batched sparse matmul |

---

## 7. GOAT gate (per Plan 407)

| Gate | Criterion | Target |
|---|---|---|
| **G1** (correctness — DEC identity) | `d∘d=0` on the consensus path: after convergence, `‖F x‖ → 0` (no edge disagreement) | < 1e-6 residual |
| **G2** (correctness — dual conservation) | `u^{k+1} − u^k = x^{k+1} − z^{k+1}` bit-exactly | exact |
| **G3** (heterogeneous consensus — d_e < d_v) | Restriction maps compress: `‖F x‖ ≤ ‖x‖` for all x | holds by construction |
| **G4** (latency) | One ADMM iteration, K=100 NPCs, d_v=8, d_e=5, T=5 | < 5 µs |
| **G5** (zero-alloc) | 0 allocations per `sheaf_admm_step` call in steady state | 0 |
| **G6** (determinism) | Same primal/consensus/dual output bit-exactly across runs | bit-exact |

**Promotion rule:** G1–G6 all pass → promote `sheaf_admm` to default-on in `katgpt-dec`. The runtime fusion (riir-ai) has its own gates in Research 314.

---

## 8. Limitations and honest risks

1. **The paper's headline numbers rely on training.** Sudoku 92.6% requires backprop-trained restriction maps. Our modelless version (identity / selector / CS-ranking-derived maps) cannot claim that number. The selling point grounds in "we already have the DEC substrate + Mind-Reading CS-rankings + Latent Functor directions" — NOT in "Sheaf-ADMM beats MPNN on Sudoku".
2. **The dual `u_i` is only useful if it produces visibly different behavior.** If the disagreement fingerprint doesn't enable emergent personality divergence or detectable anomaly patterns, the "new capability class" claim collapses. Research 314 §G8 is the kill switch.
3. **K iterations per tick may be too many for the 20Hz budget at crowd scale.** The paper uses K=20–30 for hard tasks. At K=30, T=5, K_NPCS=1000, that's 30× the per-iteration cost → ~15µs, still within budget but tighter. May need K≤5 at crowd scale.
4. **Sheaf diffusion conditioning.** The paper notes `L_F` conditioning deteriorates on large sparse graphs; they use conjugate gradient. Our shipped `hodge_laplacian` uses gradient descent. May need a CG variant for large zones — defer to Plan 407 Phase 2.
5. **Cousin saturation.** riir-ai/.research/ is dense in per-NPC runtime docs (R123, R126, R128, R133, R143, R146–R170). R314 must justify itself by clearly being **three-state sheaf-coordinated** consensus, NOT another single-state or one-way-broadcast mechanism. The primal/consensus/dual triple is the justification.

---

## 9. What NOT to do

1. **Don't implement the learned restriction maps in katgpt-rs.** That's riir-train territory (backprop through unrolled ADMM).
2. **Don't sync the primal/consensus/dual triples over the chain.** They are per-NPC subjective latent state. Only the optional `u_i` BLAKE3 hash crosses (as a commitment, not as sync data).
3. **Don't use softmax anywhere.** The ADMM penalty `ρ` and diffusion step `η` are scalars, not distributions. The primal update uses proximal maps (closed-form for quadratic), not softmax mixing.
4. **Don't claim Sudoku/MNIST numbers.** Those require training. Our modelless claim is the architectural pattern + the dual `u_i` disagreement signal.

---

## 10. Cross-references

| Research | Connection |
|---|---|
| katgpt-rs 219 (DEC substrate) | The math foundation — `exterior_derivative` = F, `hodge_laplacian` = L_F |
| katgpt-rs 296 (Stokes vocabulary) | Vocabulary crosswalk — confirms DEC ships the sheaf Laplacian |
| katgpt-rs 354 (cross-datapoint set attention) | The closest cousin — single-state peer attention; this adds the triple |
| riir-ai 143 (Latent CCE Crowd) | Crowd coordination prior art — CCE equilibria; Sheaf-ADMM adds the optimization-derived structure |
| riir-ai 167 (crowd joint inference) | The direct prior art — single-state; Sheaf-ADMM is the three-state extension |
| riir-ai 123 (Latent Functor) | Provides direction vectors as consensus keys |
| riir-ai 133 (Mind-Reading) | Provides CS-rankings as restriction-map source |
| riir-ai 314 (private Super-GOAT guide) | The selling-point doc — THIS is the moat |
| riir-neuron-db (NeuronShard) | Persists restriction maps + `u_i` fingerprint as BLAKE3-committed Pod |
| riir-chain (LatCal) | Commits `u_i` hash as forensic fixed-point signature |

## TL;DR

Sheaf-ADMM's three-state primal/consensus/dual decomposition is a new capability class for multi-agent coordination. The math (sheaf Laplacian = Hodge Laplacian, coboundary = exterior derivative, consensus projection = harmonic subspace) already ships in `katgpt-dec`. The genuinely novel transferable piece is the per-agent dual `u_i` disagreement accumulator — a per-NPC disagreement fingerprint with no analog in any shipped crowd system. **Verdict: Super-GOAT** — open primitive in katgpt-rs/407, private guide in riir-ai/314, runtime plan in riir-ai/394. Latent-vs-raw boundary respected: the primal/consensus/dual triples stay local; only the existing 5-scalar bridge crosses sync; the optional `u_i` BLAKE3 hash crosses as a forensic commitment. Six-system force multiplier (DEC + Crowd MCGS + CCE + HLA + Committed Personality + chain forensic). Modelless — restriction maps constructed from CS-rankings or incidence structure, no gradient descent.
