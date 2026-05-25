# Research 98: Eigenspace Alignment for Structural Anomaly Detection

> **Paper:** [Latent Geometry as a Structural Monitor: Eigenspace Alignment for Anomaly Detection in Anonymity Networks](https://arxiv.org/pdf/2605.20391) — Vaibhav Chhabra, May 2026
> **Date:** 2026-05, distilled 2026-05
> **Related Research:** 39 (SpectralQuant Eigenbasis), 51 (Deep Manifold Fixed-Point), 37 (REAP Model-Based/Modelless), 61 (Entropy Anomaly), 76 (SR²AM Configurator), 93 (Committee Boost), 97 (EGA Energy-Gated)
> **Related Plans:** 138 (katgpt-rs, Stiff/Soft Subspace Anomaly Gate), 139 (riir-ai, Game Structural Health Monitor)
> **Reference:** 27_mmo_goat_pillars_decision_matrix.md (Pillar 2: WASM Validators, Pillar 4: Frame-Sampling Bridge)

---

## TL;DR

The paper treats behavioral populations as **geometric energy landscapes** and detects structural anomalies by decomposing the encoder Jacobian into **stiff** (load-bearing) and **soft** (elastic) subspaces. A dual-observer pipeline (CDAE geometric + GRBM thermodynamic, bridged by CCA) detects events *before* thresholds are crossed — "structure precedes geometry." Applied to Tor relay metadata across 67 daily windows, the stiff subspace dimension k=9 is **emergent and invariant** (67/67 windows), validated at 16.8σ above Monte Carlo noise floor, with 0.0% FPR on 24 confirmed stable windows.

**Key distillation for our stack:**

1. **Stiff/soft subspace decomposition** composes directly with SpectralQuant's eigenbasis pipeline — we already compute `JᵀJ` eigenvalues for KV cache compression; the same machinery classifies population behavior as load-bearing vs. elastic.

2. **Dual-observer anomaly cascade** maps to our model-based/modelless duality — modelless (geometric: Jacobian trace, no forward pass) and model-based (thermodynamic: energy landscape, requires learned parameters). Our `ScreeningPruner`/`ConstraintPruner` trait stack already separates these modes.

3. **δmg mass-gravity divergence** (weighted vs. unweighted population centers) maps to our `BanditPruner` Q-value weighted vs. uniform routing — a principled way to detect "lightweight influx" in game populations (e.g., low-quality moves flooding MCTS) vs. genuine strategy shifts.

4. **Event taxonomy (PRECURSOR → REGIME_S → REGIME_D → REGIME_E → REGIME_K → NORMAL)** provides a principled classification framework for game state health monitoring — directly applicable to MMO server-side anomaly detection (Pillar 2/4).

**Verdict: 🟡 CONDITIONAL ADOPT — The stiff/soft subspace decomposition is high-value for katgpt-rs as a generic anomaly detection primitive (feature-gate `stiff_anomaly`). The dual-observer event taxonomy is game-specific and belongs in riir-ai as a structural health monitor for MMO game loops. The δmg discriminator is the super-GOAT piece — it distinguishes population floods from structural fractures in game populations, and this is a selling point that should stay private in riir-ai.**

---

## 1. Paper Core Architecture

### 1.1 Dual-Observer Pipeline

```
Daily Relay Snapshot
    │
    ├── CDAE (Geometric Observer) ──── 17 features → 32-dim latent
    │       Jacobian J = dz/dx at cluster center
    │       JᵀJ eigendecomposition → stiff (k=9) / soft (d-k=8)
    │
    ├── GRBM (Thermodynamic Observer) ─ 191 features
    │       F_visible = Σ (xᵢ - bᵢ)² / σᵢ²  (visible free energy)
    │       CV = std(F) / mean(F) → population fragmentation
    │
    └── CCA Bridge ── ρ₁ (agreement), θ (rotation), Δρ (drift)
```

### 1.2 Stiff/Soft Subspace Decomposition (EJT)

The encoder Jacobian `J = ∂z/∂x` evaluated at a cluster center produces:

```
JᵀJ = J_cleanᵀ J_clean   (shape: d×d, here 17×17)
```

Eigendecomposition of `JᵀJ`:
- **Stiff subspace** (top-k eigenvectors): directions the encoder is *most sensitive to* — load-bearing structure
- **Soft subspace** (remaining d-k eigenvectors): directions the encoder *barely notices* — elastic absorption

The **soft alignment ratio** `α`:
```
α = ‖V_soft V_softᵀ Δx‖² / ‖Δx‖²
```
- `α ≈ 1`: population moved through soft directions (elastic, benign)
- `α ≈ 0`: population moved into stiff axes (structural stress, anomaly)

### 1.3 Mass-Gravity Divergence (δmg)

```
δmg = ‖x_weighted_median - x_unweighted_median‖
```

Distinguishes:
- **Population surge** (lightweight influx): mass shifts, gravity stays → high δmg + high α = benign stretch
- **Structural fracture** (real anomaly): both shift similarly → low δmg + low α = genuine stress

### 1.4 Event Taxonomy

| Classification | Signal | Meaning |
|---|---|---|
| **PRECURSOR** | Ch5 CV > 3.0 only | Thermodynamic fragmentation before geometry responds |
| **REGIME_S** | α high + δmg high + sustained | Elastic stretch — population surge absorbed |
| **REGIME_D** | α high + δmg moderate + isolated | Localized deformation — contained shift |
| **REGIME_E** | Global EJT z < -2.0 | Stiff-axis fracture — population-wide stress |
| **REGIME_K** | α low + shift large + bimodal | Administrative — requires forensic checklist |
| **NORMAL** | No gates fire | Stability |

### 1.5 Key Results

| Metric | Value |
|---|---|
| Stiff subspace k=9 invariance | 67/67 windows |
| Monte Carlo null separation | 16.8σ |
| Primary gate FPR (Ch5 CV, Ch6 EJT) | 0.0% |
| Top-10 Jaccard stability | median 0.90 |
| Confirmed event detection | Feb 20, 2026 (Cloudflare BGP, z=-4.38) |

---

## 2. Mapping to Our Architecture

### 2.1 Stiff/Soft → SpectralQuant Eigenbasis

We already compute eigendecomposition of per-(layer, head) key covariance in SpectralQuant (Research 39). The participation ratio `d_eff = (Σλ)²/(Σλ²)` is the same math. The paper's EJT adds:

- **Temporal dimension**: track `α(t)` over consecutive windows (our training steps / game rounds)
- **Load-bearing interpretation**: stiff dimensions aren't just "high variance" — they're the *structure that shouldn't change*

**What we have:** `eigenvalues()`, `participation_ratio()`, eigenvector rotation in `spectralquant/`
**What's new:** soft alignment ratio `α`, temporal tracking of eigenvalue stability, z-score gating

### 2.2 Dual-Observer → Model-Based/Modelless Duality

| Paper Component | Our Equivalent | Mode |
|---|---|---|
| CDAE geometric (Jacobian, no training) | `ScreeningPruner.relevance()` | **Modelless** |
| GRBM thermodynamic (learned energy) | `BanditPruner<P>` Q-values | **Model-based** |
| CCA bridge (observer agreement) | SR²AM configurator cross-signal | **Meta** |
| EJT z-score gate | `entropy_score()` anomaly threshold | **Modelless** |
| δmg discriminator | Weighted vs. uniform routing in BanditPruner | **Model-based** |

The paper's "structure precedes geometry" maps to our "modelless before model-based" philosophy — cheap structural checks before expensive forward passes.

### 2.3 Event Taxonomy → Game State Health

Mapping the six event types to our game domains:

| Paper Event | Game Analog | Detection |
|---|---|---|
| PRECURSOR | NPC dialog quality degrading before visible failure | CV of move quality scores rising |
| REGIME_S | New players flooding a zone (elastic) | α high, δmg high (weighted positions stable) |
| REGIME_D | Single-player exploit detected (localized) | α high, δmg low, isolated to one role |
| REGIME_E | Server-wide game state corruption | Global EJT z < -2.0 across all players |
| REGIME_K | Scheduled maintenance / admin action | Bimodal restart age, forensic checklist |
| NORMAL | Stable game loop | No anomaly gates fire |

### 2.4 δmg → BanditPruner Population Discrimination

The mass-gravity divergence directly applies to MCTS/Bandit populations:

```
δmg_bandit = ‖Q_weighted_mean - Q_unweighted_mean‖
```

- **Lightweight moves** (low visit count, high variance): mass shifts, gravity stays → exploration surge, not convergence failure
- **Structural shift** (genuine strategy change): both move → real convergence event

This is the **super-GOAT insight**: distinguishing "lots of noise moves" from "the game has changed" in MCTS populations. Private to riir-ai as game-specific knowledge.

---

## 3. Distillation Breakdown

### 3.1 katgpt-rs (Open, Generic)

| Component | What | Feature Gate | GOAT Proof |
|---|---|---|---|
| `soft_alignment_ratio()` | Generic stiff/soft α computation on any `JᵀJ` eigenbasis | `stiff_anomaly` | α=1 for elastic, α=0 for stiff, test with known rotations |
| `stiff_subspace_k()` | Find invariant k at 90% trace mass threshold | `stiff_anomaly` | k stable across N synthetic windows |
| `eigenspace_zscore()` | Z-score of current EJT against frozen baseline | `stiff_anomaly` | 0.0% FPR on stable windows |
| `eigenvalue_jaccard()` | Top-k feature loading stability across windows | `stiff_anomaly` | Median Jaccard ≥ 0.85 |

These are generic linear algebra utilities that extend SpectralQuant's existing eigenbasis code. They don't encode any game-specific knowledge.

### 3.2 riir-ai (Private, Game-Specific)

| Component | What | Why Private |
|---|---|---|
| `GameStructuralHealth` struct | Event taxonomy (PRECURSOR→REGIME_K) applied to game populations | Game-specific tuning thresholds |
| `game_mass_gravity_divergence()` | δmg for MCTS/Bandit populations — distinguishes exploration surge from convergence failure | **Super GOAT**: tells you whether noise or strategy changed |
| `npc_dialog_drift_detector()` | PRECURSOR detection on NPC dialog quality scores | Private quest FSM knowledge |
| `mmo_zone_health_monitor()` | Per-zone stiff/soft tracking for MMO server | Pillar 4 integration (Frame-Sampling Bridge) |
| `fleet_restart_forensic()` | REGIME_K checklist for MMO maintenance events | Operational MMO domain knowledge |

The δmg discriminator is the key selling point: nobody else can tell you whether your MCTS is exploring or collapsing. This is private game IP.

---

## 4. GOAT Pillar Alignment

Reference: `27_mmo_goat_pillars_decision_matrix.md`

| Pillar | How This Research Strengthens It |
|---|---|
| **P1: Fourier Spatial AI** | Stiff/soft decomposition adds anomaly detection to Fourier-hashed positions — detect when spatial structure changes (rigid) vs. normal variation (elastic) |
| **P2: WASM Validators** | `GameStructuralHealth` runs inside WASM sandbox — structural anomaly detection as a validator-level gate. "Is the game population healthy?" as a validation primitive |
| **P3: NPC Dialog Engine** | `npc_dialog_drift_detector` provides PRECURSOR detection — flag when NPC responses drift before visible quality failure |
| **P4: Frame-Sampling Bridge** | `mmo_zone_health_monitor` per-zone stiff/soft tracking — decide frame sampling ratio based on structural health (healthy = decimate more, stressed = sample more) |

**LoRA independence:** All four applications work modelless. The stiff/soft decomposition is pure linear algebra. The δmg discriminator uses existing BanditPruner Q-values. No neural network required.

---

## 5. Honest Assessment

### What's Genuinely New (We Don't Have)

1. **Temporal eigenvalue tracking** — We compute eigenvalues once (SpectralQuant calibration), then freeze. The paper tracks stability across 67 windows. We should track across training steps / game rounds.

2. **Soft alignment ratio α** — We don't compute the projection of population change onto soft vs. stiff axes. This is a one-liner given existing eigenbasis, but we never compute it.

3. **δmg as discriminator** — We use weighted/unweighted means in BanditPruner but never compute their divergence as a diagnostic signal.

### What We Already Have (No Action)

- Eigenvalue decomposition, participation ratio (SpectralQuant)
- Modelless anomaly detection via `entropy_score()` (Plan 061)
- Multi-observer signal fusion (SR²AM configurator, Committee Boost)
- Event logging / game trace fork-diff (Plan 124)
- Z-score gating with FPR validation (GOAT methodology)

### What's Questionable

- **Contractive penalty (λc = 0.001)**: The CDAE's Frobenius penalty on J forces the encoder to be contractive, which produces clean stiff/soft separation. Our SpectralQuant eigendecomposition doesn't have this bias. We may need to add a contractive term for clean subspace separation — or it may emerge naturally from the covariance structure. Needs empirical validation.

- **k=9 emergence**: The paper claims k=9 is emergent and invariant *for Tor relay data*. Our game data may have a different invariant k. The framework should discover k empirically, not assume it.

- **CCA bridge complexity**: The paper's per-window CCA refit adds computational overhead. For 20Hz MMO game loops, this may be too expensive. Simplified alternatives (cosine similarity between latent vectors, KL divergence) may suffice.

---

## 6. Verdict Summary

| Dimension | Assessment |
|---|---|
| **Novelty** | Medium — stiff/soft decomposition is standard sensitivity analysis; the event taxonomy and δmg discriminator are the novel contributions |
| **Implementability** | High — all components are linear algebra primitives that compose with existing SpectralQuant code |
| **GOAT provability** | High — z-score FPR, Jaccard stability, and k-invariance are all measurable with synthetic data |
| **Moat value** | High for δmg game discriminator (private riir-ai), Medium for generic stiff/soft (open katgpt-rs) |
| **LoRA independence** | Full — pure modelless linear algebra |
| **Risk** | Low — additive feature gates, no existing code changes |

**Action:**
1. Add `stiff_anomaly` feature gate to katgpt-rs with generic stiff/soft utilities
2. Add game-specific `GameStructuralHealth` to riir-ai (private, super GOAT)
3. Wire into GOAT pillar framework for MMO server integration
