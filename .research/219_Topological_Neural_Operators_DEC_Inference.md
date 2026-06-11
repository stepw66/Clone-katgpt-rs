# Research 219: Topological Neural Operators → DEC Inference Operators

**Date:** 2026-06
**Status:** 🟢 GAIN — Modelless DEC operators for game AI spatial reasoning
**Paper:** [Topological Neural Operators (arXiv:2606.09806)](https://arxiv.org/abs/2606.09806)
**Context:** katgpt-rs modelless inference, flow fields, spectral hierarchy, Dirichlet energy

---

## TL;DR

TNOs model physical quantities as cochains on cell complexes (vertices→scalars, edges→circulations, faces→fluxes, volumes→densies) and route information through fixed DEC operators (gradient d₀, curl d₁, div d₂, Hodge Laplacian Δₖ). The key insight for us: **these are not neural operators — they're topological routing primitives** that we can implement as zero-alloc, SIMD-accelerated inference-time operators for game spatial reasoning, replacing ad-hoc gradient/flow computations.

---

## 1. Paper Core Ideas

### 1.1 Cochain Fields (Not Just Point Fields)

Current state: `FlowField` in `flow/mod.rs` stores 2D flow vectors on a grid — all rank-0 (vertex) data. TNOs show this is lossy: physical quantities have geometric type. Pressures live on vertices, circulations on edges, fluxes on faces.

**Key equation:** A k-cochain `uₖ: Kₖ → ℝ^{dₖ}` assigns features to k-cells. Rank matters:
- Rank 0 (vertices): scalars — potential, pressure, HP, value
- Rank 1 (edges): circulations — flow, gradient, threat direction  
- Rank 2 (faces): fluxes — area-normalized quantities, vorticity
- Rank 3 (volumes): densities — mass, charge, occupancy

### 1.2 Fixed DEC Operators as Routing

```
d₀ (gradient):  C₀ → C₁  (vertex → edge differences)
d₁ (curl):      C₁ → C₂  (edge → face circulation)  
d₂ (divergence): C₂ → C₃  (face → volume flux)
δₖ (codifferential): Cₖ → Cₖ₋₁  (metric adjoint of d)
Δₖ = δₖ₊₁dₖ + dₖ₋₁δₖ  (Hodge Laplacian)
```

**Critical identity:** `dₖ₊₁ ∘ dₖ = 0` (boundary of boundary is zero). This gives curl(grad) = 0 and div(curl) = 0 for free — no soft penalty needed.

### 1.3 Hodge Decomposition (4 Channels)

Every cochain decomposes into 3 orthogonal components:
```
Cₖ = im(dₖ₋₁) ⊕ ker(Δₖ) ⊕ im(δₖ₊₁)
      exact       harmonic    coexact
```

- **Exact:** Conservative fields (gradients of potentials)
- **Harmonic:** Topological modes (cycles, holes) — dimension = Betti number βₖ
- **Coexact:** Divergence-free fields (solenoidal)

The 4th channel is a local/self residual.

### 1.4 HTNO as Learned V-Cycle

Hierarchical TNO = multi-grid on cell complexes:
1. Pre-smooth on fine complex K with TNO layers
2. Restrict to coarse complex Kc
3. Solve on Kc  
4. Prolongate back to K
5. Post-smooth on K

Transfer maps ideally commute with coboundary: `dₖKc ∘ Rₖ = Rₖ₊₁ ∘ dₖK`.

---

## 2. Fusion Novel Ideas (Not Direct Mapping)

### 2.1 DEC-Structured Game Spatial Reasoning

**Current:** `LeoPotentialGrid::gradient()` computes naive finite differences on a 2D grid → `FlowField`. This is rank-0 only, no conservation guarantees.

**Fusion:** Replace with DEC operators on a cell complex. The game map IS a cell complex:
- Vertices = grid cells (rank 0): Q-values, occupancy, HP
- Edges = connections between cells (rank 1): flow vectors, threat gradients, movement costs
- Faces = rooms/areas (rank 2): area flux, room-level threat accumulation
- Volumes = zones (rank 3): zone density, zone-level resource totals

The gradient `d₀` replaces `LeoPotentialGrid::gradient()`. But now we also get:
- Curl `d₁` = "how much does threat circulate around a room?" (camp detection)
- Divergence `δ₁` = "where is threat converging/diverging?" (chokepoint detection)
- Harmonic component = "what routes are topologically guaranteed?" (regardless of obstacle placement)

### 2.2 Hodge-Decomposed NPC Navigation

NPCs currently use flow fields (rank-0 gradient of potential). TNO suggests decomposing navigation into:

1. **Exact channel:** "Go toward the goal" — gradient of distance potential. Conservative, path-independent.
2. **Coexact channel:** "Patrol the boundary" — divergence-free circulation around obstacles. Gives natural patrol behavior.
3. **Harmonic channel:** "Use the topology" — modes supported by map topology (loops, holes). These are routes that exist regardless of obstacle placement — **strategic navigation**.
4. **Self channel:** Local adjustments for dynamic obstacles.

This gives NPCs three qualitatively different navigation behaviors from a single decomposition — no hand-coded FSM needed.

### 2.3 Conservation-by-Construction for Game Physics

The identity `dₖ₊₁ ∘ dₖ = 0` gives conservation laws for free:
- `curl(grad) = 0`: A gradient field never has circulation — no energy creation
- `div(curl) = 0`: A curl field never has divergence — no flux creation

Applied to game physics:
- Resource flow: Total resources in = total resources out (mass conservation)
- Damage propagation: No damage appears from nowhere
- NPC economy: Gold in circulation is conserved modulo sinks/sources

Current code has `FlowField` which is just velocity vectors — no conservation guarantee. DEC operators would give conservation by construction.

### 2.4 Spectral Hierarchy as Hodge Spectrum

Our `spectral_hierarchy.rs` already computes eigenvalue decomposition for quality metrics. TNO shows the Hodge Laplacian Δₖ has a spectral structure where:
- Low eigenvalues → smooth/conservative fields (exact + coexact)
- Zero eigenvalues → topological modes (harmonic, count = Betti number βₖ)

This means our existing spectral infrastructure can directly compute:
- Betti numbers (topological invariants of the game map)
- Harmonic navigation routes (routes guaranteed by topology)
- Spectral quality of flow fields (how "physical" is the NPC navigation?)

### 2.5 DEC Operators as Pruner Features

The `ConstraintPruner` trait already validates token sequences. DEC operators provide new pruning features:
- **Hodge residual:** `||Δₖu - f||` — how well does a proposed game state satisfy physical constraints?
- **Exact/coexact ratio:** Is the NPC navigation mostly conservative or mostly circulating?
- **Harmonic energy:** How much of the game state is topologically constrained?

These become new `is_valid()` and `relevance()` signals for the pruner.

---

## 3. Distillation to katgpt-rs Architecture

### 3.1 What Maps Directly

| TNO Concept | katgpt-rs Component | Action |
|-------------|---------------------|--------|
| Cell complex K | Game grid/terrain | Already implicit — make explicit |
| 0-cochain C₀ | `LeoPotentialGrid::potential` | Already exists |
| 1-cochain C₁ | `FlowField::flow` | Already exists, but untyped |
| Gradient d₀ | `LeoPotentialGrid::gradient()` | Replace with proper DEC d₀ |
| Dirichlet Energy | `dirichlet.rs` | Already exists — extend to Hodge energy |
| Spectral hierarchy | `spectral_hierarchy.rs` | Extend to Hodge spectrum |
| Betti numbers | NEW | Compute from Hodge Laplacian zero eigenvalues |
| Flow field | `flow/mod.rs` | Extend to multi-rank cochain fields |

### 3.2 What's New (Novel Fusion)

| Component | Description | LOC Estimate |
|-----------|-------------|-------------|
| `CochainField` struct | Typed multi-rank cochain on cell complex | ~200 |
| `dec_operators` module | d₀, d₁, d₂, δₖ, Δₖ as sparse matrix ops | ~400 |
| `hodge_decomposition` | Exact/coexact/harmonic projector | ~300 |
| `betti_numbers` | Topological invariants from Δₖ spectrum | ~100 |
| `DecFlowField` | Hodge-decomposed NPC navigation | ~300 |
| SIMD kernels for DEC ops | Sparse matrix-vector multiply with SIMD | ~200 |

### 3.3 Feature Gate Strategy

| Feature | Gate | Why |
|---------|------|-----|
| `dec_operators` | katgpt-rs (open) | Generic math operators |
| `hodge_decomposition` | katgpt-rs (open) | Generic spectral decomposition |
| `dec_flow_field` | katgpt-rs (open) | Generic spatial reasoning |
| Game-specific cochain types | riir-ai (private) | Game domain knowledge |
| DEC pruner features | katgpt-rs (open) | Generic pruner signals |

---

## 4. GOAT Pillar Assessment

### Is It a "Super GOAT" (Keep Secret)?

**No.** The paper is publicly available. DEC operators are well-known mathematics. Our novel fusion (DEC for game spatial reasoning) is architecturally interesting but not patentable. The game-specific cochain definitions (what counts as a "face" or "edge" in each game) IS private domain knowledge.

### Verdict by 003 Strategy

| Criterion | Assessment |
|-----------|-----------|
| Fits engine/fuel split? | ✅ DEC ops = engine (MIT), game cochains = fuel (private) |
| Block anything? | ❌ No blocking dependency |
| GOAT gate candidate? | ✅ Feature flag `dec_operators`, A/B vs naive gradient |
| LoRA needed? | ❌ Pure inference-time, no training |
| riir-ai domain? | Game cochain definitions, DEC pruner features |

**Decision: GAIN.** Implement as modelless feature in katgpt-rs with GOAT gate. The DEC operators are MIT (engine), game-specific cochain schemas are riir-ai (fuel).

---

## 5. Performance Considerations

- DEC operators are sparse matrix-vector multiplies — O(nnz) per operation
- For game grids (bounded degree), this is O(n) where n = number of cells
- Hodge decomposition needs eigendecomposition — O(n²) naive, but we only need harmonic component (low-rank)
- SIMD acceleration for sparse matmul is straightforward
- Pre-compute incidence matrices once, reuse per frame — zero allocation in hot loop

### CPU/SIMD/GPU Routing

| Op Size | Backend | Threshold |
|---------|---------|-----------|
| n < 1K cells | CPU (scalar) | Under threshold |
| 1K < n < 10K | SIMD | Game grids |
| n > 10K | GPU | Large maps |

---

## 6. What NOT to Do

1. **Don't implement full TNO with learnable weights.** That's riir-ai territory (model-based).
2. **Don't add DEC as a "reasoning mode".** It should replace/augment existing spatial ops transparently.
3. **Don't pre-compute Hodge decomposition every frame.** Cache and invalidate on topology change only.

---

## Research Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ DEC for game spatial reasoning is novel fusion |
| Rigor | ⭐⭐⭐⭐⭐ Mathematical framework is rigorous |
| Relevance to us | ⭐⭐⭐⭐⭐ Directly applicable to flow fields, spatial reasoning, conservation |
| Actionability | ⭐⭐⭐⭐ DEC ops are well-defined, implementation straightforward |
| Risk | ⭐⭐ Low — DEC is proven math, we just apply it differently |

**Bottom line:** The DEC operators are a perfect fit for our modelless spatial reasoning. They replace ad-hoc gradient/flow computations with structured, conservation-guaranteed alternatives. The Hodge decomposition gives NPCs qualitatively different navigation behaviors from a single computation. Feature gate as `dec_operators`, validate against naive gradient in arenas.

---

## Related Internal Research

| Research | Connection |
|----------|-----------|
| 111 (Analogical Reasoning) | Dirichlet Energy as structural alignment metric |
| 149 (dirichlet.rs) | Already implemented — extend to Hodge energy |
| 051 (Deep Manifold) | Fixed-point boundary conditions → related geometric structure |
| 039 (SpectralQuant) | Eigenbasis alignment ≈ Hodge spectral alignment |
| 106 (Shock Confidence) | PDE verification → DEC operators verify physics constraints |
| 135 (Parallax) | Parameterized local linear attention → similar to linear TNO layer |
| 212 (Gemini Fourier LatCal) | Fourier + lattice calculus → DEC is the discrete version of this |

TL;DR: TNO's DEC operators give us conservation-by-construction game spatial reasoning. Implement as modelless feature with GOAT gate. The math is open, game cochains are private.
