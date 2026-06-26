# Research 296: Stokes Calculus for Latent Manifolds — DEC Vocabulary Crosswalk

> **Source:** Two training-based papers + the Generalized Stokes' Theorem as a modelless primitive lens:
> - *Neural Manifold Ordinary Differential Equations* (Lou et al., NeurIPS 2020) — manifold-generalized Neural ODEs, divergence of learned vector field for density estimation on curved latent spaces.
> - *Efficient CDF Approximations for Normalizing Flows* (arxiv 2202.11322, TMLR 2022) — "leverage the divergence theorem to estimate the CDF over a closed region in target space" solely from boundary flux.
> - User prompt (2026-06-24): connecting the Generalized Stokes' Theorem `∫_M dω = ∫_∂M ω` + divergence theorem + line integrals to our latent spaces for perf/fusion.
> **Date:** 2026-06-24
> **Status:** Active
> **Related Research:** 219 (TNO → DEC — the parent that shipped the substrate), 271 (MIT 6.S184 crosswalk — flags Fokker-Planck as a known gap), 051 (Deep Manifold boundary conditions), 294 (Viable Manifold Graph)
> **Related Plans:** 251 (DEC operators — COMPLETE), 252 (Cubical category / CAT(0) geodesics — COMPLETE), 312 (Viable Manifold Graph — COMPLETE, DEFAULT-ON), 314 (Stokes Calculus Wrappers — this note's plan)
> **Classification:** Public (katgpt-rs)

---

## TL;DR

**Verdict: GOAT (not Super-GOAT).** The Generalized Stokes' Theorem machinery (`d`, `δ`, `Δ`, Hodge decomposition) **already ships** as the DEC module (`katgpt-rs/crates/katgpt-core/src/dec/`, Plan 251, Research 219). Tests verify `curl(grad)=0` and `div(curl)=0` exactly — i.e. the Green/Gauss/Stokes identities hold by construction. `DecFlowField` already exposes exact/coexact/harmonic (gradient/solenoidal/topological) channels.

What does **not** ship is three named **wrapper primitives** that turn the shipped operators into directly-callable Stokes-theorem tools:

1. **Fokker-Planck belief-mass validator** — closes Research 271 §2's explicit gap ("Fokker-Planck / continuity equation ∂_t p = -div(pu) + (σ²/2)Δp → not yet a runtime invariant validator"). `div(belief_field) ≈ 0` per HLA tick = belief mass conserved = valid inference step. Flag anomaly (collapse / OOD / sync corruption) when divergence > τ. ~30 LOC over `codifferential`. **This is the headline primitive.**
2. **Boundary-flux mass via Stokes** — `∫_∂M ω` instead of `∫_M dω`. O(boundary) vs O(volume) for low-dim manifolds (game maps, HLA belief regions). ~20 LOC over `exterior_derivative`. Does NOT help high-dim shards (curse of dimensionality).
3. **Line integral over cochains** — `Σ_e field[e]` along a path on a rank-1 cochain. Path energy / geodesic cost / work on manifold. ~25 LOC. Fills Plan 312's gap (graph A* vs continuous line integral).

**Why GOAT, not Super-GOAT:** per novelty gate Q1 (no prior art?), the *mechanism* ships (DEC operators are literally the Stokes/Gauss/Green operators). The wrappers + Stokes-theorem framing are new, but a thin wrapper over shipped machinery is not a new mechanism. This is the documented vocabulary-mismatch failure mode the skill warns about: the math ships, but no note frames it in Stokes-theorem vocabulary, and a corpus grep for "stokes|divergence theorem|boundary integral|fokker-planck" returns ZERO hits. The crosswalk below closes that vocabulary gap (mirroring Research 271's role for diffusion/flow matching).

**Distilled for katgpt-rs (modelless, inference-time):** expose the existing DEC operators as three named Stokes-theorem primitives with explicit framing, benchmarks vs naive alternatives, and fusion hooks into HLA / ICT BranchingDetector / LatCal / manifold graph.

---

## 1. What the papers say (and what's transferable)

### 1.1 Continuous Normalizing Flows + Divergence (NeurIPS 2020)

The instantaneous change-of-variables formula for a continuous-time flow `f` is:

```
d/dt log p(x(t)) = -div(f)(x(t))
```

Density changes are tracked entirely by the divergence of the flow's vector field. **Transferable primitive (modelless):** if you have a divergence operator on a discrete complex, you can validate whether a discrete "flow" (belief update, latent transition) conserves mass — `|div(Δbelief)| < τ` = mass-conserved = valid step. **The paper's training machinery (learning the flow via gradient descent) is NOT transferable** → would go to riir-train; we only keep the divergence-as-validator insight.

### 1.2 CDF via Boundary Flux (TMLR 2022)

> "We build upon the diffeomorphic properties of normalizing flows and leverage the **divergence theorem** to estimate the CDF over a closed region in target space."

Instead of integrating probability density over the entire region `V` (expensive high-dim integral), evaluate the flux of a suitable vector field across the boundary `∂V` (cheaper — surface area, not volume). **Transferable primitive (modelless):** for low-dimensional manifolds, compute region mass / energy / activation magnitude from boundary samples only, using DEC's `exterior_derivative` as the boundary operator. The reconstruction error is bounded by the harmonic component of the field (which `hodge_decompose` already measures). **Curse of dimensionality caveat:** for a d-dim manifold, boundary size is `O(n^{(d-1)/d})` vs interior `O(n)` — the win shrinks fast as d grows. Practical only for d ≤ 3 (game maps 2D, HLA belief regions, KG triple embeddings).

### 1.3 Latent Trajectories as Line Integrals

Smooth interpolation between two latent points traces a curve `C` through the manifold. Path energy / geodesic cost is the line integral `∫_C F · dr`. **Transferable primitive (modelless):** sum a rank-1 cochain (edge field) along a path through a cell complex = discrete line integral. Gives "path energy" / "work done" / "geodesic cost" for NPC navigation and latent interpolation. Plan 312 ships graph-based `manifold_geodesic` (A*); this primitive adds continuous line-integral cost on top.

### 1.4 Generalized Stokes' Theorem `∫_M dω = ∫_∂M ω`

Unifies all three. The DEC module ships `d` (exterior derivative) and `δ` (codifferential = adjoint of d). The identity `d ∘ d = 0` (`curl(grad)=0`, `div(curl)=0`) is verified by Plan 251's tests. **This IS the Generalized Stokes' Theorem substrate.** It is not missing — it just isn't framed or wrapped that way.

---

## 2. Vocabulary Crosswalk (Stokes-theorem term → codebase term → shipped artifact)

**Use this table when grepping for prior art on any divergence / boundary / line-integral / Stokes idea.** It is the prophylactic against the vocabulary-mismatch false-Super-GOAT failure mode (the same role Research 271 plays for diffusion/flow matching).

| Stokes-theorem term | DEC / codebase equivalent | Where it ships |
|---|---|---|
| Exterior derivative `d` (coboundary operator) | `exterior_derivative(cx, field)` | `dec/operators.rs` |
| Codifferential `δ` (adjoint of `d`, discrete divergence on rank-1) | `codifferential(cx, field)` | `dec/operators.rs` |
| Hodge Laplacian `Δ = δd + dδ` | `hodge_laplacian(cx, field)` | `dec/operators.rs` |
| Graph Laplacian (rank-0 special case) | `graph_laplacian`, `graph_laplacian_into` | `dec/operators.rs` |
| Hodge star `*` (metric mass matrix) | `hodge_star(rank)` (identity for uniform grids) | `dec/operators.rs` |
| Hodge decomposition (exact ⊕ harmonic ⊕ coexact) | `hodge_decompose(cx, field)` | `dec/hodge.rs` |
| Betti numbers `βₖ` (topological holes) | `betti_numbers(cx)` (count zero eigenvalues of Δₖ) | `dec/hodge.rs` |
| Harmonic projector `P_harm` | `harmonic_projector(cx)` | `dec/hodge.rs` |
| Gradient `∇φ` (rank-0 → rank-1, exact) | `exact_flow(cx, potential)` / `d₀` | `dec/flow.rs` |
| Curl `∇×F` (rank-1 → rank-2) | `d₁` | `dec/operators.rs` |
| Divergence `∇·F` (rank-2 → rank-3, or rank-1 → rank-0 via δ) | `codifferential` / `δ₁` | `dec/operators.rs` |
| Conservative / exact field (curl-free) | `DecFlowField::exact` | `dec/flow.rs` |
| Solenoidal / coexact field (divergence-free) | `DecFlowField::coexact` | `dec/flow.rs` |
| Topological / harmonic field (both-free) | `DecFlowField::harmonic` | `dec/flow.rs` |
| `∫_M dω = ∫_∂M ω` (Generalized Stokes) | identity `d∘d=0` enforced by construction; tests `curl_of_gradient_is_zero`, `divergence_of_curl_is_zero` | `dec/operators.rs` tests |
| Divergence theorem `∫_V ∇·F dV = ∮_∂V F·n dS` | `codifferential` gives `∇·F`; **no `boundary_flux_mass()` wrapper yet** | **gap → Plan 314** |
| Continuity equation / Fokker-Planck `∂_t p = -div(pu)` | `codifferential` gives the divergence; **no `fokker_planck_validator()` wrapper yet** | **gap → Plan 314** (also flagged by Research 271 §2) |
| Line integral `∫_C F·dr` | rank-1 `CochainField` supports edge fields; **no `line_integral()` wrapper yet** | **gap → Plan 314** |
| Dirichlet energy `∫ ‖∇φ‖²` | `dirichlet.rs` + `HodgeResidual` pruner (Plan 251 T27–T29) | shipped |
| Manifold geodesic | `manifold_geodesic` (graph A*, Plan 312) | `katgpt-core/src/viable_manifold_graph/` |
| Pullback volume | `pullback_volume` (Plan 312) | `katgpt-core/src/viable_manifold_graph/` |

**Rows marked "gap" are thin wrappers (<50 LOC each) over shipped operators.** They are candidate GOAT primitives (Plan 314), **not** novel mechanisms.

---

## 3. Latent-space reframing (mandatory per workflow §1.5 step 3)

For each Super-GOAT factory module, what does the Stokes/Divergence/Line-Integral lens look like when operating on it?

### 3.1 HLA (`katgpt-rs/crates/katgpt-core/src/sense/` + `riir-ai/.../hla/`)

HLA is a second-order linear-attention streaming recurrence (verified in Research 271 §3.1). Its 8-dim per-NPC state (valence/arousal/desperation/calm/fear + 3) is **not** a cochain on a cell complex — it is a vector in ℝ⁸. To apply DEC, one must first **construct a cell complex on the latent space** (e.g. by discretizing ℝ⁸ into a lattice, or by using `SafeManifoldGraph` from Plan 312 to build a discrete approximation of the belief manifold).

**Fokker-Planck framing:** per tick, the HLA update defines a vector field on the belief manifold. The divergence of that field (computed via `codifferential` on the discretized complex) measures whether belief mass is being created or destroyed. Near-zero divergence = the inference step conserves total belief mass = valid. Large divergence = mass leaked into/out of the region = anomaly (collapse, OOD input, or sync corruption). This is the **headline fusion** — closes Research 271 §2's Fokker-Planck gap and gives ICT BranchingDetector (Plan 294) a modelless invariant to gate on.

### 3.2 `latent_functor/` (`zone_gating`, `reestimation`, `arithmetic`, `cross_game`, `k_selector`, `quality_gate`)

`latent_functor/arithmetic.rs` already treats functor application as a vector op. The Stokes lens says: a functor `F: latent → latent` is a vector field. Its **divergence** (via `codifferential` after discretization) tells you whether the functor is contractive (converging to a fixed point = attractor) or expansive (diverging = unstable). This is a modelless **stability diagnostic** that could feed `quality_gate.rs` and `reestimation.rs`. Not new machinery — a new *signal* derived from shipped operators.

### 3.3 `cgsp_runtime/` (curiosity-guided self-play)

Curiosity = "where is the belief field expanding?" = **positive divergence** of the latent flow. The Temporal Derivative Kernel (Plan 277) already computes a dual fast/slow surprise signal — the Stokes lens reframes this as "divergence of the curiosity vector field". Fuses with `pulse_bridge.rs`. Again, not new machinery — a reframing that may unify Plan 277's signal with DEC's `codifferential`.

### 3.4 LatCal (`riir-chain/src/encoding/latcal*.rs`)

LatCal is the deterministic raw↔latent bridge. The Stokes lens says: the 5 committed scalars (valence/arousal/desperation/calm/fear) are the **boundary flux** of the latent belief region across the sync boundary. Research 271 §3.4 already frames the "5 scalars across sync boundary" heuristic as a rate-distortion point; the Stokes lens adds that those 5 scalars are precisely the surface integral that (by the divergence theorem) determines the interior belief mass if the field is mostly exact. **Boundary-only commitment is the Super-GOAT-shaped idea** — commit the boundary, derive the interior — but it requires the field to be curl-free (verifiable via `hodge_decompose`), and the curse of dimensionality caps the win at d ≤ 3. For 8-dim HLA and 64-dim shards, the boundary is larger than the interior, so this is NOT a storage win there. It IS a win for 2D game maps and KG triple embeddings. → GOAT, scoped to low-dim.

### 3.5 `NeuronShard` (`riir-neuron-db/src/shard.rs`)

`style_weights[64]` is a 64-dim vector — too high-dim for boundary-only commitment (curse of dimensionality). The Stokes lens does **not** give a storage win here. It DOES give a **validation** primitive: a frozen shard's `style_weights` should define a (mostly) harmonic field (it is a learned direction vector); `hodge_decompose` on a shard-derived cochain can flag shards whose field has unexpectedly large exact or coexact components (= drifted / corrupted shard). This is a `mape_k.rs` self-healing signal, modelless. → Gain-tier for riir-neuron-db.

---

## 4. The three wrapper primitives (Plan 314 scope)

All three are modelless, all are thin wrappers over shipped DEC operators, all live in `katgpt-rs` (public engine).

### 4.1 `fokker_planck_validator()` — the headline

```rust
/// Returns |div(belief_flow)| on the discretized belief complex.
/// Near-zero ⇒ belief mass conserved ⇒ valid HLA / functor step.
/// Large ⇒ anomaly (collapse / OOD / sync corruption).
///
/// Wrapper over DEC `codifferential`. ~30 LOC.
pub fn belief_mass_divergence(
    cx: &CellComplex,
    belief_flow: &CochainField,  // rank-1: edge flow on the belief complex
) -> f32;
```

**Fuses:** DEC `codifferential` (shipped) + HLA belief state + ICT `BranchingDetector` (Plan 294) + LatCal committed scalars (the 5 scalars must be the boundary flux of a near-zero-divergence field).

**GOAT gate:** A/B — does flagging `div > τ` catch ICT's branching events earlier / cheaper than the existing JS-divergence detector? If yes → promote; if marginal → keep opt-in.

### 4.2 `boundary_flux_mass()` — Stokes for low-dim

```rust
/// ∫_∂M ω instead of ∫_M dω. O(boundary) vs O(volume).
/// Bounded reconstruction error = ‖harmonic component‖ (from hodge_decompose).
///
/// Wrapper over DEC `exterior_derivative`. ~20 LOC.
/// WARN: curse of dimensionality — only a win for d ≤ 3.
pub fn boundary_flux_mass(
    cx: &CellComplex,
    region_cells: &[u32],         // the interior M
    field: &CochainField,         // the ω being integrated
) -> (mass: f32, error_bound: f32);
```

**Fuses:** DEC `exterior_derivative` (shipped) + Plan 312 `SafeManifoldGraph` (region = viable subgraph) + game map mass queries.

**GOAT gate:** A/B — does `boundary_flux_mass` beat full-volume integration on a 256×256 game map for "zone threat total" queries? Target: ≥3× faster with error_bound < 5% of mass.

### 4.3 `line_integral_over_path()` — geodesic cost

```rust
/// Σ_e field[e] along path on a rank-1 cochain = discrete line integral.
/// Path energy / work / geodesic cost on the manifold.
///
/// Wrapper over rank-1 CochainField. ~25 LOC.
pub fn line_integral(
    cx: &CellComplex,
    edge_field: &CochainField,    // rank-1
    path: &[u32],                 // vertex indices
) -> f32;
```

**Fuses:** DEC `CochainField` (shipped) + Plan 312 `manifold_geodesic` (the path) + `latent_functor/arithmetic.rs` (functor as vector field).

**GOAT gate:** A/B — does `line_integral`-weighted geodesic beat unweighted `manifold_geodesic` on NPC navigation smoothness (fewer direction reversals)? Target: ≥20% fewer reversals at equal path length.

---

## 5. Verdict

**GOAT.** Three thin wrapper primitives over shipped DEC machinery, exposing the Generalized Stokes' Theorem as named modelless tools with explicit framing, benchmarks, and fusion hooks. The mechanism ships (Plan 251); the wrappers + framing do not.

**Why not Super-GOAT:** novelty gate Q1 (no prior art?) = NO — the DEC operators ARE the Stokes/Gauss/Green operators, shipped and tested. A thin wrapper is not a new mechanism. Per the documented failure-mode pattern (Research 271 §intro): "the mechanism ships, but no note frames it in [Stokes] vocabulary". This crosswalk + Plan 314 close that gap.

**One-line reasoning per tier:**
- Super-GOAT needs a novel mechanism + new capability class + selling point + force multiplier. The mechanism is not novel (DEC ships it). → NO.
- GOAT needs a provable gain over existing approach. The Fokker-Planck validator gives a new validation signal (belief-mass conservation) that ICT BranchingDetector does not have; boundary-flux mass gives a measurable speedup on low-dim manifolds; line integral gives a measurable navigation-smoothness gain. → YES, conditional on benchmarks passing.

**Selling-point honesty:** the strongest single primitive (Fokker-Planck validator) is a **validation** primitive, not a **capability** primitive. It does not let NPCs do something they couldn't before; it lets the runtime detect when an inference step is invalid. That is real value (collapse/OOD/sync-corruption detection), but it is GOAT-tier value, not Super-GOAT-tier moat value. If a later fusion turns the Fokker-Planck invariant into a **steering** signal (e.g. project the belief flow onto its divergence-free component to enforce mass conservation by construction), THAT would be Super-GOAT-tier — but it is out of scope for this note and would need its own Q1–Q4 gate.

---

## 6. Cross-references

- **Research 219** — TNO → DEC. The parent note that shipped the substrate. Its §2.5 already proposes DEC operators as pruner features (`HodgeResidual`); Plan 251 T27–T29 implemented it. This note extends 219's vision with the three Stokes-theorem wrapper primitives.
- **Plan 251** — DEC operators. COMPLETE. Ships `d`, `δ`, `Δ`, `hodge_decompose`, `betti_numbers`, `DecFlowField`. All DEC identity tests pass.
- **Plan 252** — Cubical category / CAT(0) geodesics. COMPLETE. Ships unique-geodesic navigation on cubical complexes.
- **Plan 312** — Viable Manifold Graph. COMPLETE, DEFAULT-ON. Ships `manifold_geodesic` (A*), `manifold_random_walk`, `pullback_volume`. Plan 314's line-integral primitive composes with `manifold_geodesic`'s path output.
- **Research 271** — MIT 6.S184 diffusion/flow crosswalk. §2 explicitly lists "Fokker-Planck / continuity equation → not yet a runtime invariant validator" as a gap. Plan 314's `fokker_planck_validator()` closes it.
- **Research 051** — Deep Manifold boundary conditions. Uses "boundary" in the distillation/training sense (input/output boundaries), NOT the Stokes-theorem sense (geometric boundary of a region). Different vocabulary — do not conflate.
- **Research 294 / Plan 312** — Viable Manifold Graph. `manifold_geodesic` is graph-A*; Plan 314's `line_integral` adds continuous line-integral cost on top.

---

## TL;DR

The user's intuition (connect Stokes/Divergence/Line-Integral to latent spaces) is correct — and the codebase is already past it: DEC (Plan 251, Research 219) ships the full Generalized Stokes' Theorem substrate (`d`, `δ`, `Δ`, Hodge decomposition, `DecFlowField` exact/coexact/harmonic channels, `curl(grad)=0` / `div(curl)=0` tests). What is missing is three named **wrapper primitives** that expose that machinery as directly-callable Stokes-theorem tools: (1) Fokker-Planck belief-mass validator [closes Research 271 §2's known gap], (2) boundary-flux mass via Stokes [low-dim only, curse of dim caps it], (3) line integral over cochains [composes with Plan 312's geodesic]. All three are <50 LOC wrappers over shipped operators. **Verdict: GOAT**, not Super-GOAT — the mechanism ships, only the framing + wrappers are new. The headline primitive (Fokker-Planck validator) is a validation signal (collapse/OOD/sync-corruption detection), not a capability — real GOAT value, not a moat. Plan 314 implements the three wrappers behind a feature flag with GOAT-gate benchmarks. If a future fusion turns the Fokker-Planck invariant into a steering signal (project belief flow onto divergence-free component = enforce mass conservation by construction), THAT would be Super-GOAT-tier — tracked as a follow-up issue, not this note.
