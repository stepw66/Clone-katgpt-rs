# Research 405: Topological Neural Operators (TNOs)

> **Source:** Bastian, Leventhal, Hajij, Birdal — "Topological Neural Operators", arXiv:2606.09806, Jun 2026 (Imperial College / USF). Project page: https://circle-group.github.io/research/TNO
> **Date:** 2026-07-10
> **Status:** Done
> **Related Research:** 219 (TNO → DEC operators), 296 (Stokes/DEC vocabulary crosswalk), 307 (FNO practical perspective — TNO subsumes FNO as a rank-0 special case per Prop. 4.2), 291 (Cross-Resolution Spectral Transport — the Super-GOAT that emerged from the same NO paper family)
> **Related Plans:** 251 (DEC operators — ships `d`, `δ`, `Δ`, `hodge_decompose`), 314 (Stokes-calculus wrappers), 413 (multiscale V-cycle — the HTNO coarse-correction analog)
> **Classification:** Public

---

## TL;DR

TNOs lift neural operator (NO) learning from point fields (0-cochains on vertices) to **typed cochain fields on cell complexes** — vertices, edges, faces, volumes — coupled by Discrete Exterior Calculus. The headline value is the **learned** PDE-surrogate operator (`W`, `φ_θ` channel mixing) that respects DEC structure, beating FNO/GNO/MGN on irregular-geometry PDE benchmarks. The modelless-transferable substrate — the fixed DEC operators (`d_k`, `δ_k`, `Δ_k`, `hodge_decompose`, `harmonic_projector`) that route information "where it flows" — **already ships** in `katgpt-dec` (Plan 251/314).

**Verdict: Pass** for the trained operator (→ riir-train; the channel-mixing `W`/`φ_θ` is learned via backprop with no closed-form deterministic construction — confirmed by §3.5 check) **+ Gain-tier architectural validation** that our typed-cochain discipline and DEC substrate are the correct substrate for topologically-structured operator learning. No new modelless primitive; the entire "where information flows" half that the paper distills is already shipped. One fusion idea (modelless cross-rank coupling with direction-vector mixing) is noted but unproven — needs a PoC before any verdict.

---

## 1. Paper Core Findings

### 1.1 The central design principle: decouple WHERE from HOW

> "The key design principle is to decouple **where information flows**, as governed by fixed topological operators, from **how it is transformed** (which is learned)."

- **WHERE (fixed, modelless):** the DEC operators of the cell complex — `d_k = B_{k+1}^⊤` (exterior derivative, rank k→k+1), `δ_k = M_{k-1}^{-1} B_k M_k` (codifferential, rank k→k-1), `Δ_k = δ_{k+1} d_k + d_{k-1} δ_k` (Hodge Laplacian), plus `hodge_decompose` (exact ⊕ harmonic ⊕ coexact). Topology fixes incidence; Hodge stars weight it. The identity `d_{k+1} ∘ d_k = 0` (≡ `curl(grad)=0`, `div(curl)=0`) holds **by construction** on the oriented cell complex — not learned.
- **HOW (learned, training-side):** the channel-mixing matrices `W` and nonlinear update `φ_θ` applied to the DEC-routed features (paper Eq. 9). The trainable objects are exactly `W` and `φ_θ`; the transport maps themselves are fixed DEC operators. The "copresheaf variant" makes the per-incidence fiber maps learnable but keeps the incidence support fixed.

### 1.2 The TNO layer (Eq. 9–10)

The rank-k block update:
```
H_out^k = φ_θ( d_{k-1} H^{k-1} W_↓,   // exact channel (potential-driven, from below)
                Δ^k↑ H^k W_Δ↑,         // upper Hodge-Laplacian (curl-type, via k+1 cells)
                Δ^k↓ H^k W_Δ↓,         // lower Hodge-Laplacian (div-type, via k-1 cells)
                H^k,                    // self channel
                δ_{k+1} H^{k+1} W_↑ )   // coexact channel (divergence-free, from above)
```
The four DEC channels land in the four Hodge components (`im(d_{k-1})`, `im(δ_{k+1})`, `ker(Δ_k)`, residual). The paper (App. D) gives a **structural analogy** (not a convergence theorem) to Hiptmair Hodge-compatible multigrid smoothing: the layer has the same typed correction paths, with learned channel maps in place of fixed relaxation parameters.

### 1.3 FNO/GNO as rank-0 special cases (Props 4.2, 4.3)

- **FNO** = rank-0 spectral TNO whose same-rank channel is diagonal in the Fourier basis (Prop 4.2).
- **GNO** = rank-0 graph-quadrature TNO with a learned incidence-supported kernel message on directed rank-1 cells (Prop 4.3).
Both are subsumed when all fields are forced to rank 0 (vertices only) — collapsing physical quantities to nodes, which loses geometric type.

### 1.4 HTNO — learned coarse complexes, V-cycle (App. D.5)

Hierarchical TNOs arrange TNO blocks as a learned V-cycle over coarsened complexes `K_0 ← K_1 ← … ← K_L`, connected by degree-preserving transfer maps `R, Π`. The FEEC ideal is that transfers **commute with the coboundary** (`d R = R d`, `d Π = Π d`), preserving the de Rham structure across levels. The implementation uses soft-Voronoi / k-means partitions (learned or fixed), not exact commuting-diagram transfers. This is a learned two-grid method; structurally analogous to Hodge-compatible multigrid, but **not** a convergence theorem.

### 1.5 Key empirical results

- Beats RIGNO-18/MGN/Geo-FNO/FNO/GINO/UPT on Poisson-Gauss, Elasticity, EmmiWing (Tab 1–2).
- **Anisotropic Darcy ablation (§5.3, the modelless-relevant insight):** ingesting a face-valued (rank-2) orientation field *natively* beats vertex-projecting it by ~0.5pp, because vertex projection is lossy by construction (the iid-per-face orientation averages to ~0 at every vertex). The architectural gain (TNO vs MPNN, ~4–5pp) is independent of the ingestion rank; native rank-2 adds another ~0.5pp. **Lesson: physical quantities have geometric type; collapsing them to vertices loses information.**
- **Component ablation (§5.4):** the harmonic basis is the load-bearing component (+6.8pp Darcy, +7.9pp Adv-Diff); sheaf transport is synergistic with it (helps when harmonic is on, hurts when off).

---

## 2. Distillation

### 2.1 §3.5 modelless-unblock check (MANDATORY — the channel mixing looks training-only; verify)

**Gate:** "TNO-quality cross-rank coupling appears to need training (the channel-mixing `W`, `φ_θ`, and the learned coarse complexes)."

→ Does the failure (no cross-rank coupled channel mixing) have a SYSTEMATIC, characterizable cause?
- The "failure" is the absence of a learned constitutive law. This is not a bias or offset; it is a genuinely absent capability (PDE-specific feature transformation). **Not systematic in the §3.5 sense** (§3.5 targets correctable biases like "signal doubled" or "position offset", not absent learned functions).

→ Can freeze/thaw (path 1) fix it? — **No.** No frozen snapshot produces a learned PDE constitutive law.

→ Can a deterministically constructed reader/writer LoRA (path 2) fix it? — **No.** The channel mixing is PDE-specific (anisotropic Darcy conductivity, Maxwell coupling constants, etc.). No closed-form deterministic construction exists; the paper random-inits and trains via backprop.

→ Can a latent-space projection/gate (path 3) fix it? — **Partially, but this is a different (weaker) operator.** A dot-product projection + sigmoid gate onto direction vectors could substitute for `W`/`φ_θ`, yielding a *modelless* cross-rank coupling layer. But the paper's entire value proposition is that LEARNED mixing outperforms fixed mixing on PDE surrogacy. The modelless analog is a new, unvalidated operator — not a correction to a bias. **This is a fusion idea (§2.5), not a §3.5 unblock.**

**§3.5 verdict: genuine riir-train dependency for the trained TNO operator.** The modelless cross-rank coupling layer (path 3 analog) is a *separate, unproven* primitive tracked in §2.5, not a deferral-blocker — because there is no blocked modelless gate to unblock; the paper's value is the trained operator itself.

### 2.2 What already ships (verified grep this session, 2026-07-10)

Vocabulary translation (paper → shipped code), then grep `katgpt-dec/src/**/*.rs`:

| Paper term | Codebase term | Ships? | Location |
|---|---|---|---|
| exterior derivative `d_k` | `exterior_derivative` / `exterior_derivative_into` | ✅ | `operators.rs` L48/L68 |
| codifferential `δ_k` | `codifferential` / `codifferential_into` | ✅ | `operators.rs` L122/L142 |
| Hodge Laplacian `Δ_k` | `hodge_laplacian` / `hodge_laplacian_into` | ✅ | `operators.rs` L195/L232 |
| Hodge decomposition (exact⊕harmonic⊕coexact) | `hodge_decompose` / `hodge_decompose_cached` | ✅ | `hodge.rs` L331; `cache.rs` L186 |
| harmonic projector `P^harm_k` | `harmonic_projector` | ✅ | `hodge.rs` L536 |
| Betti numbers `β_k` | `betti_numbers` | ✅ | `hodge.rs` L407 |
| typed cochain field `C^k(K; R^d)` | `CochainField { rank, n_cells, dim, data }` | ✅ | `katgpt-dec` |
| `d∘d=0` by construction | enforced; tests verify `curl(grad)=0`, `div(curl)=0` | ✅ | Plan 251 |
| HTNO V-cycle (coarse correction) | `htno_v_cycle`, `prolongate` | ✅ | Plan 413 |
| typed game cochains | `terrain_cochains` (Safety/Threat/Occupancy/Destruction) | ✅ | Plan 251 |

**The entire "where information flows" half — the modelless residue — ships.** The paper's contribution is the "how it is transformed" half (`W`, `φ_θ`, learned coarse complexes), which is training-side.

### 2.3 The one gap: cross-rank *coupled mixing* (not pure transport)

`grep "cross.?rank|coupl.*(rank|degree)|multi.?rank.*mix"` against `katgpt-dec/src/**/*.rs` → **0 matches.**

- We ship **pure cross-rank transport**: `exterior_derivative` (0→1, the geometric gradient), `codifferential` (1→0, the geometric divergence). These are fixed, topology-only.
- We do **not** ship a layer that *simultaneously* couples rank-k to ranks k±1 via d/δ/Δ with per-channel mixing (the TNO Eq. 9 pattern). That layer is the trained operator.

This gap is **not modelless-actionable without a use case + PoC** (§2.5). For our game fields, the pure DEC transport already produces the cross-rank maps we need (threat gradient → flow, occupancy divergence → flux). A coupled mixing layer would add learned constitutive laws, which is training-side.

### 2.4 §1.5 insight (the genuinely transferable lesson)

**"Physical quantities have geometric type; collapsing them to vertices loses information"** (§5.3 anisotropic Darcy ablation). Our `terrain_cochains` already follow this discipline — Safety/Threat/Occupancy are typed 0-cochains, and the DEC operators move them between ranks type-correctly. The paper's ablation **empirically validates** a design we already follow architecturally. This is architectural validation, not a new primitive. Per §3.6, this is a pure architectural redirect (no quality-parity claim against our runtime), so no PoC is required.

### 2.5 Fusion idea (novelty TBD — needs PoC before any verdict, NOT a candidate)

**Idea:** ship a *modelless* cross-rank coupling layer — TNO Eq. 9 with `W`/`φ_θ` replaced by pre-computed direction-vector projections + sigmoid gates. Input: cochains at ranks {k-1, k, k+1}. Output: updated rank-k cochain. Routing: fixed DEC operators. Mixing: dot-product + sigmoid onto frozen direction vectors (the existing `project_to_scalars` pattern).

**Why it might matter:** a coupled threat×occupancy→safe-flow field where the coupling isn't just `d(threat)` but a gated blend of `d(threat)`, `δ(occupancy)`, `Δ(threat)`. Currently computed as separate primitives; a coupled layer could be cheaper (one pass) and expose the Hodge components as explicit channels.

**Novelty gate Q1–Q4 (the reason this is NOT a candidate yet):**
- **Q1 (novel combo?):** YES — no shipped layer couples ranks via d/δ/Δ with modelless mixing.
- **Q2 (modelless constructible?):** YES mechanically, but **the use case is unproven.** Our game fields don't currently need coupled constitutive laws (threat→flow is pure `d`; occupancy→flux is pure `δ`). The coupling only pays off when two rank-different fields interact nonlinearly, and we have no such consumer identified.
- **Q3 (beats existing?):** MOOT — no benchmark exists for "coupled game-field cochain mixing."
- **Q4 (measurable win?):** MOOT — same reason.

**Verdict: closed, not pursued.** Q2–Q4 cannot be answered without (a) a concrete game-system consumer that needs cross-rank nonlinear coupling, and (b) a PoC showing the coupled layer beats computing the transports separately. Per the no-"candidate"-escape-hatch rule, this is filed as a tracked fusion idea, **not** a Super-GOAT candidate. Re-open if a consumer appears (e.g., a game system where two rank-different fields must interact nonlinearly at 20Hz).

---

## 3. Verdict

**Tier: Pass** (training-side operator) **+ Gain** (architectural validation of the typed-cochain discipline).

**One-line reasoning:** TNO's headline value is the TRAINED PDE-surrogate operator (`W`, `φ_θ` channel mixing + learned coarse complexes) — training-side, → riir-train per §3.5 (no deterministic construction; §3.5 paths 1/2 fail, path 3 yields a *different weaker operator* not a bias correction). The modelless residue — the fixed DEC operators `d`, `δ`, `Δ`, `hodge_decompose`, `harmonic_projector`, typed `CochainField`, the V-cycle — **already ships** in `katgpt-dec` (Plan 251/314/413), verified by grep this session. The paper's §5.3 ablation empirically validates the typed-cochain discipline our `terrain_cochains` already follow.

### Why NOT Super-GOAT / GOAT (novelty gate)

- **Q1 (no prior art?):** NO. The DEC operators (the modelless half) are prior art — we ship them (Plan 251, Research 219/296). The novel half (learned channel mixing) is training-side.
- **Q2 (new capability class?):** NO at the modelless level. Cross-rank pure transport ships; cross-rank coupled mixing is the trained operator. The modelless coupling layer (§2.5) is an unproven fusion idea, not a shipped-capability gap.
- **Q3 (product selling point?):** NO. "We use DEC operators" is not a moat — they're public substrate we already ship.
- **Q4 (force multiplier?):** NO. Single paper; the substrate it would multiply already exists.

All NO → NOT Super-GOAT, NOT GOAT. Gain for the architectural-validation insight; Pass for the trained operator.

### MOAT gate (katgpt-rs domain)

- **In scope:** neutrally — the paper-derived fundamental primitive (DEC operators) already ships. The architectural insight (typed cochains) is already applied in `terrain_cochains`.
- **Strengthen moat:** no. The DEC substrate is public (Plan 251); TNO validates it but adds no private moat.
- **Promote/demote:** no change. No stack-slot change.

### Routing

| Component | Destination | Rationale |
|---|---|---|
| Trained TNO operator (`W`, `φ_θ`, learned coarse complexes) | **→ riir-train** | Training method (backprop against PDE/LM loss). §3.5 paths 1/2 fail; path 3 yields a different operator, not a bias correction. |
| DEC operators (d, δ, Δ, hodge_decompose, harmonic_projector) | **Already ships** (katgpt-dec, Plan 251/314) | The modelless "where information flows" substrate. No action. |
| HTNO V-cycle | **Already ships** (`htno_v_cycle`, Plan 413) | The coarse-correction analog. No action. |
| Typed-cochain discipline | **Already applied** (`terrain_cochains`) | Architectural validation of existing design. No action. |
| Modelless cross-rank coupling layer (fusion idea) | **CLOSED** (novelty gate Q2–Q4 MOOT) | Needs a consumer + PoC before any verdict. Tracked in §2.5; re-open if a game system needs nonlinear cross-rank coupling. |

---

## 4. What to implement (modelless, katgpt-rs)

**Nothing.** No new modelless primitive. The DEC substrate ships; the trained operator → riir-train; the fusion idea (§2.5) is closed pending a consumer. This note exists to (a) record the Pass/Gain verdict, (b) document that the modelless residue ships (so a future reader doesn't re-distill this paper thinking the DEC operators are missing), and (c) close the cross-rank coupling fusion idea with the explicit Q1–Q4 reasoning.

---

## TL;DR

TNOs are TRAINED neural operators for PDE surrogacy that route information between typed cochain ranks via fixed DEC operators. **Verdict: Pass** (trained operator → riir-train; §3.5 confirms no deterministic construction for the channel mixing) **+ Gain** (the typed-cochain discipline and DEC substrate the paper validates already ship in `katgpt-dec` — Plan 251/314/413 — verified by grep). The modelless cross-rank coupling fusion idea (§2.5) is closed: Q2–Q4 are MOOT without a concrete game consumer that needs nonlinear cross-rank field coupling, and our current game fields (threat→flow via `d`, occupancy→flux via `δ`) need only the pure transport that already ships. No new primitive; no plan.
