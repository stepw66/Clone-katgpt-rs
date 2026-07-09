# Issue 043: Canvas Modelless Behavioral Gain PoC (Super-GOAT Re-evaluation Gate)

**Created:** 2026-07-09
**Research:** [katgpt-rs/.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md](../.research/398_Canvas_Engineering_Declared_Causal_Topology_Compiler.md) §2.5 (fusion idea)
**Plan:** [katgpt-rs/.plans/419_canvas_schema_compiler.md](../.plans/419_canvas_schema_compiler.md) (the compiler primitive that ships regardless of this PoC)
**Status:** Open — blocked on Plan 419 Phase 3 (reachability primitive) landing

---

## The question

Research 398 distills canvas engineering as **GOAT** (not Super-GOAT) because the modelless behavioral gain is unproven and the paper's own evidence shows modelless application (mask on an untrained-for-it backbone) is a **19% loss** (paper §5 calibration finding #2). The compiler + reachability guarantee ships on structural/correctness merits regardless.

But the §2.5 fusion idea — canvas × DEC reachability × `region_subspace_bridge` × freeze/thaw schema exchange × HLA per-NPC latent state → "a typed NPC cognitive stack with declared causal topology and reachability guarantees" — *might* produce a modelless behavioral gain **specifically because our NPC latent state is NOT a pretrained DiT**. Our HLA + latent_functor + region_subspace substrate is already structured; the canvas compiler would *unify and enforce* that structure with a reachability guarantee, not impose foreign structure on an untrained backbone.

**The PoC question:** does compiling the existing NPC cognitive operations (HLA affect projection, latent_functor composition, region_subspace steering) into a declared `CanvasSchema` with reachability constraints improve any measurable per-NPC behavior metric modellessly, over the un-unified constituents run independently?

## Why this is a PoC, not a plan

Per research skill §3.6 (defend-wrong PoC rule): a verdict that claims quality parity or "the fusion produces a new capability class" needs a head-to-head PoC on a controlled toy domain. The fusion claim is "new capability class: declared causal topology with reachability on per-NPC latent state." That's a quality/capability claim, not just an architectural one. The PoC defends OR refutes it.

## PoC design (three competitors minimum, per §3.6)

**Domain:** a controlled toy NPC cognition task — e.g. a 2D threat-avoidance + curiosity-seeking NPC where the cognitive stack is HLA (8-dim affect) + region_subspace (zone-conditioned steering) + latent_functor (action composition). Run on the existing `riir-ai/crates/riir-poc/` harness.

**Three competitors:**
1. **Baseline (no canvas):** HLA + region_subspace + latent_functor run independently, no declared topology, no reachability constraint. This is the current shipped state.
2. **Canvas-unified (the fusion):** the same operations compiled into a `CanvasSchema` with declared causal topology (perception → affect → action, memory ↔ affect) and reachability guarantees enforced via `can_reach` gating.
3. **Frozen/no-adaptation floor:** HLA only, no region_subspace, no functor. The degenerate baseline.

**Metrics (honest, no cherry-picking):**
- **Behavioral:** threat-avoidance success rate, curiosity coverage, action-coherence (does the NPC act consistently with its affect state?).
- **Reachability sanity:** does the canvas-unified version actually enforce the declared topology (perception cannot influence action without traversing affect)? This is a correctness check, not a perf metric.
- **Latency:** per-tick overhead of the canvas-unified path vs baseline. Per AGENTS.md, plasma-tier budget is sub-µs; the canvas compile is one-time at schema load, so per-tick overhead should be ~0 (just `can_reach` lookups).

## Verdict protocol (§3.6 honesty)

- **If canvas-unified BEATS baseline on ≥1 behavioral metric at no latency cost** → the fusion produces a modelless behavioral gain → re-evaluate Research 398 for **Super-GOAT**. Create the private guide in riir-ai (typed NPC cognitive stack selling point) + open a riir-ai plan for the runtime integration.
- **If canvas-unified MATCHES baseline** → the fusion is architectural-only (unification + reachability correctness, no behavioral gain) → Research 398 stays **GOAT**. The compiler ships (Plan 419) on structural merits. Close this issue with the negative result recorded.
- **If canvas-unified is WORSE than baseline** → the canvas overhead hurts without behavioral payoff → Research 398 downgraded to **Gain** or the compiler stays opt-in-only. Record the refutation honestly in Research 398 as a §"PoC Addendum".

**The PoC defends OR refutes.** A PoC that only confirms the verdict is weaker than one that honestly refutes part of it (§3.6). Do NOT silently revise the verdict to match the PoC — record raw numbers, then revise explicitly.

## Where the PoC lives

`riir-ai/crates/riir-poc/benches/canvas_npc_cognitive_stack_modelless.rs` — the "defend-wrong" R&D crate. Use `CARGO_TARGET_DIR=/tmp/canvas_poc` per the AGENTS.md rule; clean up when done. The PoC stays as a permanent regression check (§3.6) — its job is to keep settling the dispute if the shipped primitives are later tuned.

## Blockers

- Plan 419 Phase 3 (reachability primitive) must land first — the PoC needs `can_reach` / `reachability_horizon` to enforce the declared topology.
- The PoC is in riir-ai (private), so it consumes the katgpt-core primitive via the existing path dependency.

## Out of scope

- Training a DiT within declared topology (riir-train follow-up, separate).
- Looped attention (already distilled, Research 097 / Plan 136).
- Representation stability across seeds/backbones (paper §6 linchpin, open empirical question, not a PoC we can settle modellessly).
