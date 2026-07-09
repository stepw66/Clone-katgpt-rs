# katgpt-rs — Documentation

Public MIT engine: a from-scratch Rust GPT-2 style transformer plus a growing
catalog of **modelless inference primitives** — speculative decoding, KV
compression, calibration probes, memory kernels, and heuristic-learning game
arenas. The open sibling of the private `riir-ai` runtime.

> **Read it like a book organized by primitive class.** Each folder below is a
> self-contained context unit — drag the folder into an AI chat and it gets the
> complete narrative for that capability. The folder's `README.md` opens with a
> fusion map showing what combines with what.

## Convention

- **Folders have NO number prefix** (`orientation/`, `inference/`, … `audits/`) —
  they sort alphabetically; the reading order is conveyed by this README's
  doc-group section below, not the folder names. This matches `riir-ai/.docs/`.
- **Files inside have NO number prefix** — add a new doc by dropping `slug.md`
  in the right group folder and adding one line to the relevant group README.
- **Numbers in `.plans/` and `.docs/` live in independent namespaces** — they
  must NEVER share the same number for different topics (the old flat
  `NN_slug.md` scheme collided here, e.g. `191_open_ended_*` doc vs Plan 191).

## Doc groups

### Orientation — "what is this project"

| Folder | What it covers |
|---|---|
| [`orientation/`](orientation/) | Workspace overview + capability list, full core-architecture reference, paper → feature comparison matrix |

### Inference — the speculative decoding + search engine

| Folder | What it covers |
|---|---|
| [`inference/`](inference/) | Speculative decoding (DDTree + DFlash + Leviathan verification), SpecHop continuous multi-hop, KV cache compression alternatives, MTP threshold guide, Progressive MCGS graph search |

### Memory — modelless memory primitives

| Folder | What it covers |
|---|---|
| [`memory/`](memory/) | Raven RSM O(1) routing slot memory, Product Key Memory O(√N) retrieval, Engram hash-addressed conditional memory, MicroRecurrentBeliefState attractor/leaky kernel, NPC Sense Composition, Sleep consolidation at eviction |

### Calibration — probes, gates, and confidence calibration

| Folder | What it covers |
|---|---|
| [`calibration/`](calibration/) | CCE moderator, CausalHeadImportance scale-normalized fusion, FaithfulnessProbe causal-intervention diagnostic, Salience Tri-Gate per-tick emit gate, sigmoid-not-softmax universality-class escape |

### Adaptation — modelless adaptation & distillation

| Folder | What it covers |
|---|---|
| [`adaptation/`](adaptation/) | Model adaptation technique survey (LoRA / merge / spectral-quant KV), Lucebox-hub advanced techniques, PEIRA modelless distillation |

### Game Arenas — heuristic-learning proof-of-concept engines

| Folder | What it covers |
|---|---|
| [`game_arenas/`](game_arenas/) | The HL thesis arenas: Sudoku, Bomberman, Monopoly FSM, FFT Tactics, Go; HL infrastructure + arena detail, open-ended problem-evolution arena, Bomber LoRA A/B artifacts |

### Validator — constraint validation + transformer-VM

| Folder | What it covers |
|---|---|
| [`validator/`](validator/) | Deterministic constraint validator (Sudoku/Rust-AST pruning), Percepta transformer-VM in Rust |

### Performance — perf engineering

| Folder | What it covers |
|---|---|
| [`performance/`](performance/) | Throughput tables, SIMD kernels, benchmarks |

### Feature Catalog — the full opt-in / negative-results ledger

| Folder | What it covers |
|---|---|
| [`feature_catalog/`](feature_catalog/) | Opt-in & gated features (full feature-flag reference), negative results & replaced features |

### Audits — one-off consolidation / rubric audits

| Folder | What it covers |
|---|---|
| [`audits/`](audits/) | Phase 0.5 loser-sweep audit (Proposal 003), claim-rubric audit vs `Claim` fixtures, cross-repo consolidation audit (riir-ai/riir-chain/riir-neuron-db) |

## Sibling repos (private runtimes consume these primitives)

| Repo | Role | Docs location |
|---|---|---|
| [`riir-ai`](../../riir-ai/) | Private SaaS game-AI runtime | `riir-ai/.docs/` (the consolidated selling-point book) |
| [`riir-chain`](../../riir-chain/) | Neuro-symbolic chain lib + daemon | `riir-chain/README.md` (build surface only) |
| [`riir-neuron-db`](../../riir-neuron-db/) | NeuronShard leaf crate | (no `.docs/` today) |
| [`riir-train`](../../riir-train/) | GPU training methods | `.research/` only |

## What does NOT belong in `.docs/`

A `.docs/` file must do at least one of:
1. Document a **shipped primitive** (its API, feature gate, and usage), OR
2. Highlight a **bold outstanding result** (GOAT/gain proven), OR
3. Show **fusion targets** (what this primitive combines WITH to become bigger).

If none → it belongs in `.research/` (academic distillation), `.plans/`
(execution tracking), `.issues/` (open work), or `.benchmarks/` (gate results)
— not `.docs/`. One-off audits live in `audits/` rather than the repo root.

## Historical note (2026-07-09 reindex)

This `.docs/` was reindexed from a flat `NN_slug.md` scheme (39 files at the
repo root, with collisions like `191_open_ended_*` doc vs Plan 191) to the
current unnumbered-folder / unnumbered-file scheme mirroring `riir-ai/.docs/`.
All root `README.md` links, internal `.docs/` cross-references, and
`.benchmarks/` path-bearing links were updated in the same pass. See git
history (`docs:` commit) for the full migration.
