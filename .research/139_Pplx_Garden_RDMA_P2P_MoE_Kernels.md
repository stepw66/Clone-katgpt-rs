# Research 139: pplx-garden — Perplexity RDMA P2P MoE Dispatch/Combine Kernels

> **Source:** [perplexityai/pplx-garden](https://github.com/perplexityai/pplx-garden) — Perplexity AI, 2025
> **Paper:** [RDMA Point-to-Point Communication for LLM Systems](https://arxiv.org/abs/2510.27656) — Nandor Licker, Kevin Hu, Vladimir Zaytsev, Lequn Chen
> **Date:** 2026-05-30
> **Related Research:** 112 (mKernel), 092 (Five Sharding), 137 (Pplx Tokenizer), 059 (MoE+SD), 066 (TileRT), 091 (SpecHop)
> **Domain:** katgpt-rs (open, GPU infrastructure reference)

---

## TL;DR

pplx-garden is Perplexity AI's open-source Rust+Python+CUDA library for RDMA-based peer-to-peer communication in multi-node LLM systems. It provides: (1) a Rust `fabric-lib` wrapping libfabric/libibverbs for RDMA transfers, (2) P2P all-to-all MoE dispatch/combine kernels optimized for decode (128 tokens) and prefill (4096 tokens), (3) SM-free RDMA transfers during compute, (4) CUDA Graph support.

**Verdict: NO GAIN — Multi-node NVIDIA GPU cluster infrastructure, completely orthogonal to our single-device Apple Silicon CPU SIMD + Metal stack. The Rust `fabric-lib` is well-engineered but requires libfabric/libibverbs/GDRCopy/CUDA 12.8+ and runs only on Linux with GPUDirect RDMA NICs. Three concepts are already covered by prior research: (1) compute-communication overlap → mKernel (R112), (2) all-to-all MoE dispatch → Five Sharding EP axis (R092), (3) persistent kernel SM specialization → TileRT (R066). No feature gate, no plan, no code. Research file for completeness.**

---

## Key Architecture

### fabric-lib (Rust)

```
fabric-lib/src/
├── api.rs              # Public API
├── fabric_engine.rs    # Core engine
├── transfer_engine.rs  # Send/receive operations
├── provider.rs         # NIC provider abstraction
├── efa/                # AWS EFA provider
├── verbs/              # libibverbs (Mellanox/NVIDIA) provider
├── worker.rs           # Async worker
└── topo.rs             # Topology discovery
```

Supports: NVIDIA ConnectX-7, AWS EFA. Aggregates multiple NICs per GPU. Reliable unordered transport.

### p2p-all-to-all (MoE Kernel)

- Split send and recv stages for dispatch and combine
- Micro-batching support
- NVLink for intra-node, RDMA for inter-node
- SM-free during RDMA transfer (GPU compute continues)
- CUDA Graph compatible

### All-to-All Benchmarks (from README)

**Decode (128 tokens):**

| Config | Dispatch | Combine |
|--------|----------|---------|
| EP64 pplx-CX7 | 187.5 μs | 309.1 μs |
| EP32 pplx-CX7 | 153.9 μs | 266.3 μs |
| EP8 pplx-CX7  |  50.5 μs |  65.3 μs |

**Prefill (4096 tokens):**

| Config | Dispatch | Combine |
|--------|----------|---------|
| EP64 pplx-CX7 | 4665.2 μs | 8771.1 μs |
| EP8 pplx-CX7  | 5071.1 μs | 1405.1 μs |

---

## Why No Gain for Us

| Reason | Detail |
|--------|--------|
| **Hardware mismatch** | Requires NVIDIA ConnectX-7 / AWS EFA RDMA NICs + CUDA 12.8+ + GDRCopy. We run Apple Silicon (Metal + CPU SIMD). |
| **Multi-node only** | All value is in inter-node RDMA transfer optimization. Single-device gets zero benefit. |
| **MoE-specific** | All-to-all dispatch is for Mixture-of-Experts serving. We don't have MoE in our stack. |
| **GPU kernel** | The dispatch/combine kernel is CUDA. We can't run CUDA on Apple Silicon. |
| **Linux-only** | Requires Linux Kernel 5.12+ with DMA-BUF, `SYS_PTRACE`/`SYS_ADMIN` capabilities. |

---

## Prior Research Coverage

| pplx-garden Concept | Our Prior Research | Coverage |
|---------------------|-------------------|----------|
| Compute-communication overlap | R112 (mKernel) — persistent SM-specialized kernels | ✅ Conceptual |
| All-to-all MoE dispatch | R092 (Five Sharding) — EP axis | ✅ Conceptual |
| Tile-granularity pipelining | R066 (TileRT) — persistent tile pipeline | ✅ Implemented |
| Speculative dispatch | R091 (SpecHop) — continuous multi-hop | ✅ Implemented |
| Rust networking library | Our stack is single-device, no networking | N/A |
| Pplx tokenizer (Viterbi) | R137 (Pplx Tokenizer) — double-array trie | ✅ Separate research |

---

## Civilization Engine (Plan 168) Relevance

**None.** pplx-garden is GPU cluster networking infrastructure. Civilization Engine is game design composition. No overlap whatsoever.

---

## Verdict

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Research file | ✅ This file | Completeness — pplx-garden is a notable Rust ML project |
| Plan | ❌ No plan | No applicable code for our stack |
| Feature gate | ❌ Not needed | No new functionality |
| Code reference | ⬜ Future reference only | If we ever add multi-device GPU serving |
| Civilization Engine | ❌ Not related | Different domain entirely |

---

## References

- Licker, Hu, Zaytsev, Chen. "RDMA Point-to-Point Communication for LLM Systems." arXiv:2510.27656, 2025.
- Research 112: mKernel (same domain, deeper analysis)
- Research 092: Five Sharding Dimensions (EP axis)
- Research 137: Pplx Tokenizer (same organization, different component)
