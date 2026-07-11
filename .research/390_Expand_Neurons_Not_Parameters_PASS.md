# Research 390: Expand Neurons, Not Parameters — PASS

> **Source:** Kong et al., "Expand Neurons, Not Parameters", arXiv:2510.04500 (ICML 2026)
> **Date:** 2026-07-07
> **Status:** Done — PASS
> **Classification:** Public

**Verdict:** → PASS, subsumed by 180/203b/204/228.

Paper widens MLP/conv/classifier neurons at constant non-zero param count to reduce polysemanticity. Our substrates have no such unit (HLA 8-dim recurrent kernel, `NeuronShard` Lean-proven fixed Pod, DEC operators). We already ship strictly stronger modelless polysemanticity-routing (kurtosis gate / selectivity router / vocab-channel decompose) that adapts per-position at inference rather than fixing a structural width at training. Scale mismatch (20Hz tick on 8–64-dim latents vs LLM-serving memory-bandwidth argument). No files, no plan, no issue.
