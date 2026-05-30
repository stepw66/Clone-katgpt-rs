# Plan 135: Parallax Parameterized Local Linear Attention

**Research:** [135_Parallax_Parameterized_Local_Linear_Attention](../.research/135_Parallax_Parameterized_Local_Linear_Attention.md)
**Verdict:** ⚠️ Conditional — gated on Muon optimizer adoption
**Feature gate:** `parallax_attn` (opt-in, NOT default-on)

---

# Tasks

- [ ] Implement `parallax_attn` feature flag in `Cargo.toml` (opt-in, gated)
- [ ] Add R projection to `Config` types (only when `parallax_attn` enabled)
- [ ] Implement streaming covariance branch alongside SDPA in `tiled_attention`
- [ ] AHLA covariance experiment: maintain Σ_KV in AHLA state as additional O(d·dv) statistics
- [ ] Benchmark CPU decode overhead: SDPA vs SDPA+R projection (expect ~1.5–2× FLOPs)
- [ ] If `newton_schulz` becomes default, re-run evaluation with Parallax LoRA adapter
