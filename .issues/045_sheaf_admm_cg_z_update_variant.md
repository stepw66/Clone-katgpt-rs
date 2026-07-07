# Issue 045 — Sheaf-ADMM Conjugate-Gradient z-update variant

**Source:** Extracted from Plan 407 Phase 3 T3.1 (post-promotion optimization).
**Primitive:** `sheaf_admm` (katgpt-dec, **DEFAULT-ON** since 2026-07-07, G1–G6 ALL PASS).
**Parent research:** [`.research/384_Sheaf_ADMM_Multi_Agent_Coordination.md`](../.research/384_Sheaf_ADMM_Multi_Agent_Coordination.md)
**Source paper:** arXiv:2605.31005 Appendix B.2.

## Why this is an issue, not a plan task

Per `AGENTS.md`: *optimization/refactor tasks go to `.issues/`, not plans*. Plan 407's primitive is shipped and promoted; this is a post-promotion perf optimization, not core completion.

## The optimization

The shipped `sheaf_admm_step` z-update runs `T` sheaf-diffusion steps via `hodge_laplacian` (gradient descent). For **ill-conditioned large zones** (sparse graphs with poor conditioning), conjugate gradient converges in fewer iterations at fixed residual.

- **Target regime:** K=1000 vertices, condition number > 100 (server-scale zones).
- **Current floor:** GD with eta tuning. Plan 407 G4 measured **1.808 µs** at K=100/d_v=8/d_e=5/T=5 — fast at small scale, but GD convergence degrades on poor-conditioning graphs.
- **Promotion rule:** CG variant is opt-in until a bench proves it wins on **latency at fixed residual** vs the shipped GD path on an ill-conditioned fixture. If it loses, keep GD (the modelless floor) and close this issue as negative-result.

## Acceptance

- [ ] Implement `sheaf_admm_step_cg` (or a `z_update: ZUpdateStrategy` enum on the existing entry point) using CG on the sheaf-Laplacian operator.
- [ ] Bench: GD-vs-CG at K=1000, condition number > 100, fixed residual `‖L_F z‖_∞ < 1e-5`. Record wall-time + iteration count.
- [ ] If CG wins on latency at fixed residual → promote to default (or at least document as the recommended path for ill-conditioned zones). If it loses → document negative result, close issue.

## Notes

- The CG variant must remain modelless (no learned preconditioner; diagonal Jacobi preconditioner is fine — deterministic).
- Coordinate with riir-ai Plan 394 only if the Crowd MCGS runtime actually hits K>1000 ill-conditioned zones in practice. Until then this is a latent optimization.
