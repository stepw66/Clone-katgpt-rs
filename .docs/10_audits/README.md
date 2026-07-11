# Audits — One-Off Consolidation / Rubric Audits

> **What you find here.** Point-in-time audits that informed a structural
> decision. These are historical records, not living API docs — kept here so the
> reasoning behind a refactor or rubric stays traceable.

## Docs

| Doc | Role |
|---|---|
| [`loser_sweep_audit.md`](loser_sweep_audit.md) | Phase 0.5 — loser-sweep audit (Proposal 003) |
| [`claim_rubric_audit.md`](claim_rubric_audit.md) | Claim-rubric audit — research notes vs `Claim` fixtures (Plan 307 T4.2) |
| [`cross_repo_consolidation_audit.md`](cross_repo_consolidation_audit.md) | Cross-repo consolidation audit — riir-ai / riir-chain / riir-neuron-db (2026-07-06) |

## Note

These are kept under `.docs/` rather than `.research/` because each documents a
**structural decision inside this repo** (what to delete / how claims must be
evidenced / where consolidation ends), not an academic distillation. If an
audit's decision is fully absorbed into code + plans and no longer needs a
standalone record, it can be deleted; until then it lives here.
