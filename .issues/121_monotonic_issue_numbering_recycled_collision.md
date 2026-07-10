# Issue 121: Monotonic Issue Numbering — Replace Recycled `043` Collision

**Created:** 2026-07-10
**Status:** RESOLVED (2026-07-10)
**Severity:** Process / DRY violation (low immediate damage, high long-term grep-noise)
**Discovered via:** Research 398 re-audit — grepping `043` from git history

---

## The problem

Issue number `043` has been **reused 4 separate times** across unrelated issues. Each time a resolved
issue is removed (per the AGENTS.md noise-reduction rule), the next issue grabs the next free low
number, which recycles a number that external artifacts still reference.

Git add-log for `.issues/043*` (all additions across history):

| Commit | What `043` meant |
|---|---|
| `c488ae70` | Plan 367/376 post-commit survey |
| `d11edcf0` | Research 388 Fusion A refuted (Plan 409) |
| `7c9b6863` | Research 398 canvas modelless behavioral PoC |
| `3fa01716` | `ns_inv_sqrt_psd` numerical robustness (Plan 421) |

### Concrete damage

1. **Broken / ambiguous cross-repo links.** `riir-ai/.benchmarks/043_canvas_npc_cognitive_stack_modelless.md`
   links back to `katgpt-rs/.issues/043_canvas_modelless_behavioral_gain_poc.md` — a file that no longer
   exists, and whose number now means 4 different things. Any agent grepping `043` across the 5-repo
   quintet gets cross-contaminated results.
2. **Commit-message ambiguity.** Commit messages reference "Issue 043" for at least 3 distinct issues
   (`f0943ecc` canvas, `3fa01716` ns_inv_sqrt, `d11edcf0` R388/Plan 409). A `git log --grep 043` is
   no longer a precise query.
3. **Benchmark/plan/research cross-references are stale.** This was caught because Research 398's
   §7 addendum and TL;DR referenced `.issues/043` as if it still tracked the canvas PoC; the file is
   gone and the number is now ambiguous. (Fixed inline for R398 in this session; the systemic cause
   remains.)

## Root cause

The noise-reduction rule ("when done fixed issue, note to related doc and remove it") is good for
noise, but it **frees the number** with no guard. The next issue author picks the lowest free number,
recycling it. There is no monotonic high-water-mark counter.

## Proposed fix — monotonic numbering

**Rule:** issue numbers are **never reused**, even after removal. Maintain a strict monotonic
high-water mark.

### Mechanism (pick one)

- **Option A (lightest):** always allocate `max(all-time-high across git history) + 1`. Before creating
  an issue, run:
  ```bash
  git --no-pager log --all --diff-filter=A --name-only --pretty=format: -- '.issues/*' \
    | grep -oE '^\.issues/[0-9]{3,4}_' | sed 's#\.issues/##; s#_##' | sort -n | tail -1
  ```
  and use the next number. Cost: one git command per issue. Reliability: high but relies on author
  discipline.
- **Option B (file-based counter):** keep a `.issues/.NEXT` file (or `.issues/.highwater`) holding the
  integer high-water mark. Creating an issue increments it. Removed issues do NOT decrement it. Cost:
  a tiny state file + a read-modify-write. More robust than A; survives `git log` edge cases (e.g.
  orphaned branches, shallow clones).
- **Option C (date-prefixed, no counter):** use `YYYY_NNN_*.md` where `NNN` is a per-year sequence.
  Collision-free by construction because the year disambiguates. Cost: changes the naming convention
  and all existing tooling/refs; biggest migration.

**Recommendation: Option B** — minimal migration, mechanically sound, no reliance on author memory.

### Naming-convention note (AGENTS.md alignment)

The global rule says "Use `{LATEST_NUMBER e.g. 001}_foo_bar.md` for .plans, .docs, .issues." This issue
proposes that `{LATEST_NUMBER}` be interpreted as a **monotonic, never-recycled** counter, not "the
lowest currently-free number." Same convention, stricter allocation rule.

## Scope

- **katgpt-rs** is the repo where this was observed (4× `043` recycling). Primary target.
- **riir-ai, riir-chain, riir-neuron-db, riir-train** each have their own `.issues/` folders. The same
  recycling risk exists in any repo that removes resolved issues. The rule should apply repo-wide in
  all 5 repos (each repo maintains its own monotonic counter).

## Tasks

- [x] **T1** Mechanism picked: **Option B** (`.issues/.highwater` file), **per-repo counters** (each
  repo owns its counter — issue numbers are scoped per-repo, cross-repo refs use the repo path prefix).
- [x] **T2** Counters seeded at git-history all-time-high per repo (verified via `git log --diff-filter=A`):
  | Repo | All-time-high | Seeded |
  |------|--------------|--------|
  | katgpt-rs | 122 | ✅ |
  | riir-ai | 427 | ✅ |
  | riir-chain | 008 | ✅ (`.issues/` dir recreated) |
  | riir-neuron-db | 009 | ✅ |
  | riir-train | 373 | ✅ |
  Note: katgpt-rs high-water is 122 (not 120 as originally stated — `122_canvas_functor_topology_modelless_poc.md`
  was allocated then resolved-and-removed, exactly the recycling this issue addresses).
- [x] **T3** Rule documented in all 5 repo `AGENTS.md` files (riir-train's `AGENTS.md` created — it was
  missing). The rule covers `.issues/`, `.plans/`, `.docs/`, `.benchmarks/`, `.research/` uniformly.
- [x] **T4** Audited all `.issues/043` references repo-wide. Dead file links fixed in: Bench 419
  (3 locations), feature catalog §12 (3 locations), Plan 419 (header, goal, constraints, out-of-scope),
  and `riir-ai/.benchmarks/043_*` (cross-repo dead link). All now annotated
  "resolved-and-removed 2026-07-09, inconclusive; see Research 398 §7–8". References in Plan 409,
  Plan 421, and Research 388 were already properly annotated "resolved-and-removed" — no change needed.

## Out of scope

- Renumbering existing issues/plans retroactively (too disruptive, breaks all git-history grep).
- Changing the `.plans/` or `.docs/` numbering (same risk applies, but this issue scopes to `.issues/`
  where the recycling was observed; plans/docs can adopt the same counter rule in T3 if desired).

## TL;DR

Issue number `043` was reused 4× across unrelated issues because the noise-reduction rule frees the
number on removal with no monotonic guard. **RESOLVED (2026-07-10):** `.issues/.highwater` files seeded
in all 5 repos at git-history all-time-high (katgpt-rs=122, riir-ai=427, riir-chain=8,
riir-neuron-db=9, riir-train=373); numbering-discipline rule documented in all 5 `AGENTS.md` files
(riir-train's `AGENTS.md` was missing — created); dead `.issues/043` cross-references annotated as
resolved-and-removed. Per the noise-reduction rule, this resolved issue file will be removed; the
`.issues/.highwater` file and the AGENTS.md rules are the permanent fix.
