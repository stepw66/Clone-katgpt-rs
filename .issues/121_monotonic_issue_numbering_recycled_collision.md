# Issue 121: Monotonic Issue Numbering — Replace Recycled `043` Collision

**Created:** 2026-07-10
**Status:** Open
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

- [ ] **T1** Pick the mechanism (recommend B: `.issues/.highwater` file). Decide repo-wide (each repo
  owns its counter) or quintet-wide single counter.
- [ ] **T2** Seed the counter at the current all-time-high per repo (katgpt-rs all-time-high is 120 as
  of 2026-07-10). Audit riir-ai / riir-chain / riir-neuron-db / riir-train all-time-highs.
- [ ] **T3** Document the rule in each repo's `AGENTS.md` (or the global personal AGENTS.md) so future
  agents allocate monotonically. One line: "Issue/plan/doc numbers are monotonic and never reused
  after removal; allocate from `.issues/.highwater` + 1."
- [ ] **T4** (Optional, low-priority) Audit existing cross-repo links to recycled numbers (`.issues/043`
  in particular) and either point them at the persistent benchmark/record file or annotate them. R398's
  links were fixed this session; other references to recycled numbers may exist.

## Out of scope

- Renumbering existing issues/plans retroactively (too disruptive, breaks all git-history grep).
- Changing the `.plans/` or `.docs/` numbering (same risk applies, but this issue scopes to `.issues/`
  where the recycling was observed; plans/docs can adopt the same counter rule in T3 if desired).

## TL;DR

Issue number `043` was reused 4× across unrelated issues because the noise-reduction rule frees the
number on removal with no monotonic guard. Fix: allocate issue numbers monotonically from a
never-decremented high-water mark (recommend a `.issues/.highwater` file per repo), document the rule
in AGENTS.md, seed at current all-time-high. katgpt-rs all-time-high is 120 as of today.
