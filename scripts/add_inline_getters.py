#!/usr/bin/env python3
"""
Scan crates/**/*.rs (excluding tests/, benches/, examples/) for trivial
&self getter methods missing #[inline], and add #[inline] to them.

A "trivial getter" is a method whose body is a single expression that is
either:
    self.field            (bare field)
    self.field()          (call on a field)
    self.0                (tuple-struct field)

The body may be:
    - Multi-line: a single expression line + closing brace on its own line
        fn name(&self) -> T {
            self.field
        }
    - Single-line: everything on one line
        fn name(&self) -> T { self.field }

Methods that already have #[inline] or #[inline(...)] in the preceding 3
lines are skipped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Signature regex. Captures:
#   sig_indent  -- leading whitespace
#   sig_prefix  -- e.g. "pub fn " or "pub(crate) fn " or "fn "
#   sig_name    -- method name
#   sig_inner   -- what's inside parens (we require exactly &self,
#                  optionally with whitespace)
#   sig_ret     -- the return type (or empty for impl Trait/() defaults)
#   sig_rest    -- anything on the same line after the opening brace
#
# The opening `{` may be followed by an inline body OR nothing (multi-line).
# We anchor the opening brace on the sig line because trivial getters
# virtually always have their `{` there. Methods where `{` is on a later
# line are skipped (rare and ambiguous to parse).
SIG_RE = re.compile(
    r"""
    ^(?P<indent>[ \t]*)
    (?P<prefix>pub(?:\([^)]*\))?\s+)?   # optional pub / pub(...)
    fn\s+
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    \s*\(
        \s*&\s*self\s*                 # exactly &self, no other params
    \)
    (?P<ret>\s*->\s*[^{]+?)?           # optional return type
    \s*\{
    (?P<rest>.*)
    $
    """,
    re.VERBOSE,
)

# Trivial single-expression body. Allows:
#   self.field
#   self.field()
#   self.0, self.1, ...
#   self.field.method_chain()   (still trivial -- the leaf is a field)
#   self.field.clone()          (common clone getter; one call)
TRIVIAL_BODY_RE = re.compile(
    r"""
    ^\s*
    self
    \.[A-Za-z_][A-Za-z0-9_]*     # a single field name
    (?:\([^)]*\))?               # optional single no-arg call
    \s*$
    """,
    re.VERBOSE,
)

# Single-line variant: "fn name(&self) -> T { self.field }"
# Note: this matches the *entire* `{ ... }` segment (including braces)
# that appears after the signature's opening `{`.
SINGLE_LINE_BODY_RE = re.compile(
    r"""
    ^\{\s*
    self\.[A-Za-z_][A-Za-z0-9_]*   # field
    (?:\([^)]*\))?                 # optional no-arg call
    \s*\}\s*$
    """,
    re.VERBOSE,
)


def already_has_inline(lines: list[str], fn_idx: int) -> bool:
    """True if #[inline] or #[inline(...)] appears in the 3 lines above fn."""
    start = max(0, fn_idx - 3)
    for i in range(start, fn_idx):
        s = lines[i].strip()
        if s.startswith("#[inline"):
            return True
    return False


def process_file(path: Path) -> tuple[int, list[str]]:
    """Return (num_added, sample_method_names)."""
    text = path.read_text(encoding="utf-8")
    # Skip files that are clearly not real source (defensive)
    if "\x00" in text:
        return 0, []
    lines = text.split("\n")
    additions: list[tuple[int, str, str]] = []  # (fn_line_idx, indent, name)
    samples: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        m = SIG_RE.match(line)
        if not m:
            i += 1
            continue

        indent = m.group("indent")
        name = m.group("name")
        rest = m.group("rest") or ""

        # Case A: single-line body "{ ... }" on the same line as the sig.
        # rest is the text after the sig's `{`. We match the whole
        # `{ expr }` segment with SINGLE_LINE_BODY_RE.
        if SINGLE_LINE_BODY_RE.match(rest.strip()):
            if not already_has_inline(lines, i):
                additions.append((i, indent, name))
                samples.append(f"{name} (single-line)")
            i += 1
            continue

        # Case B: rest is empty or a comment -> multi-line body
        # rest must be empty (or whitespace) for a clean trivial getter.
        if rest.strip() != "":
            i += 1
            continue

        # Multi-line: next non-blank line is the body expression, the one
        # after is the closing brace (at <= indent).
        body_idx = i + 1
        while body_idx < len(lines) and lines[body_idx].strip() == "":
            body_idx += 1
        if body_idx >= len(lines):
            i += 1
            continue

        body_line = lines[body_idx]

        close_idx = body_idx + 1
        while close_idx < len(lines) and lines[close_idx].strip() == "":
            close_idx += 1
        if close_idx >= len(lines):
            i += 1
            continue

        close_line = lines[close_idx]
        # Body must be a single trivial expression, and the close brace must
        # be at column depth <= the fn indent (typically equal).
        if (
            TRIVIAL_BODY_RE.match(body_line)
            and close_line.strip() == "}"
            and len(close_line) - len(close_line.lstrip(" ")) <= len(indent) + 1
        ):
            if not already_has_inline(lines, i):
                additions.append((i, indent, name))
                samples.append(name)

        i += 1

    if not additions:
        return 0, []

    # Apply additions bottom-up so indices stay valid.
    additions.sort(key=lambda t: t[0], reverse=True)
    for fn_idx, indent, _name in additions:
        lines.insert(fn_idx, f"{indent}#[inline]")

    path.write_text("\n".join(lines), encoding="utf-8")
    return len(additions), samples


def main() -> int:
    crates = ROOT / "crates"
    total = 0
    per_file: list[tuple[Path, int, list[str]]] = []
    for rs in crates.rglob("*.rs"):
        sp = rs.as_posix()
        if "/tests/" in sp or "/benches/" in sp or "/examples/" in sp:
            continue
        if "/target/" in sp:
            continue
        # Skip auto-generated or build files defensively
        n, samples = process_file(rs)
        if n:
            per_file.append((rs, n, samples))
            total += n

    for path, n, samples in sorted(per_file, key=lambda t: str(t[0])):
        rel = path.relative_to(ROOT)
        print(f"{rel}: +{n}")
        for s in samples[:5]:
            print(f"    - {s}")
        if len(samples) > 5:
            print(f"    ... and {len(samples) - 5} more")

    print(f"\nTotal methods annotated: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
