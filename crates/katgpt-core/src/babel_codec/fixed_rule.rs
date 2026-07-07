//! FixedRuleTextCodec — deterministic BT-P8 fixed symbolic mapping rules
//! (Plan 331 Phase 2, Research 312 §1.4 / paper Appendix C.2.8).
//!
//! The modelless subset of BabelTele: a deterministic, bijective codec between
//! a **canonical verbose form** and a compact BT-P8 symbolic form. Both
//! directions parse into the same [`BabelAst`] and re-emit canonically, so
//! `decompress(compress(x)) ≡ x` bit-identically on the schema-covered subset.
//!
//! # Schema elements (paper Appendix C.2.8)
//!
//! | Verbose canonical form | BT-P8 compressed form | Description |
//! |---|---|---|
//! | `Section[topic/abbrev]` | `S[topic/abbrev]` | section anchor |
//! | `entity has key = value` | `@entity(key=value)` | entity attribute binding |
//! | `Config[target]: key = value(unit)` | `Config[target]:key=value(unit)` | exact-value config (parens around unit in BOTH forms so the parser can split it off) |
//! | `A then B then C` | `A>B>C` | pipeline / containment |
//! | `if cond then act` | `?[cond]=>[act]` | conditional branch |
//! | `except obj : detail` | `!obj:detail` | exception |
//! | `A versus B : conclusion` | `A<>B:conclusion` | comparison |
//! | `BIBREF0`, `TABREF0`, ... | `BIBREF0`, `TABREF0`, ... | preserved verbatim |
//! | `NULL` / `?` | `NULL` / `?` | missing data |
//!
//! # Round-trip contract (G1 fidelity)
//!
//! `decompress(compress(x)) ≡ x` holds **bit-identically** when `x` is in the
//! **canonical verbose form** (the form `decompress` emits). The codec is a
//! bijective mapping between the canonical verbose form and BT-P8 on its
//! supported schema. Inputs in non-canonical verbose form (extra whitespace,
//! different word ordering) are normalized to canonical on `compress`, so
//! `decompress(compress(x))` may differ from `x` by whitespace/word-choice but
//! is semantically identical.
//!
//! For the G1 gate we test only canonical-form inputs — that is the
//! well-defined schema-covered subset on which bit-identity is the contract.
//!
//! # Canonicalization choices (documented)
//!
//! 1. **Whitespace**: one ASCII space between tokens; no leading/trailing
//!    whitespace per line or per record. Newlines separate records (`\n`).
//! 2. **Record ordering**: records emit in input order (stable).
//! 3. **Placeholders** (`BIBREF\d+`, `TABREF\d+`, `FIGREF\d+`): preserved
//!    verbatim — they are opaque references the reader must not rewrite.
//! 4. **Missing data**: `NULL` for explicit null, `?` for unknown.
//! 5. **Identifier charset**: identifiers (`entity`, `key`, `target`) are
//!    `[A-Za-z0-9_]+` plus any unicode alphanumerics (matched via
//!    `char::is_alphanumeric`). Values may additionally contain `.`, `-`, `/`,
//!    digits, units, and placeholders. Values needing other punctuation should
//!    use the BT-P8 form directly (the codec round-trips BT-P8 → BT-P8
//!    bit-identically too).
//! 6. **Unit suffix**: `Config[target]:key=value(unit)` — the unit appears in
//!    parens in BOTH the compressed and canonical verbose form (verbose:
//!    `Config[target]: key = value(unit)`). This keeps the parser simple — the
//!    unit is always the last parenthesized group. Use `(_)` for "no unit".
//!
//! # Modelless / deterministic
//!
//! No float math, no RNG, no training. The codec is a pure string-rewrite
//! function — the same input always produces the same output bytes, on every
//! architecture. BLAKE3 commitments are therefore cross-architecture-stable
//! by construction.

use crate::babel_codec::BabelCodec;
use crate::babel_codec::commitment::BabelCommitment;
use core::fmt::Write as _;

/// BT-P8 fixed-rule text codec. Owns reusable emit scratch buffers so the
/// steady-state `compress`/`decompress` only allocate when the input is larger
/// than the previous call (Vec grow policy).
pub struct FixedRuleTextCodec {
    /// Compression ratio of the most recent `compress` call
    /// (`compressed_bytes / original_bytes`, UTF-8 byte counts). Lower is better.
    last_ratio: f32,
    /// Cached commitment of the most recent `compress` output.
    last_commitment: BabelCommitment,
    /// Reusable emit buffer for `compress` — pre-grown once, `clear()`-and-refill
    /// per call so the steady-state path avoids reallocation.
    compress_buf: Vec<u8>,
    /// Reusable `String` scratch for the emitter (which uses `fmt::Write`).
    /// Pre-grown once; `clear()`-and-refill per call. This is what makes
    /// `compress_into` zero-allocation on the steady-state path (T3.2 extends
    /// to the text codec's in-place variant by design).
    emit_scratch: String,
}

impl Default for FixedRuleTextCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedRuleTextCodec {
    /// Construct an empty codec with default-capacity scratch buffers.
    pub fn new() -> Self {
        Self {
            last_ratio: 1.0,
            last_commitment: BabelCommitment::zero(),
            compress_buf: Vec::with_capacity(256),
            emit_scratch: String::with_capacity(256),
        }
    }

    /// Construct with a pre-sized compress scratch buffer (avoids first-call grow).
    pub fn with_capacity(compress_cap: usize) -> Self {
        Self {
            last_ratio: 1.0,
            last_commitment: BabelCommitment::zero(),
            compress_buf: Vec::with_capacity(compress_cap),
            emit_scratch: String::with_capacity(compress_cap),
        }
    }
}

// ─── AST ────────────────────────────────────────────────────────────────────
//
// The canonical intermediate representation. `compress` parses input → AST;
// `decompress` parses BT-P8 → AST; both then re-emit canonically. Round-trip
// identity holds because both directions agree on canonicalization.

/// One BT-P8 schema record. Variants are 1:1 with the paper's schema elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BabelRecord {
    /// `S[topic/abbrev]` — section anchor.
    Section { anchor: String },
    /// `@entity(key=value)` — entity attribute binding.
    Attribute {
        entity: String,
        key: String,
        value: String,
    },
    /// `Config[target]:key=value(unit)` — exact-value config.
    Config {
        target: String,
        key: String,
        value: String,
        unit: String,
    },
    /// `A>B>C` — pipeline / containment chain.
    Pipeline { stages: Vec<String> },
    /// `?[cond]=>[act]` — conditional branch.
    Conditional { condition: String, action: String },
    /// `!obj:detail` — exception.
    Exception { object: String, detail: String },
    /// `A<>B:conclusion` — comparison.
    Comparison {
        a: String,
        b: String,
        conclusion: String,
    },
    /// A raw opaque line — preserved verbatim (placeholders, NULL, ?, etc.).
    /// Used when a line does not match any schema element.
    Raw { text: String },
}

/// A parsed document: a sequence of records. Order is preserved.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BabelAst {
    pub records: Vec<BabelRecord>,
}

impl BabelAst {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }
}

// ─── Parser ─────────────────────────────────────────────────────────────────
//
// The parser accepts BOTH the canonical verbose form AND the BT-P8 compressed
// form, producing the same AST. This dual acceptance is what makes the codec
// robust: a compressed payload can be re-compressed (idempotent), and a verbose
// payload can be compressed.

/// Parse a multi-line document into a [`BabelAst`]. Each non-empty line is one
/// record. Empty lines are dropped (canonical form has no blank lines).
pub fn parse(input: &str) -> BabelAst {
    let mut ast = BabelAst::new();
    for raw_line in input.split('\n') {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rec) = parse_line(line) {
            ast.records.push(rec);
        } else {
            // Unrecognized line: preserve verbatim as Raw.
            ast.records.push(BabelRecord::Raw {
                text: line.to_string(),
            });
        }
    }
    ast
}

/// Parse a single non-empty line into a [`BabelRecord`], or `None` if no schema
/// element matches (caller wraps as `Raw`).
fn parse_line(line: &str) -> Option<BabelRecord> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }

    // Try each schema element in priority order. Order matters: more-specific
    // patterns first.

    // Section:  `S[topic/abbrev]`  (compressed)  or  `Section[topic/abbrev]`  (verbose)
    if let Some(anchor) = parse_section(line) {
        return Some(BabelRecord::Section { anchor });
    }

    // Config: `Config[target]:key=value(unit)` (both forms identical).
    if let Some(c) = parse_config(line) {
        return Some(BabelRecord::Config {
            target: c.0,
            key: c.1,
            value: c.2,
            unit: c.3,
        });
    }

    // Conditional: `?[cond]=>[act]`  (compressed)  or  `if cond then act`  (verbose)
    if let Some(c) = parse_conditional(line) {
        return Some(BabelRecord::Conditional {
            condition: c.0,
            action: c.1,
        });
    }

    // Exception: `!obj:detail`  (compressed)  or  `except obj : detail`  (verbose)
    if let Some(e) = parse_exception(line) {
        return Some(BabelRecord::Exception {
            object: e.0,
            detail: e.1,
        });
    }

    // Comparison: `A<>B:conclusion`  (compressed)  or  `A versus B : conclusion`  (verbose)
    if let Some(c) = parse_comparison(line) {
        return Some(BabelRecord::Comparison {
            a: c.0,
            b: c.1,
            conclusion: c.2,
        });
    }

    // Attribute: `@entity(key=value)`  (compressed)  or  `entity has key = value`  (verbose)
    if let Some(a) = parse_attribute(line) {
        return Some(BabelRecord::Attribute {
            entity: a.0,
            key: a.1,
            value: a.2,
        });
    }

    // Pipeline: `A>B>C`  (compressed)  or  `A then B then C`  (verbose)
    // (Try last — "A then B then C" shares words with other forms.)
    if let Some(stages) = parse_pipeline(line) {
        return Some(BabelRecord::Pipeline { stages });
    }

    None
}

/// `S[anchor]` or `Section[anchor]`.
fn parse_section(line: &str) -> Option<String> {
    let inner = line
        .strip_prefix("S[")
        .or_else(|| line.strip_prefix("Section["))?;
    let anchor = inner.strip_suffix(']')?;
    if anchor.is_empty() {
        return None;
    }
    // Anchor may contain anything except `]`. Validate non-empty.
    Some(anchor.to_string())
}

/// `Config[target]:key=value(unit)`.
///
/// Accepts both the BT-P8 form (`Config[t]:k=v(u)`) and the canonical verbose
/// form (`Config[t]: k = v(u)` — spaces around `=` and after `:`, but the unit
/// still appears in parens so the parser can split it off unambiguously).
fn parse_config(line: &str) -> Option<(String, String, String, String)> {
    let after_prefix = line.strip_prefix("Config[")?;
    let bracket_close = after_prefix.find(']')?;
    let target = &after_prefix[..bracket_close];
    let rest = &after_prefix[bracket_close + 1..];
    let rest = rest.strip_prefix(':')?;
    // Split on the first `=`.
    let eq = rest.find('=')?;
    let key = rest[..eq].trim();
    let value_and_unit = &rest[eq + 1..];
    // value and unit: `value(unit)`. Unit is the last parenthesized group. We
    // trim the value slice (between `=` and `(`) so that the verbose form's
    // spaces don't leak into the value.
    let open_paren = value_and_unit.rfind('(')?;
    let close_paren = value_and_unit.rfind(')')?;
    if open_paren >= close_paren {
        return None;
    }
    let value = value_and_unit[..open_paren].trim();
    let unit = &value_and_unit[open_paren + 1..close_paren];
    if target.is_empty() || key.is_empty() {
        return None;
    }
    Some((
        target.to_string(),
        key.to_string(),
        value.to_string(),
        unit.to_string(),
    ))
}

/// `?[cond]=>[act]` or `if cond then act`.
fn parse_conditional(line: &str) -> Option<(String, String)> {
    // Compressed: `?[cond]=>[act]`
    if let Some(rest) = line.strip_prefix("?[") {
        let close_bracket = rest.find("]=>")?;
        let cond = &rest[..close_bracket];
        let after = &rest[close_bracket + 3..];
        let act = after.strip_prefix('[')?.strip_suffix(']')?;
        if cond.is_empty() || act.is_empty() {
            return None;
        }
        return Some((cond.to_string(), act.to_string()));
    }
    // Verbose: `if cond then act`
    if let Some(rest) = line.strip_prefix("if ") {
        // Find " then " as a word boundary.
        if let Some(idx) = rest.find(" then ") {
            let cond = rest[..idx].trim();
            let act = rest[idx + 6..].trim();
            if cond.is_empty() || act.is_empty() {
                return None;
            }
            return Some((cond.to_string(), act.to_string()));
        }
    }
    None
}

/// `!obj:detail` or `except obj : detail`.
fn parse_exception(line: &str) -> Option<(String, String)> {
    // Compressed: `!obj:detail`
    if let Some(rest) = line.strip_prefix('!') {
        let colon = rest.find(':')?;
        let obj = &rest[..colon];
        let detail = &rest[colon + 1..];
        if obj.is_empty() || detail.is_empty() {
            return None;
        }
        return Some((obj.to_string(), detail.to_string()));
    }
    // Verbose: `except obj : detail`
    if let Some(rest) = line.strip_prefix("except ") {
        // Find " : " separator.
        if let Some(idx) = rest.find(" : ") {
            let obj = rest[..idx].trim();
            let detail = rest[idx + 3..].trim();
            if obj.is_empty() || detail.is_empty() {
                return None;
            }
            return Some((obj.to_string(), detail.to_string()));
        }
    }
    None
}

/// `A<>B:conclusion` or `A versus B : conclusion`.
fn parse_comparison(line: &str) -> Option<(String, String, String)> {
    // Compressed: `A<>B:conclusion`
    if let Some(idx) = line.find("<>") {
        let a = line[..idx].trim();
        let rest = &line[idx + 2..];
        let colon = rest.find(':')?;
        let b = rest[..colon].trim();
        let conclusion = rest[colon + 1..].trim();
        if a.is_empty() || b.is_empty() || conclusion.is_empty() {
            return None;
        }
        return Some((a.to_string(), b.to_string(), conclusion.to_string()));
    }
    // Verbose: `A versus B : conclusion`
    if let Some(idx) = line.find(" versus ") {
        let a = line[..idx].trim();
        let rest = &line[idx + 8..];
        if let Some(cidx) = rest.find(" : ") {
            let b = rest[..cidx].trim();
            let conclusion = rest[cidx + 3..].trim();
            if a.is_empty() || b.is_empty() || conclusion.is_empty() {
                return None;
            }
            return Some((a.to_string(), b.to_string(), conclusion.to_string()));
        }
    }
    None
}

/// `@entity(key=value)` or `entity has key = value`.
fn parse_attribute(line: &str) -> Option<(String, String, String)> {
    // Compressed: `@entity(key=value)`
    if let Some(rest) = line.strip_prefix('@') {
        let open = rest.find('(')?;
        let close = rest.find(')')?;
        if open >= close {
            return None;
        }
        let entity = &rest[..open];
        let kv = &rest[open + 1..close];
        let eq = kv.find('=')?;
        let key = &kv[..eq];
        let value = &kv[eq + 1..];
        if entity.is_empty() || key.is_empty() {
            return None;
        }
        return Some((entity.to_string(), key.to_string(), value.to_string()));
    }
    // Verbose: `entity has key = value`
    if let Some(idx) = line.find(" has ") {
        let entity = line[..idx].trim();
        let rest = &line[idx + 5..];
        if let Some(eq) = rest.find(" = ") {
            let key = rest[..eq].trim();
            let value = rest[eq + 3..].trim();
            if entity.is_empty() || key.is_empty() {
                return None;
            }
            return Some((entity.to_string(), key.to_string(), value.to_string()));
        }
    }
    None
}

/// `A>B>C` or `A then B then C`. Requires at least 2 stages.
fn parse_pipeline(line: &str) -> Option<Vec<String>> {
    // Compressed: `A>B>C` — split on `>`.
    if line.contains('>') && !line.contains(' ') {
        let stages: Vec<&str> = line.split('>').collect();
        if stages.len() >= 2 && stages.iter().all(|s| !s.is_empty()) {
            return Some(stages.iter().map(|s| s.to_string()).collect());
        }
    }
    // Verbose: `A then B then C` — split on " then ".
    if line.contains(" then ") {
        let stages: Vec<&str> = line.split(" then ").collect();
        if stages.len() >= 2 && stages.iter().all(|s| !s.trim().is_empty()) {
            return Some(stages.iter().map(|s| s.trim().to_string()).collect());
        }
    }
    None
}

// ─── Emitters ───────────────────────────────────────────────────────────────

/// Emit the BT-P8 compressed form of `ast` into `out`.
pub fn emit_compressed(ast: &BabelAst, out: &mut String) {
    for (i, rec) in ast.records.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        emit_record_compressed(rec, out);
    }
}

fn emit_record_compressed(rec: &BabelRecord, out: &mut String) {
    match rec {
        BabelRecord::Section { anchor } => {
            let _ = write!(out, "S[{anchor}]");
        }
        BabelRecord::Attribute { entity, key, value } => {
            let _ = write!(out, "@{entity}({key}={value})");
        }
        BabelRecord::Config {
            target,
            key,
            value,
            unit,
        } => {
            let _ = write!(out, "Config[{target}]:{key}={value}({unit})");
        }
        BabelRecord::Pipeline { stages } => {
            for (i, s) in stages.iter().enumerate() {
                if i > 0 {
                    out.push('>');
                }
                out.push_str(s);
            }
        }
        BabelRecord::Conditional { condition, action } => {
            let _ = write!(out, "?[{condition}]=>[{action}]");
        }
        BabelRecord::Exception { object, detail } => {
            let _ = write!(out, "!{object}:{detail}");
        }
        BabelRecord::Comparison { a, b, conclusion } => {
            let _ = write!(out, "{a}<>{b}:{conclusion}");
        }
        BabelRecord::Raw { text } => {
            out.push_str(text);
        }
    }
}

/// Emit the canonical verbose form of `ast` into `out`.
pub fn emit_verbose(ast: &BabelAst, out: &mut String) {
    for (i, rec) in ast.records.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        emit_record_verbose(rec, out);
    }
}

fn emit_record_verbose(rec: &BabelRecord, out: &mut String) {
    match rec {
        BabelRecord::Section { anchor } => {
            let _ = write!(out, "Section[{anchor}]");
        }
        BabelRecord::Attribute { entity, key, value } => {
            let _ = write!(out, "{entity} has {key} = {value}");
        }
        BabelRecord::Config {
            target,
            key,
            value,
            unit,
        } => {
            let _ = write!(out, "Config[{target}]: {key} = {value}({unit})");
        }
        BabelRecord::Pipeline { stages } => {
            for (i, s) in stages.iter().enumerate() {
                if i > 0 {
                    out.push_str(" then ");
                }
                out.push_str(s);
            }
        }
        BabelRecord::Conditional { condition, action } => {
            let _ = write!(out, "if {condition} then {action}");
        }
        BabelRecord::Exception { object, detail } => {
            let _ = write!(out, "except {object} : {detail}");
        }
        BabelRecord::Comparison { a, b, conclusion } => {
            let _ = write!(out, "{a} versus {b} : {conclusion}");
        }
        BabelRecord::Raw { text } => {
            out.push_str(text);
        }
    }
}

// ─── BabelCodec impl ────────────────────────────────────────────────────────

impl BabelCodec for FixedRuleTextCodec {
    type Input = String;
    type Compressed = Vec<u8>;
    type Reader = ();

    fn compress(&mut self, input: &Self::Input) -> Self::Compressed {
        self.compress_str(input.as_str())
    }

    fn decompress(_reader: &Self::Reader, c: &Self::Compressed) -> Self::Input {
        // The compressed payload is UTF-8 of the BT-P8 string.
        let as_str =
            std::str::from_utf8(c).expect("BabelCodec compressed payload must be valid UTF-8");
        let ast = parse(as_str);
        let mut out = String::new();
        emit_verbose(&ast, &mut out);
        out
    }

    #[inline]
    fn last_ratio(&self) -> f32 {
        self.last_ratio
    }

    #[inline]
    fn commit(&self) -> BabelCommitment {
        self.last_commitment
    }

    fn verify(&self, c: &Self::Compressed, commitment: &BabelCommitment) -> bool {
        let recomputed = BabelCommitment::of(c);
        recomputed.as_bytes() == commitment.as_bytes()
    }
}

impl FixedRuleTextCodec {
    /// Primary ergonomic entry point: compress a `&str` into BT-P8 bytes.
    ///
    /// This is the method callers should use directly. The `BabelCodec::compress`
    /// trait method delegates here (the trait requires `&Self::Input` = `&String`,
    /// which is awkward for string literals; `compress_str` takes `&str` directly).
    ///
    /// Updates `last_ratio` and `last_commitment`.
    pub fn compress_str(&mut self, input: &str) -> Vec<u8> {
        // Reuse the compress scratch buffer: clear, parse, emit.
        self.compress_buf.clear();
        let ast = parse(input);
        self.emit_scratch.clear();
        emit_compressed(&ast, &mut self.emit_scratch);
        self.compress_buf
            .extend_from_slice(self.emit_scratch.as_bytes());

        // Record ratio (UTF-8 byte counts) + commitment.
        let in_bytes = input.len();
        let out_bytes = self.compress_buf.len();
        self.last_ratio = if in_bytes > 0 {
            out_bytes as f32 / in_bytes as f32
        } else {
            1.0
        };
        self.last_commitment = BabelCommitment::of(&self.compress_buf);
        // Clone the bytes out (the contract is `Vec<u8>` ownership).
        self.compress_buf.clone()
    }

    /// In-place variant of [`BabelCodec::compress`] that writes into a
    /// caller-provided buffer instead of returning a fresh `Vec<u8>`. Useful
    /// for callers that want to amortize the output allocation.
    ///
    /// Updates `last_ratio` and `last_commitment` exactly like the trait method.
    pub fn compress_into(&mut self, input: &str, out: &mut Vec<u8>) {
        out.clear();
        let ast = parse(input);
        // Reuse the codec's `emit_scratch` String (the emitter needs `fmt::Write`).
        // This keeps `compress_into` zero-allocation on the steady-state path —
        // the scratch is pre-sized once at construction and `clear()`-and-refilled
        // per call (same pattern as the latent codec's `scratch_scores`).
        self.emit_scratch.clear();
        emit_compressed(&ast, &mut self.emit_scratch);
        out.extend_from_slice(self.emit_scratch.as_bytes());

        let in_bytes = input.len();
        let out_bytes = out.len();
        self.last_ratio = if in_bytes > 0 {
            out_bytes as f32 / in_bytes as f32
        } else {
            1.0
        };
        self.last_commitment = BabelCommitment::of(out);
    }

    /// Decompress straight into a caller-provided `String`.
    pub fn decompress_into(_reader: &(), c: &[u8], out: &mut String) {
        out.clear();
        let as_str =
            std::str::from_utf8(c).expect("BabelCodec compressed payload must be valid UTF-8");
        let ast = parse(as_str);
        emit_verbose(&ast, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a single canonical-verbose line through compress → decompress.
    fn roundtrip_verbose(input: &str) -> String {
        let mut codec = FixedRuleTextCodec::new();
        let compressed = codec.compress_str(input);
        FixedRuleTextCodec::decompress(&(), &compressed)
    }

    // (roundtrip_compressed removed — t08 now uses decompress + compress_str directly
    // for clarity, since roundtrip_compressed returned the re-compressed BT-P8 bytes,
    // not the verbose form.)

    // ── T2.5 required tests (≥ 12) ──────────────────────────────────────────

    #[test]
    fn t01_round_trip_kg_triple_attribute() {
        // Entity-attribute pair (the KG-triple / S-V-O shape).
        let input = "Wang_Nianfang has appellant_of = Hubei_Longan_Real_Estate";
        let rt = roundtrip_verbose(input);
        assert_eq!(
            rt, input,
            "canonical verbose attribute must round-trip bit-identically"
        );
    }

    #[test]
    fn t02_round_trip_entity_attribute_multiple() {
        let input = "player_42 has hp = 100\nplayer_42 has level = 7";
        let rt = roundtrip_verbose(input);
        assert_eq!(rt, input);
    }

    #[test]
    fn t03_round_trip_config_string() {
        let input = "Config[engine]: max_fps = 60(hz)";
        let rt = roundtrip_verbose(input);
        assert_eq!(
            rt, input,
            "config with unit must round-trip bit-identically"
        );
    }

    #[test]
    fn t04_round_trip_conditional_branch() {
        let input = "if hp < 10 then flee";
        let rt = roundtrip_verbose(input);
        assert_eq!(rt, input);
    }

    #[test]
    fn t05_round_trip_comparison() {
        let input = "fire versus ice : fire wins";
        let rt = roundtrip_verbose(input);
        assert_eq!(rt, input);
    }

    #[test]
    fn t06_placeholder_preserved_verbatim() {
        // Placeholders must survive round-trip unchanged.
        let input = "Smith_2020 has citation = BIBREF0";
        let rt = roundtrip_verbose(input);
        assert_eq!(rt, input, "BIBREF placeholder must be preserved verbatim");
        // Multiple placeholders.
        let input2 = "Report has table = TABREF3 and figure = FIGREF1";
        // ^ this won't parse as a single attribute; it'll be Raw. Verify Raw round-trips.
        let rt2 = roundtrip_verbose(input2);
        assert_eq!(
            rt2, input2,
            "Raw line with placeholders must round-trip verbatim"
        );
    }

    #[test]
    fn t07_null_handling() {
        // NULL / ? pass through as Raw lines.
        let input = "NULL\n?\nunknown_field has value = ?";
        let rt = roundtrip_verbose(input);
        assert_eq!(
            rt, input,
            "NULL and ? must round-trip verbatim as Raw lines"
        );
    }

    #[test]
    fn t08_nested_structure_pipeline() {
        // BT-P8 form decompresses to verbose, then verbose re-compresses to BT-P8.
        let bt_p8 = "load>parse>validate>save";
        let verbose = FixedRuleTextCodec::decompress(&(), &bt_p8.as_bytes().to_vec());
        assert_eq!(verbose, "load then parse then validate then save");
        // And the verbose form re-compresses back to the BT-P8 form bit-identically.
        let mut codec = FixedRuleTextCodec::new();
        let recompressed = codec.compress_str(&verbose);
        assert_eq!(
            std::str::from_utf8(&recompressed).unwrap(),
            bt_p8,
            "verbose pipeline must re-compress to the BT-P8 form bit-identically"
        );
    }

    #[test]
    fn t09_empty_input() {
        let mut codec = FixedRuleTextCodec::new();
        let compressed = codec.compress_str("");
        assert!(
            compressed.is_empty(),
            "empty input → empty compressed payload"
        );
        let decompressed = FixedRuleTextCodec::decompress(&(), &compressed);
        assert_eq!(decompressed, "");
        // Ratio for empty input is defined as 1.0 (no-op).
        assert!((codec.last_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn t10_mixed_schema_multiline() {
        let input = "\
Section[quest/kill_10_rats]
player has hp = 100
Config[combat]: base_damage = 15(hp)
if enemy_level > player_level then flee
blacksmith versus merchant : blacksmith wins
load then parse then validate";
        let rt = roundtrip_verbose(input);
        assert_eq!(
            rt, input,
            "mixed-schema multi-line document must round-trip bit-identically"
        );
    }

    #[test]
    fn t11_unicode_entity_names() {
        // Unicode alphanumerics in entity names. The parser accepts any non-
        // whitespace, non-delimiter char inside identifiers (we don't restrict
        // to ASCII), so unicode names round-trip.
        let input = "玩家_42 has 等级 = 7";
        let rt = roundtrip_verbose(input);
        assert_eq!(rt, input, "unicode entity names must round-trip");
    }

    #[test]
    fn t12_section_anchor() {
        let input = "Section[quest/main_story]";
        let rt = roundtrip_verbose(input);
        assert_eq!(rt, input);
        // Compressed form starts with S[.
        let mut codec = FixedRuleTextCodec::new();
        let compressed = codec.compress_str(input);
        let compressed_str = std::str::from_utf8(&compressed).unwrap();
        assert!(
            compressed_str.starts_with("S["),
            "compressed section must start with S[: {compressed_str}"
        );
    }

    #[test]
    fn t13_exception_record() {
        let input = "except boss_immunity : physical_damage_negated";
        let rt = roundtrip_verbose(input);
        assert_eq!(rt, input);
    }

    // ── Additional edge-case tests ──────────────────────────────────────────

    #[test]
    fn t14_compress_then_compress_is_idempotent() {
        // Re-compressing a compressed payload should be a fixed point.
        let mut codec = FixedRuleTextCodec::new();
        let original = "player has hp = 100";
        let c1 = codec.compress_str(original);
        let c2 = codec.compress_str(std::str::from_utf8(&c1).unwrap());
        assert_eq!(
            c1, c2,
            "compress(compress(x)) must equal compress(x) for BT-P8 payloads"
        );
    }

    #[test]
    fn t15_last_ratio_reflects_compression() {
        let mut codec = FixedRuleTextCodec::new();
        // `entity has key = value` (23 bytes) → `@entity(key=value)` (18 bytes).
        // ratio = 18/23 ≈ 0.78 (< 1.0 → compression win).
        codec.compress_str("entity has key = value");
        let r = codec.last_ratio();
        assert!(r < 1.0, "expected ratio < 1.0 for attribute, got {r}");
        assert!(r > 0.0);
    }

    #[test]
    fn t16_commit_and_verify_round_trip() {
        let mut codec = FixedRuleTextCodec::new();
        let payload = "player_42 has hp = 100";
        let compressed = codec.compress_str(payload);
        let commitment = codec.commit();
        assert!(
            codec.verify(&compressed, &commitment),
            "verify must accept the just-committed payload"
        );
        // Tamper.
        let mut tampered = compressed.clone();
        if let Some(byte) = tampered.last_mut() {
            *byte ^= 0xFF;
        }
        assert!(
            !codec.verify(&tampered, &commitment),
            "verify must reject a tampered payload"
        );
    }

    #[test]
    fn t17_compress_into_writes_into_caller_buffer() {
        let mut codec = FixedRuleTextCodec::new();
        let mut buf = Vec::new();
        codec.compress_into("player has hp = 100", &mut buf);
        let s = std::str::from_utf8(&buf).unwrap();
        assert!(
            s.contains("@player("),
            "compress_into must write BT-P8 form into the caller buffer: {s}"
        );
        // Ratio + commitment recorded.
        assert!(codec.last_ratio() < 1.0);
        assert!(!codec.commit().is_zero());
    }

    #[test]
    fn t18_decompress_into_writes_into_caller_buffer() {
        let mut codec = FixedRuleTextCodec::new();
        let compressed = codec.compress_str("player has hp = 100");
        let mut out = String::new();
        FixedRuleTextCodec::decompress_into(&(), &compressed, &mut out);
        assert_eq!(out, "player has hp = 100");
    }

    #[test]
    fn t19_babel_pair_constructs() {
        // Smoke-test the BabelPair wrapper from the parent module.
        use crate::babel_codec::BabelPair;
        let codec1 = FixedRuleTextCodec::new();
        let codec2 = FixedRuleTextCodec::new();
        let pair = BabelPair::new(codec1, codec2);
        // Round-trip via the pair.
        let input = "player has hp = 100";
        let mut compressor = pair.compressor;
        let compressed = compressor.compress_str(input);
        let recovered = FixedRuleTextCodec::decompress(&(), &compressed);
        assert_eq!(recovered, input);
    }

    #[test]
    fn t20_ast_records_parse_correctly() {
        let ast = parse("@entity(k=v)\nSection[foo]\nif a then b");
        assert_eq!(ast.len(), 3);
        match &ast.records[0] {
            BabelRecord::Attribute { entity, key, value } => {
                assert_eq!(entity, "entity");
                assert_eq!(key, "k");
                assert_eq!(value, "v");
            }
            other => panic!("expected Attribute, got {other:?}"),
        }
        assert!(matches!(ast.records[1], BabelRecord::Section { .. }));
        assert!(matches!(ast.records[2], BabelRecord::Conditional { .. }));
    }

    #[test]
    fn t21_raw_line_passthrough() {
        // A line matching no schema element is preserved verbatim as Raw, and
        // Raw round-trips bit-identically in both directions.
        let weird = "some totally unstructured prose with no schema markers";
        let rt = roundtrip_verbose(weird);
        assert_eq!(rt, weird);
    }

    #[test]
    fn t22_config_with_no_unit_uses_underscore() {
        // The documented canonicalization: unit-less configs use `(_)` to round-trip.
        let input = "Config[engine]: version = 1(_)";
        let rt = roundtrip_verbose(input);
        assert_eq!(rt, input);
    }

    // ── Doc-example test (T2.6) ──────────────────────────────────────────────

    /// Doc-example showing before/after on a sample quest dialog.
    #[test]
    fn doc_example_quest_dialog_before_after() {
        // BEFORE (canonical verbose form, 5 lines):
        let before = "\
Section[quest/rescue_hostage]
guard_captain has hp = 250
Config[negotiation]: patience_required = 10(turns)
if guard_hostile then combat
rescue then escort then reward";

        // AFTER compress (BT-P8 form):
        let mut codec = FixedRuleTextCodec::new();
        let compressed = codec.compress_str(before);
        let after = std::str::from_utf8(&compressed).unwrap();
        let expected_after = "\
S[quest/rescue_hostage]
@guard_captain(hp=250)
Config[negotiation]:patience_required=10(turns)
?[guard_hostile]=>[combat]
rescue>escort>reward";
        assert_eq!(after, expected_after);

        // Ratio: the compressed form is meaningfully shorter than the verbose form
        // (compression win — exact ratio asserted loosely, see G2 bench for corpus numbers).
        let r = codec.last_ratio();
        assert!(r < 0.9, "expected ratio < 0.9 for the doc example, got {r}");

        // Round-trip back to the canonical verbose form bit-identically.
        let recovered = FixedRuleTextCodec::decompress(&(), &compressed);
        assert_eq!(recovered, before);
    }
}
