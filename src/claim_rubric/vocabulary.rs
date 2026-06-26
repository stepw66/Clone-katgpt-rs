//! Vocabulary enforcement — R287 §2.3.
//!
//! The rubric's teeth come from forcing claim vocabulary to match evidence
//! level. A note that says "the probe *controls* behavior" without L3
//! evidence must be rewritten to "the probe *predicts* behavior" (L1) or
//! "the probe *functionally induces* behavior change" (L2). This module
//! scans claim text for level-restricted verbs.
//!
//! ## Semantics
//!
//! Following R287 §2.3 and Plan 307 T1.9, vocabulary in the text *implies*
//! a level the author is claiming. [`vocabulary_floor`] returns the minimum
//! level the text's verbs demand. The validator computes
//! `honest_level = min(evidence_level, vocabulary_floor)` — which means
//! vocabulary only ever *raises* the bar, never silently lowers it. A
//! violation is recorded separately (see [`scan`]) whenever the claim's
//! declared level is below the floor, so the overclaim is visible in the
//! [`Grade`](crate::claim_rubric::types::Grade) and actionable via
//! [`ClaimValidator::promote_advice`](crate::claim_rubric::validator::ClaimValidator::promote_advice).
//!
//! ## Tables
//!
//! - `L1_SAFE_VERBS` — R287 §2.2 row 1 "Vocabulary allowed" column. Always
//!   safe; never produces a violation.
//! - `L2_VERBS` — R287 §2.2 row 2 "Vocabulary allowed" column. Allowed at
//!   L2 and above; forbidden below L2.
//! - `L3_VERBS` — R287 §2.2 row 3 "Vocabulary allowed" column. Allowed at
//!   L3 only; forbidden below L3.

use crate::claim_rubric::types::{EvidenceLevel, VocabularyViolation};

/// L1-safe verbs (R287 §2.2 row 1 "Vocabulary allowed"). Always safe; never
/// produce a [`VocabularyViolation`]. Listed for documentation/UI; the
/// validator only scans [`L2_VERBS`] and [`L3_VERBS`].
pub const L1_SAFE_VERBS: &[&str] = &[
    "reads",
    "detects",
    "projects to",
    "emits",
    "correlates with",
];

/// L2-only verbs (R287 §2.2 row 2 "Vocabulary allowed"). Permitted at L2
/// and above; forbidden below L2. If found in an L1 claim, they produce a
/// [`VocabularyViolation`] and raise [`vocabulary_floor`] to L2.
pub const L2_VERBS: &[&str] = &["induces", "reliably produces", "functionally steers"];

/// L3-only verbs (R287 §2.2 row 3 "Vocabulary allowed"). Permitted at L3
/// only; forbidden below L3. If found in an L1/L2 claim, they produce a
/// [`VocabularyViolation`] and raise [`vocabulary_floor`] to L3.
pub const L3_VERBS: &[&str] = &[
    "causally controls",
    "mechanistically mediates",
    "is the direction for",
];

/// Returns the minimum evidence level at which `verb` becomes permissible.
///
/// - `Some(L1)` for any verb in [`L1_SAFE_VERBS`] (i.e. never a violation).
/// - `Some(L2)` for any verb in [`L2_VERBS`].
/// - `Some(L3)` for any verb in [`L3_VERBS`].
/// - `None` for unrecognized verbs.
///
/// Case-insensitive.
#[must_use]
pub fn min_level_for_verb(verb: &str) -> Option<EvidenceLevel> {
    let lower = verb.to_lowercase();
    if L1_SAFE_VERBS.iter().any(|v| v.eq_ignore_ascii_case(&lower)) {
        return Some(EvidenceLevel::L1);
    }
    if L2_VERBS.iter().any(|v| v.eq_ignore_ascii_case(&lower)) {
        return Some(EvidenceLevel::L2);
    }
    if L3_VERBS.iter().any(|v| v.eq_ignore_ascii_case(&lower)) {
        return Some(EvidenceLevel::L3);
    }
    None
}

/// Scans `text` for verbs forbidden at `evidence_level` (R287 §2.3) and
/// returns one [`VocabularyViolation`] per offending verb.
///
/// A verb is forbidden at `evidence_level` iff its `min_level_for_verb` is
/// strictly greater than `evidence_level`. For example, `"causally controls"`
/// has min level L3; if `evidence_level` is L1 or L2, the verb produces a
/// violation. This implements R287 §2.3 "vocabulary must match evidence
/// level" — the validator calls this with the claim's honest (evidence-backed)
/// level, NOT its declared level.
///
/// # Matching caveat
///
/// Matching is case-insensitive substring (`text.to_lowercase().contains(...)`),
/// not word-boundary aware. This is acceptable for the multi-word verb
/// phrases in [`L2_VERBS`] and [`L3_VERBS`] (e.g. `"causally controls"`,
/// `"is the direction for"`) which are unlikely to substring-collide with
/// unrelated words. Single-word verbs like `"induces"` could in principle
/// collide (e.g. a token like `"noninduces"` — contrived but possible); a
/// future revision can switch to word-boundary regex when this becomes a
/// real false-positive source.
pub fn scan(text: &str, evidence_level: EvidenceLevel) -> Vec<VocabularyViolation> {
    let lower = text.to_lowercase();
    let mut out: Vec<VocabularyViolation> = Vec::new();
    // Walk high-to-low so the violation order in `out` is L3 verbs first,
    // then L2 verbs — matches R287 §2.2 table order.
    for &verb in L3_VERBS {
        if lower.contains(verb) && EvidenceLevel::L3 > evidence_level {
            out.push(VocabularyViolation {
                verb,
                found_at_level: evidence_level,
                max_allowed_level: EvidenceLevel::L3,
            });
        }
    }
    for &verb in L2_VERBS {
        if lower.contains(verb) && EvidenceLevel::L2 > evidence_level {
            out.push(VocabularyViolation {
                verb,
                found_at_level: evidence_level,
                max_allowed_level: EvidenceLevel::L2,
            });
        }
    }
    out
}

/// The minimum level the verbs in `text` demand (R287 §2.3).
///
/// - If `text` contains any L3 verb ([`L3_VERBS`]), returns `L3`.
/// - Else if `text` contains any L2 verb ([`L2_VERBS`]), returns `L2`.
/// - Else returns `L1` (no vocabulary constraint — L1-safe verbs or no
///   recognized verbs at all).
///
/// The validator computes `honest_level = min(evidence_level,
/// vocabulary_floor(text))`. Note this means vocabulary only ever *raises*
/// the floor — a claim with L3 verbs and L3 evidence is honest at L3; a
/// claim with L3 verbs and L1 evidence is honest at L1, with the L3 verb
/// recorded as a violation by [`scan`].
#[must_use]
pub fn vocabulary_floor(text: &str) -> EvidenceLevel {
    let lower = text.to_lowercase();
    if L3_VERBS.iter().any(|v| lower.contains(*v)) {
        return EvidenceLevel::L3;
    }
    if L2_VERBS.iter().any(|v| lower.contains(*v)) {
        return EvidenceLevel::L2;
    }
    EvidenceLevel::L1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_level_lookup() {
        assert_eq!(min_level_for_verb("reads"), Some(EvidenceLevel::L1));
        assert_eq!(
            min_level_for_verb("Detects"),
            Some(EvidenceLevel::L1),
            "case-insensitive"
        );
        assert_eq!(min_level_for_verb("induces"), Some(EvidenceLevel::L2));
        assert_eq!(
            min_level_for_verb("Functionally Steers"),
            Some(EvidenceLevel::L2)
        );
        assert_eq!(
            min_level_for_verb("causally controls"),
            Some(EvidenceLevel::L3)
        );
        assert_eq!(
            min_level_for_verb("is the direction for"),
            Some(EvidenceLevel::L3)
        );
        assert_eq!(min_level_for_verb("unrelated"), None);
    }

    #[test]
    fn floor_with_no_verbs_is_l1() {
        assert_eq!(vocabulary_floor("hello world"), EvidenceLevel::L1);
        // L1-safe verbs don't raise the floor.
        assert_eq!(
            vocabulary_floor("the probe reads behavior"),
            EvidenceLevel::L1
        );
    }

    #[test]
    fn floor_with_l2_verb_is_l2() {
        assert_eq!(
            vocabulary_floor("the probe induces behavior change"),
            EvidenceLevel::L2
        );
    }

    #[test]
    fn floor_with_l3_verb_is_l3() {
        assert_eq!(
            vocabulary_floor("the probe causally controls behavior"),
            EvidenceLevel::L3
        );
    }

    #[test]
    fn scan_catches_l3_verb_at_l1() {
        let v = scan("the probe causally controls desperation", EvidenceLevel::L1);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].verb, "causally controls");
        assert_eq!(v[0].found_at_level, EvidenceLevel::L1);
        assert_eq!(v[0].max_allowed_level, EvidenceLevel::L3);
    }

    #[test]
    fn scan_clean_for_l1_safe_at_l1() {
        let v = scan("the probe reads behavior", EvidenceLevel::L1);
        assert!(v.is_empty());
    }

    #[test]
    fn scan_l2_verb_passes_at_l2() {
        let v = scan("the probe induces behavior", EvidenceLevel::L2);
        assert!(v.is_empty(), "L2 verb at L2 is permitted");
    }

    #[test]
    fn scan_l3_verb_passes_at_l3() {
        let v = scan(
            "the probe causally controls behavior",
            EvidenceLevel::L3,
        );
        assert!(v.is_empty(), "L3 verb at L3 is permitted");
    }
}
