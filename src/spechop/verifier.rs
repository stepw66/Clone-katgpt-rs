//! Observation verification for speculative hop predictions.
//!
//! Implements the verification layer that compares target tool observations
//! against speculative predictions. Based on paper Appendix D.4:
//! - Exact match (after normalization)
//! - Short-answer exact match
//! - Numeric consistency
//! - Substring match
//! - Token-set Jaccard similarity ≥ 0.55
//! - Refusal pattern detection

use std::collections::HashSet;
use std::sync::LazyLock;

/// Trait for verifying speculative observations against target observations.
///
/// Paper Section 4: when the target tool returns, the verifier checks
/// equivalence → `true` = commit branch, `false` = rollback.
pub trait ObservationVerifier: Send + Sync {
    /// Verify whether speculative observation matches target.
    ///
    /// Returns `true` if observations are equivalent (commit branch),
    /// `false` if they differ (rollback branch).
    fn verify(&self, o_target: &str, o_spec: &str) -> bool;
}

/// Common refusal patterns indicating the tool/model declined to answer.
const REFUSAL_PATTERNS: &[&str] = &[
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "sorry",
    "as an ai",
    "as a language model",
    "i apologize",
    "not able to",
    "against my",
];

/// Common English stopwords to remove before computing Jaccard similarity.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "than", "too", "very", "just", "because", "if",
    "then", "else", "when", "where", "why", "how", "what", "which", "who", "whom", "this", "that",
    "these", "those", "it", "its", "he", "she", "they", "them", "we", "you", "i", "me", "my",
    "your", "his", "her", "our", "their",
];

/// Pre-computed HashSet of stopwords for O(1) lookups instead of O(n) linear scan.
static STOPWORD_SET: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| STOPWORDS.iter().copied().collect());

/// Rule-based verifier implementing paper Appendix D.4 heuristics.
///
/// Checks (in order of decreasing cost, early-exit on pass):
/// 1. Exact match after normalization
/// 2. Short-answer exact match (< 10 chars)
/// 3. Refusal pattern detection (both refused → match, one refused → mismatch)
/// 4. Numeric consistency (extract all numbers, compare sets)
/// 5. Substring containment
/// 6. Token-set Jaccard similarity ≥ 0.55
///
/// Uses thread-local scratch buffers to avoid allocations in the hot path.
/// Each thread reuses its own pre-allocated buffers via `thread_local!`.
#[derive(Clone, Debug)]
pub struct RuleBasedVerifier {
    /// Minimum Jaccard similarity threshold for paraphrase matching.
    /// Paper default: 0.55.
    pub jaccard_threshold: f64,
    /// Maximum length for "short answer" exact matching.
    /// Paper default: 10 characters.
    pub short_answer_max_len: usize,
}

impl Default for RuleBasedVerifier {
    fn default() -> Self {
        Self {
            jaccard_threshold: 0.55,
            short_answer_max_len: 10,
        }
    }
}

// ── Thread-local scratch buffers ──────────────────────────────────
//
// Each thread gets its own set of pre-allocated buffers that are reused
// across `verify()` calls. This avoids per-call heap allocations while
// keeping `RuleBasedVerifier` `Send + Sync`.

std::thread_local! {
    static NORM_BUF: std::cell::RefCell<String> = std::cell::RefCell::new(String::with_capacity(256));
    static NORM_BUF2: std::cell::RefCell<String> = std::cell::RefCell::new(String::with_capacity(256));
    static NUM_ACC: std::cell::RefCell<String> = std::cell::RefCell::new(String::with_capacity(64));
    static NUM_VEC: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::with_capacity(32));
    static NUM_VEC2: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::with_capacity(32));
    static TOKEN_SET_A: std::cell::RefCell<HashSet<&'static str>> = std::cell::RefCell::new(HashSet::with_capacity(64));
    static TOKEN_SET_B: std::cell::RefCell<HashSet<&'static str>> = std::cell::RefCell::new(HashSet::with_capacity(64));
}

impl ObservationVerifier for RuleBasedVerifier {
    fn verify(&self, o_target: &str, o_spec: &str) -> bool {
        // Normalize both inputs into thread-local scratch buffers.
        let (target_norm, spec_norm) = NORM_BUF.with(|nb| {
            NORM_BUF2.with(|nb2| {
                let mut buf1 = nb.borrow_mut();
                let mut buf2 = nb2.borrow_mut();
                normalize_into(o_target, &mut buf1);
                normalize_into(o_spec, &mut buf2);
                // Clone into owned Strings so we can release the RefCell borrows
                // before later steps that also need thread-local scratch.
                (buf1.clone(), buf2.clone())
            })
        });

        // Rule 1: Exact match after normalization
        if target_norm == spec_norm {
            return true;
        }

        // Rule 2: Short-answer exact match
        if target_norm.len() <= self.short_answer_max_len
            && spec_norm.len() <= self.short_answer_max_len
        {
            return target_norm == spec_norm;
        }

        // Rule 3: Refusal pattern detection (text is already lowercased)
        let target_refused = is_refusal(&target_norm);
        let spec_refused = is_refusal(&spec_norm);
        match (target_refused, spec_refused) {
            (true, true) => return true,
            (true, false) | (false, true) => return false,
            (false, false) => {}
        }

        // Rule 4: Numeric consistency
        if numeric_consistent_scratch(&target_norm, &spec_norm) {
            return true;
        }

        // Rule 5: Substring containment
        if target_norm.contains(&spec_norm as &str) || spec_norm.contains(&target_norm as &str) {
            return true;
        }

        // Rule 6: Token-set Jaccard similarity
        let jaccard = token_set_jaccard_scratch(&target_norm, &spec_norm);
        jaccard >= self.jaccard_threshold
    }
}

/// Normalize text for comparison: lowercase, collapse whitespace, trim.
/// Writes into a pre-allocated scratch buffer (cleared first).
fn normalize_into(text: &str, buf: &mut String) {
    buf.clear();
    buf.reserve(text.len());

    let mut prev_space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_space && !buf.is_empty() {
                buf.push(' ');
                prev_space = true;
            }
        } else {
            buf.push(c.to_ascii_lowercase());
            prev_space = false;
        }
    }

    // Trim trailing space
    if buf.ends_with(' ') {
        buf.pop();
    }
}

/// Normalize text for comparison: lowercase, collapse whitespace, trim.
/// Allocates a new String each call — use only in tests or non-hot paths.
#[allow(dead_code)]
fn normalize(text: &str) -> String {
    let mut buf = String::with_capacity(text.len());
    normalize_into(text, &mut buf);
    buf
}

/// Check whether text matches a refusal pattern.
/// Text is expected to be already lowercased (from normalize).
fn is_refusal(text: &str) -> bool {
    REFUSAL_PATTERNS
        .iter()
        .any(|pattern| text.contains(pattern))
}

/// Extract all numeric values (integers and decimals) from text into
/// pre-allocated buffers. Clears both outputs first.
fn extract_numbers_into(text: &str, numbers: &mut Vec<String>, acc: &mut String) {
    numbers.clear();
    acc.clear();
    let mut has_digit = false;

    for ch in text.chars() {
        match ch {
            '0'..='9' => {
                acc.push(ch);
                has_digit = true;
            }
            '.' if has_digit => {
                acc.push(ch);
            }
            '-' if acc.is_empty() => {
                acc.push(ch);
            }
            _ => {
                if has_digit {
                    numbers.push(acc.clone());
                }
                acc.clear();
                has_digit = false;
            }
        }
    }
    if has_digit {
        numbers.push(acc.clone());
    }
}

/// Extract all numeric values (integers and decimals) from text.
/// Allocates — use only in tests.
#[allow(dead_code)]
fn extract_numbers(text: &str) -> Vec<String> {
    let mut numbers = Vec::new();
    let mut acc = String::new();
    extract_numbers_into(text, &mut numbers, &mut acc);
    numbers
}

/// Check numeric consistency using thread-local scratch buffers.
fn numeric_consistent_scratch(target: &str, spec: &str) -> bool {
    NUM_VEC.with(|nv| {
        NUM_VEC2.with(|nv2| {
            NUM_ACC.with(|acc| {
                let mut nums1 = nv.borrow_mut();
                let mut nums2 = nv2.borrow_mut();
                let mut acc_buf = acc.borrow_mut();
                extract_numbers_into(target, &mut nums1, &mut acc_buf);
                extract_numbers_into(spec, &mut nums2, &mut acc_buf);

                if nums1.is_empty() && nums2.is_empty() {
                    return false;
                }
                if nums1.is_empty() || nums2.is_empty() {
                    return false;
                }

                nums1.sort();
                nums2.sort();
                *nums1 == *nums2
            })
        })
    })
}

/// Check numeric consistency: both texts have the same set of numbers.
/// Allocates — use only in tests.
#[allow(dead_code)]
fn numeric_consistent(target: &str, spec: &str) -> bool {
    let mut target_nums = extract_numbers(target);
    let mut spec_nums = extract_numbers(spec);

    if target_nums.is_empty() && spec_nums.is_empty() {
        return false;
    }
    if target_nums.is_empty() || spec_nums.is_empty() {
        return false;
    }

    target_nums.sort();
    spec_nums.sort();
    target_nums == spec_nums
}

/// Tokenize text into a set of words (for Jaccard computation), writing into
/// a pre-allocated scratch HashSet.
fn tokenize_set<'a>(text: &'a str, scratch: &mut HashSet<&'a str>) {
    scratch.clear();
    text.split_whitespace()
        .filter(|word| !STOPWORD_SET.contains(word))
        .for_each(|word| {
            scratch.insert(word);
        });
}

/// Compute token-set Jaccard similarity using thread-local scratch buffers.
///
/// NOTE: the `as *mut HashSet<&str>` cast below looks redundant to clippy
/// (`unnecessary_cast`) but is load-bearing: `HashSet<T>` is invariant in `T`,
/// so without the explicit pointer-type cast the borrow checker refuses to
/// view `HashSet<&'static str>` as `HashSet<&str>`. See the SAFETY comment.
#[allow(clippy::unnecessary_cast)]
fn token_set_jaccard_scratch(target: &str, spec: &str) -> f64 {
    TOKEN_SET_A.with(|sa| {
        TOKEN_SET_B.with(|sb| {
            // SAFETY: The HashSets are typed as `HashSet<&'static str>` for storage
            // convenience. We cast to `HashSet<&str>` to insert the actual borrowed
            // strings. The sets are fully cleared before those strings go out of scope.
            let set_a: &mut HashSet<&str> = unsafe {
                &mut *(&mut *sa.borrow_mut() as *mut HashSet<&'static str> as *mut HashSet<&str>)
            };
            let set_b: &mut HashSet<&str> = unsafe {
                &mut *(&mut *sb.borrow_mut() as *mut HashSet<&'static str> as *mut HashSet<&str>)
            };

            tokenize_set(target, set_a);
            tokenize_set(spec, set_b);

            if set_a.is_empty() && set_b.is_empty() {
                return 1.0;
            }
            if set_a.is_empty() || set_b.is_empty() {
                return 0.0;
            }

            let intersection = set_a.intersection(set_b).count();
            let union = set_a.len() + set_b.len() - intersection;
            if union == 0 {
                return 1.0;
            }

            intersection as f64 / union as f64
        })
    })
}

/// Compute token-set Jaccard similarity between two normalized texts.
/// Allocates — use only in tests or standalone.
///
/// J(A, B) = |A ∩ B| / |A ∪ B|
///
/// Stopwords are removed before computing to focus on content words.
pub fn token_set_jaccard(target: &str, spec: &str) -> f64 {
    let mut target_set = HashSet::new();
    tokenize_set(target, &mut target_set);
    let mut spec_set = HashSet::new();
    tokenize_set(spec, &mut spec_set);

    if target_set.is_empty() && spec_set.is_empty() {
        return 1.0;
    }
    if target_set.is_empty() || spec_set.is_empty() {
        return 0.0;
    }

    let intersection = target_set.intersection(&spec_set).count();
    let union = target_set.len() + spec_set.len() - intersection;
    if union == 0 {
        return 1.0;
    }

    intersection as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── T10: Unit tests ─────────────────────────────────────────

    fn verifier() -> RuleBasedVerifier {
        RuleBasedVerifier::default()
    }

    #[test]
    fn test_identical_observations_true() {
        assert!(verifier().verify("The answer is 42", "The answer is 42"));
        assert!(verifier().verify("hello world", "hello world"));
    }

    #[test]
    fn test_different_numbers_false() {
        assert!(!verifier().verify("The answer is 42", "The answer is 43"));
    }

    #[test]
    fn test_paraphrased_true() {
        let target = "Paris is the capital city of France located in Europe";
        let spec = "Paris is the capital of France in Europe";
        assert!(verifier().verify(target, spec));
    }

    #[test]
    fn test_short_answer_mismatch_false() {
        assert!(!verifier().verify("yes", "no"));
        assert!(!verifier().verify("42", "43"));
        assert!(!verifier().verify("true", "false"));
    }

    #[test]
    fn test_refusal_pattern_false() {
        assert!(!verifier().verify("I cannot answer that question", "The answer is 42"));
        assert!(!verifier().verify("The answer is 42", "I apologize but I cannot help"));
    }

    #[test]
    fn test_both_refusals_true() {
        assert!(verifier().verify("I cannot answer that", "I'm unable to help with that"));
    }

    #[test]
    fn test_case_insensitive_match() {
        assert!(verifier().verify("The Answer Is 42", "the answer is 42"));
        assert!(verifier().verify("HELLO WORLD", "hello world"));
    }

    #[test]
    fn test_whitespace_normalization() {
        assert!(verifier().verify("The   answer   is   42", "The answer is 42"));
        assert!(verifier().verify("  hello  world  ", "hello world"));
    }

    #[test]
    fn test_substring_match() {
        assert!(verifier().verify("The capital of France is Paris", "capital of France"));
    }

    #[test]
    fn test_numeric_consistency_same_numbers() {
        assert!(verifier().verify("Result: 3.14", "Got 3.14"));
        assert!(verifier().verify(
            "Coordinates 40.7128 -74.0060",
            "Location at -74.0060 40.7128"
        ));
    }

    // ── Normalize ───────────────────────────────────────────────

    #[test]
    fn test_normalize_collapse_whitespace() {
        assert_eq!(normalize("  hello   world  "), "hello world");
    }

    #[test]
    fn test_normalize_lowercase() {
        assert_eq!(normalize("HELLO World"), "hello world");
    }

    #[test]
    fn test_normalize_empty() {
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("   "), "");
    }

    #[test]
    fn test_normalize_into_reuse() {
        let mut buf = String::new();
        normalize_into("hello world", &mut buf);
        assert_eq!(buf, "hello world");
        normalize_into("GOODBYE", &mut buf);
        assert_eq!(buf, "goodbye");
    }

    // ── Refusal detection ───────────────────────────────────────

    #[test]
    fn test_is_refusal_positive() {
        assert!(is_refusal("i cannot answer"));
        assert!(is_refusal("i apologize but..."));
        assert!(is_refusal("as an ai, i..."));
    }

    #[test]
    fn test_is_refusal_negative() {
        assert!(!is_refusal("the answer is 42"));
        assert!(!is_refusal("paris is the capital of france"));
    }

    // ── Number extraction ───────────────────────────────────────

    #[test]
    fn test_extract_numbers_simple() {
        assert_eq!(extract_numbers("42"), vec!["42"]);
        assert_eq!(extract_numbers("3.14"), vec!["3.14"]);
    }

    #[test]
    fn test_extract_numbers_multiple() {
        let nums = extract_numbers("x=42 y=3.14 z=-7");
        assert!(nums.contains(&"42".to_string()));
        assert!(nums.contains(&"3.14".to_string()));
    }

    #[test]
    fn test_extract_numbers_none() {
        assert!(extract_numbers("no numbers here").is_empty());
    }

    #[test]
    fn test_extract_numbers_into_reuse() {
        let mut nums = Vec::new();
        let mut acc = String::new();

        extract_numbers_into("42 99", &mut nums, &mut acc);
        assert_eq!(nums, vec!["42", "99"]);

        extract_numbers_into("3.14", &mut nums, &mut acc);
        assert_eq!(nums, vec!["3.14"]);
    }

    // ── Numeric consistency ─────────────────────────────────────

    #[test]
    fn test_numeric_consistent_matching() {
        assert!(numeric_consistent("Result: 3.14", "Got 3.14"));
    }

    #[test]
    fn test_numeric_consistent_no_numbers() {
        assert!(!numeric_consistent("hello", "world"));
    }

    #[test]
    fn test_numeric_consistent_different_numbers() {
        assert!(!numeric_consistent("42", "43"));
    }

    // ── Token-set Jaccard ───────────────────────────────────────

    #[test]
    fn test_jaccard_identical() {
        let j = token_set_jaccard("hello world", "hello world");
        assert!((j - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_no_overlap() {
        let j = token_set_jaccard("alpha beta", "gamma delta");
        assert!((j).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_partial() {
        let j = token_set_jaccard(
            "paris is the capital of france",
            "paris is the capital of europe",
        );
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_both_empty() {
        let j = token_set_jaccard("", "");
        assert!((j - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_stopwords_removed() {
        let j = token_set_jaccard("a the is", "a the is");
        assert!((j - 1.0).abs() < 1e-10);
    }
}
