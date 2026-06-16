//! Causal perturbation strategies for injected memory segments.
//!
//! Each function applies one [`Intervention`](super::types::Intervention)
//! variant in-place on a `&mut [T]` slice. Zero allocation — all perturbations
//! mutate the caller-owned buffer.

use fastrand::Rng;

/// `Empty` intervention — zero-fill (replace all elements with `Default`).
///
/// Content is removed but length/format preserved. A faithful consumer
/// should fall back to baseline behavior (small delta vs no-memory baseline)
/// because zeroed memory carries no signal.
#[inline]
pub fn perturb_empty<T: Clone + Default>(memory: &mut [T]) {
    for elem in memory.iter_mut() {
        *elem = T::default();
    }
}

/// `Shuffle` intervention — Fisher-Yates on the slice.
///
/// Destroys temporal/causal structure while preserving the multiset of
/// values. A faithful consumer that depends on ordering (e.g. positional
/// weights, sequence-aware readout) produces a large behavioral delta.
#[inline]
pub fn perturb_shuffle<T>(memory: &mut [T], rng: &mut Rng) {
    // Fisher-Yates: for i from last down to 1, swap[i] <-> swap[random j in 0..=i].
    if memory.len() < 2 {
        return;
    }
    let mut i = memory.len() - 1;
    while i > 0 {
        let j = rng.usize(..=i);
        memory.swap(i, j);
        i -= 1;
    }
}

/// `Corrupt` intervention — random element displacement.
///
/// Each element is, with probability 0.5, replaced by a clone of a randomly
/// chosen *different* element from the same slice. Breaks internal coherence
/// (duplicates, lost ordering) without introducing external content.
#[inline]
pub fn perturb_corrupt<T: Clone>(memory: &mut [T], rng: &mut Rng) {
    let n = memory.len();
    if n < 2 {
        return;
    }
    for i in 0..n {
        if rng.usize(..2) == 0 {
            // Pick an index != i (guaranteed by offsetting then wrapping).
            let j = (i + 1 + rng.usize(..n - 1)) % n;
            let replacement = memory[j].clone();
            memory[i] = replacement;
        }
    }
}

/// `Irrelevant` intervention — replace elements with picks from an external pool.
///
/// Substitutes same-format unrelated content. The pool is caller-provided
/// (e.g. tokens from a different context, latent vectors from another shard).
/// No-op if the pool is empty.
#[inline]
pub fn perturb_irrelevant<T: Clone>(memory: &mut [T], rng: &mut Rng, pool: &[T]) {
    if pool.is_empty() {
        return;
    }
    let pool_len = pool.len();
    for elem in memory.iter_mut() {
        let pick = rng.usize(..pool_len);
        *elem = pool[pick].clone();
    }
}

/// `Filler` intervention — replace all elements with a constant placeholder.
///
/// Semantically-empty content (e.g. padding token, placeholder scalar).
/// Unlike [`perturb_empty`] which uses `Default::default()`, the caller
/// chooses the placeholder here, allowing a non-zero filler (e.g. `<pad>`
/// token id) that tests whether the consumer distinguishes "no content"
/// from "garbage content".
#[inline]
pub fn perturb_filler<T: Clone>(memory: &mut [T], filler: &T) {
    for elem in memory.iter_mut() {
        *elem = filler.clone();
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> Rng {
        Rng::with_seed(0xCAFE)
    }

    #[test]
    fn test_perturb_empty_zeros_all() {
        let mut m = vec![1.0_f32, 2.0, 3.0, 4.0];
        perturb_empty(&mut m);
        assert!(m.iter().all(|&v| v == 0.0));
        // Length preserved.
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn test_perturb_shuffle_preserves_multiset() {
        let mut m = vec![1_u32, 2, 3, 4, 5, 6, 7, 8];
        let original = m.clone();
        perturb_shuffle(&mut m, &mut rng());
        // Multiset unchanged.
        let mut sorted_orig = original.clone();
        sorted_orig.sort_unstable();
        let mut sorted_now = m.clone();
        sorted_now.sort_unstable();
        assert_eq!(sorted_orig, sorted_now);
        // But order changed (extremely unlikely to be identity for 8 elems).
        assert_ne!(original, m);
    }

    #[test]
    fn test_perturb_shuffle_short_slice_noop() {
        let mut m = vec![42_u32];
        let original = m.clone();
        perturb_shuffle(&mut m, &mut rng());
        assert_eq!(m, original);
    }

    #[test]
    fn test_perturb_corrupt_changes_some_elements() {
        let mut m = vec![1_u32, 2, 3, 4, 5, 6, 7, 8];
        let original = m.clone();
        perturb_corrupt(&mut m, &mut rng());
        // At least one element should differ.
        assert!(m.iter().zip(&original).any(|(&a, &b)| a != b));
    }

    #[test]
    fn test_perturb_irrelevant_substitutes_from_pool() {
        let mut m = vec![0_u32; 8];
        let pool = vec![100_u32, 200, 300];
        perturb_irrelevant(&mut m, &mut rng(), &pool);
        // Every element is now from the pool.
        assert!(m.iter().all(|&v| pool.contains(&v)));
        // And none is the original 0.
        assert!(m.iter().all(|&v| v != 0));
    }

    #[test]
    fn test_perturb_irrelevant_empty_pool_noop() {
        let mut m = vec![1_u32, 2, 3];
        let original = m.clone();
        perturb_irrelevant(&mut m, &mut rng(), &[]);
        assert_eq!(m, original);
    }

    #[test]
    fn test_perturb_filler_constant_fill() {
        let mut m = vec![1.0_f32, 2.0, 3.0];
        perturb_filler(&mut m, &7.5);
        assert!(m.iter().all(|&v| v == 7.5));
    }
}
