//! Signal-mix kernel — blends the soft embedding with a control-token embedding
//! at switch instants (paper Eq. 4).
//!
//! When the controller switches Latent→Explicit (or Explicit→Latent), the paper
//! interpolates between the *would-be* emitted soft embedding and the embedding
//! of the control token that anchors the transition (`</think>` on
//! Explicit→Latent, the leading reasoning token on Latent→Explicit). This keeps
//! the residual stream continuous across the mode boundary and prevents a
//! sharp gradient spike from destabilising subsequent layers.
//!
//! The mix ratio is a linear schedule: α_t = α_0 + (1 − α_0) · t / T, so the
//! blend weight *increases* toward the soft embedding as the run progresses —
//! early switches are mostly the anchor token, late switches are mostly the
//! soft embedding.

/// `out ← ratio · out + (1 − ratio) · control_token_embed`, in place.
///
/// - `soft_embed`: the soft-embedding scratch produced by
///   [`soft_embedding`](crate::swir::soft_embedding). Modified in place.
/// - `control_token_embed`: the embedding of the control-token anchor.
/// - `ratio`: blend weight in `[0, 1]`. `ratio = 1` → keep soft embedding; `0`
///   → take the control token verbatim.
///
/// # Panics
///
/// Debug builds assert `ratio ∈ [0, 1]` and equal lengths. Release skips.
#[inline]
pub fn mix_thinking_signal(soft_embed: &mut [f32], control_token_embed: &[f32], ratio: f32) {
    debug_assert!(
        (0.0..=1.0).contains(&ratio),
        "mix ratio out of [0,1]: {ratio}"
    );
    debug_assert_eq!(
        soft_embed.len(),
        control_token_embed.len(),
        "soft_embed and control_token_embed lengths must match"
    );

    // Hot loop — branch free, 8-wide chunk for auto-vec.
    const CHUNK: usize = 8;
    let n = soft_embed.len();
    let inv = 1.0 - ratio;
    let mut i = 0usize;
    while i + CHUNK <= n {
        unsafe {
            let s = soft_embed.get_unchecked_mut(i..i + CHUNK);
            let c = control_token_embed.get_unchecked(i..i + CHUNK);
            for k in 0..CHUNK {
                *s.get_unchecked_mut(k) = ratio * *s.get_unchecked(k) + inv * *c.get_unchecked(k);
            }
        }
        i += CHUNK;
    }
    while i < n {
        soft_embed[i] = ratio * soft_embed[i] + inv * control_token_embed[i];
        i += 1;
    }
}

/// Which kind of signal mix applies at a given switch instant.
///
/// - [`LatentEntry`](Self::LatentEntry): we just entered Latent mode — α_t blends
///   toward the soft embedding, away from the `</think>`-style anchor.
/// - [`ExplicitExit`](Self::ExplicitExit): we just exited Latent mode — β_t blends
///   toward the concrete token, away from the soft embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalMixKind {
    LatentEntry,
    ExplicitExit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_one_keeps_soft_embed() {
        let mut s = vec![1.0, 2.0, 3.0, 4.0];
        let c = vec![10.0, 20.0, 30.0, 40.0];
        mix_thinking_signal(&mut s, &c, 1.0);
        assert_eq!(s, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ratio_zero_takes_control_token() {
        let mut s = vec![1.0, 2.0, 3.0, 4.0];
        let c = vec![10.0, 20.0, 30.0, 40.0];
        mix_thinking_signal(&mut s, &c, 0.0);
        assert_eq!(s, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn ratio_half_blends() {
        let mut s = vec![1.0, 2.0, 3.0, 4.0];
        let c = vec![10.0, 20.0, 30.0, 40.0];
        mix_thinking_signal(&mut s, &c, 0.5);
        assert!((s[0] - 5.5).abs() < 1e-6);
        assert!((s[1] - 11.0).abs() < 1e-6);
        assert!((s[2] - 16.5).abs() < 1e-6);
        assert!((s[3] - 22.0).abs() < 1e-6);
    }

    #[test]
    fn works_with_unaligned_length() {
        // Length 5 — exercises both the chunk path (4-wide) and the tail (1).
        let mut s = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let c = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        mix_thinking_signal(&mut s, &c, 0.5);
        for d in 0..5 {
            assert!((s[d] - c[d] * 0.5).abs() < 1e-6, "dim {d}: {}", s[d]);
        }
    }
}
