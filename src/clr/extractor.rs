//! `FnClaimExtractor` — closure-backed [`ClaimExtractor`] (Plan 284 T1.4).
//!
//! This is the canonical adapter for wiring a domain-specific claim-extraction
//! function (e.g. "split CoT into M reasoning steps and embed each") into the
//! CLR runtime without forcing the caller to implement the trait by hand.

use core::marker::PhantomData;

use crate::clr::traits::ClaimExtractor;
use crate::clr::types::{Claim, Trajectory};

/// Claim extractor backed by a user-supplied closure.
///
/// `F: Fn(&Trajectory<T>) -> Vec<Claim<T>>` is called once per trajectory.
/// The closure MUST return exactly `m` claims — this is enforced via
/// `debug_assert_eq!` so release builds pay no bounds-check cost but debug
/// builds fail loudly on programmer error.
pub struct FnClaimExtractor<F, T> {
    /// Expected claim count (== `ClrConfig::m`). Asserted in [`Self::extract`].
    pub m: usize,
    /// User-supplied extraction closure.
    pub f: F,
    _phantom: PhantomData<T>,
}

impl<F, T> FnClaimExtractor<F, T>
where
    F: Fn(&Trajectory<T>) -> Vec<Claim<T>>,
{
    /// Construct a new extractor expecting exactly `m` claims per trajectory.
    #[inline]
    pub fn new(m: usize, f: F) -> Self {
        assert!(m > 0, "FnClaimExtractor: m must be > 0");
        Self {
            m,
            f,
            _phantom: PhantomData,
        }
    }
}

impl<F, T> ClaimExtractor<T> for FnClaimExtractor<F, T>
where
    F: Fn(&Trajectory<T>) -> Vec<Claim<T>>,
{
    /// Delegate to the stored closure and assert the returned claim count.
    #[inline(always)]
    fn extract(&self, trajectory: &Trajectory<T>) -> Vec<Claim<T>> {
        let result = (self.f)(trajectory);
        debug_assert_eq!(
            result.len(),
            self.m,
            "FnClaimExtractor: expected {} claims, got {}",
            self.m,
            result.len()
        );
        result
    }
}
