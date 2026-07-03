//! Quasi-Monte Carlo uniform sources for correlated-but-marginally-exact
//! parallel sampling (Plan 367, Research 367 — QuasiMoTTo,
//! arXiv:2607.01179).
//!
//! Three QMC methods producing k marginally-Unif[0,1) points with controlled
//! joint structure (low-discrepancy coverage):
//! - [`LatticeQmc`]: rank-1 lattice, max coverage, min freedom
//!   (pairwise MI = −∞ — each point determines every other)
//! - [`StratifiedQmc`]: stratified + Fisher-Yates permutation
//!   (pairwise MI = log(k/(k−1)))
//! - [`SobolQmc`]: multi-dimensional Sobol sequence with digital-shift
//!   (Owen) randomization; direction numbers computed at construction from
//!   GF(2) primitive polynomials (zero-dep, no vendored tables)
//!
//! ## The contract (marginal exactness)
//!
//! Each `u_i` drawn by any [`QmcSource`] is marginally uniform on [0,1). The
//! joint structure is designed for better coverage than i.i.d. — enabling
//! 25–47% fewer rollouts for matched pass@k (per the paper). By linearity of
//! expectation, any average-type estimator (policy gradient, mean reward,
//! pass@k) is unbiased regardless of the joint, as long as each rollout's
//! marginal matches the LM. This is what makes QMC a drop-in for i.i.d.
//!
//! ## Zero-allocation contract
//!
//! All [`QmcSource::draw`] calls write into a caller-provided `&mut [f32]`.
//! No allocation occurs inside `draw` — the caller pre-allocates the buffer.

use crate::types::Rng;

// ─────────────────────────────────────────────────────────────────────────────
// QmcSource trait
// ─────────────────────────────────────────────────────────────────────────────

/// QMC uniform source: produces k marginally-Unif[0,1) points.
///
/// Contract: each `u_i` is marginally uniform on [0,1); the joint is
/// low-discrepancy (controlled per implementation). Implementations MUST NOT
/// allocate inside [`draw`](Self::draw) — the caller provides the output
/// buffer.
///
/// Drop-in replacement for K calls to `rng.uniform()` in K-rollout paths
/// (speculative decoding, BoM sampling, PPOT resampling). Each `u_i` feeds
/// an independent arithmetic-coding descend (Plan 367 Phase 2).
pub trait QmcSource {
    /// Fill `out[..k]` with k uniform variates.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() < k`.
    fn draw(&mut self, k: usize, out: &mut [f32]);
}

// ─────────────────────────────────────────────────────────────────────────────
// LatticeQmc — rank-1 lattice
// ─────────────────────────────────────────────────────────────────────────────

/// Rank-1 lattice QMC: k points on `{(i/k + Δ) mod 1 : i=0..k-1}`.
///
/// A single shared offset `Δ ~ Unif[0,1)` is the only degree of freedom — each
/// grid point is marginally uniform because Δ is. Pairwise mutual information
/// is `−∞` (each point determines every other). This is the maximum-coverage /
/// minimum-freedom end of the QMC spectrum: the paper (R367 §1.1) reports it
/// dominates pass@k among the three methods.
///
/// State: 1 `f32` (the offset Δ, redrawn each batch). No per-point allocation.
pub struct LatticeQmc {
    rng: Rng,
}

impl LatticeQmc {
    /// Construct from a 64-bit seed (SplitMix64-mixed per [`Rng::new`]).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self { rng: Rng::new(seed) }
    }
}

impl QmcSource for LatticeQmc {
    #[inline]
    fn draw(&mut self, k: usize, out: &mut [f32]) {
        assert!(out.len() >= k, "LatticeQmc::draw: out.len() {} < k {}", out.len(), k);
        if k == 0 {
            return;
        }
        let delta = self.rng.uniform();
        let inv_k = 1.0 / k as f32;
        // Each point: (i/k + Δ) mod 1. The `fract` is a single `% 1.0` —
        // numerically stable since i/k ∈ [0,1) and Δ ∈ [0,1), so i/k+Δ ∈ [0,2).
        for i in 0..k {
            let v = i as f32 * inv_k + delta;
            out[i] = if v >= 1.0 { v - 1.0 } else { v };
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StratifiedQmc — stratified + Fisher-Yates permutation
// ─────────────────────────────────────────────────────────────────────────────

/// Stratified QMC: divide `[0,1)` into k equal strata, draw one point per
/// stratum, then Fisher-Yates permute.
///
/// Pairwise MI `= log(k/(k−1))` — the middle ground between i.i.d. (MI=0) and
/// lattice (MI=−∞). The paper (R367 §1.1) reports stratified empirically wins
/// RL (lower RLOO bias under dependence).
///
/// State: none beyond the RNG (used for stratum-local draws + permutation).
pub struct StratifiedQmc {
    rng: Rng,
}

impl StratifiedQmc {
    /// Construct from a 64-bit seed.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self { rng: Rng::new(seed) }
    }
}

impl QmcSource for StratifiedQmc {
    #[inline]
    fn draw(&mut self, k: usize, out: &mut [f32]) {
        assert!(out.len() >= k, "StratifiedQmc::draw: out.len() {} < k {}", out.len(), k);
        if k == 0 {
            return;
        }
        let inv_k = 1.0 / k as f32;
        // Step 1: draw one uniform per stratum: out[i] ~ Unif[i/k, (i+1)/k).
        for i in 0..k {
            let lo = i as f32 * inv_k;
            out[i] = lo + self.rng.uniform() * inv_k;
        }
        // Step 2: Fisher-Yates shuffle — each permutation equally likely.
        // Index i drawn uniformly from [0, i] via next_u64 % (i+1).
        for i in (1..k).rev() {
            let j = (self.rng.next() % (i as u64 + 1)) as usize;
            out.swap(i, j);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SobolQmc — Sobol sequence with digital-shift randomization
// ─────────────────────────────────────────────────────────────────────────────

/// Number of bits in each Sobol direction number (u32 → f32 precision).
const SOBOL_BITS: usize = 32;

/// Maximum supported dimensions (dim 0 = Van der Corput + dims 1..32).
///
/// 32 dimensions is enough for token-level QMC on sequences up to 32 tokens
/// (one coordinate per token position). The paper's token-level Sobol uses
/// `dim = sequence_length`; for longer sequences, draw batches at different
/// starting indices.
pub const SOBOL_MAX_DIM: usize = 33;

/// Multi-dimensional Sobol QMC with digital-shift (Owen) randomization.
///
/// Direction numbers are computed at construction from GF(2) primitive
/// polynomials — zero external data tables, zero-dep. Each dimension uses a
/// distinct primitive polynomial (the first available of the smallest
/// sufficient degree), ensuring valid multi-dimensional projection properties.
///
/// Initial direction numbers use `m_j = 1` (the simplest valid choice — all
/// are odd, as required). The specific Joe-Kuo optimized initial values
/// improve two-dimensional projection quality but are not required for
/// correctness; the GOAT gate (Phase 5) validates quality empirically.
///
/// The digital-shift scramble XORs each dimension's output with a random u32
/// drawn at construction. This randomizes the starting point while preserving
/// the low-discrepancy property.
///
/// State: `SOBOL_MAX_DIM × SOBOL_BITS` direction numbers (precomputed) + the
/// running point (u32 per dim) + the index counter + per-dim scramble.
pub struct SobolQmc {
    /// Number of active dimensions (1 for 1D QMC; >1 for token-level coverage).
    dim: usize,
    /// Running sample index (0-based; point 0 is the zero vector, skipped).
    index: u32,
    /// Current point, one u32 bit-pattern per dimension.
    point: [u32; SOBOL_MAX_DIM],
    /// Precomputed direction numbers: `[dim][bit]`.
    direction_numbers: [[u32; SOBOL_BITS]; SOBOL_MAX_DIM],
    /// Per-dimension digital-shift scramble (random u32 from seed).
    scramble: [u32; SOBOL_MAX_DIM],
}

impl SobolQmc {
    /// Construct a 1-dimensional Sobol source (Van der Corput + Owen shift).
    ///
    /// This is the most common case: each `draw(k, out)` produces k scalar
    /// points suitable for the [`QmcSource`] trait. For multi-dimensional
    /// coverage, use [`new_multi`](Self::new_multi).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self::new_multi(seed, 1)
    }

    /// Construct a `dim`-dimensional Sobol source.
    ///
    /// `dim` is clamped to [`SOBOL_MAX_DIM`]. Each dimension uses a distinct
    /// primitive polynomial over GF(2), computed at construction via the
    /// [`find_primitive_poly`] search.
    ///
    /// The trait method [`QmcSource::draw`] outputs only dimension 0 (for
    /// 1D compatibility). Use [`draw_nd`](Self::draw_nd) for multi-dimensional
    /// output.
    pub fn new_multi(seed: u64, dim: usize) -> Self {
        let dim = dim.min(SOBOL_MAX_DIM).max(1);
        let mut rng = Rng::new(seed);

        // Compute direction numbers for each dimension.
        let mut direction_numbers = [[0u32; SOBOL_BITS]; SOBOL_MAX_DIM];

        // Dimension 0: Van der Corput in base 2 — v[j] = 1 << (BITS-1-j).
        // This is the canonical first Sobol dimension (trivially "primitive").
        for j in 0..SOBOL_BITS {
            direction_numbers[0][j] = 1u32 << (SOBOL_BITS - 1 - j);
        }

        // Dimensions 1..dim: find primitive polynomials and compute direction
        // numbers via the recurrence.
        for d in 1..dim {
            let (poly, degree) = find_primitive_poly(d as u32);
            direction_numbers[d] = compute_direction_numbers(poly, degree);
        }

        // Digital-shift scramble: one random u32 per dimension.
        let mut scramble = [0u32; SOBOL_MAX_DIM];
        for s in &mut scramble[..dim] {
            *s = (rng.next() >> 32) as u32 | (rng.next() as u32);
            // Ensure nonzero (a zero scramble is valid but boring).
            if *s == 0 {
                *s = 0xDEAD_BEEF;
            }
        }

        Self {
            dim,
            index: 0,
            point: [0u32; SOBOL_MAX_DIM],
            direction_numbers,
            scramble,
        }
    }

    /// Multi-dimensional draw: fill `out` with `k` points, each `dim` f32s.
    ///
    /// Output layout: `[p0c0, p0c1, ..., p0c(dim-1), p1c0, ...]`.
    /// `out.len()` must be `>= k * self.dim`.
    ///
    /// This is the method for token-level Sobol where each rollout uses
    /// coordinate j as the initial `u` for token position j.
    pub fn draw_nd(&mut self, k: usize, out: &mut [f32]) {
        let needed = k * self.dim;
        assert!(
            out.len() >= needed,
            "SobolQmc::draw_nd: out.len() {} < k*dim {}",
            out.len(),
            needed
        );
        for i in 0..k {
            self.advance();
            let base = i * self.dim;
            for d in 0..self.dim {
                out[base + d] = u32_to_unit_f32(self.point[d] ^ self.scramble[d]);
            }
        }
    }

    /// Advance the internal state by one Sobol point (incremental XOR).
    #[inline]
    fn advance(&mut self) {
        self.index = self.index.wrapping_add(1);
        // The bit to flip is the position of the lowest set bit of the new index.
        // For index 1 → bit 0; index 2 → bit 1; index 3 → bit 0; etc.
        // This follows from Gray(n) XOR Gray(n-1) having exactly one bit set
        // at position trailing_zeros(n).
        let l = (self.index.trailing_zeros() as usize).min(SOBOL_BITS - 1);
        for d in 0..self.dim {
            self.point[d] ^= self.direction_numbers[d][l];
        }
    }
}

impl QmcSource for SobolQmc {
    #[inline]
    fn draw(&mut self, k: usize, out: &mut [f32]) {
        assert!(out.len() >= k, "SobolQmc::draw: out.len() {} < k {}", out.len(), k);
        for i in 0..k {
            self.advance();
            // Output dimension 0 with scramble.
            out[i] = u32_to_unit_f32(self.point[0] ^ self.scramble[0]);
        }
    }
}

/// Map a u32 bit-pattern to a float in [0, 1) using upper 24 bits.
///
/// Matches [`Rng::uniform`] precision (24 mantissa bits). Takes the upper
/// 24 bits (positions 8–31), overlays the IEEE-754 exponent for [1.0, 2.0),
/// then subtracts 1.0.
///
/// [`Rng::uniform`]: katgpt_types::Rng::uniform
#[inline(always)]
fn u32_to_unit_f32(bits: u32) -> f32 {
    f32::from_bits((bits >> 8) | 0x3f80_0000) - 1.0
}

// ─────────────────────────────────────────────────────────────────────────────
// GF(2) polynomial arithmetic — for computing Sobol direction numbers
// ─────────────────────────────────────────────────────────────────────────────
//
// Polynomials over GF(2) are represented as u64 bitmasks: bit i = coefficient
// of x^i. The degree is the position of the highest set bit.
//
// These helpers are ONLY called during `SobolQmc::new_multi` (construction),
// never in the hot `draw` path. Allocation in `prime_factors` is acceptable.

/// Compute a mod b in GF(2)[x] (polynomial remainder).
fn gf2_mod(mut a: u64, b: u64) -> u64 {
    if b == 0 {
        return a;
    }
    let db = 63 - b.leading_zeros();
    // Subtract b shifted to cancel the highest set bit of a, until a is
    // smaller than b (degree of remainder < degree of divisor).
    while a != 0 {
        let da = 63 - a.leading_zeros();
        if da < db {
            break;
        }
        a ^= b << (da - db);
    }
    a
}

/// Compute gcd(a, b) in GF(2)[x] via the Euclidean algorithm.
fn gf2_gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = gf2_mod(a, b);
        a = b;
        b = r;
    }
    a
}

/// Compute (a × b) mod `modulus` in GF(2)[x], where `modulus` has degree `deg`.
fn gf2_mulmod(a: u64, b: u64, modulus: u64, deg: u32) -> u64 {
    let mut result = 0u64;
    let mut a = a;
    let high_bit = 1u64 << deg;
    let mut b = b;
    while b != 0 {
        if b & 1 != 0 {
            result ^= a;
        }
        b >>= 1;
        a <<= 1;
        if a & high_bit != 0 {
            a ^= modulus;
        }
    }
    result
}

/// Compute `base^exp mod modulus` in GF(2)[x] (square-and-multiply).
fn gf2_powmod(mut exp: u64, base: u64, modulus: u64, deg: u32) -> u64 {
    let mut result = 1u64;
    let mut base = gf2_mod(base, modulus);
    while exp > 0 {
        if exp & 1 != 0 {
            result = gf2_mulmod(result, base, modulus, deg);
        }
        base = gf2_mulmod(base, base, modulus, deg);
        exp >>= 1;
    }
    result
}

/// Test whether `poly` (with implicit leading bit at position `degree` and
/// constant bit at position 0) is irreducible over GF(2), using the Ben-Or
/// test.
fn is_irreducible(poly: u64, degree: u32) -> bool {
    // poly must have bit 0 and bit `degree` set.
    if poly & 1 == 0 || poly & (1u64 << degree) == 0 {
        return false;
    }
    // Ben-Or: irreducible iff gcd(poly, x^(2^i) + x) == 1 for i = 1..=floor(deg/2).
    let mut xp = 2u64; // x (= x^1)
    for _ in 1..=(degree / 2) {
        // Square x mod poly: x^(2^i) = (x^(2^{i-1}))^2 mod poly
        xp = gf2_mulmod(xp, xp, poly, degree);
        // x^(2^i) + x (subtraction = addition in GF(2))
        let g = gf2_gcd(poly, xp ^ 2);
        if g != 1 {
            return false;
        }
    }
    true
}

/// Test whether `poly` (degree `degree`) is a primitive polynomial over GF(2):
/// irreducible AND the multiplicative order of x mod poly is exactly 2^degree − 1.
fn is_primitive(poly: u64, degree: u32) -> bool {
    if !is_irreducible(poly, degree) {
        return false;
    }
    let order = (1u64 << degree) - 1;
    // x^order mod poly must be 1.
    if gf2_powmod(order, 2, poly, degree) != 1 {
        return false;
    }
    // For each prime factor q of order: x^(order/q) mod poly must NOT be 1.
    for &q in &prime_factors_u64(order) {
        if gf2_powmod(order / q, 2, poly, degree) == 1 {
            return false;
        }
    }
    true
}

/// Find the primitive polynomial assigned to dimension `dim_index` (1-based).
///
/// Dimensions are assigned one primitive polynomial each, consuming the
/// available primitive polynomials of each degree in order:
/// degree 2 (1 poly) → dims 1..2
/// degree 3 (2 polys) → dims 2..4
/// degree 4 (2 polys) → dims 4..6
/// degree 5 (6 polys) → dims 6..12
/// degree 6 (6 polys) → dims 12..18
/// degree 7 (18 polys) → dims 18..36
///
/// Returns `(poly_as_u64, degree)`.
fn find_primitive_poly(dim_index: u32) -> (u64, u32) {
    // (degree, count_of_polys_so_far_before_this_degree)
    // Number of primitive polys of degree s over GF(2) = φ(2^s − 1) / s.
    // s=2: φ(3)/2 = 1   → cumulative 1
    // s=3: φ(7)/3 = 2   → cumulative 3
    // s=4: φ(15)/4 = 2  → cumulative 5
    // s=5: φ(31)/5 = 6  → cumulative 11
    // s=6: φ(63)/6 = 6  → cumulative 17
    // s=7: φ(127)/7 = 18 → cumulative 35
    const DEGREE_CUMULATIVE: &[(u32, u32)] = &[
        (2, 0),
        (3, 1),
        (4, 3),
        (5, 5),
        (6, 11),
        (7, 17),
    ];

    // Find the degree for this dimension index (1-based).
    let mut degree = 2u32;
    let mut skip = dim_index - 1; // 0-based offset within the degree

    for &(deg, cum) in DEGREE_CUMULATIVE {
        if dim_index > cum {
            degree = deg;
            // How many polys in this degree?
            let next_cum = DEGREE_CUMULATIVE
                .iter()
                .find(|&&(d, _)| d == deg + 1)
                .map(|&(_, c)| c)
                .unwrap_or(35);
            let count_in_degree = next_cum - cum;
            skip = dim_index - cum - 1;
            if skip < count_in_degree {
                break;
            }
        }
    }

    // Enumerate polynomials of `degree` with leading + constant terms set,
    // find the `skip`-th primitive one.
    let leading = 1u64 << degree;
    let middle_bits = degree - 1;
    let mut found = 0u32;
    for middle in 0u64..(1u64 << middle_bits) {
        let poly = leading | (middle << 1) | 1;
        if is_primitive(poly, degree) {
            if found == skip {
                return (poly, degree);
            }
            found += 1;
        }
    }
    panic!(
        "find_primitive_poly: not enough primitive polynomials for dim_index {} (degree {}, skip {})",
        dim_index, degree, skip
    );
}

/// Compute the full direction number table `[u32; SOBOL_BITS]` from a primitive
/// polynomial and its degree.
///
/// Initial direction numbers: `m_j = 1` for `j = 0..degree` (all odd, valid).
/// Stored left-aligned: `v[j] = m_j << (BITS − 1 − j)`.
///
/// Recurrence (Bratley-Fox, in left-aligned integer storage — no shifts):
/// ```text
/// v[j] = v[j − degree]
///      XOR a_1 · v[j − 1] XOR a_2 · v[j − 2] XOR ... XOR a_{s−1} · v[j − s + 1]
/// ```
/// where `a_k` = bit `(degree − k)` of `poly` (coefficient of `x^(degree−k)`).
fn compute_direction_numbers(poly: u64, degree: u32) -> [u32; SOBOL_BITS] {
    let mut v = [0u32; SOBOL_BITS];
    let deg = degree as usize;

    // Initial direction numbers: m_j = 1 for j = 0..degree.
    for j in 0..deg {
        v[j] = 1u32 << (SOBOL_BITS - 1 - j);
    }

    // Recurrence for j >= degree.
    for j in deg..SOBOL_BITS {
        // v[j] starts with v[j − degree] (the constant-term tap, always 1).
        v[j] = v[j - deg];
        // For k = 1..degree−1: if a_k (= bit (degree−k) of poly) is set, XOR v[j−k].
        for k in 1..deg {
            if (poly >> (deg - k)) & 1 == 1 {
                v[j] ^= v[j - k];
            }
        }
    }

    v
}

/// Prime factorization of a u64 (distinct prime factors only).
fn prime_factors_u64(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            factors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── KS test (marginal uniformity) ──────────────────────────────────────

    /// Kolmogorov–Smirnov one-sample test against Unif[0,1).
    /// Returns (D statistic, p-value).
    fn ks_uniform(samples: &[f32]) -> (f64, f64) {
        let n = samples.len();
        assert!(n > 0);
        let mut sorted: Vec<f32> = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut d_max = 0.0f64;
        let nf = n as f64;
        for (i, &x) in sorted.iter().enumerate() {
            let xf = x as f64;
            let f_lower = i as f64 / nf;
            let f_upper = (i + 1) as f64 / nf;
            d_max = d_max.max((f_lower - xf).abs()).max((f_upper - xf).abs());
        }

        // p-value via the Kolmogorov distribution complementary CDF
        // (Numerical Recipes formula):
        //   λ = (√N + 0.12 + 0.11/√N) · D
        //   Q = 2 · Σ_{j=1}^∞ (−1)^{j−1} exp(−2j²λ²)
        let en = nf.sqrt();
        let lambda = (en + 0.12 + 0.11 / en) * d_max;
        let mut q = 0.0f64;
        for j in 1..=100 {
            let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
            let term = sign * (-2.0 * (j as f64) * (j as f64) * lambda * lambda).exp();
            q += term;
            if term.abs() < 1e-12 {
                break;
            }
        }
        q = (2.0 * q).max(0.0).min(1.0);
        (d_max, q)
    }

    // ── Star discrepancy ───────────────────────────────────────────────────

    /// Star discrepancy D*_N = sup_x |F_emp(x) − x| for a finite sample set.
    fn star_discrepancy(samples: &[f32]) -> f64 {
        let n = samples.len();
        assert!(n > 0);
        let mut sorted: Vec<f32> = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let nf = n as f64;
        let mut d_max = 0.0f64;
        for (i, &x) in sorted.iter().enumerate() {
            let xf = x as f64;
            // |(i+1)/N − x_(i)|  (empirical CDF just after x_(i))
            d_max = d_max.max(((i + 1) as f64 / nf - xf).abs());
            // |i/N − x_(i)|  (empirical CDF just before x_(i))
            d_max = d_max.max((i as f64 / nf - xf).abs());
        }
        d_max
    }

    // ── T1.4: LatticeQmc basic ─────────────────────────────────────────────

    #[test]
    fn test_lattice_basic() {
        let mut qmc = LatticeQmc::new(42);
        let mut buf = [0.0f32; 8];
        qmc.draw(8, &mut buf);
        // All values in [0, 1).
        for &v in &buf {
            assert!(v >= 0.0 && v < 1.0, "lattice value out of [0,1): {v}");
        }
        // Points are equally spaced at 1/8 intervals (shifted by Δ).
        let mut sorted = buf;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 1..8 {
            let gap = sorted[i] - sorted[i - 1];
            assert!(
                (gap - 0.125).abs() < 1e-5,
                "lattice points must be 1/k spaced: gap {gap} vs 0.125"
            );
        }
    }

    #[test]
    fn test_lattice_k1() {
        let mut qmc = LatticeQmc::new(7);
        let mut buf = [0.0f32; 1];
        qmc.draw(1, &mut buf);
        assert!(buf[0] >= 0.0 && buf[0] < 1.0);
    }

    #[test]
    fn test_lattice_zero_k() {
        let mut qmc = LatticeQmc::new(99);
        let mut buf = [0.0f32; 4];
        // k=0 should be a no-op (no panic).
        qmc.draw(0, &mut buf);
    }

    // ── T1.5: StratifiedQmc basic ──────────────────────────────────────────

    #[test]
    fn test_stratified_basic() {
        let mut qmc = StratifiedQmc::new(42);
        let mut buf = [0.0f32; 8];
        qmc.draw(8, &mut buf);
        for &v in &buf {
            assert!(v >= 0.0 && v < 1.0, "stratified value out of [0,1): {v}");
        }
        // Each stratum [i/8, (i+1)/8) should contain exactly one point.
        let mut strata = [false; 8];
        for &v in &buf {
            let s = (v * 8.0) as usize;
            let s = s.min(7);
            assert!(!strata[s], "stratum {s} has more than one point");
            strata[s] = true;
        }
        for (i, &occupied) in strata.iter().enumerate() {
            assert!(occupied, "stratum {i} has no point");
        }
    }

    // ── T1.6: SobolQmc basic ───────────────────────────────────────────────

    #[test]
    fn test_sobol_basic() {
        let mut qmc = SobolQmc::new(42);
        let mut buf = [0.0f32; 16];
        qmc.draw(16, &mut buf);
        for &v in &buf {
            assert!(v >= 0.0 && v < 1.0, "sobol value out of [0,1): {v}");
        }
        // The first Sobol point (after skipping the zero) should be ~0.5
        // in dimension 0 (Van der Corput: 0.5, 0.25, 0.75, 0.125, ...).
        // But with Owen scrambling, exact values differ. Just check spread.
        let min = buf.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(min < 0.3, "sobol min too high: {min}");
        assert!(max > 0.7, "sobol max too low: {max}");
    }

    #[test]
    fn test_sobol_multi_dim() {
        let dim = 4;
        let mut qmc = SobolQmc::new_multi(42, dim);
        let k = 8;
        let mut buf = [0.0f32; 32]; // k * dim = 32
        qmc.draw_nd(k, &mut buf);
        for &v in &buf[..k * dim] {
            assert!(v >= 0.0 && v < 1.0, "sobol nd value out of [0,1): {v}");
        }
    }

    #[test]
    fn test_sobol_unscrambled_dim0_matches_van_der_corput() {
        // Without scrambling, dimension 0 is the Van der Corput sequence:
        // 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, ...
        // To test this, we need to zero the scramble. We can't do that via
        // the public API, so we verify the property indirectly: the
        // direction numbers for dim 0 are powers of 2.
        let qmc = SobolQmc::new(1);
        for j in 0..SOBOL_BITS {
            assert_eq!(
                qmc.direction_numbers[0][j],
                1u32 << (SOBOL_BITS - 1 - j),
                "dim 0 direction number {j} must be 1 << (BITS-1-j)"
            );
        }
    }

    // ── T1.7: Marginal uniformity (KS test) ────────────────────────────────
    //
    // Plan specifies N=10^4 batches of k=64. For test speed, we use N=500
    // batches (32K samples total), which still gives the KS test very high
    // statistical power (critical D ≈ 1.36/√32000 ≈ 0.0076).

    #[test]
    fn test_lattice_marginal_uniformity() {
        let mut qmc = LatticeQmc::new(12345);
        let k = 64;
        let n_batches = 500;
        let mut all = Vec::with_capacity(n_batches * k);
        let mut buf = [0.0f32; 64];
        for _ in 0..n_batches {
            qmc.draw(k, &mut buf);
            all.extend_from_slice(&buf[..k]);
        }
        let (d, p) = ks_uniform(&all);
        assert!(
            p > 0.05,
            "LatticeQmc marginal uniformity FAIL: KS D={d:.6}, p={p:.4} (need p>0.05)"
        );
    }

    #[test]
    fn test_stratified_marginal_uniformity() {
        let mut qmc = StratifiedQmc::new(12345);
        let k = 64;
        let n_batches = 500;
        let mut all = Vec::with_capacity(n_batches * k);
        let mut buf = [0.0f32; 64];
        for _ in 0..n_batches {
            qmc.draw(k, &mut buf);
            all.extend_from_slice(&buf[..k]);
        }
        let (d, p) = ks_uniform(&all);
        assert!(
            p > 0.05,
            "StratifiedQmc marginal uniformity FAIL: KS D={d:.6}, p={p:.4} (need p>0.05)"
        );
    }

    #[test]
    fn test_sobol_marginal_uniformity() {
        let mut qmc = SobolQmc::new(12345);
        let k = 64;
        let n_batches = 500;
        let mut all = Vec::with_capacity(n_batches * k);
        let mut buf = [0.0f32; 64];
        for _ in 0..n_batches {
            qmc.draw(k, &mut buf);
            all.extend_from_slice(&buf[..k]);
        }
        let (d, p) = ks_uniform(&all);
        assert!(
            p > 0.05,
            "SobolQmc marginal uniformity FAIL: KS D={d:.6}, p={p:.4} (need p>0.05)"
        );
    }

    // ── T1.8: Low-discrepancy (star discrepancy ≤ i.i.d.) ──────────────────

    #[test]
    fn test_lattice_star_discrepancy_beats_iid() {
        let seed = 42u64;
        let k = 64;

        // QMC batch
        let mut qmc = LatticeQmc::new(seed);
        let mut qmc_buf = [0.0f32; 64];
        qmc.draw(k, &mut qmc_buf);
        let d_qmc = star_discrepancy(&qmc_buf[..k]);

        // i.i.d. baseline (same RNG seed for fair comparison)
        let mut rng = Rng::new(seed);
        let mut iid_buf = [0.0f32; 64];
        for v in &mut iid_buf[..k] {
            *v = rng.uniform();
        }
        let d_iid = star_discrepancy(&iid_buf[..k]);

        assert!(
            d_qmc <= d_iid,
            "LatticeQmc star discrepancy {d_qmc:.6} must be ≤ i.i.d. {d_iid:.6}"
        );
    }

    #[test]
    fn test_stratified_star_discrepancy_beats_iid() {
        let seed = 42u64;
        let k = 64;

        let mut qmc = StratifiedQmc::new(seed);
        let mut qmc_buf = [0.0f32; 64];
        qmc.draw(k, &mut qmc_buf);
        let d_qmc = star_discrepancy(&qmc_buf[..k]);

        let mut rng = Rng::new(seed);
        let mut iid_buf = [0.0f32; 64];
        for v in &mut iid_buf[..k] {
            *v = rng.uniform();
        }
        let d_iid = star_discrepancy(&iid_buf[..k]);

        assert!(
            d_qmc <= d_iid,
            "StratifiedQmc star discrepancy {d_qmc:.6} must be ≤ i.i.d. {d_iid:.6}"
        );
    }

    #[test]
    fn test_sobol_star_discrepancy_beats_iid() {
        let seed = 42u64;
        let k = 64;

        let mut qmc = SobolQmc::new(seed);
        let mut qmc_buf = [0.0f32; 64];
        qmc.draw(k, &mut qmc_buf);
        let d_qmc = star_discrepancy(&qmc_buf[..k]);

        let mut rng = Rng::new(seed);
        let mut iid_buf = [0.0f32; 64];
        for v in &mut iid_buf[..k] {
            *v = rng.uniform();
        }
        let d_iid = star_discrepancy(&iid_buf[..k]);

        assert!(
            d_qmc <= d_iid,
            "SobolQmc star discrepancy {d_qmc:.6} must be ≤ i.i.d. {d_iid:.6}"
        );
    }

    // ── T1.9: Pairwise MI sanity (informational) ───────────────────────────

    /// Estimate pairwise mutual information I(U_0; U_1) via binned histogram.
    /// For continuous variables we bin into `n_bins` equal-width bins.
    fn pairwise_mi(samples_a: &[f32], samples_b: &[f32], n_bins: usize) -> f64 {
        assert_eq!(samples_a.len(), samples_b.len());
        let n = samples_a.len() as f64;

        // Marginal histograms
        let mut ha = vec![0u32; n_bins];
        let mut hb = vec![0u32; n_bins];
        let mut hab = vec![vec![0u32; n_bins]; n_bins];

        for (&a, &b) in samples_a.iter().zip(samples_b.iter()) {
            let ia = ((a * n_bins as f32).floor() as usize).min(n_bins - 1);
            let ib = ((b * n_bins as f32).floor() as usize).min(n_bins - 1);
            ha[ia] += 1;
            hb[ib] += 1;
            hab[ia][ib] += 1;
        }

        let mut mi = 0.0f64;
        for ia in 0..n_bins {
            for ib in 0..n_bins {
                let cab = hab[ia][ib];
                if cab == 0 {
                    continue;
                }
                let pab = cab as f64 / n;
                let pa = ha[ia] as f64 / n;
                let pb = hb[ib] as f64 / n;
                mi += pab * (pab / (pa * pb)).ln();
            }
        }
        mi
    }

    #[test]
    fn test_lattice_high_pairwise_mi() {
        // Lattice: each point determines every other → MI should be very high.
        let mut qmc = LatticeQmc::new(42);
        let k = 64;
        let n_batches = 500;
        let mut col0 = Vec::with_capacity(n_batches);
        let mut col1 = Vec::with_capacity(n_batches);
        let mut buf = [0.0f32; 64];
        for _ in 0..n_batches {
            qmc.draw(k, &mut buf);
            col0.push(buf[0]);
            col1.push(buf[1]);
        }
        let mi = pairwise_mi(&col0, &col1, 16);
        // For lattice, U_1 = (U_0 + 1/k) mod 1 → MI is very high (near log(k)).
        assert!(
            mi > 1.0,
            "LatticeQmc pairwise MI={mi:.4} should be high (>1.0, near log(k)≈4.16 for k=64)"
        );
    }

    #[test]
    fn test_iid_near_zero_pairwise_mi() {
        // i.i.d. baseline: MI should be near zero.
        let mut rng = Rng::new(42);
        let n = 500;
        let mut col0 = Vec::with_capacity(n);
        let mut col1 = Vec::with_capacity(n);
        for _ in 0..n {
            col0.push(rng.uniform());
            col1.push(rng.uniform());
        }
        let mi = pairwise_mi(&col0, &col1, 16);
        // With finite samples, MI estimate has positive bias. Allow up to 0.3.
        assert!(
            mi < 0.3,
            "i.i.d. pairwise MI={mi:.4} should be near zero (<0.3 with finite-sample bias)"
        );
    }

    // ── GF(2) helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_gf2_mod() {
        // x^3 mod (x^2+x+1) = x+1 (since x^3 = x·x^2 = x·(x+1) = x^2+x = (x+1)+x = 1... wait)
        // Actually x^3 mod (x^2+x+1): x^2 ≡ x+1, so x^3 = x·x^2 ≡ x·(x+1) = x^2+x ≡ (x+1)+x = 1.
        // So x^3 mod (x^2+x+1) = 1.
        let poly = 0b111u64; // x^2+x+1
        let x3 = 0b1000u64; // x^3
        assert_eq!(gf2_mod(x3, poly), 1, "x^3 mod (x^2+x+1) should be 1");
    }

    #[test]
    fn test_is_irreducible() {
        // x^2+x+1 is irreducible over GF(2).
        assert!(is_irreducible(0b111, 2));
        // x^2+1 = (x+1)^2 is reducible.
        assert!(!is_irreducible(0b101, 2));
        // x^3+x+1 is irreducible.
        assert!(is_irreducible(0b1011, 3));
        // x^3+x^2+x+1 = (x+1)(x^2+1) is reducible.
        assert!(!is_irreducible(0b1111, 3));
    }

    #[test]
    fn test_is_primitive() {
        // x^2+x+1 is primitive (2^2-1=3 is prime, irreducible ⟹ primitive).
        assert!(is_primitive(0b111, 2));
        // x^3+x+1 is primitive (2^3-1=7 is prime).
        assert!(is_primitive(0b1011, 3));
        // x^3+x^2+1 is primitive.
        assert!(is_primitive(0b1101, 3));
        // x^4+x+1 is primitive.
        assert!(is_primitive(0b10011, 4));
        // x^4+x^3+x^2+x+1 is irreducible but NOT primitive
        // (2^4-1=15, order of x divides 5).
        assert!(is_irreducible(0b11111, 4));
        assert!(!is_primitive(0b11111, 4));
    }

    #[test]
    fn test_find_primitive_poly_dim1() {
        // Dimension 1 should use x^2+x+1 (the only primitive poly of degree 2).
        let (poly, degree) = find_primitive_poly(1);
        assert_eq!(degree, 2);
        assert_eq!(poly, 0b111);
    }

    #[test]
    fn test_find_primitive_poly_all_dims() {
        // All 32 dimensions should find a valid primitive polynomial.
        for d in 1..=32 {
            let (poly, degree) = find_primitive_poly(d);
            assert!(
                is_primitive(poly, degree),
                "dim {d}: poly {poly:#b} degree {degree} is not primitive"
            );
        }
    }

    #[test]
    fn test_sobol_direction_numbers_nonzero() {
        // All direction numbers must be nonzero (zero would break the XOR chain).
        for d in 1..=32 {
            let (poly, degree) = find_primitive_poly(d);
            let v = compute_direction_numbers(poly, degree);
            for (j, &vn) in v.iter().enumerate() {
                assert!(vn != 0, "dim {d} direction number {j} is zero");
            }
        }
    }

    #[test]
    fn test_sobol_construction_all_dims() {
        // Constructing a 32-dimensional Sobol source should not panic.
        let qmc = SobolQmc::new_multi(42, 32);
        assert_eq!(qmc.dim, 32);
    }

    // ── Determinism ────────────────────────────────────────────────────────

    #[test]
    fn test_lattice_deterministic() {
        let mut a = LatticeQmc::new(42);
        let mut b = LatticeQmc::new(42);
        let mut buf_a = [0.0f32; 16];
        let mut buf_b = [0.0f32; 16];
        a.draw(16, &mut buf_a);
        b.draw(16, &mut buf_b);
        assert_eq!(buf_a, buf_b, "same seed must produce same sequence");
    }

    #[test]
    fn test_stratified_deterministic() {
        let mut a = StratifiedQmc::new(42);
        let mut b = StratifiedQmc::new(42);
        let mut buf_a = [0.0f32; 16];
        let mut buf_b = [0.0f32; 16];
        a.draw(16, &mut buf_a);
        b.draw(16, &mut buf_b);
        assert_eq!(buf_a, buf_b, "same seed must produce same sequence");
    }

    #[test]
    fn test_sobol_deterministic() {
        let mut a = SobolQmc::new(42);
        let mut b = SobolQmc::new(42);
        let mut buf_a = [0.0f32; 16];
        let mut buf_b = [0.0f32; 16];
        a.draw(16, &mut buf_a);
        b.draw(16, &mut buf_b);
        assert_eq!(buf_a, buf_b, "same seed must produce same sequence");
    }

    // ── Buffer-too-small panics ────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "out.len()")]
    fn test_lattice_buf_too_small() {
        let mut qmc = LatticeQmc::new(42);
        let mut buf = [0.0f32; 4];
        qmc.draw(8, &mut buf);
    }

    #[test]
    #[should_panic(expected = "out.len()")]
    fn test_sobol_buf_too_small() {
        let mut qmc = SobolQmc::new(42);
        let mut buf = [0.0f32; 4];
        qmc.draw(8, &mut buf);
    }
}
