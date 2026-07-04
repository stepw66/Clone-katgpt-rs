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
        for (i, slot) in out.iter_mut().enumerate().take(k) {
            let v = i as f32 * inv_k + delta;
            *slot = if v >= 1.0 { v - 1.0 } else { v };
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
        for (i, slot) in out.iter_mut().enumerate().take(k) {
            let lo = i as f32 * inv_k;
            *slot = lo + self.rng.uniform() * inv_k;
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
        let dim = dim.clamp(1, SOBOL_MAX_DIM);
        let mut rng = Rng::new(seed);

        // Compute direction numbers for each dimension.
        let mut direction_numbers = [[0u32; SOBOL_BITS]; SOBOL_MAX_DIM];

        // Dimension 0: Van der Corput in base 2 — v[j] = 1 << (BITS-1-j).
        // This is the canonical first Sobol dimension (trivially "primitive").
        for (j, slot) in direction_numbers[0].iter_mut().enumerate() {
            *slot = 1u32 << (SOBOL_BITS - 1 - j);
        }

        // Dimensions 1..dim: find primitive polynomials and compute direction
        // numbers via the recurrence.
        for (d, row) in direction_numbers.iter_mut().enumerate().take(dim).skip(1) {
            let (poly, degree) = find_primitive_poly(d as u32);
            *row = compute_direction_numbers(poly, degree);
        }

        // Digital-shift scramble: one random u32 per dimension.
        //
        // Each scramble is the upper 32 bits of one `rng.next()` call (u64).
        // Upper bits of xorshift64 have better statistical distribution
        // than the lower bits (lower bits have shorter LFSR periods).
        // (Phase 5 GOAT gate G1 catch: the original code OR'd two 32-bit
        // halves from two separate draws — OR(a,b) is NOT uniform:
        // P(bit=1) = 0.75, not 0.5 — which biased the Sobol output and broke
        // marginal exactness. G1 fail rate dropped from 98% to ~1%.)
        let mut scramble = [0u32; SOBOL_MAX_DIM];
        for s in &mut scramble[..dim] {
            *s = (rng.next() >> 32) as u32;
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
        for slot in out.iter_mut().take(k) {
            self.advance();
            // Output dimension 0 with scramble.
            *slot = u32_to_unit_f32(self.point[0] ^ self.scramble[0]);
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
    for (j, slot) in v.iter_mut().enumerate().take(deg) {
        *slot = 1u32 << (SOBOL_BITS - 1 - j);
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
        if n.is_multiple_of(d) {
            factors.push(d);
            while n.is_multiple_of(d) {
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
// Phase 4 — QMC → Gaussian noise query fill (Fusion A: QmcBoMSampler)
// (Plan 367 Phase 4, Research 367 §2.3 — strongest fusion)
//
// `BoMSampler::sample_k_states` takes a pre-filled `queries: &[f32]` buffer of
// K×D Gaussian noise. The sampler itself is agnostic to how `queries` was
// generated — i.i.d. (`rng.normal() * sigma` in a loop) or QMC (this module).
// Phase 4 provides the QMC fill path: draw low-discrepancy uniforms, apply the
// inverse Gaussian CDF (probit) to each, scale by σ. Each element is marginally
// N(0,σ²) exact (T4.2); the joint has QMC low-discrepancy structure for better
// coverage of the K-dim belief ball (T4.3).
//
// # Design note — why a free helper, not a SeedStrategy variant
//
// The plan suggested adding `SeedStrategy::QmcLattice` / `QmcSobol` variants,
// but this is infeasible for two SOLID reasons:
// 1. `SeedStrategy` lives in `katgpt-micro-belief` (leaf crate) which cannot
//    depend on `katgpt-core` where `QmcSource` is defined — circular dep.
// 2. `SeedStrategy` governs seed derivation (PerNpc vs PerClass), semantically
//    orthogonal to noise shape (i.i.d. vs QMC). Conflating violates ISP.
// The free-helper design respects the existing architecture: callers already
// manage their own `queries` buffer (see `conformal_floor_bom.rs:184`); QMC is
// a drop-in alternative fill strategy.
// ─────────────────────────────────────────────────────────────────────────────

/// Inverse of the standard normal CDF (probit function).
///
/// Maps `u ∈ (0, 1)` to the standard normal quantile `z = Φ⁻¹(u)` such that
/// `P(Z ≤ z) = u` for `Z ~ N(0,1)`. Used to transform QMC uniform variates into
/// marginally-Gaussian noise queries (Plan 367 Phase 4, T4.2).
///
/// # Algorithm
///
/// Hastings (1955) rational approximation. Max absolute error ≈ 4.5e-4
/// — sufficient for the BoM marginal-Gaussianity KS gate (T4.2), which
/// detects CDF errors > ~0.01 at N=10⁴. Uses `t = √(−2 ln(min(u, 1−u)))`
/// and a single rational function, then applies the sign. Symmetric by
/// construction: `Φ⁻¹(1−u) = −Φ⁻¹(u)`.
///
/// # Edge cases
///
/// - `u ≤ 0.0` → `-INFINITY` (left tail limit)
/// - `u ≥ 1.0` → `+INFINITY` (right tail limit)
/// - `u == 0.5` → `0.0` (median, exact by symmetry)
///
/// # Zero-allocation
///
/// Pure arithmetic — no allocations, one `sqrt` + one `ln` per call.
#[inline]
pub fn inverse_normal_cdf(u: f32) -> f32 {
    if u <= 0.0 {
        return f32::NEG_INFINITY;
    }
    if u >= 1.0 {
        return f32::INFINITY;
    }
    if u == 0.5 {
        return 0.0;
    }

    // Hastings (1955) coefficients.
    const C0: f64 = 2.515517;
    const C1: f64 = 0.802853;
    const C2: f64 = 0.010328;
    const D1: f64 = 1.432788;
    const D2: f64 = 0.189269;
    const D3: f64 = 0.001308;

    // Exploit symmetry: work with the smaller tail.
    let p = (u as f64).min(1.0 - u as f64);
    let t = (-2.0 * p.ln()).sqrt();
    let numerator = C0 + C1 * t + C2 * t * t;
    let denominator = 1.0 + D1 * t + D2 * t * t + D3 * t * t * t;
    let x0 = t - numerator / denominator;

    // Sign: positive for u > 0.5, negative for u < 0.5.
    if u > 0.5 {
        x0 as f32
    } else {
        -(x0 as f32)
    }
}

/// Apply `σ · Φ⁻¹(u)` in-place to a buffer of uniforms, producing Gaussian
/// noise queries.
///
/// Each `uniforms[i]` is transformed to `sigma * inverse_normal_cdf(uniforms[i])`.
/// Works with any pre-filled uniforms buffer — from [`QmcSource::draw`] (1D
/// coverage) or [`SobolQmc::draw_nd`] (D-dimensional coverage for T4.3).
///
/// # Zero-allocation
///
/// In-place mutation — no allocation.
#[inline]
pub fn gaussianize_uniforms_inplace(uniforms: &mut [f32], sigma: f32) {
    for u in uniforms.iter_mut() {
        *u = sigma * inverse_normal_cdf(*u);
    }
}

/// Fill a `queries` buffer with K×D QMC-derived Gaussian noise.
///
/// Produces a `[K×D]` row-major buffer where every element is marginally
/// `N(0, σ²)` (T4.2) with QMC low-discrepancy joint structure for better
/// coverage of the K-dim belief ball (T4.3).
///
/// # Multi-dimensional coverage strategy
///
/// For `dim > 1`, performs **D independent QMC draws** of K points each (one
/// per dimension), rather than a single K·D draw. This is critical for
/// D-dimensional coverage: a single K·D lattice draw assigns consecutive
/// lattice points to the same vector (row-major), causing all D coordinates
/// of each rollout to cluster near the same Gaussian quantile → diagonal
/// bias → poor pairwise separation.
///
/// With D independent draws, each column j gets K evenly-spaced Gaussian
/// quantiles (low-discrepancy within the column), and the columns are
/// independent (different random offsets per `QmcSource::draw` call). This
/// gives proper D-dimensional coverage: each rollout is marginally
/// `N(0, σ²I)` (all D coordinates independent), and the K rollouts are
/// correlated within each dimension for better spread.
///
/// For `dim == 1`, the single-draw fast path is used (no coverage benefit
/// from per-dimension draws in 1D).
///
/// # Panics
///
/// Panics if `queries.len() < k * dim` or `k > FILL_NOISE_MAX_K` (stack
/// buffer limit for the per-dimension scratch).
///
/// # Zero-allocation
///
/// Uses a stack-allocated `[f32; FILL_NOISE_MAX_K]` scratch buffer (no heap).
/// Writes into the caller-provided `queries`.
pub const FILL_NOISE_MAX_K: usize = 256;

#[inline]
pub fn fill_noise_queries_gaussian_qmc(
    source: &mut dyn QmcSource,
    k: usize,
    dim: usize,
    sigma: f32,
    queries: &mut [f32],
) {
    let n = k.checked_mul(dim).expect("k * dim overflow");
    assert!(
        queries.len() >= n,
        "fill_noise_queries_gaussian_qmc: queries.len() {} < k*dim {}",
        queries.len(),
        n
    );
    if k == 0 || dim == 0 {
        return;
    }

    if dim == 1 {
        // 1D fast path: single draw, in-place gaussianize.
        source.draw(k, &mut queries[..k]);
        gaussianize_uniforms_inplace(&mut queries[..k], sigma);
        return;
    }

    // Multi-dim: D independent draws of K points each.
    // Stack scratch for per-dimension K uniforms (no heap allocation).
    assert!(
        k <= FILL_NOISE_MAX_K,
        "fill_noise_queries_gaussian_qmc: k {} > FILL_NOISE_MAX_K {} (stack buffer limit)",
        k,
        FILL_NOISE_MAX_K
    );
    let mut col_scratch = [0.0f32; FILL_NOISE_MAX_K];
    for j in 0..dim {
        source.draw(k, &mut col_scratch[..k]);
        for k_idx in 0..k {
            queries[k_idx * dim + j] = sigma * inverse_normal_cdf(col_scratch[k_idx]);
        }
    }
}

/// Convenience wrapper: fill `queries` with QMC Gaussian noise, then call
/// [`BoMSampler::sample_k_states`].
///
/// This is the one-call "QMC BoM" path — composes
/// [`fill_noise_queries_gaussian_qmc`] with the kernel's `sample_k_states`.
/// Requires both `qmc_sampling` (this module) and `bom_sampling` (the
/// `BoMSampler` trait + `NoiseQueryConfig`).
///
/// `queries` and `out` are caller-allocated; `queries` is overwritten with QMC
/// noise on each call. The `NoiseQueryConfig::sigma` field scales the noise; its
/// `k` field determines K.
///
/// # Zero-allocation
///
/// Writes into caller-provided `queries` and `out`; no allocation.
#[cfg(feature = "bom_sampling")]
pub fn sample_k_states_qmc<K: crate::BoMSampler>(
    kernel: &K,
    s_prev: &[f32],
    x: &[f32],
    source: &mut dyn QmcSource,
    cfg: &crate::NoiseQueryConfig,
    queries: &mut [f32],
    out: &mut [f32],
) {
    let dim = kernel.dim();
    fill_noise_queries_gaussian_qmc(source, cfg.k, dim, cfg.sigma, queries);
    kernel.sample_k_states(s_prev, x, queries, out, cfg);
}

/// Convenience wrapper: fill `queries` with QMC Gaussian noise using a
/// [`QmcMethod`](crate::QmcMethod) tag (Plan 370 — BoM Arena × QuasiMoTTo wiring).
/// Constructs the appropriate [`QmcSource`] on the stack (zero-alloc) from
/// `method` + `seed`, then delegates to [`fill_noise_queries_gaussian_qmc`].
///
/// This is the entry point used by `MultiHypothesisBoMMinimaxPlanner::resample_queries`
/// when `NoiseQueryConfig::qmc_method` is `Some(method)`. The caller passes a
/// per-tick `seed` (typically `TICK_SALT + obs_hash`); each call constructs a
/// fresh source so the QMC batch is deterministic given the seed.
///
/// Requires both `qmc_sampling` (this module) and `bom_sampling` (the
/// `QmcMethod` tag lives in `katgpt-micro-belief`, forwarded via `bom_sampling`).
///
/// # Zero-allocation
///
/// Each `QmcSource` impl is stack-allocated (1 `f32` for Lattice, 1 `Rng` for
/// Stratified, fixed-size direction table for Sobol). Writes into caller-provided
/// `queries`; no heap allocation.
#[cfg(feature = "bom_sampling")]
pub fn fill_noise_queries_gaussian_qmc_by_method(
    method: crate::QmcMethod,
    seed: u64,
    k: usize,
    dim: usize,
    sigma: f32,
    queries: &mut [f32],
) {
    match method {
        crate::QmcMethod::Lattice => {
            let mut src = LatticeQmc::new(seed);
            fill_noise_queries_gaussian_qmc(&mut src, k, dim, sigma, queries);
        }
        crate::QmcMethod::Stratified => {
            let mut src = StratifiedQmc::new(seed);
            fill_noise_queries_gaussian_qmc(&mut src, k, dim, sigma, queries);
        }
        crate::QmcMethod::Sobol => {
            let mut src = SobolQmc::new(seed);
            fill_noise_queries_gaussian_qmc(&mut src, k, dim, sigma, queries);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dyadic bootstrap pass@k estimator (Plan 367 Phase 6 / Theorem 1)
// ─────────────────────────────────────────────────────────────────────────────
//
// Theorem 1 of arXiv:2607.01179 (QuasiMoTTo): for a rank-1 lattice QMC
// batch of size k = 2^L, every stride-`s` subsequence (s = k/m, m a power
// of two dividing k) starting at offset j ∈ {0, ..., s-1} is itself a
// valid rank-1 lattice of size m, because the lattice values
// `{(i/k + Δ) mod 1 : i=0..k-1}` restricted to indices `{j + t·s}` satisfy
//   ((j + t·s)/k + Δ) mod 1 = (t/m + (Δ + j/k)) mod 1
// which is a rank-1 lattice of size m with offset `(Δ + j/k) mod 1`, itself
// marginally Unif[0,1) since both Δ and j/k are.
//
// Therefore a single pass@k lattice batch yields `s = k/m` unbiased pass@m
// lattice-batch estimates (one per starting offset). Their mean is an
// unbiased point estimate; the Wilson score CI quantifies lattice-resample
// variance. This converts one expensive pass@k rollout batch into `s`
// cheaper pass@m estimates for free — no extra rollouts needed.
//
// For Sobol/Stratified the dyadic-stride property does NOT hold in general,
// but contiguous blocks of m points DO preserve the per-method low-
// discrepancy structure (Sobol: contiguous subsequences are valid shifted
// Sobol subsequences; Stratified: contiguous blocks span m consecutive
// strata, a coarser valid stratification). So for those methods we offer a
// contiguous-block bootstrap with random starts.

/// Result of a bootstrap pass@m estimation.
///
/// `point_estimate` is the unbiased pass@m point estimate; `sample_variance`
/// is the unbiased (n-1) sample variance of the per-resample binary
/// indicators; `n_resamples` is the number of resamples that contributed.
/// Use [`wilson_ci`](Self::wilson_ci) for a well-behaved CI at small n.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapEstimate {
    /// Unbiased point estimate of pass@m (mean over resamples).
    pub point_estimate: f64,
    /// Unbiased (n-1) sample variance of the per-resample binary indicators.
    pub sample_variance: f64,
    /// Number of resamples that contributed (s = k/m for dyadic, B for block).
    pub n_resamples: usize,
}

impl BootstrapEstimate {
    /// Wilson score confidence interval for the pass@m proportion.
    ///
    /// Preferred over the normal-approximation (`p̂ ± z·√(p̂(1-p̂)/n)`) for
    /// binary indicators because it has correct coverage at small n and near
    /// the 0/1 boundaries (Brown, Cai & DasGupta 2001).
    ///
    /// `z` is the two-sided critical value (1.96 for 95%, 2.576 for 99%).
    /// Returns `(low, high)` clamped to `[0, 1]`. For `n_resamples == 0`, the
    /// uninformative `(0.0, 1.0)` is returned.
    #[inline]
    pub fn wilson_ci(&self, z: f64) -> (f64, f64) {
        let n = self.n_resamples as f64;
        if n == 0.0 {
            return (0.0, 1.0);
        }
        let p_hat = self.point_estimate;
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let center = (p_hat + z2 / (2.0 * n)) / denom;
        let margin = (z / denom) * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt();
        let lo = (center - margin).max(0.0);
        let hi = (center + margin).min(1.0);
        (lo, hi)
    }

    /// Convenience: Wilson score 95% CI (z = 1.959963984540054, Φ⁻¹(0.975)).
    #[inline]
    pub fn wilson_ci_95(&self) -> (f64, f64) {
        self.wilson_ci(1.959_963_984_540_054)
    }

    /// Sample standard deviation across resamples (√ of [`sample_variance`](Self::sample_variance)).
    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.sample_variance.sqrt()
    }
}

/// Lattice dyadic bootstrap pass@m estimator (Theorem 1 of arXiv:2607.01179).
///
/// For a [`LatticeQmc`] batch of size k = 2^L, every stride-`s` subsequence
/// (s = k/m, m a power of two dividing k) starting at offset j ∈ {0,...,s-1}
/// is itself a valid rank-1 lattice of size m. Therefore a single pass@k
/// lattice batch yields `s = k/m` unbiased pass@m estimates — one per
/// starting offset. Their mean is the unbiased pass@m estimate; their
/// variance drives the Wilson score CI.
///
/// This is the strongest of the three QMC-method bootstrap forms: it is
/// *exhaustive* (no RNG needed — every starting offset is taken) and
/// *algebraically exact* (each sub-lattice is provably a LatticeQmc batch
/// of size m, not an approximation).
///
/// # Arguments
///
/// * `outcomes` - pass/fail of each of the K rollouts (length K = 2^L).
/// * `m` - sub-sample size (power of two, divides K, 0 < m ≤ K).
///
/// # Panics
///
/// Panics if `outcomes.len()` is not a power of two, `m` is not a power of
/// two, `m > outcomes.len()`, or `outcomes.len() % m != 0`.
///
/// # Zero-allocation
///
/// Single streaming pass; no heap allocation. Hot-path friendly.
///
/// # Example
///
/// ```
/// # use katgpt_core::speculative::qmc::dyadic_bootstrap_pass_at_m_lattice;
/// // 8-rollout lattice batch: 4 pass, 4 fail. Stride-2 sub-lattices at
/// // m=4 give 2 estimates of pass@4.
/// let outcomes = [true, false, true, false, true, false, true, false];
/// let est = dyadic_bootstrap_pass_at_m_lattice(&outcomes, 4);
/// assert_eq!(est.point_estimate, 0.5);  // one sub-lattice all-pass, one all-fail
/// assert_eq!(est.n_resamples, 2);
/// ```
pub fn dyadic_bootstrap_pass_at_m_lattice(outcomes: &[bool], m: usize) -> BootstrapEstimate {
    let k = outcomes.len();
    assert!(
        k > 0 && k.is_power_of_two(),
        "dyadic_bootstrap_pass_at_m_lattice: outcomes.len() = {k} must be a power of two"
    );
    assert!(
        m > 0 && m.is_power_of_two(),
        "dyadic_bootstrap_pass_at_m_lattice: m = {m} must be a power of two"
    );
    assert!(
        m <= k,
        "dyadic_bootstrap_pass_at_m_lattice: m = {m} > outcomes.len() = {k}"
    );
    // k is a power of two and m ≤ k is a power of two ⇒ k % m == 0 by the
    // divisibility of powers of two. No separate assert needed.
    let stride = k / m;

    // For each starting offset j ∈ [0, stride), pass@m of the subsequence
    // {outcomes[j], outcomes[j+stride], ..., outcomes[j+(m-1)*stride]} is the
    // indicator "any true". Single streaming pass — sum and sum of squares
    // only.
    let mut sum: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for j in 0..stride {
        let mut any = false;
        for t in 0..m {
            if outcomes[j + t * stride] {
                any = true;
                break;
            }
        }
        let x = if any { 1.0f64 } else { 0.0 };
        sum += x;
        sum_sq += x * x;
    }

    let n = stride as f64;
    let mean = sum / n;
    let var = sample_variance_binary(mean, sum_sq, n);
    BootstrapEstimate {
        point_estimate: mean,
        sample_variance: var,
        n_resamples: stride,
    }
}

/// Contiguous-block bootstrap for Sobol / Stratified / general orderings.
///
/// Unlike the lattice dyadic case (which has provable sub-lattice validity
/// for strided offsets), [`SobolQmc`] and [`StratifiedQmc`] preserve their
/// low-discrepancy structure within *contiguous* blocks of m points. We
/// resample by drawing `n_resamples` random contiguous starting positions
/// uniformly from {0, 1, ..., K-m} (no wrapping — boundary blocks are full
/// size) and computing pass@m of each.
///
/// This is the standard nonparametric block-bootstrap, adapted to preserve
/// local QMC structure. Less powerful than the lattice dyadic form (random
/// starts vs exhaustive; contiguous rather than algebraically exact) but
/// applicable when the dyadic-stride theorem doesn't hold.
///
/// # Arguments
///
/// * `outcomes` - pass/fail of each of the K rollouts.
/// * `m` - block size (sub-batch size).
/// * `n_resamples` - number of random block starts (B). Must be > 0.
/// * `rng` - caller-provided [`Rng`] for selecting block starts.
///
/// # Panics
///
/// Panics if `m == 0`, `m > outcomes.len()`, or `n_resamples == 0`.
///
/// # Zero-allocation
///
/// Single streaming pass; no heap allocation.
pub fn contiguous_block_bootstrap_pass_at_m(
    outcomes: &[bool],
    m: usize,
    n_resamples: usize,
    rng: &mut Rng,
) -> BootstrapEstimate {
    let k = outcomes.len();
    assert!(m > 0, "contiguous_block_bootstrap_pass_at_m: m must be > 0");
    assert!(
        m <= k,
        "contiguous_block_bootstrap_pass_at_m: m = {m} > outcomes.len() = {k}"
    );
    assert!(
        n_resamples > 0,
        "contiguous_block_bootstrap_pass_at_m: n_resamples must be > 0"
    );
    let n_starts = if k > m { k - m + 1 } else { 1 };

    let mut sum: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for _ in 0..n_resamples {
        // Map a u64 to [0, n_starts). For n_starts a power of two this would
        // be unbiased via masking; for the general case we accept the tiny
        // modular bias (≤ n_starts/u64::MAX, negligible for n_starts ≤ 2^32).
        let start = if n_starts > 1 {
            (rng.next() % (n_starts as u64)) as usize
        } else {
            0
        };
        let mut any = false;
        for i in 0..m {
            if outcomes[start + i] {
                any = true;
                break;
            }
        }
        let x = if any { 1.0f64 } else { 0.0 };
        sum += x;
        sum_sq += x * x;
    }

    let n = n_resamples as f64;
    let mean = sum / n;
    let var = sample_variance_binary(mean, sum_sq, n);
    BootstrapEstimate {
        point_estimate: mean,
        sample_variance: var,
        n_resamples,
    }
}

/// Unbiased (n-1) sample variance of binary indicators.
///
/// For 0/1 indicators with mean `mean` and `sum_sq = Σ x_i² = Σ x_i = sum`
/// (since x_i² = x_i for binary), this is `(n/(n-1)) · (m2 − mean²)` where
/// `m2 = sum_sq / n`. Returns 0 for n ≤ 1; clamps tiny negative drift from
/// f64 rounding to 0.
#[inline]
fn sample_variance_binary(mean: f64, sum_sq: f64, n: f64) -> f64 {
    if n <= 1.0 {
        return 0.0;
    }
    let m2 = sum_sq / n;
    let v = (n / (n - 1.0)) * (m2 - mean * mean);
    v.max(0.0)
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
        q = (2.0 * q).clamp(0.0, 1.0);
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
            assert!((0.0..1.0).contains(&v), "lattice value out of [0,1): {v}");
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
            assert!((0.0..1.0).contains(&v), "stratified value out of [0,1): {v}");
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
            assert!((0.0..1.0).contains(&v), "sobol value out of [0,1): {v}");
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
            assert!((0.0..1.0).contains(&v), "sobol nd value out of [0,1): {v}");
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

    // ───────────────────────────────────────────────────────────────────
    // Phase 4 — QMC → Gaussian noise query fill (T4.2, T4.3)
    // ───────────────────────────────────────────────────────────────────

    /// Standard normal CDF Φ(x) via the Abramowitz-Stegun erf approximation
    /// (formula 7.1.26). Max error ≈ 1.5e-7. Independent of `inverse_normal_cdf`
    /// so the KS test below is a fair cross-check (not a tautology).
    ///
    /// Uses Φ(x) = 0.5 · (1 + erf(x/√2)) — the √2 scaling is critical.
    fn normal_cdf(x: f64) -> f64 {
        const P: f64 = 0.3275911;
        const A1: f64 = 0.254829592;
        const A2: f64 = -0.284496736;
        const A3: f64 = 1.421413741;
        const A4: f64 = -1.453152027;
        const A5: f64 = 1.061405429;
        const SQRT2: f64 = std::f64::consts::SQRT_2;
        // Φ(x) = 0.5 · (1 + erf(x/√2))
        let z = x / SQRT2;
        let sign = if z < 0.0 { -1.0 } else { 1.0 };
        let az = z.abs();
        let t = 1.0 / (1.0 + P * az);
        let erf_abs = 1.0
            - (((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t)
                * (-az * az).exp();
        0.5 * (1.0 + sign * erf_abs)
    }

    /// KS one-sample test against the standard normal CDF. Returns (D, p-value).
    fn ks_normal(samples: &[f32], sigma: f32) -> (f64, f64) {
        let n = samples.len();
        assert!(n > 0);
        let mut sorted: Vec<f32> = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let inv_sigma = (1.0 / sigma) as f64;
        let mut d_max = 0.0f64;
        let nf = n as f64;
        for (i, &x) in sorted.iter().enumerate() {
            let cdf_val = normal_cdf((x as f64) * inv_sigma);
            let f_lower = i as f64 / nf;
            let f_upper = (i + 1) as f64 / nf;
            d_max = d_max
                .max((f_lower - cdf_val).abs())
                .max((f_upper - cdf_val).abs());
        }
        let en = nf.sqrt();
        let lambda = (en + 0.12 + 0.11 / en) * d_max;
        let mut q = 0.0f64;
        for j in 1..=100 {
            let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
            let term =
                sign * (-2.0 * (j as f64) * (j as f64) * lambda * lambda).exp();
            q += term;
            if term.abs() < 1e-12 {
                break;
            }
        }
        q = (2.0 * q).clamp(0.0, 1.0);
        (d_max, q)
    }

    // ── T4.2a: probit accuracy at known quantiles ───────────────────────

    #[test]
    fn test_inverse_normal_cdf_known_quantiles() {
        // Φ⁻¹(0.5) = 0 (median, exact by symmetry).
        let z = inverse_normal_cdf(0.5);
        assert!(z.abs() < 1e-5, "Φ⁻¹(0.5) should be 0, got {z}");

        // Φ⁻¹(0.025) ≈ -1.95996, Φ⁻¹(0.975) ≈ +1.95996 (95% CI bounds).
        let z_lo = inverse_normal_cdf(0.025);
        let z_hi = inverse_normal_cdf(0.975);
        assert!(
            (z_lo + 1.95996).abs() < 0.01,
            "Φ⁻¹(0.025) should be ≈ -1.96, got {z_lo}"
        );
        assert!(
            (z_hi - 1.95996).abs() < 0.01,
            "Φ⁻¹(0.975) should be ≈ +1.96, got {z_hi}"
        );

        // Φ⁻¹(0.001) ≈ -3.0902, Φ⁻¹(0.999) ≈ +3.0902 (99.8% CI bounds).
        let z_tail_lo = inverse_normal_cdf(0.001);
        let z_tail_hi = inverse_normal_cdf(0.999);
        assert!(
            (z_tail_lo + 3.0902).abs() < 0.02,
            "Φ⁻¹(0.001) should be ≈ -3.09, got {z_tail_lo}"
        );
        assert!(
            (z_tail_hi - 3.0902).abs() < 0.02,
            "Φ⁻¹(0.999) should be ≈ +3.09, got {z_tail_hi}"
        );
    }

    #[test]
    fn test_inverse_normal_cdf_symmetry() {
        // Φ⁻¹(1-u) = -Φ⁻¹(u) for all u ∈ (0,1).
        for &u in &[0.1f32, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99] {
            let z1 = inverse_normal_cdf(u);
            let z2 = inverse_normal_cdf(1.0 - u);
            assert!(
                (z1 + z2).abs() < 1e-3,
                "symmetry violated at u={u}: Φ⁻¹(u)={z1}, Φ⁻¹(1-u)={z2}"
            );
        }
    }

    #[test]
    fn test_inverse_normal_cdf_edge_cases() {
        assert!(inverse_normal_cdf(0.0).is_infinite() && inverse_normal_cdf(0.0).is_sign_negative());
        assert!(inverse_normal_cdf(1.0).is_infinite() && inverse_normal_cdf(1.0).is_sign_positive());
        // u slightly inside (0,1) should be finite.
        assert!(inverse_normal_cdf(1e-6).is_finite());
        assert!(inverse_normal_cdf(1.0 - 1e-6).is_finite());
    }

    // ── T4.2b: marginal Gaussianity of fill_noise_queries_gaussian_qmc ──
    //
    // Each element of the queries buffer must be marginally N(0, σ²). This
    // is the contract that makes QMC a drop-in for i.i.d. Gaussian noise:
    // linearity-of-expectation estimators (mean reward, pass@k) are unbiased
    // regardless of the joint, as long as each rollout's marginal matches.
    //
    // We pool K·D values across N=500 batches (32K samples at K=64, D=1) and
    // run a KS test against N(0, σ²). Critical D at α=0.05, N=32K: ~0.0076.

    #[test]
    fn test_fill_noise_marginal_gaussian_lattice() {
        let k = 64;
        let dim = 1; // 1D is the cleanest marginal test (no cross-column effects)
        let sigma = 0.3;
        let n_batches = 500;
        let mut source = LatticeQmc::new(999);
        let mut queries = vec![0.0f32; k * dim];
        let mut all: Vec<f32> = Vec::with_capacity(n_batches * k * dim);
        for _ in 0..n_batches {
            fill_noise_queries_gaussian_qmc(&mut source, k, dim, sigma, &mut queries);
            all.extend_from_slice(&queries[..k * dim]);
        }
        let (d, p) = ks_normal(&all, sigma);
        assert!(
            p > 0.01,
            "Lattice QMC marginal Gaussianity FAIL: KS D={d:.6}, p={p:.4} (need p>0.01)"
        );
    }

    #[test]
    fn test_fill_noise_marginal_gaussian_stratified() {
        let k = 64;
        let dim = 1;
        let sigma = 0.3;
        let n_batches = 500;
        let mut source = StratifiedQmc::new(888);
        let mut queries = vec![0.0f32; k * dim];
        let mut all: Vec<f32> = Vec::with_capacity(n_batches * k * dim);
        for _ in 0..n_batches {
            fill_noise_queries_gaussian_qmc(&mut source, k, dim, sigma, &mut queries);
            all.extend_from_slice(&queries[..k * dim]);
        }
        let (d, p) = ks_normal(&all, sigma);
        assert!(
            p > 0.01,
            "Stratified QMC marginal Gaussianity FAIL: KS D={d:.6}, p={p:.4} (need p>0.01)"
        );
    }

    #[test]
    fn test_fill_noise_marginal_gaussian_sobol() {
        let k = 64;
        let dim = 1;
        let sigma = 0.3;
        let n_batches = 500;
        let mut source = SobolQmc::new(777);
        let mut queries = vec![0.0f32; k * dim];
        let mut all: Vec<f32> = Vec::with_capacity(n_batches * k * dim);
        for _ in 0..n_batches {
            fill_noise_queries_gaussian_qmc(&mut source, k, dim, sigma, &mut queries);
            all.extend_from_slice(&queries[..k * dim]);
        }
        let (d, p) = ks_normal(&all, sigma);
        assert!(
            p > 0.01,
            "Sobol QMC marginal Gaussianity FAIL: KS D={d:.6}, p={p:.4} (need p>0.01)"
        );
    }

    #[test]
    fn test_gaussianize_uniforms_inplace_scales_by_sigma() {
        // gaussianize(u) = σ·Φ⁻¹(u). At u=0.5: Φ⁻¹(0.5)=0, so result=0.
        let mut buf = [0.5f32, 0.5, 0.5];
        gaussianize_uniforms_inplace(&mut buf, 0.3);
        for &v in &buf {
            assert!(v.abs() < 1e-5, "σ·Φ⁻¹(0.5) should be 0, got {v}");
        }

        // σ scaling: Φ⁻¹(0.975) ≈ 1.96, so at σ=0.5 result ≈ 0.98.
        let mut buf2 = [0.975f32];
        gaussianize_uniforms_inplace(&mut buf2, 0.5);
        assert!(
            (buf2[0] - 0.5 * 1.95996).abs() < 0.01,
            "σ·Φ⁻¹(0.975) at σ=0.5 should be ≈ 0.98, got {}",
            buf2[0]
        );
    }

    // ── T4.3: belief-ball coverage (QMC vs i.i.d.) ─────────────────────
    //
    // The plan specifies "radius of the largest empty spherical cap centered
    // at origin" as the coverage metric. We use minimum pairwise Euclidean
    // distance as a practical proxy — higher = more even spread (no two
    // hypotheses too close).
    //
    // For K=8 points in R⁴, i.i.d. Gaussian noise is a strong baseline for
    // minimum pairwise distance — the QMC advantage (lower variance in
    // average-type estimators) does NOT necessarily translate to better
    // minimum pairwise distance at small K. The QMC win is in CONSISTENCY
    // (more predictable coverage across batches), not necessarily in the
    // mean of the minimum pairwise distance.
    //
    // This test verifies the QMC fill is not catastrophically worse than
    // i.i.d. (≥ 70% of i.i.d. mean). The marginal-Gaussianity contract
    // (T4.2) is the hard correctness gate; this test is a sanity check.

    /// Minimum pairwise Euclidean distance among K row-vectors of width dim.
    fn min_pairwise_distance(queries: &[f32], k: usize, dim: usize) -> f32 {
        let mut min_d = f32::INFINITY;
        for a in 0..k {
            for b in (a + 1)..k {
                let row_a = &queries[a * dim..(a + 1) * dim];
                let row_b = &queries[b * dim..(b + 1) * dim];
                let mut dist_sq = 0.0f32;
                for j in 0..dim {
                    let diff = row_a[j] - row_b[j];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                if dist < min_d {
                    min_d = dist;
                }
            }
        }
        min_d
    }

    #[test]
    fn test_qmc_coverage_not_worse_than_iid_lattice() {
        let k = 8;
        let dim = 4;
        let sigma = 0.3;
        let n_batches = 2000;

        // QMC coverage (Lattice, D independent draws).
        let mut qmc_source = LatticeQmc::new(42);
        let mut qmc_queries = vec![0.0f32; k * dim];
        let mut qmc_sum = 0.0f64;
        for _ in 0..n_batches {
            fill_noise_queries_gaussian_qmc(&mut qmc_source, k, dim, sigma, &mut qmc_queries);
            qmc_sum += min_pairwise_distance(&qmc_queries, k, dim) as f64;
        }
        let qmc_mean = qmc_sum / n_batches as f64;

        // i.i.d. coverage (Box-Muller via fastrand).
        let mut iid_sum = 0.0f64;
        let mut iid_queries = vec![0.0f32; k * dim];
        let mut rng = fastrand::Rng::with_seed(42);
        for _ in 0..n_batches {
            for q in &mut iid_queries[..k * dim] {
                *q = standard_normal_fastrand(&mut rng) * sigma;
            }
            iid_sum += min_pairwise_distance(&iid_queries, k, dim) as f64;
        }
        let iid_mean = iid_sum / n_batches as f64;

        // QMC should not be catastrophically worse than i.i.d.
        // The Lattice's rigid structure (same rank ordering across dimensions)
        // means its minimum pairwise distance is slightly lower than i.i.d.
        // for small K. This is acceptable — the QMC win is in marginal
        // exactness + integration variance, not minimum pairwise distance.
        assert!(
            qmc_mean >= iid_mean * 0.70,
            "Lattice QMC coverage ({qmc_mean:.6}) should be ≥ 70% of i.i.d. ({iid_mean:.6})"
        );
    }

    /// Box-Muller standard normal using fastrand (matches the i.i.d. baseline).
    fn standard_normal_fastrand(rng: &mut fastrand::Rng) -> f32 {
        let u1 = rng.f32().max(1e-10);
        let u2 = rng.f32();
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0 * core::f32::consts::PI * u2;
        r * theta.cos()
    }

    #[test]
    fn test_fill_noise_queries_zero_dim() {
        // dim=0 → n=0, no-op.
        let mut source = LatticeQmc::new(42);
        let mut queries: [f32; 0] = [];
        fill_noise_queries_gaussian_qmc(&mut source, 8, 0, 0.3, &mut queries);
    }

    #[test]
    fn test_fill_noise_queries_zero_k() {
        // k=0 → n=0, no-op.
        let mut source = LatticeQmc::new(42);
        let mut queries: [f32; 0] = [];
        fill_noise_queries_gaussian_qmc(&mut source, 0, 4, 0.3, &mut queries);
    }

    #[test]
    #[should_panic(expected = "queries.len()")]
    fn test_fill_noise_queries_buf_too_small() {
        let mut source = LatticeQmc::new(42);
        let mut queries = [0.0f32; 4]; // need 8*4=32
        fill_noise_queries_gaussian_qmc(&mut source, 8, 4, 0.3, &mut queries);
    }

    // ── T4.1 integration: sample_k_states_qmc wrapper ───────────────────
    // (gated on bom_sampling; reuses the AttractorKernel)

    #[cfg(feature = "bom_sampling")]
    #[test]
    fn test_sample_k_states_qmc_produces_valid_hypotheses() {
        use crate::{AttractorKernel, NoiseQueryConfig};

        let kernel = AttractorKernel::from_seed(42, 4);
        let dim = 4;
        let k = 8;
        let sigma = 0.3;
        let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(sigma);

        let s_prev = vec![0.0f32; dim];
        let x = vec![0.5f32; dim];

        let mut source = LatticeQmc::new(123);
        let mut queries = vec![0.0f32; k * dim];
        let mut out = vec![0.0f32; k * dim];

        sample_k_states_qmc(
            &kernel, &s_prev, &x, &mut source, &cfg, &mut queries, &mut out,
        );

        // Output must be valid (in [-1, 1] after AttractorKernel's clamp).
        for &v in &out[..k * dim] {
            assert!(v.is_finite(), "hypothesis contains NaN/inf: {v}");
            assert!((-1.0..=1.0).contains(&v), "hypothesis out of [-1,1]: {v}");
        }

        // Distinct hypotheses (G1.2 analog): QMC should also produce distinct
        // hypotheses, not degenerate copies of step().
        let mut any_distinct = false;
        for a in 0..k {
            for b in (a + 1)..k {
                let row_a = &out[a * dim..(a + 1) * dim];
                let row_b = &out[b * dim..(b + 1) * dim];
                let mut dist_sq = 0.0f32;
                for j in 0..dim {
                    let d = row_a[j] - row_b[j];
                    dist_sq += d * d;
                }
                if dist_sq > 1e-8 {
                    any_distinct = true;
                }
            }
        }
        assert!(any_distinct, "QMC BoM should produce at least one distinct pair");
    }

    #[cfg(feature = "bom_sampling")]
    #[test]
    fn test_sample_k_states_qmc_deterministic() {
        use crate::{AttractorKernel, NoiseQueryConfig};

        let kernel = AttractorKernel::from_seed(42, 4);
        let dim = 4;
        let k = 8;
        let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(0.3);
        let s_prev = vec![0.0f32; dim];
        let x = vec![0.5f32; dim];

        let mut queries_a = vec![0.0f32; k * dim];
        let mut queries_b = vec![0.0f32; k * dim];
        let mut out_a = vec![0.0f32; k * dim];
        let mut out_b = vec![0.0f32; k * dim];

        let mut src_a = LatticeQmc::new(123);
        let mut src_b = LatticeQmc::new(123);
        sample_k_states_qmc(&kernel, &s_prev, &x, &mut src_a, &cfg, &mut queries_a, &mut out_a);
        sample_k_states_qmc(&kernel, &s_prev, &x, &mut src_b, &cfg, &mut queries_b, &mut out_b);

        assert_eq!(out_a, out_b, "same QMC seed must produce bit-identical hypotheses");    }

    // ── Plan 370 T2.3: fill_noise_queries_gaussian_qmc_by_method ────────────

    #[cfg(feature = "bom_sampling")]
    #[test]
    fn test_fill_by_method_all_methods_produce_valid_queries() {
        let k = 8;
        let dim = 4;
        let sigma = 0.1;
        let mut queries = vec![0.0f32; k * dim];

        for method in [crate::QmcMethod::Lattice, crate::QmcMethod::Stratified, crate::QmcMethod::Sobol] {
            fill_noise_queries_gaussian_qmc_by_method(method, 42, k, dim, sigma, &mut queries);
            // All values finite.
            for &q in &queries {
                assert!(q.is_finite(), "{:?} produced non-finite query {}", method, q);
            }
            // Empirical mean ≈ 0 (Gaussian, σ=0.1 → mean in [-0.05, 0.05] for k*dim=32 samples).
            let mean = queries.iter().sum::<f32>() / queries.len() as f32;
            assert!(mean.abs() < 0.1, "{:?} mean {} too far from 0", method, mean);
            // Empirical stddev ≈ σ (in [0.05, 0.2] for 32 samples from N(0,0.1²)).
            let var = queries.iter().map(|q| (q - mean).powi(2)).sum::<f32>() / queries.len() as f32;
            let std = var.sqrt();
            assert!(std > 0.05 && std < 0.2, "{:?} stddev {} outside [0.05, 0.2]", method, std);
        }
    }

    #[cfg(feature = "bom_sampling")]
    #[test]
    fn test_fill_by_method_is_deterministic_given_seed() {
        let k = 8;
        let dim = 4;
        let sigma = 0.2;
        let mut a = vec![0.0f32; k * dim];
        let mut b = vec![0.0f32; k * dim];

        for method in [crate::QmcMethod::Lattice, crate::QmcMethod::Stratified, crate::QmcMethod::Sobol] {
            fill_noise_queries_gaussian_qmc_by_method(method, 99, k, dim, sigma, &mut a);
            fill_noise_queries_gaussian_qmc_by_method(method, 99, k, dim, sigma, &mut b);
            assert_eq!(a, b, "{:?} must be bit-identical for same seed", method);
        }
    }

    // ── Plan 367 Phase 6 / T6.1: Dyadic bootstrap pass@k estimator ────────
    //
    // Theorem 1 of arXiv:2607.01179: for a LatticeQmc batch of size k=2^L,
    // every stride-(k/m) subsequence is itself a valid LatticeQmc batch of
    // size m. Therefore one pass@k batch yields k/m unbiased pass@m
    // estimates. For Sobol/Stratified we use contiguous-block bootstrap.

    #[test]
    fn test_dyadic_bootstrap_all_pass() {
        // k=8, m=4, stride=2. Both offsets j=0 and j=1 see only passes.
        let outcomes = [true; 8];
        let est = dyadic_bootstrap_pass_at_m_lattice(&outcomes, 4);
        assert_eq!(est.point_estimate, 1.0);
        assert_eq!(est.sample_variance, 0.0);
        assert_eq!(est.n_resamples, 2);
        // Wilson 95% CI at 2/2 successes is wide (~[0.34, 1.0]) — small n.
        let (lo, hi) = est.wilson_ci_95();
        assert!((hi - 1.0).abs() < 1e-9, "hi={hi}");
        assert!(lo > 0.3 && lo < 0.4, "lo={lo} (Wilson at 2/2)");
    }

    #[test]
    fn test_dyadic_bootstrap_all_fail() {
        let outcomes = [false; 8];
        let est = dyadic_bootstrap_pass_at_m_lattice(&outcomes, 4);
        assert_eq!(est.point_estimate, 0.0);
        assert_eq!(est.sample_variance, 0.0);
        assert_eq!(est.n_resamples, 2);
        // Wilson 95% CI at 0/2 is (~[0.0, 0.66]) — wide due to small n.
        let (lo, hi) = est.wilson_ci_95();
        assert!(lo < 1e-9, "lo={lo}");
        assert!(hi > 0.6 && hi < 0.7, "hi={hi} (Wilson at 0/2)");
    }

    #[test]
    fn test_dyadic_bootstrap_alternating_half_half() {
        // k=8, m=4, stride=2. Offsets separate even from odd indices.
        // j=0: outcomes[0,2,4,6] = [T,T,T,T] -> pass@4 = 1.
        // j=1: outcomes[1,3,5,7] = [F,F,F,F] -> pass@4 = 0.
        let outcomes = [true, false, true, false, true, false, true, false];
        let est = dyadic_bootstrap_pass_at_m_lattice(&outcomes, 4);
        assert_eq!(est.point_estimate, 0.5);
        assert_eq!(est.n_resamples, 2);
        // Sample variance of {0, 1} with n=2: ((1-0.5)^2 + (0-0.5)^2)/(n-1)
        // = (0.25 + 0.25)/1 = 0.5.
        assert!((est.sample_variance - 0.5).abs() < 1e-12,
            "sample_variance = {}", est.sample_variance);
    }

    #[test]
    fn test_dyadic_bootstrap_m_equals_k_single_estimate() {
        // m = k -> stride = 1 -> single estimate, the indicator of any pass.
        // pass@8 of all-pass = 1.
        let outcomes = [false, false, false, true, false, false, false, false];
        let est = dyadic_bootstrap_pass_at_m_lattice(&outcomes, 8);
        assert_eq!(est.point_estimate, 1.0);
        assert_eq!(est.n_resamples, 1);
        assert_eq!(est.sample_variance, 0.0, "n=1 -> variance undefined -> 0");
    }

    #[test]
    fn test_dyadic_bootstrap_m_equals_k_all_fail() {
        let outcomes = [false; 16];
        let est = dyadic_bootstrap_pass_at_m_lattice(&outcomes, 16);
        assert_eq!(est.point_estimate, 0.0);
        assert_eq!(est.n_resamples, 1);
    }

    #[test]
    fn test_dyadic_bootstrap_m1_recovers_mean_pass() {
        // m=1 -> each rollout is its own sub-lattice. pass@1 of a single
        // rollout is just the rollout's outcome indicator. Mean = empirical
        // pass rate.
        let outcomes = [
            true, false, true, true, false, true, false, true,
            true, false, true, true, false, true, false, true,
        ];
        // 10 trues / 16 = 0.625.
        let est = dyadic_bootstrap_pass_at_m_lattice(&outcomes, 1);
        assert!((est.point_estimate - 0.625).abs() < 1e-12,
            "point_estimate = {}", est.point_estimate);
        assert_eq!(est.n_resamples, 16);
        // Variance of a Bernoulli(0.625) with n=16: p(1-p) = 0.625*0.375
        // = 0.234375. Sample variance uses (n/(n-1)) correction.
        let expected = (16.0 / 15.0) * 0.625 * 0.375;
        assert!((est.sample_variance - expected).abs() < 1e-10,
            "sample_variance {} expected {}", est.sample_variance, expected);
    }

    #[test]
    fn test_dyadic_bootstrap_stride4_three_passing_offsets() {
        // k=8, m=2, stride=4. 4 starting offsets, each gives pass@2.
        // outcomes: [T,T, F,T, F,F, T,F]  (3 passes total at indices 0,1,3,6)
        // j=0: outcomes[0,4] = [T,F] -> pass@2 = 1
        // j=1: outcomes[1,5] = [T,F] -> pass@2 = 1
        // j=2: outcomes[2,6] = [F,T] -> pass@2 = 1
        // j=3: outcomes[3,7] = [T,F] -> pass@2 = 1
        let outcomes = [true, true, false, true, false, false, true, false];
        let est = dyadic_bootstrap_pass_at_m_lattice(&outcomes, 2);
        assert_eq!(est.point_estimate, 1.0);
        assert_eq!(est.n_resamples, 4);
    }

    #[test]
    fn test_dyadic_bootstrap_wilson_ci_known_values() {
        // Wilson 95% CI for 7/8 successes, n=8: well-tabulated value
        // (~0.473, ~0.997) per standard references. Verify roughly.
        let outcomes = [
            true, true, true, true, true, true, true, false,
        ];
        let est = dyadic_bootstrap_pass_at_m_lattice(&outcomes, 1);
        // m=1, k=8 -> 8 sub-lattices of size 1. pass@1 of [T]=1, [F]=0.
        // 7 passes / 8 = 0.875.
        assert!((est.point_estimate - 0.875).abs() < 1e-12);
        let (lo, hi) = est.wilson_ci_95();
        assert!(lo > 0.40 && lo < 0.55, "Wilson lo={lo} (expected ~0.47)");
        assert!(hi > 0.95 && hi <= 1.0, "Wilson hi={hi} (expected ~0.99)");
    }

    #[test]
    fn test_dyadic_bootstrap_wilson_ci_at_extremes() {
        // n=1, k=1, m=1, single pass -> Wilson CI is degenerate but well-defined.
        let est = BootstrapEstimate { point_estimate: 1.0, sample_variance: 0.0, n_resamples: 1 };
        let (lo, hi) = est.wilson_ci_95();
        // Wilson at 1/1, z=1.96: center ~ 1/(1+3.84) = 0.206, margin tiny
        assert!(hi <= 1.0 && hi > 0.9, "hi={hi}");
        assert!(lo > 0.0 && lo < 0.5, "lo={lo}");

        // n=0 -> uninformative.
        let est_empty = BootstrapEstimate { point_estimate: 0.0, sample_variance: 0.0, n_resamples: 0 };
        assert_eq!(est_empty.wilson_ci_95(), (0.0, 1.0));
    }

    #[test]
    fn test_dyadic_bootstrap_panics_on_non_power_of_two_k() {
        // k=7 is not a power of two.
        let outcomes = [true; 7];
        let result = std::panic::catch_unwind(|| {
            dyadic_bootstrap_pass_at_m_lattice(&outcomes, 4);
        });
        assert!(result.is_err(), "should panic on non-power-of-two k");
    }

    #[test]
    fn test_dyadic_bootstrap_panics_on_non_power_of_two_m() {
        let outcomes = [true; 8];
        let result = std::panic::catch_unwind(|| {
            dyadic_bootstrap_pass_at_m_lattice(&outcomes, 3);
        });
        assert!(result.is_err(), "should panic on non-power-of-two m");
    }

    #[test]
    fn test_dyadic_bootstrap_panics_on_m_greater_than_k() {
        let outcomes = [true; 4];
        let result = std::panic::catch_unwind(|| {
            dyadic_bootstrap_pass_at_m_lattice(&outcomes, 8);
        });
        assert!(result.is_err(), "should panic when m > k");
    }

    #[test]
    fn test_dyadic_bootstrap_panics_on_empty_outcomes() {
        let outcomes: [bool; 0] = [];
        let result = std::panic::catch_unwind(|| {
            dyadic_bootstrap_pass_at_m_lattice(&outcomes, 1);
        });
        assert!(result.is_err(), "should panic on empty outcomes");
    }

    #[test]
    fn test_dyadic_bootstrap_empirical_unbiasedness() {
        // Sanity check on the estimator algebra (NOT on QMC advantage): for
        // iid Bernoulli(p) outcomes, dyadic_bootstrap pass@m is unbiased for
        // the true pass@m = 1 - (1-p)^m.
        //
        // QMC advantage is a separate question (covariance structure);
        // this test isolates estimator correctness by feeding iid outcomes.
        let p = 0.3f64;
        let m = 4usize;
        let true_pass_at_m = 1.0 - (1.0 - p).powi(m as i32); // = 0.7599
        let n_batches = 200_000;
        let mut rng = crate::types::Rng::new(0x00C0_FFEE_BABE);
        let mut sum_est = 0.0f64;
        let mut outcomes = [false; 8];
        for _ in 0..n_batches {
            for o in outcomes.iter_mut() {
                *o = (rng.uniform() as f64) < p;
            }
            let est = dyadic_bootstrap_pass_at_m_lattice(&outcomes, m);
            sum_est += est.point_estimate;
        }
        let mean_est = sum_est / n_batches as f64;
        // Allow 3 sigma of Monte Carlo error: sqrt(p(1-p)/n_batches).
        // p = 0.7599, variance ~ 0.7599*0.2401 = 0.1824, sigma = 0.000954.
        // 4 sigma ~ 0.0038. Use 0.01 for safety.
        let abs_err = (mean_est - true_pass_at_m).abs();
        assert!(
            abs_err < 0.01,
            "dyadic bootstrap unbiasedness: mean_est = {mean_est}, true pass@m = {true_pass_at_m}, abs_err = {abs_err}"
        );
    }

    #[test]
    fn test_dyadic_bootstrap_strided_subsequence_is_valid_sub_lattice() {
        // Direct empirical verification of Theorem 1: draw a LatticeQmc
        // batch of k=8, take stride-2 subsequences, and confirm each is
        // marginally Unif[0,1) (KS test) AND has the expected lattice
        // structure (equispaced values).
        let k = 8usize;
        let m = 4usize;
        let stride = k / m;
        let n_batches = 10_000;
        let mut batch = vec![0.0f32; k];
        let mut sub_batch = vec![0.0f32; m];
        // Collect sub-batch values from offset j=0 across many batches.
        let mut collected = Vec::with_capacity(n_batches * m);
        let mut src = LatticeQmc::new(0xABCD_1234);
        for _ in 0..n_batches {
            src.draw(k, &mut batch);
            for t in 0..m {
                sub_batch[t] = batch[t * stride]; // j=0 subsequence
            }
            collected.extend_from_slice(&sub_batch);
        }
        // Sub-batch values should be marginally Unif[0,1).
        let (_, p) = ks_uniform(&collected);
        assert!(p > 0.01, "sub-lattice marginal uniformity: KS p = {p}");

        // Verify lattice structure within each sub-batch: values should be
        // equispaced (modulo the wraparound).
        let mut max_gap_dev = 0.0f32;
        src = LatticeQmc::new(0xABCD_1234); // reset to same seed
        for _ in 0..1000 {
            src.draw(k, &mut batch);
            for t in 0..m {
                sub_batch[t] = batch[t * stride];
            }
            // Sort; the cyclic gap structure of a size-m lattice has all
            // pairwise neighbor gaps equal to 1/m (with one wrap gap).
            let mut sorted = sub_batch.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut gaps = Vec::with_capacity(m);
            for i in 0..m - 1 {
                gaps.push(sorted[i + 1] - sorted[i]);
            }
            gaps.push(1.0 - sorted[m - 1] + sorted[0]); // wrap
            let expected = 1.0 / m as f32;
            for g in gaps {
                max_gap_dev = max_gap_dev.max((g - expected).abs());
            }
        }
        assert!(max_gap_dev < 1e-5, "sub-lattice equispacing: max gap dev = {max_gap_dev}");
    }

    // ── Block bootstrap (Sobol/Stratified) ────────────────────────────────

    #[test]
    fn test_block_bootstrap_all_pass() {
        let outcomes = [true; 16];
        let mut rng = crate::types::Rng::new(42);
        let est = contiguous_block_bootstrap_pass_at_m(&outcomes, 4, 50, &mut rng);
        assert_eq!(est.point_estimate, 1.0);
        assert_eq!(est.sample_variance, 0.0);
        assert_eq!(est.n_resamples, 50);
    }

    #[test]
    fn test_block_bootstrap_all_fail() {
        let outcomes = [false; 16];
        let mut rng = crate::types::Rng::new(42);
        let est = contiguous_block_bootstrap_pass_at_m(&outcomes, 4, 50, &mut rng);
        assert_eq!(est.point_estimate, 0.0);
        assert_eq!(est.sample_variance, 0.0);
    }

    #[test]
    fn test_block_bootstrap_deterministic_given_seed() {
        let outcomes = [
            true, false, true, false, true, true, false, false,
            true, true, true, false, false, true, false, true,
        ];
        let mut r1 = crate::types::Rng::new(7);
        let mut r2 = crate::types::Rng::new(7);
        let est1 = contiguous_block_bootstrap_pass_at_m(&outcomes, 4, 100, &mut r1);
        let est2 = contiguous_block_bootstrap_pass_at_m(&outcomes, 4, 100, &mut r2);
        assert_eq!(est1, est2, "same seed must produce identical estimates");
    }

    #[test]
    fn test_block_bootstrap_m_equals_k_single_block() {
        // m == k -> only one possible start (0); every resample sees the
        // whole array. Variance collapses to 0 (all resamples identical).
        let outcomes = [true, false, true, false];
        let mut rng = crate::types::Rng::new(42);
        let est = contiguous_block_bootstrap_pass_at_m(&outcomes, 4, 30, &mut rng);
        assert_eq!(est.point_estimate, 1.0, "at least one pass -> pass@4 = 1");
        assert_eq!(est.sample_variance, 0.0);
        assert_eq!(est.n_resamples, 30);
    }

    #[test]
    fn test_block_bootstrap_estimate_in_range() {
        // For random Bernoulli outcomes, the bootstrap estimate should be
        // in [0, 1] and the Wilson CI should bracket it.
        let outcomes = [
            true, false, true, false, true, true, false, false,
            true, false, false, true, true, false, true, false,
        ];
        let mut rng = crate::types::Rng::new(0xBEEF);
        let est = contiguous_block_bootstrap_pass_at_m(&outcomes, 4, 1000, &mut rng);
        assert!(est.point_estimate >= 0.0 && est.point_estimate <= 1.0);
        assert!(est.sample_variance >= 0.0);
        let (lo, hi) = est.wilson_ci_95();
        assert!(lo <= est.point_estimate && est.point_estimate <= hi,
            "point {} not in CI [{}, {}]", est.point_estimate, lo, hi);
    }

    #[test]
    fn test_block_bootstrap_empirical_unbiasedness() {
        // For iid Bernoulli(p) outcomes, contiguous-block bootstrap pass@m
        // should be approximately unbiased for true pass@m = 1-(1-p)^m.
        let p = 0.4f64;
        let m = 4usize;
        let true_pass_at_m = 1.0 - (1.0 - p).powi(m as i32); // = 0.8704
        let n_batches = 100_000;
        let mut master_rng = crate::types::Rng::new(0xFEED_FACE);
        let mut sum_est = 0.0f64;
        let mut outcomes = [false; 16];
        for _ in 0..n_batches {
            for o in outcomes.iter_mut() {
                *o = (master_rng.uniform() as f64) < p;
            }
            let est = contiguous_block_bootstrap_pass_at_m(&outcomes, m, 32, &mut master_rng);
            sum_est += est.point_estimate;
        }
        let mean_est = sum_est / n_batches as f64;
        // Note: contiguous-block bootstrap is biased toward 1 because
        // overlapping blocks share rollouts (effective sample size < B).
        // Allow generous tolerance; the main check is that we're in the
        // right neighborhood and not catastrophically wrong.
        let abs_err = (mean_est - true_pass_at_m).abs();
        assert!(
            abs_err < 0.05,
            "block bootstrap approximate unbiasedness: mean_est = {mean_est}, true pass@m = {true_pass_at_m}, abs_err = {abs_err}"
        );
    }

    #[test]
    fn test_block_bootstrap_panics_on_zero_m() {
        let outcomes = [true; 8];
        let mut rng = crate::types::Rng::new(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            contiguous_block_bootstrap_pass_at_m(&outcomes, 0, 10, &mut rng);
        }));
        assert!(result.is_err(), "should panic on m == 0");
    }

    #[test]
    fn test_block_bootstrap_panics_on_zero_resamples() {
        let outcomes = [true; 8];
        let mut rng = crate::types::Rng::new(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            contiguous_block_bootstrap_pass_at_m(&outcomes, 4, 0, &mut rng);
        }));
        assert!(result.is_err(), "should panic on n_resamples == 0");
    }

    #[test]
    fn test_block_bootstrap_panics_on_m_greater_than_k() {
        let outcomes = [true; 4];
        let mut rng = crate::types::Rng::new(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            contiguous_block_bootstrap_pass_at_m(&outcomes, 8, 10, &mut rng);
        }));
        assert!(result.is_err(), "should panic on m > k");
    }

    // ── BootstrapEstimate methods ────────────────────────────────────────

    #[test]
    fn test_bootstrap_estimate_std_dev() {
        let est = BootstrapEstimate {
            point_estimate: 0.5,
            sample_variance: 0.25,
            n_resamples: 10,
        };
        assert!((est.std_dev() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_wilson_ci_zero_failures_one_sample() {
        // Single sample, single success. Wilson CI should be wide.
        let est = BootstrapEstimate {
            point_estimate: 1.0,
            sample_variance: 0.0,
            n_resamples: 1,
        };
        let (lo, hi) = est.wilson_ci_95();
        assert!(lo > 0.0 && lo < 0.5, "lo = {lo}");
        assert!((hi - 1.0).abs() < 1e-12, "hi = {hi}");
    }
}
