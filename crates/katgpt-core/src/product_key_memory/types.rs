//! Types for the PKM retrieval primitive.
//!
//! Plan 408 Phase 1. Defines the const-generic table layout, scoring modes,
//! and fixed-size query result type. All hot-path buffers are sized by the
//! const generics; `query_into` (Phase 2) takes pre-allocated scratch.
//!
//! # Layout
//!
//! The three tables are flat `Box<[f32]>` slices (row-major), NOT nested
//! arrays — const-generic arithmetic like `[[f32; D_K/2]; SQRT_N]` requires
//! the unstable `generic_const_exprs` feature, which this crate does not use.
//! Row-major indexing is computed at runtime from the const generics:
//!
//! | Table | Logical shape | Flat length | Row `(i, ..)` start |
//! |---|---|---|---|---|
//! | `keys_1` | `SQRT_N × (D_K/2)` | `SQRT_N * D_K_HALF` | `i * D_K_HALF` |
//! | `keys_2` | `SQRT_N × (D_K/2)` | `SQRT_N * D_K_HALF` | `i * D_K_HALF` |
//! | `values` | `(SQRT_N*SQRT_N) × D_V` | `SQRT_N * SQRT_N * D_V` | `flat_idx * D_V` |
//!
//! where `D_K_HALF = D_K / 2`. This mirrors the Engram `InMemoryEngramTable`
//! pattern (flat `Box<[f32]>`, direct index by `hash mod N`).
//!
//! # Const generic floors
//!
//! `SQRT_N >= 2` (need at least a 2×2 = 4-slot table), `D_K >= 2` and even
//! (need two non-empty halves for the split). Constructors enforce these at
//! runtime — invalid monomorphizations panic on first construct, not at the
//! type level (the price of stable-Rust compatibility).

// ─── Const-generic floors ──────────────────────────────────────────────
//
// `SQRT_N >= 2`: a 2×2 = 4-slot table is the smallest meaningful PKM (smaller
// is just a flat array, no factorization benefit). `D_K >= 2` and even: needs
// two non-empty halves for the split. These floors are exported so callers can
// pattern-match on them in docs / assertion messages.

/// Minimum sub-key codebook size (`SQRT_N`). Enforced by constructors.
pub const SQRT_N_FLOOR: usize = 2;

/// Minimum key dimension (`D_K`). Must be even (split into two halves).
/// Enforced by constructors.
pub const D_K_FLOOR: usize = 2;

// ─── Scoring mode ──────────────────────────────────────────────────────

/// How a half-query is scored against a sub-key codebook row.
///
/// The PKM factorization is scoring-function-agnostic. The paper §2.2 uses
/// dot product by default; §A.2 introduces the inverse-distance weighting
/// (IDW) variant `−log(ε + ‖q − K‖²)` which encourages keys to behave as
/// cluster centroids (they cannot inflate score by growing magnitude).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ScoreFn {
    /// Dot-product scoring (paper §2.2 default).
    ///
    /// `score(q, k) = q · k`. Linear, SIMD-friendly, magnitude-sensitive.
    #[default]
    Dot,

    /// Inverse-distance-weighted scoring (paper §A.2).
    ///
    /// `score(q, k) = −log(ε + ‖q − k‖²)`. Encourages centroid-like keys —
    /// keys cannot inflate score by growing magnitude (the log bounds the
    /// best achievable score at `−log ε`).
    ///
    /// `epsilon` MUST be > 0; the constructors clamp it to a tiny floor to
    /// avoid `log(0)` blowups.
    Idw {
        /// Additive epsilon under the log (must be > 0). Default 1e-6.
        epsilon: f32,
    },
}

impl ScoreFn {
    /// IDW mode with a sane default epsilon (1e-6).
    pub const fn idw_default() -> Self {
        ScoreFn::Idw { epsilon: 1e-6 }
    }

    /// IDW mode with caller-supplied epsilon, clamped to a tiny positive floor.
    pub fn idw_with_epsilon(epsilon: f32) -> Self {
        const EPS_FLOOR: f32 = 1e-12;
        let epsilon = if !epsilon.is_finite() || epsilon <= EPS_FLOOR {
            EPS_FLOOR
        } else {
            epsilon
        };
        ScoreFn::Idw { epsilon }
    }
}

// ─── Query result type ─────────────────────────────────────────────────

/// A single (flat_index, weight) entry in a PKM top-k query result.
///
/// `flat_index = i * SQRT_N + j` over the value table — caller maps it to the
/// value row via `ProductKeyMemory::value(flat_index)`.
///
/// `weight` is the post-normalization mixing coefficient (sigmoid-gate or
/// softmax-over-k² depending on the kernel path). Always in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PkEntry {
    pub flat_index: usize,
    pub weight: f32,
}

/// Fixed-size top-k query result, zero-allocation by construction.
///
/// Compile-time `K` (the const generic) gives a stack-sized array. The kernel
/// writes into the first `n_valid` slots; trailing slots are zeroed and have
/// `flat_index = usize::MAX` (sentinel for "unfilled"). Callers should iterate
/// `result.entries()[..result.n_valid()]`.
///
/// `K` MUST be `<= SQRT_N`. The Cartesian product of the two per-codebook
/// top-k lists is `K × K = K²` candidates, then reduced back to the top `K`.
/// Plan 408 T1.5 specifies this exact fixed-size shape.
#[derive(Debug, Clone, Copy)]
pub struct PkQuery<const K: usize> {
    /// The K best `(flat_index, weight)` pairs, in score-descending order
    /// within `..n_valid`. Trailing slots are `PkEntry { flat_index: usize::MAX, weight: 0.0 }`.
    entries: [PkEntry; K],
    /// Number of slots in `entries` that hold a real result. `<= K`.
    n_valid: usize,
}

impl<const K: usize> PkQuery<K> {
    /// Construct an empty (all-sentinel) result. The kernel fills + bumps
    /// `n_valid` as it commits each entry.
    pub const fn empty() -> Self {
        Self {
            entries: [PkEntry {
                flat_index: usize::MAX,
                weight: 0.0,
            }; K],
            n_valid: 0,
        }
    }

    /// Returns the filled prefix of `entries`. Trailing sentinel slots are
    /// excluded. Callers MUST check this slice's length, not assume it equals
    /// `K` (the table may have fewer than `K` valid slots in degenerate cases).
    #[inline]
    pub fn entries(&self) -> &[PkEntry] {
        // SAFETY: `n_valid <= K` is a kernel contract; the slice is in-bounds.
        &self.entries[..self.n_valid.min(K)]
    }

    /// Number of valid entries in this result (always `<= K`).
    #[inline]
    pub const fn n_valid(&self) -> usize {
        self.n_valid
    }
}

// SAFETY: PkEntry is plain-old-data (usize + f32). PkQuery<K> is a fixed-size
// array of PkEntry. Both are Send+Sync unconditionally.
unsafe impl<const K: usize> Send for PkQuery<K> {}
unsafe impl<const K: usize> Sync for PkQuery<K> {}

// ─── The PKM table itself ──────────────────────────────────────────────

/// Product Key Memory — `O(√N)` factored retrieval table.
///
/// Const generics:
/// - `SQRT_N` — sub-key codebook size; the value table has `SQRT_N * SQRT_N`
///   rows (the "N" in the O(√N) claim is `SQRT_N * SQRT_N`).
/// - `D_K` — full key dimension; split into two `D_K / 2`-dim halves for the
///   two codebooks. MUST be even and `>= D_K_FLOOR` (2).
/// - `D_V` — value dimension (the per-slot latent pattern width).
///
/// # Memory layout
///
/// The three tables are flat `Box<[f32]>` slices (row-major) — see the module
/// docs for why nested const-generic arrays aren't used. At the target scale
/// (`SQRT_N = 1000`, `D_K = 64`, `D_V = 128`) the table is:
/// - `keys_1` / `keys_2`: `1000 * 32 * 4 = 128 KB` each (L2-resident).
/// - `values`: `1_000_000 * 128 * 4 = 512 MB` (Cold-tier-sized).
///
/// The two codebooks are the hot path; the value table is randomly indexed by
/// the resolved top-k flat indices (cold reads, ~K rows per query).
///
/// # Frozen-between-swaps
///
/// The three tables are immutable once constructed. Updates go through a
/// freeze/thaw wrapper (Plan 408 Phase 4) that swaps an entire
/// `Arc<ProductKeyMemory>` atomically. The value table has no per-cell write
/// method; Phase 5's `product_key_memory_episodic` feature adds a δ-rule write
/// path over a private mutable copy, NOT over this struct directly.
///
/// # Construct
///
/// See [`ProductKeyMemory::new`], [`ProductKeyMemory::from_random`],
/// [`ProductKeyMemory::from_centroids`].
#[derive(Debug)]
pub struct ProductKeyMemory<const SQRT_N: usize, const D_K: usize, const D_V: usize> {
    /// Sub-key codebook 1, scored against `q[..D_K/2]`. Flat row-major,
    /// `SQRT_N * (D_K/2)` floats. Row `i` starts at `i * (D_K/2)`.
    pub keys_1: Box<[f32]>,
    /// Sub-key codebook 2, scored against `q[D_K/2..]`. Flat row-major,
    /// `SQRT_N * (D_K/2)` floats. Row `i` starts at `i * (D_K/2)`.
    pub keys_2: Box<[f32]>,
    /// Value table, flat row-major `(SQRT_N*SQRT_N) * D_V` floats.
    /// Row `flat_idx` starts at `flat_idx * D_V`.
    pub values: Box<[f32]>,
}

impl<const SQRT_N: usize, const D_K: usize, const D_V: usize> ProductKeyMemory<SQRT_N, D_K, D_V> {
    /// Half key dimension (`D_K / 2`). Runtime constant — the const-generic
    /// `D_K / 2` cannot be used as an array-size on stable Rust, so it's
    /// computed here and used for row indexing.
    #[inline]
    pub const fn key_half_dim() -> usize {
        D_K / 2
    }

    /// Runtime floor check on the const generics. Mirrors `SQRT_N_FLOOR` /
    /// `D_K_FLOOR`. Called from every constructor; panics on invalid
    /// monomorphization (caller bug — the const generics are wrong).
    fn assert_dims() {
        assert!(
            D_K >= D_K_FLOOR && D_K.is_multiple_of(2),
            "ProductKeyMemory: D_K must be even and >= {}, got {}",
            D_K_FLOOR,
            D_K
        );
        assert!(
            SQRT_N >= SQRT_N_FLOOR,
            "ProductKeyMemory: SQRT_N must be >= {}, got {}",
            SQRT_N_FLOOR,
            SQRT_N
        );
    }

    /// Construct from caller-owned flat codebook + value slices.
    ///
    /// The arguments are moved in (the table owns them). The slices MUST be
    /// the correct length (`SQRT_N * D_K/2` for each codebook,
    /// `SQRT_N * SQRT_N * D_V` for values); debug_asserted. Callers building
    /// a table from scratch should use [`ProductKeyMemory::from_random`] or
    /// [`ProductKeyMemory::from_centroids`].
    ///
    /// # Panics
    ///
    /// Debug builds assert the const-generic floors (`SQRT_N >= 2`, `D_K >= 2`
    /// and even) and the slice lengths.
    pub fn new(keys_1: Box<[f32]>, keys_2: Box<[f32]>, values: Box<[f32]>) -> Self {
        Self::assert_dims();
        let half = Self::key_half_dim();
        debug_assert_eq!(keys_1.len(), SQRT_N * half, "keys_1 wrong length");
        debug_assert_eq!(keys_2.len(), SQRT_N * half, "keys_2 wrong length");
        debug_assert_eq!(values.len(), SQRT_N * SQRT_N * D_V, "values wrong length");
        Self {
            keys_1,
            keys_2,
            values,
        }
    }

    /// Construct from a seed — random uniform keys in `[-1, 1]` and random
    /// uniform values in `[-1, 1]`. Used by tests + benchmarks; NOT a
    /// meaningful initialization for retrieval quality (the IDW-scoring
    /// centroid mode expects [`ProductKeyMemory::from_centroids`]).
    ///
    /// `seed` drives a deterministic `SplitMix64`-style PRNG so the same seed
    /// yields the same table bit-identically across runs (G6 determinism gate).
    pub fn from_random(seed: u64) -> Self {
        Self::assert_dims();
        // Tiny deterministic PRNG — splitmix64. Same seed → same table.
        let mut rng = SeededRng::new(seed);
        let half = Self::key_half_dim();

        let mut keys_1 = vec![0.0f32; SQRT_N * half].into_boxed_slice();
        let mut keys_2 = vec![0.0f32; SQRT_N * half].into_boxed_slice();
        for v in keys_1.iter_mut() {
            *v = rng.next_f32_in_range(-1.0, 1.0);
        }
        for v in keys_2.iter_mut() {
            *v = rng.next_f32_in_range(-1.0, 1.0);
        }
        let mut values = vec![0.0f32; SQRT_N * SQRT_N * D_V].into_boxed_slice();
        for v in values.iter_mut() {
            *v = rng.next_f32_in_range(-1.0, 1.0);
        }
        Self {
            keys_1,
            keys_2,
            values,
        }
    }

    /// Construct from caller-supplied cluster centroids (k-means or any other
    /// deterministic clustering output).
    ///
    /// `centroids_1` and `centroids_2` are the per-codebook centroids (flat
    /// row-major `SQRT_N * (D_K/2)` floats each); `values` is the value table
    /// (flat row-major `SQRT_N * SQRT_N * D_V` floats; caller initializes —
    /// typically random or zeros for an empty table).
    ///
    /// This is the **modelless initialization path** for IDW scoring: the
    /// paper's `L_addr` GD step would *learn* centroids to maximize entropy,
    /// which is forbidden. We instead take caller-supplied centroids — the
    /// caller is responsible for producing them via a deterministic process
    /// (TEMP `sleep_diverse` Plan 005, k-means on a frozen corpus, etc.).
    ///
    /// The centroids are NOT trained at runtime — they are a frozen snapshot.
    /// Future centroid refresh goes through the Phase 4 freeze/thaw wrapper.
    pub fn from_centroids(
        centroids_1: Box<[f32]>,
        centroids_2: Box<[f32]>,
        values: Box<[f32]>,
    ) -> Self {
        // `from_centroids` is the same as `new` semantically; the distinct
        // name signals intent (IDW-mode initialization) and keeps the audit
        // trail clear in callers.
        Self::new(centroids_1, centroids_2, values)
    }

    /// Total slot count of the value table (`SQRT_N * SQRT_N`).
    #[inline]
    pub const fn num_slots() -> usize {
        SQRT_N * SQRT_N
    }

    /// Key dimension (the full `D_K`, before the split).
    #[inline]
    pub const fn key_dim() -> usize {
        D_K
    }

    /// Value dimension (`D_V`).
    #[inline]
    pub const fn value_dim() -> usize {
        D_V
    }

    /// Borrow a codebook-1 row as a `&[f32]` of length `D_K/2`.
    ///
    /// Panics if `i >= SQRT_N` (caller bug — `i` comes from a top-k index
    /// over the codebook, always `< SQRT_N`).
    #[inline]
    pub fn keys_1_row(&self, i: usize) -> &[f32] {
        let half = Self::key_half_dim();
        &self.keys_1[i * half..(i + 1) * half]
    }

    /// Borrow a codebook-2 row as a `&[f32]` of length `D_K/2`.
    #[inline]
    pub fn keys_2_row(&self, j: usize) -> &[f32] {
        let half = Self::key_half_dim();
        &self.keys_2[j * half..(j + 1) * half]
    }

    /// Borrow the value row at `flat_index` as a `&[f32]` of length `D_V`.
    ///
    /// Panics if `flat_index >= num_slots()` — callers must resolve
    /// `flat_index` only from the top-k kernel output, which is always
    /// `< num_slots()`.
    ///
    /// This is a cold read (top-k fetches ~K rows per query); the value table
    /// is randomly indexed by the resolved flat indices and lives outside the
    /// L2-resident codebook hot path.
    #[inline]
    pub fn value(&self, flat_index: usize) -> &[f32] {
        &self.values[flat_index * D_V..(flat_index + 1) * D_V]
    }

    /// Borrow the entire value table as a flat `&[f32]` (for hashing /
    /// commitment in Phase 4). Row-major `SQRT_N * SQRT_N * D_V` floats.
    pub fn values_flat(&self) -> &[f32] {
        &self.values
    }

    /// Borrow the entire codebook 1 as a flat `&[f32]`.
    pub fn keys_1_flat(&self) -> &[f32] {
        &self.keys_1
    }

    /// Borrow the entire codebook 2 as a flat `&[f32]`.
    pub fn keys_2_flat(&self) -> &[f32] {
        &self.keys_2
    }

    /// Mutably borrow the value row at `flat_index` as a `&mut [f32]` of length
    /// `D_V`.
    ///
    /// Used by the δ-rule write path (Phase 5, `PkmEpisodicStore::write`).
    /// Panics if `flat_index >= num_slots()` — callers must resolve
    /// `flat_index` only from the top-k kernel output, which is always
    /// `< num_slots()`.
    ///
    /// This is the mutable counterpart of [`value`](Self::value). The
    /// codebook rows (`keys_1`, `keys_2`) are intentionally NOT exposed as
    /// `&mut` — the modelless mandate (Plan 408 constraint #1) forbids
    /// gradient descent on keys; key refresh goes through the freeze/thaw
    /// wrapper (Phase 4), not through in-place mutation.
    #[inline]
    pub fn value_mut(&mut self, flat_index: usize) -> &mut [f32] {
        &mut self.values[flat_index * D_V..(flat_index + 1) * D_V]
    }
}

/// Deep-clone all three flat slices.
///
/// Intentionally a manual impl (not `#[derive(Clone)]`) to surface the cost
/// at call sites: cloning a production-scale table is `O(SQRT_N² × D_V)` —
/// 16 MB for `SQRT_N=1000, D_V=4`, 512 MB for `D_V=128`. The primary consumer
/// is [`crate::product_key_memory::PkmEpisodicStore::publish`], which clones
/// the working table into the freeze slot at sleep-cycle cadence (seconds-
/// scale, not per-tick). The Phase 4 bit-identity tests deliberately avoided
/// `Clone` (using `clone_table_slices` instead) to keep the commitment hash
/// path allocation-free; Phase 5 lifts that constraint for the publish path.
impl<const SQRT_N: usize, const D_K: usize, const D_V: usize> Clone
    for ProductKeyMemory<SQRT_N, D_K, D_V>
{
    fn clone(&self) -> Self {
        Self {
            keys_1: self.keys_1.clone(),
            keys_2: self.keys_2.clone(),
            values: self.values.clone(),
        }
    }
}

// ─── Tiny deterministic PRNG (splitmix64) ──────────────────────────────
//
// `from_random` needs a deterministic RNG so the same seed yields the same
// table bit-identically (G6 determinism gate). We ship our own splitmix64
// rather than depending on the `rand` crate — PKM is leaf-clean (constraint
// #5 of Plan 408: "no default deps — leaf-clean per tier-0 substrate rule").

/// Deterministic `splitmix64` PRNG. NOT cryptographically secure — used only
/// for test/bench initialization. Bit-identical across runs for the same seed.
struct SeededRng {
    state: u64,
}

impl SeededRng {
    const fn new(seed: u64) -> Self {
        Self {
            // Mix the seed once so `seed == 0` isn't a degenerate path.
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    /// Next 64-bit unsigned. (splitmix64 — see Steele et al. 2014, "Fast
    /// splittable pseudorandom number generators".)
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next `f32` in `[lo, hi)`. Uniform — fine for test/bench init; not for
    /// any production retrieval-quality claim.
    fn next_f32_in_range(&mut self, lo: f32, hi: f32) -> f32 {
        // Map the high 24 bits of next_u64 to [0, 1) — matches the standard
        // `(u >> 40) as f32 / (1u64 << 24) as f32` pattern.
        let u = (self.next_u64() >> 40) as u32; // 24 bits
        let unit = u as f32 / ((1u32 << 24) as f32);
        lo + unit * (hi - lo)
    }
}

// ─── Phase 1 unit tests (type invariants, no kernel yet) ───────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_floor_enforced_at_compile_time() {
        // Smallest legal table: 2×2 = 4 slots, D_K = 2, D_V = 1. The construct
        // itself must succeed (it calls assert_dims); the associated-function
        // checks below verify the dimensions.
        let _table: ProductKeyMemory<2, 2, 1> = ProductKeyMemory::from_random(42);
        assert_eq!(ProductKeyMemory::<2, 2, 1>::num_slots(), 4);
        assert_eq!(ProductKeyMemory::<2, 2, 1>::key_dim(), 2);
        assert_eq!(ProductKeyMemory::<2, 2, 1>::key_half_dim(), 1);
        assert_eq!(ProductKeyMemory::<2, 2, 1>::value_dim(), 1);
    }

    #[test]
    fn from_random_is_deterministic_same_seed() {
        // G6 determinism gate — same seed must produce bit-identical tables.
        let a: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(123);
        let b: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(123);
        assert_eq!(a.keys_1.as_ref(), b.keys_1.as_ref());
        assert_eq!(a.keys_2.as_ref(), b.keys_2.as_ref());
        assert_eq!(a.values.as_ref(), b.values.as_ref());
    }

    #[test]
    fn from_random_different_seeds_differ() {
        // Sanity — different seeds produce (almost certainly) different tables.
        let a: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(1);
        let b: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(2);
        assert_ne!(a.values.as_ref(), b.values.as_ref());
    }

    #[test]
    fn from_random_values_in_range() {
        // Sanity — `next_f32_in_range(-1, 1)` must stay in `[-1, 1)`.
        let table: ProductKeyMemory<32, 16, 8> = ProductKeyMemory::from_random(7);
        for v in table.values.iter() {
            assert!(*v >= -1.0 && *v < 1.0, "value {v} out of [-1, 1)");
        }
        for v in table.keys_1.iter() {
            assert!(*v >= -1.0 && *v < 1.0, "key1 {v} out of [-1, 1)");
        }
        for v in table.keys_2.iter() {
            assert!(*v >= -1.0 && *v < 1.0, "key2 {v} out of [-1, 1)");
        }
    }

    #[test]
    fn from_random_slice_lengths_match_const_generics() {
        // The flat slices must be sized per the const generics:
        //   keys_1.len() == SQRT_N * (D_K/2)
        //   keys_2.len() == SQRT_N * (D_K/2)
        //   values.len() == SQRT_N * SQRT_N * D_V
        let table: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(5);
        assert_eq!(table.keys_1.len(), 16 * 4);
        assert_eq!(table.keys_2.len(), 16 * 4);
        assert_eq!(table.values.len(), 16 * 16 * 4);
    }

    #[test]
    fn row_accessors_return_correct_lengths() {
        let table: ProductKeyMemory<8, 6, 3> = ProductKeyMemory::from_random(11);
        // codebook rows are D_K/2 = 3 wide.
        assert_eq!(table.keys_1_row(0).len(), 3);
        assert_eq!(table.keys_1_row(7).len(), 3);
        assert_eq!(table.keys_2_row(0).len(), 3);
        assert_eq!(table.keys_2_row(7).len(), 3);
        // value rows are D_V = 3 wide; flat_index in [0, 64).
        assert_eq!(table.value(0).len(), 3);
        assert_eq!(table.value(63).len(), 3);
    }

    #[test]
    fn row_accessors_are_contiguous_views() {
        // `value(flat_idx)` and `values_flat()` must overlap exactly.
        let table: ProductKeyMemory<4, 4, 2> = ProductKeyMemory::from_random(99);
        let flat = table.values_flat();
        assert_eq!(flat.len(), 4 * 4 * 2);
        // Row 5 in the flat view == value(5).
        let row5 = table.value(5);
        assert_eq!(row5[0], flat[5 * 2]);
        assert_eq!(row5[1], flat[5 * 2 + 1]);
    }

    #[test]
    fn pkquery_empty_is_all_sentinel() {
        let q: PkQuery<8> = PkQuery::empty();
        assert_eq!(q.n_valid(), 0);
        assert!(q.entries().is_empty());
    }

    #[test]
    fn scorefn_default_is_dot() {
        assert_eq!(ScoreFn::default(), ScoreFn::Dot);
    }

    #[test]
    fn scorefn_idw_default_epsilon() {
        match ScoreFn::idw_default() {
            ScoreFn::Idw { epsilon } => assert_eq!(epsilon, 1e-6),
            _ => panic!("expected Idw"),
        }
    }

    #[test]
    fn scorefn_idw_clamps_bad_epsilon() {
        // NaN / negative / zero all clamp to a tiny positive floor — never
        // `log(0)` or `log(negative)`.
        for bad in [f32::NAN, -1.0, 0.0, f32::NEG_INFINITY] {
            match ScoreFn::idw_with_epsilon(bad) {
                ScoreFn::Idw { epsilon } => {
                    assert!(epsilon > 0.0 && epsilon.is_finite());
                }
                _ => panic!("expected Idw"),
            }
        }
    }
}
