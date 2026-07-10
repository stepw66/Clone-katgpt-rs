//! CGSP types — decoupled structs used by [`crate::cgsp`] (Plan 274).
//!
//! All types here are POD-friendly, fixed-size, and cloneable so they can be
//! reused across cycles without re-allocation. No game semantics.
//!
//! Per Plan 274 §Latent vs Raw boundary:
//! - `Direction` / `Target` / `Priority` are **latent** (local, not synced).
//! - `solve_rate` scalar is **raw** (may be used for anti-cheat / cold-tier).
//! - `collapse_triggered` bool is **raw** (event crossing sync boundary).

use blake3::Hasher;

// ── Constants ─────────────────────────────────────────────────────────────

/// Magic bytes identifying a `CuriosityPrioritySnapshot`.
pub(crate) const SNAPSHOT_MAGIC: [u8; 4] = *b"CGSP";

/// Current snapshot serialization version.
pub(crate) const SNAPSHOT_VERSION: u32 = 1;

/// Default maximum supported dimension for HLA direction vectors.
pub const DEFAULT_HLA_DIM: usize = 64;

/// Default maximum number of candidates sampled per cycle (k).
pub const DEFAULT_K: usize = 4;

/// Default maximum number of arms in the priority table.
pub const DEFAULT_POOL_SIZE: usize = 64;

// ── Latent types ──────────────────────────────────────────────────────────

/// A latent direction vector. POD-style fixed-size buffer.
///
/// Operates in latent space — never crosses the sync boundary. The
/// `dot-product + sigmoid` projection onto a `Target` yields a scalar
/// quality score (the only thing that may cross the boundary).
#[derive(Clone, Debug, PartialEq)]
pub struct Direction {
    /// Latent coordinates. Length is the HLA dimension.
    pub coords: Vec<f32>,
}

impl Direction {
    /// Create a zero direction with `dim` coordinates.
    pub fn zeros(dim: usize) -> Self {
        Self {
            coords: vec![0.0; dim],
        }
    }

    /// Create from an existing slice (clones into owned storage).
    pub fn from_slice(src: &[f32]) -> Self {
        Self {
            coords: src.to_vec(),
        }
    }

    /// Dimensionality of this direction.
    #[inline]
    pub fn dim(&self) -> usize {
        self.coords.len()
    }

    /// Dot product with another direction. Returns 0.0 if dims mismatch.
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        if self.coords.len() != other.coords.len() {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for (a, b) in self.coords.iter().zip(other.coords.iter()) {
            // FMA: sum = a * b + sum (single rounding, matches bridge::dot_f32_i8).
            sum = a.mul_add(*b, sum);
        }
        sum
    }

    /// L2 norm squared.
    #[inline]
    pub fn norm_sq(&self) -> f32 {
        let mut s = 0.0f32;
        for c in &self.coords {
            // FMA: s = c * c + s (single rounding).
            s = c.mul_add(*c, s);
        }
        s
    }

    /// Mutable coordinate access.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.coords
    }

    /// Read-only coordinate access.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.coords
    }
}

/// The unsolved target the loop is currently curious about.
///
/// A target carries a direction vector (latent, local) and a scalar
/// `priority_hint` used to seed initial sampling weights. Game-agnostic.
#[derive(Clone, Debug)]
pub struct Target {
    /// Direction the loop wants to make progress against.
    pub direction: Direction,
    /// Optional scalar weight hint in `[0, 1]`.
    pub priority_hint: f32,
}

impl Target {
    /// Build a target from a direction vector.
    pub fn new(direction: Direction) -> Self {
        Self {
            direction,
            priority_hint: 0.5,
        }
    }

    /// Override the priority hint.
    pub fn with_priority_hint(mut self, hint: f32) -> Self {
        self.priority_hint = hint.clamp(0.0, 1.0);
        self
    }

    /// Dimensionality (delegates to inner direction).
    #[inline]
    pub fn dim(&self) -> usize {
        self.direction.dim()
    }
}

/// A single candidate emitted by the conjecturer for the current cycle.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// The proposed direction vector.
    pub direction: Direction,
    /// Index into the conjecturer pool (`usize::MAX` for off-pool samples).
    pub pool_index: usize,
}

impl Candidate {
    /// Build a candidate from a pool slot.
    pub fn new(direction: Direction, pool_index: usize) -> Self {
        Self {
            direction,
            pool_index,
        }
    }
}

// ── Raw/scalar-crossable types ────────────────────────────────────────────

/// Solve-rate scalar in `[0, 1]` produced by the solver attempt.
///
/// This is the **raw** value that may cross the sync boundary. The latent
/// `Direction` it was produced from stays local.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolveRate(pub f32);

impl SolveRate {
    /// Build from a raw solve-rate, clamped to `[0, 1]`.
    #[inline]
    pub fn new(rate: f32) -> Self {
        Self(rate.clamp(0.0, 1.0))
    }

    /// Raw scalar.
    #[inline]
    pub fn value(self) -> f32 {
        self.0
    }
}

// ── Priority table ────────────────────────────────────────────────────────

/// Per-pool-slot priority weight in `[0, 1]`.
///
/// Priority is a latent, local quantity — never synced directly. Snapshots
/// (committed via BLAKE3) are what cross to the cold tier.
pub type Priority = f32;

// ── Hint policy ───────────────────────────────────────────────────────────

/// How much a [`Solver`](crate::cgsp::traits::Solver) benefits from injected
/// branching-order hints (the priority-weighted candidate ordering).
///
/// Distilled from G-RRM §3 (arXiv 2607.02491): overhead-dominated solvers
/// (cadical3, 0.896× mean slowdown) see a net regression from hints because
/// their fixed startup cost dominates the search savings; search-dominated
/// solvers (backtracking, glucose4 — up to 33.3× speedup) benefit greatly.
///
/// The `Skip` policy short-circuits the bandit absorb so an overhead-dominated
/// solver's noise-dominated solve-rates cannot corrupt the hint priority table.
///
/// # Default
///
/// `OrderOnly` — every shipped solver is search-dominated and hint-receptive.
/// This is a non-breaking default: the absorb runs exactly as before.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum HintPolicy {
    /// Hints only reorder the search — never injected as hard constraints.
    /// Correct for custom backtracking + DDTree speculation (all shipped solvers).
    #[default]
    OrderOnly,
    /// Hints seed phase initialization (warm-start), then fall back to
    /// order-only. Future hook for warm-start solvers.
    PhaseInit,
    /// Skip hint injection entirely — this solver is overhead-dominated
    /// (e.g. a future cadical/glucose binding with fixed startup cost).
    /// The loop suppresses the bandit absorb for this solver.
    Skip,
}

/// Aggregated cycle statistics (raw scalars).
#[derive(Clone, Copy, Debug, Default)]
pub struct CycleStats {
    /// Number of candidates the conjecturer emitted this cycle.
    pub candidates_sampled: u32,
    /// Number of candidates admitted by the difficulty filter.
    pub candidates_admitted: u32,
    /// Number of candidates the solver successfully solved.
    pub candidates_solved: u32,
    /// Mean guide score over admitted candidates.
    pub mean_guide_score: f32,
    /// Mean synthetic reward over admitted candidates.
    pub mean_r_synth: f32,
    /// Entropy of the priority table after this cycle's update.
    pub priority_entropy: f32,
}

// ── CycleResult ───────────────────────────────────────────────────────────

/// Outcome of one CGSP cycle.
///
/// Carries the **raw** cross-boundary scalars (`solve_rate`, collapse event).
/// The latent vectors stay inside `ScratchBuffers` and never leave the cycle.
#[derive(Clone, Copy, Debug, Default)]
pub struct CycleResult {
    /// True when the priority table entropy dropped below `τ_low` and
    /// exploration was injected.
    pub collapse_triggered: bool,
    /// True when the batch was flagged degenerate and the bandit update
    /// was skipped (`data_gate`-equivalent behaviour).
    pub batch_degenerate: bool,
    /// Aggregated raw statistics for sync / observability.
    pub stats: CycleStats,
}

// ── ScratchBuffers ────────────────────────────────────────────────────────

/// Pre-allocated scratch space reused across `cycle()` calls.
///
/// All hot-path writes go into this struct — no `Vec::new` in `cycle()`.
/// Sized for `k` candidates and `pool_size` priority slots.
#[derive(Clone, Debug)]
pub struct ScratchBuffers {
    /// Candidate directions emitted by the conjecturer (length = k).
    pub candidates: Vec<Candidate>,
    /// Guide score per candidate (length = k).
    pub guide_scores: Vec<f32>,
    /// Admit/reject decision per candidate (length = k).
    pub admitted: Vec<bool>,
    /// Per-candidate solver solve-rate (length = k).
    pub solve_rates: Vec<f32>,
    /// Per-candidate synthetic reward `(1 - solve_rate) * guide_score` (length = k).
    pub r_synth: Vec<f32>,
    /// CDF scratch used by the priority-weighted roulette sampler.
    pub cdf_scratch: Vec<f32>,
}

impl ScratchBuffers {
    /// Allocate scratch buffers for `k` candidates and `pool_size` priority slots.
    ///
    /// Capacity is reserved but no slots are materialised — call
    /// [`ensure_len`](Self::ensure_len) before the first `cycle()` (or let
    /// `CgspLoop::cycle` call it for you) to pre-fill the `candidates` Vec
    /// with reusable `Candidate` slots.
    pub fn new(k: usize, pool_size: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(k),
            guide_scores: Vec::with_capacity(k),
            admitted: Vec::with_capacity(k),
            solve_rates: Vec::with_capacity(k),
            r_synth: Vec::with_capacity(k),
            cdf_scratch: Vec::with_capacity(pool_size),
        }
    }

    /// Ensure the `candidates` Vec contains exactly `k` reusable slots, each
    /// carrying a `Direction` of dimension `dim`.
    ///
    /// This is the Option-B fix for issue 021 Site 1: instead of clearing
    /// and re-growing the Vec every cycle (which clones a `Vec<f32>` per
    /// slot), we materialise the slots **once** and let the conjecturer
    /// overwrite their `coords` in place via `clone_from` (zero allocation
    /// in steady state). On subsequent calls with the same `(k, dim)` this
    /// method is a no-op apart from a length check.
    ///
    /// If `k` grows between calls, only the newly-appended slots allocate.
    /// If `dim` changes, existing slots are resized in place (which may
    /// reallocate, but only on the dimension-change cycle, never in steady
    /// state).
    pub fn ensure_len(&mut self, k: usize, dim: usize) {
        // Shrink is unusual but keep the contract exact.
        if self.candidates.len() > k {
            self.candidates.truncate(k);
        }
        while self.candidates.len() < k {
            self.candidates
                .push(Candidate::new(Direction::zeros(dim), usize::MAX));
        }
        // Resize any existing slot whose dim doesn't match (e.g. caller
        // changed target dimensionality). `Vec::resize` on `f32` is
        // allocation-free when the existing capacity already covers `dim`,
        // which holds for every cycle after the first.
        for slot in self.candidates.iter_mut() {
            if slot.direction.coords.len() != dim {
                slot.direction.coords.resize(dim, 0.0);
            }
        }
        // Keep the scalar buffers at length k so direct index assignment in
        // `cycle()` is in bounds. `resize` on `f32` / `bool` is alloc-free
        // once capacity covers k (which it does after the first call).
        self.guide_scores.resize(k, 0.0);
        self.admitted.resize(k, false);
        self.solve_rates.resize(k, 0.0);
        self.r_synth.resize(k, 0.0);
    }

    /// Reset the per-cycle scalar buffers.
    ///
    /// As of issue 021 (Option B) this is a **no-op on `candidates`** — the
    /// reusable slots are owned by `ScratchBuffers` and overwritten in place
    /// by the conjecturer each cycle. Only the scalar working buffers are
    /// cleared (their capacity is preserved; `clear` is alloc-free).
    ///
    /// `cdf_scratch` is also cleared here; `PoolConjecturer::build_cdf` does
    /// its own `clear` + `push` pattern that is alloc-free in steady state.
    #[inline]
    pub fn reset(&mut self) {
        self.guide_scores.clear();
        self.admitted.clear();
        self.solve_rates.clear();
        self.r_synth.clear();
        self.cdf_scratch.clear();
    }
}

// ── Snapshot (Phase 2) ────────────────────────────────────────────────────

/// Atomic snapshot of a CGSP priority table + direction pool.
///
/// Used by [`crate::cgsp::loop_::CgspLoop::snapshot`] / `restore` for the
/// freeze/thaw cycle. Serialized form is fixed-size bytes; BLAKE3 commitment
/// provides tamper-evidence for the cold tier.
///
/// # Layout
///
/// ```text
/// [magic:4][version:4][dim:4][pool_size:4]
/// [directions: pool_size * dim * 4 bytes]
/// [priorities: pool_size * 4 bytes]
/// ```
#[derive(Clone, Debug)]
pub struct CuriosityPrioritySnapshot {
    /// Magic bytes (`CGSP`).
    pub magic: [u8; 4],
    /// Serialization version.
    pub version: u32,
    /// HLA dimension of every direction in the pool.
    pub dim: u32,
    /// Snapshot ordering key (Uuid v7 — time-ordered).
    pub snapshot_id: [u8; 16],
    /// Snapshot of the direction pool (latent, committed).
    pub directions: Vec<Direction>,
    /// Snapshot of the priority weights (latent, committed).
    pub priorities: Vec<f32>,
}

impl CuriosityPrioritySnapshot {
    /// Build a snapshot from a pool + priority slice.
    pub fn new(directions: Vec<Direction>, priorities: Vec<f32>) -> Self {
        debug_assert_eq!(
            directions.len(),
            priorities.len(),
            "cgsp: directions and priorities must have equal length"
        );
        let dim = directions.first().map(|d| d.dim() as u32).unwrap_or(0);
        Self {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
            dim,
            snapshot_id: snapshot_id_now(),
            directions,
            priorities,
        }
    }

    /// Number of slots in the pool.
    #[inline]
    pub fn pool_size(&self) -> usize {
        self.priorities.len()
    }

    /// Encode to a fixed-shape byte vector (no serde, no allocation in the
    /// hot path — caller pre-allocates the destination).
    ///
    /// Layout matches the doc above. Output length is deterministic given
    /// `dim` and `pool_size`.
    pub fn encode_to(&self, out: &mut Vec<u8>) {
        out.clear();
        let pool_size = self.pool_size() as u32;
        // Header: 16 bytes.
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.dim.to_le_bytes());
        out.extend_from_slice(&pool_size.to_le_bytes());
        out.extend_from_slice(&self.snapshot_id);
        // Directions: pool_size * dim * 4 bytes.
        for d in &self.directions {
            // Pad/truncate to declared dim so the layout is fixed-shape.
            let mut written = 0usize;
            for c in d.coords.iter().take(self.dim as usize) {
                out.extend_from_slice(&c.to_le_bytes());
                written += 1;
            }
            for _ in written..(self.dim as usize) {
                out.extend_from_slice(&0.0f32.to_le_bytes());
            }
        }
        // Priorities: pool_size * 4 bytes.
        for p in &self.priorities {
            out.extend_from_slice(&p.to_le_bytes());
        }
    }

    /// Decode from a byte slice previously produced by [`encode_to`](Self::encode_to).
    ///
    /// Returns `Err` with a human-readable reason on any mismatch.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 32 {
            return Err(format!(
                "cgsp snapshot: header too short ({} < 32)",
                bytes.len()
            ));
        }
        let magic = <[u8; 4]>::try_from(&bytes[0..4]).unwrap();
        if magic != SNAPSHOT_MAGIC {
            return Err(format!(
                "cgsp snapshot: bad magic {:?}, expected {:?}",
                magic, SNAPSHOT_MAGIC
            ));
        }
        let version = u32::from_le_bytes(<[u8; 4]>::try_from(&bytes[4..8]).unwrap());
        if version != SNAPSHOT_VERSION {
            return Err(format!(
                "cgsp snapshot: bad version {version}, expected {SNAPSHOT_VERSION}"
            ));
        }
        let dim = u32::from_le_bytes(<[u8; 4]>::try_from(&bytes[8..12]).unwrap());
        let pool_size = u32::from_le_bytes(<[u8; 4]>::try_from(&bytes[12..16]).unwrap());
        let snapshot_id = <[u8; 16]>::try_from(&bytes[16..32]).unwrap();

        let dim_us = dim as usize;
        let pool_us = pool_size as usize;
        let expected_len = 32 + pool_us * dim_us * 4 + pool_us * 4;
        if bytes.len() != expected_len {
            return Err(format!(
                "cgsp snapshot: length {} != expected {} (dim={}, pool={})",
                bytes.len(),
                expected_len,
                dim,
                pool_size,
            ));
        }

        let mut off = 32usize;
        let mut directions = Vec::with_capacity(pool_us);
        for _ in 0..pool_us {
            let mut coords = Vec::with_capacity(dim_us);
            for _ in 0..dim_us {
                let chunk =
                    <[u8; 4]>::try_from(&bytes[off..off + 4]).expect("cgsp: dir slice in range");
                coords.push(f32::from_le_bytes(chunk));
                off += 4;
            }
            directions.push(Direction { coords });
        }

        let mut priorities = Vec::with_capacity(pool_us);
        for _ in 0..pool_us {
            let chunk =
                <[u8; 4]>::try_from(&bytes[off..off + 4]).expect("cgsp: prio slice in range");
            priorities.push(f32::from_le_bytes(chunk));
            off += 4;
        }

        Ok(Self {
            magic,
            version,
            dim,
            snapshot_id,
            directions,
            priorities,
        })
    }

    /// Compute the BLAKE3 commitment over the encoded snapshot.
    ///
    /// Deterministic — the same snapshot always produces the same hash.
    /// The hash is the **raw** commitment that may cross to the cold tier.
    pub fn blake3_hash(&self) -> [u8; 32] {
        let mut buf = Vec::with_capacity(32 + self.pool_size() * (self.dim as usize + 1) * 4);
        self.encode_to(&mut buf);
        let mut hasher = Hasher::new();
        hasher.update(&buf);
        *hasher.finalize().as_bytes()
    }
}

/// Produce a time-ordered 16-byte snapshot id (Uuid v7 layout).
///
/// Uses the high 48 bits for unix epoch millis and the low bits for a
/// fastrand counter — sufficient for ordering within a process without
/// pulling in the `uuid` crate as a hard dependency here.
fn snapshot_id_now() -> [u8; 16] {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut id = [0u8; 16];
    // bytes 0..8 — timestamp with version nibble
    id[0] = ((now_ms >> 40) & 0xFF) as u8;
    id[1] = ((now_ms >> 32) & 0xFF) as u8;
    id[2] = ((now_ms >> 24) & 0xFF) as u8;
    id[3] = ((now_ms >> 16) & 0xFF) as u8;
    id[4] = ((now_ms >> 8) & 0xFF) as u8;
    id[5] = (now_ms & 0xFF) as u8;
    // version 7 (0x7 in high nibble of byte 6)
    let rand_hi = fastrand::u8(..0x10) | 0x70;
    id[6] = rand_hi;
    id[7] = fastrand::u8(..);
    // variant (0b10xx_xxxx) in high bits of byte 8
    id[8] = (fastrand::u8(..0x40)) | 0x80;
    for b in id.iter_mut().take(16).skip(9) {
        *b = fastrand::u8(..);
    }
    id
}

// ── Math helpers (sigmoid never softmax) ──────────────────────────────────

/// Numerically stable sigmoid.
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Shannon entropy (nats) of a probability-like vector.
///
/// Used for collapse detection: low entropy = degenerate priority table.
pub fn entropy_nats(weights: &[f32]) -> f32 {
    let total: f32 = weights.iter().copied().sum();
    if total <= 0.0 || weights.is_empty() {
        return 0.0;
    }
    let mut h = 0.0f32;
    for &w in weights {
        if w > 0.0 {
            let p = w / total;
            h -= p * p.ln();
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_basics() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(1e6) > 0.999);
        assert!(sigmoid(-1e6) < 1e-3);
        // Bounds
        assert!(sigmoid(0.0) >= 0.0 && sigmoid(0.0) <= 1.0);
    }

    #[test]
    fn entropy_uniform_is_max() {
        let n = 8.0f32;
        let uniform = vec![1.0; n as usize];
        let h = entropy_nats(&uniform);
        let expected = n.ln();
        assert!((h - expected).abs() < 1e-5, "got {h}, expected {expected}");
    }

    #[test]
    fn entropy_degenerate_is_zero() {
        let degenerate = vec![1.0f32, 0.0, 0.0, 0.0];
        let h = entropy_nats(&degenerate);
        assert!(h.abs() < 1e-5, "expected 0, got {h}");
    }

    #[test]
    fn direction_dot() {
        let a = Direction::from_slice(&[1.0, 2.0, 3.0]);
        let b = Direction::from_slice(&[4.0, -5.0, 6.0]);
        assert!((a.dot(&b) - (4.0 - 10.0 + 18.0)).abs() < 1e-6);
        // Mismatched dims -> 0.0
        let c = Direction::from_slice(&[1.0, 2.0]);
        assert_eq!(a.dot(&c), 0.0);
    }

    #[test]
    fn snapshot_roundtrip() {
        let dirs = vec![
            Direction::from_slice(&[1.0, 0.0, 0.0]),
            Direction::from_slice(&[0.0, 1.0, 0.0]),
        ];
        let prios = vec![0.7, 0.3];
        let snap = CuriosityPrioritySnapshot::new(dirs, prios);
        let mut buf = Vec::new();
        snap.encode_to(&mut buf);
        let back = CuriosityPrioritySnapshot::decode(&buf).expect("decode");
        assert_eq!(back.version, snap.version);
        assert_eq!(back.dim, snap.dim);
        assert_eq!(back.priorities, snap.priorities);
        assert_eq!(back.directions.len(), snap.directions.len());
        for (a, b) in back.directions.iter().zip(snap.directions.iter()) {
            assert_eq!(a.coords, b.coords);
        }
    }

    #[test]
    fn snapshot_blake3_deterministic() {
        let dirs = vec![
            Direction::from_slice(&[1.0, 0.0]),
            Direction::from_slice(&[0.0, 1.0]),
        ];
        let prios = vec![0.5, 0.5];
        let s1 = CuriosityPrioritySnapshot::new(dirs.clone(), prios.clone());
        // Force identical snapshot_id to compare deterministically.
        let mut s2 = CuriosityPrioritySnapshot::new(dirs, prios);
        s2.snapshot_id = s1.snapshot_id;
        let h1 = s1.blake3_hash();
        let h2 = s2.blake3_hash();
        assert_eq!(h1, h2, "BLAKE3 must be deterministic for identical content");
    }

    #[test]
    fn snapshot_rejects_bad_magic() {
        let bogus = b"XXXX";
        let mut buf = bogus.to_vec();
        buf.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 16]);
        let err = CuriosityPrioritySnapshot::decode(&buf).unwrap_err();
        assert!(err.contains("magic"), "err = {err}");
    }
}
