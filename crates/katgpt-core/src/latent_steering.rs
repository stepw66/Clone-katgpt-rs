//! Latent Field Steering — top-down direction-vector injection into latent state.
//!
//! See `katgpt-rs/.research/290_latent_field_steering_open_primitive.md` and
//! Plan 309. This is the **missing top-down control direction** complement to
//! the existing bottom-up emotion computation (`EmotionDirections::project`,
//! Plan 162). Instead of reading affect from latent state, we **inject** a
//! designer/environment-supplied direction vector directly into mutable
//! per-tick latent state — the "wave interference" mechanism: linear
//! superposition of the NPC's current field with an injected steering field.
//!
//! ## Math
//!
//! Given NPC latent state `s ∈ R^d` (d=8 for HLA) and a unit-norm steering
//! direction `v ∈ R^d` with strength `α ∈ [0, 1]`:
//!
//! ```text
//! s' = s + α · v
//! ```
//!
//! For localized fields (only NPCs within a support region R), the effective
//! strength is modulated by a sigmoid-falloff kernel:
//!
//! ```text
//! s'_i = s_i + α · kernel(distance(i, center), bandwidth) · v
//! ```
//!
//! The kernel is `sigmoid((bandwidth - distance) · steepness)` — ~1 inside
//! the bandwidth, ~0 outside, smooth at the boundary. Per AGENTS.md: **sigmoid,
//! never softmax/Gaussian** for projections.
//!
//! ## Why modelless
//!
//! Direction vectors are **frozen, BLAKE3-committed artifacts** loaded at init.
//! No gradients, no training. The steering is an **additive overlay** on mutable
//! per-tick state — it does NOT mutate the frozen personality shard. This is
//! the freeze/thaw pattern: the shard is read-only, the steering field is a
//! separate mutable overlay (atomic Arc swap for hot-swap).
//!
//! ## Zero-alloc hot path
//!
//! [`apply_latent_steering`] and [`apply_field_to_crowd`] take borrowed slices
//! only — no heap allocation. The steering loop is an element-wise SAXPY
//! (`s[i] += α · v[i]`) that auto-vectorizes at d=8 (HLA scale).
//!
//! ## Design decision: `Vec<f32>` vs `[f32; D]`
//!
//! Uses `Vec<f32>` for the direction to match `EmotionDirections` storage
//! (dynamically sized, same artifact format for read-side project and write-
//! side steer). Game-side hot path can wrap in a typed `HLAField([f32; 8])`
//! alias in riir-ai once G1–G5 pass.

use blake3::Hasher;

// ── HLA axis indices ──────────────────────────────────────────────

/// Index of the valence axis in the 8-dim HLA activation vector.
pub const HLA_VALENCE: usize = 0;
/// Index of the arousal axis.
pub const HLA_AROUSAL: usize = 1;
/// Index of the desperation axis.
pub const HLA_DESPERATION: usize = 2;
/// Index of the calm axis.
pub const HLA_CALM: usize = 3;
/// Index of the fear axis.
pub const HLA_FEAR: usize = 4;
/// Standard HLA dimensionality (5 affective + 3 reserved).
pub const HLA_DIM: usize = 8;

// ── Errors ────────────────────────────────────────────────────────

/// Errors returned by [`LatentSteeringVector::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatentSteeringError {
    /// Direction vector is not unit-norm within the constructor tolerance.
    NotUnitNorm,
    /// Strength `alpha` is outside `[0.0, 1.0]`.
    AlphaOutOfRange,
}

// ── Steering vector ───────────────────────────────────────────────

/// Unit-norm direction in latent space + scalar strength, BLAKE3-committed.
///
/// Reuses the same artifact shape as `EmotionDirections` (Plan 162) so the
/// same frozen direction works for read-side (project) and write-side (steer).
/// The `commitment` field is `BLAKE3(direction_le || alpha_le)`, mirroring the
/// `MerkleFrozenEnvelope` pattern in riir-neuron-db.
#[derive(Debug, Clone)]
pub struct LatentSteeringVector {
    /// Unit-norm direction, d ≤ 64 (HLA d=8).
    pub direction: Vec<f32>,
    /// Strength α ∈ [0, 1]. Sigmoid-bounded at construction.
    pub alpha: f32,
    /// `BLAKE3(direction_le || alpha_le)` — content-addressed commitment.
    pub commitment: [u8; 32],
}

impl LatentSteeringVector {
    /// Construct a steering vector, validating unit-norm and alpha range.
    ///
    /// Returns [`LatentSteeringError::NotUnitNorm`] if `||direction||` deviates
    /// from 1.0 by more than `norm_tol`. Returns
    /// [`LatentSteeringError::AlphaOutOfRange`] if `alpha` is outside `[0, 1]`.
    pub fn new(
        direction: Vec<f32>,
        alpha: f32,
        norm_tol: f32,
    ) -> Result<Self, LatentSteeringError> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(LatentSteeringError::AlphaOutOfRange);
        }
        let norm = direction_norm(&direction);
        if (norm - 1.0).abs() > norm_tol {
            return Err(LatentSteeringError::NotUnitNorm);
        }
        let commitment = compute_commitment(&direction, alpha);
        Ok(Self { direction, alpha, commitment })
    }

    /// Construct without validation. Caller guarantees unit-norm + alpha range.
    /// Used when the direction comes from a trusted frozen artifact.
    pub fn new_unchecked(direction: Vec<f32>, alpha: f32) -> Self {
        let commitment = compute_commitment(&direction, alpha);
        Self { direction, alpha, commitment }
    }

    /// Re-check unit-norm (within `tol`) AND that the stored commitment matches
    /// the current contents. Returns `false` if either check fails.
    pub fn verify(&self, tol: f32) -> bool {
        let norm = direction_norm(&self.direction);
        if (norm - 1.0).abs() > tol {
            return false;
        }
        compute_commitment(&self.direction, self.alpha) == self.commitment
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.direction.len()
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.direction
    }
}

// ── Field support ─────────────────────────────────────────────────

/// Support descriptor for a localized steering field.
#[derive(Debug, Clone, Copy)]
pub enum FieldSupport {
    /// Global — applies to all entities regardless of position/zone.
    Global,
    /// Radius-banded — applies within `bandwidth` of `center` (Euclidean).
    /// Kernel: `sigmoid((bandwidth - distance) * steepness)`.
    Radius {
        center: [f32; 2],
        bandwidth: f32,
        steepness: f32,
    },
    /// Zone-keyed — applies to entities whose zone hash matches.
    Zone { zone_hash: u64 },
}

/// A steering vector + support descriptor.
#[derive(Debug, Clone)]
pub struct LatentField {
    pub steering: LatentSteeringVector,
    pub support: FieldSupport,
}

// ── Steering kernels ──────────────────────────────────────────────

/// Apply steering to a single latent state slice. Zero-alloc.
///
/// `state` is d-dimensional (e.g., HLA 8-dim). Computes `state[i] += alpha * direction[i]`.
/// The support descriptor is ignored — use [`kernel_weight`] + [`apply_latent_steering_weighted`]
/// for localized fields, or [`apply_field_to_crowd`] for batch application.
#[inline]
pub fn apply_latent_steering(state: &mut [f32], steering: &LatentSteeringVector) {
    debug_assert_eq!(
        state.len(),
        steering.dim(),
        "state dim {} != steering dim {}",
        state.len(),
        steering.dim()
    );
    let alpha = steering.alpha;
    let dir = steering.as_slice();
    // Element-wise SAXPY: auto-vectorizes at d=8 (HLA). No manual SIMD needed
    // until a deployment shows this is the bottleneck (unlikely at d ≤ 64).
    for (s, d) in state.iter_mut().zip(dir.iter()) {
        *s += alpha * d;
    }
}

/// Apply steering with an explicit kernel weight `w`. Zero-alloc.
/// Effective strength is `alpha * w`. Use for localized fields where the
/// caller has already computed the kernel weight via [`kernel_weight`].
#[inline]
pub fn apply_latent_steering_weighted(
    state: &mut [f32],
    steering: &LatentSteeringVector,
    w: f32,
) {
    if w <= 0.0 {
        return;
    }
    debug_assert_eq!(state.len(), steering.dim());
    let effective_alpha = steering.alpha * w;
    let dir = steering.as_slice();
    for (s, d) in state.iter_mut().zip(dir.iter()) {
        *s += effective_alpha * d;
    }
}

/// Kernel weight for an entity given support. Returns 0.0 outside support.
///
/// - `Global` → always 1.0.
/// - `Radius` → `sigmoid((bandwidth - distance) * steepness)`. Returns 0.0 if
///   `entity_pos` is `None` (entity has no position, cannot be in a radius field).
/// - `Zone` → 1.0 if zone matches, 0.0 otherwise.
#[inline]
pub fn kernel_weight(
    support: &FieldSupport,
    entity_pos: Option<[f32; 2]>,
    entity_zone: Option<u64>,
) -> f32 {
    match support {
        FieldSupport::Global => 1.0,
        FieldSupport::Radius { center, bandwidth, steepness } => {
            let pos = match entity_pos {
                Some(p) => p,
                None => return 0.0,
            };
            let dx = pos[0] - center[0];
            let dy = pos[1] - center[1];
            let dist = (dx * dx + dy * dy).sqrt();
            sigmoid((bandwidth - dist) * steepness)
        }
        FieldSupport::Zone { zone_hash } => match entity_zone {
            Some(z) if z == *zone_hash => 1.0,
            _ => 0.0,
        },
    }
}

/// Apply a field to a crowd of latent states. Zero-alloc given borrowed slices.
///
/// `states` is flattened `[e0d0, e0d1, ..., eNd(D-1)]` (N*D). `positions` and
/// `zones` are per-entity (length N). For `Global` fields, positions/zones are
/// ignored (kernel weight is always 1.0) but must still be the correct length.
///
/// For `Radius`/`Zone` fields, entities outside the support are skipped (no
/// mutation), achieving zero leakage.
pub fn apply_field_to_crowd(
    states: &mut [f32],
    entity_dim: usize,
    positions: &[Option<[f32; 2]>],
    zones: &[Option<u64>],
    field: &LatentField,
) {
    debug_assert_eq!(states.len(), positions.len() * entity_dim);
    debug_assert_eq!(positions.len(), zones.len());
    let dir = field.steering.as_slice();
    let base_alpha = field.steering.alpha;
    for (i, entity_state) in states.chunks_mut(entity_dim).enumerate() {
        let w = kernel_weight(&field.support, positions[i], zones[i]);
        if w <= 0.0 {
            continue;
        }
        let effective_alpha = base_alpha * w;
        for (s, d) in entity_state.iter_mut().zip(dir.iter()) {
            *s += effective_alpha * d;
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────

/// Compute `BLAKE3(direction_le || alpha_le)`.
///
/// Direction is serialized as little-endian `f32` per-element (matches the
/// `engram/commitment.rs::build_merkle_root` and `cross_resolution.rs`
/// conventions). Alpha is little-endian `f32`.
fn compute_commitment(direction: &[f32], alpha: f32) -> [u8; 32] {
    let mut hasher = Hasher::new();
    for &f in direction {
        hasher.update(&f.to_le_bytes());
    }
    hasher.update(&alpha.to_le_bytes());
    let mut out = [0u8; 32];
    hasher.finalize_xof().fill(&mut out);
    out
}

#[inline]
fn direction_norm(direction: &[f32]) -> f32 {
    direction.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_unit_direction(d: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed;
        let mut v: Vec<f32> = (0..d)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng >> 33) as f32) / (1u64 << 31) as f32 - 1.0
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm.max(1e-12);
        }
        v
    }

    #[test]
    fn smoke_global_field_shifts_state() {
        // T1.5: Global field shifts state by exactly alpha * direction.
        let dir = make_unit_direction(8, 42);
        let steering = LatentSteeringVector::new(dir.clone(), 0.5, 1e-4).unwrap();
        let field = LatentField { steering, support: FieldSupport::Global };

        let mut state = vec![0.1f32; 8];
        apply_latent_steering(&mut state, &field.steering);
        for i in 0..8 {
            let expected = 0.1 + 0.5 * dir[i];
            assert!(
                (state[i] - expected).abs() < 1e-5,
                "state[{i}] = {} expected {expected}",
                state[i]
            );
        }
    }

    #[test]
    fn smoke_radius_field_localizes() {
        // T1.5: Radius field applies inside, skips outside.
        let dir = make_unit_direction(8, 7);
        let steering = LatentSteeringVector::new(dir, 0.3, 1e-4).unwrap();
        let field = LatentField {
            steering,
            support: FieldSupport::Radius {
                center: [0.0, 0.0],
                bandwidth: 10.0,
                steepness: 2.0,
            },
        };

        let n = 100;
        let mut states = vec![0.0f32; n * 8];
        let positions: Vec<Option<[f32; 2]>> = (0..n)
            .map(|i| {
                if i < 50 {
                    Some([5.0, 0.0]) // inside (d=5 < bandwidth=10)
                } else {
                    Some([15.0, 0.0]) // outside (d=15 > bandwidth=10)
                }
            })
            .collect();
        let zones = vec![None; n];

        apply_field_to_crowd(&mut states, 8, &positions, &zones, &field);

        // Inside: shifted (sigmoid((10-5)*2) = sigmoid(10) ≈ 1.0)
        let inside_shift = states[0..50 * 8].iter().map(|x| x.abs()).sum::<f32>();
        assert!(inside_shift > 0.0, "inside entities must be shifted");

        // Outside: sigmoid((10-15)*2) = sigmoid(-10) ≈ 4.5e-5 — effectively 0
        let outside_shift = states[50 * 8..].iter().map(|x| x.abs()).sum::<f32>();
        let ratio = outside_shift / inside_shift.max(1e-12);
        assert!(
            ratio < 0.01,
            "outside/inside shift ratio = {ratio} should be < 0.01"
        );
    }

    #[test]
    fn smoke_constructor_rejects_non_unit_norm() {
        let dir = vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // norm = 2.0
        let result = LatentSteeringVector::new(dir, 0.5, 1e-4);
        assert_eq!(result.unwrap_err(), LatentSteeringError::NotUnitNorm);
    }

    #[test]
    fn smoke_constructor_rejects_alpha_out_of_range() {
        let dir = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = LatentSteeringVector::new(dir, 1.5, 1e-4);
        assert_eq!(result.unwrap_err(), LatentSteeringError::AlphaOutOfRange);
    }

    #[test]
    fn smoke_commitment_roundtrip() {
        let dir = make_unit_direction(8, 99);
        let sv = LatentSteeringVector::new(dir, 0.7, 1e-4).unwrap();
        assert!(sv.verify(1e-3), "commitment + norm must verify");
    }

    #[test]
    fn smoke_zone_field_matches_only_matching_zone() {
        let dir = make_unit_direction(8, 3);
        let steering = LatentSteeringVector::new(dir, 0.4, 1e-4).unwrap();
        let field = LatentField {
            steering,
            support: FieldSupport::Zone { zone_hash: 12345 },
        };

        let n = 4;
        let mut states = vec![0.0f32; n * 8];
        let positions = vec![None; n];
        let zones = vec![Some(1u64), Some(12345), Some(999), None];

        apply_field_to_crowd(&mut states, 8, &positions, &zones, &field);

        // Only entity 1 (zone 12345) should be shifted.
        let e0 = &states[0..8];
        let e1 = &states[8..16];
        let e2 = &states[16..24];
        let e3 = &states[24..32];
        assert!(e0.iter().all(|x| x.abs() < 1e-9), "e0 wrong zone, no shift");
        assert!(e1.iter().any(|x| x.abs() > 1e-5), "e1 matching zone, must shift");
        assert!(e2.iter().all(|x| x.abs() < 1e-9), "e2 wrong zone, no shift");
        assert!(e3.iter().all(|x| x.abs() < 1e-9), "e3 no zone, no shift");
    }
}
