//! Core types for the Speculative Reconciliation Engine.
//!
//! All types are `#[repr(C)]` with zero heap allocation, suitable for
//! real-time game state verification.

/// Fixed-layout trajectory point: 8 floats describing a game state snapshot.
///
/// Layout: `[pos_x, pos_y, vel_x, vel_y, kills, deaths, assists, direction]`
///
/// - `pos_x`, `pos_y`: player position in world coordinates
/// - `vel_x`, `vel_y`: player velocity (units/sec)
/// - `kills`, `deaths`, `assists`: cumulative combat stats
/// - `direction`: facing angle in radians [0, 2π)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrajectoryPoint {
    /// Fixed-layout state: [pos_x, pos_y, vel_x, vel_y, kills, deaths, assists, direction]
    pub data: [f32; 8],
}

impl TrajectoryPoint {
    // Field indices
    pub const POS_X: usize = 0;
    pub const POS_Y: usize = 1;
    pub const VEL_X: usize = 2;
    pub const VEL_Y: usize = 3;
    pub const KILLS: usize = 4;
    pub const DEATHS: usize = 5;
    pub const ASSISTS: usize = 6;
    pub const DIRECTION: usize = 7;

    /// Create a new trajectory point from raw data.
    #[inline]
    pub const fn new(data: [f32; 8]) -> Self {
        Self { data }
    }

    /// Create a trajectory point from named fields.
    #[inline]
    #[allow(clippy::too_many_arguments)] // Named-field constructor for 8-axis trajectory point
    pub fn from_fields(
        pos_x: f32,
        pos_y: f32,
        vel_x: f32,
        vel_y: f32,
        kills: f32,
        deaths: f32,
        assists: f32,
        direction: f32,
    ) -> Self {
        Self {
            data: [
                pos_x, pos_y, vel_x, vel_y, kills, deaths, assists, direction,
            ],
        }
    }

    #[inline]
    pub fn pos_x(&self) -> f32 {
        self.data[Self::POS_X]
    }

    #[inline]
    pub fn pos_y(&self) -> f32 {
        self.data[Self::POS_Y]
    }

    #[inline]
    pub fn vel_x(&self) -> f32 {
        self.data[Self::VEL_X]
    }

    #[inline]
    pub fn vel_y(&self) -> f32 {
        self.data[Self::VEL_Y]
    }

    #[inline]
    pub fn kills(&self) -> f32 {
        self.data[Self::KILLS]
    }

    #[inline]
    pub fn deaths(&self) -> f32 {
        self.data[Self::DEATHS]
    }

    #[inline]
    pub fn assists(&self) -> f32 {
        self.data[Self::ASSISTS]
    }

    #[inline]
    pub fn direction(&self) -> f32 {
        self.data[Self::DIRECTION]
    }

    /// Position as (x, y).
    #[inline]
    pub fn position(&self) -> (f32, f32) {
        (self.data[Self::POS_X], self.data[Self::POS_Y])
    }

    /// Velocity as (vx, vy).
    #[inline]
    pub fn velocity(&self) -> (f32, f32) {
        (self.data[Self::VEL_X], self.data[Self::VEL_Y])
    }

    /// Speed magnitude.
    #[inline]
    pub fn speed(&self) -> f32 {
        let (vx, vy) = self.velocity();
        (vx * vx + vy * vy).sqrt()
    }

    /// Euclidean distance to another point (position only).
    #[inline]
    pub fn distance_to(&self, other: &Self) -> f32 {
        let dx = self.pos_x() - other.pos_x();
        let dy = self.pos_y() - other.pos_y();
        (dx * dx + dy * dy).sqrt()
    }

    /// Delta (difference) from `prev` to `self`.
    #[inline]
    pub fn delta_from(&self, prev: &Self) -> Self {
        let mut data = [0.0f32; 8];
        for (dst, (&s, &p)) in data.iter_mut().zip(self.data.iter().zip(prev.data.iter())) {
            *dst = s - p;
        }
        Self { data }
    }
}

/// Reconciliation verdict — result of verifying an offline trajectory
/// against the plausibility manifold.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconciliationVerdict {
    /// Trajectory is within expected bounds. Accept immediately.
    Accept,
    /// Trajectory violates hard or soft constraints. Quarantine for review.
    Quarantine,
    /// Insufficient confidence to accept or reject. Flag for manual review.
    Uncertain,
}

/// Configuration for the reconciliation engine.
///
/// All thresholds are tunable per-game. `#[repr(C)]` for FFI compatibility.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReconciliationConfig {
    /// Number of speculative trajectories to generate (manifold width).
    pub k: usize,
    /// Maximum allowed player speed (world units/sec).
    pub max_speed: f32,
    /// Map bounds: (min_x, min_y, max_x, max_y).
    pub map_bounds: [f32; 4],
    /// Cosine similarity threshold for soft scoring. Above this → Accept.
    pub accept_threshold: f32,
    /// Cosine similarity threshold below which we Quarantine.
    pub quarantine_threshold: f32,
    /// Kill-rate Chebyshev bound in standard deviations (default: 5.0).
    pub kill_rate_sigma: f32,
    /// Gaussian noise standard deviation for manifold generation.
    pub noise_sigma: f32,
    /// Delta time for velocity bound computation.
    pub dt: f32,
}

impl Default for ReconciliationConfig {
    fn default() -> Self {
        Self {
            k: 16,
            max_speed: 600.0,
            map_bounds: [0.0, 0.0, 4096.0, 4096.0],
            accept_threshold: 0.85,
            quarantine_threshold: 0.5,
            kill_rate_sigma: 5.0,
            noise_sigma: 0.1,
            dt: 1.0 / 60.0,
        }
    }
}

impl ReconciliationConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.k == 0 {
            return Err("k must be > 0".into());
        }
        if self.max_speed <= 0.0 {
            return Err("max_speed must be > 0".into());
        }
        if self.map_bounds[0] >= self.map_bounds[2] || self.map_bounds[1] >= self.map_bounds[3] {
            return Err("map_bounds must be (min_x, min_y, max_x, max_y) with min < max".into());
        }
        if self.accept_threshold <= self.quarantine_threshold {
            return Err("accept_threshold must be > quarantine_threshold".into());
        }
        if self.kill_rate_sigma <= 0.0 {
            return Err("kill_rate_sigma must be > 0".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_point_construction() {
        let tp = TrajectoryPoint::from_fields(100.0, 200.0, 5.0, -3.0, 2.0, 0.0, 1.0, 1.57);
        assert!((tp.pos_x() - 100.0).abs() < f32::EPSILON);
        assert!((tp.pos_y() - 200.0).abs() < f32::EPSILON);
        assert!((tp.vel_x() - 5.0).abs() < f32::EPSILON);
        assert!((tp.vel_y() - (-3.0)).abs() < f32::EPSILON);
        assert!((tp.kills() - 2.0).abs() < f32::EPSILON);
        assert!((tp.direction() - 1.57).abs() < f32::EPSILON);
        assert!((tp.speed() - (25.0f32 + 9.0).sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_trajectory_point_default() {
        let tp = TrajectoryPoint::default();
        assert_eq!(tp.data, [0.0; 8]);
    }

    #[test]
    fn test_trajectory_point_delta() {
        let a = TrajectoryPoint::from_fields(10.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let b = TrajectoryPoint::from_fields(15.0, 23.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.5);
        let d = b.delta_from(&a);
        assert!((d.pos_x() - 5.0).abs() < f32::EPSILON);
        assert!((d.pos_y() - 3.0).abs() < f32::EPSILON);
        assert!((d.kills() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_trajectory_point_distance() {
        let a = TrajectoryPoint::from_fields(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let b = TrajectoryPoint::from_fields(3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_verdict_debug_clone() {
        let v = ReconciliationVerdict::Accept;
        assert_eq!(format!("{v:?}"), "Accept");
        let v2 = v;
        assert_eq!(v, v2);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = ReconciliationConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_bad_k() {
        let cfg = ReconciliationConfig {
            k: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_thresholds() {
        let cfg = ReconciliationConfig {
            accept_threshold: 0.3,
            quarantine_threshold: 0.5,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_repr_c_layout() {
        // TrajectoryPoint should be exactly 32 bytes (8 × f32)
        assert_eq!(std::mem::size_of::<TrajectoryPoint>(), 32);
        // ReconciliationVerdict should be small (discriminant + padding)
        assert!(std::mem::size_of::<ReconciliationVerdict>() <= 8);
    }
}
