//! Plausibility manifold generation — K speculative trajectories from a known last-good state.
//!
//! The manifold is a set of K forward-simulated trajectories that represent plausible
//! future states given a starting point. Each trajectory is a sequence of `TrajectoryPoint`s
//! advanced by a physics-like step with Gaussian noise injection.
//!
//! This is the *modelless* (no neural forward pass) implementation: goal directions are
//! sampled either from LEO Q-values or uniformly at random.

use std::f32::consts::{PI, TAU};

use super::types::{ReconciliationConfig, TrajectoryPoint};
use katgpt_types::Rng;

// ── Trait ───────────────────────────────────────────────────────

/// Generates a plausibility manifold: K speculative trajectories from `h_last`.
pub trait ManifoldGenerator: Send + Sync {
    /// Generate `k` trajectories, each with `steps` points, starting from `h_last`.
    ///
    /// * `h_last`  — last known-good state
    /// * `q_goals` — LEO Q-values (goal weights); may be empty for modelless mode
    /// * `k`       — number of trajectories
    /// * `dt`      — time step per advancement
    /// * `steps`   — number of steps per trajectory
    /// * `rng`     — deterministic RNG
    fn generate(
        &self,
        h_last: &TrajectoryPoint,
        q_goals: &[f32],
        k: usize,
        dt: f32,
        steps: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<TrajectoryPoint>>;
}

// ── Helper ──────────────────────────────────────────────────────

/// Box-Muller transform: produce a single standard-normal sample.
#[inline]
pub fn gaussian_sample(rng: &mut Rng) -> f32 {
    // Box-Muller: z = sqrt(-2 * ln(u1)) * cos(2π * u2)
    let u1 = rng.uniform().max(1e-10);
    let u2 = rng.uniform();
    (-2.0f32 * u1.ln()).sqrt() * (TAU * u2).cos()
}

// ── Default Implementation ──────────────────────────────────────

/// Modelless manifold generator — no neural forward pass.
///
/// Uses LEO Q-values for goal weighting when available, otherwise samples
/// a random direction uniformly.
pub struct DefaultManifoldGenerator {
    /// Tunable thresholds and bounds (used for max_speed, noise_sigma, map_bounds).
    pub config: ReconciliationConfig,
}

impl DefaultManifoldGenerator {
    /// Create a new generator with the given config.
    #[inline]
    pub fn new(config: ReconciliationConfig) -> Self {
        Self { config }
    }

    /// Sample a goal direction from Q-value weights.
    ///
    /// Each Q-value corresponds to a direction bin (uniformly spaced over [0, 2π)).
    /// Returns the sampled direction angle in radians.
    ///
    /// Hot-path optimization: pre-compute shifted weights once in a small
    /// `Vec<f32>` instead of recomputing `exp(q - max_q)` twice per element
    /// (which dominates cost once `q_goals.len()` exceeds ~4).
    fn sample_goal_from_q(&self, q_goals: &[f32], rng: &mut Rng) -> f32 {
        if q_goals.is_empty() {
            return rng.uniform() * TAU;
        }

        // Shift by max_q for numerical stability (same as before — this is
        // categorical sampling, not an activation; the project sigmoid rule
        // applies to bounded mappings, not probability normalization).
        let max_q = q_goals.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute weights once, reuse for total + sampling. Capacity hint
        // avoids the realloc-on-first-push.
        let weights: Vec<f32> = q_goals.iter().map(|q| (q - max_q).exp()).collect();
        let total: f32 = weights.iter().copied().sum();
        if total <= 0.0 {
            return rng.uniform() * TAU;
        }

        // Weighted sample using cached weights — no recomputation.
        let mut r = rng.uniform() * total;
        let mut chosen = weights.len() - 1;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }

        // Map bin to direction angle with jitter.
        let bin_count = weights.len();
        let bin_width = TAU / bin_count as f32;
        chosen as f32 * bin_width + rng.uniform() * bin_width
    }

    /// Clamp velocity to `max_speed`.
    ///
    /// Hot-path optimization: compare squared speed against the squared limit
    /// to avoid the `sqrt` on the fast (already-bounded) path. Only when we
    /// need to scale do we compute the actual magnitude.
    #[inline]
    fn clamp_velocity(vx: f32, vy: f32, max_speed: f32) -> (f32, f32) {
        let speed_sq = vx * vx + vy * vy;
        let max_sq = max_speed * max_speed;
        if speed_sq > max_sq && speed_sq > 0.0 {
            // scale = max_speed / sqrt(speed_sq)
            let inv_speed = max_speed / speed_sq.sqrt();
            (vx * inv_speed, vy * inv_speed)
        } else {
            (vx, vy)
        }
    }

    /// Clamp position within map bounds [min_x, min_y, max_x, max_y].
    #[inline]
    fn clamp_position(x: f32, y: f32, bounds: &[f32; 4]) -> (f32, f32) {
        (x.clamp(bounds[0], bounds[2]), y.clamp(bounds[1], bounds[3]))
    }
}

impl ManifoldGenerator for DefaultManifoldGenerator {
    fn generate(
        &self,
        h_last: &TrajectoryPoint,
        q_goals: &[f32],
        k: usize,
        dt: f32,
        steps: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<TrajectoryPoint>> {
        let sigma = self.config.noise_sigma;
        let max_speed = self.config.max_speed;
        let map_bounds = &self.config.map_bounds;
        let sqrt_dt = dt.sqrt();

        let mut trajectories = Vec::with_capacity(k);

        for _ in 0..k {
            // Sample a goal direction for this trajectory.
            let goal_dir = self.sample_goal_from_q(q_goals, rng);
            let goal_vx = goal_dir.cos() * max_speed * 0.5;
            let goal_vy = goal_dir.sin() * max_speed * 0.5;

            let mut traj = Vec::with_capacity(steps);

            // Start from h_last.
            let mut pos_x = h_last.pos_x();
            let mut pos_y = h_last.pos_y();
            let mut vel_x = h_last.vel_x();
            let mut vel_y = h_last.vel_y();
            let mut kills = h_last.kills();
            let mut deaths = h_last.deaths();
            let mut assists = h_last.assists();
            let mut direction = h_last.direction();

            for _ in 0..steps {
                // Blend velocity toward goal direction with some noise.
                let noise_vx = gaussian_sample(rng) * sigma * sqrt_dt;
                let noise_vy = gaussian_sample(rng) * sigma * sqrt_dt;

                vel_x = vel_x * 0.8 + goal_vx * 0.2 + noise_vx;
                vel_y = vel_y * 0.8 + goal_vy * 0.2 + noise_vy;

                // Clamp velocity.
                (vel_x, vel_y) = Self::clamp_velocity(vel_x, vel_y, max_speed);

                // Advance position: pos += vel * dt + noise * sigma * sqrt(dt).
                let noise_px = gaussian_sample(rng) * sigma * sqrt_dt;
                let noise_py = gaussian_sample(rng) * sigma * sqrt_dt;

                pos_x += vel_x * dt + noise_px;
                pos_y += vel_y * dt + noise_py;

                // Clamp position within map bounds.
                (pos_x, pos_y) = Self::clamp_position(pos_x, pos_y, map_bounds);

                // Kills increase stochastically: with small probability, add max(0, noise).
                let kill_noise = gaussian_sample(rng);
                if kill_noise > 2.0 {
                    kills += (kill_noise - 2.0).max(0.0);
                }

                // Deaths stay stable with very small probability of increment.
                let death_noise = gaussian_sample(rng);
                if death_noise > 3.0 {
                    deaths += 1.0;
                }

                // Assists have small chance of incrementing.
                let assist_noise = gaussian_sample(rng);
                if assist_noise > 2.5 {
                    assists += 1.0;
                }

                // Direction updates smoothly toward velocity direction.
                let vel_dir = vel_y.atan2(vel_x);
                // Normalize delta to [-π, π] via one add + one remainder
                // rather than a `while` loop (cheap and bounded).
                let mut delta = (vel_dir - direction) % TAU;
                if delta > PI {
                    delta -= TAU;
                } else if delta < -PI {
                    delta += TAU;
                }
                direction += delta * 0.3;
                // Normalize to [0, 2π).
                direction = ((direction % TAU) + TAU) % TAU;

                traj.push(TrajectoryPoint::from_fields(
                    pos_x, pos_y, vel_x, vel_y, kills, deaths, assists, direction,
                ));
            }

            trajectories.push(traj);
        }

        trajectories
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> Rng {
        Rng::new(42)
    }

    fn make_config() -> ReconciliationConfig {
        ReconciliationConfig::default()
    }

    fn make_origin_point() -> TrajectoryPoint {
        TrajectoryPoint::from_fields(2048.0, 2048.0, 10.0, 5.0, 2.0, 0.0, 1.0, 0.0)
    }

    #[test]
    fn test_gaussian_sample_produces_reasonable_values() {
        let mut rng = make_rng();
        let mut sum = 0.0f32;
        let n = 10_000;
        for _ in 0..n {
            let v = gaussian_sample(&mut rng);
            assert!(v.is_finite(), "gaussian_sample produced non-finite value");
            sum += v;
        }
        let mean = sum / n as f32;
        // Mean of standard normal should be close to 0.
        assert!(
            mean.abs() < 0.1,
            "mean of gaussian samples should be near 0, got {mean}"
        );
    }

    #[test]
    fn test_generates_correct_trajectory_count() {
        let generator = DefaultManifoldGenerator::new(make_config());
        let mut rng = make_rng();
        let h = make_origin_point();
        let k = 16;
        let steps = 10;

        let result = generator.generate(&h, &[], k, 1.0 / 60.0, steps, &mut rng);

        assert_eq!(result.len(), k, "should generate K trajectories");
        for (i, traj) in result.iter().enumerate() {
            assert_eq!(
                traj.len(),
                steps,
                "trajectory {i} should have {steps} steps"
            );
        }
    }

    #[test]
    fn test_trajectories_start_from_h_last() {
        let generator = DefaultManifoldGenerator::new(make_config());
        let mut rng = make_rng();
        let h = make_origin_point();
        let k = 4;
        let steps = 5;

        let result = generator.generate(&h, &[], k, 1.0 / 60.0, steps, &mut rng);

        for (ti, traj) in result.iter().enumerate() {
            // The first point should be close to h_last (noise is small relative to position).
            let first = &traj[0];
            let dx = (first.pos_x() - h.pos_x()).abs();
            let dy = (first.pos_y() - h.pos_y()).abs();
            // With default sigma=0.1, the displacement from the origin should be small.
            assert!(
                dx < 5.0,
                "trajectory {ti}: first point x should be near h_last, dx={dx}"
            );
            assert!(
                dy < 5.0,
                "trajectory {ti}: first point y should be near h_last, dy={dy}"
            );
        }
    }

    #[test]
    fn test_sigma_grows_with_dt() {
        let config = make_config();
        let generator = DefaultManifoldGenerator::new(config);
        let h = make_origin_point();
        let k = 8;
        let steps = 50;

        // Short dt → tight spread.
        let mut rng1 = make_rng();
        let tight = generator.generate(&h, &[], k, 1.0 / 60.0, steps, &mut rng1);

        // Long dt → wider spread.
        let mut rng2 = make_rng();
        let wide = generator.generate(&h, &[], k, 1.0, steps, &mut rng2);

        // Measure average distance from h_last to final point.
        let avg_tight: f32 = tight
            .iter()
            .map(|t| t.last().unwrap().distance_to(&h))
            .sum::<f32>()
            / k as f32;

        let avg_wide: f32 = wide
            .iter()
            .map(|t| t.last().unwrap().distance_to(&h))
            .sum::<f32>()
            / k as f32;

        assert!(
            avg_wide > avg_tight,
            "longer dt should produce wider spread: wide={avg_wide}, tight={avg_tight}"
        );
    }

    #[test]
    fn test_goals_sampled_from_q_values() {
        let config = make_config();
        let generator = DefaultManifoldGenerator::new(config);
        let h = make_origin_point();
        let mut rng = make_rng();
        let k = 8;
        let steps = 20;

        // Strongly weighted Q-values pointing in direction bin 0 (angle 0, rightward).
        let q_goals = vec![10.0, 0.0, 0.0, 0.0];

        let result = generator.generate(&h, &q_goals, k, 1.0 / 60.0, steps, &mut rng);

        // With strong goal weight in direction 0, trajectories should drift rightward (positive x).
        let drift_right = result
            .iter()
            .filter(|t| t.last().unwrap().pos_x() > h.pos_x())
            .count();

        // At least half should drift right (probabilistic, but with weight 10.0 it should be most).
        assert!(
            drift_right >= k / 2,
            "expected majority of trajectories to drift right with goal weight, got {drift_right}/{k}"
        );
    }

    #[test]
    fn test_empty_q_goals_fallback_to_random() {
        let config = make_config();
        let generator = DefaultManifoldGenerator::new(config);
        let h = make_origin_point();
        let mut rng = make_rng();
        let k = 16;
        let steps = 20;

        // Empty Q-values should still work (random direction fallback).
        let result = generator.generate(&h, &[], k, 1.0 / 60.0, steps, &mut rng);

        assert_eq!(result.len(), k);
        for traj in &result {
            assert_eq!(traj.len(), steps);
        }
    }

    #[test]
    fn test_map_bounds_respected() {
        let config = make_config();
        let generator = DefaultManifoldGenerator::new(config);
        // Start near the edge of the map.
        let h = TrajectoryPoint::from_fields(
            config.map_bounds[2] - 10.0, // near max_x
            config.map_bounds[3] - 10.0, // near max_y
            100.0,
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        let mut rng = make_rng();
        let k = 8;
        let steps = 100;

        let result = generator.generate(&h, &[], k, 1.0 / 60.0, steps, &mut rng);

        for (ti, traj) in result.iter().enumerate() {
            for (si, pt) in traj.iter().enumerate() {
                assert!(
                    pt.pos_x() >= config.map_bounds[0] && pt.pos_x() <= config.map_bounds[2],
                    "trajectory {ti} step {si}: x={} out of bounds [{}, {}]",
                    pt.pos_x(),
                    config.map_bounds[0],
                    config.map_bounds[2]
                );
                assert!(
                    pt.pos_y() >= config.map_bounds[1] && pt.pos_y() <= config.map_bounds[3],
                    "trajectory {ti} step {si}: y={} out of bounds [{}, {}]",
                    pt.pos_y(),
                    config.map_bounds[1],
                    config.map_bounds[3]
                );
            }
        }
    }
}
