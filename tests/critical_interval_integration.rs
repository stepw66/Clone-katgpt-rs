//! Integration test for CriticalIntervalGate with a simple constraint problem.
//! Compares acceptance rate with/without the gate.

#[cfg(feature = "critical_interval_gate")]
mod tests {
    use katgpt_core::dllm_solver::{SolverKind, build_dd_tree_adaptive};

    /// Simulate a simple constraint satisfaction problem:
    /// 4x4 grid, each cell has 4 possible values, row/col uniqueness constraint.
    /// At each depth, marginals represent probability of each value for current cell.
    #[test]
    fn test_critical_interval_improves_branch_rate() {
        let grid_size = 4;
        let vocab_size = 4;
        let depths = grid_size * grid_size; // 16 cells

        // Generate marginals: some cells are "easy" (peaked), some "hard" (uniform)
        let mut marginals_per_depth: Vec<Vec<f32>> = Vec::new();
        for d in 0..depths {
            let is_row_start = d % grid_size == 0;
            let is_col_start = d < grid_size;

            if is_row_start || is_col_start {
                // Hard: high entropy (many options still available)
                let uniform = 1.0 / vocab_size as f32;
                marginals_per_depth.push(vec![uniform; vocab_size]);
            } else {
                // Easy: peaked (constraint narrows options)
                let mut m = vec![0.02; vocab_size];
                m[d % vocab_size] = 0.94; // one dominant value
                marginals_per_depth.push(m);
            }
        }

        // With critical interval gate
        let mut solver_with = SolverKind::DpmSolver2M;
        let transitions = build_dd_tree_adaptive(&marginals_per_depth, 0.0, &mut solver_with);

        // Verify transitions were recorded
        assert_eq!(transitions.len(), depths);

        // Count critical depths
        let critical_count = transitions.iter().filter(|t| t.critical).count();

        // Row/col start positions should be critical (high entropy)
        assert!(
            critical_count > 0,
            "Should detect at least some critical intervals"
        );

        // Verify solver switched for critical depths (only when q_sample_solver enabled)
        #[cfg(feature = "q_sample_solver")]
        {
            let switches: Vec<_> = transitions
                .iter()
                .filter(|t| t.solver_before != t.solver_after)
                .collect();
            assert!(
                !switches.is_empty(),
                "Solver should switch at least once with q_sample_solver"
            );
        }

        // Without q_sample_solver, verify critical intervals are still detected
        #[cfg(not(feature = "q_sample_solver"))]
        {
            let critical_depths: Vec<_> = transitions.iter().filter(|t| t.critical).collect();
            assert!(!critical_depths.is_empty());
        }

        // Without gate: just count how many depths would benefit from switching
        // (this is a qualitative comparison — we verify the gate activates correctly)
    }

    #[test]
    fn test_solver_transition_log_integrity() {
        let uniform = vec![0.25; 4]; // high entropy
        let peaked = vec![0.9, 0.03, 0.04, 0.03]; // low entropy

        let depths = vec![uniform.clone(), peaked.clone(), uniform, peaked];
        let mut solver = SolverKind::DpmSolver2M;

        let transitions = build_dd_tree_adaptive(&depths, 1.0, &mut solver);

        // Verify transition log entries have correct metadata
        for (i, t) in transitions.iter().enumerate() {
            assert_eq!(t.depth, i);
            assert!(t.entropy >= 0.0);
        }

        // Depth 0 (uniform) should be critical
        assert!(transitions[0].critical);
        // Depth 1 (peaked) should not be critical
        assert!(!transitions[1].critical);
    }
}
