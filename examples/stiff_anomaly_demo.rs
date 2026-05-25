//! Stiff/Soft Subspace Anomaly Gate demo (Plan 138).
//!
//! Demonstrates:
//! 1. Create synthetic eigenvalue windows (50 stable + 5 anomalous)
//! 2. Freeze baseline, track k-invariance
//! 3. Run anomaly gate with FPR validation
//! 4. Show Jaccard stability

use katgpt_rs::stiff_anomaly::{
    EigenvalueTracker, StiffAnomalyGate, decompose, monte_carlo_null_test, soft_alignment_ratio,
    stiff_subspace_k,
};

fn main() {
    println!("=== Stiff/Soft Subspace Anomaly Gate Demo (Plan 138) ===\n");

    let dim = 6;
    let base_eigenvalues = vec![12.0, 8.0, 5.0, 2.0, 0.5, 0.1];

    // Step 1: Generate synthetic eigenvalue windows
    println!("Step 1: Generating 50 stable + 5 anomalous windows (dim={dim})");
    let mut rng = fastrand::Rng::with_seed(42);

    let stable_windows: Vec<Vec<f32>> = (0..50)
        .map(|_| {
            base_eigenvalues
                .iter()
                .map(|&v| (v + (rng.f32() - 0.5) * 2.0 * 0.3).max(0.001))
                .collect()
        })
        .collect();

    let anomalous_windows: Vec<Vec<f32>> = (0..5)
        .map(|_| {
            base_eigenvalues
                .iter()
                .enumerate()
                .map(|(i, &v)| if i < 2 { v * 0.05 } else { v })
                .collect()
        })
        .collect();

    println!(
        "  Base spectrum: {:?}",
        base_eigenvalues
            .iter()
            .map(|v| format!("{v:.1}"))
            .collect::<Vec<_>>()
    );
    println!("  Anomalous:     collapsed first 2 eigenvalues to 5%\n");

    // Step 2: Freeze baseline, track k-invariance
    println!("Step 2: Freeze baseline from 50 stable windows");
    let tracker = EigenvalueTracker::freeze_baseline(&stable_windows);
    println!(
        "  Baseline mean: {:?}",
        tracker
            .baseline_mean
            .iter()
            .map(|v| format!("{v:.2}"))
            .collect::<Vec<_>>()
    );
    println!(
        "  Baseline std:  {:?}",
        tracker
            .baseline_std
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
    );

    let k90 = stiff_subspace_k(&tracker.baseline_mean, 0.90);
    println!("  Stiff k at 90% trace mass: {k90}");

    let is_invariant = tracker.k_invariant(0.90);
    println!("  k invariant across baseline: {is_invariant}\n");

    // Step 3: Run anomaly gate with FPR validation
    println!("Step 3: Anomaly gate evaluation");
    let gate = StiffAnomalyGate::new(&tracker);
    println!("  z_threshold: {}", gate.z_threshold);
    println!("  alpha_threshold: {}", gate.alpha_threshold);

    // Identity eigenvectors for demo
    let eigenvectors: Vec<Vec<f32>> = (0..dim)
        .map(|i| {
            let mut v = vec![0.0f32; dim];
            v[i] = 1.0;
            v
        })
        .collect();

    let fpr = gate.validate_fpr(&tracker, &stable_windows, &eigenvectors, 0.90);
    println!("  FPR on stable windows: {:.3}", fpr);

    println!("\n  Anomalous window detection:");
    for (i, w) in anomalous_windows.iter().enumerate() {
        let z_scores = tracker.eigenspace_zscore(w);
        let min_z = z_scores.iter().cloned().reduce(f32::min).unwrap_or(0.0);
        let result = gate.evaluate(&tracker, w, &eigenvectors, &[1.0; 6], 0.90);
        println!("    Window {i}: z_min={:.2} -> {result:?}", min_z);
    }

    // Step 4: Jaccard stability
    println!("\nStep 4: Jaccard stability analysis");
    let mut stable_jaccards = Vec::new();
    for i in 1..stable_windows.len() {
        let j =
            EigenvalueTracker::eigenvalue_jaccard(&stable_windows[i - 1], &stable_windows[i], 3);
        stable_jaccards.push(j);
    }
    let stable_mean = stable_jaccards.iter().sum::<f32>() / stable_jaccards.len() as f32;

    let anomaly_j =
        EigenvalueTracker::eigenvalue_jaccard(&stable_windows[49], &anomalous_windows[0], 3);

    println!("  Stable window Jaccard (top-3): mean={:.3}", stable_mean);
    println!("  Last stable -> first anomalous Jaccard: {:.3}", anomaly_j);

    // Decomposition demo
    println!("\n  Stiff/soft decomposition of base spectrum:");
    let decomp = decompose(base_eigenvalues.clone(), eigenvectors.clone(), 0.90);
    println!("    k = {}", decomp.k);
    println!(
        "    Stiff eigenvalues: {:?}",
        decomp
            .stiff_eigenvalues
            .iter()
            .map(|v| format!("{v:.1}"))
            .collect::<Vec<_>>()
    );
    println!(
        "    Soft eigenvalues:  {:?}",
        decomp
            .soft_eigenvalues
            .iter()
            .map(|v| format!("{v:.1}"))
            .collect::<Vec<_>>()
    );

    // Soft alignment ratio demo
    let delta_stiff = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let delta_soft = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let alpha_stiff = soft_alignment_ratio(&decomp, &delta_stiff);
    let alpha_soft = soft_alignment_ratio(&decomp, &delta_soft);
    println!("\n  Soft alignment ratio:");
    println!("    delta_x along stiff axis: alpha = {:.3}", alpha_stiff);
    println!("    delta_x along soft axis:  alpha = {:.3}", alpha_soft);

    // Monte Carlo null test
    println!("\n  Monte Carlo null test:");
    let null_result = monte_carlo_null_test(dim, 200, 42, |data| {
        // Participation ratio of variance diagonal
        let n = data.len();
        let d = data[0].len();
        let mut col_vars = vec![0.0f64; d];
        for j in 0..d {
            let mean: f64 = data.iter().map(|row| row[j] as f64).sum::<f64>() / n as f64;
            let var: f64 = data
                .iter()
                .map(|row| {
                    let diff = row[j] as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / n as f64;
            col_vars[j] = var;
        }
        let sum: f64 = col_vars.iter().sum();
        let sum_sq: f64 = col_vars.iter().map(|x| x * x).sum();
        if sum_sq < 1e-12 {
            0.0
        } else {
            ((sum * sum) / sum_sq) as f32
        }
    });
    println!("    Null mean: {:.3}", null_result.null_mean);
    println!("    Null std:  {:.3}", null_result.null_std);
    println!("    sigma separation: {:.1}", null_result.sigma_separation);

    println!("\n=== Demo complete ===");
}
