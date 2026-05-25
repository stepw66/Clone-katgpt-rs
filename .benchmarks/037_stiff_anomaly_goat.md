# GOAT Proof: Stiff/Soft Subspace Anomaly Gate (Plan 138)

## Module: `stiff_anomaly`

| GOAT | Target | Proof Test |
|------|--------|------------|
| G1 | Known rotation → k matches rank at 90% trace mass; isotropic → k = d; rank-3 → k = 3 | `subspace::test_g1_stiff_subspace_k`, `test_g1_decompose_rank3` |
| G2 | 100 stable windows → median Jaccard ≥ 0.85; perturbed → Jaccard drops | `stability::test_g2_jaccard_stable`, `test_g2_jaccard_perturbed` |
| G3 | FPR ≤ 4% on 50 stable windows; 100% detection on 5 anomalous | `stability::test_g3_fpr_zero_and_full_detection` |
| G4 | Known structure → σ separation ≥ 5.0; pure noise → σ ≈ 0.0 | `baseline::test_g4_structured_high_sigma`, `test_g4_noise_sigma_near_zero` |
| G5 | Integration with eigenvalue data → k matches expected effective dimension | `subspace::test_g5_effective_dimension` |
| G6 | End-to-end example runs without errors | `cargo run --example stiff_anomaly_demo --features stiff_anomaly` |

## Status: GOAT 6/6
