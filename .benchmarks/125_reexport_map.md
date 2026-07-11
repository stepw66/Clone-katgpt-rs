# Issue 125 Re-export Map

Source: `katgpt-rs/src/lib.rs` — 136 `pub use katgpt_*` re-exports.

## Aliases (root path → leaf path)

### → katgpt-core
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::breakeven` | `katgpt_core::breakeven` |
| `katgpt_rs::channel_simd` | `katgpt_core::channel_simd` |
| `katgpt_rs::cgsp` | `katgpt_core::cgsp` |
| `katgpt_rs::cumprodsum` | `katgpt_core::cumprodsum` |
| `katgpt_rs::compaction` | `katgpt_core::compaction` |
| `katgpt_rs::ssd_block` | `katgpt_core::ssd_block` |
| `katgpt_rs::data_probe::*` | `katgpt_core::data_probe::*` |
| `katgpt_rs::cubical_nerve` | `katgpt_core::cubical_nerve` |
| `katgpt_rs::compression_drafter` | `katgpt_core::compression_drafter` |
| `katgpt_rs::dllm_solver` | `katgpt_core::dllm_solver` |
| `katgpt_rs::cce` | `katgpt_core::cce` |
| `katgpt_rs::newton_schulz` | `katgpt_core::newton_schulz` |
| `katgpt_rs::pipeline_pruner` | `katgpt_core::pipeline_pruner` |
| `katgpt_rs::alloc` | `katgpt_core::alloc` |
| `katgpt_rs::mux_demux` | `katgpt_core::mux_demux` |
| `katgpt_rs::mux_latent` | `katgpt_core::mux_latent` |
| `katgpt_rs::memory_soup_lora` | `katgpt_core::memory_soup_lora` |
| `katgpt_rs::llmexec_guard` | `katgpt_core::llmexec_guard` |
| `katgpt_rs::closure::mining` | `katgpt_core::closure::mining` |
| `katgpt_rs::salience` | `katgpt_core::salience` |
| `katgpt_rs::trigger_gate` | `katgpt_core::trigger_gate` |
| `katgpt_rs::skill_opt` | `katgpt_core::skill_opt` |

### → katgpt-claim
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::clr` | `katgpt_claim::clr` |
| `katgpt_rs::claim_rubric` | `katgpt_claim::claim_rubric` |

### → katgpt-band
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::band_conditioner` | `katgpt_band::band_conditioner` |
| `katgpt_rs::bckvss` | `katgpt_band::bckvss` |
| `katgpt_rs::collider_pruner` | `katgpt_band::collider_pruner` |
| `katgpt_rs::adaptive_cot_stopper` | `katgpt_band::adaptive_cot_stopper` |

### → katgpt-pruners
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::freq_bandit` | `katgpt_pruners::freq_bandit` |
| `katgpt_rs::interval_pruner` | `katgpt_pruners::interval_pruner` |
| `katgpt_rs::lattice_operad` | `katgpt_pruners::lattice_operad` |
| `katgpt_rs::closure_wire` | `katgpt_pruners::closure_wire` |
| `katgpt_rs::screening` | `katgpt_pruners::screening` |

### → katgpt-attn
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::diagonal_gate` | `katgpt_attn::diagonal_gate` |
| `katgpt_rs::ega_attn` | `katgpt_attn::ega_attn` |
| `katgpt_rs::dash_attn::*` | `katgpt_attn::dash_attn::*` |
| `katgpt_rs::gdn2::*` | `katgpt_attn::gdn2::*` |
| `katgpt_rs::rat_bridge` | `katgpt_attn::rat_bridge` |
| `katgpt_rs::chiaroscuro` | `katgpt_attn::chiaroscuro` |
| `katgpt_rs::funcattn_compose` | `katgpt_attn::funcattn_compose` |
| `katgpt_rs::static_cal` | `katgpt_attn::static_cal` |

### → katgpt-attn-match
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::attn_match` | `katgpt_attn_match` |
| `katgpt_rs::rerank` | `katgpt_attn_match::rerank` |

### → katgpt-kv
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::async_qdq` | `katgpt_kv::async_qdq` |
| `katgpt_rs::cache_prune` | `katgpt_kv::cache_prune` |
| `katgpt_rs::segment_checkpoint` | `katgpt_kv::segment_checkpoint` |
| `katgpt_rs::cs_kv_probe` | `katgpt_kv::cs_kv_probe` |
| `katgpt_rs::kv_share` | `katgpt_kv::kv_share` |
| `katgpt_rs::kvarn` | `katgpt_kv::kvarn` |
| `katgpt_rs::osc_kv` | `katgpt_kv::osc_kv` |
| `katgpt_rs::shard_kv` | `katgpt_kv::shard_kv` |
| `katgpt_rs::still_kv` | `katgpt_kv::still_kv` |
| `katgpt_rs::targeted_precision` | `katgpt_kv::targeted_precision` |
| `katgpt_rs::sp_kv::*` | `katgpt_kv::sp_kv::*` |

### → katgpt-quant
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::hybrid_oct_pq` | `katgpt_quant::hybrid_oct_pq` |
| `katgpt_rs::iso_quant` | `katgpt_quant::iso_quant` |
| `katgpt_rs::octopus` | `katgpt_quant::octopus` |
| `katgpt_rs::planar_quant` | `katgpt_quant::planar_quant` |
| `katgpt_rs::turboquant` | `katgpt_quant::turboquant` |

### → katgpt-spectral
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::peira` | `katgpt_spectral::peira` |
| `katgpt_rs::gauge_invariant` | `katgpt_spectral::gauge_invariant` |
| `katgpt_rs::manifold_power_iter_router` | `katgpt_spectral::manifold_power_iter_router` |
| `katgpt_rs::spectral_retract` | `katgpt_spectral::spectral_retract` |
| `katgpt_rs::hla_eigenbasis` | `katgpt_spectral::hla_eigenbasis` |
| `katgpt_rs::off_principal` | `katgpt_spectral::off_principal` |
| `katgpt_rs::spectral_rewire` | `katgpt_spectral::spectral_rewire` |
| `katgpt_rs::procrustes` | `katgpt_spectral::procrustes` |
| `katgpt_rs::river_valley` | `katgpt_spectral::river_valley` |
| `katgpt_rs::spectral_budget` | `katgpt_spectral::spectral_budget` |
| `katgpt_rs::spectral_concentration` | `katgpt_spectral::spectral_concentration` |
| `katgpt_rs::stiff_anomaly` | `katgpt_spectral::stiff_anomaly` |
| `katgpt_rs::spectralquant::*` | `katgpt_spectral::*` |

### → katgpt-speculative
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::ilc` | `katgpt_speculative::distill::ilc` |
| `katgpt_rs::trd` | `katgpt_speculative::distill::trd` |
| `katgpt_rs::fold` | `katgpt_speculative::fold` |
| `katgpt_rs::precision_aware_draft` | `katgpt_speculative::precision_aware_draft` |
| `katgpt_rs::progressive_mcgs` | `katgpt_speculative::progressive_mcgs` |
| `katgpt_rs::rt_turbo` | `katgpt_speculative::rt_turbo` |
| `katgpt_rs::spec_reconciliation` | `katgpt_speculative::spec_reconciliation` |
| `katgpt_rs::spechop` | `katgpt_speculative::spechop` |

### → katgpt-sparse
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::sparse_compose` | `katgpt_sparse::sparse_compose` |
| `katgpt_rs::sparse_task_vector` | `katgpt_sparse::sparse_task_vector` |
| `katgpt_rs::specialist_projection` | `katgpt_sparse::specialist_projection` |

### → katgpt-transformer
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::mbu` | `katgpt_transformer::mbu` |
| `katgpt_rs::dense_mesh::*` | `katgpt_transformer::dense_mesh::*` |
| `katgpt_rs::swir` | `katgpt_transformer::swir` |
| `katgpt_rs::thinking_cot` | `katgpt_transformer::thinking_cot` |
| `katgpt_rs::{ContiguousWeights, load_ternary_bits}` | `katgpt_transformer::{ContiguousWeights, load_ternary_bits}` |

### → katgpt-tokenizer
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::tokenizer` | `katgpt_tokenizer` |

### → katgpt-ruliology
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::ruliology` | `katgpt_ruliology` |

### → katgpt-validator
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::validator` | `katgpt_validator` |

### → katgpt-proof-cert
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::proof_cert` | `katgpt_proof_cert` |
| `katgpt_rs::conditional_proof` | `katgpt_proof_cert::conditional_proof` |

### → katgpt-deprecated
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::feedback` | `katgpt_deprecated::feedback` |
| `katgpt_rs::unit_distance` | `katgpt_deprecated::unit_distance` |
| `katgpt_rs::alien_sampler` | `katgpt_deprecated::alien_sampler` |

### → katgpt-backend
| Root alias | Leaf path |
|---|---|
| `katgpt_rs::inference_backend` | `katgpt_backend` |
| `katgpt_rs::ane_backend::*` | `katgpt_backend::{AneBackend, AneError}` |
| `katgpt_rs::gpu_backend::*` | `katgpt_backend::GpuBackend` |

## NOT re-exports (genuine root modules — leave alone)

These are `pub mod` declarations with real root code (possibly containing their
own internal re-exports, but the module ITSELF is root):
- `katgpt_rs::pruners` (mod → `src/pruners/mod.rs`)
- `katgpt_rs::speculative` (mod → `src/speculative/mod.rs`)
- `katgpt_rs::types` (mod → `src/types.rs`)
- `katgpt_rs::transformer` (mod → `src/transformer.rs`)
- `katgpt_rs::inference_router` (mod → `src/inference_router.rs`)
- `katgpt_rs::dllm` (mod → `src/dllm.rs`)
- `katgpt_rs::benchmark`, `katgpt_rs::plot`, `katgpt_rs::sleep`, `katgpt_rs::tf_loop`
- `katgpt_rs::hla` (mod with internal re-export from katgpt-core + katgpt-forward)
- `katgpt_rs::gdn2` (mod with internal re-export from katgpt-attn)
- `katgpt_rs::data_probe` (mod with internal re-export from katgpt-core)
- `katgpt_rs::dash_attn` (mod with internal re-export from katgpt-attn)
- `katgpt_rs::dense_mesh` (mod with internal re-export from katgpt-transformer)
- `katgpt_rs::sp_kv` (mod with internal re-export from katgpt-kv)
- `katgpt_rs::spectralquant` (mod with internal re-export from katgpt-spectral)
- `katgpt_rs::distill` (mod with internal re-exports)

These internal-module re-exports are a SEPARATE concern (deeper refactor,
entangled with real root code). This issue only removes the lib.rs-level
`pub use katgpt_*` shims.
