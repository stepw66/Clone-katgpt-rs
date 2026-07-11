# Adaptation — Modelless Adaptation & Distillation

> **What we sell here.** How a frozen base model is reshaped at runtime without
> gradient descent: LoRA-style adapters, weight merging, calibrated KV
> quantization, and distillation-free capability transfer.

## Docs

| Doc | Role |
|---|---|
| [`model_adaptation.md`](model_adaptation.md) | Survey of model adaptation techniques (LoRA, merge, spectral-quant KV, ELF SDE, CNA steering) |
| [`lucebox_techniques.md`](lucebox_techniques.md) | Advanced techniques — TurboQuant → SpectralQuant migration, PFlash block-sparse prefill |
| [`peira_distillation.md`](peira_distillation.md) | PEIRA — modelless distillation (feature `peira_distill`) |
| [`tilr_subspace_family.md`](tilr_subspace_family.md) | Subspace-projection family — TILR alignment-gated correction (Plan 425), cross-referencing `subspace_steering`/`spectral_rewire`/`river_valley` |

## See also

- [`../02_inference/kv_compression.md`](../02_inference/kv_compression.md) — the KV-compression research that the spectral-quant technique builds on
- [`../09_feature_catalog/opt_in_features.md`](../09_feature_catalog/opt_in_features.md) — feature gates for the opt-in adaptation techniques
