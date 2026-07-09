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

## See also

- [`../inference/kv_compression.md`](../inference/kv_compression.md) — the KV-compression research that the spectral-quant technique builds on
- [`../feature_catalog/opt_in_features.md`](../feature_catalog/opt_in_features.md) — feature gates for the opt-in adaptation techniques
