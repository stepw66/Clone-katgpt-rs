# Calibration — Probes, Gates, and Confidence Calibration

> **What we sell here.** The primitives that make a frozen model's outputs
> trustworthy: confidence calibration, causal head-importance fusion, injected-
> memory faithfulness probing, per-tick emit gating, and the sigmoid-not-softmax
> design principle that runs through all of them.

## Fusion map — calibration composition

```
   universality_class_escape.md (sigmoid-not-softmax: the shared gate design)
        │
        ▼
   cce_moderator.md (calibrated confidence at the token level)
        │
        ├── causal_head_importance.md (which attention heads carry causal signal)
        └── faithfulness_probe.md (did the injected memory actually change the answer?)
              │
              ▼
        salience_tri_gate.md (per-tick Speak / Silent / Delegate)
```

| Doc | Role |
|---|---|
| [`cce_moderator.md`](cce_moderator.md) | CCE Moderator — API reference & worked examples |
| [`causal_head_importance.md`](causal_head_importance.md) | CausalHeadImportance — causal head-importance calibration + scale-normalized fusion |
| [`faithfulness_probe.md`](faithfulness_probe.md) | FaithfulnessProbe — causal-intervention diagnostic for injected memory (Plan 278) |
| [`salience_tri_gate.md`](salience_tri_gate.md) | Salience Tri-Gate — per-tick `Speak` / `Silent` / `Delegate` (Plan 303) |
| [`universality_class_escape.md`](universality_class_escape.md) | Sigmoid-not-softmax: the universality-class escape (Research 315, Liu & Gore 2606.25008) |

## See also

- [`../03_memory/engram.md`](../03_memory/engram.md) — the memory primitive that `faithfulness_probe.md` diagnoses
- [`../10_audits/claim_rubric_audit.md`](../10_audits/claim_rubric_audit.md) — audit tying calibration claims to `Claim` fixtures
