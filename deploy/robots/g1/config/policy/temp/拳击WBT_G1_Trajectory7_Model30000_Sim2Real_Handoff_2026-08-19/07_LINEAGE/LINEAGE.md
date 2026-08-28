# Lineage

## Current model

```text
trajectory7 new default stance
        ↓
M_ShortMove12 punch rebuilt
        ↓
m_shortmove12_frames280_350_trajectory7_stance_v1
        ↓
Tracking-Flat-G1-Wo-State-Estimation-v0
        ↓
scratch training, resume=false
        ↓
model_30000.pt
        ↓
policy.onnx
```

Training run:

```text
2026-08-18_19-54-59_trajectory7_punch_scratch_30001
```

## Historical context

The project previously had a highly stable old-default-pose policy:

```text
model_46500.pt
Tracking-Flat-G1-Wo-State-Estimation-Robust-v0
```

That historical checkpoint is NOT the source of the current model weights.

Its robustness design may be useful for later training, but the present package
is intentionally the new-stance scratch baseline.
