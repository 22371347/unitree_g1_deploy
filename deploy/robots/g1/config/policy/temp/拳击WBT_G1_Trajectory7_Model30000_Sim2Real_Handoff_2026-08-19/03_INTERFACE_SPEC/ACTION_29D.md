# Action Contract — 29D

## Raw policy output

ONNX output:

```text
actions: [1, 29]
```

Order is the policy joint order in `JOINT_ORDER.md`.

## Training action transformation

Collected training config:

```text
JointPositionAction
use_default_offset = True
clip = None
```

Deployment target:

```python
q_des_policy = q_default_policy + action_scale_policy * raw_action
```

`q_default_policy` and `action_scale_policy` are listed per joint in
`JOINT_INTERFACE.csv`.

Then reorder the resulting `q_des_policy` into the hardware SDK order.

## Important

- trajectory7 stance is the motion reference stance, not the action offset;
- do not add trajectory7 stance to policy actions;
- do not reorder the raw action before applying the policy-space per-joint scale;
- hardware safety clamps should be implemented independently and tested
  conservatively before full motion execution.

## Training PD metadata

`JOINT_INTERFACE.csv` includes training Kp/Kd, effort limits and velocity limits.
These are useful for reproducing the training actuator behavior, but they are
**not an instruction to bypass the real robot's safety/controller layer**.

The Sim2Real team should reconcile these values with the official G1 low-level
controller and hardware limits.
