# Validation Report

## Result

**Core package checks: PASS**

### Checkpoint

- `checkpoint["iter"] = 30000`
- Actor shapes:
  - `(512, 154)`
  - `(256, 512)`
  - `(128, 256)`
  - `(29, 128)`
- Critic shapes:
  - `(512, 286)`
  - `(256, 512)`
  - `(128, 256)`
  - `(1, 128)`

### ONNX ↔ checkpoint identity

The raw float32 byte sequences of all 8 actor parameter tensors from
`model_30000.pt` were searched inside `policy.onnx`.

Result:

```text
actor.0.weight  PASS
actor.0.bias    PASS
actor.2.weight  PASS
actor.2.bias    PASS
actor.4.weight  PASS
actor.4.bias    PASS
actor.6.weight  PASS
actor.6.bias    PASS
```

Therefore the exported ONNX Actor weights match this `model_30000.pt`
checkpoint, rather than an older checkpoint such as model_6000.

### Motion

- fps: `50`
- frames: `159`
- joint_pos: `(159, 29)`
- joint_vel: `(159, 29)`
- body_pos_w: `(159, 30, 3)`
- body_quat_w: `(159, 30, 4)`
- duration: `3.18 s`

Trajectory7 stance consistency:

```text
max |motion[0].joint_pos  - stance| = 0.000e+00
max |motion[-1].joint_pos - stance| = 0.000e+00
```

Generated CSV consistency:

```text
CSV shape = (159, 36)
root columns = 7
joint columns = 29 (grouped / SDK order)
max error after grouped→policy remap vs motion.npz = 0.000e+00
```

### ONNX metadata observed

```text
observation_names = command,motion_anchor_ori_b,base_ang_vel,joint_pos,joint_vel,actionsrB
anchor_body_name  = torso_link
```

Joint names, training Kp/Kd, nominal default joint positions, action scales,
anchor body and selected body names are embedded in ONNX metadata.

## Known packaging limitation

The newly collected input archive had an empty `05_ROBOT_RESOURCE` directory.
This handoff supplements one historical `g1_29dof_unitree_official.xml` from the
previous project handoff for naming/reference only. Its referenced STL meshes
are not included.

For real deployment, use the Sim2Real team's current official Unitree G1 29DoF
robot description/SDK and validate its ordering/signs against `JOINT_INTERFACE.csv`.
