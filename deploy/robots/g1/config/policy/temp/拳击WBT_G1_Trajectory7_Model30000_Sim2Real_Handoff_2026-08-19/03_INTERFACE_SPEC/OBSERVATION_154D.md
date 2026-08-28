# Observation Contract — 154D

The exact policy observation term order stored in the exported ONNX metadata is:

```text
command,motion_anchor_ori_b,base_ang_vel,joint_pos,joint_vel,actions
```

## Layout

| Slice | Dim | Meaning |
|---|---:|---|
| `0:29` | 29 | reference joint position `ref_q` |
| `29:58` | 29 | reference joint velocity `ref_dq` |
| `58:64` | 6 | torso/anchor orientation error 6D |
| `64:67` | 3 | base angular velocity / body-frame gyro |
| `67:96` | 29 | `q_policy - q_default_policy` |
| `96:125` | 29 | joint velocity in policy order |
| `125:154` | 29 | previous raw policy action |

Total: `58 + 6 + 3 + 29 + 29 + 29 = 154`.

## Reference command

Training `MotionCommand.command` is exactly:

```python
torch.cat([reference_joint_pos, reference_joint_vel], dim=1)
```

Both are in **policy joint order**.

## motion_anchor_ori_b

Training computes the desired anchor orientation relative to the current robot
anchor (`torso_link`) and converts the relative quaternion to a rotation matrix.

Equivalent convention used by the aligned project Sim2Sim reference:

```python
q_rel = inverse(current_torso_quat_wxyz) * reference_torso_quat_wxyz
ori6 = rotation_matrix(q_rel)[:, :2].reshape(-1)
```

Result order is:

```text
[R00, R01, R10, R11, R20, R21]
```

Do not replace this with Euler-angle error.

## base_ang_vel

The training term is IsaacLab `base_ang_vel`. Deployment must match its frame
convention. The project's aligned MuJoCo implementation used the local 3-axis
gyro as this term.

Validate IMU axis signs and body frame convention on hardware before enabling
full-body actions.

## joint_pos

This is **relative to the training nominal default pose**:

```python
joint_pos_rel = q_policy - q_default_policy
```

It is NOT relative to trajectory7 stance.

## joint_vel

29D joint velocity in policy order, rad/s.

## actions

Previous cycle's **raw ONNX action output**, 29D.

Initialize to zeros on policy activation.

## Normalization

`agent.yaml` says `empirical_normalization: true`.

The ONNX export path passes the runner normalizer into the exporter, so treat
`policy.onnx` as the canonical inference artifact and feed the raw 154D
observation described above.

**Do not apply a second external normalization unless the ONNX graph has first
been inspected and the deployment stack is intentionally replacing the
embedded preprocessing.**
