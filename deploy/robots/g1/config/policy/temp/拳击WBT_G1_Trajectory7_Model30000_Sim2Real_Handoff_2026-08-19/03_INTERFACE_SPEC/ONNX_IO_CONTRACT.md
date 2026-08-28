# ONNX I/O Contract

The project exporter creates two inputs:

```text
obs        shape [1, 154]
time_step  shape [1, 1]
```

and seven outputs:

```text
actions         [1, 29]
joint_pos       [1, 29]
joint_vel       [1, 29]
body_pos_w      [1, 14, 3]
body_quat_w     [1, 14, 4]
body_lin_vel_w  [1, 14, 3]
body_ang_vel_w  [1, 14, 3]
```

`time_step` is converted to integer inside the export graph and clamped to the
last motion frame.

Selected reference bodies, in ONNX metadata order:

```text
pelvis
left_hip_roll_link
left_knee_link
left_ankle_roll_link
right_hip_roll_link
right_knee_link
right_ankle_roll_link
torso_link
left_shoulder_roll_link
left_elbow_link
left_wrist_yaw_link
right_shoulder_roll_link
right_elbow_link
right_wrist_yaw_link
```

Anchor body: `torso_link`.

## Recommendation

Use the ONNX outputs `joint_pos` and `joint_vel` as the reference command at the
same `time_step` used for inference. This makes the deployed reference come from
the same exported artifact as the Actor.

The external `motion.npz` is kept for replay, validation and debugging.
