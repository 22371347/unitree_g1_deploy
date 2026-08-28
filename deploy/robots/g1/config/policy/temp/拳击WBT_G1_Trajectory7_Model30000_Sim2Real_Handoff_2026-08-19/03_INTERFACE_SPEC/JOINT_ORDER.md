# Joint Order

## Policy order — authoritative for ONNX observation/action

| Policy idx | Joint | SDK idx |
|---:|---|---:|
| 0 | `left_hip_pitch_joint` | 0 |
| 1 | `right_hip_pitch_joint` | 6 |
| 2 | `waist_yaw_joint` | 12 |
| 3 | `left_hip_roll_joint` | 1 |
| 4 | `right_hip_roll_joint` | 7 |
| 5 | `waist_roll_joint` | 13 |
| 6 | `left_hip_yaw_joint` | 2 |
| 7 | `right_hip_yaw_joint` | 8 |
| 8 | `waist_pitch_joint` | 14 |
| 9 | `left_knee_joint` | 3 |
| 10 | `right_knee_joint` | 9 |
| 11 | `left_shoulder_pitch_joint` | 15 |
| 12 | `right_shoulder_pitch_joint` | 22 |
| 13 | `left_ankle_pitch_joint` | 4 |
| 14 | `right_ankle_pitch_joint` | 10 |
| 15 | `left_shoulder_roll_joint` | 16 |
| 16 | `right_shoulder_roll_joint` | 23 |
| 17 | `left_ankle_roll_joint` | 5 |
| 18 | `right_ankle_roll_joint` | 11 |
| 19 | `left_shoulder_yaw_joint` | 17 |
| 20 | `right_shoulder_yaw_joint` | 24 |
| 21 | `left_elbow_joint` | 18 |
| 22 | `right_elbow_joint` | 25 |
| 23 | `left_wrist_roll_joint` | 19 |
| 24 | `right_wrist_roll_joint` | 26 |
| 25 | `left_wrist_pitch_joint` | 20 |
| 26 | `right_wrist_pitch_joint` | 27 |
| 27 | `left_wrist_yaw_joint` | 21 |
| 28 | `right_wrist_yaw_joint` | 28 |

## Unitree grouped / SDK order used for mapping

| SDK idx | Joint | Policy idx |
|---:|---|---:|
| 0 | `left_hip_pitch_joint` | 0 |
| 1 | `left_hip_roll_joint` | 3 |
| 2 | `left_hip_yaw_joint` | 6 |
| 3 | `left_knee_joint` | 9 |
| 4 | `left_ankle_pitch_joint` | 13 |
| 5 | `left_ankle_roll_joint` | 17 |
| 6 | `right_hip_pitch_joint` | 1 |
| 7 | `right_hip_roll_joint` | 4 |
| 8 | `right_hip_yaw_joint` | 7 |
| 9 | `right_knee_joint` | 10 |
| 10 | `right_ankle_pitch_joint` | 14 |
| 11 | `right_ankle_roll_joint` | 18 |
| 12 | `waist_yaw_joint` | 2 |
| 13 | `waist_roll_joint` | 5 |
| 14 | `waist_pitch_joint` | 8 |
| 15 | `left_shoulder_pitch_joint` | 11 |
| 16 | `left_shoulder_roll_joint` | 15 |
| 17 | `left_shoulder_yaw_joint` | 19 |
| 18 | `left_elbow_joint` | 21 |
| 19 | `left_wrist_roll_joint` | 23 |
| 20 | `left_wrist_pitch_joint` | 25 |
| 21 | `left_wrist_yaw_joint` | 27 |
| 22 | `right_shoulder_pitch_joint` | 12 |
| 23 | `right_shoulder_roll_joint` | 16 |
| 24 | `right_shoulder_yaw_joint` | 20 |
| 25 | `right_elbow_joint` | 22 |
| 26 | `right_wrist_roll_joint` | 24 |
| 27 | `right_wrist_pitch_joint` | 26 |
| 28 | `right_wrist_yaw_joint` | 28 |

## Mapping arrays

```python
POLICY_TO_SDK = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
SDK_TO_POLICY = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
```

Semantics:

```python
q_policy[i] = q_sdk[POLICY_TO_SDK[i]]
```

To send a policy-space target to SDK order:

```python
q_sdk = zeros(29)
for policy_i, sdk_i in enumerate(POLICY_TO_SDK):
    q_sdk[sdk_i] = q_target_policy[policy_i]
```

**Do not send the ONNX 29D action vector directly as SDK joint positions.**
