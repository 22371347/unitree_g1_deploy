# WBT G1 Trajectory7 Model30000 — Sim2Real Handoff

## 先看结论

这是 **trajectory7 新默认姿态**下重新从 scratch 训练得到的 G1 Whole-Body Tracking 打拳 baseline：

- Checkpoint: `model_30000.pt`
- Deployment model: `policy.onnx`
- Training task: `Tracking-Flat-G1-Wo-State-Estimation-v0`
- Policy observation: **154D**
- Action: **29D**
- Policy rate: **50 Hz**
- Reference motion: **50 Hz, 159 frames, 3.18 s**
- Anchor body: `torso_link`
- 这是新 stance baseline，**不是**历史 `model_46500 / Robust-v0`。

## 部署时优先使用哪个文件

**优先使用 `01_POLICY/policy.onnx`。**

`model_30000.pt` 主要用于训练恢复、复现和审计。ONNX 是本交接包的 canonical deployment artifact，因为导出器会把 motion reference 接口和运行时 normalizer（如果 runner 提供）一起包装进导出图。

## 最危险的两个坑

### 1. Policy joint order != Unitree grouped / SDK order

不要直接把 29D policy 向量按 SDK 0..28 原样发送。

必须看：

- `03_INTERFACE_SPEC/JOINT_INTERFACE.csv`
- `03_INTERFACE_SPEC/JOINT_ORDER.md`

映射定义：

```text
q_policy[i] = q_sdk[POLICY_TO_SDK[i]]

POLICY_TO_SDK =
[0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
```

### 2. trajectory7 stance 不是 action 的 default offset

训练 action 使用 `use_default_offset=True`。

也就是说 target 逻辑是：

```text
q_des_policy = q_default_policy + action_scale * raw_action
```

其中 `q_default_policy` 是训练机器人配置里的 nominal default pose，**不是 trajectory7 stance**。

trajectory7 是 reference motion 的首尾姿态。两者不要混。

## Policy observation

154D 顺序固定为：

```text
[0:58]    command = reference_joint_pos(29) + reference_joint_vel(29)
[58:64]   motion_anchor_ori_b (6D)
[64:67]   base_ang_vel / body-frame gyro (3D)
[67:96]   joint_pos_rel = q - q_default (29D)
[96:125]  joint_vel (29D)
[125:154] last_raw_action (29D)
```

本模型 **不使用** `base_lin_vel`，也 **不使用** `motion_anchor_pos_b`。

详见 `03_INTERFACE_SPEC/OBSERVATION_154D.md`。

## ONNX 接口

导出器接口：

```text
Inputs
  obs       [1, 154]
  time_step [1, 1]

Outputs
  actions        [1, 29]
  joint_pos      [1, 29]
  joint_vel      [1, 29]
  body_pos_w     [1, 14, 3]
  body_quat_w    [1, 14, 4]
  body_lin_vel_w [1, 14, 3]
  body_ang_vel_w [1, 14, 3]
```

建议 Sim2Real 直接使用 ONNX 输出的 `joint_pos/joint_vel/body_*` 作为当前 `time_step` 的 reference，避免外部 motion 和 ONNX reference 错帧。

## Timing

训练：

```text
physics dt = 0.005 s
decimation = 4
policy dt = 0.020 s
policy frequency = 50 Hz
motion frequency = 50 Hz
```

因此正常播放时每个 policy tick：

```text
time_step += 1
```

范围 `0..158`。动作播放结束后建议先 **hold frame 158**，不要默认 loop。

## 首次真机前

必须先完成 `DEPLOYMENT_CHECKLIST.md`，尤其是：

1. 验证 29 个关节名字、顺序、方向；
2. 验证 `q_default` 和 action scale；
3. 验证 IMU gyro / torso quaternion 的 frame convention；
4. 先在安全支撑/吊架/急停就绪条件下做低风险 bring-up；
5. 先完成 safe stand -> trajectory7 stance 的平滑过渡，再 enable policy。

## 机器人资源说明

用户本次收集包里的 `05_ROBOT_RESOURCE` 实际为空。

本包额外附了之前 handoff 中保存的
`g1_29dof_unitree_official_REFERENCE_ONLY.xml`，仅供 joint/body 命名和 Sim2Sim 对照。
该 XML 引用了外部 STL，**不是自包含机器人资源，也不能代替团队正式的 Unitree G1 29DoF SDK/URDF/MJCF**。

真机部署必须以 Sim2Real 团队当前正式 robot description / SDK 为准，再对照本包的 joint interface 做一致性检查。
