# Sim2Real Deployment Checklist

> 目标：在给真机发送任何较大动作前，把接口错误和 frame/order 错误排除掉。

## A. 文件与模型

- [ ] 校验 `99_MANIFEST/SHA256SUMS.txt`
- [ ] 部署使用 `01_POLICY/policy.onnx`
- [ ] 确认模型是 `model_30000.pt` 对应导出，不是旧 6000/46500
- [ ] Reference 使用 `02_REFERENCE_MOTION/motion.npz`
- [ ] 确认 motion 为 50 Hz / 159 frames

## B. Joint interface — 必须逐项验证

- [ ] 真机 SDK joint 名字与 `JOINT_INTERFACE.csv` 对齐
- [ ] 验证 policy order ↔ SDK order mapping
- [ ] 验证每个 joint 正方向
- [ ] 验证 encoder 单位是 rad
- [ ] 验证 velocity 单位是 rad/s
- [ ] 验证 `q_default` 使用训练 nominal pose，而不是 trajectory7 stance
- [ ] 验证 raw action 到 q_des 的 scale
- [ ] 真机侧设置合理 joint position / velocity / torque safety limits

## C. Observation

- [ ] observation 总长度恰好 154
- [ ] reference q/dq 为 policy order
- [ ] torso reference quaternion 与当前 torso quaternion 都使用同一 `wxyz` convention
- [ ] `motion_anchor_ori_b` 按文档计算 6D rotation representation
- [ ] base angular velocity/gyro frame 与训练一致
- [ ] `joint_pos_rel = q_policy - q_default_policy`
- [ ] `joint_vel` 为 policy order
- [ ] `last_action` 是上一周期 raw 29D policy output；首次置 0
- [ ] 不额外拼接 base linear velocity
- [ ] 不额外拼接 anchor position
- [ ] 不对 ONNX 输入做第二次 normalization

## D. Timing

- [ ] policy loop 50 Hz / 20 ms
- [ ] motion `time_step` 与 policy tick 一一对应
- [ ] `time_step` 从 0 开始
- [ ] frame 158 后先 hold，不自动 loop
- [ ] 监控实际 loop jitter、deadline miss 和通信 latency

## E. First bring-up

- [ ] 急停可用
- [ ] 机器人处于安全支撑/吊架/保护条件
- [ ] 先做低幅度单关节 mapping/sign test
- [ ] 再做全身 `q_default` hold test
- [ ] 再做 safe stand → trajectory7 stance 平滑 transition
- [ ] stance 稳定后才 enable policy
- [ ] 第一次只跑 motion 前几帧/低风险区间
- [ ] 监控 position error、velocity、torque/current、IMU tilt
- [ ] 任一 joint 方向/顺序异常立即停机

## F. Punch run

- [ ] trajectory7 stance 首帧正确
- [ ] 进入 punch 无瞬时大跳
- [ ] waist/肩/腕方向正确
- [ ] 下肢不出现 joint-order 交叉
- [ ] frame 158 后能回 trajectory7 stance
- [ ] hold 末帧稳定后再扩大测试范围
