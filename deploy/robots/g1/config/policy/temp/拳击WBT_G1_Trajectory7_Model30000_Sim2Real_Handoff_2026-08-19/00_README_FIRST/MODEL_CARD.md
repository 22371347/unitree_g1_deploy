# Model Card

## Identity

| Field | Value |
|---|---|
| Model | `model_30000.pt` |
| ONNX | `policy.onnx` |
| Training run | `2026-08-18_19-54-59_trajectory7_punch_scratch_30001` |
| Task | `Tracking-Flat-G1-Wo-State-Estimation-v0` |
| Iteration | `30000` |
| Policy obs | 154 |
| Critic obs | 286 |
| Action | 29 |
| Actor | 154 → 512 → 256 → 128 → 29, ELU |
| Critic | 286 → 512 → 256 → 128 → 1, ELU |
| Reference | `m_shortmove12_frames280_350_trajectory7_stance_v1` |
| Motion | 50 Hz, 159 frames |
| Anchor | `torso_link` |

## Training lineage

这是 trajectory7 新 stance 的 **scratch baseline**。

`agent.yaml` 显示：

- `resume: false`
- `max_iterations: 30001`
- `save_interval: 500`
- `empirical_normalization: true`

本 checkpoint 不应描述成历史 Robust 模型的 continuation。

历史 `model_46500.pt / Tracking-Flat-G1-Wo-State-Estimation-Robust-v0`
只是旧默认姿态下的稳定性方法参考，不属于当前模型权重 lineage。

## Canonical deployment artifact

推荐真机部署使用 `policy.onnx`。

原因：

- Actor 参数已验证与 `model_30000.pt` 完全匹配；
- ONNX 导出器同时暴露 reference motion outputs；
- exporter 会包装 runner 提供的 observation normalizer；
- 可以从 ONNX metadata 读取 joint names、default joint pos、gains、action scale、body names。

不要仅凭 `.pt` 自己重新拼 observation normalization，除非完整复现训练/导出 stack。
