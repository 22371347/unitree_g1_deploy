# my_policy —— 空白策略模板

本目录是一个**空白的策略模板**，配合教程（README / `doc/ppt` 组会 PPT）使用。

## 目录结构

```
config/policy/my_policy/
├── exported/
│   ├── README.md        # 说明（本目录放置策略文件）
│   ├── policy.onnx      # ⬅ 把你的策略网络放这里
│   └── policy.onnx.data # ⬅ 归一化参数（若有）
└── params/
    └── deploy.yaml      # ⬅ 部署配置模板（基于 RLBase 速度类，含 TODO 注释）
```

## 使用方法（三步）

1. **放置策略文件**：把训练导出的 `policy.onnx`（+ `policy.onnx.data`）放入 `exported/`
2. **填写配置**：按你的模型修改 `params/deploy.yaml` 中的
   `joint_ids_map / step_dt / stiffness / damping / default_joint_pos / actions / observations`
   （观测项书写顺序 = ONNX 输入拼接顺序，务必与训练一致）
3. **运行**（状态已在 `config/config_mypolicy.yaml` 注册好）：
   ```bash
   ./g1_ctrl -n lo -c config_mypolicy.yaml   # 仿真
   ./g1_ctrl -c config_mypolicy.yaml          # 实机
   ```

> 参考样例：
> - 速度控制（RLBase）：`config/policy/velocity/`
> - 动作模仿（Mimic）：`config/policy/mimic/dance1_subject2/`
