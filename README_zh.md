# Unitree RL Mjlab


## ✳️ 概述

Unitree RL Mjlab 是一个基于 [mjlab](https://github.com/mujocolab/mjlab.git) 构建的强化学习项目，
使用 MuJoCo 作为物理仿真后端，当前支持 Unitree Go2, A2, As2, G1, R1, H1_2 和 H2 机器人。

Mjlab 结合了 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的成熟高层 API 与 
[MuJoCo](https://github.com/google-deepmind/mujoco_warp) 的高精度物理引擎，
为强化学习机器人研究与 Sim-to-Real（仿真到实机） 部署提供了一个轻量化、模块化的框架。

<div align="center">

| <div align="center">  MuJoCo </div>                                                                                                                                           | <div align="center"> Physical </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>


## 📦 安装配置

安装和配置步骤请参考 [setup.md](doc/setup_zh.md)


## 🔁 流程概览

使用强化学习实现机器人运动控制的基本流程如下：

`训练` → `仿真验证` → `仿真到实机`

- **训练**: 在 MuJoCo 模拟环境中让机器人与环境交互，并通过奖励函数最大化学习策略。
- **仿真验证**: 加载训练好的策略进行回放，验证策略行为是否符合预期。
- **仿真到实机**: 将策略部署到物理机器人上，实现真实环境中的运动控制。


## 🛠️ 使用指南

### 1. 速度跟踪训练

运行以下命令进行速度跟踪训练：

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

多 GPU 训练：使用 --gpu-ids 扩展到多块 GPU：

```bash
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

- 第一个参数(如 Mjlab-Velocity-Flat-Unitree-G1)为必选参数，确定要启用的训练环境。可选：
  - Unitree-Go2-Flat
  - Unitree-G1-Flat
  - Unitree-G1-23Dof-Flat
  - Unitree-H1_2-Flat
  - Unitree-A2-Flat
  - Unitree-R1-Flat

> [!NOTE]
> 更多有关详细说明，请参阅 mjlab 文档
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

### 2. 动作模仿训练

训练 Unitree G1 模仿参考动作序列。

<div style="margin-left: 20px;">

#### 2.1 准备动作文件

将准备好的 csv 格式的动作文件保存在 mjlab/motions/g1/ 目录下，执行下面的指令将其转为训练可用的 npz 文件：

```bash
python scripts/csv_to_npz.py \
--input-file src/assets/motions/g1/dance1_subject2.csv \
--output-name dance1_subject2.npz \
--input-fps 30 \
--output-fps 50 \
--robot g1 # g1 or g1_23dof
```

**npz文件默认保存路径为**：`src/motions/g1/...`

#### 2.2 训练

确保有可用的npz文件之后，执行以下指令进行训练：

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096
```

可用任务:
  - Unitree-G1-Tracking-No-State-Estimation
  - Unitree-G1-23Dof-Tracking-No-State-Estimation

</div>

> [!NOTE]
> 有关动作模仿训练的详细说明，请参阅BeyondMimic 文档
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### ⚙️  参数说明
- `--env.scene`: 仿真场景配置，包括环境数量（num_envs）、物理仿真步长、地面类型、重力、随机扰动等参数。
- `--env.observations`: 观测空间配置，控制训练时输入到策略网络的状态信息，如关节位置、速度、IMU等内容。
- `--env.rewards`: 奖励函数配置，定义每步训练时的优化目标。
- `--env.commands`: 控制命令配置，用于生成训练时随机或指定的速度 / 姿态 / 动作指令。
- `--env.terminations`: 终止条件配置，定义训练 episode 的结束条件。
- `--agent.seed`: 训练随机种子，用于结果复现，不同 seed 会导致策略略有差异。
- `--agent.resume`: 是否从上次中断的 checkpoint 继续训练。 设置为 True 时，会自动加载最近一次保存的 .pt 模型文件。
- `--agent.policy`: 策略网络结构配置，例如 MLP 层数、隐藏维度、激活函数等。
- `--agent.algorithm`: 强化学习算法配置。可设置优化超参数，如学习率、批量大小、GAE λ 等。

**默认保存训练结果**：`logs/rsl_rl/<robot>_(velocity | tracking)/<date_time>/model_<iteration>.pt`

### 3. 仿真验证

如果想要在 MuJoCo 中查看训练效果，可以运行以下命令：

查看速度跟踪训练效果：
```bash
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

查看动作模仿训练效果：
```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --checkpoint_file=logs/rsl_rl/g1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**说明**：

- 训练时在每次保存模型时会同步导出 policy.onnx 文件在同层目录下，可用于实物部署。

**效果**：

| Go2                              | G1                             | H1_2                               | G1_mimic                          |
|----------------------------------|--------------------------------|------------------------------------|-----------------------------------|
| ![go2](doc/gif/go2-velocity.gif) | ![g1](doc/gif/g1-velocity.gif) | ![h1_2](doc/gif/h1_2-velocity.gif) | ![g1_mimic](doc/gif/g1-mimic.gif) |

### 4. 策略部署与测试指南

> 下面以 G1 为例，所有路径均以**仓库根目录**为基准。先进入仓库根目录（下称 `<REPO>`）：
> ```bash
> cd <REPO>   # 例如 cd ~/g1_ws/unitree_rl_mjlab
> ```

#### 4.1 注册新策略（配置文件修改指南）

一个待部署的 ONNX 策略要接入本框架，一般需要完成以下 5 步：

> 📌 **工作方式**：默认 `config.yaml` 只保留当前核心策略。要接入新策略时，请【复制 `config.yaml`
> 为 `config_<策略名>.yaml`】并在其中新增/修改该策略的状态与配置（用 `--config` 启动），
> 而不是直接修改默认文件——见下方“多配置并行测试”。

1. **准备策略文件**：将训练导出的 `policy.onnx` 与 `policy.onnx.data` 放入
   `deploy/robots/g1/config/policy/<策略名>/exported/`
2. **编写部署配置** `deploy/robots/g1/config/policy/<策略名>/params/deploy.yaml`：
   - `joint_ids_map`：模型关节顺序 → 硬件关节顺序 的重映射（长度 = 模型 DOF）
   - `step_dt`：控制周期（如 `0.02` = 50Hz）
   - `stiffness` / `damping` / `default_joint_pos`：关节刚度、阻尼、默认位置（按**模型顺序**给出）
   - `actions`：动作配置（`target = action * scale + offset`）
   - `observations`：观测项配置，**书写顺序即 ONNX 输入张量的拼接顺序，必须与训练一致**
   - > ⚠️ 关节映射 / 观测顺序不匹配，是部署后动作异常的最常见原因。
3. **注册 FSM 状态**：在 `deploy/robots/g1/config/config.yaml` 的 `FSM._` 中新增状态，指定**唯一 id** 与 `type`：
   - `type: RLBase` —— 无参考轨迹的策略（如速度控制，对应 `State_RLBase`）
   - `type: Mimic` —— 动作模仿策略，需额外提供 `motion_file`（对应 `State_Mimic`）
   - `type: <自定义>` —— 观测逻辑差异较大时，见第 5 步
4. **添加状态配置块**：配置 `transitions`（手柄切换条件）、`policy_dir`；Mimic 类型还需 `motion_file`、`time_start/end`
5. **观测逻辑差异较大时**：在 `deploy/robots/g1/include/` 与 `deploy/robots/g1/src/` 下编写自定义
   `State_<名字>.h/.cpp`（继承 `FSMState`，重写 `enter/run/exit` 与观测计算，用 `REGISTER_FSM` 注册），
   并在 `deploy/robots/g1/main.cpp` 中 `#include` 对应头文件。

> [!TIP]
> `config.yaml` 内已内置完整的注册引导注释与可直接照抄的示例
> （`Velocity` = RLBase 示例，`Fight1` / `AMP` = 策略状态示例）。

#### 多配置并行测试（配置预设）

在 `deploy/robots/g1/config/` 下保存为 `config_<策略名>.yaml`（如 `config_fight.yaml`、`config_AMP.yaml`），
启动时用 `--config` 选择即可——无需复制粘贴，无需重新编译：

```bash
cd deploy/robots/g1/build
./g1_ctrl -n lo -c config_fight1.yaml    # 仿真，fight1（WBT）策略
./g1_ctrl -n lo -c config_AMP.yaml       # 仿真，AMP 策略
./g1_ctrl -c config_fight1.yaml          # 实机（默认 DDS 配置）
```

- 未指定 `--config` 时默认加载 `config.yaml`。
- 各预设共享公共状态（`Passive` / `FixStand` / `Velocity`），仅在启用的策略状态（`Fight1` / `AMP` / …）上不同，
  编辑时请保持公共部分一致。

#### 4.2 Sim2Real：MuJoCo 仿真验证

上真机前先在 MuJoCo 仿真中验证策略。需要两个程序：仿真器 `unitree_mujoco` 与控制程序 `g1_ctrl`。

**编译控制程序**（以 G1 为例）：

```bash
cd deploy/robots/g1
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
```

**编译仿真器**：

```bash
cd simulate
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
```

**运行仿真**（需要连接手柄）：

终端 1 —— 启动仿真器：

```bash
cd simulate/build
./unitree_mujoco
```

终端 2 —— 启动控制程序：

```bash
cd deploy/robots/g1/build
./g1_ctrl -n lo
```

进入后可通过键盘 `a` `b` `c` `d` 切换四个状态进行观测
（`a`=Passive，`b`=FixStand，`c`=Velocity，`d`=Mimic_Dance1_subject2）。

> [!NOTE]
> `-n` / `--network` 指定 DDS 网络接口。仿真时用 `lo`（本机回环）。

#### 4.3 实机部署

> 前置依赖：需先安装通信库
> - [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
> - [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

**4.3.1 开机与调试模式**
- 将机器人悬挂悬空启动，等待进入 `zero-torque`（零力矩）模式
- 在零力矩模式下按手柄 `L2 + R2`，进入 `debug mode`（关节阻尼生效）

**4.3.2 网络连接**
用网线连接 PC 与机器人，配置 IPv4：
- 地址：`192.168.123.222`
- 子网掩码：`255.255.255.0`

机器人默认地址通常为 `192.168.123.164`（以实际为准，可用 `ifconfig` / 宇树文档确认）。

**4.3.3 传输项目到机器人**

```bash
ssh unitree@192.168.123.164        # 或对应的机器人地址
scp -r deploy unitree@192.168.123.164:~/wu/
```



**4.3.4 在机器人上编译并运行**
手柄按下R2+L2、L2+A，进入调试模式
```bash
ssh unitree@192.168.123.164
cd ~/wu/deploy/robots/g1              # 按实际 scp 路径调整
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)

tmux new -s 1
./g1_ctrl
```

拉起后启动
> [!NOTE]
> 在机器人板载计算机内运行 `g1_ctrl` 时无需指定网络参数（使用默认 DDS 配置）。
> 若在 PC 上做仿真则使用 `-n lo`；若 PC 通过网口直连机器人运行，则改用对应网卡名（如 `-n enp5s0`，用 `ifconfig` 查看）。

#### 4.4 手柄操作指南

| 当前状态              | 按键     | 目标状态               | 说明                          |
|----------------------|---------|-----------------------|------------------------------|
| Passive（阻尼模式）    | L2 + Up | FixStand              | 进入力矩模式，调整机器人姿态至触地 |
| FixStand             | R2 + A  | Velocity              | 进入运控模式（速度控制）         |
| Velocity / FixStand  | R1 + A  | 拳击姿态               | 进入拳击姿态动作                 |
| 任意状态              | L2 + B  | Passive               | 回到阻尼模式（**急停**）         |

> 键盘辅助（仿真调试）：`a`=Passive，`b`=FixStand，`c`=Velocity，`d`=自定义动作


## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [mjlab](https://github.com/mujocolab/mjlab.git): 构建训练与运行代码的基础。
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): 用于动作跟踪的通用人形机器人控制框架。
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): 提供 GPU 加速渲染与仿真接口。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大仿真功能。

