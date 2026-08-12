# Unitree RL Mjlab


## ✳️ Overview
Unitree RL Mjlab is a reinforcement learning project built upon the
[mjlab](https://github.com/mujocolab/mjlab.git), using MuJoCo as its 
physics simulation backend, currently supporting Unitree Go2, A2, As2, G1, R1, H1_2 and H2.

Mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API
with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp)
physics to provide lightweight, modular abstractions for RL robotics research
and sim-to-real deployment.

<div align="center">

| <div align="center">  MuJoCo </div>                                                                                                                                           | <div align="center"> Physical </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>


## 📦 Installation and Configuration

Please refer to [setup.md](doc/setup_en.md) for installation and configuration steps.


## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Real`

- **Train**: The agent interacts with the MuJoCo simulation and optimizes policies through reward maximization.
- **Play**: Replay trained policies to verify expected behavior.
- **Sim2Real**: Deploy trained policies to physical Unitree robots for real-world execution.


## 🛠️ Usage Guide

### 1. Velocity Tracking Training

Run the following command to train a velocity tracking policy:

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

Multi-GPU Training: Scale to multiple GPUs using --gpu-ids:

```bash
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

- The first argument (e.g., Mjlab-Velocity-Flat-Unitree-G1) specifies the training task.
Available velocity tracking tasks:
  - Unitree-Go2-Flat
  - Unitree-G1-Flat
  - Unitree-G1-23Dof-Flat
  - Unitree-H1_2-Flat
  - Unitree-A2-Flat
  - Unitree-R1-Flat

> [!NOTE]
> For more details, refer to the mjlab documentation:
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

### 2. Motion Imitation Training

Train a Unitree G1 to mimic reference motion sequences.

<div style="margin-left: 20px;">

#### 2.1 Prepare Motion Files

Prepare csv motion files in mjlab/motions/g1/ and convert them to npz format:

```bash
python scripts/csv_to_npz.py \
--input-file src/assets/motions/g1/dance1_subject2.csv \
--output-name dance1_subject2.npz \
--input-fps 30 \
--output-fps 50 \
--robot g1 # g1 or g1_23dof
```

**npz files will be stored at:**：`src/motions/g1/...`

#### 2.2 Training

After generating the NPZ file, launch imitation training:

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096
```

Available tasks:
  - Unitree-G1-Tracking-No-State-Estimation
  - Unitree-G1-23Dof-Tracking-No-State-Estimation

</div>

> [!NOTE]
> For detailed motion imitation instructions, refer to the BeyondMimic documentation:
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### ⚙️  Parameter Description
- `--env.scene`: simulation scene configuration (e.g., num_envs, dt, ground type, gravity, disturbances)
- `--env.observations`: observation space configuration (e.g., joint state, IMU, commands, etc.)
- `--env.rewards`: reward terms used for policy optimization
- `--env.commands`: task commands (e.g., velocity, pose, or motion targets)
- `--env.terminations`: termination conditions for each episode
- `--agent.seed`: random seed for reproducibility
- `--agent.resume`: resume from the last saved checkpoint when enabled
- `--agent.policy`: policy network architecture configuration
- `--agent.algorithm`: reinforcement learning algorithm configuration (PPO, hyperparameters, etc.)

**Training results are stored at**：`logs/rsl_rl/<robot>_(velocity | tracking)/<date_time>/model_<iteration>.pt`

### 3. Simulation Validation

To visualize policy behavior in MuJoCo:

Velocity tracking:
```bash
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

Motion imitation:
```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --checkpoint_file=logs/rsl_rl/g1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**Note**：

- During training, policy.onnx and policy.onnx.data are also exported for deployment onto physical robots.

**Visualization**：

| Go2                              | G1                             | H1_2                               | G1_mimic                          |
|----------------------------------|--------------------------------|------------------------------------|-----------------------------------|
| ![go2](doc/gif/go2-velocity.gif) | ![g1](doc/gif/g1-velocity.gif) | ![h1_2](doc/gif/h1_2-velocity.gif) | ![g1_mimic](doc/gif/g1-mimic.gif) |

### 4. 策略部署与测试指南 (Deployment & Testing Guide)

> 下面以 G1 为例，所有路径均以**仓库根目录**为基准。先进入仓库根目录（下称 `<REPO>`）：
> ```bash
> cd <REPO>   # 例如 cd ~/g1_ws/unitree_rl_mjlab
> ```

#### 4.1 注册新策略（配置文件修改指南）

一个待部署的 ONNX 策略要接入本框架，一般需要完成以下 5 步：

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
> `config.yaml` 内已内置完整的注册引导注释与两个可直接照抄的示例
> （`Velocity` = RLBase 示例，`Mimic_Dance1_subject2` = Mimic 示例）。

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
scp -r deploy unitree@192.168.123.164:~/
```

**4.3.4 在机器人上编译并运行**

```bash
ssh unitree@192.168.123.164
cd ~/deploy/robots/g1              # 按实际 scp 路径调整
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
./g1_ctrl -n lo
```

> [!NOTE]
> 若在机器人板载计算机内运行 `g1_ctrl`，DDS 走本机回环，一般用 `-n lo`；
> 若通过网口与机器人通信（PC 直连运行），则改用对应网卡名（如 `-n enp5s0`，用 `ifconfig` 查看）。

#### 4.4 手柄操作指南

| 当前状态 | 按键 | 目标状态 | 说明 |
|---------|------|---------|------|
| Passive（阻尼模式） | `L2 + Up` | FixStand | 进入力矩/站立模式，调整机器人姿态至触地 |
| FixStand | `R2 + A` | Velocity | 进入运控模式（速度控制） |
| Velocity / FixStand | `R1 + A` | Mimic_Dance1_subject2 | 进入自定义动作（建议先取下背后挂钩） |
| 任意状态 | `L2 + B` | Passive | 回到阻尼模式（**急停**） |

> 键盘辅助（仿真调试）：`a`=Passive，`b`=FixStand，`c`=Velocity，`d`=Mimic_Dance1_subject2

**Deployment Results**：

| Go2                                                    | G1                                                    | H1_2           | G1_mimic                                           |
|--------------------------------------------------------|-------------------------------------------------------|----------------|----------------------------------------------------|
| <img src="doc/gif/go2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-velocity-real.gif" width="300"/> | <img src="doc/gif/h1_2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-mimic-real.gif" width="300"/> |


## 🎉  Acknowledgements

This project would not be possible without the contributions of the following repositories:

- [mjlab](https://github.com/mujocolab/mjlab.git): training and execution framework
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): versatile humanoid motion tracking framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): reinforcement learning algorithm implementation
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): GPU-accelerated rendering and simulation interface
- [mujoco](https://github.com/google-deepmind/mujoco.git): high-fidelity rigid-body physics engine
