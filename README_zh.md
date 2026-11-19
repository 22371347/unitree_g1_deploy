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
   - > ⚠️ **历史观测布局**也必须与训练一致：部署 `use_gym_history: true` = 按帧组织
     （`[帧0: 全部项, 帧1: 全部项, ...]`，mjlab `history_ordering="time"`）；`false` = 按项组织
     （`[项0: 全部帧, 项1: 全部帧, ...]`，IsaacLab `concatenate_terms`）。
     布局错误会让策略收到错乱观测却照常推理（无报错）→ 抽搐 / 无法响应。
3. **注册 FSM 状态**：在 `deploy/robots/g1/config/config_<策略名>.yaml`（从 `config.yaml` 复制而来，见下方“多配置并行测试”）的 `FSM._` 中新增状态，指定**唯一 id** 与 `type`：
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

#### 4.1.1 一次性动作策略部署（后空翻 backflip 示例，State_Mimic 播放式）⭐

适用场景：**后空翻 / 拳击 / 踢腿** 等「一次性动作」策略。

**两种部署方式**：
- `type: Mimic`（`State_Mimic`，推荐）：仿照 `beyondmimic_nostate`，用 `observation_manager` + `env->step()` 自动构建观测，播放到 `time_end` 自动回 `end_state`。
- 自定义 State（`State_Backflip` / `State_Fight1` / `State_Fight2`）：手动 `build_obs_map`，支持「进入停末帧 + 按 `r` 重播」模式。

**行为模式（State_Mimic 播放式）**：
- 进入状态 → 从头播放动作 → 播放到 `time_end` 自动切回 `end_state`（如 `Velocity`）
- 若需「停末帧 + 重播」，改用自定义 State（见 `State_Backflip`）

**实现要点**（backflip 当前用 `type: Mimic`，`config_backflip.yaml`）：
```yaml
FSM:
  _:
    Backflip:
      id: 5
      type: Mimic          # 用 State_Mimic + observation_manager + env->step()
  Backflip:
    transitions: { Passive: LT + B.on_pressed, FixStand: LT + up.on_pressed, Velocity: RT + A.on_pressed }
    policy_dir: config/policy/backflip
    motion_file: config/policy/backflip/params/motion.npz
    # backflip motion 为 IsaacLab 关节序，腰部在 2/5/8（State_Mimic 默认 12/13/14，需显式覆盖）
    waist_yaw_idx: 2
    waist_roll_idx: 5
    waist_pitch_idx: 8
    # 后空翻全程翻转（倾角远超 57°），禁用 bad_orientation，靠 time_end 结束回 Velocity
    disable_bad_orientation: true
    end_state: Velocity
```
**deploy.yaml 要点**（`config/policy/backflip/params/deploy.yaml`）：
1. 观测用 6 项（`motion_command` 58 + `motion_anchor_ori_b` 6 + `base_ang_vel` 3 + `joint_pos_rel` 29 + `joint_vel_rel` 29 + `last_action` 29 = 154），由 `State_Mimic` 的 `observation_manager` 自动生成——**与手动 `build_obs_map` 相比能排除观测拼序/漏项问题**。
2. `stiffness/damping` 按 **IsaacLab 对称序**（`[40.2,40.2,40.2,99.1,99.1,28.5,...]`），避免左右腿刚度不平衡导致失衡。
3. `actions.joint_ids: [0,6,12,...]` 显式指定模型→硬件映射。

> ⚠️ **重要：motion 的 joint_vel 必须平滑！**
> 后空翻等快速动作的 `csv` 通常不含关节速度，差分得到的 `joint_vel` 含**高频噪声**。AMP 系策略
> （无世界模型）对 `motion_command` 里的 joint_vel 噪声敏感 → 起跳/翻转阶段机器人跟不上 → 输出
> 大动作 → 正反馈**指数爆炸**（act 可到 1e21+，实机表现为剧烈抽搐）。
> 解决：对差分后的 `joint_vel` 做**滑动平均（窗口 5）**再写回 motion.npz（见 `scripts` 或
> `deploy/robots/g1/config/policy/backflip/params/motion_orig_vel.npz` 备份对比）。

> 📌 关节序速查：**backflip 为 IsaacLab 序**（`joint_ids_map = [0,6,12,...]`，腰 2/5/8）；AMP_mjlab 系
> （`fight2` / `amp_recovery`）为 **qpos 序**（恒等映射 `[0..28]`，腰 12/13/14）；WBT 系
> （`fight1` / `fight_right` / `beyondmimic_nostate`）为 **IsaacLab 序**。
> 用错关节序会导致肢体错乱——判断依据：csv 中 knee 位置（qpos 在 col3/9）与 stiffness 分组
> （腿6+腿6+腰3+臂7+臂7）；且 **backflip 的 onnx 为单输出（AMP），无 time_step**。

#### 多配置并行测试（配置预设）

在 `deploy/robots/g1/config/` 下保存为 `config_<策略名>.yaml`（如 `config_fight.yaml`、`config_AMP.yaml`），
启动时用 `--config` 选择即可——无需复制粘贴，无需重新编译：

```bash
cd deploy/robots/g1/build
./g1_ctrl -n lo -c config_fight1.yaml    # 仿真，fight1（WBT）策略
./g1_ctrl -n lo -c config_AMP.yaml       # 仿真，AMP 策略
./g1_ctrl -n lo -c config_php.yaml       # 仿真，PHP 感知跑酷策略
./g1_ctrl -c config_fight1.yaml          # 实机（默认 DDS 配置）
```

- 未指定 `--config` 时默认加载 `config.yaml`。
- 各预设共享公共状态（`Passive` / `FixStand` / `Velocity`），仅在启用的策略状态（`Fight1` / `AMP` / `Php` / …）上不同，
  编辑时请保持公共部分一致。

**键盘键位与状态 id 对齐**：各预设可在**文件顶层**用 `keyboard:` 段显式声明「键盘字母 → 状态名」，
`CtrlFSM` 会按 `FSM._` 里该状态的 `id` 自动解析，避免出现“按了键但与 id 对不上”的问题：

```yaml
keyboard:
  a: Passive
  b: FixStand
  c: Velocity
  i: Php
```

- 写了 `keyboard:` 段 → **完全以配置为准**（未注册的状态会告警并被忽略）；
- 不写 → 使用内置默认映射 `a=1 b=2 c=3 d=5 e=6 f=7 g=9 h=10 i=11`。
- 方向键（`↑ ↓ ← →`）始终是“模拟摇杆 / 策略方向指令”，与上面的字母键互不干扰。

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
（`a`=Passive，`b`=FixStand，`c`=Velocity，`d`=当前预设里的自定义策略）。

> [!TIP]
> 上面的 `./g1_ctrl -n lo` 加载的是默认 `config.yaml`。若要跑某个预设（如 PHP），
> 加上 `-c`：`./g1_ctrl -n lo -c config_php.yaml`。

> [!NOTE]
> `-n` / `--network` 指定 DDS 网络接口。仿真时用 `lo`（本机回环）。

#### 4.2.1 PHP 感知跑酷策略（State_Php，预设 `config_php.yaml`，键盘 `d` 进入）

- **启动**：本策略在**独立预设** `config_php.yaml` 中，默认 `config.yaml` 不含 `Php` 状态：
  ```bash
  cd deploy/robots/g1/build
  ./g1_ctrl -n lo -c config_php.yaml
  ```
- **状态号**：`Php` 使用 `id: 4` —— 设计时预留给自定义策略的序号，对应键盘 `d`。
- **进入**：在 `g1_ctrl` 终端按键盘 `d`（或手柄 `RB + Y`）进入 PHP；**进入时默认 0 指令（站立）**。
- **切出**：`LT + B` 回 Passive（急停），`LT + up` 回 FixStand，`RT + A` 回 Velocity。
- **速度指令**：与 `Velocity` 等策略**共用同一套摇杆 / 键盘 toggle 设计**——状态只读摇杆量，
  方向键只是把虚拟摇杆置位（按一次置位，再按一次归零 → 站立）：

  | 按键 | 虚拟摇杆 | 行为 | `command` |
  |---|---|---|---|
  | ↑ | `ly=+5` | 前进（**再按一下归零 = 站立**） | 1 |
  | ↓ | `ly=-5` | 后退（再按一下归零） | 11 |
  | ← / → | `lx=∓5` | 左 / 右横移 90°（再按一下归零） | 3 / 5 |
  | q / e | `rx=∓5` | 左 / 右原地转（再按一下归零） | 3 / 5 |
  | ↑ + ← / ↑ + → | `ly=+5, lx=∓5` | 左前 / 右前 45° | 2 / 4 |
  | `z` | 全部归零 | 停止 / 站立 | 0 |

  实机直接推左摇杆（前后 / 左右）、右摇杆（转向），回中即站立。

- **观测构成**：`[ actions(29), base_ang_vel(3), dof_pos(29), dof_vel(29), projected_gravity(3) ]`（共 93）
  `+ velocity_command(15) + depth_latent(32) = 140`；`time_step` 恒为 0。
- **深度采集（与 holosoma 官方 sim2sim 同一路径）**：`scene_g1.xml` 中 `torso_link` 下的
  `<camera name="d435i_depth">`（`pos="0.01 0.01 0.44"`、D435i 训练标定四元数、`fovy=58.5953°`），
  `unitree_mujoco` 用隐藏 GLFW 上下文离屏渲染（`mjr_render` + `mjr_readPixels`，reverse-Z 反解成米制
  z 深度），以 10Hz 把 `106×60` 原始平面深度写入 `/dev/shm/depth_raw_shm`。
  - 用**渲染**而非射线投射，是为了让画面里包含机器人自身的近场几何（胸口/手臂/大腿）——
    训练时就是这样的分布；用 `mj_multiRay` 时相机在躯干网格内部必须排除 torso，
    会丢掉约 8% 的近场像素，导致策略步态偏保守（长距离前进时会逐渐下蹲）。
  - 深度单位/定义/内外参/预处理与官方逐项对齐，实测同姿态下 C++ 输出与 `mujoco.Renderer`
    的结果逐像素一致（`mean|diff| = 0.0000`）。
- **实物越障软垫**：`scene_g1.xml` 中 `php_pad_stack`（1×1 m 软垫堆，前沿 `x=1.1m`）。
  **实测高度结论（2026-11-19）**：
  | 高度 | 表现 |
  |---|---|
  | 0.30 m | 策略不响应，直接走过（太低） |
  | **0.40 m** | **能成功看到并越上 ← 当前采用** |
  改高度：`size` 第 3 个数 = 高度/2，且 `pos` 的 z 必须等于该值（否则会埋进地面）。
- **深度相机探测范围可视化**：`torso_link` 下的 `depth_fov` 子体（橙色半透明视锥，near0.3~far3.0）。
  它位于 geom **group 4**，深度渲染时通过 `mjvOption.geomgroup[4]=0` 排除，因此不会被当成障碍物。
  在 MuJoCo 查看器中可按需切换显示 geom 分组。

> [!NOTE]
> 深度共享内存 `depth_raw_shm` 由 `unitree_mujoco` 创建；若先启动 `g1_ctrl` 再启动仿真，进入 PHP 时会自动重连该 shm。


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
| Velocity / FixStand  | R1 + A  | 自定义locomotion       | 进入自定义locomotion           |
| 自定义locomotion      | R1 + Up/left/down/right | 自定义动作  | 进入自定义动作1                |

| 任意状态              | L2 + B  | Passive               | 回到阻尼模式（**急停**）         |

> 键盘辅助（仿真调试）：`a`=Passive，`b`=FixStand，`c`=Velocity，`d`=自定义策略（PHP 预设即为 PHP，需 `-c config_php.yaml` 启动）


## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [mjlab](https://github.com/mujocolab/mjlab.git): 构建训练与运行代码的基础。
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): 用于动作跟踪的通用人形机器人控制框架。
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): 提供 GPU 加速渲染与仿真接口。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大仿真功能。

终端1：
cd ~/g1_ws/unitree_rl_mjlab/simulate/build
./unitree_mujoco 
终端2：
cd ~/g1_ws/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl -n lo



