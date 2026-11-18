# PHP Sim2Sim 集成方案（mjlab / unitree_rl_mjlab）— 设计蓝图

> 目标：在 `unitree_rl_mjlab` 中新增 PHP(Perceptive Humanoid Parkour) depth 策略的 sim2sim 测试能力。
> 本文件记录已从源码核实的规格 + 分阶段实现计划。编码前请先对齐第 6 节架构选型。

## 0. 版本基线
- PHP 部署前基线：`unitree_rl_mjlab` commit `c33f9d1`，tag `php-pre-deploy`。

## 1. 官方产物（已确认可得，无需网上另找）
- 学生模型组件：`release-assets.json` → `components.student`（status=ready）
  - 文件：`depth_backbone.onnx` + `student.onnx`
  - 下载：`python scripts/download_assets.py student`（默认到 `~/.cache/php-parkour/student`）
- 部署示例：`run_php_sim.sh`（仿真）+ `run_php_inference.sh`（推理，holosoma_inference `run_policy.py`）
- 注意：`thirdparty/holosoma` 为空子模块，需 `git submodule update --init --recursive thirdparty/holosoma`

## 2. 数据接口规格（源码核实）
### 2.1 传感器（权威：holosoma 源码）
**深度相机 D435i（`RsD435iConfig`）**
- 原始 106(w)×60(h)，HFOV 89.5°，range [0.3, 3.0] m
- 安装：torso_link，offset_pos (0.01,0.01,0.44) m，offset_rot (1,27,1)°，offset_rot_base (-90,0,-90)°
- 训练随机化：位置 ±0.025 m、旋转 ±2.5~3°（部署可不加）

**深度预处理（仿真侧，写入 shm 前完成；`depth_shm_plugin.py`）**
1. 光线投射得每像素**米制深度**；无命中→far
2. crop `depth[2:, 4:-4]`（top2,left4,right4）→ 58×98
3. `clip(near=0.3, far=3.0)`（**resize 前**）
4. **torch 双三次抗锯齿** resize → 58×87（对齐训练 `Resize(BICUBIC)`；软性bilinear会掉台阶足落精度）
5. 后处理：`min(far)`；`where(<empty_threshold=0.15, far)`；归一化 `(d-near)/(far-near)-0.5` → 约 [-0.5,0.5]
6. dtype float32

**共享内存契约 `depth_img_shm`**
- 布局：float32，shape **(1,1,58,87)**，NCHW，20184 bytes
- 可选心跳：zmq PUB `ipc:///tmp/depth_img_shm.heartbeat`，JSON {v,seq,t_capture,state}
- 渲染频率 render_hz（默认10），latency_frames 环形缓冲（部署默认0）
- ⚠️ 预处理已在仿真侧完成 → 策略侧(A-2)读到的就是**可直接喂 backbone 的归一化张量**

**本体（student actor，concatenate=True，history_length=1）**
  - 观测项按 **名称排序** 拼接（observation manager 约定）：
    1. `actions` (29)  — 上一步动作
    2. `base_ang_vel` (3)
    3. `dof_pos` (29)
    4. `dof_vel` (29)
    5. `robot_anchor_projected_gravity` (3)
    6. `velocity_command` (15) — one-hot 离散速度命令
  - 本体合计 = 29+3+29+29+3+15 = **108 维**
  - 深度 latent 由 `depth_backbone.onnx` 编码后与本体一起送入 `student.onnx`
  - ⚠️ student.onnx 输入拼接顺序（latent 在前/在后，depth 是否含多帧）需在拿到 ONNX 后核对其 input shape 确认

### 2.2 15 维离散速度命令（VEL_CMD_MAPPING）
| idx | 命令 | idx | 命令 |
|----|------|----|------|
| 0 | stand | 6 | run_forward (~2.2 m/s) |
| 1 | forward (~0.8 m/s) | 7 | run_left_45 |
| 2 | left_45 | 8 | run_left_90 |
| 3 | left_90 | 9 | run_right_45 |
| 4 | right_45 | 10 | run_right_90 |
| 5 | right_90 | 11 | backward |
| — | 12-14 未使用 | | |

> 障碍技能（step/climb/vault/roll/jump/drop）由策略**根据深度自主选择**；操作员只给速度命令。

### 2.3 动作输出
- 29 维关节位置目标；`target = action_scale * action + default_joint_pos`，PD 控制（增益见 holosoma g1 配置）

## 2.4 学生双 ONNX 与 140 维线序（已用官方模型实测确认）
**模型 IO**
- `depth_backbone.onnx`：IN `depth_image [1,58,87]`（float32，归一化[-0.5,0.5]）→ OUT `depth_latent [1,32]`
- `student.onnx`：IN `obs [1,140]` + `time_step [1,1]`（恒置 **0**）→ OUT `actions [1,29]`（另有 joint_pos/vel/body_* 附加输出，忽略）

**obs(140) 线序 = `[actor_obs(93) | velocity_command(15) | depth_latent(32)]`**，逐项：
| 维度段 | 项 | 缩放 |
|---|---|---|
| 0:29 | actions（上一步动作，原始） | 1.0 |
| 29:32 | base_ang_vel（机体角速度） | 1.0 |
| 32:61 | dof_pos（关节位置） | 1.0 |
| 61:90 | dof_vel（关节速度） | 1.0 |
| 90:93 | projected_gravity（**机器人 anchor/torso 系**重力投影） | 1.0 |
| 93:108 | velocity_command（15 维 one-hot） | 1.0 |
| 108:140 | depth_latent（32，由 backbone 产出） | — |

**⚠️ 关键注意（官方注释强调，顺序错会静默退化）**
- actor_obs 必须按**字母序**拼接（勿按 ONNX metadata 的 `observation_names` 重排，那是误导）
- `projected_gravity` 是 **robot_anchor(torso_link) 系**投影（含腰部关节链），不是 pelvis 系
- `dof_pos` 需与训练定义一致（绝对或相对 default，待读项实现最终确认）
- depth backbone 输入为**单帧** (1,58,87)（部署 depth_buffer_len=1）
- 所有项 scale=1.0，无需额外缩放

## 2.5 A-1 实现记录（仿真侧深度发布，已完成）
**新增/修改文件（`unitree_rl_mjlab/simulate/`）**
- `src/depth_camera.{h,cc}`：从 `torso_link` 位姿合成相机外参（`cam_pos = base_pos + R(base_quat)·offset_pos`；`cam_quat = base_quat ⊗ euler(offset_rot) ⊗ euler(offset_rot_base)`，wxyz），用 `mj_multiRay` 对 **106×60** 像素批量光线投射，得**平面度量深度**（`depth = dist·dot(ray,principal)`，未命中→far=3.0）；排除 `torso_link` 自遮。~10Hz 限频。
- `src/depth_shm.h`：POSIX 共享内存写入器。
- `src/main.cc`：`Init`（模型加载后）、`MaybeUpdate`（`mj_step` 后、持物理锁内，线程安全）、`Shutdown`（退出前）。
- `CMakeLists.txt`：加入 `src/depth_camera.cc` 与 `rt`。

**输出契约**：`/dev/shm/depth_raw_shm` = float32 **106×60=6360**（行主序，y×106+x，单位 m，未命中=3.0，原始平面度量）。
> 预处理（crop[2:,4:-4]→clip[0.3,3.0]→torch 双三次抗锯齿→min(far)/where(<0.15,far)→归一化[-0.5,0.5]）放在 **A-2 Python 侧**完成，用 torch 保证与训练逐位一致。

**已验证**（无显示 C++ 烟测，与 Python 复现逐值一致）：`min=0.381 max=57900 mean=20.53 ~far=3053; center=2.813 bottom-center=1.413`；torch 预处理后形状 (58,87)、范围 [-0.492,0.500]；可视化显示地面梯度+底部双腿，几何正确。

## 2.6 A-2 实现记录（C++ deploy 状态，已完成编译+离线验证）
**为何用 C++ deploy 而非 Python runner**：本机各 conda 环境均无 `unitree_sdk2py`（不在镜像源，需 git 装且网络不稳）；而 `unitree_rl_mjlab/deploy` 已具备 DDS+ONNX Runtime+PD，且同样是真实 G1 的部署路径。

**新增/修改文件（`deploy/robots/g1/`）**
- `include/State_Php.h`、`src/State_Php.cpp`：`State_Php` 状态 + `REGISTER_OBSERVATION(php_obs)`
- `include/PhpDepthPreprocess.h`：深度预处理（crop/clip/双三次抗锯齿/归一化），生产与测试共用
- `config/policy/php/params/deploy.yaml`：identity 关节映射、kp/kd/default/action_scale（全部取自 ONNX 元数据）
- `config/policy/php/exported/{student.onnx,depth_backbone.onnx}`
- `config/config_php.yaml`（**独立预设**，不动默认 `config.yaml`）：`FSM._` 含 `Php{id:4,type:Php}`（id 4 = 预留给自定义策略的序号，键盘 `d`），FixStand/Velocity 增加 `RB+Y → Php`，新增 `Php` 块；顶层 `keyboard:` 段声明 `a/b/c/d → Passive/FixStand/Velocity/Php`
- `include/FSM/CtrlFSM.h`：键盘映射改为配置驱动（`keyboard:` 段，缺省回退内置 `a=1 b=2 c=3 d=5 e=6 f=7 g=9 h=10 i=11`）
- `main.cpp`：`#include "State_Php.h"`

**观测(140)**：`[actions(29) base_ang_vel(3) dof_pos(29) dof_vel(29) projected_gravity(3) | velocity_command(15) | depth_latent(32)]`
- `projected_gravity` 在 **torso/锚点系**：`R(root_quat_w ⊗ waist_yaw ⊗ waist_roll ⊗ waist_pitch)ᵀ·[0,0,-1]`（仿真 `imu` site 在 pelvis，故必须链腰；已核实 scene_g1.xml）
- `dof_pos = joint_pos − default`；`time_step` 恒 0（自定义循环，不经 `env->step` 的 episode 递增）
- 动作：`target = 1.0*action + default_joint_pos`；kp/kd 用元数据值；`joint_ids_map` 为恒等（桥接以 `sensordata[i]` 直传=Unitree 顺序）

**已验证（离线）**
- 双三次抗锯齿 vs torch：`mean|diff|=4.9e-5, p99=1.8e-5`（单一边界像素 0.013，属"近似[-0.5,0.5]"容差内）
- 双 ONNX 链路：backbone→latent(32)；student(obs140+time_step0)→actions(29) 有限、量级 ~0.2–0.5，站立/前进/左45 输出不同
- `g1_ctrl` 编译链接通过

**运行**：先启动 `simulate`（发布 `rt/lowstate`/`rt/lowcmd` + `depth_raw_shm`），再启动
`deploy/robots/g1/build/g1_ctrl -n lo -c config_php.yaml`，键盘 `d` 进入 PHP；
速度指令与其他策略统一走**摇杆/toggle**：`↑` 前进（再按一下归零=站立）、`↓` 后退、`←/→` 横移、`q/e` 转向、`z` 停止，
映射为 `command`（0=站,1=前进,2=左前45°,3=左90°,4=右前45°,5=右90°,11=后退）。

## 3. 技能/障碍能力边界（训练分布，实机测试须匹配高度）
| 技能 | 数据库 | 障碍顶面高度/尺寸 |
|---|---|---|
| locomotion | database_g1_locomotion | 平地 |
| step 上/下 | database_g1_step | STEP_TOP ≈ **0.352 m**（单级台阶） |
| jump | database_g1_jump_35 | ≈ **0.35 m** |
| climb | climb_58/76/94/134 | 顶面 ≈ **0.586 / 0.764 / 0.938 / 1.34 m** |
| vault | cat_vault / dash_vault(76) / speed_vault | 矮箱 / ≈ **0.76 m** |
| hurdle | database_g1_hurdle | 跨栏（≈0.35–0.5m 级） |
| roll | roll_78/90/136 | 从 **0.779 / 0.905 / 1.36 m** 台面翻滚下 |
| drop | drop_92 | 从 **0.92 m** 跳下 |

**不支持**：通用连续楼梯（建筑楼梯踏步 ~15–18cm 连续，与训练里 ~35cm 单级 step 分布不符）。因此"直接走楼道楼梯"**不是**可靠测试方式。

## 3.5 实验室障碍几何设计范围（权威依据：论文 Terrain Randomization）
训练对障碍做随机化，故实机障碍须落在该分布内：
- **高度 H**：取对应技能参考高度，容差 **±5 cm**。评测用 36 / 58 / 76 cm，climb 最高 125 cm。
- **宽度 W（左右跨距）**：从"参考所需最小值"到 **1.5 m** 采样 → 实机取 **0.8–1.2 m**（上限 1.5 m）。
- **深度 D（沿行进方向）**：属"其余维度"，参考值 **±5 cm**。
  - step / hurdle / vault（薄障碍）：约 **0.15–0.4 m** 薄箱/横杆
  - climb / roll / drop（需落脚/翻滚平台）：顶面深度 **≥0.6–0.8 m**，保证能站立/落地
- **偏航**：训练随机 **±45°** → 方形截面更稳，或允许箱体绕竖轴 ±45° 摆放。
- 体积参考（W×D×H）：
  - 单级 step / jump：1.0 × 0.35 × 0.35 m ≈ **0.12 m³**
  - climb 58：1.0 × 0.8 × 0.586 m ≈ **0.47 m³**
  - climb 76：1.0 × 0.8 × 0.764 m ≈ **0.61 m³**
  - vault 76（薄）：1.0 × 0.3 × 0.76 m ≈ **0.23 m³**
> 精确的逐技能深度见 `terrain/<skill>/box.json`（在 databases 资产包内）；需要我可拉取该包仅提取这些 json。

## 4. 实验室实机最简测试方案（无成品障碍物时）
- **阶段0（零障碍，先验管线）**：空走廊/空地，命令 stand/forward。验证：深度流→backbone→student→DDS→站立/行走。用现有 AR 空地即可，无需搭建。
- **阶段1（~35cm 单级，最省事）**：一个稳固 ~35cm 高箱子（纸箱/木箱，或泡沫垫堆到 35cm，宽≥1m、不晃）→ 测 step 上/下、jump。**这是最简可落地障碍**。
- **阶段2（可选，climb 58cm）**：稳固 ~58cm 平台（长凳/矮桌），能承重站立。
- （可选 vault）稳固 ~76cm 横栏/箱体。
- 关键：障碍顶面高度尽量贴 0.35 / 0.586 / 0.76 m（±数 cm）；深度相机需按训练位姿装 D435i。
- 安全：动态 vault/climb/roll 需专人保护 + 软垫 + 急停(阻尼/Passive) + 建议吊绳限高。

## 5. 观测/推理数据流（sim2sim 目标）
```
MuJoCo(unitree_mujoco) --DDS lowstate--> [PHP runner] 得到 q,dq,imu(关节信息来自宇树SDK)
MuJoCo 渲染深度58x87 --(shm/DDS)--> [PHP runner] 得到 depth
PHP runner:
  depth -> depth_backbone.onnx -> latent
  [latent + 本体108 + cmd15] -> student.onnx -> 29 关节目标
  -> DDS lowcmd -> MuJoCo
```

## 6. 架构选型：✅ 已选定 A（复用 DDS 全链路）
- **A. 复用现有 DDS sim2sim**（贴合"从宇树SDK取关节信息"）
  - C++ `simulate/`(unitree_mujoco)：新增 MuJoCo 离屏深度渲染(58×87, D435i内外参) + 共享内存发布（仿 holosoma DepthShmPlugin）
  - PHP runner：订阅 lowstate(DDS) + 深度(shm) + 双 ONNX 推理 + 发布 lowcmd
- **B. 兜底**：若 A 出现链路问题，用独立 Python sim2sim（单进程 MuJoCo+深度+双ONNX+PD）快速确认部署链路可行性。

## 7. 分阶段实现计划（A 为主，B 兜底）
1. 环境：`git submodule update --init thirdparty/holosoma`；`download_assets.py student`；核对两 ONNX 的 IO shape
2. A-1 仿真器：深度用 **MuJoCo `mj_multiRay` 光线投射**（得平面深度 = ray_dist·(−dir_z)，非 GL zbuffer），复现预处理 crop[2:,4:-4]→clip[0.3,3.0]→双三次抗锯齿 resize 58×87→normalize[-0.5,0.5]，写 `depth_img_shm` float32(1,1,58,87)（+可选心跳）。⚠️ 双三次抗锯齿 resize 保真是难点，需与 holosoma 输出逐帧比对验证
3. A-2 PHP runner：DDS 订阅 lowstate（关节/IMU）+ 读深度 shm + 双 ONNX（backbone→student）+ 发布 lowcmd
4. 本体108维观测组装（sorted-name 顺序）+ 深度 latent 拼接 + 15 维命令，与训练对齐
5. 命令输入：15 维 one-hot（键盘映射）
6. 校验：sim2sim 站立/行走 → 35cm 箱 step → 记录
7. 兜底：若 A 卡住，切 B 单进程快速验证部署链路

## 8. 待确认/风险
- student.onnx 输入顺序与 depth 是否多帧（buffer_len=6）——需拿模型核对
- 相机内外参与安装位姿需与 D435i 训练配置一致
- 实机深度与仿真域差异（sim2real gap）
