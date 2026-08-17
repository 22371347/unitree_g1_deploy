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

### 4. Deployment & Testing Guide

> The following uses G1 as an example. All paths are relative to the **repository root**. First enter the repo root (referred to as `<REPO>`):
> ```bash
> cd <REPO>   # e.g. cd ~/g1_ws/unitree_rl_mjlab
> ```

#### 4.1 Registering a New Policy (Config File Guide)

To integrate a new ONNX policy into this framework, generally complete the following 5 steps:

> 📌 **Workflow**: keep the default `config.yaml` for the current core policy only. To add a new
> policy, **copy `config.yaml` to `config_<policy_name>.yaml`** and add the new state there (launch
> with `--config`), instead of editing the default file — see *Parallel Testing* below.

1. **Prepare the policy files**: place the exported `policy.onnx` and `policy.onnx.data` into
   `deploy/robots/g1/config/policy/<policy_name>/exported/`
2. **Write the deployment config** `deploy/robots/g1/config/policy/<policy_name>/params/deploy.yaml`:
   - `joint_ids_map`: remapping from model joint order → hardware joint order (length = model DOF)
   - `step_dt`: control period (e.g. `0.02` = 50 Hz)
   - `stiffness` / `damping` / `default_joint_pos`: joint gains and default positions (given in **model order**)
   - `actions`: action config (`target = action * scale + offset`)
   - `observations`: observation config; **the written order is the concatenation order of the ONNX input tensor and must match training**
   - > ⚠️ A mismatch in joint mapping / observation order is the most common cause of abnormal behavior after deployment.
   - > ⚠️ **History layout** must match training too: deploy `use_gym_history: true` = frame-major
     (`[frame0: all terms, frame1: ...]`, mjlab `history_ordering="time"`); `false` = term-major
     (`[term0: all frames, term1: ...]`, IsaacLab `concatenate_terms`).
     A wrong layout scrambles the observations but the policy still runs without error → twitching / no response.
3. **Register the FSM state**: add a new state in the `FSM._` section of `deploy/robots/g1/config/config.yaml`, with a **unique id** and `type`:
   - `type: RLBase` — policies without a reference trajectory (e.g. velocity tracking, `State_RLBase`)
   - `type: Mimic` — motion imitation; additionally requires `motion_file` (`State_Mimic`)
   - `type: <custom>` — if the observation logic differs significantly, see step 5
4. **Add the state config block**: configure `transitions` (gamepad triggers) and `policy_dir`; Mimic type also needs `motion_file` and `time_start/end`
5. **Custom implementation for differing observation logic**: write `State_<name>.h/.cpp` under
   `deploy/robots/g1/include/` and `deploy/robots/g1/src/` (inherit `FSMState`, override `enter/run/exit`
   and observation computation, register with `REGISTER_FSM`), and `#include` the header in
   `deploy/robots/g1/main.cpp`.

> [!TIP]
> `config.yaml` already contains a complete registration guide comment and ready-to-copy examples
> (`Velocity` = RLBase example, `Fight1` / `AMP` = policy state examples).

#### Parallel Testing with Config Presets

When testing several policies alternately, keep **one config file per policy** instead of overwriting
`config.yaml`. Save presets as `config_<policy>.yaml` under `deploy/robots/g1/config/`
(e.g. `config_fight.yaml`, `config_AMP.yaml`) and select them at launch with `--config` — no copy-paste,
no recompile:

```bash
cd deploy/robots/g1/build
./g1_ctrl -n lo -c config_fight1.yaml    # simulation, fight1 (WBT) policy
./g1_ctrl -n lo -c config_AMP.yaml       # simulation, AMP policy
./g1_ctrl -c config_fight1.yaml          # real robot (default DDS config)
```

- `config.yaml` is loaded by default when `--config` is omitted.
- Each preset shares the common states (`Passive` / `FixStand` / `Velocity`) and only differs in the
  enabled policy states (`Fight1` / `AMP` / ...) — keep the common parts consistent when editing.

#### 4.2 Sim2Real: MuJoCo Simulation Validation

Validate the policy in MuJoCo simulation before going to the real robot. Two programs are required: the
simulator `unitree_mujoco` and the control program `g1_ctrl`.

**Build the control program** (G1 as an example):

```bash
cd deploy/robots/g1
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Build the simulator**:

```bash
cd simulate
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Run the simulation** (a gamepad must be connected):

Terminal 1 — start the simulator:

```bash
cd simulate/build
./unitree_mujoco
```

Terminal 2 — start the control program:

```bash
cd deploy/robots/g1/build
./g1_ctrl -n lo
```

Once running, press the keyboard keys `a` `b` `c` `d` to switch between the four states for observation
(`a`=Passive, `b`=FixStand, `c`=Velocity, `d`=Mimic_Dance1_subject2).

> [!NOTE]
> `-n` / `--network` specifies the DDS network interface. Use `lo` (loopback) for simulation.

#### 4.3 Real-Robot Deployment

> Prerequisites: install the communication libraries first
> - [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
> - [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

**4.3.1 Power On & Debug Mode**
- Power on the robot in a suspended state and wait until it enters `zero-torque` mode
- While in `zero-torque` mode, press `L2 + R2` on the gamepad to enter `debug mode` (joint damping enabled)

**4.3.2 Network Connection**
Connect the PC to the robot via Ethernet and configure IPv4:
- Address: `192.168.123.222`
- Netmask: `255.255.255.0`

The robot default address is usually `192.168.123.164` (confirm with `ifconfig` / Unitree docs).

**4.3.3 Transfer the Project to the Robot**

```bash
ssh unitree@192.168.123.164        # or the corresponding robot address
scp -r deploy unitree@192.168.123.164:~/
```

**4.3.4 Build & Run on the Robot**

```bash
ssh unitree@192.168.123.164
cd ~/deploy/robots/g1              # adjust to the actual scp path
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
./g1_ctrl
```

> [!NOTE]
> When running `g1_ctrl` on the robot's onboard computer, no network parameter is needed (default DDS config).
> For simulation on the PC, use `-n lo`; if connecting to the robot directly from the PC via Ethernet, use the
> corresponding NIC name (e.g. `-n enp5s0`, check with `ifconfig`).

#### 4.4 Gamepad Operation Guide

| Current State | Buttons | Target State | Description |
|---------|------|---------|------|
| Passive (damping mode) | `L2 + Up` | FixStand | Enter torque/standing mode, adjust posture until feet touch the ground |
| FixStand | `R2 + A` | Velocity | Enter motion control mode (velocity) |
| Velocity / FixStand | `R1 + A` | Mimic_Dance1_subject2 | Play the custom motion |
| Any state | `L2 + B` | Passive | Return to damping mode (**emergency stop**) |

> Keyboard shortcuts (simulation debugging): `a`=Passive, `b`=FixStand, `c`=Velocity, `d`=Mimic_Dance1_subject2

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
