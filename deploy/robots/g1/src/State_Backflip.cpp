#include "State_Backflip.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "g1_health_logger.h"
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>

// 从 ONNX 模型的 joint_names metadata 中解析腰部关节索引（IsaacLab 顺序），
// 写入该策略的 MotionLoader_（供 motion_torso_quat 计算使用）
static void load_waist_indices_from_onnx(const std::string& onnx_path,
                                         std::shared_ptr<State_Backflip::MotionLoader_> loader)
{
    // 默认值（IsaacLab G1 顺序：yaw=2, roll=5, pitch=8）
    int yaw = 2, roll = 5, pitch = 8;
    bool found = false;

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "waist_idx_reader");
        Ort::SessionOptions session_options;
        Ort::Session session(env, onnx_path.c_str(), session_options);

        auto model_metadata = session.GetModelMetadata();
        Ort::AllocatorWithDefaultOptions allocator;

        auto joint_names_str = model_metadata.LookupCustomMetadataMapAllocated("joint_names", allocator);
        if (joint_names_str) {
            std::string names_str(joint_names_str.get());
            std::vector<std::string> joint_names;
            size_t start = 0, end;
            while ((end = names_str.find(',', start)) != std::string::npos) {
                joint_names.push_back(names_str.substr(start, end - start));
                start = end + 1;
            }
            joint_names.push_back(names_str.substr(start));

            for (size_t i = 0; i < joint_names.size(); ++i) {
                if (joint_names[i] == "waist_yaw_joint")   { yaw = i;   found = true; }
                if (joint_names[i] == "waist_roll_joint")  { roll = i;  found = true; }
                if (joint_names[i] == "waist_pitch_joint") { pitch = i; found = true; }
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("[backflip waist_idx] Failed to read ONNX metadata: {}", e.what());
    }

    loader->waist_yaw_idx   = yaw;
    loader->waist_roll_idx  = roll;
    loader->waist_pitch_idx = pitch;

    spdlog::info("[backflip] waist indices (model order): yaw={}, roll={}, pitch={}",
                 yaw, roll, pitch);
}


State_Backflip::State_Backflip(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // 播放模式：true = 进入即播放一次；false = 进入停末帧按 r 重播（默认）
    if (cfg["auto_play"]) {
        auto_play_ = cfg["auto_play"].as<bool>();
        spdlog::info("[backflip] auto_play={}", auto_play_);
    }

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if (!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }

    // Motion
    motion_ = std::make_shared<MotionLoader_>(motion_file.string());
    num_frames_ = motion_->num_frames;
    spdlog::info("[backflip] Loaded motion file '{}' frames={} duration={:.2f}s",
                 motion_file.filename().string(), num_frames_, motion_->duration);

    // 必须在 env（ObservationManager）构造前设置静态 motion：
    // ObservationManager 构造时就会调用 motion_command 观测取初始值，
    // 它读取 State_Mimic::motion，若为 nullptr 会段错误（与 State_Mimic 相同做法）
    State_Mimic::motion = motion_;

    // env + alg
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    auto onnx_path = policy_dir / "exported" / "policy.onnx";
    env->alg = std::make_unique<isaaclab::OrtRunner>(onnx_path);
    load_waist_indices_from_onnx(onnx_path.string(), motion_);

    // ---- 姿态保护（可配置，默认开启）：倾角超限 -> 目标状态（默认 Passive，可配 Recovery）----
    bool enable_bad_orientation_check = true;
    if (cfg["enable_bad_orientation_check"])
    {
        enable_bad_orientation_check =
            cfg["enable_bad_orientation_check"].as<bool>();
    }
    float bad_orientation_limit = 1.0f;
    if (cfg["bad_orientation_limit"])
    {
        bad_orientation_limit =
            cfg["bad_orientation_limit"].as<float>();
    }
    std::string bad_orientation_target = "Passive";
    if (cfg["bad_orientation_target"])
    {
        bad_orientation_target =
            cfg["bad_orientation_target"].as<std::string>();
    }
    int bad_orientation_confirm_ms = 0;
    if (cfg["bad_orientation_confirm_ms"])
    {
        bad_orientation_confirm_ms =
            cfg["bad_orientation_confirm_ms"].as<int>();
    }
    if (enable_bad_orientation_check)
    {
        if (!FSMStringMap.right.count(bad_orientation_target))
        {
            throw std::runtime_error(
                "Unknown bad_orientation_target: " + bad_orientation_target
            );
        }
        const int target_mode = FSMStringMap.right.at(bad_orientation_target);
        this->registered_checks.emplace_back(
            [this, bad_orientation_limit, bad_orientation_confirm_ms]() -> bool
            {
                return condition_confirmed(
                    isaaclab::mdp::bad_orientation(
                        env.get(), bad_orientation_limit
                    ),
                    bad_orientation_since_,
                    bad_orientation_confirm_ms
                );
            },
            target_mode
        );
    }

    // ---- 摔倒检测：倾角过大（持续 confirm_ms）-> 自动切目标（如 Recovery）----
    // 注：仅用 IMU 倾角判据（projected_gravity），不依赖高度（实机无 SportModeState 高度源）
    if (cfg["fall_detection"])
    {
        const auto fall_cfg = cfg["fall_detection"];
        const std::string target_name =
            fall_cfg["target"] ?
                fall_cfg["target"].as<std::string>() :
                "Recovery";
        const float max_tilt =
            fall_cfg["max_tilt"] ?
                fall_cfg["max_tilt"].as<float>() :
                0.872665f;   // 50 度
        const int confirm_ms =
            fall_cfg["confirm_ms"] ?
                fall_cfg["confirm_ms"].as<int>() :
                50;

        if (!FSMStringMap.right.count(target_name))
        {
            throw std::runtime_error(
                "Unknown fall detection target state: " + target_name
            );
        }

        const int target_mode = FSMStringMap.right.at(target_name);
        this->registered_checks.emplace_back(
            [this, max_tilt, confirm_ms]() -> bool
            {
                const bool bad_tilt =
                    isaaclab::mdp::bad_orientation(env.get(), max_tilt);
                return condition_confirmed(
                    bad_tilt,
                    fall_condition_since_,
                    confirm_ms
                );
            },
            target_mode
        );
    }

    // ---- 播完自动切回目标状态（如 Fight）：已触发过播放且播放到末帧 ----
    if (cfg["end_state"])
    {
        const std::string end_state = cfg["end_state"].as<std::string>();
        if (!FSMStringMap.right.count(end_state))
        {
            throw std::runtime_error("Unknown end_state: " + end_state);
        }
        const int end_id = FSMStringMap.right.at(end_state);
        this->registered_checks.emplace_back(
            std::make_pair(
                [this, end_id]() -> bool
                {
                    return has_played_ && frame_ >= num_frames_ - 1;
                },
                end_id
            )
        );
    }

    spdlog::info("[backflip] ctor done, frames={}", num_frames_);
}


void State_Backflip::enter()
{
    // 设置增益（经 joint_ids_map 将模型顺序重映射到硬件顺序）
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        int hw_idx = static_cast<int>(env->robot->data.joint_ids_map[i]);
        lowcmd->msg_.motor_cmd()[hw_idx].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[hw_idx].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[hw_idx].dq() = 0;
        lowcmd->msg_.motor_cmd()[hw_idx].tau() = 0;
    }

    env->robot->update();
    // 播放模式（auto_play_，config 可快速切换，无需重编译）：
    //   true  -> 进入即从头播放一次（has_played_=true，播完自动回 end_state）
    //   false -> 进入稳定在 motion 最后一帧（末帧站姿），按 r / RB+down 触发重播
    if (auto_play_) {
        frame_ = 0;
        motion_->frame = 0;
        has_played_ = true;   // 进入即播放，播完回 end_state 的检查生效
    } else {
        frame_ = num_frames_ - 1;
        motion_->frame = frame_;
        has_played_ = false;   // 未触发播放，播完回 end_state 的检查不生效
    }
    enter_ramp_ = 0;       // 重置进入平滑

    // 让 observation_manager 的 motion_command / motion_anchor_ori_b 读到本策略的 motion
    State_Mimic::motion = motion_;
    spdlog::info("[backflip] enter: motion set to State_Mimic::motion");

    // 初始偏航对齐（写入 State_Mimic::init_quat，供 motion_anchor_ori_b 观测）
    recompute_init_quat();
    spdlog::info("[backflip] enter: init_quat done, motion={}", (bool)State_Mimic::motion);

    env->action_manager->reset();
    env->reset();
    spdlog::info("[backflip] enter: env reset done");

    // Start policy thread (50Hz, 与 motion fps 同步)
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;
        while (policy_thread_running)
        {
            env->robot->update();

            // 重播信号：进入该状态的组合键（RB+down）上升沿 -> 重新播放动作。
            // 注意：该组合键不能出现在本状态的 transitions 里（避免与切走冲突）
            auto & joy = FSMState::lowstate->joystick;
            bool restart_key = joy.RB.pressed && joy.down.pressed;
            // 键盘检测
            std::string key = FSMState::keyboard->key();
            if (restart_key && !last_restart_key_) {
                replay_motion();
            }
            else if (key == "r") {  // 按 'r' 键重播
                replay_motion();
            }
            last_restart_key_ = restart_key;

            advance_frame();
            //spdlog::info("[backflip] f={} before step", frame_);

            // 仿照 State_Mimic：observation_manager 自动构建观测 + 策略推理 + 动作下发
            // （motion_command / motion_anchor_ori_b 从 State_Mimic::motion / init_quat 读取）
            env->step();
            //spdlog::info("[backflip] f={} after step", frame_);

            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}


void State_Backflip::run()
{
    auto action = env->action_manager->processed_actions();
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    // ---- G1 health logger ----
    g1health::G1HealthLogger::instance().logCycle(
        lowstate->msg_, lowcmd->msg_,
        g1health::actionToHardware(env->robot->data.joint_ids_map, action, 29),
        g1health::nanv(), env->last_inference_s.load(), g1health::nanv());
}


void State_Backflip::advance_frame()
{
    // 固定 hold 逻辑：从头播放，播到最后一帧（站立）后保持
    frame_ = std::min(frame_ + 1, num_frames_ - 1);
    motion_->frame = frame_;
}


void State_Backflip::replay_motion()
{
    // 1) 帧回零：从动作开头重新播放
    frame_ = 0;
    motion_->frame = 0;
    has_played_ = true;   // 标记已触发播放（播完自动回 end_state）

    // 2) 重新对齐：以当前机器人朝向为基准重算偏航对齐（供 motion_anchor_ori_b 观测）
    recompute_init_quat();

    // 3) 清零上一步动作反馈（last_action 观测）
    env->action_manager->reset();

    spdlog::info("[backflip] Replay triggered (frame -> 0)");
}


// 机器人 torso 朝向 = IMU(pelvis) 朝向 * 腰部关节旋转
// 注：模拟器/实机 IMU 位于 pelvis（site "imu" 挂在 pelvis body），故需补偿腰
Eigen::Quaternionf State_Backflip::robot_torso_quat()
{
    auto & motors = FSMState::lowstate->msg_.motor_state();
    Eigen::Quaternionf root_quat = env->robot->data.root_quat_w;

    // 硬件 qpos 顺序: 12=waist_yaw, 13=waist_roll, 14=waist_pitch
    return root_quat
        * Eigen::AngleAxisf(motors[12].q(), Eigen::Vector3f::UnitZ())
        * Eigen::AngleAxisf(motors[13].q(), Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(motors[14].q(), Eigen::Vector3f::UnitY());
}


// 重新计算初始偏航对齐（世界系 -> 机器人初始朝向系），写入 State_Mimic::init_quat
// （observation_manager 的 motion_anchor_ori_b 观测会用到它，与 State_Mimic 一致）
void State_Backflip::recompute_init_quat()
{
    auto ref_yaw   = isaaclab::yawQuaternion(motion_->root_quaternion()).toRotationMatrix();
    auto robot_yaw = isaaclab::yawQuaternion(robot_torso_quat()).toRotationMatrix();
    State_Mimic::init_quat = Eigen::Quaternionf(robot_yaw * ref_yaw.transpose());
}


// 从 unitree_mujoco 发布的 SportModeState 读取 root 高度（仿真用）；真机无高度源返回 false
bool State_Backflip::read_base_height(float &height) const
{
    if (!sport_mode_state || sport_mode_state->isTimeout())
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(sport_mode_state->mutex_);
    height = static_cast<float>(sport_mode_state->msg_.position()[2]);
    return std::isfinite(height);
}


// 条件持续满足 confirm_ms 毫秒后才返回 true（防抖）
bool State_Backflip::condition_confirmed(
    bool condition,
    std::optional<std::chrono::steady_clock::time_point> &since,
    int confirm_ms
)
{
    if (!condition)
    {
        since.reset();
        return false;
    }

    if (confirm_ms <= 0)
    {
        return true;
    }

    const auto now = std::chrono::steady_clock::now();
    if (!since)
    {
        since = now;
        return false;
    }

    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - *since
        ).count();
    return elapsed >= confirm_ms;
}
