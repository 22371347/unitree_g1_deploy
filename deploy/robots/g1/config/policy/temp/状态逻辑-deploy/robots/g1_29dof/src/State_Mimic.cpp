#include "State_Mimic.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cctype>
#include <unordered_map>

// ObservationManager evaluates terms during construction, before enter().
// Eigen's default Quaternion constructor leaves coefficients uninitialized.
static Eigen::Quaternionf init_quat = Eigen::Quaternionf::Identity();
std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;


static Eigen::Quaternionf normalized_quaternion(
    const Eigen::Quaternionf& quaternion,
    const char* source
)
{
    const float squared_norm = quaternion.squaredNorm();
    if (!quaternion.coeffs().allFinite() || !std::isfinite(squared_norm) || squared_norm < 1.0e-8f)
    {
        throw std::runtime_error(std::string("Invalid quaternion from ") + source);
    }
    return quaternion.normalized();
}

Eigen::Quaternionf robot_quat_w(
    isaaclab::ManagerBasedRLEnv* env,
    const std::shared_ptr<State_Mimic::MotionLoader_>& loader
)
{
    const auto root_quat = normalized_quaternion(env->robot->data.root_quat_w, "robot root IMU");
    const auto& joint_pos = env->robot->data.joint_pos;
    const int waist_indices[] = {
        loader->waist_yaw_idx,
        loader->waist_roll_idx,
        loader->waist_pitch_idx
    };
    for (const int index : waist_indices)
    {
        if (index < 0 || static_cast<std::size_t>(index) >= joint_pos.size() || !std::isfinite(joint_pos[index]))
        {
            throw std::runtime_error("Invalid robot waist joint index/value in Mimic observation");
        }
    }

    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(joint_pos[loader->waist_yaw_idx], Eigen::Vector3f::UnitZ()) \
        * Eigen::AngleAxisf(joint_pos[loader->waist_roll_idx], Eigen::Vector3f::UnitX()) \
        * Eigen::AngleAxisf(joint_pos[loader->waist_pitch_idx], Eigen::Vector3f::UnitY());

    return normalized_quaternion(torso_quat, "robot torso reconstruction");
}

Eigen::Quaternionf motion_anchor_quat_w(std::shared_ptr<State_Mimic::MotionLoader_> loader)
{
    const auto root_quat = normalized_quaternion(loader->root_quaternion(), "motion root");
    const auto joint_pos = loader->joint_pos();
    // 从 MotionLoader_ 读取该策略自身的腰部关节索引
    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(joint_pos[loader->waist_yaw_idx], Eigen::Vector3f::UnitZ()) \
        * Eigen::AngleAxisf(joint_pos[loader->waist_roll_idx], Eigen::Vector3f::UnitX()) \
        * Eigen::AngleAxisf(joint_pos[loader->waist_pitch_idx], Eigen::Vector3f::UnitY());

    //std::cout << "waist_yaw_idx: " << loader->waist_yaw_idx << std::endl;
    //std::cout << "waist_roll_idx: " << loader->waist_roll_idx << std::endl;
    //std::cout << "waist_pitch_idx: " << loader->waist_pitch_idx << std::endl;


//    return root_quat;
    return normalized_quaternion(torso_quat, "motion torso reconstruction");
}

static float reference_tilt_error(
    isaaclab::ManagerBasedRLEnv* env,
    const std::shared_ptr<State_Mimic::MotionLoader_>& loader
)
{
    const auto robot_quat = robot_quat_w(env, loader);
    const auto reference_quat = normalized_quaternion(
        init_quat * motion_anchor_quat_w(loader),
        "aligned motion torso"
    );
    const Eigen::Vector3f robot_up = robot_quat * Eigen::Vector3f::UnitZ();
    const Eigen::Vector3f reference_up = reference_quat * Eigen::Vector3f::UnitZ();
    const float cosine = std::clamp(robot_up.dot(reference_up), -1.0f, 1.0f);
    return std::acos(cosine);
}

static std::string normalized_joint_name(std::string name)
{
    name.erase(
        std::remove_if(
            name.begin(), name.end(),
            [](unsigned char c) {
                return std::isspace(c) || c == '[' || c == ']' || c == '\'' || c == '"';
            }
        ),
        name.end()
    );
    return name;
}

static const std::unordered_map<std::string, int>& g1_sdk_joint_ids()
{
    static const std::unordered_map<std::string, int> ids = {
        {"left_hip_pitch_joint", 0}, {"left_hip_roll_joint", 1},
        {"left_hip_yaw_joint", 2}, {"left_knee_joint", 3},
        {"left_ankle_pitch_joint", 4}, {"left_ankle_roll_joint", 5},
        {"right_hip_pitch_joint", 6}, {"right_hip_roll_joint", 7},
        {"right_hip_yaw_joint", 8}, {"right_knee_joint", 9},
        {"right_ankle_pitch_joint", 10}, {"right_ankle_roll_joint", 11},
        {"waist_yaw_joint", 12}, {"waist_roll_joint", 13},
        {"waist_pitch_joint", 14},
        {"left_shoulder_pitch_joint", 15}, {"left_shoulder_roll_joint", 16},
        {"left_shoulder_yaw_joint", 17}, {"left_elbow_joint", 18},
        {"left_wrist_roll_joint", 19}, {"left_wrist_pitch_joint", 20},
        {"left_wrist_yaw_joint", 21},
        {"right_shoulder_pitch_joint", 22}, {"right_shoulder_roll_joint", 23},
        {"right_shoulder_yaw_joint", 24}, {"right_elbow_joint", 25},
        {"right_wrist_roll_joint", 26}, {"right_wrist_pitch_joint", 27},
        {"right_wrist_yaw_joint", 28},
    };
    return ids;
}

/// Read policy joint order from ONNX and verify every policy-to-SDK mapping.
static void load_and_validate_joint_mapping(
    const std::string& onnx_path,
    const std::shared_ptr<State_Mimic::MotionLoader_>& loader,
    const std::vector<float>& joint_ids_map
)
{
    int yaw = 2, roll = 5, pitch = 8;
    bool found_yaw = false;
    bool found_roll = false;
    bool found_pitch = false;
    std::vector<std::string> joint_names;

    if (joint_ids_map.size() != 29)
    {
        throw std::runtime_error(
            "Mimic joint_ids_map must contain 29 entries, got " +
            std::to_string(joint_ids_map.size())
        );
    }

    std::array<bool, 29> used_motor_ids{};
    for (std::size_t i = 0; i < joint_ids_map.size(); ++i)
    {
        const float value = joint_ids_map[i];
        const int motor_id = static_cast<int>(std::lround(value));
        if (!std::isfinite(value) || std::abs(value - motor_id) > 1.0e-5f ||
            motor_id < 0 || motor_id >= 29 || used_motor_ids[motor_id])
        {
            throw std::runtime_error(
                "Invalid or duplicate Mimic joint_ids_map entry at policy index " +
                std::to_string(i)
            );
        }
        used_motor_ids[motor_id] = true;
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "waist_idx_reader");
        Ort::SessionOptions session_options;
        Ort::Session session(env, onnx_path.c_str(), session_options);

        auto model_metadata = session.GetModelMetadata();
        Ort::AllocatorWithDefaultOptions allocator;

        auto joint_names_str = model_metadata.LookupCustomMetadataMapAllocated("joint_names", allocator);
        if (joint_names_str) {
            std::string names_str(joint_names_str.get());
            size_t start = 0, end;
            while ((end = names_str.find(',', start)) != std::string::npos) {
                joint_names.push_back(normalized_joint_name(names_str.substr(start, end - start)));
                start = end + 1;
            }
            joint_names.push_back(normalized_joint_name(names_str.substr(start)));

            for (size_t i = 0; i < joint_names.size(); ++i) {
                const auto& name = joint_names[i];
                if (name == "waist_yaw_joint")   { yaw = i;   found_yaw = true; }
                if (name == "waist_roll_joint")  { roll = i;  found_roll = true; }
                if (name == "waist_pitch_joint") { pitch = i; found_pitch = true; }
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("[waist_idx] Failed to read ONNX metadata: {}", e.what());
    }

    if (!(found_yaw && found_roll && found_pitch))
    {
        yaw = 2;
        roll = 5;
        pitch = 8;
    }

    if (!joint_names.empty())
    {
        if (joint_names.size() != joint_ids_map.size())
        {
            throw std::runtime_error(
                "ONNX joint_names count does not match joint_ids_map: " +
                std::to_string(joint_names.size()) + " vs " +
                std::to_string(joint_ids_map.size())
            );
        }

        const auto& sdk_ids = g1_sdk_joint_ids();
        for (std::size_t i = 0; i < joint_names.size(); ++i)
        {
            const auto sdk_it = sdk_ids.find(joint_names[i]);
            if (sdk_it == sdk_ids.end())
            {
                throw std::runtime_error(
                    "Unknown G1 joint name in ONNX metadata: " + joint_names[i]
                );
            }
            const int configured_id = static_cast<int>(std::lround(joint_ids_map[i]));
            if (configured_id != sdk_it->second)
            {
                throw std::runtime_error(
                    "Mimic joint mapping mismatch for '" + joint_names[i] +
                    "': config=" + std::to_string(configured_id) +
                    ", expected SDK id=" + std::to_string(sdk_it->second)
                );
            }
        }
        spdlog::info("[Mimic mapping] validated all {} policy joints against G1 SDK order", joint_names.size());
    }
    else
    {
        spdlog::warn("[Mimic mapping] ONNX has no joint_names metadata; only permutation checks were applied");
    }

    loader->waist_yaw_idx   = yaw;
    loader->waist_roll_idx  = roll;
    loader->waist_pitch_idx = pitch;

    if (found_yaw && found_roll && found_pitch) {
        spdlog::info("[waist_idx] '{}' -> yaw={}, roll={}, pitch={}",
                     onnx_path.substr(onnx_path.find_last_of('/') + 1),
                     yaw, roll, pitch);
    } else {
        spdlog::warn("[waist_idx] '{}' -> incomplete joint_names metadata, using policy defaults ({},{},{})",
                     onnx_path.substr(onnx_path.find_last_of('/') + 1),
                     yaw, roll, pitch);
    }
}


namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(motion_command)
{
    auto loader = State_Mimic::motion;
    std::vector<float> data;

    auto motion_joint_pos = loader->joint_pos();
    auto motion_joint_vel = loader->joint_vel();

    data.insert(data.end(),
                motion_joint_pos.data(),
                motion_joint_pos.data() + motion_joint_pos.size());
    data.insert(data.end(),
                motion_joint_vel.data(),
                motion_joint_vel.data() + motion_joint_vel.size());
    return data;
}

REGISTER_OBSERVATION(motion_anchor_ori_b)
{
    auto loader = State_Mimic::motion;
    std::vector<float> out;

    auto real_quat_w = robot_quat_w(env, loader);
    auto ref_quat_w  = motion_anchor_quat_w(loader);

    const auto aligned_ref = normalized_quaternion(init_quat * ref_quat_w, "aligned motion torso");
    const auto rot_ = normalized_quaternion(
        aligned_ref.conjugate() * real_quat_w,
        "relative Mimic torso orientation"
    );
    auto rot = rot_.toRotationMatrix().transpose();

    if (!rot.allFinite())
    {
        throw std::runtime_error("Mimic torso rotation matrix contains a non-finite value");
    }

    Eigen::Matrix<float, 6, 1> data;
    data << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    return std::vector<float>(data.data(), data.data() + data.size());
}

   //调试：尝试修改计算方法
   /*
REGISTER_OBSERVATION(motion_anchor_ori_b)
{
    auto loader = State_Mimic::motion;
    std::vector<float> out;

    auto real_quat_w = robot_quat_w(env);
    auto ref_quat_w  = motion_anchor_quat_w(loader);

    // 移除 init_quat，直接计算相对旋转
    auto q_relative = real_quat_w.conjugate() * ref_quat_w;

    // 不转置
    auto rot = q_relative.toRotationMatrix();

    Eigen::Matrix<float, 6, 1> data;
    data << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    return std::vector<float>(data.data(), data.data() + data.size());
}
    */

// 注册项motion_anchor_pos_b和base_lin_vel
REGISTER_OBSERVATION(motion_anchor_pos_b)
{
    /*auto loader = State_Mimic::motion;
    if (!loader) {
        return std::vector<float>{0.0f, 0.0f, 0.0f};
    }

    // 1. 获取机器人锚点的世界坐标位置和朝向
    // 注意：asset->data 中没有 root_pos_w！我们需要从其他地方获取位置。
    // 解决方案：使用 projected_gravity_b 来推断，或者从 MuJoCo 的 d.xpos 获取。
    // 在实机部署中，位置需要从状态估计器获取。

    // 临时方案：使用零位置（假设锚点在世界原点）
    // 这会导致锚点位置偏差，但可以验证数据流
    Eigen::Vector3f robot_anchor_pos = Eigen::Vector3f::Zero();  // 需要替换为真实位置

    // 获取机器人锚点的朝向
    auto & asset = env->robot;
    Eigen::Quaternionf robot_anchor_quat = asset->data.root_quat_w;

    // 2. 获取运动锚点的世界坐标位姿
    Eigen::Vector3f motion_anchor_pos = loader->root_position();
    Eigen::Quaternionf motion_anchor_quat = loader->root_quaternion();

    // 3. 计算相对变换（运动锚点相对于机器人锚点）
    Eigen::Quaternionf robot_anchor_quat_inv = robot_anchor_quat.conjugate();
    Eigen::Quaternionf relative_quat = robot_anchor_quat_inv * motion_anchor_quat;
    Eigen::Vector3f relative_pos = robot_anchor_quat_inv * (motion_anchor_pos - robot_anchor_pos);

    // 4. 调试输出
    spdlog::info("[motion_anchor_pos_b] robot_anchor_pos: ({:.4f}, {:.4f}, {:.4f})",
                 robot_anchor_pos[0], robot_anchor_pos[1], robot_anchor_pos[2]);
    spdlog::info("[motion_anchor_pos_b] relative_pos: ({:.4f}, {:.4f}, {:.4f})",
                 relative_pos[0], relative_pos[1], relative_pos[2]);

    return std::vector<float>{relative_pos[0], relative_pos[1], relative_pos[2]};*/
    return std::vector<float>{0.0f, 0.0f, 0.0f};
}

REGISTER_OBSERVATION(base_lin_vel)
{
    /*
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;

    std::cout << "[base_lin_vel] projected_gravity: "
              << data[0] << ", " << data[1] << ", " << data[2] << std::endl;

    data[2] = 0;

    return std::vector<float>(data.data(), data.data() + data.size());
    */
    return std::vector<float>{0.0f, 0.0f, 0.0f};
}

REGISTER_OBSERVATION(motion_time_step)
{
    return {static_cast<float>(State_Mimic::motion->frame_index())};
}

REGISTER_OBSERVATION(wbt_time_step)
{
    return {static_cast<float>(State_Mimic::motion->frame_index())};
}


}
}


State_Mimic::State_Mimic(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if(!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }

    const YAML::Node deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    const float csv_fps = cfg["fps"] ? cfg["fps"].as<float>() : 50.0f;
    const auto joint_ids_map = deploy_cfg["joint_ids_map"].as<std::vector<float>>();

    // Motion
    motion_ = std::make_shared<MotionLoader_>(
        motion_file.string(), csv_fps, joint_ids_map
    );
    spdlog::info("Loaded motion file '{}' with duration {:.2f}s", motion_file.stem().string(), motion_->duration);
    motion = motion_;
    if(cfg["time_start"]) {
        float time_start = cfg["time_start"].as<float>();
        time_range_[0] = std::clamp(time_start, 0.0f, motion_->duration);
    } else {
        time_range_[0] = 0.0f;
    }
    if(cfg["time_end"]) {
        float time_end = cfg["time_end"].as<float>();
        time_range_[1] = std::clamp(time_end, 0.0f, motion_->duration);
    } else {
        time_range_[1] = motion_->duration;
    }
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(deploy_cfg, articulation);
    auto onnx_path = policy_dir / "exported" / "policy.onnx";
    env->alg = std::make_unique<isaaclab::OrtRunner>(onnx_path);

    load_and_validate_joint_mapping(
        onnx_path.string(), motion_, env->robot->data.joint_ids_map
    );

    action_abort_limit_ = cfg["action_abort_limit"] ?
        cfg["action_abort_limit"].as<float>() : 100.0f;
    if (!std::isfinite(action_abort_limit_) || action_abort_limit_ <= 0.0f)
    {
        throw std::runtime_error("Mimic action_abort_limit must be positive");
    }

    const std::string policy_fault_target = cfg["policy_fault_target"] ?
        cfg["policy_fault_target"].as<std::string>() :
        (FSMStringMap.right.count("Recovery") ? "Recovery" : "Passive");
    if (!FSMStringMap.right.count(policy_fault_target))
    {
        throw std::runtime_error("Unknown Mimic policy-fault target: " + policy_fault_target);
    }
    this->registered_checks.emplace_back(
        [this]() -> bool { return policy_fault_.load(); },
        FSMStringMap.right.at(policy_fault_target)
    );

    bad_orientation_enabled_ = cfg["enable_bad_orientation_check"] ?
        cfg["enable_bad_orientation_check"].as<bool>() : true;
    if (bad_orientation_enabled_)
    {
        const std::string bad_orientation_target = cfg["bad_orientation_target"] ?
            cfg["bad_orientation_target"].as<std::string>() : "Passive";
        if (!FSMStringMap.right.count(bad_orientation_target))
        {
            throw std::runtime_error("Unknown Mimic bad-orientation target: " + bad_orientation_target);
        }
        bad_orientation_limit_ = cfg["bad_orientation_limit"] ?
            cfg["bad_orientation_limit"].as<float>() : 1.0f;
        constexpr float pi = 3.14159265358979323846f;
        if (!std::isfinite(bad_orientation_limit_) ||
            bad_orientation_limit_ <= 0.0f || bad_orientation_limit_ > pi)
        {
            throw std::runtime_error("Mimic bad_orientation_limit must be in (0, pi]");
        }
        const int confirm_ms = cfg["bad_orientation_confirm_ms"] ?
            cfg["bad_orientation_confirm_ms"].as<int>() : 0;
        if (confirm_ms < 0)
        {
            throw std::runtime_error("Mimic bad_orientation_confirm_ms cannot be negative");
        }
        bad_orientation_confirm_steps_ = std::max(
            1,
            static_cast<int>(std::ceil(
                confirm_ms / (env->step_dt * 1000.0f)
            ))
        );
        this->registered_checks.emplace_back(
            [this]() -> bool {
                return bad_orientation_steps_.load() >= bad_orientation_confirm_steps_;
            },
            FSMStringMap.right.at(bad_orientation_target)
        );
    }

    const std::string completion_target = cfg["completion_target"] ?
        cfg["completion_target"].as<std::string>() :
        (cfg["end_state"] ? cfg["end_state"].as<std::string>() : "Velocity");
    if (!FSMStringMap.right.count(completion_target))
    {
        throw std::runtime_error("Unknown Mimic completion target: " + completion_target);
    }
    const float completion_hold_s = cfg["completion_hold_s"] ?
        cfg["completion_hold_s"].as<float>() : 0.0f;
    if (!std::isfinite(completion_hold_s) || completion_hold_s < 0.0f)
    {
        throw std::runtime_error("Mimic completion_hold_s cannot be negative");
    }
    this->registered_checks.emplace_back(
        [this, completion_hold_s]()->bool{
            return (policy_steps_.load() * env->step_dt) >
                (time_range_[1] + completion_hold_s);
        },
        FSMStringMap.right.at(completion_target)
    );
}

bool State_Mimic::update_action_snapshot()
{
    const auto raw_action = env->action_manager->action();
    const auto processed_action = env->action_manager->processed_actions();
    const std::size_t expected_size = env->robot->data.joint_ids_map.size();

    const auto report_fault = [this](const std::string& message) -> bool {
        if (!policy_fault_.exchange(true))
        {
            spdlog::error("Mimic policy fault: {}. Holding the last valid target.", message);
        }
        return false;
    };

    if (raw_action.size() != expected_size || processed_action.size() != expected_size)
    {
        return report_fault(
            "Mimic action size mismatch: raw=" + std::to_string(raw_action.size()) +
            ", processed=" + std::to_string(processed_action.size()) +
            ", expected=" + std::to_string(expected_size)
        );
    }

    for (std::size_t i = 0; i < expected_size; ++i)
    {
        if (!std::isfinite(raw_action[i]) || !std::isfinite(processed_action[i]))
        {
            return report_fault(
                "non-finite action at policy index " + std::to_string(i)
            );
        }
        if (std::abs(raw_action[i]) > action_abort_limit_)
        {
            return report_fault(
                "raw action exceeded limit at policy index " + std::to_string(i) +
                ": " + std::to_string(raw_action[i])
            );
        }
    }

    if (bad_orientation_enabled_)
    {
        const float tilt_error = reference_tilt_error(env.get(), motion);
        if (tilt_error > bad_orientation_limit_)
        {
            const int confirmed_steps = bad_orientation_steps_.fetch_add(1) + 1;
            if (confirmed_steps == bad_orientation_confirm_steps_)
            {
                spdlog::warn(
                    "Mimic reference tilt error {:.3f} rad exceeded {:.3f} rad for {} policy steps",
                    tilt_error, bad_orientation_limit_, confirmed_steps
                );
            }
        }
        else
        {
            bad_orientation_steps_ = 0;
        }
    }

    {
        std::lock_guard<std::mutex> lock(action_mutex_);
        action_snapshot_ = processed_action;
    }
    return true;
}

void State_Mimic::enter()
{
    policy_thread_running = false;
    policy_fault_ = false;
    policy_steps_ = 0;
    bad_orientation_steps_ = 0;
    {
        std::lock_guard<std::mutex> lock(action_mutex_);
        action_snapshot_.clear();
    }

    try
    {
        motion = motion_;
        env->robot->update();
        motion->reset(env->robot->data, time_range_[0]);

        const auto ref_anchor_quat = motion_anchor_quat_w(motion);
        const auto robot_anchor_quat = robot_quat_w(env.get(), motion);
        const auto ref_yaw = isaaclab::yawQuaternion(ref_anchor_quat).toRotationMatrix();
        const auto robot_yaw = isaaclab::yawQuaternion(robot_anchor_quat).toRotationMatrix();
        init_quat = normalized_quaternion(
            Eigen::Quaternionf(robot_yaw * ref_yaw.transpose()),
            "Mimic initial heading alignment"
        );

        env->reset();
        motion->update(time_range_[0]);
        env->step();
        policy_steps_ = env->episode_length;
        if (!update_action_snapshot())
        {
            return;
        }
    }
    catch (const std::exception& error)
    {
        policy_fault_ = true;
        spdlog::error("Mimic initialization failed: {}", error.what());
        return;
    }

    std::vector<float> initial_action;
    {
        std::lock_guard<std::mutex> lock(action_mutex_);
        initial_action = action_snapshot_;
    }
    for (std::size_t i = 0; i < env->robot->data.joint_ids_map.size(); ++i)
    {
        const auto motor_id = static_cast<std::size_t>(env->robot->data.joint_ids_map[i]);
        lowcmd->msg_.motor_cmd()[motor_id].q() = initial_action[i];
    }

    // Stiffness and damping are stored in SDK motor order. Install gains only
    // after the first valid policy target is ready, so no stale target is used.
    for (std::size_t i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;

        while (policy_thread_running)
        {
            std::this_thread::sleep_until(sleepTill);
            if (!policy_thread_running)
            {
                break;
            }

            try
            {
                motion->update(policy_steps_.load() * env->step_dt + time_range_[0]);
                env->step();
                policy_steps_ = env->episode_length;
                if (!update_action_snapshot())
                {
                    break;
                }
            }
            catch (const std::exception& error)
            {
                if (!policy_fault_.exchange(true))
                {
                    spdlog::error(
                        "Mimic policy thread failed: {}. Holding the last valid target.",
                        error.what()
                    );
                }
                break;
            }

            sleepTill += dt;
        }
        policy_thread_running = false;
    });
}


void State_Mimic::run()
{
    std::vector<float> action;
    {
        std::lock_guard<std::mutex> lock(action_mutex_);
        action = action_snapshot_;
    }

    if (action.size() != env->robot->data.joint_ids_map.size())
    {
        return;
    }

    for (std::size_t i = 0; i < env->robot->data.joint_ids_map.size(); ++i)
    {
        const auto motor_id = static_cast<std::size_t>(env->robot->data.joint_ids_map[i]);
        lowcmd->msg_.motor_cmd()[motor_id].q() = action[i];
    }
}
