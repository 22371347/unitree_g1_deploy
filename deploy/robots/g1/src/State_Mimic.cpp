#include "State_Mimic.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <onnxruntime_cxx_api.h>

static Eigen::Quaternionf init_quat;
std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;


Eigen::Quaternionf robot_quat_w(isaaclab::ManagerBasedRLEnv* env)
{
    using G1Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
    G1Type* robot = dynamic_cast<G1Type*>(env->robot.get());

    auto root_quat = env->robot->data.root_quat_w;
    auto & motors = robot->lowstate->msg_.motor_state();

    // robot_quat_w 使用 motor_state 硬件索引（12,13,14 永远是腰部关节）
    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(motors[12].q(), Eigen::Vector3f::UnitZ()) \
        * Eigen::AngleAxisf(motors[13].q(), Eigen::Vector3f::UnitX()) \
        * Eigen::AngleAxisf(motors[14].q(), Eigen::Vector3f::UnitY());

//    return root_quat;
    return torso_quat;
}

Eigen::Quaternionf motion_anchor_quat_w(std::shared_ptr<State_Mimic::MotionLoader_> loader)
{
    const auto root_quat = loader->root_quaternion();
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
    return torso_quat;
}

/// 从 ONNX 模型的 joint_names metadata 中解析腰部关节索引，写入 MotionLoader_
static void load_waist_indices_from_onnx(const std::string& onnx_path,
                                          std::shared_ptr<State_Mimic::MotionLoader_> loader)
{
    // 默认值（硬件顺序）
    int yaw = 12, roll = 13, pitch = 14;
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
                const auto& name = joint_names[i];
                if (name == "waist_yaw_joint")   { yaw = i;   found = true; }
                if (name == "waist_roll_joint")  { roll = i;  found = true; }
                if (name == "waist_pitch_joint") { pitch = i; found = true; }
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("[waist_idx] Failed to read ONNX metadata: {}", e.what());
    }

    // 写入 loader（每个策略各自的 MotionLoader_）
    loader->waist_yaw_idx   = yaw;
    loader->waist_roll_idx  = roll;
    loader->waist_pitch_idx = pitch;

    if (found) {
        spdlog::info("[waist_idx] '{}' → yaw={}, roll={}, pitch={}",
                     onnx_path.substr(onnx_path.find_last_of('/') + 1),
                     yaw, roll, pitch);
    } else {
        spdlog::warn("[waist_idx] '{}' → joint_names not found, using defaults ({},{},{})",
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

    auto real_quat_w = robot_quat_w(env);
    auto ref_quat_w  = motion_anchor_quat_w(loader);

    auto rot_ = (init_quat * ref_quat_w).conjugate() * real_quat_w;
    auto rot = rot_.toRotationMatrix().transpose();

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

    // Motion
    motion_ = std::make_shared<MotionLoader_>(motion_file.string());
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
    std::string end_state = "Velocity";
    if (cfg["end_state"]) {
        end_state = cfg["end_state"].as<std::string>();
    }

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    auto onnx_path = policy_dir / "exported" / "policy.onnx";
    env->alg = std::make_unique<isaaclab::OrtRunner>(onnx_path);

    // 从 ONNX metadata 动态解析腰部关节索引，写入该策略的 MotionLoader_
    load_waist_indices_from_onnx(onnx_path.string(), motion_);

    const auto & joy = FSMState::lowstate->joystick;
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return (env->episode_length * env->step_dt) > time_range_[1]; }, // time out
            FSMStringMap.right.at(end_state)
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); }, // bad orientation
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_Mimic::enter()
{
    // set gain (通过 joint_ids_map 将模型顺序重映射到硬件顺序)
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); i++)
    {
        int hw_idx = static_cast<int>(env->robot->data.joint_ids_map[i]);
        lowcmd->msg_.motor_cmd()[hw_idx].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[hw_idx].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[hw_idx].dq() = 0;
        lowcmd->msg_.motor_cmd()[hw_idx].tau() = 0;
    }

    motion = motion_; // set for specific motion
    env->reset();
    // Start policy thread
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        // Initialize timing
        const auto start = clock::now();
        auto sleepTill = start + dt;

        motion->reset(env->robot->data, time_range_[0]);
        auto ref_yaw = isaaclab::yawQuaternion(motion->root_quaternion()).toRotationMatrix();
        auto robot_yaw = isaaclab::yawQuaternion(robot_quat_w(env.get())).toRotationMatrix();
        init_quat = robot_yaw * ref_yaw.transpose();
        env->reset();

        while (policy_thread_running)
        {
            env->robot->update();
            motion->update(env->episode_length * env->step_dt + time_range_[0]);
            env->step();

            // Sleep
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}


void State_Mimic::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}