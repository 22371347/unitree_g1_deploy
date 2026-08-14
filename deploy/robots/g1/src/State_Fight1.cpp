#include "State_Fight1.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "g1_health_logger.h"
#include <onnxruntime_cxx_api.h>

// 从 ONNX 模型的 joint_names metadata 中解析腰部关节索引（模型/IsaacLab 顺序），
// 写入该策略的 MotionLoader_（供 motion_torso_quat 计算使用）
static void load_waist_indices_from_onnx(const std::string& onnx_path,
                                         std::shared_ptr<State_Fight1::MotionLoader_> loader)
{
    // 默认值（IsaacLab G1 模型顺序）
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
        spdlog::warn("[fight1 waist_idx] Failed to read ONNX metadata: {}", e.what());
    }

    loader->waist_yaw_idx   = yaw;
    loader->waist_roll_idx  = roll;
    loader->waist_pitch_idx = pitch;

    spdlog::info("[fight1] waist indices (model order): yaw={}, roll={}, pitch={}",
                 yaw, roll, pitch);
}


State_Fight1::State_Fight1(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if (!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }

    // Motion
    motion_ = std::make_shared<MotionLoader_>(motion_file.string());
    num_frames_ = motion_->num_frames;
    spdlog::info("[fight1] Loaded motion file '{}' frames={} duration={:.2f}s",
                 motion_file.filename().string(), num_frames_, motion_->duration);

    if (cfg["motion_mode"]) motion_mode_ = cfg["motion_mode"].as<std::string>();
    if (cfg["end_state"])   end_state_   = cfg["end_state"].as<std::string>();

    // env + alg
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    auto onnx_path = policy_dir / "exported" / "policy.onnx";
    env->alg = std::make_unique<isaaclab::OrtRunner>(onnx_path);
    load_waist_indices_from_onnx(onnx_path.string(), motion_);

    // 退出检查
    if (motion_mode_ == "once") {
        // motion 播放结束 -> 切换到 end_state
        int end_id = FSMStringMap.right.at(end_state_);
        this->registered_checks.emplace_back(
            std::make_pair(
                [&]()->bool{ return frame_ >= num_frames_ - 1; },
                end_id
            )
        );
    }
    // 姿态异常 -> Passive
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}


void State_Fight1::enter()
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
    frame_ = 0;

    // 初始偏航对齐：把 motion 参考旋转到机器人初始朝向系（消除两者初始偏航差）
    auto ref_yaw   = isaaclab::yawQuaternion(motion_->root_quaternion()).toRotationMatrix();
    auto robot_yaw = isaaclab::yawQuaternion(robot_torso_quat()).toRotationMatrix();
    init_rot_ = robot_yaw * ref_yaw.transpose();

    env->action_manager->reset();
    env->reset();

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
            advance_frame();
            auto obs_map = build_obs_map();
            auto action = env->alg->act(obs_map);
            env->action_manager->process_action(action);
            env->episode_length++;

            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}


void State_Fight1::run()
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


void State_Fight1::advance_frame()
{
    if (motion_mode_ == "loop") {
        frame_ = (frame_ + 1) % num_frames_;
    } else {  // once / hold：播放到最后一帧后保持
        frame_ = std::min(frame_ + 1, num_frames_ - 1);
    }
    motion_->frame = frame_;
}


// 机器人 torso 朝向 = IMU(pelvis) 朝向 * 腰部关节旋转
// 注：模拟器/实机 IMU 位于 pelvis（site "imu" 挂在 pelvis body），故需补偿腰
Eigen::Quaternionf State_Fight1::robot_torso_quat()
{
    auto & motors = FSMState::lowstate->msg_.motor_state();
    Eigen::Quaternionf root_quat = env->robot->data.root_quat_w;

    // 硬件 qpos 顺序: 12=waist_yaw, 13=waist_roll, 14=waist_pitch
    return root_quat
        * Eigen::AngleAxisf(motors[12].q(), Eigen::Vector3f::UnitZ())
        * Eigen::AngleAxisf(motors[13].q(), Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(motors[14].q(), Eigen::Vector3f::UnitY());
}


// motion torso 朝向 = motion root(pelvis) 朝向 * motion 腰部关节旋转
Eigen::Quaternionf State_Fight1::motion_torso_quat()
{
    auto root_quat = motion_->root_quaternion();
    auto joint_pos = motion_->joint_pos();  // 模型顺序

    return root_quat
        * Eigen::AngleAxisf(joint_pos[waist_yaw_idx_],   Eigen::Vector3f::UnitZ())
        * Eigen::AngleAxisf(joint_pos[waist_roll_idx_],  Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(joint_pos[waist_pitch_idx_], Eigen::Vector3f::UnitY());
}


std::unordered_map<std::string, std::vector<float>> State_Fight1::build_obs_map()
{
    // 1) command: motion 目标关节位置(29) + 目标关节速度(29)，模型顺序
    auto ref_q  = motion_->joint_pos();
    auto ref_dq = motion_->joint_vel();

    // 2) motion_anchor_ori_b: inv(robot_torso) * aligned_motion_torso 旋转矩阵前两列
    auto robot_torso = robot_torso_quat();
    auto ref_torso   = motion_torso_quat();
    // 用初始偏航对齐 motion 参考
    Eigen::Quaternionf aligned_ref(init_rot_ * ref_torso.toRotationMatrix());
    Eigen::Quaternionf rel = robot_torso.conjugate() * aligned_ref;
    Eigen::Matrix3f R = rel.toRotationMatrix();
    // 与 sim2sim (numpy C-order) 一致: [R00,R01,R10,R11,R20,R21]
    float ori6[6] = {R(0,0), R(0,1), R(1,0), R(1,1), R(2,0), R(2,1)};

    // 3) base_ang_vel: IMU 角速度（机体系）
    auto & gyro = env->robot->data.root_ang_vel_b;

    // 4/5) joint_pos_rel / joint_vel_rel（模型顺序，data 已经 joint_ids_map 重映射）
    auto & jpos     = env->robot->data.joint_pos;
    auto & jvel     = env->robot->data.joint_vel;
    auto & default_ = env->robot->data.default_joint_pos;

    // 6) last_action: 上一步原始动作
    auto last = env->action_manager->action();

    std::vector<float> obs_vec;
    obs_vec.reserve(154);
    obs_vec.insert(obs_vec.end(), ref_q.data(),  ref_q.data()  + ref_q.size());
    obs_vec.insert(obs_vec.end(), ref_dq.data(), ref_dq.data() + ref_dq.size());
    obs_vec.insert(obs_vec.end(), ori6, ori6 + 6);
    obs_vec.insert(obs_vec.end(), gyro.data(), gyro.data() + 3);
    for (int i = 0; i < static_cast<int>(jpos.size()); ++i) {
        obs_vec.push_back(jpos[i] - default_[i]);
    }
    obs_vec.insert(obs_vec.end(), jvel.data(), jvel.data() + jvel.size());
    obs_vec.insert(obs_vec.end(), last.begin(), last.end());

    std::unordered_map<std::string, std::vector<float>> obs_map;
    obs_map["obs"] = obs_vec;
    obs_map["time_step"] = std::vector<float>{static_cast<float>(frame_)};
    return obs_map;
}
