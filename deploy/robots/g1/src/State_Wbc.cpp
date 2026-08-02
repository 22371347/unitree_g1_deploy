#include "State_Wbc.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <cmath>
#include <deque>

// ============================================================================
// 静态成员定义
// ============================================================================
std::shared_ptr<State_Wbc::MotionLoader_> State_Wbc::motion = nullptr;
std::vector<float> State_Wbc::robot_state_hist;
unsigned int State_Wbc::refer_idx = 0;
int State_Wbc::start_idx = 0;
int State_Wbc::end_idx = -1;
bool State_Wbc::refill_history = true;

namespace
{
// ============================================================================
// 四元数/旋转数学工具（移植自 wbc_fsm mathTools.h，确保观测完全一致）
// 四元数格式均为 (w, x, y, z)
// ============================================================================

std::vector<float> q_to_vec(const Eigen::Quaternionf& q)
{
    return {q.w(), q.x(), q.y(), q.z()};
}
std::vector<float> v_to_vec(const Eigen::Vector3f& v)
{
    return {v.x(), v.y(), v.z()};
}

Eigen::Matrix3f matrix_from_quat(const std::vector<float>& quaternion)
{
    float w = quaternion[0], x = quaternion[1], y = quaternion[2], z = quaternion[3];
    float two_s = 2.0f / (w * w + x * x + y * y + z * z);

    Eigen::Matrix3f R;
    R << 1.0f - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w),
         two_s * (x * y + z * w), 1.0f - two_s * (x * x + z * z), two_s * (y * z - x * w),
         two_s * (x * z - y * w), two_s * (y * z + x * w), 1.0f - two_s * (x * x + y * y);
    return R;
}

std::vector<float> quat_conjugate(const std::vector<float>& q)
{
    return {q[0], -q[1], -q[2], -q[3]};
}

std::vector<float> quat_inv(const std::vector<float>& q, float eps = 1e-9f)
{
    float nsq = std::max(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3], eps);
    auto qc = quat_conjugate(q);
    return {qc[0]/nsq, qc[1]/nsq, qc[2]/nsq, qc[3]/nsq};
}

std::vector<float> quat_multiply(const std::vector<float>& q1, const std::vector<float>& q2)
{
    float w1=q1[0], x1=q1[1], y1=q1[2], z1=q1[3];
    float w2=q2[0], x2=q2[1], y2=q2[2], z2=q2[3];
    return {
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    };
}

std::vector<float> quat_mul(const std::vector<float>& q1, const std::vector<float>& q2)
{
    return quat_multiply(q1, q2);
}

std::vector<float> quat_apply(const std::vector<float>& quat, const std::vector<float>& vec)
{
    float w = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];
    float vx = vec[0], vy = vec[1], vz = vec[2];

    float tx = 2.0f * (qy * vz - qz * vy);
    float ty = 2.0f * (qz * vx - qx * vz);
    float tz = 2.0f * (qx * vy - qy * vx);

    float cross_x = qy * tz - qz * ty;
    float cross_y = qz * tx - qx * tz;
    float cross_z = qx * ty - qy * tx;

    return {vx + w * tx + cross_x, vy + w * ty + cross_y, vz + w * tz + cross_z};
}

std::vector<float> quat_apply_inverse(const std::vector<float>& quat, const std::vector<float>& vec)
{
    std::vector<float> xyz = {quat[1], quat[2], quat[3]};
    std::vector<float> cross1(3);
    cross1[0] = xyz[1] * vec[2] - xyz[2] * vec[1];
    cross1[1] = xyz[2] * vec[0] - xyz[0] * vec[2];
    cross1[2] = xyz[0] * vec[1] - xyz[1] * vec[0];

    std::vector<float> t = {2.0f * cross1[0], 2.0f * cross1[1], 2.0f * cross1[2]};

    std::vector<float> cross2(3);
    cross2[0] = xyz[1] * t[2] - xyz[2] * t[1];
    cross2[1] = xyz[2] * t[0] - xyz[0] * t[2];
    cross2[2] = xyz[0] * t[1] - xyz[1] * t[0];

    return {
        vec[0] - quat[0] * t[0] + cross2[0],
        vec[1] - quat[0] * t[1] + cross2[1],
        vec[2] - quat[0] * t[2] + cross2[2]
    };
}

std::vector<float> yaw_quat(const std::vector<float>& quat)
{
    float qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];
    float yaw = std::atan2(2.0f * (qw * qz + qx * qy), 1.0f - 2.0f * (qy * qy + qz * qz));

    std::vector<float> quat_yaw = {std::cos(yaw / 2.0f), 0.0f, 0.0f, std::sin(yaw / 2.0f)};
    float norm = std::sqrt(quat_yaw[0] * quat_yaw[0] + quat_yaw[3] * quat_yaw[3]);
    quat_yaw[0] /= norm;
    quat_yaw[3] /= norm;
    return quat_yaw;
}

/// 相对变换: 计算坐标系2相对坐标系1的位姿（结果在坐标系1局部系）
std::pair<std::vector<float>, std::vector<float>> subtract_frame_transforms(
    const std::vector<float>& t01, const std::vector<float>& q01,
    const std::vector<float>& t02, const std::vector<float>& q02)
{
    std::vector<float> q10 = quat_inv(q01);
    std::vector<float> q12 = quat_mul(q10, q02);

    std::vector<float> diff = {t02[0]-t01[0], t02[1]-t01[1], t02[2]-t01[2]};
    std::vector<float> t12 = quat_apply(q10, diff);

    return {t12, q12};
}

constexpr int kNumDof = 29;
constexpr int kRobotStateDim = 93;      // 3+3+29+29+29
constexpr int kReferenceDim = 67;       // 29+29+3+6
constexpr int kHistoryLength = 4;
constexpr int kFrameInterval = 5;       // 参考锚点速度的帧间隔（与 wbc_fsm 一致）
constexpr float kClipObs = 100.0f;
const std::vector<float> kGravityVec = {0.0f, 0.0f, -1.0f};

} // anonymous namespace


namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(wbc_obs)
{
    auto loader = State_Wbc::motion;
    if (!loader) {
        throw std::runtime_error("[wbc_obs] motion loader not initialized.");
    }

    // ========================================================================
    // 1. 当前机器人状态 (93 维)
    // ========================================================================
    std::vector<float> current_state;
    current_state.reserve(kRobotStateDim);

    // 1a. 基座角速度 (3) — wbc_fsm 中 scale_lin_vel=1, scale_ang_vel=1
    const auto& ang_vel = env->robot->data.root_ang_vel_b;
    current_state.insert(current_state.end(), ang_vel.data(), ang_vel.data() + 3);

    // 1b. 投影重力 (3)
    const auto& gravity = env->robot->data.projected_gravity_b;
    current_state.insert(current_state.end(), gravity.data(), gravity.data() + 3);

    // 1c. 关节位置偏差 (29) — joint_pos[i] 已是 joint_ids_map（policy）顺序
    for (int i = 0; i < kNumDof; ++i) {
        current_state.push_back(env->robot->data.joint_pos[i] - env->robot->data.default_joint_pos[i]);
    }

    // 1d. 关节速度 (29)
    for (int i = 0; i < kNumDof; ++i) {
        current_state.push_back(env->robot->data.joint_vel[i]);
    }

    // 1e. 上一步动作 (29) — 策略输出（未缩放）
    auto action = env->action_manager->action();
    current_state.insert(current_state.end(), action.begin(), action.end());

    // ========================================================================
    // 2. 维护机器人状态历史 (4×93)
    // ========================================================================
    if (State_Wbc::refill_history || State_Wbc::robot_state_hist.empty()) {
        State_Wbc::robot_state_hist.clear();
        for (int i = 0; i < kHistoryLength; ++i) {
            State_Wbc::robot_state_hist.insert(State_Wbc::robot_state_hist.end(),
                                               current_state.begin(), current_state.end());
        }
        State_Wbc::refill_history = false;
    } else {
        State_Wbc::robot_state_hist.erase(State_Wbc::robot_state_hist.begin(),
                                          State_Wbc::robot_state_hist.begin() + kRobotStateDim);
        State_Wbc::robot_state_hist.insert(State_Wbc::robot_state_hist.end(),
                                           current_state.begin(), current_state.end());
    }

    // ========================================================================
    // 3. 参考轨迹 (67 维)
    // ========================================================================
    // 机器人基座四元数（世界）
    Eigen::Quaternionf base_q = env->robot->data.root_quat_w;
    std::vector<float> base_quat = q_to_vec(base_q);
    std::vector<float> base_yaw_quat = yaw_quat(base_quat);

    int end_idx = (State_Wbc::end_idx < 0) ? (loader->num_frames - 1) : State_Wbc::end_idx;
    int idx = static_cast<int>(State_Wbc::refer_idx);
    int last_idx = idx - kFrameInterval;
    idx = std::clamp(idx, 1, end_idx);
    last_idx = std::clamp(last_idx, 1, end_idx);

    // 当前参考帧
    Eigen::Vector3f cur_anchor_pos = loader->anchor_pos(idx);
    Eigen::Quaternionf cur_anchor_quat = loader->anchor_quat(idx);
    const Eigen::VectorXf& cur_dof_pos = loader->dof_pos(idx);
    const Eigen::VectorXf& cur_dof_vel = loader->dof_vel(idx);

    // yaw 对齐：base_yaw * ref_yaw^-1，再施加到参考锚点朝向上
    std::vector<float> ref_yaw_quat = yaw_quat(q_to_vec(cur_anchor_quat));
    std::vector<float> yaw_quat_delta = quat_multiply(base_yaw_quat, quat_conjugate(ref_yaw_quat));
    std::vector<float> aligned_anchor_quat = quat_multiply(yaw_quat_delta, q_to_vec(cur_anchor_quat));

    // 上一参考帧（用于计算锚点相对速度/姿态变化）
    Eigen::Vector3f last_anchor_pos = loader->anchor_pos(last_idx);
    Eigen::Quaternionf last_anchor_quat = loader->anchor_quat(last_idx);

    // 相对变换: last → cur
    auto [cur_target_pos, cur_target_quat] = subtract_frame_transforms(
        v_to_vec(last_anchor_pos), q_to_vec(last_anchor_quat),
        v_to_vec(cur_anchor_pos), q_to_vec(cur_anchor_quat));

    // 6D 朝向表示（旋转矩阵前两列展平）
    Eigen::Matrix3f mat = matrix_from_quat(cur_target_quat);
    std::vector<float> tgt_anchor_ori_b = {
        mat(0,0), mat(0,1),
        mat(1,0), mat(1,1),
        mat(2,0), mat(2,1)};

    // 拼接参考轨迹 (67)
    std::vector<float> mimic_obs;
    mimic_obs.reserve(kReferenceDim);
    mimic_obs.insert(mimic_obs.end(), cur_dof_pos.data(), cur_dof_pos.data() + cur_dof_pos.size());
    mimic_obs.insert(mimic_obs.end(), cur_dof_vel.data(), cur_dof_vel.data() + cur_dof_vel.size());
    mimic_obs.insert(mimic_obs.end(), cur_target_pos.begin(), cur_target_pos.end());
    mimic_obs.insert(mimic_obs.end(), tgt_anchor_ori_b.begin(), tgt_anchor_ori_b.end());

    // ========================================================================
    // 4. 拼接完整观测 [参考67, 历史372] = 439
    // ========================================================================
    std::vector<float> obs;
    obs.reserve(kReferenceDim + kRobotStateDim * kHistoryLength);
    obs.insert(obs.end(), mimic_obs.begin(), mimic_obs.end());
    obs.insert(obs.end(), State_Wbc::robot_state_hist.begin(), State_Wbc::robot_state_hist.end());

    for (auto& v : obs) {
        v = std::max(-kClipObs, std::min(v, kClipObs));
    }

    return obs;
}

} // namespace mdp
} // namespace isaaclab


// ============================================================================
// State_Wbc 实现
// ============================================================================

State_Wbc::State_Wbc(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    // 运动文件
    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if (!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }
    motion_ = std::make_shared<MotionLoader_>(motion_file.string());
    spdlog::info("Loaded WBC motion '{}' with {} frames ({:.2f}s)",
                 motion_file.stem().string(), motion_->num_frames, motion_->duration);
    motion = motion_;

    // 起始/结束帧（可选配置）
    start_idx = cfg["start_idx"] ? cfg["start_idx"].as<int>() : 0;
    end_idx   = cfg["end_idx"]   ? cfg["end_idx"].as<int>()   : -1;
    start_idx = std::clamp(start_idx, 0, motion_->num_frames - 1);

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // 安全检测：姿态过差时回到 Passive
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_Wbc::enter()
{
    // 设置 PD 增益（通过 joint_ids_map 重映射到硬件顺序）
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); i++)
    {
        int hw_idx = static_cast<int>(env->robot->data.joint_ids_map[i]);
        lowcmd->msg_.motor_cmd()[hw_idx].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[hw_idx].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[hw_idx].dq() = 0;
        lowcmd->msg_.motor_cmd()[hw_idx].tau() = 0;
    }

    // 初始化观测状态
    refer_idx = static_cast<unsigned int>(start_idx);
    refill_history = true;
    robot_state_hist.clear();
    motion = motion_;

    env->reset();

    // 启动策略线程（50Hz）
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        const auto start = clock::now();
        auto sleepTill = start + dt;

        env->reset(); // 再次重置，确保历史使用进入时的真实状态

        while (policy_thread_running)
        {
            env->robot->update();   // 刷新机器人状态（读取 lowstate）
            refer_idx++;            // 推进参考轨迹帧
            env->step();            // 观测 → 推理 → 更新动作
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_Wbc::run()
{
    // 将策略最新动作（policy 顺序）映射到硬件电机目标位置
    auto action = env->action_manager->processed_actions();
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
