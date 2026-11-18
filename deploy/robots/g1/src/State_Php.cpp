#include "State_Php.h"

#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/algorithms/algorithms.h"
#include "PhpDepthPreprocess.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace
{
constexpr int   RAW_W = php_depth::RAW_W, RAW_H = php_depth::RAW_H;
constexpr float FAR = php_depth::FAR;
constexpr int   NUM_DOF = 29;
constexpr int   ACTOR_DIM = 93, CMD_DIM = 15, LATENT_DIM = 32, OBS_DIM = 140;

const char* kShmName = "depth_raw_shm";

// ---------------- 四元数（w,x,y,z） ----------------
inline Eigen::Quaternionf quat_axis_angle(float angle, const Eigen::Vector3f& axis)
{
    return Eigen::Quaternionf(Eigen::AngleAxisf(angle, axis.normalized()));
}

} // anonymous namespace


// ============================================================================
// 静态成员
// ============================================================================
std::unique_ptr<isaaclab::OrtRunner> State_Php::backbone = nullptr;
std::atomic<int> State_Php::cmd_index{0};
std::array<int, 3> State_Php::waist_policy_idx = {-1, -1, -1};
std::vector<float> State_Php::depth_norm;

namespace {
float* g_depth_shm = nullptr;
size_t g_depth_bytes = 0;
std::vector<float> g_raw_depth(static_cast<size_t>(RAW_W) * RAW_H, FAR);
bool g_debug = false;            // config: FSM.Php.debug_log
bool g_disable_depth = false;    // config: FSM.Php.disable_depth (A/B 测试用)
std::atomic<double> g_policy_hz{0.0};   // 实测策略循环频率
std::atomic<float> g_latent_norm{0.0f}; // 最近一次深度 latent 的范数
bool g_gravity_pelvis = false;         // config: FSM.Php.gravity_mode == "pelvis"
std::atomic<float> g_gx{0.0f}, g_gy{0.0f}, g_gz{0.0f};  // 最近一次重力向量
} // namespace

bool State_Php::AttachDepth(const std::string& name)
{
    DetachDepth();
    int fd = shm_open(name.c_str(), O_RDONLY, 0);
    if (fd < 0) {
        spdlog::warn("[State_Php] depth shm '{}' not found; using all-far depth.", name);
        return false;
    }
    g_depth_bytes = static_cast<size_t>(RAW_W) * RAW_H * sizeof(float);
    void* p = mmap(nullptr, g_depth_bytes, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        spdlog::warn("[State_Php] mmap depth shm failed.");
        return false;
    }
    g_depth_shm = static_cast<float*>(p);
    spdlog::info("[State_Php] attached depth shm '{}' ({}x{}).", name, RAW_W, RAW_H);
    return true;
}

void State_Php::DetachDepth()
{
    if (g_depth_shm) {
        munmap(g_depth_shm, g_depth_bytes);
        g_depth_shm = nullptr;
    }
}

void State_Php::ReadRawDepth(std::vector<float>& out_raw)
{
    if (g_depth_shm) {
        std::memcpy(g_raw_depth.data(), g_depth_shm, g_depth_bytes);
    } else {
        std::fill(g_raw_depth.begin(), g_raw_depth.end(), FAR);
    }
    out_raw = g_raw_depth;
}


// ============================================================================
// 观测项: [actions, base_ang_vel, dof_pos, dof_vel, projected_gravity,
//          velocity_command(15), depth_latent(32)] = 140
// ============================================================================
namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(php_obs)
{
    auto& d = env->robot->data;
    std::vector<float> obs;
    obs.reserve(OBS_DIM);

    // 1) actions (29) — 上一步原始动作（policy 顺序）
    auto action = env->action_manager->action();
    obs.insert(obs.end(), action.begin(), action.begin() + std::min<size_t>(action.size(), NUM_DOF));

    // 2) base_ang_vel (3) — IMU 陀螺
    for (int i = 0; i < 3; ++i) obs.push_back(d.root_ang_vel_b[i]);

    // 3) dof_pos (29) — 相对默认
    for (int i = 0; i < NUM_DOF; ++i) obs.push_back(d.joint_pos[i] - d.default_joint_pos[i]);

    // 4) dof_vel (29)
    for (int i = 0; i < NUM_DOF; ++i) obs.push_back(d.joint_vel[i]);

    // 5) projected_gravity (3)
    Eigen::Vector3f gravity;
    if (g_gravity_pelvis) {
        gravity = d.projected_gravity_b;   // pelvis 系（与 Velocity 策略一致）
    } else {
        // torso/锚点系: base ⊗ waist_yaw ⊗ waist_roll ⊗ waist_pitch
        Eigen::Quaternionf anchor = d.root_quat_w;
        const Eigen::Vector3f ax_yaw(0, 0, 1), ax_roll(1, 0, 0), ax_pitch(0, 1, 0);
        if (State_Php::waist_policy_idx[0] >= 0)
            anchor = anchor * quat_axis_angle(d.joint_pos[State_Php::waist_policy_idx[0]], ax_yaw);
        if (State_Php::waist_policy_idx[1] >= 0)
            anchor = anchor * quat_axis_angle(d.joint_pos[State_Php::waist_policy_idx[1]], ax_roll);
        if (State_Php::waist_policy_idx[2] >= 0)
            anchor = anchor * quat_axis_angle(d.joint_pos[State_Php::waist_policy_idx[2]], ax_pitch);
        gravity = anchor.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    }
    g_gx.store(gravity.x()); g_gy.store(gravity.y()); g_gz.store(gravity.z());
    obs.push_back(gravity.x());
    obs.push_back(gravity.y());
    obs.push_back(gravity.z());

    // 6) velocity_command (15) — one-hot，由 cmd_index 决定（方向键控制）
    {
        int ci = State_Php::cmd_index.load();
        if (ci < 0 || ci >= CMD_DIM) ci = 0;
        for (int i = 0; i < CMD_DIM; ++i) obs.push_back(i == ci ? 1.0f : 0.0f);
    }

    // 7) depth_latent (32)
    if (State_Php::backbone && !g_disable_depth) {
        std::vector<float> raw;
        State_Php::ReadRawDepth(raw);
        php_depth::preprocess_depth(raw, State_Php::depth_norm);
        auto latent = State_Php::backbone->act({{"depth_image", State_Php::depth_norm}});
        if (static_cast<int>(latent.size()) >= LATENT_DIM) {
            float ss = 0.0f;
            for (int i = 0; i < LATENT_DIM; ++i) ss += latent[i] * latent[i];
            g_latent_norm.store(std::sqrt(ss));
            obs.insert(obs.end(), latent.begin(), latent.begin() + LATENT_DIM);
        } else {
            obs.insert(obs.end(), LATENT_DIM, 0.0f);
        }
    } else {
        obs.insert(obs.end(), LATENT_DIM, 0.0f);   // 禁用深度: 零 latent
    }

    return obs;
}

} // namespace mdp
} // namespace isaaclab


// ============================================================================
// State_Php 实现
// ============================================================================
State_Php::State_Php(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"), articulation);

    // 学生策略（obs[1,140] + time_step[1,1] -> actions[1,29]）
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "student.onnx");
    // 深度编码骨干（depth_image[1,58,87] -> depth_latent[1,32]）
    backbone = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "depth_backbone.onnx");

    // 腰关节在 policy 顺序中的索引（硬件 12/13/14）
    waist_policy_idx = {-1, -1, -1};
    const auto& map = env->robot->data.joint_ids_map;
    const int hw_waist[3] = {12, 13, 14};
    for (int k = 0; k < 3; ++k) {
        for (size_t i = 0; i < map.size(); ++i) {
            if (static_cast<int>(map[i]) == hw_waist[k]) { waist_policy_idx[k] = static_cast<int>(i); break; }
        }
    }

    // 速度指令：进入后由方向键切换，默认 0（站立）
    cmd_index.store(0);
    g_debug = cfg["debug_log"] ? cfg["debug_log"].as<bool>() : false;
    g_disable_depth = cfg["disable_depth"] ? cfg["disable_depth"].as<bool>() : false;
    if (g_disable_depth) spdlog::warn("[State_Php] disable_depth=true -> feeding zero depth latent (A/B test)");
    std::string gm = cfg["gravity_mode"] ? cfg["gravity_mode"].as<std::string>() : "torso";
    g_gravity_pelvis = (gm == "pelvis");
    spdlog::info("[State_Php] gravity_mode={}", gm);

    // 安全检测：姿态过差回 Passive
    registered_checks.emplace_back(
        std::make_pair(
            [&]() -> bool { return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")));
}

void State_Php::enter()
{
    AttachDepth(kShmName);
    cmd_index.store(0);   // 进入时为 0 指令（站立）
    FSMState::reset_joystick_request.store(true);   // 清空键盘虚拟摇杆，避免带着上一次的速度指令进入
    spdlog::info("[State_Php] entered, cmd=0(stand). Keys: up=fwd(toggle), down=back, left/right=strafe,"
                 " q/e=turn, z=stop; or gamepad stick");

    for (int i = 0; i < env->robot->data.joint_stiffness.size(); i++) {
        int hw_idx = static_cast<int>(env->robot->data.joint_ids_map[i]);
        lowcmd->msg_.motor_cmd()[hw_idx].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[hw_idx].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[hw_idx].dq() = 0;
        lowcmd->msg_.motor_cmd()[hw_idx].tau() = 0;
    }

    env->reset();

    policy_thread_running = true;
    policy_thread = std::thread([this] {
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);
        auto next = clock::now() + dt;

        // 预热：用进入时的真实状态再重置一次观测缓冲
        env->robot->update();
        env->reset();

        auto last_t0 = clock::now();
        double hz_acc = 0.0;
        int hz_n = 0;
        while (policy_thread_running) {
            auto t0 = clock::now();
            try {
                env->robot->update();
                auto obs = env->observation_manager->compute();
                obs["time_step"] = std::vector<float>{0.0f};   // PHP: time_step 恒 0
                auto action = env->alg->act(obs);
                env->action_manager->process_action(action);
            } catch (const std::exception& e) {
                spdlog::error("[State_Php] policy step error: {}", e.what());
            }
            // 实测控制频率
            double inst = std::chrono::duration<double>(t0 - last_t0).count();
            last_t0 = t0;
            if (inst > 1e-6) {
                hz_acc += 1.0 / inst;
                if (++hz_n >= 50) { g_policy_hz.store(hz_acc / hz_n); hz_acc = 0.0; hz_n = 0; }
            }
            // 固定节拍：若落后就丢弃错过的节拍，绝不为“追帧”而爆发式补跑
            auto now = clock::now();
            if (next < now) next = now;
            std::this_thread::sleep_until(next);
            next += dt;
        }
    });
}

void State_Php::run()
{
    // 速度指令：与其他策略统一 —— 一律走“摇杆”语义（仿真里由 CtrlFSM 把键盘 toggle 注入摇杆，
    // 实机由手柄摇杆直接提供），再把摇杆方向映射成 PHP 的 15 维 one-hot 方向指令：
    //   ↑=前进（再按一下归零=站立） ↓=后退 ←/→=左右横移 q/e=左右转向 z=停止
    //   ↑+← / ↑+→ = 左前 45° / 右前 45°
    int ni = 0; // 0 = stand
    if (FSMState::lowstate) {
        auto & joy = FSMState::lowstate->joystick;   // Axis::operator()() 为非 const，需非 const 引用
        constexpr float kDead = 0.2f; // 摇杆死区（键盘注入恒为 ±5，实机推杆超过死区即可）
        const float ly = joy.ly(), lx = joy.lx(), rx = joy.rx();
        const bool fwd = (ly >  kDead), back = (ly < -kDead);
        const bool lft = (lx < -kDead), rgt  = (lx >  kDead);
        if (std::fabs(rx) > kDead) ni = (rx > 0) ? 5 : 3;              // 原地转：right_90 / left_90
        else if (fwd)              ni = lft ? 2 : (rgt ? 4 : 1);       // 前 / 左前45° / 右前45°
        else if (back)             ni = 11;                            // 后退
        else if (rgt)              ni = 5;                             // 右横移 90°
        else if (lft)              ni = 3;                             // 左横移 90°
    }
    if (ni != cmd_index.load()) {
        cmd_index.store(ni);
        spdlog::info("[State_Php] command -> {}", ni);
    }

    auto action = env->action_manager->processed_actions();
    const auto& map = env->robot->data.joint_ids_map;
    for (size_t i = 0; i < map.size(); ++i) {
        lowcmd->msg_.motor_cmd()[map[i]].q() = action[i];
    }

    if (g_debug) {
        static int dbg_cnt = 0;
        if ((++dbg_cnt % 50) == 0) {
            const auto& jp = env->robot->data.joint_pos;
            const auto& av = env->robot->data.root_ang_vel_b;
            float amax = 0.0f;
            for (float a : action) amax = std::max(amax, std::fabs(a));
            spdlog::info("[State_Php] cmd={} depth={} hz={:.1f} dn={:.2f} g=({:.2f},{:.2f},{:.2f}) | kneeL tgt={:.3f} act={:.3f} | kneeR tgt={:.3f} act={:.3f} | |a|max={:.2f} angvel={:.2f}",
                         cmd_index.load(), (g_depth_shm ? 1 : 0), g_policy_hz.load(), g_latent_norm.load(),
                         g_gx.load(), g_gy.load(), g_gz.load(),
                         action[3], jp[3], action[9], jp[9],
                         amax, std::sqrt(av[0]*av[0] + av[1]*av[1] + av[2]*av[2]));
        }
    }
}
