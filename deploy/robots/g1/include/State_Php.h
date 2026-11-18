#pragma once

#include "FSM/State_RLBase.h"

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace isaaclab { class OrtRunner; }

/**
 * State_Php: Perceptive Humanoid Parkour (PHP) 感知跑酷策略。
 *
 * 观测 (140) = [ actor_obs(93) | velocity_command(15) | depth_latent(32) ]
 *   actor_obs(93) 按训练字母序拼接:
 *     actions(29) @0:29 | base_ang_vel(3) @29:32 | dof_pos(29) @32:61
 *     | dof_vel(29) @61:90 | projected_gravity(3) @90:93   ← 锚点(torso)系
 *   velocity_command(15): one-hot 方向指令 (0=stand,1=forward,...)
 *   depth_latent(32): depth_backbone.onnx( 深度 58x87 -> latent )
 *
 * 动作: target = action_scale(1.0) * action + default_joint_pos
 * 模型: exported/student.onnx (obs[1,140]+time_step[1,1] -> actions[1,29])
 *       exported/depth_backbone.onnx (depth_image[1,58,87] -> depth_latent[1,32])
 * 深度: 从 /dev/shm/depth_raw_shm 读取仿真发布的原始平面深度 (106x60 float32)，
 *       预处理 (crop/clip/双三次抗锯齿/归一化) 后送入骨干网络。
 */
class State_Php : public FSMState
{
public:
    State_Php(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        DetachDepth();
    }

    // ===== 供 REGISTER_OBSERVATION(php_obs) 使用的静态运行期状态 =====
    static std::unique_ptr<isaaclab::OrtRunner> backbone;   // depth_backbone.onnx
    static std::atomic<int> cmd_index;                      // 当前速度指令索引(0=站,1=前,...)
    static std::array<int, 3> waist_policy_idx;             // 腰 yaw/roll/pitch 的 policy 序号
    static std::vector<float> depth_norm;                   // 58*87 归一化深度(复用缓冲)

    // 深度共享内存
    static bool AttachDepth(const std::string& name);
    static void DetachDepth();
    static void ReadRawDepth(std::vector<float>& out_raw);

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_Php)
