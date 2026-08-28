#pragma once

#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include "unitree/dds_wrapper/robots/go2/go2.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"  // 复用 State_Mimic::MotionLoader_ (npz 加载/帧推进)
#include <cnpy.h>

/**
 * State_Backflip: G1 后空翻动作策略（AMP_mjlab, IsaacLab 关节序, 无 time_step）
 *
 * 观测构建：仿照 State_Mimic 使用 observation_manager + env->step()
 *   （motion_command / motion_anchor_ori_b / base_ang_vel / joint_pos_rel /
 *    joint_vel_rel / last_action = 154 维，由 deploy.yaml 自动构建）
 *   - 输入仅 obs[1,154]（无 time_step）
 *   - 动作: target = default_joint_pos + action_scale * actions（IsaacLab 序）
 *   - 关节顺序: IsaacLab（joint_ids_map [0,6,12,...]）== motion.npz 顺序
 *
 * 行为（保留“停末帧 + 重播”控制方案）:
 *   - 进入后先稳定在 motion 最后一帧（站姿），不自动播放
 *   - 按 r（键盘）或再次按下进入组合键（RB+down）触发从头播放
 *   - 播完自动切回 end_state（如 Velocity）
 * 配置（config yaml 中该状态块）:
 *   end_state                      : 播完自动切换的目标状态（如 Velocity）
 *   enable_bad_orientation_check / bad_orientation_limit : 姿态保护
 *   bad_orientation_target / bad_orientation_confirm_ms  : 姿态异常目标/防抖
 */
class State_Backflip : public FSMState
{
public:
    State_Backflip(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

    using MotionLoader_ = State_Mimic::MotionLoader_;

private:
    void advance_frame();
    void replay_motion();                     // 重播：帧回零 + 重新对齐 + 清 last_action
    void recompute_init_quat();               // 重算初始偏航对齐（写入 State_Mimic::init_quat）
    Eigen::Quaternionf robot_torso_quat();    // IMU(pelvis) 朝向 + 腰部关节 -> torso 朝向

    // 摔倒/恢复辅助（与 State_RLBase 一致）
    bool read_base_height(float &height) const;
    bool condition_confirmed(bool condition,
                             std::optional<std::chrono::steady_clock::time_point> &since,
                             int confirm_ms);

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<MotionLoader_> motion_;

    std::thread policy_thread;
    bool policy_thread_running = false;

    int frame_ = 0;               // 当前 motion 帧
    int num_frames_ = 0;

    // 播放模式：true = 进入即从头播放一次（播完回 end_state）；false = 进入停末帧按 r 重播（默认）
    bool auto_play_ = false;

    // 重播信号（RB + A 同时按下）边沿检测
    bool last_restart_key_ = false;
    bool has_played_ = false;   // 已触发过播放（用于播完自动回 end_state）
    int enter_ramp_ = 0;        // 进入平滑（保留字段）

    // 仿真 root 高度数据源（摔倒检测用）；真机为 nullptr（保留备用，当前不依赖高度）
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> sport_mode_state;
    std::optional<std::chrono::steady_clock::time_point> fall_condition_since_;
    std::optional<std::chrono::steady_clock::time_point> bad_orientation_since_;
};


REGISTER_FSM(State_Backflip)
