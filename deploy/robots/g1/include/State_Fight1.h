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
 * State_Fight1: WBT G1 全身动作跟踪策略 (model_46500)
 *
 * 与 State_Mimic 类似，但观测结构与输入不同（按 sim2sim 权威实现）:
 *   - 观测 154 维（无历史、无归一化）:
 *       [command(58) + motion_anchor_ori_b(6) + base_ang_vel(3)
 *        + joint_pos_rel(29) + joint_vel_rel(29) + actions(29)]
 *     command          = motion 目标关节位置(29) + 目标关节速度(29)，模型顺序
 *     motion_anchor_ori_b = inv(robot_torso)*aligned_motion_torso 旋转矩阵前两列
 *   - 额外输入 time_step = motion 帧索引 (0~158)
 *   - 动作: target = default_joint_pos + action_scale * action（模型顺序）
 *   - 关节顺序: IsaacLab 模型顺序 == motion.npz 顺序，经 joint_ids_map 重映射
 *
 * 行为: 进入即从头播放 motion；RB + A（或键盘 r）可随时重新播放；
 *      播完自动切回 end_state（如 Fight）。
 * 配置（config yaml 中该状态块）:
 *   end_state                      : 播完自动切换的目标状态（如 Fight）
 *   enable_bad_orientation_check / bad_orientation_limit : 姿态保护
 *   fall_detection                 : 摔倒自动切目标（如 Recovery）
 */
class State_Fight1 : public FSMState
{
public:
    State_Fight1(int state_mode, std::string state_string);

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
    Eigen::Quaternionf robot_torso_quat();    // IMU(pelvis) 朝向 + 腰部关节 -> torso 朝向
    Eigen::Quaternionf motion_torso_quat();   // motion root(pelvis) + 腰部关节 -> torso 朝向
    std::unordered_map<std::string, std::vector<float>> build_obs_map();

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

    // 腰部关节在模型 joint_names（IsaacLab 序）中的索引，从 ONNX metadata 解析
    int waist_yaw_idx_   = 2;
    int waist_roll_idx_  = 5;
    int waist_pitch_idx_ = 8;

    // 初始偏航对齐（世界系 -> 机器人初始朝向系），消除机器人与 motion 初始朝向差
    Eigen::Matrix3f init_rot_ = Eigen::Matrix3f::Identity();

    // 重播信号（RB + A 同时按下）边沿检测
    bool last_restart_key_ = false;
    bool has_played_ = false;   // 已触发过播放（用于播完自动回 end_state）

    // 仿真 root 高度数据源（摔倒检测用）；真机为 nullptr（保留备用，当前不依赖高度）
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> sport_mode_state;
    std::optional<std::chrono::steady_clock::time_point> fall_condition_since_;
    std::optional<std::chrono::steady_clock::time_point> bad_orientation_since_;
};


REGISTER_FSM(State_Fight1)
