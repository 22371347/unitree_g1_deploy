#pragma once

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
 * motion 播放模式: once（播完切 end_state）/ loop（循环）/ hold（保持最后一帧）
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
    Eigen::Quaternionf robot_torso_quat();    // IMU(pelvis) 朝向 + 腰部关节 -> torso 朝向
    Eigen::Quaternionf motion_torso_quat();   // motion root(pelvis) + 腰部关节 -> torso 朝向
    std::unordered_map<std::string, std::vector<float>> build_obs_map();

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<MotionLoader_> motion_;

    std::thread policy_thread;
    bool policy_thread_running = false;

    int frame_ = 0;               // 当前 motion 帧
    int num_frames_ = 0;
    std::string motion_mode_ = "hold";   // once / loop / hold
    std::string end_state_ = "Velocity";

    // 腰部关节在模型 joint_names（IsaacLab 序）中的索引，从 ONNX metadata 解析
    int waist_yaw_idx_   = 2;
    int waist_roll_idx_  = 5;
    int waist_pitch_idx_ = 8;

    // 初始偏航对齐（世界系 -> 机器人初始朝向系），消除机器人与 motion 初始朝向差
    Eigen::Matrix3f init_rot_ = Eigen::Matrix3f::Identity();
};


REGISTER_FSM(State_Fight1)
