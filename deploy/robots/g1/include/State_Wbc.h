#pragma once

#include "FSM/State_RLBase.h"
#include <cnpy.h>

/**
 * State_Wbc: wbc_fsm 动作追踪策略（LAFAN1 WBC）
 *
 * 与 State_Mimic 类似，但观测结构不同：
 *   - 观测: [参考轨迹67 + 机器人状态历史 4×93] = 439 维
 *   - 参考轨迹: 目标关节pos(29) + 目标关节vel(29) + 锚点相对pos(3) + 锚点相对ori(6)
 *   - 状态历史: 每帧 = 角速度(3) + 投影重力(3) + 关节pos偏差(29) + 关节vel(29) + 上一步动作(29)
 *   - 动作: 29 维全身，target = action*0.25 + default_dof_pos
 */
class State_Wbc : public FSMState
{
public:
    State_Wbc(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

    class MotionLoader_;

    static std::shared_ptr<MotionLoader_> motion; // for obs computation

    // ========== 观测状态（跨步保持，供 REGISTER_OBSERVATION 使用） ==========
    static std::vector<float> robot_state_hist; // 4 帧 × 93 = 372
    static unsigned int refer_idx;              // 参考轨迹当前帧索引
    static int start_idx;                       // 起始帧
    static int end_idx;                         // 结束帧（-1 表示运动末尾）
    static bool refill_history;                 // 进入状态时置 true，触发历史重建

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<MotionLoader_> motion_; // for saving

    std::thread policy_thread;
    bool policy_thread_running = false;
};


/**
 * 运动数据加载器。
 * 从 NPZ 加载 LAFAN1 重定向运动数据，仅保留 wbc 观测需要的锚点(索引0)位姿与关节轨迹。
 */
class State_Wbc::MotionLoader_
{
public:
    MotionLoader_(std::string motion_file)
    : dt(1.0f / 50.0f)
    {
        load_data_from_npz(motion_file);
        num_frames = static_cast<int>(dof_positions.size());
        duration = num_frames * dt;
        frame = 0;
    }

    void load_data_from_npz(const std::string& motion_file)
    {
        cnpy::npz_t npz_data = cnpy::npz_load(motion_file);

        auto body_pos_w  = npz_data["body_pos_w"];   // [frame, body_id, 3]
        auto body_quat_w = npz_data["body_quat_w"];  // [frame, body_id, 4]
        auto joint_pos   = npz_data["joint_pos"];    // [frame, dof]
        auto joint_vel   = npz_data["joint_vel"];    // [frame, dof]

        anchor_positions.clear();
        anchor_quaternions.clear();
        dof_positions.clear();
        dof_velocities.clear();

        const size_t num_frames_npz = body_pos_w.shape[0];
        const size_t num_bodies     = body_pos_w.shape[1];
        const size_t anchor_idx     = std::min<size_t>(anchor_body_index, num_bodies - 1);

        const size_t body_stride_pos  = num_bodies * 3;
        const size_t body_stride_quat = num_bodies * 4;

        for (size_t i = 0; i < num_frames_npz; i++)
        {
            // 锚点位置（世界坐标系）
            Eigen::Vector3f anchor_pos(
                body_pos_w.data<float>()[i * body_stride_pos + anchor_idx * 3 + 0],
                body_pos_w.data<float>()[i * body_stride_pos + anchor_idx * 3 + 1],
                body_pos_w.data<float>()[i * body_stride_pos + anchor_idx * 3 + 2]);
            anchor_positions.push_back(anchor_pos);

            // 锚点四元数（世界坐标系, w,x,y,z）
            Eigen::Quaternionf anchor_quat(
                body_quat_w.data<float>()[i * body_stride_quat + anchor_idx * 4 + 0],
                body_quat_w.data<float>()[i * body_stride_quat + anchor_idx * 4 + 1],
                body_quat_w.data<float>()[i * body_stride_quat + anchor_idx * 4 + 2],
                body_quat_w.data<float>()[i * body_stride_quat + anchor_idx * 4 + 3]);
            anchor_quaternions.push_back(anchor_quat.normalized());

            Eigen::VectorXf joint_position(joint_pos.shape[1]);
            Eigen::VectorXf joint_velocity(joint_vel.shape[1]);
            for (int j = 0; j < joint_pos.shape[1]; j++) {
                joint_position[j] = joint_pos.data<float>()[i * joint_pos.shape[1] + j];
                joint_velocity[j] = joint_vel.data<float>()[i * joint_vel.shape[1] + j];
            }
            dof_positions.push_back(joint_position);
            dof_velocities.push_back(joint_velocity);
        }
    }

    void update(unsigned int idx)
    {
        frame = std::min(static_cast<int>(idx), num_frames - 1);
    }

    // ---- 按帧索引访问 ----
    Eigen::Vector3f anchor_pos(int idx) const
    {
        int i = std::clamp(idx, 0, num_frames - 1);
        return anchor_positions[i];
    }
    Eigen::Quaternionf anchor_quat(int idx) const
    {
        int i = std::clamp(idx, 0, num_frames - 1);
        return anchor_quaternions[i];
    }
    const Eigen::VectorXf& dof_pos(int idx) const
    {
        int i = std::clamp(idx, 0, num_frames - 1);
        return dof_positions[i];
    }
    const Eigen::VectorXf& dof_vel(int idx) const
    {
        int i = std::clamp(idx, 0, num_frames - 1);
        return dof_velocities[i];
    }

    float dt;
    int num_frames;
    float duration;
    int frame;
    size_t anchor_body_index = 0;  // wbc_fsm 使用 root (索引0) 作为锚点

    std::vector<Eigen::Vector3f> anchor_positions;
    std::vector<Eigen::Quaternionf> anchor_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;
};


REGISTER_FSM(State_Wbc)
