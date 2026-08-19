// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include "unitree/dds_wrapper/robots/go2/go2.h"
#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        fall_condition_since_.reset();
        recovery_ready_since_.reset();

        // set gain（通过 joint_ids_map 将模型顺序重映射到硬件顺序）
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            int hw_idx = static_cast<int>(env->robot->data.joint_ids_map[i]);
            lowcmd->msg_.motor_cmd()[hw_idx].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[hw_idx].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[hw_idx].dq() = 0;
            lowcmd->msg_.motor_cmd()[hw_idx].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    // 从 unitree_mujoco 发布的 SportModeState 读取 root 高度（仿真用）。
    // 真机 G1 lowstate 无对应高度字段，返回 false。
    bool read_base_height(float &height) const
    {
        if (!sport_mode_state || sport_mode_state->isTimeout())
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(sport_mode_state->mutex_);
        height = static_cast<float>(sport_mode_state->msg_.position()[2]);
        return std::isfinite(height);
    }

    // 条件持续满足 confirm_ms 毫秒后才返回 true（防抖），避免瞬时误触发。
    bool condition_confirmed(
        bool condition,
        std::optional<std::chrono::steady_clock::time_point> &since,
        int confirm_ms
    )
    {
        if (!condition)
        {
            since.reset();
            return false;
        }

        if (confirm_ms <= 0)
        {
            return true;
        }

        const auto now = std::chrono::steady_clock::now();
        if (!since)
        {
            since = now;
            return false;
        }

        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - *since
            ).count();
        return elapsed >= confirm_ms;
    }

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    // 仿真 root 高度数据源（unitree_mujoco 发布）；真机为 nullptr
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState>
        sport_mode_state;

    std::optional<std::chrono::steady_clock::time_point>
        fall_condition_since_;
    std::optional<std::chrono::steady_clock::time_point>
        recovery_ready_since_;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
