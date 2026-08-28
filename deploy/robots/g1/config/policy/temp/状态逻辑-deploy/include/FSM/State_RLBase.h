// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
// 第三个状态：根据当前状态名称找到 policy_dir，加载 params/deploy.yaml 和 exported/policy.onnx

#pragma once

#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
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

        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
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

    // unitree_mujoco publishes the simulated root position on this topic.
    // Real G1 lowstate normally has no equivalent root-height field.
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
