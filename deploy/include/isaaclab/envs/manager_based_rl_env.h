// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include <atomic>
#include <chrono>
#include "isaaclab/utils/utils.h"

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        command.assign(3, 0.0f); // 限速后的速度指令复位
        robot->update();
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute();

        // ========== 调试模式：固定输入开关 ==========
        const bool DEBUG_FIXED_INPUT = false;  // true: 使用固定观测, false: 使用真实观测
        const bool DEBUG_OBS_OUTPUT = false;  // true: 使用固定观测, false: 使用真实观测
                   
        if (DEBUG_FIXED_INPUT) {
                        std::vector<float> fixed_obs_vec = {
            0.0648432, 0.106284, -0.0033956, 0.0431186, -0.144412, -0.0231454, 0.166343, -0.409542, 0.0719466, 0.159991, 0.204351, -0.0304434, 0.23664, -0.0800348, -0.173099, 1.18581, -1.12624, -0.0234694, 0.0399944, 0.551842, -0.861379, 0.83864, 0.834059, 1.09944, -0.740294, 0.376521, 0.267098, -0.0230269, -0.0403144, -0.00517499, 0.00056494, 0.00486497, -0.00724504, -0.00535995, -0.00303499, 0.000784919, 0.00109002, 0.007765, 0.0062447, -0.00576526, 0.0525599, 0.0202753, -0.000715069, 0.00866055, 0.0199258, 0.014016, 0.01024, 0.00713998, -0.0273913, -0.0214815, 0.0140056, -0.00141412, 0.109929, -0.0203401, 0.00369474, -0.00342503, -0.169545, 0.0110401, 0, 0, 0, 0.937757, -0.237195, 0.247405, 0.968874, 0.243725, -0.0708693, 0, 0, 0, 0, 0, 0, 0.312, 0.312, 0, 0, 0, 0, 0, 0, 0, -0.669, -0.669, -0.2, -0.2, 0.363, 0.363, -0.2, 0.2, 0, 0, 0, 0, -0.6, -0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.50119, 3.5683, 1.17118, -0.567263, -0.594273, -2.54616, 0.954486, -1.00324, -1.23805, -2.09124, -0.114601, -0.930399, -0.316535, -1.01122, 6.25829, 2.18793, -1.66646, -0.413492, 1.2978, 1.22624, -1.38061, -0.229903, -1.23906, 1.81632, -1.64272, 2.23488, -0.703247, -1.96358, 0.4386
            };
            
            // 替换 obs["obs"] 为固定值
            obs["obs"] = fixed_obs_vec;
            episode_length = 1; // 重置 episode_length 为 1
        }
        
        obs["time_step"] = std::vector<float>{static_cast<float>(episode_length)};
        //spdlog::info("[TIME_STEP] episode_length: {}, time_step: {}", episode_length, episode_length);

        
        

        if ((DEBUG_OBS_OUTPUT)&&(episode_length % 50 == 0)) {
        //if (DEBUG_OBS_OUTPUT) {
            spdlog::info("=== OBS DEBUG (step {}) ===", episode_length.load());
            spdlog::info("obs size: {}", obs.size());
            for (const auto& [group_name, group_obs] : obs) {
                spdlog::info("Observation Group: '{}', Size: {}", group_name, group_obs.size());
            }
            for (const auto& [group_name, group_obs] : obs) {
                std::ostringstream oss;
                oss << "Observation Group: '" << group_name << "', Values: [";
                for (size_t i = 0; i < group_obs.size(); ++i) {
                    oss << group_obs[i];
                    if (i < group_obs.size() - 1) {
                        oss << ", ";
                    }
                }
                oss << "]";
                spdlog::info(oss.str());
            }
        }
        

        // ===== 执行策略推理 =====        
        auto infer_begin = std::chrono::high_resolution_clock::now();
        auto action = alg->act(obs);
        last_inference_s.store(std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - infer_begin).count());
                
        // ===== 调试输出：置零输出/打印策略输出 =====
        std::vector<float> locked_action(29, 0.0f);
        //locked_action[15] = 1.0f;
        //action = locked_action;
        if (DEBUG_FIXED_INPUT) {
            std::ostringstream action_oss;
            action_oss << "[FIXED INPUT] action output: [";
            for (size_t i = 0; i < action.size(); ++i) {
                action_oss << action[i];
                if (i < action.size() - 1) {
                    action_oss << ", ";
                }
            }
            action_oss << "]";
            spdlog::info(action_oss.str());
            
            // 可选：只打印第一个动作值
            // spdlog::info("[FIXED INPUT] action[0] = {}", action[0]);
        }
        
        action_manager->process_action(action);
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    std::atomic<long> episode_length{0};
    std::atomic<double> last_inference_s{0.0};   // 最近一次策略推理耗时（s），供健康日志使用
    float global_phase = 0.0f;
    std::vector<float> command{0.0f, 0.0f, 0.0f}; // 经 max_acceleration 限速后的速度指令

};
};