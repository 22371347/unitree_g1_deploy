#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "g1_health_logger.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    const float vx_min = cfg["lin_vel_x"][0].as<float>();
    const float vx_max = cfg["lin_vel_x"][1].as<float>();
    const float vy_min = cfg["lin_vel_y"][0].as<float>();
    const float vy_max = cfg["lin_vel_y"][1].as<float>();
    const float wz_min = cfg["ang_vel_z"][0].as<float>();
    const float wz_max = cfg["ang_vel_z"][1].as<float>();

    std::vector<float> target_command = {0.0f, 0.0f, 0.0f};
    if (key == "w") target_command = {vx_max, 0.0f, 0.0f};
    if (key == "s") target_command = {vx_min, 0.0f, 0.0f};
    if (key == "a") target_command = {0.0f, vy_max, 0.0f};
    if (key == "d") target_command = {0.0f, vy_min, 0.0f};
    if (key == "q") target_command = {0.0f, 0.0f, wz_max};
    if (key == "e") target_command = {0.0f, 0.0f, wz_min};

    // max_acceleration 由 export_deploy_cfg.py 从训练配置写入 deploy.yaml。
    // 没有该字段的旧策略保持原始键盘指令，不改变旧部署行为。
    const auto max_acceleration_node =
        env->cfg["commands"]["base_velocity"]["max_acceleration"];
    if (!max_acceleration_node)
    {
        return target_command;
    }
    const auto max_acceleration = max_acceleration_node.as<std::vector<float>>();
    if (max_acceleration.size() != target_command.size())
    {
        throw std::runtime_error(
            "base_velocity.max_acceleration must contain x, y, and yaw values."
        );
    }

    // env->step_dt 是策略控制周期；每个周期最多改变 a_max * dt。
    // 松开按键时 target_command 为 0，因此停止同样会经过减速斜坡。
    static std::vector<float> filtered_command = {0.0f, 0.0f, 0.0f};
    if (env->episode_length <= 1)
    {
        filtered_command = {0.0f, 0.0f, 0.0f};
    }

    for (std::size_t i = 0; i < filtered_command.size(); ++i)
    {
        const float max_step = max_acceleration[i] * env->step_dt;
        if (max_step <= 0.0f)
        {
            throw std::runtime_error(
                "base_velocity.max_acceleration values must be positive."
            );
        }
        const float command_error = target_command[i] - filtered_command[i];
        filtered_command[i] += std::clamp(command_error, -max_step, max_step);
    }

    return filtered_command;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // ---- 姿态保护（可配置，默认开启）：倾角超限 -> Passive ----
    bool enable_bad_orientation_check = true;
    if (cfg["enable_bad_orientation_check"])
    {
        enable_bad_orientation_check =
            cfg["enable_bad_orientation_check"].as<bool>();
    }
    float bad_orientation_limit = 1.0f;
    if (cfg["bad_orientation_limit"])
    {
        bad_orientation_limit =
            cfg["bad_orientation_limit"].as<float>();
    }

    if (enable_bad_orientation_check)
    {
        this->registered_checks.emplace_back(
            std::make_pair(
                [this, bad_orientation_limit]() -> bool
                {
                    return isaaclab::mdp::bad_orientation(
                        env.get(), bad_orientation_limit
                    );
                },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    // ---- 摔倒检测：倾角过大（持续 confirm_ms）-> 切到目标状态（如 Recovery）----
    // 注：仅用 IMU 倾角判据（projected_gravity），不依赖高度（实机无 SportModeState 高度源）
    if (cfg["fall_detection"])
    {
        const auto fall_cfg = cfg["fall_detection"];
        const std::string target_name =
            fall_cfg["target"] ?
                fall_cfg["target"].as<std::string>() :
                "Recovery";
        const float max_tilt =
            fall_cfg["max_tilt"] ?
                fall_cfg["max_tilt"].as<float>() :
                0.872665f;   // 50 度
        const int confirm_ms =
            fall_cfg["confirm_ms"] ?
                fall_cfg["confirm_ms"].as<int>() :
                50;

        if (!FSMStringMap.right.count(target_name))
        {
            throw std::runtime_error(
                "Unknown fall detection target state: " + target_name
            );
        }

        const int target_mode = FSMStringMap.right.at(target_name);
        this->registered_checks.emplace_back(
            [this, max_tilt, confirm_ms]() -> bool
            {
                const bool bad_tilt =
                    isaaclab::mdp::bad_orientation(env.get(), max_tilt);
                return condition_confirmed(
                    bad_tilt,
                    fall_condition_since_,
                    confirm_ms
                );
            },
            target_mode
        );
    }

    // ---- 恢复完成门控：直立 + 角速度小（持续 confirm_ms）-> 切到目标状态（如 Fight）----
    // 注：仅用 IMU 倾角 + 角速度判据，不依赖高度（实机无高度源）
    if (cfg["recovery_gate"])
    {
        const auto recovery_cfg = cfg["recovery_gate"];
        const std::string target_name =
            recovery_cfg["target"] ?
                recovery_cfg["target"].as<std::string>() :
                "Fight";
        const float max_tilt =
            recovery_cfg["max_tilt"] ?
                recovery_cfg["max_tilt"].as<float>() :
                0.30f;
        const float max_root_ang_vel =
            recovery_cfg["max_root_ang_vel"] ?
                recovery_cfg["max_root_ang_vel"].as<float>() :
                1.0f;
        const int confirm_ms =
            recovery_cfg["confirm_ms"] ?
                recovery_cfg["confirm_ms"].as<int>() :
                500;

        if (!FSMStringMap.right.count(target_name))
        {
            throw std::runtime_error(
                "Unknown recovery gate target state: " + target_name
            );
        }

        const int target_mode = FSMStringMap.right.at(target_name);
        this->registered_checks.emplace_back(
            [this, max_tilt, max_root_ang_vel, confirm_ms]() -> bool
            {
                const bool upright =
                    !isaaclab::mdp::bad_orientation(env.get(), max_tilt);
                const bool angularly_stable =
                    env->robot->data.root_ang_vel_b.norm() <
                    max_root_ang_vel;
                return condition_confirmed(
                    upright && angularly_stable,
                    recovery_ready_since_,
                    confirm_ms
                );
            },
            target_mode
        );
    }
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    // ---- G1 health logger：记录真正发出去的 LowCmd（Publish 之前） ----
    g1health::G1HealthLogger::instance().logCycle(
        lowstate->msg_, lowcmd->msg_,
        g1health::actionToHardware(env->robot->data.joint_ids_map, action, 29),
        g1health::nanv(), env->last_inference_s.load(), g1health::nanv());
}