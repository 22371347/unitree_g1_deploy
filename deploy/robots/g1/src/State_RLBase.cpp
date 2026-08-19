#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "g1_health_logger.h"
#include <stdexcept>
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
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

    // ---- 摔倒检测：倾角过大 / 高度过低（持续 confirm_ms）-> 切到目标状态（如 Recovery）----
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
                0.523592f;   // 30 度
        const float min_height =
            fall_cfg["min_height"] ?
                fall_cfg["min_height"].as<float>() :
                0.35f;
        const bool height_enabled =
            fall_cfg["height_enabled"] ?
                fall_cfg["height_enabled"].as<bool>() :
                true;
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

        if (height_enabled && !sport_mode_state)
        {
            sport_mode_state = std::make_shared<
                unitree::robot::go2::subscription::SportModeState
            >();
        }

        const int target_mode = FSMStringMap.right.at(target_name);
        this->registered_checks.emplace_back(
            [this, max_tilt, min_height, height_enabled, confirm_ms]() -> bool
            {
                const bool bad_tilt =
                    isaaclab::mdp::bad_orientation(env.get(), max_tilt);

                bool low_height = false;
                if (height_enabled)
                {
                    float base_height = 0.0f;
                    low_height =
                        read_base_height(base_height) &&
                        base_height < min_height;
                }

                return condition_confirmed(
                    bad_tilt || low_height,
                    fall_condition_since_,
                    confirm_ms
                );
            },
            target_mode
        );
    }

    // ---- 恢复完成门控：直立 + 高度达标 + 角速度小（持续 confirm_ms）-> 切到目标状态（如 Fight）----
    if (cfg["recovery_gate"])
    {
        const auto recovery_cfg = cfg["recovery_gate"];
        const std::string target_name =
            recovery_cfg["target"] ?
                recovery_cfg["target"].as<std::string>() :
                "Fight";
        const float min_height =
            recovery_cfg["min_height"] ?
                recovery_cfg["min_height"].as<float>() :
                0.55f;
        const float max_tilt =
            recovery_cfg["max_tilt"] ?
                recovery_cfg["max_tilt"].as<float>() :
                0.30f;
        const float max_root_ang_vel =
            recovery_cfg["max_root_ang_vel"] ?
                recovery_cfg["max_root_ang_vel"].as<float>() :
                1.0f;
        const bool height_enabled =
            recovery_cfg["height_enabled"] ?
                recovery_cfg["height_enabled"].as<bool>() :
                true;
        const bool require_height_source =
            recovery_cfg["require_height_source"] ?
                recovery_cfg["require_height_source"].as<bool>() :
                false;
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

        if (height_enabled && !sport_mode_state)
        {
            sport_mode_state = std::make_shared<
                unitree::robot::go2::subscription::SportModeState
            >();
        }

        const int target_mode = FSMStringMap.right.at(target_name);
        this->registered_checks.emplace_back(
            [this, min_height, max_tilt, max_root_ang_vel,
             height_enabled, require_height_source, confirm_ms]() -> bool
            {
                const bool upright =
                    !isaaclab::mdp::bad_orientation(env.get(), max_tilt);
                const bool angularly_stable =
                    env->robot->data.root_ang_vel_b.norm() <
                    max_root_ang_vel;

                bool height_ready = true;
                if (height_enabled)
                {
                    float base_height = 0.0f;
                    const bool height_valid =
                        read_base_height(base_height);
                    height_ready =
                        height_valid ?
                            base_height >= min_height :
                            !require_height_source;
                }

                return condition_confirmed(
                    upright && angularly_stable && height_ready,
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