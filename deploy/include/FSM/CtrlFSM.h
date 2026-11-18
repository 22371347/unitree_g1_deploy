// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/common/thread/recurrent_thread.hpp>
#include "BaseState.h"
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <map>
#include <string>

#include "FSMState.h"

class CtrlFSM
{
public:
    CtrlFSM(std::shared_ptr<BaseState> initstate)
    {
        // Initialize FSM states
        states.push_back(std::move(initstate));

    }

    CtrlFSM(YAML::Node cfg)
    {
        auto fsms = cfg["_"]; // enabled FSMs

        // register FSM string map; used for state transition
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            FSMStringMap.insert({id, fsm_name});
        }

        // Initialize FSM states
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
            auto fsm_class = getFsmMap().find("State_" + fsm_type);
            if (fsm_class == getFsmMap().end()) {
                throw std::runtime_error("FSM: Unknown FSM type " + fsm_type);
            }
            auto state_instance = fsm_class->second(id, fsm_name);
            add(state_instance);
        }

        // 键盘字母 -> 状态 id（配置文件顶层 keyboard: {键: 状态名} 可覆盖内置默认）
        init_keyboard_map();

        spdlog::info("Keyboard->joystick (sim): ↑/↓/←/→=move(ly/lx), q/e=turn(rx), z=stop(toggle)");
    }

    // 内置默认键盘映射：a=Passive(1) b=FixStand(2) c=Velocity(3)
    //                      d=5 e=6 f=7 g=9 h=10 i=11（自定义策略状态按各自 id 对号入座）
    // 若配置文件顶层写了 keyboard:，则完全以配置为准（按状态名解析 id，未注册的状态会告警并忽略）
    void init_keyboard_map()
    {
        key_to_state_ = {
            {"a", 1}, {"b", 2}, {"c", 3}, {"d", 5}, {"e", 6},
            {"f", 7}, {"g", 9}, {"h", 10}, {"i", 11},
        };

        // 兜底：默认 d 指向 5；若当前配置没有 id 5，而自定义策略占了预留的 id 4，则让 d 指向 4
        if (!FSMStringMap.left.count(5) && FSMStringMap.left.count(4)) {
            key_to_state_["d"] = 4;
        }

        auto kb = param::config["keyboard"];
        if (kb && kb.IsMap() && kb.size() > 0) {
            key_to_state_.clear();
            for (auto it = kb.begin(); it != kb.end(); ++it) {
                const std::string k    = it->first.as<std::string>();
                const std::string name = it->second.as<std::string>();
                if (!FSMStringMap.right.count(name)) {
                    spdlog::warn("FSM: keyboard '{}' -> unknown state '{}', ignored", k, name);
                    continue;
                }
                key_to_state_[k] = FSMStringMap.right.at(name);
            }
        }

        std::string desc;
        for (const auto &kv : key_to_state_) {
            const int id = kv.second;
            desc += kv.first + "=" +
                    (FSMStringMap.left.count(id) ? FSMStringMap.left.at(id) : std::string("?")) +
                    "(" + std::to_string(id) + ") ";
        }
        spdlog::info("FSM: keyboard map: {}", desc);
    }

    void start() 
    {
        // Start From State_Passive
        currentState = states[0];
        currentState->enter();

        fsm_thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "FSM", 0, this->dt * 1e6, &CtrlFSM::run_, this);
        spdlog::info("FSM: Start {}", currentState->getStateString());
    }

    void add(std::shared_ptr<BaseState> state)
    {
        for(auto & s : states)
        {
            if(s->isState(state->getState()))
            {
                spdlog::error("FSM: State_{} already exists", state->getStateString());
                std::exit(0);
            }
        }

        states.push_back(std::move(state));
    }
    
    ~CtrlFSM()
    {
        states.clear();
    }

    std::vector<std::shared_ptr<BaseState>> states;
private:
    const double dt = 0.001;

    // 键盘模拟手柄摇杆（仿真用）：↑/↓=ly, ←/→=lx, q/e=rx, z=急停（toggle 式）
    float kbd_ly_ = 0.0f, kbd_lx_ = 0.0f, kbd_rx_ = 0.0f;
    bool  kbd_active_ = false;   // 上一帧是否在注入虚拟摇杆（用于回零时写一次后停止注入）

    // 键盘字母 -> 状态 id
    std::map<std::string, int> key_to_state_;

    void run_()
    {
        currentState->pre_run();
        currentState->run();
        currentState->post_run();

        // 键盘模拟手柄摇杆：必须在 registered_checks 之前注入（DDS 每帧会清零 joystick）
        apply_keyboard_joystick();

        // Check if need to change state
        int nextStateMode = 0;
        //手柄输入修改为键盘检测
        for(int i(0); i<currentState->registered_checks.size(); i++)
        {
            if(currentState->registered_checks[i].first())
            {
                nextStateMode = currentState->registered_checks[i].second;
                break;
            }
        }
        
        
        //增加键盘检测（映射见 init_keyboard_map，可由配置文件的 keyboard: 段覆盖）
        std::string key = FSMState::keyboard->key();
        if (!key.empty()) {
            auto it = key_to_state_.find(key);
            if (it != key_to_state_.end()) nextStateMode = it->second;
        }
        

        if(nextStateMode != 0 && !currentState->isState(nextStateMode))
        {
            for(auto & state : states)
            {
                if(state->isState(nextStateMode))
                {
                    spdlog::info("FSM: Change state from {} to {}", currentState->getStateString(), state->getStateString());
                    currentState->exit();
                    currentState = state;
                    currentState->enter();
                    break;
                }
            }
        }
    }

    void apply_keyboard_joystick()
    {
        // 状态主动请求清空虚拟摇杆（如进入 PHP 时希望默认 0 指令 = 站立）
        if (FSMState::reset_joystick_request.exchange(false)) {
            kbd_ly_ = kbd_lx_ = kbd_rx_ = 0.0f;
        }

        // 键盘模拟手柄摇杆（toggle 式）：按一次置位，再按一次归零
        if (FSMState::keyboard->on_pressed) {
            const std::string key = FSMState::keyboard->key();
            if      (key == "up")    kbd_ly_ = (kbd_ly_ > 0.0f) ? 0.0f : 5.0f;   // 前进
            else if (key == "down")  kbd_ly_ = (kbd_ly_ < 0.0f) ? 0.0f : -5.0f;  // 后退
            else if (key == "right")  kbd_lx_ = (kbd_lx_ > 0.0f) ? 0.0f : 5.0f;   // 右移
            else if (key == "left") kbd_lx_ = (kbd_lx_ < 0.0f) ? 0.0f : -5.0f;  // 左移
            else if (key == "e")     kbd_rx_ = (kbd_rx_ > 0.0f) ? 0.0f : 5.0f;   // 右转
            else if (key == "q")     kbd_rx_ = (kbd_rx_ < 0.0f) ? 0.0f : -5.0f;  // 左转
            else if (key == "z")     { kbd_ly_ = kbd_lx_ = kbd_rx_ = 0.0f; }     // 急停
        }
        // 写回 joystick 摇杆（velocity_commands 观测读取；DDS 每帧会清零，故在仿真里需持续注入）。
        // 门控：仅当键盘确实有非零摇杆指令、或刚从非零回零时写一次——
        // 实机不在推虚拟摇杆时就不去覆盖真实手柄的摇杆值。
        const bool active = (kbd_ly_ != 0.0f || kbd_lx_ != 0.0f || kbd_rx_ != 0.0f);
        if (active || kbd_active_) {
            auto & joy = FSMState::lowstate->joystick;
            joy.ly(kbd_ly_);
            joy.lx(kbd_lx_);
            joy.rx(kbd_rx_);
            kbd_active_ = active;
        }
    }

    std::shared_ptr<BaseState> currentState;
    unitree::common::RecurrentThreadPtr fsm_thread_;
};