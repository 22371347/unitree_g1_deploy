// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/common/thread/recurrent_thread.hpp>
#include "BaseState.h"
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

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

        spdlog::info("Keyboard->joystick (sim): ↑/↓/←/→=move(ly/lx), q/e=turn(rx), z=stop(toggle); a/b/c/d/e/f/g=FSM switch");
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
        
        
        //增加键盘检测
        std::string key = FSMState::keyboard->key();
        if (!key.empty()) {
            //spdlog::info("FSM: Keyboard input received: {}", key);
            // 使用键盘字母键 a, b, c, d, e, f... 切换状态
            if (key == "a") nextStateMode = 1;   // Passive
            else if (key == "b") nextStateMode = 2; // FixStand
            else if (key == "c") nextStateMode = 3; // Velocity
            else if (key == "d") nextStateMode = 5; // 自定义动作
            else if (key == "e") nextStateMode = 6; // Fight1
            else if (key == "f") nextStateMode = 7; // Fight2
            else if (key == "g") nextStateMode = 9; // FightRight
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
        // 键盘模拟手柄摇杆（toggle 式）：按一次置位，再按一次归零
        if (FSMState::keyboard->on_pressed) {
            const std::string key = FSMState::keyboard->key();
            if      (key == "up")    kbd_ly_ = (kbd_ly_ > 0.0f) ? 0.0f : 1.0f;   // 前进
            else if (key == "down")  kbd_ly_ = (kbd_ly_ < 0.0f) ? 0.0f : -1.0f;  // 后退
            else if (key == "right")  kbd_lx_ = (kbd_lx_ > 0.0f) ? 0.0f : 1.0f;   // 右移
            else if (key == "left") kbd_lx_ = (kbd_lx_ < 0.0f) ? 0.0f : -1.0f;  // 左移
            else if (key == "e")     kbd_rx_ = (kbd_rx_ > 0.0f) ? 0.0f : 1.0f;   // 右转
            else if (key == "q")     kbd_rx_ = (kbd_rx_ < 0.0f) ? 0.0f : -1.0f;  // 左转
            else if (key == "z")     { kbd_ly_ = kbd_lx_ = kbd_rx_ = 0.0f; }     // 急停
        }
        // 写入 joystick 摇杆（velocity_commands 观测读取；DDS 每帧清零，故在此重注入）
        auto & joy = FSMState::lowstate->joystick;
        joy.ly(kbd_ly_);
        joy.lx(kbd_lx_);
        joy.rx(kbd_rx_);
    }

    std::shared_ptr<BaseState> currentState;
    unitree::common::RecurrentThreadPtr fsm_thread_;
};