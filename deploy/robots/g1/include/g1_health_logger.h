// Copyright (c) 2026, G1 health logger for Unitree G1 29-DoF deployment.
//
// C++ port of g1_health/g1_health_logger.py (see g1_health/README_G1_HEALTH_DEBUG.md).
//
// 设计目标（与 Python 版保持一致）：
//   1) 记录 29 个电机反馈状态 + 真正发出去的 LowCmd；
//   2) 记录 BMS / 控制周期 / LowState age；
//   3) 滚动保存故障前 ring buffer，motorstate / BMS 状态变化时自动 dump；
//   4) 终端只打印精简摘要，避免日志本身影响控制循环。
//
// 用法：
//   // main.cpp（可选，读取 config.yaml 的 health_log: 节）
//   G1HealthLogger::initFromConfig(param::config["health_log"]);
//
//   // 各策略状态 run() 中（FSM 线程，Publish 之前）：
//   G1HealthLogger::instance().logCycle(
//       lowstate->msg_, lowcmd->msg_,
//       g1health::actionToHardware(env->robot->data.joint_ids_map,
//                                  env->action_manager->processed_actions(), 29),
//       g1health::nanv(), env->last_inference_s.load(), g1health::nanv());
//
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/BmsState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <yaml-cpp/yaml.h>

namespace g1health
{

// NaN
inline double nanv() { return std::numeric_limits<double>::quiet_NaN(); }

// 标准 G1 29-DoF 硬件顺序关节名（与 motor_state / motor_cmd 索引一致）。
// 若你的部署顺序不同，请换用你自己的 joint 顺序。
inline const std::vector<std::string>& g1JointNames29()
{
    static const std::vector<std::string> names = {
        "left_hip_pitch",    "left_hip_roll",    "left_hip_yaw",
        "left_knee",         "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch",   "right_hip_roll",   "right_hip_yaw",
        "right_knee",        "right_ankle_pitch","right_ankle_roll",
        "waist_yaw",         "waist_roll",       "waist_pitch",
        "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw",
        "left_elbow",        "left_wrist_roll",  "left_wrist_pitch","left_wrist_yaw",
        "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw",
        "right_elbow",       "right_wrist_roll", "right_wrist_pitch","right_wrist_yaw",
    };
    return names;
}

// 策略输出（模型顺序）-> 29 个硬件电机顺序；未被策略控制的关节填 NaN
inline std::vector<float> actionToHardware(const std::vector<float>& joint_ids_map,
                                           const std::vector<float>& processed_actions,
                                           size_t num_joints = 29)
{
    std::vector<float> out(num_joints, std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < joint_ids_map.size() && i < processed_actions.size(); ++i)
    {
        const int hw = static_cast<int>(joint_ids_map[i]);
        if (hw >= 0 && static_cast<size_t>(hw) < num_joints)
            out[static_cast<size_t>(hw)] = processed_actions[i];
    }
    return out;
}

// 控制循环 / LowState age 跟踪（tick 变化 -> 最近一次收到 LowState 的时刻）
class HealthTiming
{
public:
    // tick = low_state.tick()；返回 {loop_dt_s, lowstate_age_s}
    std::pair<double, double> update(uint32_t tick)
    {
        const auto now = std::chrono::steady_clock::now();
        double loop_dt_s = 0.0;
        if (initialized_)
            loop_dt_s = std::chrono::duration<double>(now - last_call_).count();
        last_call_ = now;
        initialized_ = true;

        if (tick != last_tick_) { last_tick_ = tick; last_tick_time_ = now; }
        const double age_s =
            std::chrono::duration<double>(now - last_tick_time_).count();
        return {loop_dt_s, age_s};
    }

private:
    bool initialized_ = false;
    std::chrono::steady_clock::time_point last_call_{};
    uint32_t last_tick_ = 0;
    std::chrono::steady_clock::time_point last_tick_time_{};
};

// 配置（与 Python logger 构造参数对应）
struct HealthConfig
{
    bool enabled = true;
    std::string out_dir = "logs/g1_health";
    double control_hz = 1000.0;  // FSM 1kHz 控制循环
    double log_hz = 50.0;        // CSV 采样率
    double print_hz = 1.0;       // 终端摘要频率
    double ring_seconds = 30.0;  // 故障前预存缓冲（秒）
    bool enable_bms = true;      // 订阅 rt/bms/state
};

class G1HealthLogger
{
public:
    G1HealthLogger(const std::vector<std::string>& joint_names = g1JointNames29(),
                   const HealthConfig& cfg = HealthConfig{});
    ~G1HealthLogger();

    G1HealthLogger(const G1HealthLogger&) = delete;
    G1HealthLogger& operator=(const G1HealthLogger&) = delete;

    // 每个控制周期调用一次（策略状态 run()，Publish 之前）。
    // loop_dt_s / inference_s / lowstate_age_s 传 nan 时由 logger 自行测量。
    void logCycle(const unitree_hg::msg::dds_::LowState_& low_state,
                  const unitree_hg::msg::dds_::LowCmd_& low_cmd,
                  const std::vector<float>& action_by_joint,
                  double loop_dt_s = nanv(),
                  double inference_s = nanv(),
                  double lowstate_age_s = nanv());

    void markEvent(const std::string& text, bool dump_ring = false);
    void close();
    bool enabled() const { return enabled_; }

    // ---- 全局单例（一个进程一个健康日志，随状态切换持续记录） ----
    static G1HealthLogger& instance();
    static void configure(const HealthConfig& cfg);
    static void initFromConfig(const YAML::Node& node);  // 解析 config.yaml 的 health_log: 节

private:
    std::vector<std::string> buildFieldNames() const;
    std::vector<double> buildRow(const unitree_hg::msg::dds_::LowState_& low_state,
                                 const unitree_hg::msg::dds_::LowCmd_& low_cmd,
                                 const std::vector<float>& action,
                                 double elapsed, double loop_dt_ms,
                                 double inference_ms, double lowstate_age_ms);
    void writeRow(const std::vector<double>& row);
    void updateWindows(const std::vector<double>& row);
    std::string detectStateChanges(const std::vector<double>& row, double elapsed);
    void printSummary(const std::vector<double>& row);
    std::string dumpRing(const std::string& tag);

    size_t n_ = 0;
    HealthConfig cfg_;
    bool enabled_ = false;
    bool closed_ = false;

    std::vector<std::string> joint_names_;
    std::vector<std::string> fieldnames_;

    std::string run_id_;
    std::string csv_path_;
    std::string events_path_;
    std::ofstream csv_fs_;

    double t0_ = 0.0;
    double last_log_t_ = -1e9, last_print_t_ = -1e9;
    double last_flush_t_ = -1e9, last_snapshot_t_ = -1e9;
    long row_count_ = 0;

    size_t ring_cap_ = 0;
    std::deque<std::vector<double>> ring_;

    std::vector<std::optional<int64_t>> motorstate_baseline_;
    std::optional<int64_t> bmsstate_baseline_;

    std::vector<std::deque<double>> tau_hist_;
    std::vector<std::deque<double>> qerr_hist_;
    size_t hist_cap_ = 0;

    HealthTiming timing_;

    std::shared_ptr<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::BmsState_>> bms_sub_;
    unitree_hg::msg::dds_::BmsState_ latest_bms_;
    std::mutex bms_mutex_;
    bool have_bms_ = false;

    // 字段布局（与 analyze_g1_health_log.py 一致）
    enum CommonIdx {
        C_WALL = 0, C_ELAPSED, C_LOOP, C_INFER, C_AGE,
        C_MODE_MACHINE, C_MODE_PR,
        C_BMS_V, C_BMS_I, C_BMS_SOC, C_BMS_SOH, C_BMS_T0, C_BMS_T1, C_BMS_STATE,
        C_NUM
    };
    enum SuffixIdx {
        S_Q = 0, S_DQ, S_TAU_EST, S_TEMP0, S_TEMP1, S_MOTORSTATE, S_MODE,
        S_Q_DES, S_DQ_DES, S_KP, S_KD, S_TAU_FF, S_Q_ERR, S_TAU_PD, S_ACTION,
        S_NUM
    };
};

} // namespace g1health
