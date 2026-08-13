// Copyright (c) 2026, G1 health logger implementation.
// C++ port of g1_health/g1_health_logger.py

#include "g1_health_logger.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace g1health
{

namespace
{
const char* kCommonFieldNames[] = {
    "wall_time", "elapsed_s", "loop_dt_ms", "inference_ms", "lowstate_age_ms",
    "mode_machine", "mode_pr",
    "bms_voltage", "bms_current", "bms_soc", "bms_soh", "bms_temp0", "bms_temp1", "bms_state",
};
const char* kSuffixFieldNames[] = {
    "q", "dq", "tau_est", "temp0", "temp1", "motorstate", "mode",
    "q_des", "dq_des", "kp", "kd", "tau_ff", "q_err", "tau_pd_est", "action",
};

double nowS()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string timestampStr()
{
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
    return buf;
}

double wallTimeS()
{
    return static_cast<double>(std::time(nullptr));
}

bool finite(double v) { return std::isfinite(v); }

void writeNum(std::ostream& os, double v)
{
    if (std::isnan(v)) { os << "nan"; return; }
    if (std::isinf(v)) { os << "inf"; return; }
    if (v == std::floor(v) && std::fabs(v) < 1e15) {
        os << static_cast<long long>(v);
    } else {
        os << std::setprecision(9) << v;
    }
}
} // namespace

// ---------------------------------------------------------------------------
// 全局单例
// ---------------------------------------------------------------------------
namespace
{
HealthConfig g_config;
std::once_flag g_singleton_flag;
std::unique_ptr<G1HealthLogger> g_singleton;
} // namespace

G1HealthLogger& G1HealthLogger::instance()
{
    std::call_once(g_singleton_flag, [] {
        g_singleton = std::make_unique<G1HealthLogger>(g1JointNames29(), g_config);
    });
    return *g_singleton;
}

void G1HealthLogger::configure(const HealthConfig& cfg) { g_config = cfg; }

void G1HealthLogger::initFromConfig(const YAML::Node& node)
{
    HealthConfig cfg;
    if (node && !node.IsNull())
    {
        if (node["enabled"])      cfg.enabled      = node["enabled"].as<bool>();
        if (node["out_dir"])      cfg.out_dir      = node["out_dir"].as<std::string>();
        if (node["control_hz"])   cfg.control_hz   = node["control_hz"].as<double>();
        if (node["log_hz"])       cfg.log_hz       = node["log_hz"].as<double>();
        if (node["print_hz"])     cfg.print_hz     = node["print_hz"].as<double>();
        if (node["ring_seconds"]) cfg.ring_seconds = node["ring_seconds"].as<double>();
        if (node["enable_bms"])   cfg.enable_bms   = node["enable_bms"].as<bool>();
    }
    g_config = cfg;
}

// ---------------------------------------------------------------------------
// 构造 / 析构
// ---------------------------------------------------------------------------
G1HealthLogger::G1HealthLogger(const std::vector<std::string>& joint_names,
                               const HealthConfig& cfg)
    : n_(joint_names.size()), cfg_(cfg), joint_names_(joint_names)
{
    if (!cfg_.enabled || n_ == 0) return;

    try { std::filesystem::create_directories(cfg_.out_dir); }
    catch (const std::exception& e) {
        std::cerr << "[G1-HEALTH] failed to create out_dir " << cfg_.out_dir
                  << ": " << e.what() << "\n";
        return;
    }

    run_id_ = timestampStr();
    csv_path_ = (std::filesystem::path(cfg_.out_dir) / (run_id_ + "_full.csv")).string();
    events_path_ = (std::filesystem::path(cfg_.out_dir) / (run_id_ + "_events.txt")).string();

    fieldnames_ = buildFieldNames();

    csv_fs_.open(csv_path_, std::ios::out | std::ios::trunc);
    if (!csv_fs_.is_open()) {
        std::cerr << "[G1-HEALTH] cannot open " << csv_path_ << "\n";
        return;
    }
    for (size_t i = 0; i < fieldnames_.size(); ++i) {
        if (i) csv_fs_ << ",";
        csv_fs_ << fieldnames_[i];
    }
    csv_fs_ << "\n";
    csv_fs_.flush();

    t0_ = nowS();
    ring_cap_ = static_cast<size_t>(std::max(1.0, cfg_.log_hz * cfg_.ring_seconds));
    hist_cap_ = static_cast<size_t>(std::max(1.0, cfg_.log_hz * 10.0));
    motorstate_baseline_.assign(n_, std::optional<int64_t>{});
    tau_hist_.resize(n_);
    qerr_hist_.resize(n_);

    enabled_ = true;
    markEvent("LOGGER_START");

    // BMS 单独订阅（G1 的 hg LowState 不含 BMS）
    if (cfg_.enable_bms)
    {
        try {
            bms_sub_ = std::make_shared<
                unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::BmsState_>>("rt/bms/state");
            bms_sub_->InitChannel([this](const void* msg) {
                std::lock_guard<std::mutex> lk(bms_mutex_);
                latest_bms_ = *(const unitree_hg::msg::dds_::BmsState_*)msg;
                have_bms_ = true;
            });
        } catch (const std::exception& e) {
            std::cerr << "[G1-HEALTH] BMS subscription failed: " << e.what() << "\n";
            bms_sub_.reset();
        }
    }
}

G1HealthLogger::~G1HealthLogger() { close(); }

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------
std::vector<std::string> G1HealthLogger::buildFieldNames() const
{
    std::vector<std::string> f;
    for (auto* n : kCommonFieldNames) f.emplace_back(n);
    for (const auto& name : joint_names_)
        for (auto* suf : kSuffixFieldNames) f.push_back(name + "." + suf);
    return f;
}

std::vector<double> G1HealthLogger::buildRow(const unitree_hg::msg::dds_::LowState_& ls,
                                             const unitree_hg::msg::dds_::LowCmd_& lc,
                                             const std::vector<float>& action,
                                             double elapsed, double loop_dt_ms,
                                             double inference_ms, double lowstate_age_ms)
{
    std::vector<double> r(C_NUM + S_NUM * n_, nanv());
    r[C_WALL] = wallTimeS();
    r[C_ELAPSED] = elapsed;
    r[C_LOOP] = loop_dt_ms;
    r[C_INFER] = inference_ms;
    r[C_AGE] = lowstate_age_ms;
    r[C_MODE_MACHINE] = ls.mode_machine();
    r[C_MODE_PR] = ls.mode_pr();

    {
        std::lock_guard<std::mutex> lk(bms_mutex_);
        if (have_bms_)
        {
            r[C_BMS_V] = latest_bms_.bmsvoltage()[0];
            r[C_BMS_I] = latest_bms_.current();
            r[C_BMS_SOC] = latest_bms_.soc();
            r[C_BMS_SOH] = latest_bms_.soh();
            const auto& t = latest_bms_.temperature();
            r[C_BMS_T0] = t[0];
            r[C_BMS_T1] = t[1];
            const auto& st = latest_bms_.bmsstate();
            r[C_BMS_STATE] = st[0];
        }
    }

    const auto& motors = ls.motor_state();
    const auto& cmds = lc.motor_cmd();
    for (size_t i = 0; i < n_; ++i)
    {
        const size_t base = C_NUM + i * S_NUM;
        const auto& ms = motors[i];
        const auto& mc = cmds[i];
        const double q = ms.q(), dq = ms.dq(), tau = ms.tau_est();
        const double t0 = ms.temperature()[0], t1 = ms.temperature()[1];
        const double q_des = mc.q(), dq_des = mc.dq();
        const double kp = mc.kp(), kd = mc.kd(), tau_ff = mc.tau();

        r[base + S_Q] = q;
        r[base + S_DQ] = dq;
        r[base + S_TAU_EST] = tau;
        r[base + S_TEMP0] = t0;
        r[base + S_TEMP1] = t1;
        r[base + S_MOTORSTATE] = ms.motorstate();
        r[base + S_MODE] = ms.mode();
        r[base + S_Q_DES] = q_des;
        r[base + S_DQ_DES] = dq_des;
        r[base + S_KP] = kp;
        r[base + S_KD] = kd;
        r[base + S_TAU_FF] = tau_ff;
        r[base + S_Q_ERR] = (finite(q_des) && finite(q)) ? (q_des - q) : nanv();
        r[base + S_TAU_PD] = (finite(kp) && finite(q_des) && finite(q) &&
                              finite(kd) && finite(dq_des) && finite(dq) && finite(tau_ff))
                                 ? (kp * (q_des - q) + kd * (dq_des - dq) + tau_ff)
                                 : nanv();
        r[base + S_ACTION] = (i < action.size()) ? action[i] : nanv();
    }
    return r;
}

void G1HealthLogger::writeRow(const std::vector<double>& row)
{
    std::ostringstream os;
    os << std::setprecision(9);
    for (size_t i = 0; i < row.size(); ++i) {
        if (i) os << ",";
        writeNum(os, row[i]);
    }
    os << "\n";
    csv_fs_ << os.str();
    ++row_count_;
}

void G1HealthLogger::updateWindows(const std::vector<double>& row)
{
    for (size_t i = 0; i < n_; ++i)
    {
        const size_t base = C_NUM + i * S_NUM;
        const double tau = row[base + S_TAU_EST];
        const double qerr = row[base + S_Q_ERR];
        if (finite(tau))
        {
            tau_hist_[i].push_back(tau);
            if (tau_hist_[i].size() > hist_cap_) tau_hist_[i].pop_front();
        }
        if (finite(qerr))
        {
            qerr_hist_[i].push_back(qerr);
            if (qerr_hist_[i].size() > hist_cap_) qerr_hist_[i].pop_front();
        }
    }
}

std::string G1HealthLogger::detectStateChanges(const std::vector<double>& row, double elapsed)
{
    // 始终学习基线（首 2s 内的样本也用于学习，但不触发事件）
    for (size_t i = 0; i < n_; ++i)
    {
        const double v = row[C_NUM + i * S_NUM + S_MOTORSTATE];
        if (!motorstate_baseline_[i].has_value() && finite(v))
            motorstate_baseline_[i] = static_cast<int64_t>(v);
    }
    const double b = row[C_BMS_STATE];
    if (!bmsstate_baseline_.has_value() && finite(b))
        bmsstate_baseline_ = static_cast<int64_t>(b);

    if (elapsed < 2.0) return "";  // 启动瞬态窗口内不判定变化

    std::string out;
    for (size_t i = 0; i < n_; ++i)
    {
        const double cur = row[C_NUM + i * S_NUM + S_MOTORSTATE];
        if (motorstate_baseline_[i].has_value() && finite(cur) &&
            static_cast<int64_t>(cur) != *motorstate_baseline_[i])
        {
            out += "MOTORSTATE_CHANGE " + joint_names_[i] + ": baseline=" +
                   std::to_string(*motorstate_baseline_[i]) + " current=" +
                   std::to_string(static_cast<int64_t>(cur)) + " | ";
        }
    }
    const double cur_bms = row[C_BMS_STATE];
    if (bmsstate_baseline_.has_value() && finite(cur_bms) &&
        static_cast<int64_t>(cur_bms) != *bmsstate_baseline_)
    {
        out += "BMSSTATE_CHANGE baseline=" + std::to_string(*bmsstate_baseline_) +
               " current=" + std::to_string(static_cast<int64_t>(cur_bms)) + " | ";
    }
    if (!out.empty()) out.resize(out.size() - 3);
    return out;
}

void G1HealthLogger::printSummary(const std::vector<double>& row)
{
    double hot = nanv(), taup = nanv(), qep = nanv(), tau_rms = nanv();
    std::string hot_j = "n/a", taup_j = "n/a", qep_j = "n/a", tau_rms_j = "n/a";

    for (size_t i = 0; i < n_; ++i)
    {
        const size_t base = C_NUM + i * S_NUM;
        const double tm = std::max(row[base + S_TEMP0], row[base + S_TEMP1]);
        if (finite(tm) && (!finite(hot) || tm > hot)) { hot = tm; hot_j = joint_names_[i]; }

        const double tau = row[base + S_TAU_EST];
        const double at = std::fabs(tau);
        if (finite(at) && (!finite(taup) || at > taup)) { taup = at; taup_j = joint_names_[i]; }

        const double aq = std::fabs(row[base + S_Q_ERR]);
        if (finite(aq) && (!finite(qep) || aq > qep)) { qep = aq; qep_j = joint_names_[i]; }

        if (!tau_hist_[i].empty())
        {
            double acc = 0.0;
            for (double v : tau_hist_[i]) acc += v * v;
            const double rms = std::sqrt(acc / tau_hist_[i].size());
            if (!finite(tau_rms) || rms > tau_rms) { tau_rms = rms; tau_rms_j = joint_names_[i]; }
        }
    }

    // 摘要里最多显示前 3 个状态变化
    std::string changes;
    int cnt = 0;
    for (size_t i = 0; i < n_ && cnt < 3; ++i)
    {
        const double cur = row[C_NUM + i * S_NUM + S_MOTORSTATE];
        if (motorstate_baseline_[i].has_value() && finite(cur) &&
            static_cast<int64_t>(cur) != *motorstate_baseline_[i])
        {
            if (!changes.empty()) changes += ",";
            changes += joint_names_[i] + ":" + std::to_string(*motorstate_baseline_[i]) +
                       "->" + std::to_string(static_cast<int64_t>(cur));
            ++cnt;
        }
    }
    const std::string state_str = changes.empty() ? "none" : changes;

    std::printf(
        "[G1-HEALTH] t=%7.1fs | hot=%s %.1fC | tau_peak=%s %.1fNm | tauRMS10=%s %.1fNm | "
        "qerr=%s %.3frad | BMS V=%.1f I=%.1f | loop=%.1fms infer=%.1fms stateAge=%.1fms | "
        "state_change=%s\n",
        row[C_ELAPSED], hot_j.c_str(), hot, taup_j.c_str(), taup,
        tau_rms_j.c_str(), tau_rms, qep_j.c_str(), qep,
        row[C_BMS_V], row[C_BMS_I], row[C_LOOP], row[C_INFER], row[C_AGE],
        state_str.c_str());
    std::fflush(stdout);
}

std::string G1HealthLogger::dumpRing(const std::string& tag)
{
    if (ring_.empty()) return "";
    std::string safe;
    for (char c : tag)
        safe += (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_';
    if (safe.size() > 80) safe.resize(80);

    const auto path = std::filesystem::path(cfg_.out_dir) /
        (run_id_ + "_" + timestampStr() + "_" + safe +
         "_PRE" + std::to_string(static_cast<int>(cfg_.ring_seconds)) + "S.csv");
    std::ofstream f(path);
    if (!f.is_open()) return "";

    for (size_t i = 0; i < fieldnames_.size(); ++i) { if (i) f << ","; f << fieldnames_[i]; }
    f << "\n";
    std::ostringstream os;
    os << std::setprecision(9);
    for (const auto& r : ring_)
    {
        os.str(""); os.clear();
        for (size_t i = 0; i < r.size(); ++i) { if (i) os << ","; writeNum(os, r[i]); }
        os << "\n";
        f << os.str();
    }
    f.flush();
    return path.string();
}

// ---------------------------------------------------------------------------
// 事件 / 周期
// ---------------------------------------------------------------------------
void G1HealthLogger::markEvent(const std::string& text, bool dump_ring)
{
    if (!events_path_.empty())
    {
        std::ofstream ev(events_path_, std::ios::app);
        if (ev.is_open())
        {
            std::ostringstream os;
            os << std::fixed << std::setprecision(6) << wallTimeS()
               << "\t" << std::fixed << std::setprecision(3) << (nowS() - t0_) << "s\t"
               << text << "\n";
            ev << os.str();
            ev.flush();
        }
    }
    std::printf("[G1-HEALTH EVENT] %s\n", text.c_str());
    std::fflush(stdout);
    if (dump_ring)
    {
        const std::string p = dumpRing(text);
        if (!p.empty())
            std::printf("[G1-HEALTH] pre-event buffer saved: %s\n", p.c_str());
    }
}

void G1HealthLogger::logCycle(const unitree_hg::msg::dds_::LowState_& low_state,
                              const unitree_hg::msg::dds_::LowCmd_& low_cmd,
                              const std::vector<float>& action_by_joint,
                              double loop_dt_s, double inference_s, double lowstate_age_s)
{
    if (!enabled_) return;

    const double now = nowS();

    // 动作含 inf -> 立即标记（即使未到采样时刻）
    for (float a : action_by_joint)
        if (std::isinf(a)) { markEvent("INFINITE_ACTION", true); break; }

    // 未显式提供的 timing 由 logger 自行测量
    const auto timing = timing_.update(low_state.tick());
    if (!finite(loop_dt_s)) loop_dt_s = timing.first;
    if (!finite(lowstate_age_s)) lowstate_age_s = timing.second;

    const double elapsed = now - t0_;

    // 降采样到 log_hz
    if (now - last_log_t_ < 1.0 / std::max(cfg_.log_hz, 1e-6)) return;
    last_log_t_ = now;

    auto row = buildRow(low_state, low_cmd, action_by_joint, elapsed,
                        loop_dt_s * 1e3, inference_s * 1e3, lowstate_age_s * 1e3);
    writeRow(row);
    ring_.push_back(row);
    if (ring_.size() > ring_cap_) ring_.pop_front();

    updateWindows(row);

    const std::string changes = detectStateChanges(row, elapsed);
    if (!changes.empty() && now - last_snapshot_t_ > 2.0)
    {
        last_snapshot_t_ = now;
        markEvent(changes, true);
    }

    if (now - last_print_t_ >= 1.0 / std::max(cfg_.print_hz, 1e-6))
    {
        last_print_t_ = now;
        printSummary(row);
    }

    if (now - last_flush_t_ >= 1.0)
    {
        last_flush_t_ = now;
        csv_fs_.flush();
    }
}

void G1HealthLogger::close()
{
    if (closed_) return;
    closed_ = true;
    if (enabled_)
    {
        markEvent("LOGGER_STOP");
        if (csv_fs_.is_open()) { csv_fs_.flush(); csv_fs_.close(); }
    }
    bms_sub_.reset();
}

} // namespace g1health
