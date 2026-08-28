#pragma once

#include "FSM/State_RLBase.h"
#include <atomic>
#include <cmath>
#include <cnpy.h>
#include <mutex>

class State_Mimic : public FSMState
{
public:
    State_Mimic(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable())
        {
            policy_thread.join();
        }
    }

    class MotionLoader_;
    static std::shared_ptr<MotionLoader_> motion;

private:
    bool update_action_snapshot();

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<MotionLoader_> motion_;

    std::thread policy_thread;
    std::atomic_bool policy_thread_running{false};
    std::atomic_bool policy_fault_{false};
    std::atomic<long> policy_steps_{0};
    std::atomic<int> bad_orientation_steps_{0};
    std::mutex action_mutex_;
    std::vector<float> action_snapshot_;
    std::array<float, 2> time_range_;
    float action_abort_limit_ = 100.0f;
    float bad_orientation_limit_ = 1.0f;
    int bad_orientation_confirm_steps_ = 1;
    bool bad_orientation_enabled_ = false;
};

class State_Mimic::MotionLoader_
{
public:
    MotionLoader_(
        const std::string &motion_file,
        float csv_fps,
        const std::vector<float> &joint_ids_map
    )
    {
        const auto extension = std::filesystem::path(motion_file).extension().string();
        if (extension == ".npz")
        {
            load_data_from_npz(motion_file);
        }
        else if (extension == ".csv")
        {
            load_data_from_csv(motion_file, csv_fps, joint_ids_map);
        }
        else
        {
            throw std::runtime_error("Unsupported Mimic motion format: " + motion_file);
        }
        num_frames = static_cast<int>(dof_positions.size());
        duration = num_frames * dt;
        update(0.0f);
    }

    void load_data_from_csv(
        const std::string &motion_file,
        float fps,
        const std::vector<float> &joint_ids_map
    )
    {
        if (!std::isfinite(fps) || fps <= 0.0f)
        {
            throw std::runtime_error("CSV motion fps must be positive: " + motion_file);
        }
        if (joint_ids_map.size() != 29)
        {
            throw std::runtime_error("CSV motion requires a 29-entry joint_ids_map");
        }

        const auto rows = isaaclab::load_csv(motion_file);
        if (rows.empty())
        {
            throw std::runtime_error("CSV motion is empty: " + motion_file);
        }
        const std::size_t columns = rows.front().size();
        if (columns != 36 && columns != 65)
        {
            throw std::runtime_error(
                "CSV motion must have 36 (q) or 65 (q+dq) columns, got " +
                std::to_string(columns)
            );
        }

        dt = 1.0f / fps;
        root_positions.reserve(rows.size());
        root_quaternions.reserve(rows.size());
        dof_positions.reserve(rows.size());
        if (columns == 65)
        {
            dof_velocities.reserve(rows.size());
        }

        for (std::size_t frame_index = 0; frame_index < rows.size(); ++frame_index)
        {
            const auto &row = rows[frame_index];
            if (row.size() != columns)
            {
                throw std::runtime_error(
                    "Inconsistent CSV column count at frame " + std::to_string(frame_index)
                );
            }

            root_positions.emplace_back(Eigen::Vector3f::Map(row.data()));
            root_quaternions.emplace_back(row[6], row[3], row[4], row[5]);

            Eigen::VectorXf policy_pos(29);
            Eigen::VectorXf policy_vel(29);
            for (std::size_t policy_index = 0; policy_index < 29; ++policy_index)
            {
                const float mapped_id = joint_ids_map[policy_index];
                const int sdk_id = static_cast<int>(std::lround(mapped_id));
                if (!std::isfinite(mapped_id) || std::abs(mapped_id - sdk_id) > 1.0e-5f ||
                    sdk_id < 0 || sdk_id >= 29)
                {
                    throw std::runtime_error(
                        "Invalid joint_ids_map entry at policy index " +
                        std::to_string(policy_index)
                    );
                }
                policy_pos[policy_index] = row[7 + sdk_id];
                if (columns == 65)
                {
                    policy_vel[policy_index] = row[36 + sdk_id];
                }
            }
            dof_positions.push_back(std::move(policy_pos));
            if (columns == 65)
            {
                dof_velocities.push_back(std::move(policy_vel));
            }
        }

        if (columns == 36)
        {
            dof_velocities = compute_derivative(dof_positions, dt);
        }
    }

    void load_data_from_npz(const std::string &motion_file)
    {
        const cnpy::npz_t npz_data = cnpy::npz_load(motion_file);
        const auto &body_pos_w = require_array(npz_data, "body_pos_w", motion_file);
        const auto &body_quat_w = require_array(npz_data, "body_quat_w", motion_file);
        const auto &joint_pos = require_array(npz_data, "joint_pos", motion_file);
        const auto &joint_vel = require_array(npz_data, "joint_vel", motion_file);
        const auto &fps = require_array(npz_data, "fps", motion_file);

        validate_float_array(body_pos_w, "body_pos_w", 3);
        validate_float_array(body_quat_w, "body_quat_w", 3);
        validate_float_array(joint_pos, "joint_pos", 2);
        validate_float_array(joint_vel, "joint_vel", 2);

        if (body_pos_w.shape[2] != 3 || body_quat_w.shape[2] != 4)
        {
            throw std::runtime_error("Motion body arrays have invalid trailing dimensions: " + motion_file);
        }
        if (joint_pos.shape[1] != 29 || joint_vel.shape[1] != 29)
        {
            throw std::runtime_error("Motion must contain 29 joints: " + motion_file);
        }

        const std::size_t frames = joint_pos.shape[0];
        if (frames == 0 || joint_vel.shape[0] != frames ||
            body_pos_w.shape[0] != frames || body_quat_w.shape[0] != frames)
        {
            throw std::runtime_error("Motion arrays have inconsistent frame counts: " + motion_file);
        }

        if (fps.num_vals == 0)
        {
            throw std::runtime_error("Motion fps array is empty: " + motion_file);
        }

        double fps_value = 0.0;
        if (fps.word_size == sizeof(double))
        {
            fps_value = fps.data<double>()[0];
        }
        else if (fps.word_size == sizeof(float))
        {
            fps_value = fps.data<float>()[0];
        }
        else
        {
            throw std::runtime_error("Motion fps must be float32 or float64: " + motion_file);
        }
        if (!std::isfinite(fps_value) || fps_value <= 0.0)
        {
            throw std::runtime_error("Motion fps must be positive: " + motion_file);
        }
        dt = 1.0f / static_cast<float>(fps_value);

        const float *body_pos_data = body_pos_w.data<float>();
        const float *body_quat_data = body_quat_w.data<float>();
        const float *joint_pos_data = joint_pos.data<float>();
        const float *joint_vel_data = joint_vel.data<float>();
        const std::size_t body_pos_stride = body_pos_w.shape[1] * body_pos_w.shape[2];
        const std::size_t body_quat_stride = body_quat_w.shape[1] * body_quat_w.shape[2];

        root_positions.reserve(frames);
        root_quaternions.reserve(frames);
        dof_positions.reserve(frames);
        dof_velocities.reserve(frames);

        for (std::size_t i = 0; i < frames; ++i)
        {
            root_positions.emplace_back(Eigen::Vector3f::Map(body_pos_data + i * body_pos_stride));
            root_quaternions.emplace_back(
                body_quat_data[i * body_quat_stride + 0],
                body_quat_data[i * body_quat_stride + 1],
                body_quat_data[i * body_quat_stride + 2],
                body_quat_data[i * body_quat_stride + 3]
            );
            dof_positions.emplace_back(
                Eigen::VectorXf::Map(joint_pos_data + i * joint_pos.shape[1], joint_pos.shape[1])
            );
            dof_velocities.emplace_back(
                Eigen::VectorXf::Map(joint_vel_data + i * joint_vel.shape[1], joint_vel.shape[1])
            );
        }
    }

    void update(float time)
    {
        const float clamped_time = std::clamp(time, 0.0f, duration);
        frame = std::min(static_cast<int>(std::floor(clamped_time / dt)), num_frames - 1);
    }

    void reset(const isaaclab::ArticulationData &data, float time = 0.0f)
    {
        update(time);
        const auto init_to_anchor = isaaclab::yawQuaternion(root_quaternion()).toRotationMatrix();
        const auto world_to_anchor = isaaclab::yawQuaternion(data.root_quat_w).toRotationMatrix();
        world_to_init_ = world_to_anchor * init_to_anchor.transpose();
    }

    Eigen::VectorXf root_position() const { return root_positions[frame]; }
    Eigen::Quaternionf root_quaternion() const { return root_quaternions[frame]; }
    Eigen::VectorXf joint_pos() const { return dof_positions[frame]; }
    Eigen::VectorXf joint_vel() const { return dof_velocities[frame]; }
    int frame_index() const { return frame; }

    int waist_yaw_idx = 2;
    int waist_roll_idx = 5;
    int waist_pitch_idx = 8;

    float dt = 0.02f;
    int num_frames = 0;
    float duration = 0.0f;
    Eigen::Matrix3f world_to_init_;

private:
    static std::vector<Eigen::VectorXf> compute_derivative(
        const std::vector<Eigen::VectorXf> &values,
        float dt
    )
    {
        std::vector<Eigen::VectorXf> derivative;
        derivative.reserve(values.size());
        if (values.size() == 1)
        {
            derivative.push_back(Eigen::VectorXf::Zero(values.front().size()));
            return derivative;
        }
        for (std::size_t i = 0; i + 1 < values.size(); ++i)
        {
            derivative.push_back((values[i + 1] - values[i]) / dt);
        }
        derivative.push_back(derivative.back());
        return derivative;
    }

    static const cnpy::NpyArray &require_array(
        const cnpy::npz_t &data,
        const std::string &name,
        const std::string &motion_file
    )
    {
        const auto it = data.find(name);
        if (it == data.end())
        {
            throw std::runtime_error("Motion is missing array '" + name + "': " + motion_file);
        }
        return it->second;
    }

    static void validate_float_array(
        const cnpy::NpyArray &array,
        const std::string &name,
        std::size_t dimensions
    )
    {
        if (array.word_size != sizeof(float) || array.shape.size() != dimensions || array.fortran_order)
        {
            throw std::runtime_error(
                "Motion array '" + name + "' must be a C-contiguous float32 array with " +
                std::to_string(dimensions) + " dimensions"
            );
        }
    }

    int frame = 0;
    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;
};

REGISTER_FSM(State_Mimic)
