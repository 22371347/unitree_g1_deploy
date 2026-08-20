#pragma once

#include <cmath>
#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

// 机体坐标系下的重力投影：直立时 z = -1，倾角 θ 时 z = -cos(θ)，
// 故机体倾角 = acos(-z)。超过 limit_angle（弧度）判定为姿态异常（摔倒）。
// 数据源为 IMU 四元数（projected_gravity_b），仿真/实机通用。
inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::fabs(std::acos(-data[2])) > limit_angle;
}

} 
} 