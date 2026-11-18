#include "depth_camera.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <mujoco/mujoco.h>

#include "depth_shm.h"

namespace depth_camera {
namespace {

// ---------------------------------------------------------------------------
// Quaternion helpers (w, x, y, z) — MuJoCo's native convention.
// ---------------------------------------------------------------------------
struct Quat {
  double w, x, y, z;
};

// IsaacLab's quat_from_euler_xyz_tensor: roll about X, pitch about Y, yaw about Z,
// composed as R = Rz(yaw) * Ry(pitch) * Rx(roll).  Matches training.
inline Quat EulerXyzToQuat(double roll_deg, double pitch_deg, double yaw_deg) {
  const double r = roll_deg * M_PI / 180.0;
  const double p = pitch_deg * M_PI / 180.0;
  const double y = yaw_deg * M_PI / 180.0;
  const double cy = std::cos(y * 0.5), sy = std::sin(y * 0.5);
  const double cr = std::cos(r * 0.5), sr = std::sin(r * 0.5);
  const double cp = std::cos(p * 0.5), sp = std::sin(p * 0.5);
  return Quat{
      cr * cp * cy + sr * sp * sy,  // w
      sr * cp * cy - cr * sp * sy,  // x
      cr * sp * cy + sr * cp * sy,  // y
      cr * cp * sy - sr * sp * cy,  // z
  };
}

inline Quat QuatMul(const Quat& a, const Quat& b) {
  return Quat{
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
  };
}

// Rotate vector v by quaternion q (w,x,y,z).
inline void QuatRotate(const Quat& q, const double v[3], double out[3]) {
  const double w = q.w, x = q.x, y = q.y, z = q.z;
  const double R[3][3] = {
      {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
      {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
      {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)},
  };
  for (int i = 0; i < 3; ++i) {
    out[i] = R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2];
  }
}

inline double Dot3(const double a[3], const double b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
depth_shm::Writer g_shm;
int g_body_id = -1;
Quat g_q_local{};    // euler(offset_rot)
Quat g_q_data{};     // euler(offset_rot_base)
double g_offset_pos[3] = {0.01, 0.01, 0.44};
double g_fx = 0.0, g_fy = 0.0, g_cx = 0.0, g_cy = 0.0;
// Precomputed, normalised camera-local ray directions, 3 * (W*H), row-major.
std::vector<double> g_local_dirs;
std::vector<double> g_world_dirs;   // scratch, 3 * (W*H)
std::vector<double> g_multiplier;   // scratch, W*H
std::vector<int> g_geomid;          // scratch, W*H
std::vector<mjtNum> g_dist;         // scratch, W*H
std::chrono::steady_clock::time_point g_last;

// ---------------------------------------------------------------------------
// Pose helpers
// ---------------------------------------------------------------------------
inline Quat BodyQuat(const mjData* d, int bid) {
  const mjtNum* q = d->xquat + 4 * bid;  // MuJoCo world quat is w,x,y,z
  return Quat{q[0], q[1], q[2], q[3]};
}

}  // namespace

void Init(const mjModel* m) {
  g_body_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
  if (g_body_id < 0) {
    g_body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
  }
  if (g_body_id < 0) {
    std::printf("[depth_camera] WARNING: no torso_link/base_link body; depth disabled\n");
    return;
  }

  g_q_local = EulerXyzToQuat(1.0, 27.0, 1.0);
  g_q_data = EulerXyzToQuat(-90.0, 0.0, -90.0);

  // Pinhole intrinsics (square pixels: focal_length from HFOV, VFOV derived).
  g_cx = 0.5 * kWidth;
  g_cy = 0.5 * kHeight;
  const double hfov = kHfovDeg * M_PI / 180.0;
  g_fx = g_cx / std::tan(hfov * 0.5);
  g_fy = g_fx;  // vertical FOV derived from the same focal length

  // Precompute normalised camera-local ray directions.
  const int n = kWidth * kHeight;
  g_local_dirs.assign(3 * n, 0.0);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const int idx = y * kWidth + x;
      double d[3] = {(x - g_cx) / g_fx, (y - g_cy) / g_fy, 1.0};
      const double inv = 1.0 / std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      g_local_dirs[3 * idx + 0] = d[0] * inv;
      g_local_dirs[3 * idx + 1] = d[1] * inv;
      g_local_dirs[3 * idx + 2] = d[2] * inv;
    }
  }
  g_world_dirs.assign(3 * n, 0.0);
  g_multiplier.assign(n, 1.0);
  g_geomid.assign(n, -1);
  g_dist.assign(n, 0.0);

  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  g_shm.Create(kShmName, bytes);
  g_last = std::chrono::steady_clock::time_point{};

  // Log the camera forward axis once so operators can sanity-check the mount.
  double z_axis_init[3] = {0.0, 0.0, 1.0};
  double fwd[3];
  Quat qc = QuatMul(g_q_local, g_q_data);
  QuatRotate(qc, z_axis_init, fwd);
  std::printf("[depth_camera] torso body id=%d  cam forward=(%.3f, %.3f, %.3f)\n",
              g_body_id, fwd[0], fwd[1], fwd[2]);
}

void MaybeUpdate(const mjModel* m, mjData* d) {
  if (!g_shm.ok() || g_body_id < 0) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  if (g_last.time_since_epoch().count() != 0) {
    const double dt = std::chrono::duration<double>(now - g_last).count();
    if (dt < 1.0 / kRenderHz) {
      return;
    }
  }
  g_last = now;

  // --- world camera pose -----------------------------------------------------
  const double* base_pos = d->xpos + 3 * g_body_id;
  const Quat base_quat = BodyQuat(d, g_body_id);

  double rot_off[3];
  QuatRotate(base_quat, g_offset_pos, rot_off);
  double cam_pos[3] = {base_pos[0] + rot_off[0], base_pos[1] + rot_off[1],
                       base_pos[2] + rot_off[2]};

  const Quat cam_quat = QuatMul(QuatMul(base_quat, g_q_local), g_q_data);

  double z_axis[3] = {0.0, 0.0, 1.0};
  double principal[3];
  QuatRotate(cam_quat, z_axis, principal);

  // --- per-pixel world rays + planar multipliers -----------------------------
  const int n = kWidth * kHeight;
  double min_mult = 1.0;
  for (int idx = 0; idx < n; ++idx) {
    const double* dl = &g_local_dirs[3 * idx];
    double dw[3];
    QuatRotate(cam_quat, dl, dw);
    g_world_dirs[3 * idx + 0] = dw[0];
    g_world_dirs[3 * idx + 1] = dw[1];
    g_world_dirs[3 * idx + 2] = dw[2];
    double mult = Dot3(dw, principal);
    if (mult < 1e-6) mult = 1e-6;
    if (mult > 1.0) mult = 1.0;
    g_multiplier[idx] = mult;
    if (mult < min_mult) min_mult = mult;
  }

  // Cutoff chosen so that planar <= far for every ray (matches warp's
  // far_plane / multiplier per-ray cutoff, collapsed to the worst case).
  const mjtNum cutoff = static_cast<mjtNum>(kFar / (min_mult > 1e-3 ? min_mult : 1e-3));

  // --- raycast ---------------------------------------------------------------
  // 只包含 group 0/1（碰撞体+视觉体），排除 group 4 的深度FOV可视化胶囊，
  // 避免把视锥自身当成障碍物。
  static const mjtByte kGeomGroup[6] = {1, 1, 0, 0, 0, 0};
  mj_multiRay(m, d, cam_pos, g_world_dirs.data(), kGeomGroup,
              /*flg_static=*/1, /*bodyexclude=*/g_body_id, g_geomid.data(),
              g_dist.data(), n, cutoff);

  // --- planar depth -> shm ---------------------------------------------------
  float* out = g_shm.data();
  for (int idx = 0; idx < n; ++idx) {
    if (g_geomid[idx] < 0 || g_dist[idx] < 0) {
      out[idx] = static_cast<float>(kFar);  // non-hit -> maximally far
    } else {
      out[idx] = static_cast<float>(g_multiplier[idx] * static_cast<double>(g_dist[idx]));
    }
  }
}

void Shutdown() {
  if (g_shm.ok()) {
    g_shm.Unlink();
    g_shm.Close();
  }
}

}  // namespace depth_camera
