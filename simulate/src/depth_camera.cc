#include "depth_camera.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include "depth_shm.h"

namespace depth_camera {
namespace {

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
depth_shm::Writer g_shm;

GLFWwindow* g_win = nullptr;      // hidden offscreen window + its own GL context
bool        g_ctx_ready = false;  // hidden context created (main thread)
bool        g_ready = false;      // mjrContext + scene built, publishing enabled

mjrContext g_con;
mjvScene   g_scn;
mjvOption  g_opt;
mjvCamera  g_cam;
int        g_cam_id = -1;

std::vector<float> g_norm;        // W*H raw depth buffer (reverse-Z, [0,1])
std::chrono::steady_clock::time_point g_last;

// Hyperbolic inverse-projection coefficients (mirrors mujoco.Renderer).
float g_coef_c = 0.0f;
float g_coef_d = 0.0f;
float g_far_m = 0.0f;
float g_near_m = 0.0f;

inline bool Due(std::chrono::steady_clock::time_point now) {
  const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(1.0 / kRenderHz));
  if (g_last.time_since_epoch().count() == 0) return true;   // first frame
  return (now - g_last) >= period;
}

}  // namespace

// ---------------------------------------------------------------------------
// PrepareContext: main-thread GL setup (hidden window + context)
// ---------------------------------------------------------------------------
bool PrepareContext() {
  if (g_ctx_ready) return true;

  // GLFW is already initialised by the viewer's PlatformUIAdapter. Keep the
  // viewer's context current on the caller's thread; we only create ours here.
  GLFWwindow* prev = glfwGetCurrentContext();

  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
  glfwWindowHint(GLFW_DEPTH_BITS, 24);
  g_win = glfwCreateWindow(kWidth, kHeight, "php_depth_offscreen", nullptr, nullptr);
  glfwDefaultWindowHints();

  if (!g_win) {
    std::printf("[depth_camera] WARNING: failed to create offscreen GL context; depth disabled\n");
    g_ctx_ready = false;
    return false;
  }
  g_ctx_ready = true;
  // Leave the caller's (viewer) context current until Init() runs on the
  // physics thread.
  glfwMakeContextCurrent(prev);
  std::printf("[depth_camera] offscreen GL context created (%dx%d)\n", kWidth, kHeight);
  std::fflush(stdout);
  return true;
}

// ---------------------------------------------------------------------------
// Init: bind the offscreen context to the physics thread and build the renderer
// ---------------------------------------------------------------------------
void Init(const mjModel* m) {
  // The shared-memory block is created unconditionally so the policy side can
  // attach (and see a constant far plane) even if rendering is unavailable.
  g_norm.assign(static_cast<size_t>(kWidth) * kHeight, 0.0f);
  g_shm.Create(kShmName, g_norm.size() * sizeof(float));
  g_last = std::chrono::steady_clock::time_point{};

  g_cam_id = mj_name2id(m, mjOBJ_CAMERA, kCameraName);
  if (g_cam_id < 0) {
    std::printf("[depth_camera] WARNING: camera '%s' not found in the model; depth disabled\n",
                kCameraName);
    return;
  }

  if (!g_ctx_ready || !g_win) {
    std::printf("[depth_camera] WARNING: no offscreen GL context (call PrepareContext first); "
                "depth disabled\n");
    return;
  }

  // Take ownership of the offscreen context on this (physics) thread.
  glfwMakeContextCurrent(g_win);

  mjr_makeContext(m, &g_con, mjFONTSCALE_150);
  g_con.readDepthMap = mjDEPTH_ZEROFAR;   // reversed-Z, same as mujoco.Renderer
  mjr_setBuffer(mjFB_OFFSCREEN, &g_con);

  mjv_defaultOption(&g_opt);
  g_opt.geomgroup[4] = 0;                 // group 4 = FOV visualisation only

  mjv_defaultCamera(&g_cam);
  g_cam.type = mjCAMERA_FIXED;
  g_cam.fixedcamid = g_cam_id;

  mjv_makeScene(m, &g_scn, 2000);

  // Clip planes: MuJoCo uses a GLOBAL frustum (vis.map * stat.extent), shared
  // with the viewer. Non-hits land on the far plane; we remap those later.
  const double extent = static_cast<double>(m->stat.extent);
  const double near_m = static_cast<double>(m->vis.map.znear) * extent;
  const double far_m  = static_cast<double>(m->vis.map.zfar) * extent;
  const float zfar  = static_cast<float>(far_m);
  const float znear = static_cast<float>(near_m);
  g_near_m = near_m;
  g_far_m  = far_m;
  // Perspective (reverse-Z) inverse projection — identical maths/logic to
  // mujoco/rendering/classic/renderer.py.
  g_coef_c = -(zfar + znear) / (zfar - znear);
  g_coef_d = -(static_cast<float>(2) * zfar * znear) / (zfar - znear);
  g_coef_c = static_cast<float>(-0.5) * g_coef_c - static_cast<float>(0.5);
  g_coef_d = static_cast<float>(-0.5) * g_coef_d;

  g_ready = true;
  std::printf("[depth_camera] camera '%s' (id=%d) ready | clip [%.4f, %.2f] m | render %dx%d @ %.0f Hz\n",
              kCameraName, g_cam_id, g_near_m, g_far_m, kWidth, kHeight, kRenderHz);
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// MaybeUpdate: render + publish (physics thread, simulator mutex held)
// ---------------------------------------------------------------------------
void MaybeUpdate(const mjModel* m, mjData* d) {
  if (!g_ready) return;
  const auto now = std::chrono::steady_clock::now();
  if (!Due(now)) return;
  g_last = now;

  mjv_updateScene(m, d, &g_opt, nullptr, &g_cam, mjCAT_ALL, &g_scn);

  // Extra depth precision at long range (as mujoco.Renderer does).
  g_scn.flags[mjRND_SEGMENT] = 1;
  g_scn.flags[mjRND_IDCOLOR] = 1;

  const mjrRect rect{0, 0, kWidth, kHeight};
  mjr_render(rect, &g_scn, &g_con);
  mjr_readPixels(nullptr, g_norm.data(), rect, &g_con);

  g_scn.flags[mjRND_SEGMENT] = 0;
  g_scn.flags[mjRND_IDCOLOR] = 0;

  // Reverse-Z -> metric image-plane depth (metres).
  // NOTE: GL framebuffers are bottom-up, so mjr_readPixels gives row 0 = BOTTOM.
  // The shm contract (and the D435i training pipeline) is row 0 = TOP, so flip
  // vertically here — this mirrors `out[:] = np.flipud(out)` in
  // mujoco/rendering/classic/renderer.py.
  float* out = g_shm.data();
  const double far_thresh = 0.99 * static_cast<double>(g_far_m);
  for (int row = 0; row < kHeight; ++row) {
    const int src_row = kHeight - 1 - row;
    for (int col = 0; col < kWidth; ++col) {
      const int src = src_row * kWidth + col;
      const double ndc = static_cast<double>(g_norm[src]);
      const double z = static_cast<double>(g_coef_d) / (ndc + static_cast<double>(g_coef_c));
      // Misses read the global far plane; treat them as "maximally far" so the
      // policy-side preprocessing clamps them to kFar (same as holosoma).
      out[row * kWidth + col] =
          (!std::isfinite(z) || z <= 0.0 || z >= far_thresh)
              ? static_cast<float>(kFar)
              : static_cast<float>(z);
    }
  }
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------
void Shutdown() {
  if (g_ready) {
    mjv_freeScene(&g_scn);
    mjr_freeContext(&g_con);
    g_ready = false;
  }
  if (g_win) {
    glfwMakeContextCurrent(nullptr);
    glfwDestroyWindow(g_win);
    g_win = nullptr;
    g_ctx_ready = false;
  }
  g_shm.Close();
  g_shm.Unlink();
  std::printf("[depth_camera] shutdown\n");
  std::fflush(stdout);
}

}  // namespace depth_camera
