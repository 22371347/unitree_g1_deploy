// Depth camera publisher for the G1 sim2sim rig (PHP / Perceptive Humanoid Parkour).
//
// Publishes RAW planar metric depth (106x60, float32, row-major, metres) to a
// POSIX shared-memory block.  The crop / clip / bicubic-resize / normalise
// preprocessing is intentionally NOT done here: the deployment-side runner (food
// for depth_backbone.onnx) performs it with torch so that it is bit-identical to
// training (`torch.nn.functional.interpolate(mode="bicubic", antialias=True)`).
//
// Camera model (D435i training rig, holosoma:
//   sensors/warp/camera_config/d435i_depth_config.py -> G1FlatRsD435iConfig):
//   - mounted on `torso_link`
//   - offset_pos      = (0.01, 0.01, 0.44) m
//   - offset_rot      = (1.0, 27.0, 1.0) deg   (sensor -> sensor_base_link)
//   - offset_rot_base = (-90.0, 0.0, -90.0) deg
//   - 106 x 60 px, horizontal FOV 89.5 deg, near 0.3 m, far 3.0 m
// World pose composition (holosoma managers/observation/terms/depth.py):
//   cam_pos  = base_pos + R(base_quat) * offset_pos
//   cam_quat = base_quat (x) q_local (x) q_data          (wxyz throughout)
// Ray model (holosoma sensors/warp/camera_kernels_warp.py):
//   d_local  = normalize( ((x-cx)/fx, (y-cy)/fy, 1) )
//   d_world  = R(cam_quat) * d_local
//   planar   = dist * clamp(dot(d_world, R(cam_quat)*[0,0,1]), 1e-6, 1)
#pragma once

#include <mujoco/mujoco.h>

namespace depth_camera {

// Which shared-memory block to publish into ("/dev/shm/<name>").
inline constexpr const char* kShmName = "depth_raw_shm";
inline constexpr int    kWidth    = 106;
inline constexpr int    kHeight   = 60;
inline constexpr double kHfovDeg  = 89.5;
inline constexpr double kNear     = 0.3;
inline constexpr double kFar      = 3.0;
inline constexpr double kRenderHz = 10.0;

// Resolve the camera body, precompute per-pixel ray directions and create the
// shared-memory block.  Safe to call once after the model is loaded.
void Init(const mjModel* m);

// Cast the depth frame and publish it to shared memory, rate-limited to
// kRenderHz.  Must be called from the physics thread (it mutates mjData via
// mj_multiRay) right after a step.  Cheap enough to call every step.
void MaybeUpdate(const mjModel* m, mjData* d);

// Release / unlink the shared-memory block.
void Shutdown();

}  // namespace depth_camera
