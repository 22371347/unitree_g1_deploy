// Depth camera publisher for the G1 sim2sim rig (PHP / Perceptive Humanoid Parkour).
//
// Publishes RAW metric depth (106x60, float32, row-major, metres, image-plane
// Z depth) to a POSIX shared-memory block.  The crop / clip / bicubic-resize /
// normalise preprocessing is intentionally NOT done here: the deployment-side
// runner (food for depth_backbone.onnx) performs it with torch so that it is
// bit-identical to training.
//
// Acquisition path (matches holosoma's official sim2sim EXACTLY):
//   MJCF <camera name="d435i_depth">  (torso_link, pos/quat/fovy per training rig)
//   -> mjr_render + mjr_readPixels    (offscreen, reverse-Z depth buffer)
//   -> hyperbolic inverse projection  (same maths as mujoco.Renderer)
//   => metric Z depth in metres, with the robot's own near-field geometry
//      (chest rim / arms / thighs) included, because GL backface culling makes
//      it visible from inside the torso shell just like in training.
//
// Camera model (D435i training rig, holosoma: sensor.py -> d435i_front_depth_camera):
//   - mounted on `torso_link`
//   - offset_pos = (0.01, 0.01, 0.44) m
//   - quat (wxyz) = (0.60290764, 0.37585401, -0.362958, -0.60290764)
//     == euler(1, 27, 1) deg (sensor -> base) composed with euler(-90, 0, -90) deg
//   - 106 x 60 px, vertical FOV 58.5953 deg (<=> HFOV 89.5 deg), clip [0.3, 3.0] m
//
// NOTE: the camera element lives in `scene_g1.xml`; the numbers must stay in
// sync with holosoma's `config_values/wbt/g1/sensor.py`.
#pragma once

#include <mujoco/mujoco.h>

namespace depth_camera {

// Which shared-memory block to publish into ("/dev/shm/<name>").
inline constexpr const char* kShmName = "depth_raw_shm";
inline constexpr int    kWidth    = 106;
inline constexpr int    kHeight   = 60;
inline constexpr double kFar      = 3.0;    // non-hit / out-of-range sentinel
inline constexpr double kRenderHz = 10.0;
inline constexpr const char* kCameraName = "d435i_depth";

// Create a hidden offscreen GL context (shares GLFW with the viewer, but its own
// window+context).  MUST be called from the main thread BEFORE the physics
// thread starts, so that the physics thread can take ownership of it later.
// Returns true on success; on failure depth publishing is disabled.
bool PrepareContext();

// Resolve the camera element, bind the offscreen context to the calling
// (physics) thread, build the mjrContext / mjvScene and create the shm block.
// Must be called once after the model is loaded, from the thread that will call
// MaybeUpdate().
void Init(const mjModel* m);

// Render one depth frame (rate-limited to kRenderHz) and publish it to shared
// memory.  Must be called from the physics thread right after a step, while the
// simulator mutex is held (it reads mjModel/mjData).
void MaybeUpdate(const mjModel* m, mjData* d);

// Release the GL resources and unlink the shared-memory block.
void Shutdown();

}  // namespace depth_camera
