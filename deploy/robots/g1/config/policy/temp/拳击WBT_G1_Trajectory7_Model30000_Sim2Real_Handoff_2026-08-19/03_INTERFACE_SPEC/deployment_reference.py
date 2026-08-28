"""Reference math/interface for WBT G1 trajectory7 model_30000.

This file does NOT communicate with a robot. It only documents the vector
conventions used by the policy.
"""
import numpy as np

POLICY_JOINT_NAMES = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
SDK_JOINT_NAMES = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
POLICY_TO_SDK = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28], dtype=np.int64)
SDK_TO_POLICY = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28], dtype=np.int64)

Q_DEFAULT_POLICY = np.array([-0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.669, 0.669, 0.2, 0.2, -0.363, -0.363, 0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ACTION_SCALE_POLICY = np.array([0.5475464651829304, 0.5475464651829304, 0.5475464651829304, 0.35066146635877904, 0.35066146635877904, 0.43857731389829635, 0.5475464651829304, 0.5475464651829304, 0.43857731389829635, 0.35066146635877904, 0.35066146635877904, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.43857731389829635, 0.07450087032524835, 0.07450087032524835, 0.07450087032524835, 0.07450087032524835], dtype=np.float32)
TRAJECTORY7_STANCE_POLICY = np.array([-0.6452410221099854, -0.4149630069732666, 0.0, 0.15130600333213806, -0.25377100706100464, 0.0, 0.0012659999774768949, -0.20854699611663818, 0.0, 0.9383400082588196, 1.0250810384750366, -0.7280430197715759, -0.755416989326477, -0.2698360085487366, -0.572281002998352, 0.10822299867868423, -0.2237520068883896, -0.09753499925136566, 0.07596000283956528, -0.0092569999396801, 0.24097299575805664, 0.08785399794578552, -0.08818800002336502, -0.12570199370384216, 0.12570199370384216, -0.06600599735975266, -0.06600599735975266, -0.024761000648140907, 0.03080200031399727], dtype=np.float32)


def sdk_to_policy(x_sdk):
    x_sdk = np.asarray(x_sdk)
    if x_sdk.shape[-1] != 29:
        raise ValueError("Expected last dimension 29")
    return x_sdk[..., POLICY_TO_SDK]


def policy_to_sdk(x_policy):
    x_policy = np.asarray(x_policy)
    if x_policy.shape[-1] != 29:
        raise ValueError("Expected last dimension 29")
    out = np.empty_like(x_policy)
    out[..., POLICY_TO_SDK] = x_policy
    return out


def quat_inv_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    denom = np.sum(q*q, axis=-1, keepdims=True)
    return out / denom


def quat_mul_wxyz(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def matrix_from_quat_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def motion_anchor_ori_6d(current_torso_quat_wxyz, reference_torso_quat_wxyz):
    """Returns [R00,R01,R10,R11,R20,R21]."""
    q_rel = quat_mul_wxyz(
        quat_inv_wxyz(current_torso_quat_wxyz),
        reference_torso_quat_wxyz,
    )
    R = matrix_from_quat_wxyz(q_rel)
    return R[..., :, :2].reshape(R.shape[:-2] + (6,)).astype(np.float32)


def action_to_q_des_policy(raw_action):
    raw_action = np.asarray(raw_action, dtype=np.float32)
    if raw_action.shape[-1] != 29:
        raise ValueError("Expected 29D raw action")
    return Q_DEFAULT_POLICY + ACTION_SCALE_POLICY * raw_action


def build_obs(
    ref_q_policy,
    ref_dq_policy,
    current_torso_quat_wxyz,
    reference_torso_quat_wxyz,
    base_gyro_body_xyz,
    q_policy,
    dq_policy,
    last_raw_action,
):
    ref_q_policy = np.asarray(ref_q_policy, dtype=np.float32)
    ref_dq_policy = np.asarray(ref_dq_policy, dtype=np.float32)
    gyro = np.asarray(base_gyro_body_xyz, dtype=np.float32)
    q_policy = np.asarray(q_policy, dtype=np.float32)
    dq_policy = np.asarray(dq_policy, dtype=np.float32)
    last_raw_action = np.asarray(last_raw_action, dtype=np.float32)

    ori6 = motion_anchor_ori_6d(
        current_torso_quat_wxyz,
        reference_torso_quat_wxyz,
    )

    obs = np.concatenate([
        ref_q_policy,
        ref_dq_policy,
        ori6,
        gyro,
        q_policy - Q_DEFAULT_POLICY,
        dq_policy,
        last_raw_action,
    ]).astype(np.float32)

    if obs.shape != (154,):
        raise ValueError(f"Expected observation shape (154,), got {obs.shape}")
    return obs


if __name__ == "__main__":
    print("policy joints:", len(POLICY_JOINT_NAMES))
    print("obs dim: 154")
    print("action dim: 29")
    print("policy dt: 0.02 s")
