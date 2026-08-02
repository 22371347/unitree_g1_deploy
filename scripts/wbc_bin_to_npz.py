#!/usr/bin/env python3
"""wbc_fsm 运动数据 (.bin) → NPZ 转换脚本

将 wbc_fsm 项目的 LAFAN1 运动数据（二进制 .bin 文件）转换为
unitree_rl_mjlab 部署框架所需的 NPZ 格式（与 BeyondMimic 的 motion.npz 结构兼容）。

用法:
    python wbc_bin_to_npz.py --input_dir <wbc_fsm的motion_data路径> --output <输出的motion.npz路径>

示例:
    python wbc_bin_to_npz.py \
        --input_dir /home/leo/g1_ws/wbc_fsm/motion_data/lafan1/dance12 \
        --output /home/leo/g1_ws/unitree_rl_mjlab/deploy/robots/g1/config/policy/wbc/params/motion.npz
"""

import argparse
import struct
import os

import numpy as np


def read_bin_array(filepath):
    """读取 wbc_fsm 自定义二进制格式 (.bin)

    文件头格式:
        magic:  'NPZ\\0' (4 bytes)
        ndims:  uint32
        shape:  uint32 × ndims
        dtype_size: uint32
        dtype_code: char (1 byte)   'f'=float32, 'l'=int64 等
        reserved: 3 bytes
        data:   dtype 数据

    返回:
        numpy.ndarray
    """
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != b"NPZ\x00":
            raise ValueError(f"Invalid file format (magic={magic!r}): {filepath}")

        ndims = struct.unpack("<I", f.read(4))[0]
        shape = struct.unpack("<" + "I" * ndims, f.read(4 * ndims))
        dtype_size = struct.unpack("<I", f.read(4))[0]
        dtype_code = f.read(1).decode("ascii")  # 单字节 dtype 标识
        f.read(3)  # reserved

        dtype_map = {
            "f": (np.float32, 4),
            "d": (np.float64, 8),
            "l": (np.int64, 8),
            "i": (np.int32, 4),
        }
        if dtype_code not in dtype_map:
            raise ValueError(f"Unsupported dtype code: {dtype_code!r} in {filepath}")
        np_dtype, expect_size = dtype_map[dtype_code]
        if dtype_size != expect_size:
            raise ValueError(f"dtype_size mismatch in {filepath}: {dtype_size}")

        n_elements = int(np.prod(shape))
        data = np.fromfile(f, dtype=np_dtype, count=n_elements)
        if data.size != n_elements:
            raise ValueError(f"Data size mismatch in {filepath}: expected {n_elements}, got {data.size}")

    return data.reshape(shape)


def convert(input_dir, output_path):
    """读取所有 .bin 文件并打包成 NPZ"""
    required = {
        "body_pos_w": os.path.join(input_dir, "body_pos_w.bin"),
        "body_quat_w": os.path.join(input_dir, "body_quat_w.bin"),
        "joint_pos": os.path.join(input_dir, "joint_pos.bin"),
        "joint_vel": os.path.join(input_dir, "joint_vel.bin"),
    }

    # 可选数据（wbc 观测不需要，但保留以备扩展）
    optional = {
        "body_lin_vel_w": os.path.join(input_dir, "body_lin_vel_w.bin"),
        "body_ang_vel_w": os.path.join(input_dir, "body_ang_vel_w.bin"),
        "fps": os.path.join(input_dir, "fps.bin"),
    }

    out_dict = {}
    for key, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")
        arr = read_bin_array(path)
        out_dict[key] = arr
        print(f"[OK] {key}: shape={arr.shape} dtype={arr.dtype}")

    for key, path in optional.items():
        if os.path.exists(path):
            arr = read_bin_array(path)
            out_dict[key] = arr
            print(f"[OK] {key}: shape={arr.shape} dtype={arr.dtype}")
        else:
            print(f"[SKIP] {key}: not found")

    # 打印帧数与帧率，便于核对
    n_frames = out_dict["body_pos_w"].shape[0]
    fps = int(out_dict.get("fps", np.array([50]))[0])
    print(f"\nFrames: {n_frames}, fps: {fps}, duration: {n_frames/fps:.2f}s")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **out_dict)
    print(f"\n[SUCCESS] Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="wbc_fsm bin → NPZ 转换")
    parser.add_argument("--input_dir", required=True, help="wbc_fsm motion_data 文件夹路径")
    parser.add_argument("--output", required=True, help="输出的 NPZ 文件路径")
    args = parser.parse_args()

    convert(args.input_dir, args.output)
