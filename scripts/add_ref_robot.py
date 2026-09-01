#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 unitree_mujoco 场景添加一个"参考机器人"（静态显示当前策略 default_joint_pos 默认姿态），
与真实运动机器人并行对比观察。本方案完全不影响真实机器人（不加 keyframe、不改 qpos0）。

原理：
- 复制 G1 机器人本体，body/joint/site 加 ref_ 前缀，整体平移到 (0, 1.5, 0.793)
- 参考机器人所有关节改为 type="fixed"，并把每个 body 的 pos/quat 替换为
  "默认姿态下的 FK 相对位姿"（用 MuJoCo mj_forward 烘焙得到）→ 参考机器人
  变成"默认姿态的静态雕像"，不会倒下、不参与物理

注意：FK 数据由 /tmp/dump_fk 在 AMP default_joint_pos 下导出。
换策略时重新导出 FK 或用对应 deploy.yaml 的 default_joint_pos 重算。
用法： python3 add_ref_robot.py [scene_g1.xml]（自动备份 .bak）
"""
import re
import shutil
import sys

REF_OFFSET = "0 1.5 0.793"   # 参考机器人位置（y 方向并排）

# 默认姿态 FK 相对位姿：body 名 -> (pos[3], quat[4])（相对父级 body，AMP default_joint_pos 下烘焙）
FK = {
    "pelvis":               ([0.0, 0.0, 0.793],          [1.0, 0.0, 0.0, 0.0]),
    "left_hip_pitch_link":  ([0.0, 0.064452, -0.1027],   [0.98785666, 0.0, -0.15536803, 0.0]),
    "left_hip_roll_link":   ([0.0, 0.052, -0.030465],    [0.99617869, 0.0, -0.08733857, 0.0]),
    "left_hip_yaw_link":    ([0.025001, 0.0, -0.12412],   [1.0, 0.0, 0.0, 0.0]),
    "left_knee_link":       ([-0.078273, 0.0021489, -0.17734], [0.91229208, 0.0, 0.40954019, 0.0]),
    "left_ankle_pitch_link":([0.0, -0.00009444, -0.30001], [0.98357404, 0.0, -0.18050514, 0.0]),
    "left_ankle_roll_link": ([0.0, 0.0, -0.017558],       [1.0, 0.0, 0.0, 0.0]),
    "right_hip_pitch_link": ([0.0, -0.064452, -0.1027],  [0.98785666, 0.0, -0.15536803, 0.0]),
    "right_hip_roll_link":  ([0.0, -0.052, -0.030465],   [0.99617869, 0.0, -0.08733857, 0.0]),
    "right_hip_yaw_link":   ([0.025001, 0.0, -0.12412],   [1.0, 0.0, 0.0, 0.0]),
    "right_knee_link":      ([-0.078273, -0.0021489, -0.17734], [0.91229208, 0.0, 0.40954019, 0.0]),
    "right_ankle_pitch_link":([0.0, 0.00009444, -0.30001], [0.98357404, 0.0, -0.18050514, 0.0]),
    "right_ankle_roll_link":([0.0, 0.0, -0.017558],       [1.0, 0.0, 0.0, 0.0]),
    "waist_yaw_link":       ([0.0, 0.0, 0.0],             [1.0, 0.0, 0.0, 0.0]),
    "waist_roll_link":      ([-0.0039635, 0.0, 0.044],    [1.0, 0.0, 0.0, 0.0]),
    "torso_link":           ([0.0, 0.0, 0.0],             [1.0, 0.0, 0.0, 0.0]),
    "left_shoulder_pitch_link": ([0.0039563, 0.10022, 0.24778], [0.98531556, 0.13851545, 0.09887526, 0.01379872]),
    "left_shoulder_roll_link":  ([0.0, 0.038, -0.013831],  [0.99921502, -0.03961489, 0.0, 0.0]),
    "left_shoulder_yaw_link":   ([0.0, 0.00624, -0.1032],  [1.0, 0.0, 0.0, 0.0]),
    "left_elbow_link":      ([0.015783, 0.0, -0.080518],  [0.95533649, 0.0, 0.29552021, 0.0]),
    "left_wrist_roll_link": ([0.1, 0.00188791, -0.01],    [1.0, 0.0, 0.0, 0.0]),
    "left_wrist_pitch_link":([0.038, 0.0, 0.0],           [1.0, 0.0, 0.0, 0.0]),
    "left_wrist_yaw_link":  ([0.046, 0.0, 0.0],           [1.0, 0.0, 0.0, 0.0]),
    "right_shoulder_pitch_link": ([0.0039563, -0.10021, 0.24778], [0.98531556, -0.13851545, 0.09887526, -0.01379872]),
    "right_shoulder_roll_link":  ([0.0, -0.038, -0.013831], [0.99921502, 0.03961489, 0.0, 0.0]),
    "right_shoulder_yaw_link":   ([0.0, -0.00624, -0.1032], [1.0, 0.0, 0.0, 0.0]),
    "right_elbow_link":     ([0.015783, 0.0, -0.080518],  [0.95533649, 0.0, 0.29552021, 0.0]),
    "right_wrist_roll_link":([0.1, -0.00188791, -0.01],   [1.0, 0.0, 0.0, 0.0]),
    "right_wrist_pitch_link":([0.038, 0.0, 0.0],          [1.0, 0.0, 0.0, 0.0]),
    "right_wrist_yaw_link": ([0.046, 0.0, 0.0],           [1.0, 0.0, 0.0, 0.0]),
}


def fmt_vec(v, nd=6):
    return " ".join(f"{x:+.6f}" if x != 0 else "0" for x in v)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "src/assets/robots/unitree_g1/xmls/scene_g1.xml"

    with open(path) as f:
        lines = f.readlines()

    # 1) 定位机器人本体 <body name="pelvis"> ... </body>
    start = next(i for i, ln in enumerate(lines) if '<body name="pelvis"' in ln)
    end_wb = next(i for i in range(start, len(lines)) if "</worldbody>" in lines[i])
    close = None
    for i in range(start, end_wb):
        if "</body>" in lines[i]:
            close = i
    assert close is not None, "找不到 pelvis body 的闭合标签"
    block = lines[start:close + 1]

    # 2) 生成参考机器人：改名 + 平移 + 全 fixed + 烘焙 FK 位姿
    new_block = []
    for ln in block:
        nl = ln

        m = re.search(r'<body name="([^"]+)"', nl)
        if m:
            nm = m.group(1)
            fk_pos, fk_quat = FK[nm]
            nl = nl.replace(f'name="{nm}"', f'name="ref_{nm}"', 1)
            if nm == "pelvis":
                nl = re.sub(r'pos="[^"]*"', f'pos="{REF_OFFSET}"', nl, count=1)
                if re.search(r'quat="[^"]*"', nl):
                    nl = re.sub(r'quat="[^"]*"', 'quat="1 0 0 0"', nl, count=1)
                else:
                    nl = nl.replace(">", ' quat="1 0 0 0">', 1)
            else:
                nl = re.sub(r'pos="[^"]*"', f'pos="{fmt_vec(fk_pos)}"', nl, count=1)
                if re.search(r'quat="[^"]*"', nl):
                    nl = re.sub(r'quat="[^"]*"', f'quat="{fmt_vec(fk_quat)}"', nl, count=1)
                else:
                    nl = nl.replace(">", f' quat="{fmt_vec(fk_quat)}">', 1)

        m = re.search(r'<joint name="([^"]+)"', nl)
        if m:
            # 参考机器人不加任何 joint：MuJoCo 中 body 无 joint 即刚性固定（fixed）
            continue  # 直接跳过该行

        # 参考机器人仅作视觉对比：全部 geom 设为不参与碰撞
        if "<geom" in nl:
            nl = re.sub(r'contype="\d+"', 'contype="0"', nl)
            nl = re.sub(r'conaffinity="\d+"', 'conaffinity="0"', nl)
            if 'contype="' not in nl:
                nl = nl.replace("/>", ' contype="0" conaffinity="0"/>', 1) if "/>" in nl \
                    else nl.replace(">", ' contype="0" conaffinity="0">', 1)

        m = re.search(r'<site name="([^"]+)"', nl)
        if m:
            nm = m.group(1)
            nl = nl.replace(f'name="{nm}"', f'name="ref_{nm}"', 1)

        new_block.append(nl)

    # 3) 插入参考机器人（在真实机器人之后）
    ref_comment = [
        "\n    <!-- =======================================================\n",
        "         参考机器人（ref_）：静态显示当前策略 default_joint_pos 默认姿态\n",
        "         所有关节 fixed + FK 烘焙位姿，不参与物理；如需改姿态/位置\n",
        "         重新用 scripts/add_ref_robot.py 生成（改 REF_OFFSET / FK）\n",
        "         ======================================================= -->\n",
    ]
    new_lines = lines[:start] + block + ref_comment + new_block + lines[close + 1:]

    # 4) 写回（备份）
    shutil.copy(path, path + ".bak")
    with open(path, "w") as f:
        f.writelines(new_lines)

    print(f"OK: {path}（已备份 {path}.bak）")
    print(f"  参考机器人位置: {REF_OFFSET}，全部关节 fixed，FK 烘焙默认姿态")


if __name__ == "__main__":
    main()
