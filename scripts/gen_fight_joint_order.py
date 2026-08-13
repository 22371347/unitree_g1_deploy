#!/usr/bin/env python3
"""
为 fight 策略生成不同关节顺序对应的部署数组，用于逐个测试确定模型的真实关节顺序。

用法:
    python scripts/gen_fight_joint_order.py            # 默认打印三种顺序
    python scripts/gen_fight_joint_order.py isaaclab   # 只打印 isaaclab

三种候选顺序（模型索引 i -> 硬件 qpos 索引）:
    beyondmimic : BeyondMimic/whole_body_tracking 序（髋部优先），当前配置
    isaaclab    : Isaac Lab 标准 G1 序（肩/臂优先），unitree_rl_lab 训练最可能用
    qpos        : 与 velocity/mjlab 一致（腿->腰->臂）

测试方法:
    1. 运行本脚本得到目标顺序的 joint_ids_map + stiffness/damping/default_joint_pos/offset
    2. 把数组替换进 deploy/robots/g1/config/policy/fight/params/deploy.yaml
    3. 重启 g1_ctrl（deploy.yaml 运行时加载，无需重新编译）
    4. 在仿真中进入 Fight 状态、指令为 0，观察机器人是否保持"拳击站姿"
       （髋 -1.0、膝 0.9、肩 -0.92、肘 -0.23、腕偏转等），姿态正确则该顺序正确
"""
import sys
import yaml
from pathlib import Path

# 仓库根目录（基于脚本位置解析，与运行目录无关）
REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY = REPO_ROOT / "deploy/robots/g1/config/policy/fight/params/deploy.yaml"

ORDERS = {
    "beyondmimic": [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
        17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ],
    "isaaclab": [
        15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 2, 8, 12, 0, 1, 3, 6, 7, 9,
        20, 21, 27, 28, 13, 14, 4, 5, 10, 11,
    ],
    "qpos": list(range(29)),
}


def reorder(arr_bm, bm, target):
    """把 BeyondMimic 序的数组 arr_bm 重排为目标顺序 target。"""
    inv_bm = {}
    for i, j in enumerate(bm):
        inv_bm[j] = i
    qpos_val = [arr_bm[inv_bm[j]] for j in range(len(bm))]
    return [qpos_val[target[i]] for i in range(len(target))]


def fmt(arr, per_line=10):
    parts = []
    for i in range(0, len(arr), per_line):
        parts.append(", ".join(str(x) for x in arr[i : i + per_line]))
    return "[" + ", ".join(parts) + "]"


def main():
    d = yaml.safe_load(DEPLOY.read_text())
    bm = d["joint_ids_map"]
    arr_bm = {
        "stiffness": d["stiffness"],
        "damping": d["damping"],
        "default_joint_pos": d["default_joint_pos"],
        "offset": d["actions"]["JointPositionAction"]["offset"],
    }

    only = sys.argv[1] if len(sys.argv) > 1 else None
    orders = [only] if only else list(ORDERS)
    for order in orders:
        if order not in ORDERS:
            print(f"未知顺序 {order}，可用: {list(ORDERS)}")
            sys.exit(1)
        target = ORDERS[order]
        print(f"# ============ 顺序: {order} ============")
        print(f"joint_ids_map: {fmt(target)}")
        for key in arr_bm:
            print(f"{key}: {fmt(reorder(arr_bm[key], bm, target))}")
        print()


if __name__ == "__main__":
    main()
