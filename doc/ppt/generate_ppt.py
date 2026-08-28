#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成《组会汇报：基于 MJLAB 的机器人策略部署全流程》PPT 草稿骨架。
在支持 python-pptx 的机器上运行:  python3 generate_ppt.py
产物: 组会汇报_基于MJLAB的策略部署.pptx（可在 Windows/PowerPoint/WPS 继续编辑美化）
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ---------------- 配色 ----------------
DARK = RGBColor(0x1F, 0x3A, 0x5F)      # 深蓝（主色）
ACCENT = RGBColor(0xF5, 0x8A, 0x1F)    # 橙（强调）
LIGHT = RGBColor(0xF4, 0xF6, 0xFA)     # 浅灰蓝（底色）
GRAY = RGBColor(0x55, 0x5E, 0x6B)      # 正文灰
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TABBG = RGBColor(0xE8, 0xEE, 0xF7)     # 表格表头

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

# ---------------- 辅助函数 ----------------
def add_slide():
    return prs.slides.add_slide(BLANK)

def add_rect(slide, x, y, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
    sp.shadow.inherit = False
    return sp

def set_text(tf, text, size=18, bold=False, color=GRAY, align=PP_ALIGN.LEFT,
             font='微软雅黑'):
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font

def add_textbox(slide, x, y, w, h, text, size=18, bold=False, color=GRAY,
                align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    set_text(tf, text, size, bold, color, align)
    return tb

def add_title(slide, num, title):
    """统一页眉：序号 + 标题 + 橙色竖条"""
    add_rect(slide, Inches(0.55), Inches(0.42), Inches(0.12), Inches(0.62), ACCENT)
    add_textbox(slide, Inches(0.82), Inches(0.40), Inches(11.5), Inches(0.8),
                title, size=30, bold=True, color=DARK)
    add_rect(slide, Inches(0.55), Inches(1.30), Inches(12.2), Emu(9525), RGBColor(0xD9, 0xDF, 0xEA))

def add_bullets(slide, items, x=Inches(0.9), y=Inches(1.7), w=Inches(11.5),
                h=Inches(5.3), size=18, gap=8):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        if isinstance(it, tuple):
            text, lvl = it
        else:
            text, lvl = it, 0
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.space_after = Pt(gap)
        prefix = "• " if lvl == 0 else "– "
        run = p.add_run()
        run.text = prefix + text
        run.font.size = Pt(size if lvl == 0 else size - 2)
        run.font.color.rgb = DARK if lvl == 0 else GRAY
        run.font.name = '微软雅黑'
    return tb

def add_table(slide, headers, rows, x=Inches(0.9), y=Inches(1.8), w=Inches(11.5),
              col_w=None, size=14, header_size=14):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    gt = slide.shapes.add_table(n_rows, n_cols, x, y, w, Inches(0.45 * n_rows))
    tbl = gt.table
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(w * cw / total))
    # 表头
    for j, htxt in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.06)
        cell.margin_right = Inches(0.06)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run(); run.text = htxt
        run.font.size = Pt(header_size); run.font.bold = True
        run.font.color.rgb = WHITE; run.font.name = '微软雅黑'
    # 数据
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if i % 2 == 0 else WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            run = p.add_run(); run.text = str(val)
            run.font.size = Pt(size); run.font.color.rgb = GRAY
            run.font.name = '微软雅黑'
    return tbl

def add_code(slide, code, x=Inches(0.9), y=Inches(1.7), w=Inches(11.5), h=None, size=13):
    tb = slide.shapes.add_textbox(x, y, w, h if h else Inches(0.32 * (code.count('\n') + 1)))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = code.split('\n')
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(2)
        run = p.add_run(); run.text = line
        run.font.name = 'Consolas'
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor(0x2B, 0x2B, 0x2B)
    return tb

def add_image_placeholder(slide, label, x=Inches(0.9), y=Inches(1.9), w=Inches(5.6),
                          h=Inches(4.8)):
    box = add_rect(slide, x, y, w, h, RGBColor(0xEA, 0xEF, 0xF7))
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run(); run.text = label
    run.font.size = Pt(16); run.font.color.rgb = GRAY; run.font.name = '微软雅黑'
    return box

def add_footer(slide, text="基于 MJLAB 的机器人策略部署全流程"):
    add_textbox(slide, Inches(0.55), Inches(7.08), Inches(8), Inches(0.35),
                text, size=10, color=RGBColor(0x9A, 0xA3, 0xAF))

# ============================================================
# 第 1 页 · 封面
# ============================================================
s = add_slide()
add_rect(s, 0, 0, prs.slide_width, prs.slide_height, DARK)
add_rect(s, 0, Inches(6.7), prs.slide_width, Inches(0.8), ACCENT)
add_textbox(s, Inches(1.0), Inches(2.2), Inches(11.3), Inches(1.6),
            "基于 MJLAB 的机器人策略部署全流程", size=48, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER)
add_textbox(s, Inches(1.0), Inches(3.9), Inches(11.3), Inches(1.0),
            "从训练到实机 —— sim2sim / sim2real 部署方法教程", size=22,
            color=RGBColor(0xC9, 0xD6, 0xE8), align=PP_ALIGN.CENTER)
add_textbox(s, Inches(1.0), Inches(5.6), Inches(11.3), Inches(0.6),
            "汇报人：XXX　|　日期：XXXX-XX-XX", size=16,
            color=RGBColor(0xC9, 0xD6, 0xE8), align=PP_ALIGN.CENTER)
add_textbox(s, Inches(0.9), Inches(0.5), Inches(6), Inches(0.5),
            "Unitree RL Mjlab", size=16, bold=True, color=RGBColor(0x8F, 0xA8, 0xC8))

# ============================================================
# 第 2 页 · 汇报大纲
# ============================================================
s = add_slide()
add_title(s, "02", "汇报大纲")
items = [
    "框架简介与整体流程（训练 → 仿真验证 → 实机）",
    "环境搭建与仓库拉取",
    "训练一个策略（简述）",
    "接入新策略：5 步法（文件 → 配置 → 注册）",
    "观测体系详解（速度类 / 动作模仿类 / 历史观测）",
    "FSM 状态机逻辑与工程实现",
    "Sim2Sim 仿真验证",
    "Sim2Real 实机部署",
    "注意事项与常见坑",
]
add_bullets(s, items, size=20, gap=12)

# ============================================================
# 第 3 页 · 框架简介与流程总览
# ============================================================
s = add_slide()
add_title(s, "03", "框架简介与整体流程")
add_bullets(s, [
    "MJLAB = MuJoCo 物理引擎 + Isaac Lab 风格高层 API 的轻量化强化学习框架",
    "支持机器人：Go2 / A2 / As2 / G1 / R1 / H1_2 / H2（本汇报以 G1·29DOF 为例）",
    "RL 运动控制基本流程：训练 → 仿真验证 → 仿真到实机",
    "策略 = ONNX 网络 + 部署配置（deploy.yaml），部署侧为纯 C++",
], y=Inches(1.6), h=Inches(2.2))
add_textbox(s, Inches(0.9), Inches(4.0), Inches(11.5), Inches(0.5),
            "整体流程", size=20, bold=True, color=DARK)
add_rect(s, Inches(1.0), Inches(4.6), Inches(3.2), Inches(1.3), LIGHT)
add_rect(s, Inches(4.9), Inches(4.6), Inches(3.2), Inches(1.3), LIGHT)
add_rect(s, Inches(8.8), Inches(4.6), Inches(3.2), Inches(1.3), LIGHT)
for x, t in [(1.15, "① 训练\nMuJoCo 环境"), (5.05, "② 导出策略\npolicy.onnx"),
             (8.95, "③ Sim2Sim→实机\nunitree_mujoco + g1_ctrl")]:
    add_textbox(s, Inches(x), Inches(4.9), Inches(2.9), Inches(0.9),
                t, size=15, bold=True, color=DARK, align=PP_ALIGN.CENTER)
for x in [4.2, 8.0]:
    add_textbox(s, Inches(x), Inches(5.05), Inches(0.7), Inches(0.5),
                "▶", size=20, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
add_image_placeholder(s, "【图2】仿真 vs 实机对比 gif\n(仓库 doc/gif/g1-velocity*.gif)",
                      x=Inches(8.8), y=Inches(1.6), w=Inches(3.7), h=Inches(2.2))
add_footer(s)

# ============================================================
# 第 4 页 · 环境搭建与仓库拉取
# ============================================================
s = add_slide()
add_title(s, "04", "环境搭建与仓库拉取")
add_bullets(s, [
    "系统要求：Ubuntu 22.04，Nvidia 显卡（驱动 ≥ 550）",
    "训练在 PC 上跑；部署侧 deploy/ 为纯 C++，仅需 cmake/make",
], y=Inches(1.55), h=Inches(1.5))
add_code(s, """git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
conda create -n unitree_rl_mjlab python=3.11 && conda activate unitree_rl_mjlab

# 系统依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# Python 依赖（setup.py）
cd unitree_rl_mjlab && pip install -e .""",
         y=Inches(3.0), h=Inches(2.4), size=15)
add_footer(s)

# ============================================================
# 第 5 页 · 训练一个策略（简述）
# ============================================================
s = add_slide()
add_title(s, "05", "训练一个策略（简述）")
add_table(s, ["任务类型", "任务 ID", "训练命令（示例）"],
          [["速度跟踪", "Unitree-G1-Flat",
            "python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096"],
           ["动作模仿", "Unitree-G1-Tracking-No-State-Estimation",
            "python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/xxx.npz"]],
          y=Inches(1.7), col_w=[2, 6, 8], size=13)
add_bullets(s, [
    ("动作文件准备（模仿类）：csv → npz（重采样到 50Hz）", 0),
    ("python scripts/csv_to_npz.py --input-file xxx.csv --output-name xxx.npz --input-fps 30 --output-fps 50 --robot g1", 1),
    ("训练产物：checkpoint model_<iter>.pt + 同步导出 policy.onnx（部署用）", 0),
    ("保存路径：logs/rsl_rl/<robot>_<task>/<date>/", 1),
], y=Inches(4.3), h=Inches(2.6), size=15)
add_image_placeholder(s, "【图3】训练曲线截图\n【图4】g1-mimic.gif 回放", x=Inches(9.6), y=Inches(1.7), w=Inches(2.9), h=Inches(2.4))
add_footer(s)

# ============================================================
# 第 6 页 · 策略部署文件总览
# ============================================================
s = add_slide()
add_title(s, "06", "策略部署文件总览（G1 为例）")
add_code(s, """deploy/robots/g1/
├── config/
│   ├── config.yaml            # FSM 状态注册（复制为 config_<策略>.yaml 使用）
│   └── policy/<策略名>/
│       ├── exported/
│       │   ├── policy.onnx        # 策略网络
│       │   └── policy.onnx.data   # 归一化参数
│       ├── params/
│       │   └── deploy.yaml        # 部署配置（关节/动作/观测）
│       └── motion/                # （仅动作模仿类）
│           └── xxx.npz            # 参考轨迹
├── include/                   # 自定义 State_*.h
├── src/                       # 自定义 State_*.cpp
└── main.cpp                   # 程序入口（include 各 State）""",
         y=Inches(1.7), h=Inches(4.4), size=15)
add_footer(s)

# ============================================================
# 第 7 页 · 接入新策略 5 步法
# ============================================================
s = add_slide()
add_title(s, "07", "接入新策略 5 步法（总览）")
add_table(s, ["步骤", "做什么", "改哪个文件", "需重编译？"],
          [["①", "放置策略文件", "config/policy/<名>/exported/", "否"],
           ["②", "编写部署配置 deploy.yaml", "config/policy/<名>/params/", "否"],
           ["③", "注册 FSM 状态", "config_<名>.yaml 的 FSM._", "否"],
           ["④", "添加状态配置块", "config_<名>.yaml（transitions 等）", "否"],
           ["⑤", "观测逻辑差异大时自定义状态", "include/State_<名>.h + src/State_<名>.cpp + main.cpp", "是"]],
          y=Inches(1.7), col_w=[1.5, 5, 5.5, 2], size=14)
add_rect(s, Inches(0.9), Inches(5.4), Inches(11.5), Inches(1.3), LIGHT)
add_textbox(s, Inches(1.1), Inches(5.55), Inches(11.1), Inches(1.0),
            "三条黄金原则：① joint_ids_map 与训练一致　② 观测书写顺序 = ONNX 输入顺序　③ 历史布局(time/term-major) 与训练一致",
            size=15, bold=True, color=ACCENT)
add_footer(s)

# ============================================================
# 第 8 页 · ①文件放置 ②deploy.yaml（上）
# ============================================================
s = add_slide()
add_title(s, "08", "① 放置策略文件　② deploy.yaml（上：关节与增益）")
add_bullets(s, [
    ("① 放置文件：exported/policy.onnx + policy.onnx.data", 0),
    ("② deploy.yaml 第一部分（关节映射 / 控制周期 / 增益）：", 0),
], y=Inches(1.5), h=Inches(1.0), size=16)
add_code(s, """joint_ids_map: [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
               5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]  # 模型→硬件
step_dt: 0.02                                  # 0.02 = 50Hz
stiffness: [40.2, 99.1, ...]                   # 29 个，模型顺序
damping:   [2.56, 6.31, ...]                   # 29 个，模型顺序
default_joint_pos: [-0.302, ...]               # 策略默认位（模型顺序）""",
         y=Inches(2.6), h=Inches(2.6), size=15)
add_textbox(s, Inches(0.9), Inches(5.4), Inches(11.5), Inches(0.9),
            "注意：stiffness/damping/default_joint_pos 均按【模型顺序】给出，代码会经 joint_ids_map 自动重映射到硬件顺序。",
            size=14, bold=True, color=ACCENT)
add_image_placeholder(s, "【图5】G1 关节编号示意图（腿6+腿6+腰3+臂7+臂7）",
                      x=Inches(9.2), y=Inches(1.5), w=Inches(3.4), h=Inches(1.1))
add_footer(s)

# ============================================================
# 第 9 页 · deploy.yaml（下：动作+观测）
# ============================================================
s = add_slide()
add_title(s, "09", "② deploy.yaml（下：动作 + 观测）")
add_code(s, """actions:
  JointPositionAction:
    clip: null
    joint_names: [.*]
    scale:  [0.548, 0.548, ...]    # target = action * scale + offset
    offset: [-0.302, -0.319, ...]

observations:     # ← 书写顺序 = ONNX 输入拼接顺序（务必与训练一致）
  motion_command:      { params: {command_name: motion}, history_length: 1 }  # 58
  motion_anchor_ori_b: { params: {command_name: motion}, history_length: 1 }  # 6
  base_ang_vel:        { params: {},                   history_length: 1 }    # 3
  joint_pos_rel:       { params: {},                   history_length: 1 }    # 29
  joint_vel_rel:       { params: {},                   history_length: 1 }    # 29
  last_action:         { params: {},                   history_length: 1 }    # 29
  # 154 = 58 + 6 + 3 + 29 + 29 + 29""",
         y=Inches(1.6), h=Inches(4.4), size=14)
add_textbox(s, Inches(0.9), Inches(6.2), Inches(11.5), Inches(0.9),
            "⚠️ scale 长度必须与该项维度一致；观测顺序错位 = 动作异常但无报错",
            size=14, bold=True, color=ACCENT)
add_footer(s)

# ============================================================
# 第 10 页 · 观测体系（上）速度类
# ============================================================
s = add_slide()
add_title(s, "10", "观测体系（上）：速度控制类（RLBase）")
add_bullets(s, ["特点：无参考轨迹，7 项观测，history_length=1"], y=Inches(1.5),
            h=Inches(0.6), size=16)
add_table(s, ["#", "观测项", "维度", "含义", "数据来源"],
          [["1", "base_ang_vel", "3", "机体角速度（机体系）", "IMU"],
           ["2", "projected_gravity", "3", "重力方向投影（倾角）", "IMU"],
           ["3", "velocity_commands", "3", "速度指令 [vx,vy,yaw]", "手柄/键盘"],
           ["4", "gait_phase", "2", "步态相位 sin/cos", "内部生成"],
           ["5", "joint_pos_rel", "29", "关节位置（相对默认位）", "编码器"],
           ["6", "joint_vel_rel", "29", "关节速度", "编码器"],
           ["7", "last_action", "29", "上一步动作", "内部"]],
          y=Inches(2.2), col_w=[1, 4, 1.5, 4.5, 2.5], size=14)
add_bullets(s, [("FSM 类型 type: RLBase → State_RLBase（无需参考轨迹）", 0),
                ("示例目录：config/policy/velocity/v0/", 0)],
            y=Inches(6.2), h=Inches(0.9), size=14)
add_footer(s)

# ============================================================
# 第 11 页 · 观测体系（下）Mimic 154维
# ============================================================
s = add_slide()
add_title(s, "11", "观测体系（下）：动作模仿类（Mimic，154 维）")
add_bullets(s, ["特点：多一项参考轨迹观测，6 项 = 154 维"], y=Inches(1.5),
            h=Inches(0.6), size=16)
add_table(s, ["#", "观测项", "维度", "含义", "备注"],
          [["1", "motion_command", "58", "参考轨迹：目标关节位置(29)+速度(29)", "motion.npz 当前帧"],
           ["2", "motion_anchor_ori_b", "6", "参考锚点相对姿态（旋转矩阵前两列）", "含初始偏航对齐"],
           ["3", "base_ang_vel", "3", "机体角速度", "IMU"],
           ["4", "joint_pos_rel", "29", "当前关节位置（相对默认）", "编码器"],
           ["5", "joint_vel_rel", "29", "当前关节速度", "编码器"],
           ["6", "last_action", "29", "上一步动作", "内部"]],
          y=Inches(2.2), col_w=[1, 4.5, 1.5, 5, 3], size=13)
add_textbox(s, Inches(0.9), Inches(6.15), Inches(11.5), Inches(0.7),
            "154 = 58 + 6 + 3 + 29 + 29 + 29　|　训练全集另含 motion_anchor_pos_b(3)+base_lin_vel(3)=160，无状态估计部署版为 154",
            size=14, bold=True, color=ACCENT)
add_footer(s)

# ============================================================
# 第 12 页 · 时序/历史观测
# ============================================================
s = add_slide()
add_title(s, "12", "带时序/历史状态的观测")
add_bullets(s, [
    ("A. 多帧历史观测（history_length > 1）：如 AMP 速度策略 6 项 × 4 帧 = 384 维", 0),
    ("布局由 use_gym_history 决定（必须与训练 history_ordering 一致）：", 0),
    ("use_gym_history: true → time-major（mjlab）：[帧0: 全部项, 帧1: 全部项, ...]", 1),
    ("use_gym_history: false → term-major（IsaacLab）：[项0: 全部帧, 项1: 全部帧, ...]", 1),
    ("⚠️ 布局错误 → 策略收到错乱观测却照常推理（无报错）→ 抽搐/不响应", 0),
    ("B. 显式时间条件 time_step（1 维）：部分世界模型类策略需要，env->step() 自动填充", 0),
], y=Inches(1.6), h=Inches(3.4), size=16)
add_code(s, """observations:
  base_ang_vel:
    history_length: 4
  ...
  use_gym_history: true   # 与训练 history_ordering 一致""",
         y=Inches(5.1), h=Inches(1.5), size=14)
add_footer(s)

# ============================================================
# 第 13 页 · FSM 状态机概念
# ============================================================
s = add_slide()
add_title(s, "13", "FSM 状态机概念")
add_table(s, ["状态", "作用", "备注"],
          [["Passive", "阻尼/零力矩模式", "上电默认，L2+B 急停回到此"],
           ["FixStand", "力矩站立", "手柄调整姿态触地"],
           ["Velocity", "速度运控（RLBase 策略）", "日常行走"],
           ["<策略状态>", "你部署的策略", "如 AMP / Backflip / Mimic"]],
          y=Inches(1.6), col_w=[2.5, 6, 5], size=14)
add_textbox(s, Inches(0.9), Inches(5.2), Inches(11.5), Inches(0.5),
            "转移条件 = 手柄按键 DSL：RT + A.on_pressed、LT + B.on_pressed …",
            size=15, bold=True, color=DARK)
add_code(s, """Passive -> FixStand: L2+Up     FixStand -> Velocity: R2+A
Velocity -> <策略状态>: RB+A   任意状态 -> Passive: L2+B (急停)""",
         y=Inches(5.7), h=Inches(0.9), size=14)
add_footer(s)

# ============================================================
# 第 14 页 · FSM 工程实现
# ============================================================
s = add_slide()
add_title(s, "14", "FSM 工程实现（自定义状态）")
add_bullets(s, [
    ("继承 FSMState，重写 enter() / run() / exit()，末尾 REGISTER_FSM 注册", 0),
    ("构造：读配置 → 创建 env（ManagerBasedRLEnv + ONNX）→ 注册自动转移", 0),
    ("enter()：设增益(kp/kd) → env->reset() → 启动策略线程（step_dt 节拍）", 0),
    ("run()：每 1ms 控制周期把策略输出写入电机指令", 0),
], y=Inches(1.5), h=Inches(2.2), size=15)
add_code(s, """class State_MyPolicy : public FSMState {
  State_MyPolicy(int m, std::string s): FSMState(m, s) { /* 建env+onnx */ }
  void enter() { /* 设增益、reset、启动线程 */ }
  void run()   { /* 动作 -> motor_cmd */ }
  void exit()  { /* 停线程 */ }
};
REGISTER_FSM(State_MyPolicy)""",
         y=Inches(3.6), h=Inches(2.3), size=13)
add_textbox(s, Inches(0.9), Inches(6.2), Inches(11.5), Inches(0.6),
            "env->step() = 按 deploy.yaml 算观测 → ONNX 推理 → action 反算关节目标",
            size=14, bold=True, color=ACCENT)
add_footer(s)

# ============================================================
# 第 15 页 · 状态注册与键位配置
# ============================================================
s = add_slide()
add_title(s, "15", "状态注册与键位配置（config 实战）")
add_bullets(s, [("复制 config.yaml → config_<策略>.yaml（不直接改默认文件）", 0),
                ("在 FSM._ 注册新状态（唯一 id + type）：", 0)], y=Inches(1.5),
            h=Inches(1.0), size=15)
add_code(s, """FSM:
  _:
    Passive:  { id: 1 }
    FixStand: { id: 2 }
    Velocity: { id: 3, type: RLBase }
    MyPolicy: { id: 5, type: RLBase }   # RLBase / Mimic / 自定义State名

  MyPolicy:
    transitions:
      Passive: LT + B.on_pressed
      FixStand: LT + up.on_pressed
      Velocity: RT + A.on_pressed
    policy_dir: config/policy/mypolicy
    # Mimic 类还需: motion_file / time_start / time_end""",
         y=Inches(2.5), h=Inches(3.4), size=14)
add_code(s, """./g1_ctrl -n lo -c config_mypolicy.yaml   # 仿真
./g1_ctrl -c config_mypolicy.yaml          # 实机""",
         y=Inches(6.0), h=Inches(0.8), size=13)
add_footer(s)

# ============================================================
# 第 16 页 · 需修改的文件清单
# ============================================================
s = add_slide()
add_title(s, "16", "需要修改的代码文件清单（汇总）")
add_table(s, ["步骤", "文件", "改什么", "重编译？"],
          [["1", "config/policy/<名>/exported/", "放 onnx + data", "否"],
           ["2", "config/policy/<名>/params/deploy.yaml", "新建：关节/动作/观测", "否"],
           ["3", "config/config_<名>.yaml", "复制并注册 FSM 状态", "否"],
           ["4", "config/config_<名>.yaml", "状态配置块 + transitions", "否"],
           ["5", "include/State_<名>.h + src/State_<名>.cpp", "自定义状态类 + REGISTER_FSM", "是"],
           ["5", "main.cpp", "#include 新状态头文件", "是"]],
          y=Inches(1.7), col_w=[1.2, 6, 5.5, 2], size=14)
add_rect(s, Inches(0.9), Inches(5.7), Inches(11.5), Inches(1.2), LIGHT)
add_textbox(s, Inches(1.1), Inches(5.9), Inches(11.1), Inches(0.9),
            "结论：90% 的情况只改 yaml，无需动 C++；只有观测逻辑差异大时才自定义状态",
            size=16, bold=True, color=ACCENT)
add_footer(s)

# ============================================================
# 第 17 页 · Sim2Sim 仿真验证
# ============================================================
s = add_slide()
add_title(s, "17", "Sim2Sim 仿真验证（上机前必做）")
add_bullets(s, [
    ("两个程序：仿真器 unitree_mujoco + 控制程序 g1_ctrl（DDS loopback 通信）", 0),
    ("编译（两个工程）：", 0),
], y=Inches(1.5), h=Inches(1.1), size=15)
add_code(s, """# 控制程序
cd deploy/robots/g1 && rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)
# 仿真器
cd simulate && rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)""",
         y=Inches(2.6), h=Inches(1.3), size=13)
add_bullets(s, [
    ("运行：终端1 → ./unitree_mujoco；终端2 → ./g1_ctrl -n lo（-n lo=本机回环）", 0),
    ("状态切换：键盘 a=Passive b=FixStand c=Velocity d=自定义动作；手柄 L2+Up→FixStand、R2+A→Velocity、RB+A→自定义策略", 0),
], y=Inches(4.0), h=Inches(1.6), size=15)
add_image_placeholder(s, "【图7】Sim2Sim 仿真运行截图", x=Inches(0.9), y=Inches(5.6),
                      w=Inches(5.6), h=Inches(1.3))
add_footer(s)

# ============================================================
# 第 18 页 · Sim2Real 实机部署
# ============================================================
s = add_slide()
add_title(s, "18", "Sim2Real 实机部署（G1 为例）")
add_code(s, """# ① 前置依赖: cyclonedds + unitree_sdk2
# ② 开机: 悬挂上电 → zero-torque → 手柄 L2+R2 进 debug mode
# ③ 网络: 网线直连 PC(192.168.123.222) ↔ 机器人(192.168.123.164)
# ④ 传输: scp -r deploy unitree@192.168.123.164:~/wu/
# ⑤ 机器人端:
cd ~/wu/deploy/robots/g1 && rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc) && tmux new -s 1 && ./g1_ctrl   # 实机无需 -n""",
         y=Inches(1.6), h=Inches(3.4), size=14)
add_bullets(s, [
    ("手柄：L2+R2 / L2+A 进调试模式；L2+B 任意状态急停回 Passive", 0),
    ("PC 直连机器人时用对应网卡名：./g1_ctrl -n enp5s0", 0),
], y=Inches(5.2), h=Inches(1.6), size=14)
add_image_placeholder(s, "【图8/9/10】实机照片、网络示意、实机动作视频",
                      x=Inches(9.2), y=Inches(1.6), w=Inches(3.4), h=Inches(2.6))
add_footer(s)

# ============================================================
# 第 19 页 · 常见坑与注意事项
# ============================================================
s = add_slide()
add_title(s, "19", "常见坑与注意事项（实战经验）")
add_table(s, ["#", "坑", "现象", "解决办法"],
          [["1", "关节序用错(qpos/IsaacLab)", "肢体错乱", "对照 csv knee 位置 / stiffness 分组"],
           ["2", "观测书写顺序不一致", "动作异常无报错", "严格按训练 actor_terms 顺序"],
           ["3", "历史布局错误", "抽搐/不响应", "use_gym_history 与训练一致"],
           ["4", "scale/offset 不一致", "幅度过大/过小", "从训练导出包复制"],
           ["5", "motion.npz 的 vel 含噪声", "动作发散/抽搐", "差分速度做滑动平均(窗口5)"],
           ["6", "腰部索引默认值不对", "姿态观测错", "ONNX metadata 或 config 显式指定"]],
          y=Inches(1.6), col_w=[1, 4.5, 3.5, 5.5], size=13)
add_footer(s)

# ============================================================
# 第 20 页 · 参考资料 / Q&A
# ============================================================
s = add_slide()
add_title(s, "20", "参考资料 / Q&A")
add_bullets(s, [
    "仓库：github.com/unitreerobotics/unitree_rl_mjlab",
    "文档：README_zh.md、doc/setup_zh.md",
    "上游：mjlab / IsaacLab / rsl_rl / MuJoCo / BeyondMimic / whole_body_tracking",
    "致谢：所有开源项目贡献者",
], y=Inches(1.8), h=Inches(2.5), size=18)
add_rect(s, Inches(0.9), Inches(5.2), Inches(11.5), Inches(1.2), LIGHT)
add_textbox(s, Inches(0.9), Inches(5.5), Inches(11.5), Inches(0.8),
            "谢谢聆听 · Q&A", size=28, bold=True, color=DARK, align=PP_ALIGN.CENTER)
add_footer(s)

# ---------------- 保存 ----------------
OUT = "组会汇报_基于MJLAB的策略部署.pptx"
prs.save(OUT)
print(f"已生成: {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} 页)")
