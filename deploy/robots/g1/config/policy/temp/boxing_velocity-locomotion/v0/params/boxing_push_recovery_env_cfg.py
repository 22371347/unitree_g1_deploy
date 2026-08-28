import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_BOXING_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# boxing_pose_newversion.csv 使用单位 root quaternion，因此目标重力为标准竖直方向。
DESIRED_GRAVITY = [0.0, 0.0, -1.0]
TARGET_ROOT_HEIGHT = 0.710

# Start from light disturbances and progressively reach a moderate push range.
START_PUSH_VELOCITY_RANGE = {
    "x": (-0.80, 0.80),
    "y": (-0.64, 0.64),
    "z": (0.0, 0.0),
    "roll": (-0.0, 0.0),
    "pitch": (-0.0, 0.0),
    "yaw": (-0.0, 0.0),
}
FINAL_PUSH_VELOCITY_RANGE = {
    "x": (-1.2, 1.2),
    "y": (-1.0, 1.0),
    "z": (0.0, 0.0),
    "roll": (-0.0, 0.0),
    "pitch": (-0.0, 0.0),
    "yaw": (-0.0, 0.0),
}

# 第二阶段在稳定 locomotion 策略上逐步加入抗推训练。
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # decimation=4，保存完整控制周期内的接触力用于落地冲击峰值统计。
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # 覆盖常见地面；过低摩擦会让碎步/打滑主导优化。
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            # 保留躯干质量鲁棒性，但缩小初期分布避免破坏已有步态。
            "mass_distribution_params": (-0.5, 1.5),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-0.2, 0.2),
        },
    )

    # Each environment receives independent interval pushes.  A long enough
    # interval leaves complete locomotion segments between disturbances.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        is_global_time=False,
        params={"velocity_range": START_PUSH_VELOCITY_RANGE.copy()},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # 基线训练暂时不对 command 做加速度/斜率限制，直接跟踪重采样目标。
    base_velocity = mdp.RateLimitedUniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.20,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # 从较小指令开始，课程逐步扩展到实机目标范围。
        max_acceleration=(2.0, 2.0, 5.0),
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-2.0, 2.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-2.0, 2.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.2)

    # -- 躯干稳定与停车
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_world_l2, weight=-1.5)
    # 该项过大会迫使机器人用高频碎步冻结躯干；推力恢复需要允许短暂俯仰。
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_world_l2, weight=-0.25)
    # -- 动作与能耗正则化
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    # 二阶动作差分针对高频反向修正，与一阶 action_rate 互补。
    action_second_order = RewTerm(func=mdp.ActionSecondOrderPenalty, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        # Keep the boxing upper-body posture while allowing normal arm swing.
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    # 新版默认姿态三个腰关节均为 0，只保留统一且温和的默认姿态约束。
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        # A stronger waist constraint reduces torso sway.
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_.*_joint"],
            )
        },
    )

    joint_deviation_legs_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        # Penalize hip-yaw deviation without locking hip-roll for lateral motion.
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_yaw_joint"],
            )
        },
    )

    # -- 搏击姿态
    # 目标重力与新版 boxing 资产的单位 root quaternion 严格一致。
    pelvis_orientation = RewTerm(
        func=mdp.desired_orientation_l2,
        # Strengthen the existing pelvis roll/pitch deviation penalty.
        weight=-8.0,
        params={"desired_gravity": DESIRED_GRAVITY},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-8.0,
        params={"target_height": TARGET_ROOT_HEIGHT},
    )
    # -- 足端安全与基本质量
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        # Penalize clearance error only when the feet are moving horizontally.
        func=mdp.foot_clearance_weighted_by_velocity,
        weight=-0.6,
        params={
            "target_height": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_with_maximum,
    #     weight=0.5,
    #     params={
    #         "command_name": "base_velocity",
    #         "command_threshold": 0.1,
    #         "actual_velocity_threshold": 0.1,
    #         "minimum_air_time": 0.40,
    #         "maximum_air_time": 0.75,
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=".*ankle_roll.*"
    #         ),
    #     },
    # )

    # feet_contact_time = RewTerm(
    #     func=mdp.feet_contact_time_with_maximum,
    #     # Reward longer, bounded support phases and penalize rapid shuffling.
    #     weight=0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=".*ankle_roll.*"
    #         ),
    #         "command_name": "base_velocity",
    #         "command_threshold": 0.1,
    #         "actual_velocity_threshold": 0.1,
    #         "minimum_contact_time": 0.45,
    #         "maximum_contact_time": 0.85,
    #     },
    # )

    feet_lateral_distance = RewTerm(
        func=mdp.feet_lateral_distance_l2,
        weight=-20.0,
        params={
            "minimum_distance": 0.20,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
                preserve_order=True,
            ),
        },
    )
    # 只惩罚超过阈值的落地冲击峰值，不惩罚正常承重接触力。
    feet_impact = RewTerm(
        func=mdp.feet_impact_l2,
        weight=-2.0e-6,
        params={
            "force_threshold": 350.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*ankle_roll.*"
            ),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still,
        # 零指令时回到新版默认关节姿态。
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation_from_desired_gravity,
        params={
            "limit_angle": 1.0,
            "desired_gravity": DESIRED_GRAVITY,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # 当前 terrain generator 只有平面，不启用没有实际意义的地形课程。
    terrain_levels = None
    # 新版姿态从头训练，先学低速步态，再根据追踪表现扩展指令范围。
    lin_vel_cmd_levels = None
    ang_vel_cmd_levels = None
    push_velocity_levels = CurrTerm(
        func=mdp.push_velocity_levels,
        params={
            "event_term_name": "push_robot",
            "start_velocity_range": START_PUSH_VELOCITY_RANGE,
            "final_velocity_range": FINAL_PUSH_VELOCITY_RANGE,
            "curriculum_steps": 80_000,
        },
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        # Play uses the final push range directly instead of replaying the curriculum.
        self.curriculum.push_velocity_levels = None
        self.events.push_robot.params[
            "velocity_range"
        ] = FINAL_PUSH_VELOCITY_RANGE.copy()


        # # 关闭 reset 姿态随机化
        # self.events.reset_base.params["pose_range"] = {
        #     "x": (0.0, 0.0),
        #     "y": (0.0, 0.0),
        #     "z": (0.0, 0.0),
        #     "roll": (0.0, 0.0),
        #     "pitch": (0.0, 0.0),
        #     "yaw": (0.0, 0.0),
        # }
