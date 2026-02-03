import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
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

from himloco_lab.assets.G1_asset_cfg import G1_29DOF_CFG, G1_29DOF_JOINT_ORDER, G1_29DOF_ACTION_SCALE
from himloco_lab.tasks.locomotion import mdp
import himloco_lab.terrains as him_terrains

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0), # 每个地形的大小
    border_width=30.0, # border是在地形区域外扩充一圈平地，设置平地的宽度
    border_height = 1.0, # border的高度，负数是向上砌墙，正数改了没有反应
    num_rows=10, # 列数，一共多少级
    num_cols=20, # 行数，一共多少个赛道
    horizontal_scale=0.1, # 地形水平分辨率
    vertical_scale=0.005, # 地形高度分辨率
    slope_threshold=0.75, # tan超过此值的斜坡会变成墙
    use_cache=False, # 不能开，可能导致地形修改不成功
    curriculum=True,
    sub_terrains={
        # 上楼梯
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2, 
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # 下楼梯
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        # 同心沟壑地形 (concentric moats)
        "concentric_moats": him_terrains.MeshConcentricMoatsTerrainCfg(
            proportion=0.2,
            platform_width=3.0,
            moat_width_range=(0.0, 0.3),
            moat_depth_range=(0.0, 1.5),
            num_moats=3,
            platform_ring_width=0.7,
        ),
        # 平地
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, COBBLESTONE_ROAD_CFG
        max_init_terrain_level=5,
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
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mid360_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.35, 0.95], ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],  
    )
    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mid360_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.3, 0.4]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],  
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
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

    """Configuration for events."""
    # startup
    # link的摩擦力、弹性系数随机化
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 256,
        },
    )
    # 关节摩擦力随机化
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.01, 0.2),
            # "armature_distribution_params": (0.01, 0.01),
            "operation": "abs",
        },
    )
    # base link 的质量随机化
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-0.5, 2.0),
            "operation": "add",
        },
    )
    # 随机每一个link的质量
    add_all_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.97, 1.03),
            "operation": "scale",
        },
    )

    # 惯量随机化
    randomize_base_inertias = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    # 质心随机化
    randomize_base_coms = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # 随机推动
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        min_step_count_between_reset= 100,
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.30, 0.30), "y": (-0.30, 0.30), "z": (-0.10, 0.10)}},
    )

    # 随机化kp kd
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # # 复位时会受到力的作用
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    # 这个注释了就不会reset了，不能注
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    # 复位时关节默认角度的随机化
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.85, 1.15),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.TerrainAdaptiveVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.8,
        debug_vis=True,
        # 速度采样范围（所有地形）
        ranges=mdp.TerrainAdaptiveVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2), 
            lin_vel_y=(-0.5, 0.5), 
            ang_vel_z=(-1.0, 1.0), 
            heading=(-math.pi, math.pi)
        ),
        moats_min_speed_threshold=0.8,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_JOINT_ORDER,
        scale=G1_29DOF_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        clip={
            ".*": (-50.0, 50.0),
        }
    )

# observation compute step in lab: noise clip scale
# observation compute step in gym: clip scale noise
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        height_scanner = ObsTerm(func=mdp.height_scan_clip,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-100, 100),
            noise=Unoise(n_min=-0.015, n_max=0.015)
        )

        def __post_init__(self):
            self.enable_corruption = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-100, 100), noise=Unoise(n_min=-0.1, n_max=0.1))
        # base_external_force = ObsTerm(
        #     func=mdp.base_external_force,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
        #     clip=(-100, 100),
        # )
        # height_scanner = ObsTerm(func=mdp.height_scan_clip,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-100, 100),
        # )

        def __post_init__(self):
            self.enable_corruption = True

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    """E1奖励配置"""

    # 速度跟踪奖励
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=5.0, 
        params={"command_name": "base_velocity", "std": 0.5})

    # 机体平衡奖励
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    adaptive_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, weight=-4.0
    )

    # 运动平滑奖励
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01) 
    action_smooth_l2 = RewTerm(func=mdp.action_smooth_l2, weight=-0.01)

    # 安全奖励
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-6)
    joint_vel_soft_limits = RewTerm(func=mdp.joint_vel_soft_limits, weight=-1.0, params={"soft_ratio": 0.9})
    joint_tor_soft_limits = RewTerm(func=mdp.joint_tor_soft_limits, weight=-1.0, params={"soft_ratio": 0.9})
    joint_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.2)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-250.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*shoulder_.*", ".*elbow_.*", ".*wrist_.*", ".*hand", ".*hip_yaw.*", ".*torso_.*"]), "threshold": 1.0},
    )

    base_height_exp = RewTerm(
        func=mdp.base_height_exp, 
        weight=1.0,  # 正权重
        params={
            "target_height": 0.75, 
            "std": 0.05,  # 调整敏感度
            "sensor_cfg": SceneEntityCfg("height_scanner")
        }
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # stand_still_vel = RewTerm(
    #     func=mdp.stand_still_without_cmd_vel_exp, # 走复杂地形的时候不能使用普通standstill
    #     weight=0.8,
    #     params={
    #         "command_name": "base_velocity",
    #         "std": 10.0,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #     },
    # )

    stand_still_pos = RewTerm(
        func=mdp.stand_still_without_cmd_exp, # 减少这一项的权重
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": 2.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    stand_still_pos_arm = RewTerm(
        func=mdp.stand_still_without_cmd_exp, # 只对手臂进行位置的standstill约束
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": 2.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*shoulder_.*", ".*elbow_.*", ".*wrist_.*"]),
        },
    )
    # 关节偏差惩罚-需要大幅度运动的关节少惩罚,反之多惩罚
    joint_deviation_movement_joints = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_pitch.*", ".*knee.*", ".*ankle_pitch.*", ".*shoulder_pitch.*", ".*elbow.*", ".*wrist.*"])}
    )

    joint_deviation_static_joints = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_roll.*", ".*hip_yaw.*", ".*ankle_roll.*", "waist.*", ".*shoulder_roll.*", ".*shoulder_yaw.*", ".*wrist.*"])}
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=7.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=[
                "pelvis.*",
                ".*torso_.*",
                ".*head.*",
                # ".*hip_yaw.*",
                # ".*shoulder.*",
                # ".*elbow.*"
            ]
        ),
            "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


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
        super().__post_init__()
        # general settings
        self.decimation = 4
        self.episode_length_s = 40.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

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
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.curriculum = True
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(1, 1), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0, 0), 
        )
        self.commands.base_velocity.low_vel_env_lin_x_ranges=(1,1)
        
        # Disable randomization events for play mode
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.push_robot = None
