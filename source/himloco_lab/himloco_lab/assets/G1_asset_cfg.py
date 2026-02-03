
import re
import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

# 设置最大最小命令延迟时间
JOINT_MIN_DELAY_STEP = 1
JOINT_MAX_DELAY_STEP = 2

# G1 29dof的关节顺序
G1_29DOF_JOINT_ORDER = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]

# 抄beyondmimic的写法来设置电机参数
ARMATURE_N7520_14_3 = 0.01017752
ARMATURE_N7520_22_5 = 0.025101925
ARMATURE_N5020_16 = 0.003609725
ARMATURE_W4010_25 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_N7520_14_3 = ARMATURE_N7520_14_3 * NATURAL_FREQ**2
STIFFNESS_N7520_22_5 = ARMATURE_N7520_22_5 * NATURAL_FREQ**2
STIFFNESS_N5020_16 = ARMATURE_N5020_16 * NATURAL_FREQ**2
STIFFNESS_W4010_25 = ARMATURE_W4010_25 * NATURAL_FREQ**2

DAMPING_N7520_14_3 = 2.0 * DAMPING_RATIO * ARMATURE_N7520_14_3 * NATURAL_FREQ
DAMPING_N7520_22_5 = 2.0 * DAMPING_RATIO * ARMATURE_N7520_22_5 * NATURAL_FREQ
DAMPING_N5020_16 = 2.0 * DAMPING_RATIO * ARMATURE_N5020_16 * NATURAL_FREQ
DAMPING_W4010_25 = 2.0 * DAMPING_RATIO * ARMATURE_W4010_25 * NATURAL_FREQ

@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # 全box碰撞，脚掌为box
        usd_path="/home/jyz/project/himloco_lab_ws/robot_description/G1/usd/G1_29dof_full_collision/G1_29dof_full_collision.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
           
            "left_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.16,
            
            "right_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            
            ".*_elbow_joint": 0.87,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_yaw_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_yaw_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_N7520_14_3,
                ".*_hip_roll_joint": STIFFNESS_N7520_22_5,
                ".*_hip_yaw_joint": STIFFNESS_N7520_14_3,
                ".*_knee_joint": STIFFNESS_N7520_22_5,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_N7520_14_3,
                ".*_hip_roll_joint": DAMPING_N7520_22_5,
                ".*_hip_yaw_joint": DAMPING_N7520_14_3,
                ".*_knee_joint": DAMPING_N7520_22_5,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_N7520_14_3,
                ".*_hip_roll_joint": ARMATURE_N7520_22_5,
                ".*_hip_yaw_joint": ARMATURE_N7520_14_3,
                ".*_knee_joint": ARMATURE_N7520_22_5,
            },
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
        ),
        "feet": DelayedPDActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_N5020_16,
            damping=2.0 * DAMPING_N5020_16,
            armature=2.0 * ARMATURE_N5020_16,
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
        ),
        "waist_roll_pitch": DelayedPDActuatorCfg(
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_N5020_16,
            damping=2.0 * DAMPING_N5020_16,
            armature=2.0 * ARMATURE_N5020_16,
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
        ),
        "waist_yaw": DelayedPDActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_N7520_14_3,
            damping=DAMPING_N7520_14_3,
            armature=ARMATURE_N7520_14_3,
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_N5020_16,
                ".*_shoulder_roll_joint": STIFFNESS_N5020_16,
                ".*_shoulder_yaw_joint": STIFFNESS_N5020_16,
                ".*_elbow_joint": STIFFNESS_N5020_16,
                ".*_wrist_roll_joint": STIFFNESS_N5020_16,
                ".*_wrist_pitch_joint": STIFFNESS_W4010_25,
                ".*_wrist_yaw_joint": STIFFNESS_W4010_25,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_N5020_16,
                ".*_shoulder_roll_joint": DAMPING_N5020_16,
                ".*_shoulder_yaw_joint": DAMPING_N5020_16,
                ".*_elbow_joint": DAMPING_N5020_16,
                ".*_wrist_roll_joint": DAMPING_N5020_16,
                ".*_wrist_pitch_joint": DAMPING_W4010_25,
                ".*_wrist_yaw_joint": DAMPING_W4010_25,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_N5020_16,
                ".*_shoulder_roll_joint": ARMATURE_N5020_16,
                ".*_shoulder_yaw_joint": ARMATURE_N5020_16,
                ".*_elbow_joint": ARMATURE_N5020_16,
                ".*_wrist_roll_joint": ARMATURE_N5020_16,
                ".*_wrist_pitch_joint": ARMATURE_W4010_25,
                ".*_wrist_yaw_joint": ARMATURE_W4010_25,
            },
            min_delay=JOINT_MIN_DELAY_STEP,
            max_delay=JOINT_MAX_DELAY_STEP,
        ),
    },
    joint_sdk_names = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]
)

G1_29DOF_ACTION_SCALE = {}
for a in G1_29DOF_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            G1_29DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
