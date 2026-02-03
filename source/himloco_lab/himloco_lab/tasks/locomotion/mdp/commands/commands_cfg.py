from dataclasses import MISSING

from isaaclab.utils import configclass

from .commands import UniformLevelVelocityCommand, TerrainAdaptiveVelocityCommand

from isaaclab.envs.mdp import UniformVelocityCommandCfg

@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    
    class_type: type = UniformLevelVelocityCommand
    
    curriculums_limit_ranges: tuple[float, float] = MISSING
    
    low_vel_env_lin_x_ranges: tuple[float, float] = MISSING
    
    rel_high_vel_envs: float = MISSING
    
    min_command_norm: float = MISSING

@configclass
class TerrainAdaptiveVelocityCommandCfg(UniformVelocityCommandCfg):
    """基于地形的自适应速度命令配置
    
    继承自UniformVelocityCommandCfg，保留所有原有功能（heading控制、standing环境、可视化等）。
    额外增加：对于concentric_moats地形，难度越高，采样的速度越接近极限值，排除低速区域。
    例如: 速度范围[-1.2, 1.2]，难度9时排除中间[-1.0, 1.0]，只采样[-1.2, -1.0]和[1.0, 1.2]
    """
    
    class_type: type = TerrainAdaptiveVelocityCommand
    
    moats_min_speed_threshold: float = 0.83
    """concentric_moats地形在最高难度时排除的中间低速区域比例
    例如: 0.83表示在最高难度时排除±83%以内的速度，只采样剩余的17%高速区域
    设置为0.0表示不排除任何区域（难度对采样范围无影响）
    """