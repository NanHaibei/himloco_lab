from __future__ import annotations

from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import TYPE_CHECKING
import torch
from collections.abc import Sequence
import isaaclab.utils.math as math_utils

from isaaclab.envs.mdp import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import UniformLevelVelocityCommandCfg, TerrainAdaptiveVelocityCommandCfg



class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.
    """

    cfg: UniformLevelVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.low_vel_env_lin_x_ranges)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        
        high_vel_env_ids = env_ids <= (self.num_envs * self.cfg.rel_high_vel_envs)
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        r_high = torch.empty(len(high_vel_env_ids), device=self.device)
        self.vel_command_b[high_vel_env_ids, 0] = r_high.uniform_(*self.cfg.ranges.lin_vel_x)
        # set y commands of high vel envs to zero
        low_vel_x_min = self.cfg.low_vel_env_lin_x_ranges[0]
        low_vel_x_max = self.cfg.low_vel_env_lin_x_ranges[1]
        in_low_vel_range = (self.vel_command_b[high_vel_env_ids, 0:1] >= low_vel_x_min) & \
                            (self.vel_command_b[high_vel_env_ids, 0:1] <= low_vel_x_max)
        self.vel_command_b[high_vel_env_ids, 1:2] *= in_low_vel_range
        
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > \
                                            self.cfg.min_command_norm).unsqueeze(1)
        
    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

class TerrainAdaptiveVelocityCommand(UniformVelocityCommand):
    """基于地形类型和难度的自适应速度命令
    
    继承自UniformVelocityCommand，保留所有原有功能（heading控制、standing环境、metrics、debug可视化等）。
    额外增加：当机器人处于concentric_moats地形时，根据难度等级动态调整速度命令范围。
    其他地形使用默认速度范围。
    """
    
    cfg: TerrainAdaptiveVelocityCommandCfg
    
    def __init__(self, cfg: TerrainAdaptiveVelocityCommandCfg, env: ManagerBasedEnv):
        # 调用父类初始化（会设置所有必要的buffers和metrics）
        super().__init__(cfg, env)
        
        # 获取terrain信息
        self.terrain = self._env.scene.terrain
        
        # 确定concentric_moats地形在terrain_types中的索引
        if hasattr(self.terrain, 'terrain_origins') and self.terrain.terrain_origins is not None:
            # 获取地形生成器配置
            terrain_gen_cfg = self._env.cfg.scene.terrain.terrain_generator
            if terrain_gen_cfg is not None:
                sub_terrain_names = list(terrain_gen_cfg.sub_terrains.keys())
                self.concentric_moats_idx = next((i for i, name in enumerate(sub_terrain_names) if 'concentric_moats' in name.lower()), None)
            else:
                self.concentric_moats_idx = None
        else:
            self.concentric_moats_idx = None
    
    def __str__(self) -> str:
        msg = "TerrainAdaptiveVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
        msg += f"\tConcentric moats terrain index: {self.concentric_moats_idx}\n"
        msg += f"\tMoats speed threshold: {self.cfg.moats_min_speed_threshold}"
        return msg
    
    def _resample_command(self, env_ids: Sequence[int]):
        """根据地形类型和难度重采样速度命令（向量化版本）
        
        对于concentric_moats地形：
        - 难度低时: 在整个范围内采样（包括低速）
        - 难度高时: 只在高速区域采样，排除中间低速区域
        例如: 范围[-1.2, 1.2]，难度9时只采样[-1.2, -1.0]和[1.0, 1.2]
        
        其他地形使用标准的均匀采样（与UniformVelocityCommand相同）。
        """
        if len(env_ids) == 0:
            return
        
        num_resampled = len(env_ids)
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        # 获取每个环境的地形类型和难度等级
        if hasattr(self.terrain, 'terrain_types') and hasattr(self.terrain, 'terrain_levels'):
            terrain_types = self.terrain.terrain_types[env_ids_tensor]
            terrain_levels = self.terrain.terrain_levels[env_ids_tensor]
            max_level = self.terrain.max_terrain_level
        else:
            terrain_types = None
            terrain_levels = None
            max_level = 1
        
        # 批量判断是否是concentric_moats地形
        is_moats_terrain = torch.zeros(num_resampled, dtype=torch.bool, device=self.device)
        if terrain_types is not None and self.concentric_moats_idx is not None:
            is_moats_terrain = (terrain_types == self.concentric_moats_idx)
        
        # 计算难度比例和排除比例
        difficulty_ratio = torch.zeros(num_resampled, device=self.device)
        if terrain_levels is not None:
            difficulty_ratio = terrain_levels.float() / max(max_level - 1, 1)
        excluded_ratio = difficulty_ratio * self.cfg.moats_min_speed_threshold
        
        # 准备采样
        r = torch.empty(num_resampled, device=self.device)
        
        # 准备速度范围
        vel_ranges = torch.tensor([
            [self.cfg.ranges.lin_vel_x[0], self.cfg.ranges.lin_vel_x[1]],
            [self.cfg.ranges.lin_vel_y[0], self.cfg.ranges.lin_vel_y[1]],
            [self.cfg.ranges.ang_vel_z[0], self.cfg.ranges.ang_vel_z[1]]
        ], device=self.device)  # [3, 2]
        
        # 批量采样速度（对于moats地形使用排除中间区域的采样，其他地形使用均匀采样）
        for vel_idx in range(3):
            vel_min, vel_max = vel_ranges[vel_idx]
            abs_max = max(abs(vel_min), abs(vel_max))
            
            # 计算每个环境的阈值
            threshold = abs_max * excluded_ratio  # [num_resampled]
            
            # 随机选择正向或负向
            direction_mask = torch.rand(num_resampled, device=self.device) < 0.5
            
            # 计算采样范围的下界和上界
            # 对于moats地形：负向[vel_min, -threshold]，正向[threshold, vel_max]
            # 对于其他地形：统一[vel_min, vel_max]
            lower_bound = torch.where(
                is_moats_terrain & direction_mask,
                torch.full((num_resampled,), vel_min, device=self.device),
                torch.where(
                    is_moats_terrain,
                    threshold,
                    torch.full((num_resampled,), vel_min, device=self.device)
                )
            )
            
            upper_bound = torch.where(
                is_moats_terrain & direction_mask,
                -threshold,
                torch.full((num_resampled,), vel_max, device=self.device)
            )
            
            # 批量采样
            sampled_vel = lower_bound + torch.rand(num_resampled, device=self.device) * (upper_bound - lower_bound)
            self.vel_command_b[env_ids_tensor, vel_idx] = sampled_vel
        
        # 采样heading目标（如果启用了heading命令）
        if self.cfg.heading_command:
            self.heading_target[env_ids_tensor] = r.uniform_(*self.cfg.ranges.heading)
            # 更新heading环境标记
            self.is_heading_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        
        # 更新standing环境标记
        self.is_standing_env[env_ids_tensor] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs