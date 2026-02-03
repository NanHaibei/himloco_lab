# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration classes for custom HIMLOCO terrains.
"""

from dataclasses import MISSING

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.utils import configclass
from . import him_terrains  
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

@configclass
class HfPyramidSlopeWithNoiseCfg(HfTerrainBaseCfg):
    """Configuration for pyramid sloped terrain with random noise overlay.
    """

    function = him_terrains.hf_pyramid_slope_with_noise_terrain
    """Name of the function to generate the terrain."""

    slope_range: tuple[float, float] = MISSING
    """The slope of the terrain (in radians)."""
    
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    
    inverted: bool = False
    """Whether the pyramid is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the slopes are upwards.
    """
    
    noise_amplitude_range: tuple[float, float] = MISSING
    """Range of noise amplitude as (min, max). 
    """
    
    noise_step: float = MISSING
    """Step size for noise height sampling in meters.
    """
    
    downsampled_scale: float = MISSING
    """Scale for downsampled noise generation in meters.
    Noise is generated at lower resolution then interpolated up.
    """


@configclass
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for discrete obstacles terrain.
    
    Generates random rectangular obstacles with varying heights scattered
    across the terrain. Matches HIMLOCO_GO2 implementation.
    """
    
    function = him_terrains.hf_discrete_obstacles_terrain
    """Name of the function to generate the terrain."""
    
    max_height_range: tuple[float, float] = MISSING
    """Range of maximum obstacle height as (min, max) in meters.
    Actual max_height = min + difficulty * (max - min).
    Obstacles will have heights in range: [-max, -max/2, max/2, max].
    """
    
    obstacle_size_range: tuple[float, float] = MISSING
    """Range of obstacle size as (min_size, max_size) in meters.
    Each obstacle is a rectangle with width and length sampled from this range.
    """
    
    num_obstacles: int = MISSING
    """Number of randomly generated rectangular obstacles."""
    
    platform_width: float = 1.0
    """The width of the square flat platform at the center of the terrain in meters.
    This is the spawn area for the robot. Defaults to 1.0.
    """

@configclass
class MeshConcentricMoatsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with concentric moats around a central platform.
    
    This terrain creates a challenging environment with:
    - A central square platform
    - Alternating moats and platform rings
    
    The moat width, depth, and number of moats are configurable.
    """

    function = him_terrains.concentric_moats_terrain

    platform_width: float = 2.0
    """The width of the central square platform (in m). Defaults to 2.0."""

    moat_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of each moat (in m).
    
    The actual width is interpolated based on the difficulty parameter.
    """

    moat_depth_range: tuple[float, float] = MISSING
    """The minimum and maximum depth of the moats (in m).
    
    The actual depth is interpolated based on the difficulty parameter.
    """

    num_moats: int = 1
    """The number of moats around the central platform. Defaults to 1."""

    platform_ring_width: float = MISSING
    """The width of each platform ring between moats (in m)."""
