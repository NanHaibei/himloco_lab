# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom terrain generation functions for HIMLOCO.
"""

from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import him_terrains_cfg

import trimesh
if TYPE_CHECKING:
    from . import him_terrains_cfg


@height_field_to_mesh
def hf_pyramid_slope_with_noise_terrain(
    difficulty: float, 
    cfg: him_terrains_cfg.HfPyramidSlopeWithNoiseCfg
) -> np.ndarray:
    """
    Generate pyramid sloped terrain with random uniform noise overlay.
    
    Args:
        difficulty: The difficulty of the terrain (0 to 1).
        cfg: The configuration for the terrain.
    """
    # Import the pyramid slope terrain function
    from isaaclab.terrains.height_field import hf_terrains
    
    # Use the unwrapped function to get raw height field without mesh conversion
    pyramid_func = hf_terrains.pyramid_sloped_terrain.__wrapped__
    height_field = pyramid_func(difficulty, cfg)  # Returns int16 array in discrete units
    
    # Calculate noise parameters based on difficulty
    amplitude = cfg.noise_amplitude_range[0] + difficulty * (
        cfg.noise_amplitude_range[1] - cfg.noise_amplitude_range[0]
    )
    
    # Generate downsampled random noise
    downsampled_scale = cfg.downsampled_scale
    noise_step = cfg.noise_step
    
    # Calculate downsampled dimensions
    width = height_field.shape[0]
    length = height_field.shape[1]
    horizontal_scale = cfg.horizontal_scale
    vertical_scale = cfg.vertical_scale
    
    width_downsampled = int(width * horizontal_scale / downsampled_scale)
    length_downsampled = int(length * horizontal_scale / downsampled_scale)
    
    # CRITICAL: Convert noise heights to discrete units (divide by vertical_scale)
    # This matches how isaacgym terrain_utils works
    height_min = int(-amplitude / vertical_scale)
    height_max = int(amplitude / vertical_scale)
    height_step = int(noise_step / vertical_scale)
    
    # Generate random heights in discrete units
    heights_range = np.arange(height_min, height_max + height_step, height_step)
    height_field_downsampled = np.random.choice(heights_range, (width_downsampled, length_downsampled))
    
    # Interpolate to full resolution using spline interpolation
    x = np.linspace(0, width * horizontal_scale, width_downsampled)
    y = np.linspace(0, length * horizontal_scale, length_downsampled)
    f = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    
    x_upsampled = np.linspace(0, width * horizontal_scale, width)
    y_upsampled = np.linspace(0, length * horizontal_scale, length)
    noise_field = f(x_upsampled, y_upsampled)
    
    # Round to nearest integer to maintain int16 discrete representation
    height_field = height_field + np.rint(noise_field).astype(np.int16)
    
    return height_field


@height_field_to_mesh
def hf_discrete_obstacles_terrain(
    difficulty: float,
    cfg: him_terrains_cfg.HfDiscreteObstaclesTerrainCfg
) -> np.ndarray:
    """
    Generate a terrain with discrete rectangular obstacles.
    
    This matches HIMLOCO_GO2 implementation with random rectangular obstacles
    of varying heights scattered across the terrain.
    
    From HIMLOCO_GO2 legged_gym (using isaacgym terrain_utils):
    ```python
    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 
                              rectangle_min_size, rectangle_max_size, 
                              num_rectangles, platform_size=3.)
    ```
    
    Args:
        difficulty: The difficulty of the terrain (0 to 1).
        cfg: The configuration for the terrain.
    
    Returns:
        The height field as a 2D numpy array with discretized heights (int16).
    """
    # Calculate obstacle height based on difficulty
    max_height = cfg.max_height_range[0] + difficulty * (
        cfg.max_height_range[1] - cfg.max_height_range[0]
    )
    
    # Switch parameters to discrete units
    max_height_discrete = int(max_height / cfg.vertical_scale)
    min_size = int(cfg.obstacle_size_range[0] / cfg.horizontal_scale)
    max_size = int(cfg.obstacle_size_range[1] / cfg.horizontal_scale)
    platform_size = int(cfg.platform_width / cfg.horizontal_scale)
    
    # Calculate terrain dimensions
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    # Initialize height field
    height_field = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    
    # Define height range for obstacles (matching HIMLOCO)
    # [-max_height, -max_height // 2, max_height // 2, max_height]
    height_range = [
        -max_height_discrete,
        -max_height_discrete // 2,
        max_height_discrete // 2,
        max_height_discrete
    ]
    
    # Define size ranges with step of 4 (matching HIMLOCO)
    width_range = list(range(min_size, max_size, 4))
    length_range = list(range(min_size, max_size, 4))
    
    # Generate random rectangular obstacles
    for _ in range(cfg.num_obstacles):
        if len(width_range) == 0 or len(length_range) == 0:
            break
            
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        
        # Ensure we don't go out of bounds
        if width_pixels - width <= 0 or length_pixels - length <= 0:
            continue
            
        start_i = np.random.choice(range(0, width_pixels - width, 4))
        start_j = np.random.choice(range(0, length_pixels - length, 4))
        
        # Set obstacle height
        height_field[start_i:start_i + width, start_j:start_j + length] = np.random.choice(height_range)
    
    # Create flat platform at center (for robot spawn)
    x1 = (width_pixels - platform_size) // 2
    x2 = (width_pixels + platform_size) // 2
    y1 = (length_pixels - platform_size) // 2
    y2 = (length_pixels + platform_size) // 2
    height_field[x1:x2, y1:y2] = 0
    
    return height_field

def concentric_moats_terrain(
    difficulty: float, cfg: him_terrains_cfg.MeshConcentricMoatsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with concentric moats around a central platform.
    
    The terrain structure from center outward:
    1. Central square platform
    2. Moat 1
    3. Platform ring
    4. Moat 2
    5. Platform ring
    6. ... (repeats based on num_moats)
    
    Uses boolean difference operations to create moats from a base platform.
    
    .. note::
        The moat width, depth, and number of moats are configurable.
    
    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.
    
    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration based on difficulty
    moat_depth = cfg.moat_depth_range[0] + difficulty * (cfg.moat_depth_range[1] - cfg.moat_depth_range[0])
    moat_width = cfg.moat_width_range[0] + difficulty * (cfg.moat_width_range[1] - cfg.moat_width_range[0])
    
    # compute the position of the terrain center
    terrain_center = [cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0]
    
    # Calculate available space for moats and platforms
    max_radius = min(cfg.size[0], cfg.size[1]) / 2.0
    
    # Calculate the required radius starting from center platform
    # Center platform is a square with side length cfg.platform_width
    # Use half the side length as the "radius" for box calculations
    center_platform_half = cfg.platform_width / 2.0
    
    # Required radius = center + moats + platform rings between moats
    # Structure from center: platform -> moat1 -> ring1 -> moat2 -> ring2 -> ...
    required_radius = center_platform_half + cfg.num_moats * moat_width + (cfg.num_moats - 1) * cfg.platform_ring_width
    
    # Make sure we don't exceed the terrain size
    if required_radius > max_radius:
        raise ValueError(
            f"Terrain configuration exceeds available space. "
            f"Required radius: {required_radius:.2f}m, "
            f"Available radius: {max_radius:.2f}m. "
            f"Please reduce moat_width, num_moats, or platform_width."
        )
    
    # Create the base platform (entire terrain)
    # Make it thicker than moat_depth so there's material below the moats
    # Top surface at z=0, bottom deeper than the moats will dig
    platform_base_thickness = 0.5  # Extra thickness below moats
    platform_total_height = moat_depth + platform_base_thickness
    result_platform = trimesh.creation.box(
        (cfg.size[0], cfg.size[1], platform_total_height),
        trimesh.transformations.translation_matrix([terrain_center[0], terrain_center[1], -platform_total_height / 2.0])
    )
    
    # Dig moats from center outward to maintain fixed center platform size
    # Structure from center: center_platform -> moat1 -> ring1 -> moat2 -> ring2 -> ...
    current_inner_radius = center_platform_half  # Start from the edge of center platform
    
    for i in range(cfg.num_moats):
        # Calculate the i-th moat from center (i=0 is closest to center)
        moat_inner_radius = current_inner_radius
        moat_outer_radius = current_inner_radius + moat_width
        
        # Create the moat ring by subtracting inner box from outer box
        # The moat depth is from z=0 (top surface) down to z=-moat_depth
        # We add small epsilon to ensure clean boolean operations
        outer_box = trimesh.creation.box(
            (2 * moat_outer_radius, 2 * moat_outer_radius, moat_depth + 0.01),
            trimesh.transformations.translation_matrix([terrain_center[0], terrain_center[1], -moat_depth / 2.0])
        )
        
        inner_box = trimesh.creation.box(
            (2 * moat_inner_radius, 2 * moat_inner_radius, moat_depth + 0.02),
            trimesh.transformations.translation_matrix([terrain_center[0], terrain_center[1], -moat_depth / 2.0])
        )
        
        # Create moat ring by subtracting inner from outer (this is the material to remove)
        moat_ring = outer_box.difference(inner_box)
        
        # Dig the moat by subtracting the ring from the platform
        result_platform = result_platform.difference(moat_ring)
        
        # Move to next layer: add moat width + platform ring width to get to next moat's inner radius
        current_inner_radius = moat_outer_radius + cfg.platform_ring_width
    
    # Origin of the terrain (spawn point on the central platform at height 0)
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])
    
    return [result_platform], origin
