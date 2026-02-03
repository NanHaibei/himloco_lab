import torch
import isaaclab.utils.math as math_utils
from typing import TYPE_CHECKING, Literal
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.events import _randomize_prop_by_op


from isaaclab.envs import ManagerBasedEnv


def apply_periodic_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    period_step: int,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply periodic external forces and torques.
    
    This function applies random forces and torques sampled from given ranges, but only
    when the call count reaches a multiple of period_step. At other times, zero forces
    and torques are applied. The call count is tracked using an attribute on the environment.
    
    Args:
        env: The RL environment
        env_ids: IDs of environments to apply forces to. If None, applies to all environments.
        period_step: Period in simulation steps for applying random forces/torques
        force_range: (min, max) range for force magnitude
        torque_range: (min, max) range for torque magnitude
        asset_cfg: Configuration for the asset to apply forces to
    """
    # Initialize call count on first call
    if not hasattr(env, "external_force_call_count") or env.external_force_call_count is None:
        env.external_force_call_count = 0
    
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    
    # resolve number of bodies
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )
    
    # Create zero forces and torques
    size = (len(env_ids), num_bodies, 3)
    zero_forces = torch.zeros(size, device=asset.device, dtype=torch.float32)
    zero_torques = torch.zeros(size, device=asset.device, dtype=torch.float32)
    
    # Only apply random forces/torques when call count is multiple of period_step
    if env.external_force_call_count % period_step == 0:
        # sample random forces and torques
        forces = math_utils.sample_uniform(*force_range, size, asset.device)
        torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    else:
        # apply zero forces and torques
        forces = zero_forces
        torques = zero_torques
    
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(
        forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
    )
    
    # increment call count
    env.external_force_call_count += 1

def randomize_rigid_body_inertia(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)