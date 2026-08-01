# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact-rich Stage-1 mimic expert for whole-body yoga support.

This is an opt-in derivative of :mod:`mlp_2x_no_contact_rew`.  It deliberately
keeps that experiment's 6x2048 actor, 4x2048 critic, optimizers, terminations,
tracking rewards, and zero-weight contact-match diagnostic unchanged.  Only the
contact observation pathway changes:

* the legacy binary flags are removed from ``max_coords_obs``;
* ``contact_obs_v1`` supplies the current aggregate per-body force and temporal
  contact state; and
* ``contact_proximity_obs`` supplies body-origin-to-nearest-sampled-surface
  vectors for the same bodies in the same order.

All 24 SMPL bodies are sensed because hands, forearms, knees, torso, and head
can support yoga poses.  The contact-match diagnostic remains feet-only, so
expanding observation coverage cannot silently expand reward semantics.
For ``smpl_yogi`` this makes ``contact_obs_v1`` 412 values and proximity 72
values; the complete concatenated policy input is 1487 values (versus 1027 in
``mlp_2x_no_contact_rew``).  The changed input shape requires a new checkpoint.

Isaac Lab compatibility
-----------------------
Isaac Lab's ``ContactSensorData.net_forces_w`` is the sum of *normal* contact
force vectors on a sensor body; its documentation explicitly says tangential
forces are not included.  Therefore the raw heading-frame vector, magnitude,
active state, rate, duration, and load-distribution channels are useful with
Isaac Lab, while ``horizontal_force_proxy`` and
``ground_friction_utilization_proxy`` must not be interpreted as measured
Coulomb friction.  On non-horizontal contacts they mainly reflect the
orientation of the summed normal forces.

The current ProtoMotions Isaac Lab bridge reads the most recent sensor sample
at the end of each four-substep control interval.  Although Isaac Lab retains a
four-sample sensor history, ``contact_obs_v1`` does not aggregate it, so a short
impact that ends before the final substep can be missed.

The proximity key is a vector to the nearest sampled terrain/object point.  It
is neither signed collider distance nor a contact point, penetration depth, or
self-contact query.  Its accuracy and cost depend on terrain and object
point-cloud sampling resolution.

Train::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_contact_rich.py \
      --experiment-name smpl_yogi_balance_contact_rich \
      --motion-file data/smpl/yoga_yogi_balance_grounded.pt \
      --num-envs 4096 --batch-size 16384 --headless True --use-wandb \
      --overrides env.ref_respawn_offset=0.005

Short simulator smoke test (omit ``--use-wandb``)::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_contact_rich.py \
      --experiment-name smpl_yogi_contact_rich_smoke \
      --motion-file data/smpl/yoga_yogi_balance_grounded.pt \
      --num-envs 8 --batch-size 32 --training-max-steps 256 --headless True \
      --overrides env.ref_respawn_offset=0.005 env.contact_diagnostics_interval=1
"""

from __future__ import annotations

import argparse

from examples.experiments.mimic import mlp_2x_no_contact_rew as base
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


# Reuse all non-contact experiment construction and inference behavior.
terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
apply_inference_overrides = base.apply_inference_overrides


CONTACT_INPUT_KEYS = [
    "max_coords_obs",
    "contact_obs_v1",
    "contact_proximity_obs",
    "mimic_target_poses",
    "previous_actions",
]

# Explicit experiment values make the checkpoint contract auditable.  They are
# initial scaling choices rather than robot-specific physical constants.
CONTACT_OBS_V1_PARAMS = {
    "force_reference_n": 100.0,
    "force_clip_n": 5000.0,
    "force_rate_reference_n_per_s": 1000.0,
    "force_rate_clip_n_per_s": 50000.0,
    "friction_mu": 1.0,
    "friction_utilization_clip": 2.0,
    "velocity_reference_mps": 2.0,
    "contact_age_clip_s": 2.0,
    "air_age_clip_s": 2.0,
}


def _contact_observation_body_ids(robot_cfg: RobotConfig) -> list[int]:
    """Return the resolved deterministic kinematic-order observation IDs."""
    body_names = robot_cfg.contact_observation_bodies or []
    if not body_names:
        raise ValueError(
            "mlp_contact_rich requires at least one contact observation body. "
            "Call configure_robot_and_simulator() before env_config()."
        )
    name_to_id = {
        name: body_id
        for body_id, name in enumerate(robot_cfg.kinematic_info.body_names)
    }
    return [name_to_id[name] for name in body_names]


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
):
    """Sense all bodies, expose all to observations, and reward only the feet."""
    robot_cfg.update_fields(
        contact_bodies="all",
        contact_observation_bodies="all",
        contact_reward_bodies=[
            "all_left_foot_bodies",
            "all_right_foot_bodies",
        ],
    )


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build the 2x no-contact-reward baseline with rich contact inputs."""
    from protomotions.envs.component_factories import (
        contact_obs_v1_factory,
        max_coords_obs_factory,
        nearest_surface_obs_factory,
    )

    cfg = base.env_config(robot_cfg, args)
    body_ids = _contact_observation_body_ids(robot_cfg)

    # Rebuild the mapping in model input order.  Reuse the baseline target/action
    # components so their semantics and parameters remain byte-for-byte equal.
    baseline_observations = cfg.observation_components
    cfg.observation_components = {
        "max_coords_obs": max_coords_obs_factory(observe_contacts=False),
        "contact_obs_v1": contact_obs_v1_factory(
            body_ids=body_ids,
            **CONTACT_OBS_V1_PARAMS,
        ),
        "contact_proximity_obs": nearest_surface_obs_factory(body_ids=body_ids),
        "mimic_target_poses": baseline_observations["mimic_target_poses"],
        "previous_actions": baseline_observations["previous_actions"],
    }

    # Hysteresis belongs to the environment state tracker, not the pure
    # observation kernel.  Diagnostics are intentionally opt-in with this file.
    cfg.contact_force_on_threshold_n = 5.0
    cfg.contact_force_off_threshold_n = 2.0
    cfg.contact_diagnostics_interval = 100
    return cfg


def agent_config(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    args: argparse.Namespace,
) -> PPOAgentConfig:
    """Add the two contact keys without changing the baseline 2x networks."""
    cfg = base.agent_config(robot_config, env_config, args)
    cfg.model.in_keys = list(CONTACT_INPUT_KEYS)
    cfg.model.actor.in_keys = list(CONTACT_INPUT_KEYS)
    cfg.model.actor.mu_model.in_keys = list(CONTACT_INPUT_KEYS)
    cfg.model.critic.in_keys = list(CONTACT_INPUT_KEYS)
    return cfg
