# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact-rich Stage-1 expert *with* the contact-matching reward switched on.

Identical to :mod:`mlp_contact_rich` -- same 6x2048 actor, 4x2048 critic,
optimizers, terminations, tracking rewards, and the same ``contact_obs_v1`` +
proximity observation pathway -- except for the contact-matching term, which
changes in three ways at once because none of them is useful alone:

1. **Scored over every body, not the feet.** Yoga is held on hands, forearms,
   head, knees and trunk as often as on feet; a feet-only term is blind to the
   contacts that actually carry the pose.
2. **Reference labels replaced.** The shipped labels come from a joint-centre
   height/speed heuristic that audits at 0-16 % precision here, and has no notion
   of body-body contact. This experiment expects a motion file annotated by
   ``data/scripts/annotate_contacts_geometric.py``, which runs the validated
   geometric detector (exact surface distances between typed collision geoms,
   per-zone calibrated hysteresis, and the COM/support-polygon static model).
   Pass ``--motion-file data/smpl/yoga_yogi_balance_grounded_contacts.pt``.
3. **Normalized.** ``compute_contact_match_rew`` sums per-body mismatches, so
   scoring 24 bodies instead of 4 would multiply the term's scale by six and
   swamp the tracking rewards. ``normalize=True`` makes it a mean mismatch
   fraction in [0, 1], independent of how many bodies are scored.

Semantics on both sides is "in contact with **anything**": the simulated flag is
per-body net contact force above a threshold, which self-collision makes true for
body-body contact, and the reference annotator marks ground and body-body contact
the same way. Scoring a ground-only reference against an any-contact simulation
would penalise the policy for the self-contacts the pose requires.

Known tension worth watching
----------------------------
About 13 % of the reference labels are *inferred* rather than strictly planted:
the SMPL fit floats genuinely load-bearing parts (the headstand head sits
13.5-14.2 cm off the floor), and the static-support model recovers them. For
those frames the contact term asks for a contact the reference *pose* does not
literally have, so it pulls against the tracking reward. That is deliberate --
the human was in contact and the float is a fit artifact -- but it means
``CONTACT_MATCH_WEIGHT`` trades pose fidelity against contact fidelity. Watch
``raw_r/contact_match_rew`` against ``raw_r/gt_rew``; if tracking degrades,
lower the weight before concluding the labels are wrong.

Train::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_contact_rich_rew.py \
      --experiment-name smpl_yogi_balance_contact_rich_rew_74kg \
      --motion-file data/smpl/yoga_yogi_balance_grounded_contacts.pt \
      --num-envs 4096 --batch-size 16384 --headless True --use-wandb \
      --overrides env.ref_respawn_offset=0.005
"""

from __future__ import annotations

import argparse

from examples.experiments.mimic import mlp_contact_rich as base
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
apply_inference_overrides = base.apply_inference_overrides
agent_config = base.agent_config

# The one knob this experiment exists to turn. Normalized, so this is the
# penalty at total contact disagreement; a typical mismatch costs a fraction of
# it. Sized to sit well below the tracking terms (gt 0.5 / gr 0.3) so contact
# shapes behaviour without overriding the pose.
CONTACT_MATCH_WEIGHT = -1.0


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
):
    """Sense, observe and now also *reward* contact on every body."""
    robot_cfg.update_fields(
        contact_bodies="all",
        contact_observation_bodies="all",
        contact_reward_bodies="all",
    )


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.component_factories import contact_match_rew_factory

    cfg = base.env_config(robot_cfg, args)
    cfg.reward_components["contact_match_rew"] = contact_match_rew_factory(
        weight=CONTACT_MATCH_WEIGHT,
        zero_during_grace_period=True,
        normalize=True,
    )
    return cfg
