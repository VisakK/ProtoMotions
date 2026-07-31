# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill-graph **node** co-adaptation: fine-tune a node to CATCH its edge's arrivals.

Second half of the bidirectional hand-off (``notes/brainstorm_skill_graph.MD``
§4.3). The v1 exit failure showed the shape of the problem: tadasana could not
leave a crossed-leg configuration "because escaping requires stepping and its
reference is a frozen stance" -- a training-distribution gap, not a dynamics
impossibility. Here the TARGET node is fine-tuned with half of its episodes
initialised at the edge policy's *actual arrival states* (collected by
``collect_state_bank.py --mode edge-arrival`` at the pre-registered switch
steps), so its basin grows toward where the edge actually lands.

Identical to ``mlp_node_small.py`` except:

1. **BankResetEnv** with ``bank_prob = 0.5``: half nominal resets (frozen pose
   + reset noise -- the frozen-identity guard, so the node cannot drift away
   from what it is) and half arrival resets. Reset noise applies on top of both.
2. **``advantage_normalization.use_ema = False``** -- the documented real fix
   for the downdog3 epoch-1126 collapse (notes lessons §4.1): a near-perfect
   hold shrinks the EMA advantage std, and the first real failures then divide
   by a stale denominator (16x oversized updates -> KL runaway). Catch training
   *introduces* failures on purpose, which is exactly the trigger.
3. Warm start from the trained node via ``--checkpoint``.

No-regression requirement (evaluated, not assumed): the fine-tuned node must
keep its nominal hold and must not degrade composition under the old protocol.

Train::

    python protomotions/train_agent.py --robot-name smpl_yogi --simulator isaaclab \
      --experiment-path examples/experiments/mimic/mlp_node_catch.py \
      --experiment-name node_handstand_catch \
      --motion-file data/smpl/skill_graph_handstand/node_handstand.pt \
      --checkpoint results/node_handstand/final.ckpt \
      --num-envs 4096 --batch-size 16384 --headless True --use-wandb
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig

import mlp_node_small as base

# Reuse the node expert's config wholesale; only reset distribution and
# advantage normalisation differ.
terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
configure_robot_and_simulator = base.configure_robot_and_simulator

BANK_DIR = "data/smpl/skill_graph_handstand"
NODE_ARRIVAL_BANKS = {
    "node_handstand": f"{BANK_DIR}/bank_arrivals_edge_kickup_v3.pt",
    "node_tadasana": f"{BANK_DIR}/bank_arrivals_edge_exit_v3.pt",
}
BANK_PROB = float(os.environ.get("NODE_BANK_PROB", "0.5"))


def _node_key(args) -> str:
    mf = os.path.basename(getattr(args, "motion_file", "") or "")
    for k in NODE_ARRIVAL_BANKS:
        if k in mf:
            return k
    raise ValueError(
        f"Cannot infer node from '{mf}'. Expected one of {list(NODE_ARRIVAL_BANKS)}.")


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from bank_reset import BankResetEnvConfig

    bank_file = NODE_ARRIVAL_BANKS[_node_key(args)]
    if BANK_PROB > 0 and not os.path.exists(bank_file):
        raise FileNotFoundError(
            f"arrival bank {bank_file} not found -- collect it from the trained edge first:\n"
            f"  python data/scripts/collect_state_bank.py --mode edge-arrival "
            f"--checkpoint results/<edge_v3>/final.ckpt --switch-steps <window> --out {bank_file}"
        )

    import dataclasses

    cfg = base.env_config(robot_cfg, args)
    fields = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
              if f.init and f.name != "_target_"}
    return BankResetEnvConfig(**fields, bank_file=bank_file, bank_prob=BANK_PROB)


def agent_config(robot_config: RobotConfig, env_config: EnvConfig,
                 args: argparse.Namespace) -> PPOAgentConfig:
    from protomotions.agents.ppo.config import AdvantageNormalizationConfig

    cfg = base.agent_config(robot_config, env_config, args)
    cfg.advantage_normalization = AdvantageNormalizationConfig(
        enabled=True, shift_mean=True, use_ema=False)
    return cfg


def apply_inference_overrides(robot_cfg, simulator_cfg, env_cfg, agent_cfg,
                              terrain_cfg, motion_lib_cfg, scene_lib_cfg, args):
    base.apply_inference_overrides(robot_cfg, simulator_cfg, env_cfg, agent_cfg,
                                   terrain_cfg, motion_lib_cfg, scene_lib_cfg, args)
    if hasattr(env_cfg, "bank_prob"):
        env_cfg.bank_prob = 0.0
