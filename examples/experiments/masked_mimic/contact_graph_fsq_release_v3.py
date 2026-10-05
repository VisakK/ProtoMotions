# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 FSQ student on release v3, distilled from the goal-conditioned expert (card S1 of
``expert_revist/graph_growth_2026_10_03/PLAN.MD``).

v9's recipe -- one-token code, the DAgger action loss, root-relative student targets -- ported to a
Design-B teacher: G3's ``epoch_3420.ckpt`` (``results/smpl_yogi_v2_expert56_g3_2f132f4299``), trained on
``holds_repaired_ftC_posefix.release_v3.2f132f4299`` and plant v2. Every earlier student distilled trackers,
whose copied observation components read proprioception and the dense reference future and so computed the
same thing in any env. This teacher also reads its *goal*, through components bound to ``ctx.masked_mimic`` /
``ctx.contact_goal``; in this env those carry the student's view (5 slots, 6 bodies, masked), not the 2 slots
x 24 bodies, always visible, the teacher was trained on. The port, every piece read from the teacher's
resolved training config:

1. **The teacher's view of the goal.** ``ContactGraphControl`` publishes slots ``[:2]`` of the student's own
   window, unmasked, as ``ctx.expert_masked_mimic`` / ``ctx.expert_contact_goal`` (``expert_view_*``), and the
   copied ``expert_*`` components are rewired onto it (``expert_port.rewire_expert_goal_view``). The window is
   a prefix of the student's (``ContactGraph.next_goal_indices`` serves the same first slots for any count),
   so the teacher sees what it saw in training at the same state.
2. **The student's schedule has the teacher's semantics**: ``include_current_segment``, ``dwell_channels``,
   ``min_lead_s``, ``interval_schedule`` and the dwell scale. They are copied here and the control asserts them
   against the recorded contract at every construction, inference builds included.
3. **The env is the teacher's env**: its graph, its release record and artifacts (verified by sha256 at
   construction), its ``ref_respawn_offset``, contact hysteresis, reference smoothing and realignment rule.
4. **Identity asserts** (``expert_port.port_identity_problems``): robot ``smpl_yogi_v2`` and its plant sha256,
   the simulator rates, the motion package, the graph by sha256 and pair vocabulary, one expert and an
   all-zeros routing table over the graph's motions (``make_single_expert_routing.py``).
5. **The merge-loop crash is fixed in the base** (``contact_graph_transformer.env_config``): the teacher's
   future window feeds its critic-only ``mimic_target_poses`` and is skipped.

Not changed: the FSQ head, the DAgger mixture, the student's own observations. S2 trains this; S1's parity
tests (``data/scripts/s1_port_parity.py``) are what say the port is exact.

    CONFIG_ONLY=1 bash data/scripts/run_student_distill_release_v3.sh
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
from pathlib import Path

from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

ROUTE_PLANS = "data/scripts/plans_release_v3_route"
RELEASE_PLANS = "data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v3.2f132f4299/plans"
# The panel: the five edges on their own training clips and from a held S, and family forks on training's
# deadline (make_route_probe_plans.py), all run with SequenceVizConfig.timing = "training".
DEFAULT_VIZ_PLANS = [
    *(f"{ROUTE_PLANS}/route_edge_{e}.json" for e in ("E1", "E3", "B1", "E2", "E5")),
    *(f"{RELEASE_PLANS}/edge_{e}.json" for e in ("E1", "E3", "B1", "E2", "E5")),
    f"{ROUTE_PLANS}/fork_Crane_Crow_Pose_or_Bakasana.json",
    f"{ROUTE_PLANS}/fork_Warrior_II_Pose_or_Virabhadrasana_II.json",
    f"{ROUTE_PLANS}/fork_Warrior_III_Pose_or_Virabhadrasana_III.json",
    f"{ROUTE_PLANS}/fork_Tree_Pose_or_Vrksasana.json",
    f"{ROUTE_PLANS}/fork_Side_Plank_Pose_or_Vasisthasana.json",
    f"{ROUTE_PLANS}/fork_Downward_Facing_Dog_pose_or_Adho_Mukha_S.json",
]


def _load_base_module():
    """v9, by path (the experiment files are not a package)."""
    path = Path(__file__).with_name("contact_graph_fsq_v9.py")
    spec = importlib.util.spec_from_file_location("contact_graph_fsq_v9_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

NUM_GOAL_STEPS = base.NUM_GOAL_STEPS
STATE_KEYS = base.STATE_KEYS
UNNORMALIZED_STATE_KEYS = base.UNNORMALIZED_STATE_KEYS
CONTACT_EVENT_FLAG_KEY = base.CONTACT_EVENT_FLAG_KEY
ENCODER_FUTURE_STEPS = base.ENCODER_FUTURE_STEPS


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """v9's arguments. The graph defaults to the teacher's, and the panel to S1's plans."""
    base.additional_experiment_arguments(parser)
    parser.set_defaults(
        contact_graph_file=None,
        viz_plan_files=list(DEFAULT_VIZ_PLANS),
        viz_num_sequences=len(DEFAULT_VIZ_PLANS),
        viz_max_seconds=28.0,
    )


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config


def _expert_path(args: argparse.Namespace) -> str:
    paths = base.base.base._expert_paths(args)
    if len(paths) != 1:
        raise ValueError(f"the release-v3 student distils one goal-conditioned expert, got {len(paths)}: {paths}")
    return paths[0]


@functools.lru_cache(maxsize=4)
def _resolved(path: str) -> dict:
    from protomotions.utils.config_utils import load_resolved_configs_from_checkpoint

    return load_resolved_configs_from_checkpoint(path)


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """The base's sensing (every body and body pair, as the teacher), and the teacher's simulator rates."""
    base.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    expert = _resolved(_expert_path(args))
    mine, theirs = getattr(simulator_cfg, "sim", None), getattr(expert["simulator"], "sim", None)
    if mine is not None and theirs is not None and mine != theirs:
        raise ValueError(f"simulator params {mine} are not the expert's {theirs}")


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """v9's environment, ported to the teacher (module docstring, items 1-4)."""
    from protomotions.agents.supervised.expert_port import (
        GOAL_COMPONENT_KEYS,
        expert_conditionable_body_count,
        expert_goal_contract,
        goal_view_bindings,
        port_identity_problems,
        reads_goal,
        rewire_expert_goal_view,
    )
    from protomotions.envs.control.contact_graph_control import ContactGraphControl

    path = _expert_path(args)
    expert = _resolved(path)
    if not reads_goal(expert["agent"]):
        raise ValueError(f"{path} is not a goal-conditioned expert; distil trackers with contact_graph_fsq_v9.py")
    contract = expert_goal_contract(expert["env"])
    if contract["num_history_events"] or contract["far_goal_prob"]:
        raise ValueError("the expert reads contact-event history or far-goal promotion, which its view does not carry")
    args.contact_graph_file = getattr(args, "contact_graph_file", None) or contract["graph_file"]

    cfg = base.env_config(robot_cfg, args)

    control = cfg.control_components["contact_graph"]
    for key in ContactGraphControl.EXPERT_SCHEDULE_KEYS:
        setattr(control, key, contract[key])
    for key in ("release_file", "physics_tables_file", "contact_targets_file"):
        setattr(control, key, contract[key])
    control.far_goal_prob = 0.0
    control.expert_view_steps = contract["num_goal_steps"]
    control.expert_view_num_bodies = expert_conditionable_body_count(expert["env"])
    control.expert_view_contract = contract
    for key in ContactGraphControl.EXPERT_ENV_KEYS:
        setattr(cfg, key, contract[key])
    cfg.motion_manager.realign_motion_with_humanoid_on_each_step = contract[
        "realign_motion_with_humanoid_on_each_step"
    ]

    copies = {k: v for k, v in cfg.observation_components.items() if k.startswith("expert_")}
    rewired = rewire_expert_goal_view(copies)
    left = goal_view_bindings(copies)
    if left:
        raise ValueError(f"expert components still read the student's goal view: {left}")
    missing = [f"expert_{k}" for k in GOAL_COMPONENT_KEYS if f"expert_{k}" in copies and f"expert_{k}" not in rewired]
    if missing:
        raise ValueError(f"expert goal components were not rewired: {missing}")

    problems = port_identity_problems(
        robot_config=robot_cfg,
        expert_robot_config=expert["robot"],
        motion_file=args.motion_file,
        expert_motion_file=expert["motion_lib"].motion_file,
        graph_file=args.contact_graph_file,
        contract=contract,
        expert_paths=[path],
        routing_file=getattr(args, "motion_expert_file", None),
    )
    if problems:
        raise ValueError("the student env is not the expert's:\n  " + "\n  ".join(problems))
    print(f"release-v3 port: {len(rewired)} expert components rewired onto the expert view "
          f"({sorted(rewired)}); schedule {[(k, contract[k]) for k in ContactGraphControl.EXPERT_SCHEDULE_KEYS]}")
    return cfg


def agent_config(robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace):
    """v9's agent; the panel serves goals on training's timing (S0's dwell fix)."""
    cfg = base.agent_config(robot_config, env_config, args)
    if cfg.sequence_viz is not None:
        cfg.sequence_viz.timing = "training"
    return cfg


def apply_inference_overrides(
    robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg, motion_lib_cfg, scene_lib_cfg, args
):
    """The base's (the experts and their components go), and the expert view with them."""
    base.apply_inference_overrides(
        robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg, motion_lib_cfg, scene_lib_cfg, args
    )
    control = (getattr(env_cfg, "control_components", None) or {}).get("contact_graph")
    if control is not None:
        control.expert_view_steps = 0
