# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distil a goal-conditioned (Design-B) expert in a student env.

Card S1 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``. A tracker expert (every expert before
Design B) reads proprioception and the dense reference future, so its deep-copied observation components
compute the same thing in any env that plays the same clips. A goal-conditioned expert also reads the
*goal*: its ``masked_mimic_*`` and ``contact_goal_*`` components are bound to ``ctx.masked_mimic`` and
``ctx.contact_goal``, which in a student env carry the student's view (5 slots, 6 bodies, masked) instead
of the expert's (2 slots, 24 bodies, always visible). This module

* records the expert's goal contract from its resolved config (:func:`expert_goal_contract`): its slot and
  body counts, schedule semantics, the env settings its inputs depend on, its graph by sha256 and pair
  vocabulary, and its release artifacts. ``ContactGraphControl`` asserts its own config against it;
* rewires the copied components onto the unmasked view that control publishes as
  ``ctx.expert_masked_mimic`` / ``ctx.expert_contact_goal`` (:func:`rewire_expert_goal_view`);
* checks the student's identity against the expert's (:func:`port_identity_problems`): robot, plant,
  simulator rates, motion package, graph and a single-expert routing table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Context views a copied expert component reads its goal from -> the expert's unmasked view.
EXPERT_GOAL_VIEW = {
    "masked_mimic": "expert_masked_mimic",
    "contact_goal": "expert_contact_goal",
}

# A goal-conditioned expert's goal inputs (``mlp_goal_conditioned.ACTOR_KEYS``).
GOAL_COMPONENT_KEYS = (
    "masked_mimic_target_poses",
    "masked_mimic_target_masks",
    "masked_mimic_target_times",
    "masked_mimic_target_poses_masks",
    "contact_goal_obs",
    "contact_goal_masks",
)

_VISIBILITY_ONE = ("pose_visible_prob", "contact_visible_prob", "full_pose_prob")
_VISIBILITY_ZERO = ("force_max_conditioned_bodies_prob", "force_small_num_conditioned_bodies_prob")


def field_path(path: str):
    """The ``EnvContext`` FieldPath for a dotted path, e.g. ``"expert_masked_mimic.ref_pos"``."""
    from protomotions.envs.context_paths import FieldPath
    from protomotions.envs.context_views import EnvContext

    node: Any = EnvContext
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            raise ValueError(f"{path!r} is not a field of EnvContext")
    if not isinstance(node, FieldPath) or node.path != path:
        raise ValueError(f"{path!r} is not a field of EnvContext")
    return node


def contact_graph_control_config(env_config):
    """The env config's one contact-graph control component config."""
    components = getattr(env_config, "control_components", None) or {}
    hits = [c for c in components.values() if hasattr(c, "graph_file") and hasattr(c, "num_goal_steps")]
    if len(hits) != 1:
        raise ValueError(f"expected one contact-graph control component, found {len(hits)}")
    return hits[0]


def reads_goal(expert_agent_config) -> bool:
    """Whether the expert's actor consumes any goal input (a Design-B expert)."""
    from protomotions.agents.supervised.expert_utils import get_expert_actor_in_keys

    keys = set(get_expert_actor_in_keys(expert_agent_config))
    return any(k in keys for k in GOAL_COMPONENT_KEYS)


def expert_conditionable_body_count(expert_env_config) -> int:
    """Bodies per goal slot in the expert's pose half (its components' ``conditionable_body_ids``)."""
    counts = set()
    for key in ("masked_mimic_target_poses", "masked_mimic_target_masks"):
        component = (expert_env_config.observation_components or {}).get(key)
        if component is None:
            continue
        counts.add(len(component.static_params["conditionable_body_ids"]))
    if len(counts) != 1:
        raise ValueError(f"the expert's pose-half components disagree on their body count: {sorted(counts)}")
    return counts.pop()


def expert_goal_contract(expert_env_config) -> Dict[str, Any]:
    """What a student env must match for the expert's copied goal inputs to be its training inputs.

    Plain Python values only: the dict is stored in the student's resolved config
    (``ContactGraphControlConfig.expert_view_contract``) and checked at every construction.
    """
    from protomotions.components.contact_graph import ContactGraph
    from protomotions.utils.release_identity import file_sha256

    ctrl = contact_graph_control_config(expert_env_config)
    env = expert_env_config
    graph_file = str(ctrl.graph_file)
    full = all(float(getattr(ctrl, k, 0.0)) == 1.0 for k in _VISIBILITY_ONE) and all(
        float(getattr(ctrl, k, 0.0) or 0.0) == 0.0 for k in _VISIBILITY_ZERO
    )
    return {
        "num_goal_steps": int(ctrl.num_goal_steps),
        "full_visibility": bool(full),
        "include_current_segment": bool(getattr(ctrl, "include_current_segment", False)),
        "dwell_channels": bool(getattr(ctrl, "dwell_channels", False)),
        "min_lead_s": float(ctrl.min_lead_s),
        "interval_schedule": bool(getattr(ctrl, "interval_schedule", False)),
        "history_time_clip_s": float(getattr(ctrl, "history_time_clip_s", 10.0)),
        "far_goal_prob": float(getattr(ctrl, "far_goal_prob", 0.0) or 0.0),
        "num_history_events": int(getattr(ctrl, "num_history_events", 0) or 0),
        "ref_respawn_offset": float(env.ref_respawn_offset),
        "contact_force_on_threshold_n": float(getattr(env, "contact_force_on_threshold_n", 5.0)),
        "contact_force_off_threshold_n": float(getattr(env, "contact_force_off_threshold_n", 2.0)),
        "ref_contact_smooth_window": int(getattr(env, "ref_contact_smooth_window", 0) or 0),
        "num_state_history_steps": int(getattr(env, "num_state_history_steps", 0) or 0),
        # Per-step realignment re-anchors the reference, and with it the expert's world-anchored goal poses.
        "realign_motion_with_humanoid_on_each_step": bool(
            getattr(getattr(env, "motion_manager", None), "realign_motion_with_humanoid_on_each_step", False)
        ),
        "graph_file": graph_file,
        "graph_sha256": file_sha256(graph_file),
        "pair_names": list(ContactGraph.from_file(graph_file).pair_names),
        "release_file": str(getattr(ctrl, "release_file", "") or ""),
        "physics_tables_file": str(getattr(ctrl, "physics_tables_file", "") or ""),
        "contact_targets_file": str(getattr(ctrl, "contact_targets_file", "") or ""),
    }


def rewire_expert_goal_view(
    components: Mapping[str, Any], mapping: Mapping[str, str] = EXPERT_GOAL_VIEW
) -> Dict[str, List[str]]:
    """Point every copied component's goal bindings at the expert view, in place.

    ``components`` are the ``expert_*`` copies from ``get_expert_observation_components`` (deep copies, so
    nothing shared is touched). Returns ``{component: [rewired parameters]}``.
    """
    rewired: Dict[str, List[str]] = {}
    for name, component in components.items():
        bindings = getattr(component, "dynamic_vars", None) or {}
        for param, path in list(bindings.items()):
            head, dot, rest = path.path.partition(".")
            if dot and head in mapping:
                bindings[param] = field_path(f"{mapping[head]}.{rest}")
                rewired.setdefault(name, []).append(param)
    return rewired


def goal_view_bindings(components: Mapping[str, Any]) -> Dict[str, List[str]]:
    """``{component: [paths]}`` of every binding still reading the student's goal view (should be empty)."""
    left: Dict[str, List[str]] = {}
    for name, component in components.items():
        for path in (getattr(component, "dynamic_vars", None) or {}).values():
            if path.path.partition(".")[0] in EXPERT_GOAL_VIEW:
                left.setdefault(name, []).append(path.path)
    return left


def single_expert_routing(motion_names: List[str]) -> Dict[str, Any]:
    """A routing table (``MultiExpertSupervisedAgent``) sending every motion to expert 0."""
    return {"motion_expert": [0] * len(motion_names), "motion_names": list(motion_names)}


def port_identity_problems(
    *,
    robot_config,
    expert_robot_config,
    motion_file: str,
    expert_motion_file: str,
    graph_file: str,
    contract: Mapping[str, Any],
    expert_paths: List[str],
    routing_file: Optional[str],
    simulator_config=None,
    expert_simulator_config=None,
) -> List[str]:
    """Everything that would make the student env not the expert's env, as messages (empty: identical).

    The robot (class, assets, plant sha256), the simulator rates, the motion package (sha256), the graph
    (sha256 and pair vocabulary, against the contract) and a single-expert routing table over the graph's
    motions in order.
    """
    from protomotions.components.contact_graph import ContactGraph
    from protomotions.utils import plant_identity
    from protomotions.utils.release_identity import file_sha256

    problems: List[str] = []
    if len(expert_paths) != 1:
        problems.append(f"a release-v3 port distils one expert, got {len(expert_paths)}")

    if type(robot_config).__name__ != type(expert_robot_config).__name__:
        problems.append(f"robot {type(robot_config).__name__} is not the expert's {type(expert_robot_config).__name__}")
    for field in ("asset_root", "asset_file_name", "usd_asset_file_name"):
        mine = getattr(getattr(robot_config, "asset", None), field, None)
        theirs = getattr(getattr(expert_robot_config, "asset", None), field, None)
        if mine != theirs:
            problems.append(f"robot asset {field} {mine!r} is not the expert's {theirs!r}")
    mjcf, expert_mjcf = plant_identity.robot_mjcf(robot_config), plant_identity.robot_mjcf(expert_robot_config)
    if mjcf and expert_mjcf and plant_identity.sha256(mjcf) != plant_identity.sha256(expert_mjcf):
        problems.append("the robot MJCF's plant sha256 is not the expert's")

    if simulator_config is not None and expert_simulator_config is not None:
        mine, theirs = getattr(simulator_config, "sim", None), getattr(expert_simulator_config, "sim", None)
        if mine is not None and theirs is not None and mine != theirs:
            problems.append(f"simulator params {mine} are not the expert's {theirs}")

    if file_sha256(motion_file) != file_sha256(expert_motion_file):
        problems.append(f"motion package {motion_file} is not the expert's {expert_motion_file}")

    if file_sha256(graph_file) != contract["graph_sha256"]:
        problems.append(f"graph {graph_file} is not the expert's {contract['graph_file']}")
    graph = ContactGraph.from_file(graph_file)
    if list(graph.pair_names) != list(contract["pair_names"]):
        problems.append("the graph's pair vocabulary is not the expert's")

    if not routing_file:
        problems.append("no routing table (motion_expert_file)")
    else:
        table = json.loads(Path(routing_file).read_text())
        if table.get("motion_names") != list(graph.motion_names):
            problems.append(f"routing table {routing_file} does not list the graph's motions in order")
        if any(int(e) != 0 for e in table.get("motion_expert", [])) or len(table.get("motion_expert", [])) != len(
            graph.motion_names
        ):
            problems.append(f"routing table {routing_file} is not single-expert over the graph's motions")
    return problems


__all__ = [
    "EXPERT_GOAL_VIEW",
    "GOAL_COMPONENT_KEYS",
    "contact_graph_control_config",
    "expert_conditionable_body_count",
    "expert_goal_contract",
    "field_path",
    "goal_view_bindings",
    "port_identity_problems",
    "reads_goal",
    "rewire_expert_goal_view",
    "single_expert_routing",
]
