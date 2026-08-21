# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ask a trained contact-graph student for the motion that reaches a goal.

This is the thing the Stage-2 run exists to make possible.  During training the
goals come from the clip an environment happens to be playing; here they come
from the command line:

    "reach L_HAND:G | R_HAND:G @ inverted, holding this pose, within 6 seconds"

and the student is rolled out from a chosen starting state with that goal pinned.
The pose half and the contact half can each be withheld, which is how the two are
told apart:

    --no-pose      the contact configuration alone -- the interesting query
    --no-contacts  the pose alone -- vanilla MaskedMimic behaviour, as a control

What comes back, per query: the fraction of the rollout on which the policy's
*measured* ground-contact zones match the goal's, the zone set it actually ended
in, and the per-body distance to the goal pose.  Contact is read from the
simulator's own per-body contact forces, so "did it get there" is measured, not
inferred from the kinematics -- the same reason the graph was built from rollouts
in the first place.

Only the **ground** half of a configuration can be scored at runtime:
ProtoMotions' contact sensors are filtered against the terrain, so body-body
pairs (crow's shank-on-upper-arm) are not observable without the offline pair
view.  That is a limit of the measurement, not of the goal -- body-body pairs are
still part of the specification handed to the policy.

Usage::

    # what can I ask for?
    PYTHONPATH=. python data/scripts/query_contact_goal.py \
      --checkpoint results/smpl_yogi_contact_graph_student_s2/last.ckpt --list-nodes

    # reach the handstand contact configuration from standing, pose hidden
    PYTHONPATH=. python data/scripts/query_contact_goal.py \
      --checkpoint results/smpl_yogi_contact_graph_student_s2/last.ckpt \
      --goal-contacts L_HAND:G R_HAND:G --goal-orientation inverted \
      --start-clip Handstand --start-time 0 --horizon 6 --no-pose \
      --out-dir results/contact_goal_queries
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_setup import add_common_args  # noqa: E402  (no torch import yet)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--out-dir", type=str, default=None)
parser.add_argument("--list-nodes", action="store_true",
                    help="print the graph's contact configurations and exit")
parser.add_argument("--list-limit", type=int, default=40)
parser.add_argument("--goal-node", type=int, default=None,
                    help="graph node id (see --list-nodes)")
parser.add_argument("--goal-contacts", type=str, nargs="*", default=None,
                    help="contact pair names, e.g. L_HAND:G R_HAND:G L_SHANK+L_UPPER_ARM")
parser.add_argument("--goal-orientation", type=str, default=None,
                    help="trunk orientation bin (upright/inverted/prone/supine/side_l/side_r)")
parser.add_argument("--pose-clip", type=str, default=None,
                    help="clip to take the goal pose from (substring); default: a clip "
                         "where the graph says this configuration was held")
parser.add_argument("--pose-time", type=float, default=None)
parser.add_argument("--start-clip", type=str, default=None,
                    help="clip to start from (substring); default: the pose clip")
parser.add_argument("--start-time", type=float, default=0.0)
parser.add_argument("--horizon", type=float, default=6.0,
                    help="seconds allowed to reach the goal")
parser.add_argument("--settle", type=float, default=3.0,
                    help="extra seconds to keep holding after the deadline")
parser.add_argument("--no-pose", action="store_true", help="hide the pose half")
parser.add_argument("--no-contacts", action="store_true", help="hide the contact half")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stochastic", action="store_true")
args = parser.parse_args()

args.num_envs = 1
args.headless = True

from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from policy_setup import build, motion_names  # noqa: E402


def log(fmt, *fmt_args) -> None:
    print("query_contact_goal: " + (fmt % fmt_args if fmt_args else fmt), flush=True)


def resolve_node(graph, args) -> int:
    """Node id from an explicit id, or from a contact set + orientation."""
    if args.goal_node is not None:
        if not 0 <= args.goal_node < graph.num_nodes:
            raise SystemExit(f"--goal-node out of range [0, {graph.num_nodes - 1}]")
        return args.goal_node
    if not args.goal_contacts:
        raise SystemExit("pass --goal-node or --goal-contacts (or --list-nodes)")

    unknown = [p for p in args.goal_contacts if p not in graph.pair_names]
    if unknown:
        raise SystemExit(
            f"unknown contact pairs {unknown}\nknown ground pairs: "
            f"{[p for p in graph.pair_names if p.endswith(':G')]}"
        )
    wanted = torch.zeros(graph.num_pairs)
    for pair in args.goal_contacts:
        wanted[graph.pair_names.index(pair)] = 1.0

    contact = graph.node_contact.cpu()
    distance = (contact - wanted).abs().sum(dim=1)
    if args.goal_orientation is not None:
        if args.goal_orientation not in graph.orientation_names:
            raise SystemExit(
                f"--goal-orientation must be one of {graph.orientation_names}"
            )
        target = graph.orientation_names.index(args.goal_orientation)
        distance = distance + 100.0 * (graph.node_orient.cpu() != target).float()
    best = int(distance.argmin())
    if float(distance[best]) > 0:
        log("no exact match; nearest configuration differs by %.0f pair(s): %s",
            float(distance[best]) % 100, graph.node_keys[best])
    return best


def graph_json_for(graph_file: Path):
    path = Path(graph_file).parent / "contact_graph.json"
    return json.loads(path.read_text()) if path.is_file() else None


def find_hold(graph_json, node_id: int):
    """A (clip, time) where the graph observed this node being held."""
    if graph_json is None:
        return None
    best = None
    for clip, record in graph_json["clips"].items():
        for segment in record["segments"]:
            if segment["node"] == node_id and segment["trusted"]:
                if best is None or segment["duration_s"] > best[2]:
                    best = (clip, segment["t_hold"], segment["duration_s"])
    return best


def pick_motion(names, needle, what):
    if needle is None:
        return None
    matches = [i for i, n in enumerate(names) if needle.lower() in n.lower()]
    if not matches:
        raise SystemExit(f"no clip matches --{what} '{needle}'")
    if len(matches) > 1:
        log("--%s '%s' matches %d clips; using %s", what, needle, len(matches),
            names[matches[0]])
    return matches[0]


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]
    motion_lib, simulator = built["motion_lib"], built["simulator"]

    control = None
    for component in env.control_manager.components.values():
        if hasattr(component, "set_manual_goal"):
            control = component
            break
    if control is None:
        raise SystemExit(
            "this checkpoint's environment has no contact-graph control component"
        )
    graph = control.graph

    if args.list_nodes:
        graph_json = graph_json_for(control.config.graph_file)
        dwell = {}
        if graph_json is not None:
            for node in graph_json["nodes"]:
                dwell[node["key"]] = (node["total_dwell_s"], len(node["motions"]))
        rows = [
            (dwell.get(key, (0.0, 0))[0], dwell.get(key, (0.0, 0))[1], i, key)
            for i, key in enumerate(graph.node_keys)
        ]
        rows.sort(reverse=True)
        print(f"\n{graph.num_nodes} contact configurations "
              f"({graph.num_pairs} pairs, {graph.num_orientations} orientation bins)\n")
        print(f"{'id':>5} {'dwell_s':>9} {'clips':>6}  configuration")
        for total, clips, node_id, key in rows[: args.list_limit]:
            print(f"{node_id:>5} {total:>9.1f} {clips:>6}  {key}")
        return 0

    node_id = resolve_node(graph, args)
    log("goal node %d: %s", node_id, graph.node_keys[node_id])

    names = motion_names(motion_lib)
    graph_json = graph_json_for(control.config.graph_file)
    pose_motion = pick_motion(names, args.pose_clip, "pose-clip")
    pose_time = args.pose_time

    if pose_motion is None or pose_time is None:
        observed = find_hold(graph_json, node_id)
        if observed is None:
            raise SystemExit(
                "the graph has no recorded hold for this node; pass --pose-clip "
                "and --pose-time explicitly"
            )
        clip, hold_time, duration = observed
        if pose_motion is None:
            pose_motion = names.index(clip)
        if pose_time is None:
            pose_time = hold_time
        log("goal pose from %s @ %.2f s (held %.2f s in the graph)",
            names[pose_motion], pose_time, duration)

    start_motion = pick_motion(names, args.start_clip, "start-clip")
    if start_motion is None:
        start_motion = pose_motion
    log("starting from %s @ %.2f s", names[start_motion], args.start_time)

    steps_total = int(round((args.horizon + args.settle) / float(env.dt)))
    env_ids = torch.arange(env.num_envs, device=env.device)
    device = env.device
    goal_steps = control.config.num_goal_steps

    env.motion_manager.motion_ids[env_ids] = start_motion
    env.motion_manager.motion_times[env_ids] = args.start_time
    obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)

    def column(value, dtype=torch.float32):
        """One goal in slot 0; the remaining slots are unspecified."""
        out = torch.zeros(env.num_envs, goal_steps, dtype=dtype, device=device)
        out[:, 0] = value
        return out

    node_column = torch.full(
        (env.num_envs, goal_steps), -1, dtype=torch.long, device=device
    )
    node_column[:, 0] = node_id
    control.set_manual_goal(
        node_ids=node_column,
        pose_motion_ids=torch.full(
            (env.num_envs, goal_steps), pose_motion, dtype=torch.long, device=device
        ),
        pose_times=torch.full((env.num_envs, goal_steps), float(pose_time), device=device),
        time_offsets=column(float(args.horizon)),
        pose_visible=column(not args.no_pose, dtype=torch.bool),
        contact_visible=column(not args.no_contacts, dtype=torch.bool),
    )
    log("query: pose %s, contacts %s, horizon %.1f s (+%.1f s settle)",
        "hidden" if args.no_pose else "given",
        "hidden" if args.no_contacts else "given", args.horizon, args.settle)

    # env.reset() computed observations from the *clip's* goal schedule; rebuild
    # them so the very first action already sees the goal that was asked for.
    env._current_context = None
    env._current_context = env._build_global_context(simulator.get_robot_state())
    env.compute_observations(context=env._current_context)
    obs = env.get_obs()

    goal_pose = motion_lib.get_motion_state(
        torch.full((1,), pose_motion, dtype=torch.long, device=device),
        torch.full((1,), float(pose_time), device=device),
    )
    goal_body_pos = goal_pose.rigid_body_pos[0].clone()

    zone_names = control._ground_zone_names
    goal_zone = (
        graph.node_contact[node_id].index_select(0, control._ground_pair_ids) > 0.5
    )

    agent.eval()
    agent.pre_collect_step(0)
    obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
    action_key = "action" if args.stochastic else "mean_action"

    iou_trace, zone_trace, body_pos_trace = [], [], []
    for _ in range(steps_total):
        with torch.no_grad():
            outputs = agent.model.forward_inference(obs_td)
        action = outputs.get(action_key, outputs.get("action"))
        obs, _, dones, _terminated, _ = env.step(action)
        obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

        state = simulator.get_robot_state()
        magnitude = state.rigid_body_contact_forces.norm(dim=-1)
        zone_force = torch.zeros(env.num_envs, len(zone_names), device=device)
        zone_force.index_add_(
            1, control._ground_zone_rows, magnitude[:, control._ground_zone_cols]
        )
        current = zone_force[0] > control.config.ground_contact_threshold_n
        union = float((current | goal_zone).sum())
        iou_trace.append(
            float((current & goal_zone).sum()) / union if union else 1.0
        )
        zone_trace.append(current.cpu().numpy())
        body_pos_trace.append(state.rigid_body_pos[0].cpu().numpy())
        if bool(dones.any()):
            log("episode ended early after %d steps", len(iou_trace))
            break

    iou = np.asarray(iou_trace)
    zones = np.stack(zone_trace)
    body_pos = np.stack(body_pos_trace).astype(np.float32)
    hold_from = min(int(round(args.horizon / float(env.dt))), len(iou) - 1)

    # Register both to their own root before comparing: the query is about the
    # pose reached, not about where in the world it was reached.
    final = torch.from_numpy(body_pos[-1]).to(goal_body_pos.device)
    body_error = (
        (final - final[0:1]) - (goal_body_pos - goal_body_pos[0:1])
    ).norm(dim=-1)

    achieved = [zone_names[i] for i in np.nonzero(zones[-1])[0]]
    wanted = [zone_names[i] for i in torch.nonzero(goal_zone).flatten().cpu().numpy()]
    summary = {
        "goal_node": node_id,
        "goal_config": graph.node_keys[node_id],
        "goal_ground_zones": wanted,
        "pose_clip": names[pose_motion],
        "pose_time_s": float(pose_time),
        "start_clip": names[start_motion],
        "start_time_s": float(args.start_time),
        "pose_given": not args.no_pose,
        "contacts_given": not args.no_contacts,
        "horizon_s": float(args.horizon),
        "steps": int(len(iou)),
        "ground_iou_mean": float(iou.mean()),
        "ground_iou_after_horizon": float(iou[hold_from:].mean()),
        "ground_iou_final": float(iou[-1]),
        "final_ground_zones": achieved,
        "goal_reached_exactly": sorted(achieved) == sorted(wanted),
        "body_pos_error_mean_m": float(body_error.mean()),
        "body_pos_error_max_m": float(body_error.max()),
    }

    print("\n" + "=" * 70)
    for key, value in summary.items():
        print(f"  {key:26s} {value}")
    print("=" * 70)

    if args.out_dir:
        out_dir = Path(args.out_dir) / f"node{node_id:04d}_{names[start_motion][:40]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "query.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(
            out_dir / "trajectory.npz",
            ground_iou=iou,
            ground_zones=zones,
            zone_names=np.array(zone_names),
            body_pos=body_pos,
            goal_body_pos=goal_body_pos.cpu().numpy(),
            dt_ctrl=np.array(float(env.dt)),
        )
        log("wrote %s", out_dir)

    if hasattr(simulator, "shutdown"):
        simulator.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
