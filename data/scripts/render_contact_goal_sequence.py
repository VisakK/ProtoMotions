# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive a contact-graph student through a scripted sequence of goals, on video.

:mod:`query_contact_goal` asks the student **one** question. This asks a series of
them in a single unbroken episode -- "stand, then handstand, then stand, then
warrior II" -- and records the result, so the thing you watch is the policy's own
transitions between contact configurations rather than four separate rollouts.

A plan is JSON::

    {
      "start": {"clip": "220923_Handstand", "time": 0.0},
      "goals": [
        {"name": "tadasana",  "node": 0,   "pose_clip": "...", "pose_time": 1.83,
         "reach_s": 3.0, "hold_s": 2.0},
        ...
      ]
    }

``clip``/``pose_clip`` are case-insensitive substrings, resolved against the
library and required to be unambiguous.

Five things this gets right that a naive driver does not, each of which is a real
failure mode of :meth:`ContactGraphControl.set_manual_goal`:

* **The plan lives in all five goal slots, not just slot 0.** That is what the
  policy saw in training -- slot *k* is the *k*-th upcoming hold, with offsets
  increasing across slots -- and all five reach the prior's transformer.
* **Observations are rebuilt after every goal change.** ``set_manual_goal`` only
  touches the control component; without a rebuild the first action after each
  switch is taken on the *previous* goal.
* **The deadline is re-armed rather than allowed to expire.** The countdown
  clamps at ``min_lead_s`` (0.2 s) and parks there, which in training only ever
  occurred for a single step before the schedule advanced. Holding a pose by
  letting it sit at 0.2 s is off-distribution; re-arming to ~1.2 s -- the median
  hold-to-hold gap in the graph -- is not.
* **Fresh tensors every time.** ``set_manual_goal`` stores what it is handed
  without copying when dtype and device already match, so a reused buffer would
  mutate the live goal from under the policy.
* **``dones`` is ignored.** The inference config sets ``max_episode_length`` to
  1e6 and strips terminations; the flag only means the *start* clip ran out of
  frames, which is irrelevant to a manual goal and must not end the recording.

Usage::

    PYTHONPATH=. python data/scripts/render_contact_goal_sequence.py \
      --checkpoint results/smpl_yogi_contact_graph_student_s2/last.ckpt \
      --plan data/scripts/plans/tadasana_handstand_warrior2.json \
      --out-dir output/renderings/contact_goal_sequence
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
parser.add_argument("--plan", type=str, required=True, help="JSON plan file")
parser.add_argument("--out-dir", type=str, default="output/renderings/contact_goal_sequence")
parser.add_argument("--label", type=str, default=None,
                    help="output basename (default: the plan file's stem)")
parser.add_argument("--no-video", action="store_true",
                    help="measure only; skip the viewport recording")
parser.add_argument("--no-pose", action="store_true",
                    help="hide the pose half of every goal (contact set only)")
parser.add_argument("--no-contacts", action="store_true",
                    help="hide the contact half of every goal (pose only)")
parser.add_argument("--reissue-every", type=float, default=0.5,
                    help="seconds between re-arming the plan")
parser.add_argument("--hold-lead", type=float, default=1.2,
                    help="deadline held during a goal's hold phase, seconds. The "
                         "graph's median hold-to-hold gap is 1.43 s; the countdown's "
                         "0.2 s floor is not a value the policy saw sustained.")
parser.add_argument("--settle-steps", type=int, default=10,
                    help="policy steps after reset before the plan starts")
parser.add_argument("--flush-seconds", type=float, default=2.0,
                    help="wait for Isaac's async viewport writer before compiling")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stochastic", action="store_true")
args = parser.parse_args()

args.num_envs = 1
# RecordingMixin.render() is gated on `not headless` and IsaacLab captures the
# active viewport, so a headless run would compile an empty video.
args.headless = bool(args.no_video)

from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from policy_setup import build, motion_names  # noqa: E402


def log(fmt, *fmt_args) -> None:
    print("render_contact_goal_sequence: " + (fmt % fmt_args if fmt_args else fmt),
          flush=True)


def resolve_clip(names: list[str], needle: str, what: str) -> int:
    matches = [i for i, n in enumerate(names) if needle.lower() in n.lower()]
    if not matches:
        raise SystemExit(f"no clip matches {what} '{needle}'")
    if len(matches) > 1:
        raise SystemExit(
            f"{what} '{needle}' is ambiguous, matches {len(matches)}: "
            f"{[names[i] for i in matches[:5]]}"
        )
    return matches[0]


def resolve_node(graph, entry: dict) -> int:
    """Node id for one plan goal, preferring its ``config`` string over its ``node``.

    Node ids are renumbered by every graph rebuild, and the low ids survive a
    rebuild while meaning something else -- so a plan that names goals only by id
    keeps running against a new graph and silently asks for a different pose.
    ``config`` is the canonical configuration string and is stable, so it wins
    when both are given and the disagreement is reported rather than ignored.
    """
    config = entry.get("config")
    node = entry.get("node")
    if config is None and node is None:
        raise SystemExit(
            f"goal '{entry.get('name', '?')}' names neither 'config' nor 'node'"
        )
    if config is None:
        return int(node)

    try:
        resolved = graph.node_id_for_key(config)
    except KeyError as error:
        raise SystemExit(
            f"goal '{entry.get('name', '?')}': {error.args[0]}\n"
            "This is what a plan written against a different graph build looks "
            "like. Re-derive the configuration strings from the graph you are "
            "querying."
        ) from None

    if node is not None and int(node) != resolved:
        print(
            f"render_contact_goal_sequence: goal '{entry.get('name', '?')}' says "
            f"node {int(node)} but '{config}' is node {resolved} in this graph; "
            f"using {resolved}. The plan was written against a different build.",
            flush=True,
        )
    return resolved


class GoalDriver:
    """Issues a plan into the control component's goal slots and keeps it fresh."""

    def __init__(self, env, agent, control, simulator, goals: list[dict], args):
        self.env, self.agent, self.control = env, agent, control
        self.simulator = simulator
        self.goals = goals
        self.args = args
        self.slots = control.config.num_goal_steps
        self.num_envs = env.num_envs
        self.device = env.device
        for goal in goals:
            node = goal["node"]
            if not 0 <= node < control.graph.num_nodes:
                raise SystemExit(
                    f"goal '{goal['name']}' node {node} outside [0, "
                    f"{control.graph.num_nodes - 1}]"
                )
        # Absolute end time of each goal on the sequence's own clock.
        self.ends, elapsed = [], 0.0
        for goal in goals:
            elapsed += goal["reach_s"] + goal["hold_s"]
            self.ends.append(elapsed)
        self.total_s = elapsed

    def active_index(self, t: float) -> int:
        for i, end in enumerate(self.ends):
            if t < end:
                return i
        return len(self.goals) - 1

    def issue(self, t: float):
        """Write the plan from time ``t`` onward into the five goal slots."""
        n, k, dev = self.num_envs, self.slots, self.device
        node = torch.full((n, k), -1, dtype=torch.long, device=dev)
        pose_motion = torch.zeros(n, k, dtype=torch.long, device=dev)
        pose_time = torch.zeros(n, k, device=dev)
        offset = torch.zeros(n, k, device=dev)
        pose_visible = torch.zeros(n, k, dtype=torch.bool, device=dev)
        contact_visible = torch.zeros(n, k, dtype=torch.bool, device=dev)

        start = self.active_index(t)
        for slot, index in enumerate(range(start, min(start + k, len(self.goals)))):
            goal = self.goals[index]
            # Slot 0 counts down to the end of its *reach* window and then holds
            # at --hold-lead rather than decaying to the 0.2 s floor.
            remaining = self.ends[index] - goal["hold_s"] - t
            deadline = max(remaining, self.args.hold_lead)
            node[:, slot] = goal["node"]
            pose_motion[:, slot] = goal["pose_motion"]
            pose_time[:, slot] = goal["pose_time"]
            offset[:, slot] = deadline
            pose_visible[:, slot] = not self.args.no_pose
            contact_visible[:, slot] = not self.args.no_contacts
        if not bool(pose_visible[:, 0].any() or contact_visible[:, 0].any()):
            raise SystemExit(
                "the nearest goal would specify neither a pose nor a contact set; "
                "--no-pose and --no-contacts cannot both be given"
            )

        self.control.set_manual_goal(
            node_ids=node,
            pose_motion_ids=pose_motion,
            pose_times=pose_time,
            time_offsets=offset,
            pose_visible=pose_visible,
            contact_visible=contact_visible,
        )
        return self.refresh_obs()

    def refresh_obs(self):
        """Rebuild observations so the next action sees the goal just issued."""
        env = self.env
        env._current_context = None
        env._current_context = env._build_global_context(
            self.simulator.get_robot_state()
        )
        env.compute_observations(context=env._current_context)
        return self.agent.obs_dict_to_tensordict(
            self.agent.add_agent_info_to_obs(env.get_obs())
        )


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    plan = json.loads(Path(args.plan).read_text())
    label = args.label or Path(args.plan).stem

    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]
    motion_lib, simulator = built["motion_lib"], built["simulator"]

    if not args.no_video and getattr(simulator, "headless", True):
        raise SystemExit("simulator came up headless; cannot capture a viewport")

    control = None
    for component in env.control_manager.components.values():
        if hasattr(component, "set_manual_goal"):
            control = component
            break
    if control is None:
        raise SystemExit("this checkpoint has no contact-graph control component")

    names = motion_names(motion_lib)
    goals = []
    for entry in plan["goals"]:
        goals.append(
            {
                "name": entry["name"],
                "node": resolve_node(control.graph, entry),
                "pose_motion": resolve_clip(names, entry["pose_clip"], "pose_clip"),
                "pose_time": float(entry["pose_time"]),
                "reach_s": float(entry["reach_s"]),
                "hold_s": float(entry["hold_s"]),
            }
        )
    start_motion = resolve_clip(names, plan["start"]["clip"], "start.clip")
    start_time = float(plan["start"].get("time", 0.0))

    log("plan '%s': %s", label, " -> ".join(g["name"] for g in goals))
    for goal in goals:
        log("  %-12s node %-4d %s  pose %s @ %.2f s  (reach %.1f s, hold %.1f s)",
            goal["name"], goal["node"],
            control.graph.describe_node(goal["node"])[:60],
            names[goal["pose_motion"]][:44], goal["pose_time"],
            goal["reach_s"], goal["hold_s"])
    log("start: %s @ %.2f s", names[start_motion], start_time)

    # Reset first: set_manual_goal does not set _initialized, and without it the
    # countdown in step() silently never runs.
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.motion_manager.motion_ids[env_ids] = start_motion
    env.motion_manager.motion_times[env_ids] = start_time
    obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)

    agent.eval()
    agent.pre_collect_step(0)
    obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
    action_key = "action" if args.stochastic else "mean_action"
    dt = float(env.dt)

    for _ in range(max(args.settle_steps, 0)):
        with torch.no_grad():
            outputs = agent.model.forward_inference(obs_td)
        action = outputs.get(action_key, outputs.get("action"))
        obs, _, _, _, _ = env.step(action)
        obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

    driver = GoalDriver(env, agent, control, simulator, goals, args)
    total_steps = int(round(driver.total_s / dt))
    reissue_every = max(int(round(args.reissue_every / dt)), 1)
    log("%d policy steps (%.1f s at dt=%.4f), re-arming every %d steps",
        total_steps, driver.total_s, dt, reissue_every)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    recording_dir = None
    if not args.no_video:
        simulator._camera_target["env"] = 0
        simulator._user_recording_video_path = str(out_dir / f"{label}-%s")
        simulator._toggle_video_record()
        log("recording armed -> %s", simulator._user_recording_video_path)

    zone_names = control._ground_zone_names
    goal_zone_ids = control._ground_pair_ids
    trace = {"t": [], "goal": [], "iou": [], "zones": [], "root_z": []}

    obs_td = driver.issue(0.0)
    for step in range(total_steps):
        t = step * dt
        if step > 0 and step % reissue_every == 0:
            obs_td = driver.issue(t)

        with torch.no_grad():
            outputs = agent.model.forward_inference(obs_td)
        action = outputs.get(action_key, outputs.get("action"))
        # `dones` is deliberately ignored: it only says the START clip ran out of
        # frames, which has nothing to do with a manual goal, and the inference
        # config already disables terminations and the episode-length cap.
        obs, _, _dones, _terminated, _ = env.step(action)
        obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

        state = simulator.get_robot_state()
        magnitude = state.rigid_body_contact_forces.norm(dim=-1)
        zone_force = torch.zeros(env.num_envs, len(zone_names), device=env.device)
        zone_force.index_add_(
            1, control._ground_zone_rows, magnitude[:, control._ground_zone_cols]
        )
        current = zone_force[0] > control.config.ground_contact_threshold_n
        index = driver.active_index(t)
        goal = control.graph.node_contact[goals[index]["node"]]
        wanted = goal.index_select(0, goal_zone_ids) > 0.5
        union = float((current | wanted).sum())
        trace["t"].append(t)
        trace["goal"].append(goals[index]["name"])
        trace["iou"].append(float((current & wanted).sum()) / union if union else 1.0)
        trace["zones"].append([zone_names[i] for i in torch.nonzero(current).flatten().tolist()])
        trace["root_z"].append(float(state.rigid_body_pos[0, 0, 2]))

        if (step + 1) % 60 == 0:
            log("  t=%5.1f s  goal=%-11s iou=%.2f  root_z=%.2f  zones=%s",
                t, goals[index]["name"], trace["iou"][-1], trace["root_z"][-1],
                ",".join(trace["zones"][-1]) or "-")

    if not args.no_video:
        simulator._toggle_video_record()
        time.sleep(max(args.flush_seconds, 0.0))
        recording_dir = simulator._curr_user_recording_name
        landed = len(list(Path(recording_dir).glob("*.png"))) if recording_dir else 0
        log("captured %d frames; compiling", landed)
        simulator.render()  # compiles the mp4 and deletes the PNG folder
        produced = Path(f"{recording_dir}.mp4")
        if produced.exists():
            produced.replace(out_dir / f"{label}.mp4")
            for suffix in (".motion", ".markers.pt", ".objects.pt"):
                side = Path(f"{recording_dir}{suffix}")
                if side.exists():
                    side.replace(out_dir / f"{label}{suffix}")
            log("wrote %s", out_dir / f"{label}.mp4")
        else:
            log("WARNING: no mp4 produced at %s", produced)
        shutil.rmtree(recording_dir, ignore_errors=True)

    # Per-goal summary on the frames of that goal's hold phase.
    summary = []
    ends = driver.ends
    for i, goal in enumerate(goals):
        hold_from = ends[i] - goal["hold_s"]
        sel = [j for j, t in enumerate(trace["t"]) if hold_from <= t < ends[i]]
        if not sel:
            continue
        ious = [trace["iou"][j] for j in sel]
        final_zones = trace["zones"][sel[-1]]
        wanted = sorted(
            zone_names[i2] for i2 in torch.nonzero(
                control.graph.node_contact[goal["node"]].index_select(0, goal_zone_ids) > 0.5
            ).flatten().tolist()
        )
        summary.append({
            "goal": goal["name"],
            "node": goal["node"],
            "config": control.graph.describe_node(goal["node"]),
            "hold_window_s": [round(hold_from, 2), round(ends[i], 2)],
            "ground_iou_mean_over_hold": round(float(np.mean(ious)), 3),
            "wanted_ground_zones": wanted,
            "final_ground_zones": sorted(final_zones),
            "reached_exactly": sorted(final_zones) == wanted,
            "root_z_mean": round(float(np.mean([trace["root_z"][j] for j in sel])), 3),
        })

    print("\n" + "=" * 78)
    for row in summary:
        print(f"  {row['goal']:<12} IoU {row['ground_iou_mean_over_hold']:.2f}  "
              f"root_z {row['root_z_mean']:.2f}  exact={row['reached_exactly']}")
        print(f"      wanted {row['wanted_ground_zones']}")
        print(f"      got    {row['final_ground_zones']}")
    print("=" * 78)

    (out_dir / f"{label}.json").write_text(json.dumps(
        {
            "plan": plan,
            "label": label,
            "checkpoint": args.checkpoint,
            "pose_given": not args.no_pose,
            "contacts_given": not args.no_contacts,
            "hold_lead_s": args.hold_lead,
            "summary": summary,
            "trace": trace,
        },
        indent=2,
    ) + "\n")
    log("wrote %s", out_dir / f"{label}.json")

    if hasattr(simulator, "shutdown"):
        simulator.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
