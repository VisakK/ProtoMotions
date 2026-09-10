# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode EVERY intent code at a commanded state, and see what the decoder does.

The v9 head is 4 FSQ scalars x 5 levels = **625 codes**, one AR token, held 8
control steps.  That is small enough to enumerate exhaustively -- one env per
code, one batch -- which turns the question every previous sweep had to infer
into a direct measurement:

    at a state where the policy is commanded to hold and does not, is there ANY
    intent code whose decode holds?

``notes/V9_crucial_investigations/Student_v9_tier0_report.MD`` eliminated ten
inference-time arms (temperature, greedy, hysteresis, the contact-event refresh,
the deadline) and every one of them acts on *which code is selected*.  If no code
holds, code selection was never the question: the decoder's conditional behaviour
at that state is departure whatever intent it is handed, and the fix belongs in
the training task.  If some codes hold and the prior does not choose them, the
Tier-0 nulls need re-reading and the fix belongs in the prior.

Protocol, per commanded state (one probe plan each):

* env ``i`` is initialised **at** the commanded pose (``--start-at-goal``, the
  default) so arrival is not part of the measurement -- Tier-0 §3 already showed
  all four hold probes arrive on 36/36 replicas;
* the manual goal is installed **before** any stepping and the held intent is
  flushed (the fixed panel protocol, not ``legacy_settle``), then re-armed every
  ``--reissue-every`` s exactly as the panel does;
* ``FSQMaskedMimicModel._advance_stream_hysteretic`` is monkey-patched so that
  env ``i`` executes a **fixed** code for the whole window: the chunk clock, the
  contact-event trigger and nucleus sampling are all removed from the loop.  The
  patch touches the deployment path only and is reverted in a ``finally``;
* two control blocks share the batch and are *not* pinned -- the policy's own
  sampled stream and its greedy stream -- so the enumeration is read against the
  behaviour the probes actually measure.

Nothing is written to the checkpoint or the config.

Outputs (``--out-dir``): ``summary.json``, ``per_code.csv``, a **25x25
stick-figure montage** of every code's terminal pose (outer grid = scalars 0/1,
inner = scalars 2/3, tinted by goal-pose error, commanded pose ghosted behind),
a pose-error / departure-time heat map over the same layout, a pelvis-height
sparkline grid, and stick-figure videos of selected codes.

Usage::

    PYTHONPATH=. python data/scripts/decode_all_codes.py \\
      --checkpoint results/<run>/last.ckpt --headless \\
      --plans data/scripts/plans/hold_probe_standing.json \\
              data/scripts/plans/hold_probe_sideplank.json \\
      --out-dir output/decode_all_codes/last
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_setup import add_common_args  # noqa: E402  (no torch import yet)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--plans", type=str, nargs="+", required=True,
                    help="probe plans; goal 0 of each is the commanded state")
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--seconds", type=float, default=4.0,
                    help="rollout length. Tier-0 §3: the standing probe leaves "
                         "within ~2 s, so 4 s classifies hold-vs-leave")
parser.add_argument("--controls", type=int, default=32,
                    help="unpinned replicas per state, per stream (sampled, greedy)")
parser.add_argument("--hold-lead", type=float, default=1.2)
parser.add_argument("--reissue-every", type=float, default=0.5)
parser.add_argument("--start-at-goal", type=lambda s: s.lower() != "false", default=True,
                    help="False starts at the plan's own `start` clip/time instead")
parser.add_argument("--hold-threshold-m", type=float, default=0.15,
                    help="goal-pose error under which the pose counts as held "
                         "(the panel's pose_arrive_m)")
parser.add_argument("--videos", type=int, default=6,
                    help="stick-figure videos per state (best/median/worst codes "
                         "plus the two control streams)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

args.headless = True

from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

import logging  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from policy_setup import build, motion_names  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("decode_all_codes")


# --------------------------------------------------------------------------- #
def resolve_clip(needle: str, names) -> int:
    """Same rule as ``SequenceViz._resolve_clip``: prefer the non-synthetic clip."""
    needle = Path(str(needle)).stem
    hits = [i for i, n in enumerate(names) if needle.lower() in n.lower()]
    if len(hits) > 1:
        originals = [i for i in hits if not names[i].startswith("hold_")]
        if len(originals) == 1:
            return originals[0]
    if not hits:
        raise SystemExit(f"clip '{needle}' is not in this corpus")
    return hits[0]


def enumerate_codes(num_scalars: int, num_levels: int) -> torch.Tensor:
    """All ``num_levels ** num_scalars`` codes, in odd-symmetric FSQ values.

    ``FSQQuantizer`` rounds a bounded tanh to the integers centred on zero, so a
    5-level scalar takes values ``{-2,-1,0,1,2}``.  Row ``i`` is the mixed-radix
    expansion of ``i``, which makes the montage layout below a pure reshape.
    """
    half = (num_levels - 1) // 2
    grids = torch.meshgrid(
        *[torch.arange(num_levels) - half for _ in range(num_scalars)], indexing="ij"
    )
    return torch.stack([g.reshape(-1) for g in grids], dim=-1).float()


# --------------------------------------------------------------------------- #
def main() -> int:
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plans = []
    for path in args.plans:
        plan = json.loads(Path(path).read_text())
        plans.append((Path(path).stem, plan))

    # The batch layout has to be known before the simulator is built.
    probe_cfg = torch.load(
        Path(args.checkpoint).parent / "resolved_configs_inference.pt",
        map_location="cpu", weights_only=False,
    )
    fsq_cfg = probe_cfg["agent"].model.fsq
    num_scalars = int(fsq_cfg.num_fsq_scalars)
    num_levels = int(fsq_cfg.num_fsq_levels)
    num_codes = num_levels ** num_scalars
    per_state = num_codes + 2 * args.controls
    args.num_envs = per_state * len(plans)
    log.info(
        "%d codes (%d scalars x %d levels), %d controls x 2, %d states -> %d envs",
        num_codes, num_scalars, num_levels, args.controls, len(plans), args.num_envs,
    )

    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]
    device = env.device
    names = motion_names(built["motion_lib"])
    control = next(
        (c for c in env.control_manager.components.values()
         if hasattr(c, "set_manual_goal")), None
    )
    if control is None:
        raise SystemExit("no control component exposes set_manual_goal")
    graph = control.graph
    slots = control.config.num_goal_steps
    dt = float(env.dt)

    from protomotions.agents.evaluators.sequence_viz import resolve_config

    # ---------------- batch layout ---------------- #
    env_ids = torch.arange(args.num_envs, device=device)
    state_of_env = (env_ids // per_state).cpu()
    slot_of_env = (env_ids % per_state).cpu()
    pinned = slot_of_env < num_codes
    sampled_ctrl = (slot_of_env >= num_codes) & (slot_of_env < num_codes + args.controls)
    greedy_ctrl = slot_of_env >= num_codes + args.controls

    codes = enumerate_codes(num_scalars, num_levels).to(device)
    pinned_codes = torch.zeros(args.num_envs, num_scalars, device=device)
    pinned_codes[pinned] = codes[slot_of_env[pinned].to(device)]

    # ---------------- goal / start tensors ---------------- #
    node_ids = torch.full((args.num_envs, slots), -1, dtype=torch.long)
    pose_motion = torch.zeros(args.num_envs, slots, dtype=torch.long)
    pose_time = torch.zeros(args.num_envs, slots)
    offsets = torch.zeros(args.num_envs, slots)
    visible = torch.zeros(args.num_envs, slots, dtype=torch.bool)
    start_motion = torch.zeros(args.num_envs, dtype=torch.long)
    start_time = torch.zeros(args.num_envs)

    state_meta = []
    for s, (name, plan) in enumerate(plans):
        goal = plan["goals"][0]
        node = resolve_config(graph, goal["config"])
        if node is None:
            raise SystemExit(f"{name}: goal config '{goal['config']}' not in this graph")
        gm = resolve_clip(goal["pose_clip"], names)
        gt = float(goal["pose_time"])
        rows = state_of_env == s
        node_ids[rows, 0] = node
        pose_motion[rows, 0] = gm
        pose_time[rows, 0] = gt
        offsets[rows, 0] = args.hold_lead
        visible[rows, 0] = True
        if args.start_at_goal:
            start_motion[rows], start_time[rows] = gm, gt
        else:
            start_motion[rows] = resolve_clip(plan["start"]["clip"], names)
            start_time[rows] = float(plan["start"].get("time", 0.0))
        state_meta.append(
            dict(plan=name, goal=goal.get("name"), node=int(node),
                 node_key=graph.node_keys[node], pose_clip=names[gm], pose_time=gt,
                 start_clip=names[int(start_motion[rows][0])],
                 start_time=float(start_time[rows][0]))
        )
        log.info("state %d: %s -> node %d %s", s, name, node, graph.node_keys[node])

    goal_kwargs = dict(
        node_ids=node_ids.to(device), pose_motion_ids=pose_motion.to(device),
        pose_times=pose_time.to(device), time_offsets=offsets.to(device),
        pose_visible=visible.to(device), contact_visible=visible.to(device),
        # One pinned goal for the whole window; see set_manual_goal on why an
        # omitted hold reads as "stay 0 s" once the dwell channels are on.
        hold_seconds=(visible.float() * args.seconds).to(device),
    )

    # ---------------- pin the intent ---------------- #
    model = agent.model
    original = model._advance_stream_hysteretic
    pinned_dev = pinned.to(device)
    sampled_dev = sampled_ctrl.to(device)
    greedy_dev = greedy_ctrl.to(device)

    captured = {}

    def patched(tensordict, held, context, refresh, greedy):
        """Fixed code on the enumerated block; the real streams on the controls.

        ``greedy`` from the caller is ignored: both streams are generated here so
        the sampled and greedy controls can share one batch without touching
        ``fsq.inference_argmax``, which is global.

        The first call also stashes the AR prior's **full** distribution over the
        vocabulary at the commanded state.  That is what turns "no code the prior
        picks holds" into the sharper question "what probability does the prior
        put on the codes that DO hold" -- i.e. whether the holding region is
        merely low-ranked or effectively invisible to nucleus sampling.
        """
        rows = held.shape[0]
        if "logits" not in captured:
            with torch.no_grad():
                captured["logits"] = (
                    model._ar_head.next_logits_from_context(context, token_indices=None)
                    .float().cpu().clone()
                )
        out = pinned_codes[:rows].clone()
        s_codes = model._advance_stream(held, context, refresh, greedy=False)
        g_codes = model._advance_stream(held, context, refresh, greedy=True)
        out = torch.where(sampled_dev[:rows].unsqueeze(-1), s_codes, out)
        out = torch.where(greedy_dev[:rows].unsqueeze(-1), g_codes, out)
        return out

    model._advance_stream_hysteretic = patched

    steps = int(round(args.seconds / dt))
    reissue = max(int(round(args.reissue_every / dt)), 1)
    pose_err = np.zeros((steps, args.num_envs), dtype=np.float32)
    iou = np.zeros((steps, args.num_envs), dtype=np.float32)
    num_bodies = len(env.robot_config.kinematic_info.body_names)
    positions = np.zeros((steps, args.num_envs, num_bodies, 3), dtype=np.float32)
    root_rot = np.zeros((steps, args.num_envs, 4), dtype=np.float32)
    terminated = np.zeros((steps, args.num_envs), dtype=bool)

    agent.eval()
    try:
        env.motion_manager.motion_ids[env_ids] = start_motion.to(device)
        env.motion_manager.motion_times[env_ids] = start_time.to(device)
        obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)
        model.flush_held_intent()
        agent.pre_collect_step(0)

        control.set_manual_goal(**goal_kwargs)
        model.flush_held_intent()
        env._current_context = env._build_global_context(env.simulator.get_robot_state())
        env.compute_observations(context=env._current_context)
        obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(env.get_obs()))

        pelvis = control._pelvis_body_index
        for step in range(steps):
            if step > 0 and step % reissue == 0:
                # Re-arm the deadline exactly as the panel does; no flush, since
                # the active goal never changes in a single-goal plan.
                control.set_manual_goal(**goal_kwargs)
                env._current_context = env._build_global_context(
                    env.simulator.get_robot_state()
                )
                env.compute_observations(context=env._current_context)
                obs_td = agent.obs_dict_to_tensordict(
                    agent.add_agent_info_to_obs(env.get_obs())
                )

            outputs = model.forward_inference(obs_td)
            obs, *_ = env.step(outputs["action"])
            agent.pre_collect_step(step + 1)
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

            ctx = env._current_context
            pose_err[step] = ctx.contact_goal.pose_error.cpu().numpy()
            iou[step] = ctx.contact_goal.reached.cpu().numpy()
            if step == 0:
                seen = float(ctx.contact_goal.pose_error_visible.min())
                assert seen > 0.5, "some rows carry no pose measurement"
            state = env.simulator.get_robot_state()
            positions[step] = state.rigid_body_pos.cpu().numpy()
            root_rot[step] = state.rigid_body_rot[:, pelvis].cpu().numpy()
            terminated[step] = env.terminate_buf.cpu().numpy()
    finally:
        model._advance_stream_hysteretic = original
        try:
            control.clear_manual_goal()
        except Exception:  # pragma: no cover - teardown best effort
            pass

    np.savez_compressed(
        out_dir / "rollouts.npz",
        pose_err=pose_err, iou=iou, positions=positions, root_rot=root_rot,
        terminated=terminated, codes=codes.cpu().numpy(),
        prior_logits_step0=captured.get(
            "logits", torch.zeros(0)
        ).numpy(),
        code_token=model._codes_to_tokens(codes).reshape(-1).cpu().numpy(),
        state_of_env=state_of_env.numpy(), slot_of_env=slot_of_env.numpy(),
        pinned=pinned.numpy(), sampled=sampled_ctrl.numpy(), greedy=greedy_ctrl.numpy(),
        dt=np.float32(dt),
    )

    # ---------------- scoring ---------------- #
    from decode_all_codes_report import report  # noqa: E402  (local module)

    report(
        out_dir=out_dir, plans=state_meta, pose_err=pose_err, iou=iou,
        positions=positions, root_rot=root_rot, terminated=terminated,
        codes=codes.cpu().numpy(), state_of_env=state_of_env.numpy(),
        slot_of_env=slot_of_env.numpy(), num_codes=num_codes,
        num_levels=num_levels, num_scalars=num_scalars, controls=args.controls,
        dt=dt, hold_threshold=args.hold_threshold_m, videos=args.videos,
        body_names=list(env.robot_config.kinematic_info.body_names),
        common_body_names=list(env.robot_config.kinematic_info.body_names),
        parent_indices=list(env.robot_config.kinematic_info.parent_indices),
        conditionable=[int(i) for i in control.conditionable_body_ids.cpu()],
        goal_poses=_goal_pose_table(env, control, goal_kwargs, state_of_env, num_codes),
    )
    log.info("wrote %s", out_dir)
    return 0


def _goal_pose_table(env, control, goal_kwargs, state_of_env, num_codes):
    """Commanded pose per state, heading-normalised, for the montage ghost."""
    rows = []
    for s in range(int(state_of_env.max()) + 1):
        env_id = int((state_of_env == s).nonzero()[0])
        motion = goal_kwargs["pose_motion_ids"][env_id, 0].reshape(1)
        time = goal_kwargs["pose_times"][env_id, 0].reshape(1)
        ref = env.motion_lib.get_motion_state(motion, time)
        rows.append(
            (
                ref.rigid_body_pos[0].cpu().numpy(),
                ref.rigid_body_rot[0, control._pelvis_body_index].cpu().numpy(),
            )
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
