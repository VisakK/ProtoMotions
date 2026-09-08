# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Can the OTHER clip's expert take over at a matched state?

``matched_state_pairs.py`` finds points where two clips pass through the same
physical state with different commanded goals.  Switching clip *and* expert there
would break the (state, goal) correlation that
``goal_state_redundancy.py`` measures, without inventing an action label -- but
only if the incoming expert can actually track its own reference from the state
the outgoing clip produced.  Every reviewer of this project has asked for that
measurement (``notes/V9_crucial_investigations/yogi_method_review_2026-09-04/report.md``
§5: "measure whether each selected expert can recover from the student's actual
off-reference states"); this is it, with a concrete candidate list instead of an
open-ended study.

Protocol, per candidate pair and per replica:

1. reset at clip A's frame ``t_a - pre_s`` and let **A's own expert** drive for
   ``pre_s`` seconds, so the switch state is *physically produced* by A rather
   than teleported from the reference;
2. at the switch, set ``motion_id -> B`` and ``motion_time -> t_b`` and call
   ``env.align_motion_with_humanoid`` so B's reference is anchored at the robot
   (XY only -- the yaw caveat is why the candidate list is filtered on heading);
3. drive with **B's expert** for ``post_s`` seconds.  Expert routing is by
   ``motion_id`` (``multi_expert.py:161``), so the switch selects the new teacher
   automatically.

Three arms share the batch:

* ``switch``   -- the above;
* ``continue`` -- identical, but no switch: A keeps going with A's expert. The
  ceiling, since the teacher is on its own reference throughout;
* ``shuffled`` -- switch to a **random** frame of clip B. The negative control:
  if this scores as well as ``switch``, the pair filter is doing nothing.

The score is the max-body tracking error against the *active* reference, and the
gate is v9's own training termination threshold (0.25 m,
``resolved_configs.yaml:1775``): a switch that stays inside it is a switch the
training loop would not have killed.

Needs the frozen teachers, which ``apply_inference_overrides`` strips, so it
loads ``resolved_configs.pt`` rather than the inference config.

Usage::

    PYTHONPATH=. python data/scripts/expert_handoff_screen.py \\
      --checkpoint results/<run>/last.ckpt --headless \\
      --switch-points output/matched_state_pairs/switch_points.json \\
      --out-dir output/expert_handoff
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
parser.add_argument("--switch-points", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--pairs", type=int, default=48)
parser.add_argument("--replicas", type=int, default=6)
parser.add_argument("--pre-s", type=float, default=0.5,
                    help="seconds A's expert drives before the switch, so the "
                         "handoff state is an arrival and not a teleport")
parser.add_argument("--post-s", type=float, default=1.0)
parser.add_argument("--terminate-m", type=float, default=0.25,
                    help="v9's training tracking-error termination threshold")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

args.headless = True
args.resolved_configs = "resolved_configs.pt"   # the teachers live here

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

from policy_setup import build  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("expert_handoff_screen")

ARMS = ("switch", "continue", "shuffled")


def tracking_error(env) -> torch.Tensor:
    """Max-body distance to the active reference — v9's termination quantity."""
    ctx = env._current_context
    ref = ctx.mimic.ref_state.rigid_body_pos
    cur = ctx.current.rigid_body_pos
    return (ref - cur).pow(2).sum(-1).sqrt().max(dim=-1)[0]


def main() -> int:
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = json.loads(Path(args.switch_points).read_text())[: args.pairs]
    num_pairs = len(points)
    per_arm = num_pairs * args.replicas
    args.num_envs = per_arm * len(ARMS)
    log.info("%d pairs x %d replicas x %d arms -> %d envs",
             num_pairs, args.replicas, len(ARMS), args.num_envs)

    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]
    device = env.device
    if not getattr(agent, "expert_actors", None):
        raise SystemExit(
            "the loaded agent has no frozen experts; this needs resolved_configs.pt"
        )
    log.info("experts: %d, routing table over %d clips",
             len(agent.expert_actors), len(agent._motion_expert))

    dt = float(env.dt)
    pre_steps = int(round(args.pre_s / dt))
    post_steps = int(round(args.post_s / dt))
    env_ids = torch.arange(args.num_envs, device=device)
    arm_of_env = (env_ids // per_arm).cpu().numpy()
    pair_of_env = ((env_ids % per_arm) // args.replicas).cpu().numpy()

    motion_a = torch.tensor([p["motion_a"] for p in points], dtype=torch.long)
    motion_b = torch.tensor([p["motion_b"] for p in points], dtype=torch.long)
    t_a = torch.tensor([p["t_a"] for p in points], dtype=torch.float32)
    t_b = torch.tensor([p["t_b"] for p in points], dtype=torch.float32)
    lengths = env.motion_lib.motion_lengths.cpu()

    pair_idx = torch.as_tensor(pair_of_env, dtype=torch.long)
    arm_idx = torch.as_tensor(arm_of_env, dtype=torch.long)
    start_motion = motion_a[pair_idx]
    start_time = (t_a[pair_idx] - args.pre_s).clamp(min=0.0)

    generator = torch.Generator().manual_seed(args.seed)
    shuffled_t = torch.rand(args.num_envs, generator=generator) * (
        lengths[motion_b[pair_idx]] - 0.05
    )
    target_motion = torch.where(arm_idx == 1, motion_a[pair_idx], motion_b[pair_idx])
    target_time = torch.where(
        arm_idx == 1,
        t_a[pair_idx],
        torch.where(arm_idx == 2, shuffled_t, t_b[pair_idx]),
    )

    errors = np.zeros((post_steps, args.num_envs), dtype=np.float32)
    pre_errors = np.zeros((pre_steps, args.num_envs), dtype=np.float32)

    agent.eval()
    with torch.no_grad():
        env.motion_manager.motion_ids[env_ids] = start_motion.to(device)
        env.motion_manager.motion_times[env_ids] = start_time.to(device)
        obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)
        agent.pre_collect_step(0)
        obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

        # --- phase 1: A's expert produces the arrival state --- #
        for step in range(pre_steps):
            action = agent._collect_external_expert_action(obs_td)
            obs, *_ = env.step(action)
            agent.pre_collect_step(step + 1)
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            pre_errors[step] = tracking_error(env).cpu().numpy()

        # --- the switch --- #
        env.motion_manager.motion_ids[env_ids] = target_motion.to(device)
        env.motion_manager.motion_times[env_ids] = target_time.to(device)
        # Re-anchor the new reference to where the robot actually is. XY only;
        # the candidate list is filtered so the headings already agree.
        env.align_motion_with_humanoid(
            env_ids, env.simulator.get_root_state().root_pos
        )
        env._current_context = env._build_global_context(env.simulator.get_robot_state())
        env.compute_observations(context=env._current_context)
        obs_td = agent.obs_dict_to_tensordict(
            agent.add_agent_info_to_obs(env.get_obs())
        )
        switch_error = tracking_error(env).cpu().numpy()

        # --- phase 2: the incoming expert drives --- #
        for step in range(post_steps):
            action = agent._collect_external_expert_action(obs_td)
            obs, *_ = env.step(action)
            agent.pre_collect_step(pre_steps + step + 1)
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            errors[step] = tracking_error(env).cpu().numpy()

    np.savez_compressed(
        out_dir / "handoff.npz", errors=errors, pre_errors=pre_errors,
        switch_error=switch_error, arm_of_env=arm_of_env, pair_of_env=pair_of_env,
        dt=np.float32(dt),
    )

    # ------------------------------------------------------------------ #
    peak = errors.max(axis=0)
    survives = peak <= args.terminate_m
    rows = []
    print(f"\n{'arm':<12}{'n':>6}{'survives 0.25 m':>17}{'peak p50':>10}"
          f"{'peak p90':>10}{'at switch':>11}{'final':>9}")
    summary = {}
    for a, name in enumerate(ARMS):
        sel = arm_of_env == a
        summary[name] = dict(
            n=int(sel.sum()),
            survive_rate=float(survives[sel].mean()),
            peak_p50=float(np.median(peak[sel])),
            peak_p90=float(np.percentile(peak[sel], 90)),
            error_at_switch_p50=float(np.median(switch_error[sel])),
            final_p50=float(np.median(errors[-1, sel])),
        )
        s = summary[name]
        print(f"{name:<12}{s['n']:>6}{s['survive_rate']:>16.3f}"
              f"{s['peak_p50']:>10.3f}{s['peak_p90']:>10.3f}"
              f"{s['error_at_switch_p50']:>11.3f}{s['final_p50']:>9.3f}")

    # per-pair, on the switch arm
    print(f"\nper-pair survival on the switch arm ({args.replicas} replicas each):")
    print(f"{'surv':>5}{'peak':>8}{'goal gap':>10}  clip A -> clip B")
    for p in range(num_pairs):
        sel = (arm_of_env == 0) & (pair_of_env == p)
        ctrl = (arm_of_env == 1) & (pair_of_env == p)
        point = points[p]
        rows.append(dict(
            pair=p, node=point["node"],
            clip_a=point["clip_a"], t_a=point["t_a"],
            clip_b=point["clip_b"], t_b=point["t_b"],
            goal_gap_m=point["goal_gap_m"], pose_gap_m=point["pose_gap_m"],
            yaw_gap_deg=point["yaw_gap_deg"],
            switch_survive=float(survives[sel].mean()),
            switch_peak_m=float(np.median(peak[sel])),
            switch_error_m=float(np.median(switch_error[sel])),
            continue_survive=float(survives[ctrl].mean()),
            continue_peak_m=float(np.median(peak[ctrl])),
        ))
    for row in sorted(rows, key=lambda r: (-r["switch_survive"], r["switch_peak_m"]))[:15]:
        print(f"{row['switch_survive']:>5.2f}{row['switch_peak_m']:>8.3f}"
              f"{row['goal_gap_m']:>10.3f}  {row['clip_a'][:34]}@{row['t_a']:.2f}"
              f" -> {row['clip_b'][:34]}@{row['t_b']:.2f}")

    usable = [r for r in rows if r["switch_survive"] >= 0.5]
    print(f"\n{len(usable)}/{num_pairs} pairs survive the switch on >=50% of replicas")
    (out_dir / "summary.json").write_text(json.dumps(
        dict(pairs=num_pairs, replicas=args.replicas, pre_s=args.pre_s,
             post_s=args.post_s, terminate_m=args.terminate_m,
             arms=summary, usable_pairs=len(usable), per_pair=rows), indent=1))

    _plot(out_dir / "handoff_curves.png", errors, arm_of_env, dt, args.terminate_m)
    log.info("wrote %s", out_dir)
    return 0


def _plot(path: Path, errors, arm_of_env, dt, threshold):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    t = np.arange(errors.shape[0]) * dt
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=120)
    colours = {"switch": "#1f77b4", "continue": "#2ca02c", "shuffled": "#d62728"}
    for a, name in enumerate(ARMS):
        sel = arm_of_env == a
        band = np.percentile(errors[:, sel], [25, 50, 75], axis=1)
        ax.plot(t, band[1], color=colours[name], lw=2.0, label=f"{name} (median)")
        ax.fill_between(t, band[0], band[2], color=colours[name], alpha=0.15)
    ax.axhline(threshold, color="#333333", ls="--", lw=1.0,
               label=f"v9 training termination ({threshold:g} m)")
    ax.set_xlabel("seconds after the clip/expert switch")
    ax.set_ylabel("max-body tracking error vs the ACTIVE reference (m)")
    ax.set_title("Can the incoming clip's expert take over at a matched state?",
                 fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
