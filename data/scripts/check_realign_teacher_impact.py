# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Do the frozen teachers still track their own clips with the reference re-anchored?

``check_student_xy_invariance.py`` shows that
``realign_motion_with_humanoid_on_each_step`` is what makes the *teacher* query
and the tracking termination translation-invariant, which is what the student's
``root_relative_xy`` patch needs on the label side.  But the three experts were
trained with it **off**, so turning it on changes their input distribution: they
would always see zero horizontal tracking error.

That is a restriction to the subset of inputs where the expert is already doing
well in XY, so it ought to be benign -- but "ought to be" is not a measurement,
and the whole point of the distillation is that the teacher's action is correct.
This runs both arms from identical starts and compares what the teachers do.

Reported per arm: the max-body tracking error against the active reference over
the rollout, the fraction of environments that would trip v9's own 0.25 m
training termination, and the horizontal drift of the root away from where the
reference says it should be -- the quantity re-anchoring deliberately discards.

Usage::

    PYTHONPATH=. python data/scripts/check_realign_teacher_impact.py \\
      --checkpoint results/<run>/last.ckpt --headless --num-envs 256 --seconds 4
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
parser.add_argument("--out-dir", type=str, default=None)
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--terminate-m", type=float, default=0.25,
                    help="v9's training tracking-error termination threshold")
parser.add_argument("--driver", choices=("expert", "student"), default="expert",
                    help="expert: does re-anchoring hurt the teachers. student: "
                         "how much robot-vs-reference XY drift training actually "
                         "visits, which is what converts the per-metre label "
                         "sensitivity into an expected label noise.")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

args.headless = True
args.resolved_configs = "resolved_configs.pt"

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
log = logging.getLogger("check_realign_teacher_impact")


def main() -> int:
    torch.manual_seed(args.seed)
    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]
    device = env.device
    if not getattr(agent, "expert_actors", None):
        raise SystemExit("no frozen experts; this needs resolved_configs.pt")

    env_ids = torch.arange(env.num_envs, device=device)
    steps = int(round(args.seconds / float(env.dt)))
    agent.eval()

    # One reset, captured, replayed for both arms so the starts are identical.
    torch.manual_seed(args.seed)
    env.reset(env_ids, sample_flat=True)
    snapshot = env.save_state()
    motion_ids = env.motion_manager.motion_ids.clone()
    motion_times = env.motion_manager.motion_times.clone()

    results = {}
    for arm in (False, True):
        env.motion_manager.motion_ids.copy_(motion_ids)
        env.motion_manager.motion_times.copy_(motion_times)
        env.restore_state(snapshot)
        env.motion_manager.config.realign_motion_with_humanoid_on_each_step = arm

        env._current_context = env._build_global_context(
            env.simulator.get_robot_state()
        )
        env.compute_observations(context=env._current_context)
        agent.pre_collect_step(0)
        obs_td = agent.obs_dict_to_tensordict(
            agent.add_agent_info_to_obs(env.get_obs())
        )
        track = np.zeros((steps, env.num_envs), dtype=np.float32)
        drift = np.zeros((steps, env.num_envs), dtype=np.float32)
        with torch.no_grad():
            for step in range(steps):
                action = (
                    agent._collect_external_expert_action(obs_td)
                    if args.driver == "expert"
                    else agent.model.forward_inference(obs_td)["action"]
                )
                obs, *_ = env.step(action)
                agent.pre_collect_step(step + 1)
                obs_td = agent.obs_dict_to_tensordict(
                    agent.add_agent_info_to_obs(obs)
                )
                ctx = env._current_context
                ref = ctx.mimic.ref_state.rigid_body_pos
                cur = ctx.current.rigid_body_pos
                track[step] = (
                    (ref - cur).pow(2).sum(-1).sqrt().max(dim=-1)[0].cpu().numpy()
                )
                # Horizontal root drift from the reference. Under re-anchoring
                # this is zero by construction, which is the point: it is the
                # error the teacher no longer sees and no longer corrects.
                drift[step] = (
                    (ref[:, 0, :2] - cur[:, 0, :2]).norm(dim=-1).cpu().numpy()
                )
        peak = track.max(axis=0)
        results["realign" if arm else "baseline"] = dict(
            track_p50=float(np.median(track[-1])),
            track_p90=float(np.percentile(track[-1], 90)),
            peak_p50=float(np.median(peak)),
            peak_p90=float(np.percentile(peak, 90)),
            terminate_rate=float((peak > args.terminate_m).mean()),
            drift_p50=float(np.median(drift[-1])),
            drift_p90=float(np.percentile(drift[-1], 90)),
            drift_all_p50=float(np.median(drift)),
            drift_all_p90=float(np.percentile(drift, 90)),
            drift_all_p99=float(np.percentile(drift, 99)),
            track_trace=track.mean(axis=1).tolist(),
        )
        log.info("arm realign=%s done", arm)

    env.motion_manager.config.realign_motion_with_humanoid_on_each_step = False

    print(f"\n{env.num_envs} envs, {args.seconds:g} s driven by the "
          f"{args.driver}, identical starts\n")
    print(f"{'arm':<12}{'final track p50':>17}{'p90':>9}{'peak p50':>10}{'p90':>9}"
          f"{'would terminate':>17}{'root drift p50':>16}")
    for name in ("baseline", "realign"):
        r = results[name]
        print(f"{name:<12}{r['track_p50']:>17.4f}{r['track_p90']:>9.4f}"
              f"{r['peak_p50']:>10.4f}{r['peak_p90']:>9.4f}"
              f"{r['terminate_rate']:>16.3f} {r['drift_p50']:>15.4f}")

    b = results["baseline"]
    print(f"\nrobot-vs-reference horizontal drift, baseline arm, over every step: "
          f"p50 {b['drift_all_p50']:.4f}  p90 {b['drift_all_p90']:.4f}  "
          f"p99 {b['drift_all_p99']:.4f} m")
    delta = results["realign"]["peak_p50"] - results["baseline"]["peak_p50"]
    print(f"\npeak tracking error, realign - baseline (p50): {delta:+.4f} m")
    print("A negative number means re-anchoring makes the teachers track BETTER, "
          "which is what removing an error they cannot fix should do.")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "realign_teacher_impact.json").write_text(json.dumps(results, indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
