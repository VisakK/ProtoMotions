# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Does anything the student sees still depend on where it is on the floor?

``Student_v9_input_frames.MD`` audited this at the level of the observation
*kernels* and found two leaks -- the sparse pose goals and the privileged dense
future both carried the reference's floor placement -- then fixed both behind
``root_relative_xy``.  That audit ran the kernels on synthetic tensors.  This
runs the **whole environment**: build it from a checkpoint, capture every
observation the student's three networks consume, perturb the world, rebuild the
context through the real component graph, and diff.  It therefore also covers
what a kernel test cannot -- context-view bindings, terrain sampling, cached
state, the contact stack, and whatever the experiment file actually wired up.

Three translations, each isolating a different invariance:

* ``ref_shift``   -- move the **reference** (``respawn_root_offset``), robot
  fixed. Original v9 fails this: the goal is an error to a fixed floor point.
* ``robot_shift`` -- move the **robot**, reference fixed. Original v9 fails this
  too, and it is the one that matters at deployment, where a policy that has
  drifted must not be told to walk back.
* ``both_shift``  -- move both together. v9 already passes; a failure here would
  mean an absolute world coordinate, not a relative one.

Every observation key is classified by which network consumes it, read off the
loaded model rather than assumed, so a key that is merely *present* in the
environment is not reported as a student leak.

The **label** side is measured too, and it is the part the flag does not fix:
the frozen teachers keep ``expert_mimic_target_poses`` with its original floor
anchor, so two physically identical student states at different floor positions
can receive different expert actions.  ``--realign`` re-anchors the reference to
the robot before the context is rebuilt -- exactly what
``realign_motion_with_humanoid_on_each_step`` does inside ``env.step`` -- so the
run also reports whether that one flag closes the label gap and the
world-anchored tracking termination along with it.

Usage::

    PYTHONPATH=. python data/scripts/check_student_xy_invariance.py \\
      --checkpoint results/<run>/last.ckpt --headless --num-envs 64 \\
      --experiment-overrides masked_mimic_target_poses.root_relative_xy=True \\
                             mimic_target_poses.root_relative_xy=True
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
parser.add_argument("--delta", type=float, nargs="+", default=(1.7, -2.3),
                    help="horizontal translation(s) in metres, as x y [x y ...]. "
                         "Several pairs run the whole check once per delta on one "
                         "build, which is how a terrain-grid quantization artifact "
                         "is told apart from a real world anchor: a systematic "
                         "anchor leaks at every delta, a grid artifact does not.")
parser.add_argument("--settle-steps", type=int, default=30,
                    help="steps under the policy before the check, so contact "
                         "forces and history are populated rather than post-reset")
parser.add_argument("--tol", type=float, default=1e-4)
parser.add_argument("--realign", action="store_true",
                    help="re-anchor the reference to the robot before rebuilding "
                         "the context (what realign_motion_with_humanoid_on_each_step does)")
parser.add_argument("--student-relative-xy", type=lambda s: s.lower() != "false",
                    default=None,
                    help="force the flag on/off on the two student pose components; "
                         "default leaves the saved config alone")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

args.headless = True
args.resolved_configs = "resolved_configs.pt"      # the frozen teachers live here

from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

import logging  # noqa: E402

import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from policy_setup import build  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("check_student_xy_invariance")


def consumer_map(agent) -> dict:
    """``key -> {'P','E','D','expert'}``, read off the loaded model."""
    model = agent.model
    roles: dict = {}

    def add(keys, tag):
        for key in keys or []:
            roles.setdefault(str(key), set()).add(tag)

    add(getattr(model, "_prior", None) and model._prior.in_keys, "P")
    add(getattr(model, "_encoder", None) and model._encoder.in_keys, "E")
    produced = {"vae_latent", "fsq_chunk_phase"}
    trunk = getattr(model, "_trunk", None)
    add([k for k in (trunk.in_keys if trunk else []) if str(k) not in produced], "D")
    add([f"expert_{k}" for k in getattr(agent, "expert_actor_in_keys", []) or []],
        "expert")
    return {k: "".join(sorted(v)) for k, v in roles.items()}


def snapshot_obs(env, agent):
    env._current_context = env._build_global_context(env.simulator.get_robot_state())
    env.compute_observations(context=env._current_context)
    obs = agent.add_agent_info_to_obs(env.get_obs())
    return {k: v.detach().clone() for k, v in obs.items() if torch.is_tensor(v)}


def tracking_error(env) -> torch.Tensor:
    ctx = env._current_context
    ref = ctx.mimic.ref_state.rigid_body_pos
    cur = ctx.current.rigid_body_pos
    return (ref - cur).pow(2).sum(-1).sqrt().max(dim=-1)[0]


def main() -> int:
    torch.manual_seed(args.seed)
    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]
    device = env.device
    env_ids = torch.arange(env.num_envs, device=device)
    assert len(args.delta) % 2 == 0, "--delta takes x y pairs"
    deltas = [
        torch.tensor([args.delta[i], args.delta[i + 1], 0.0], device=device)
        for i in range(0, len(args.delta), 2)
    ]

    if args.student_relative_xy is not None:
        for key in ("masked_mimic_target_poses", "mimic_target_poses"):
            component = env.config.observation_components.get(key)
            if component is not None:
                component.static_params["root_relative_xy"] = bool(
                    args.student_relative_xy
                )
                log.info("forced %s.root_relative_xy=%s", key, args.student_relative_xy)

    flags = {}
    for key in ("masked_mimic_target_poses", "mimic_target_poses",
                "expert_mimic_target_poses"):
        component = env.config.observation_components.get(key)
        if component is not None:
            flags[key] = bool(component.static_params.get("root_relative_xy", False))
    log.info("root_relative_xy: %s", flags)
    log.info("realign_motion_with_humanoid_on_each_step: %s",
             env.motion_manager.config.realign_motion_with_humanoid_on_each_step)

    roles = consumer_map(agent)
    agent.eval()

    obs, _ = env.reset(env_ids, sample_flat=True)
    agent.pre_collect_step(0)
    obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
    with torch.no_grad():
        for step in range(args.settle_steps):
            action = agent.model.forward_inference(obs_td)["action"]
            obs, *_ = env.step(action)
            agent.pre_collect_step(step + 1)
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

    from protomotions.simulator.base_simulator.simulator_state import ResetState

    def rebuild():
        if args.realign:
            env.align_motion_with_humanoid(
                env_ids, env.simulator.get_root_state().root_pos
            )
        return snapshot_obs(env, agent)

    def expert_action(observations):
        td = agent.obs_dict_to_tensordict(observations)
        with torch.no_grad():
            return agent._collect_external_expert_action(td).detach().clone()

    base_obs = rebuild()
    base_expert = expert_action(base_obs)
    base_track = tracking_error(env).clone()
    base_offset = env.respawn_root_offset.clone()
    base_state = env.simulator.get_robot_state()
    base_reset = ResetState.from_robot_state(base_state)
    base_root = base_state.root_pos.clone()

    def restore():
        env.respawn_root_offset.copy_(base_offset)
        env.simulator.reset_envs(ResetState.from_robot_state(base_state), None, env_ids)

    def move_robot(delta):
        moved = ResetState.from_robot_state(base_state)
        moved.root_pos = moved.root_pos + delta
        env.simulator.reset_envs(moved, None, env_ids)

    all_reports = []
    for delta in deltas:
        cases = {}
        # --- reference moves, robot fixed ------------------------------ #
        env.respawn_root_offset[:, :2] += delta[:2]
        cases["ref_shift"] = rebuild()
        cases["ref_shift_expert"] = expert_action(cases["ref_shift"])
        cases["ref_shift_track"] = tracking_error(env).clone()
        restore()

        # --- robot moves, reference fixed ------------------------------ #
        move_robot(delta)
        cases["robot_shift"] = rebuild()
        cases["robot_shift_expert"] = expert_action(cases["robot_shift"])
        cases["robot_shift_track"] = tracking_error(env).clone()
        restore()

        # --- both move together ---------------------------------------- #
        env.respawn_root_offset[:, :2] += delta[:2]
        move_robot(delta)
        cases["both_shift"] = rebuild()
        cases["both_shift_expert"] = expert_action(cases["both_shift"])
        cases["both_shift_track"] = tracking_error(env).clone()
        restore()

        order = ("ref_shift", "robot_shift", "both_shift")
        keys = sorted(base_obs, key=lambda k: (roles.get(k, "zz"), k))
        report = {"flags": flags, "delta": [float(delta[0]), float(delta[1])],
                  "realign": args.realign, "num_envs": int(env.num_envs),
                  "tol": args.tol, "keys": []}

        print(f"\n=== translation ({float(delta[0]):+.4f}, {float(delta[1]):+.4f}) m ==="
              f"   {env.num_envs} envs, tolerance {args.tol:g}\n")
        print(f"{'observation key':<30}{'uses':>7}{'width':>6}"
              + "".join(f"{c + ' (max,%chg)':>19}" for c in order))
        worst_student = 0.0
        for key in keys:
            role = roles.get(key, "-")
            width = int(base_obs[key].reshape(env.num_envs, -1).shape[1])
            diffs = []
            for case in order:
                other = cases[case].get(key)
                if other is None or other.shape != base_obs[key].shape:
                    diffs.append((float("nan"), float("nan"), float("nan")))
                    continue
                gap = (other.float() - base_obs[key].float()).abs()
                diffs.append((float(gap.max()), float(gap.mean()),
                              float((gap > args.tol).float().mean())))
            if role != "-" and "expert" not in role:
                worst_student = max(
                    worst_student, max(d[0] for d in diffs if d[0] == d[0])
                )
            marks = "".join(
                f"{('OK' if d[0] <= args.tol else 'LEAK') + f' {d[0]:.1e}':>13}"
                f"{d[2] * 100:>6.1f}%"
                for d in diffs
            )
            print(f"{key[:29]:<30}{role:>7}{width:>6}{marks}")
            report["keys"].append(dict(
                key=key, role=role, width=width,
                **{c: dict(max=d[0], mean=d[1], frac_changed=d[2])
                   for c, d in zip(order, diffs)}))

        print(f"\nworst deviation over all STUDENT-consumed keys: {worst_student:.3e}"
              f"  -> {'PASS' if worst_student <= args.tol else 'LEAK'}")
        print(f"\n{'label / termination side':<32}"
              + "".join(f"{c:>14}" for c in order))
        # The expert-action delta is the label noise a translation-invariant
        # student cannot explain, so report its distribution -- a max over
        # 69 dims x num_envs is an extreme-value statistic, not a typical one --
        # and normalise by the action's own spread across environments, which is
        # the scale the imitation MSE actually works against.
        scale = float(base_expert.std(dim=0).mean().clamp(min=1e-9))
        for label, key, base, stats in (
            ("expert action |d| p50/p90/max", "_expert", base_expert, True),
            ("  ... as a fraction of its sd", "_expert", base_expert, "norm"),
            ("tracking error (max abs)", "_track", base_track, False),
        ):
            row = []
            for case in order:
                gap = (cases[case + key] - base).abs()
                if stats is True:
                    row.append((float(gap.median()), float(gap.quantile(0.9)),
                                float(gap.max())))
                elif stats == "norm":
                    row.append((float(gap.median()) / scale,
                                float(gap.quantile(0.9)) / scale,
                                float(gap.max()) / scale))
                else:
                    row.append((float(gap.max()),))
            if len(row[0]) == 3:
                print(f"{label:<32}"
                      + "".join(f"{v[0]:>5.3f}/{v[1]:.3f}/{v[2]:.3f}".rjust(14)
                                for v in row))
            else:
                print(f"{label:<32}" + "".join(f"{v[0]:>14.3e}" for v in row))
            report[label] = {c: list(v) for c, v in zip(order, row)}
        report["expert_action_sd"] = scale
        report["worst_student_deviation"] = worst_student
        all_reports.append(report)

    worst_student = max(r["worst_student_deviation"] for r in all_reports)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"xy_invariance{'_realign' if args.realign else ''}.json").write_text(
            json.dumps(all_reports, indent=1)
        )
        print(f"\nwrote {out}")
    return 0 if worst_student <= args.tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
