# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train/eval parity probe: a one-flip-at-a-time bisection lattice.

The open bug this closes (``notes/Skill_graph_handstand_lessons.MD`` §5.6):
``edge_kickup_join`` reports mean episode length **315/319** in training, but
``eval_composition.py`` measures only **0.32** survival to step 307 under
deterministic rollout with disturbances *off* -- which should be easier. Those
two numbers are jointly impossible under any shared-condition failure
distribution (with 68 % of envs failing before 307, the maximum achievable mean
episode length is ~311 < 315), so a train/eval condition difference is
guaranteed to exist. This script finds it by attribution instead of argument.

The eval changes several variables at once. Each cell of the lattice flips
exactly one of them away from the training configuration:

    cell 0  TRAIN     sampled actions, reset noise ON, pushes ON,
                      env terminations + auto-reset (the training loop pattern:
                      reset done envs, step with sampled action)
    cell 1  =0 but deterministic actions (mean_action)
    cell 2  =0 but reset noise OFF
    cell 3  =0 but pushes OFF
    cell 4  =0 but failure = manual sticky max-body 0.5 m, terminations off,
            single un-reset rollout (the eval_composition statistic)
    cell 5  EVAL      all four flips together (replicates eval_composition)

Every cell reports BOTH statistics where defined: the mean completed-episode
length (training's number) and first-episode sticky survival to step ``k``
(eval's number). IsaacLab cannot rebuild the sim in-process, so each cell is
one process; ``run_parity_probe.sh`` drives all six and ``--aggregate`` merges
the per-cell JSONs into ``parity_report.json``.

Success = attribution: cell 0 must reproduce training ep_len within ~5 %
(else the harness itself is broken -- also a finding), and the ranked per-flip
deltas should roughly add up to the cell0->cell5 gap.

Usage::

    python data/scripts/parity_probe.py --checkpoint results/edge_kickup_join/final.ckpt \
        --cell 0 --k 307 --num-envs 1024 --seed 0 --out-dir results/edge_kickup_join/parity
    ...
    python data/scripts/parity_probe.py --aggregate results/edge_kickup_join/parity
"""
import argparse

p = argparse.ArgumentParser()
p.add_argument("--checkpoint")
p.add_argument("--cell", type=int, choices=range(6))
p.add_argument("--k", type=int, default=307, help="Survival horizon (steps).")
p.add_argument("--horizon-mult", type=float, default=4.0,
               help="Cells 0-3 run horizon_mult * clip_steps to sample several episodes.")
p.add_argument("--num-envs", type=int, default=1024)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--simulator", default="isaaclab")
p.add_argument("--out-dir", default=None)
p.add_argument("--aggregate", default=None, metavar="DIR",
               help="Merge cell_*.json in DIR into parity_report.json and exit (no sim).")
args, _ = p.parse_known_args()

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

CELL_FLIPS = {
    0: {},
    1: {"deterministic": True},
    2: {"no_reset_noise": True},
    3: {"no_push": True},
    4: {"sticky": True},
    5: {"deterministic": True, "no_reset_noise": True, "no_push": True, "sticky": True},
}
CELL_NAMES = {
    0: "train (all training conditions)",
    1: "flip: deterministic actions",
    2: "flip: reset noise off",
    3: "flip: pushes off",
    4: "flip: sticky 0.5m failure, no terminations/resets",
    5: "eval (all four flips = eval_composition)",
}


def aggregate(dir_):
    dir_ = Path(dir_)
    cells = {}
    for f in sorted(dir_.glob("cell_*.json")):
        d = json.load(open(f))
        cells[d["cell"]] = d
    if 0 not in cells:
        sys.exit("aggregate: cell_0.json missing -- run the cells first")
    base = cells[0]
    report = {"schema": "parity_probe/v1",
              "checkpoint": base["checkpoint"], "k": base["k"],
              "num_envs": base["num_envs"], "seed": base["seed"],
              "cells": [], "attribution": []}
    for i in sorted(cells):
        c = cells[i]
        report["cells"].append({k_: c.get(k_) for k_ in (
            "cell", "name", "mean_ep_len", "num_completed_episodes",
            "first_ep_survival_to_k", "sticky_survival_to_k",
            "mean_first_fail_step", "failure_termination_frac")})
    surv = lambda c: (c.get("sticky_survival_to_k")  # noqa: E731
                      if c.get("sticky_survival_to_k") is not None
                      else c.get("first_ep_survival_to_k"))
    if 5 in cells:
        gap = surv(cells[0]) - surv(cells[5])
        for i in (1, 2, 3, 4):
            if i in cells:
                report["attribution"].append({
                    "flip": CELL_NAMES[i],
                    "delta_survival_vs_train": round(surv(cells[i]) - surv(cells[0]), 4),
                })
        report["total_gap_train_to_eval"] = round(gap, 4)
        report["attribution"].sort(key=lambda r: r["delta_survival_vs_train"])
    out = dir_ / "parity_report.json"
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if args.aggregate:
    aggregate(args.aggregate)
    sys.exit(0)

if not (args.checkpoint and args.cell is not None and args.out_dir):
    sys.exit("need --checkpoint, --cell and --out-dir (or --aggregate DIR)")

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402
AppLauncher = import_simulator_before_torch(args.simulator)

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "examples/experiments/mimic"))
torch.manual_seed(args.seed)

flips = CELL_FLIPS[args.cell]
ck = Path(args.checkpoint)
c = torch.load(ck.parent / "resolved_configs.pt", map_location="cpu", weights_only=False)
robot_config = c["robot"]; simulator_config = c["simulator"]; terrain_config = c.get("terrain")
scene_lib_config = c["scene_lib"]; motion_lib_config = c["motion_lib"]
env_config = c["env"]; agent_config = c["agent"]

simulator_config.num_envs = args.num_envs
simulator_config.headless = True
if flips.get("no_push") and getattr(simulator_config, "domain_randomization", None) is not None:
    simulator_config.domain_randomization.push = None
if flips.get("no_reset_noise"):
    robot_config.reset_noise = None
if flips.get("sticky"):
    env_config.termination_components = {}
    env_config.max_episode_length = 10_000_000
env_config.motion_manager.init_start_prob = 1.0
# The join reward loads a second critic per step for nothing here.
env_config.reward_components = {k: v for k, v in env_config.reward_components.items()
                                if k != "node_value_join_rew"}

from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
fabric = Fabric(**FabricConfig(accelerator="gpu", devices=1, num_nodes=1,
                               loggers=[], callbacks=[]).as_kwargs())
fabric.launch()
sim_extra = {"simulation_app": AppLauncher({"headless": True, "device": str(fabric.device)}).app}
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator  # noqa: E402
terrain_config, simulator_config = convert_friction_for_simulator(terrain_config, simulator_config)
from protomotions.utils.component_builder import build_all_components  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
from protomotions.agents.base_agent.agent import BaseAgent  # noqa: E402

comp = build_all_components(
    terrain_config=terrain_config, scene_lib_config=scene_lib_config,
    motion_lib_config=motion_lib_config, simulator_config=simulator_config,
    robot_config=robot_config, device=fabric.device, save_dir=None, **sim_extra)
env: BaseEnv = get_class(env_config._target_)(
    config=env_config, robot_config=robot_config, device=fabric.device,
    terrain=comp["terrain"], scene_lib=comp["scene_lib"],
    motion_lib=comp["motion_lib"], simulator=comp["simulator"])
agent: BaseAgent = get_class(agent_config._target_)(
    config=agent_config, env=env, fabric=fabric, root_dir=ck.parent)
agent.setup(); agent.load(str(ck), load_env=False, load_training_state=False); agent.eval()

n = args.num_envs
dt = env.dt
clip_steps = int(round(float(env.motion_lib.motion_lengths[0]) / dt))
K = args.k
action_key = "mean_action" if flips.get("deterministic") else "action"


def policy_action(obs):
    with torch.no_grad():
        td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
        return agent.model(td)[action_key]


result = {
    "schema": "parity_probe/v1", "cell": args.cell, "name": CELL_NAMES[args.cell],
    "flips": flips, "checkpoint": str(ck),
    "checkpoint_epoch": torch.load(str(ck), map_location="cpu",
                                   weights_only=False).get("epoch"),
    "k": K, "clip_steps": clip_steps, "num_envs": n, "seed": args.seed,
    "action_key": action_key,
    "mean_ep_len": None, "num_completed_episodes": None,
    "first_ep_survival_to_k": None, "sticky_survival_to_k": None,
    "mean_first_fail_step": None, "failure_termination_frac": None,
}

obs, _ = env.reset()

if flips.get("sticky"):
    # eval_composition's statistic: single un-reset rollout, sticky max-body
    # 0.5 m against the edge reference (spawn-offset corrected).
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    first_fail = torch.full((n,), K, dtype=torch.long, device=env.device)
    for t in range(K):
        obs, *_ = env.step(policy_action(obs))
        st = env.simulator.get_robot_state()
        mm = env.motion_manager
        ref = env.motion_lib.get_motion_state(mm.motion_ids, mm.motion_times)
        rp = ref.rigid_body_pos.clone()
        rp += env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(rp)
        err = (rp - st.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]
        newly = alive & (err > 0.5)
        first_fail[newly] = t
        alive &= err <= 0.5
    result["sticky_survival_to_k"] = float(alive.float().mean())
    failed = first_fail < K
    result["mean_first_fail_step"] = (float(first_fail[failed].float().mean())
                                      if failed.any() else None)
else:
    # The training-loop pattern (base_agent.agent lines ~685-720): reset done
    # envs at the top of each iteration, step with the chosen action key,
    # collect dones from env.step. Episode length = progress_buf at done.
    T = int(args.horizon_mult * clip_steps)
    ep_lens = []
    first_ep_len = torch.zeros(n, dtype=torch.long, device=env.device)
    first_ep_terminated = torch.zeros(n, dtype=torch.bool, device=env.device)
    first_ep_done = torch.zeros(n, dtype=torch.bool, device=env.device)
    done_indices = torch.tensor([], dtype=torch.long, device=env.device)
    for t in range(T):
        if len(done_indices) > 0:
            obs, _ = env.reset(done_indices)
        obs, rew, dones, terminated, extras = env.step(policy_action(obs))
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_indices) > 0:
            lens = env.progress_buf[done_indices]
            ep_lens.append(lens.clone())
            new_first = ~first_ep_done[done_indices]
            fidx = done_indices[new_first]
            first_ep_len[fidx] = lens[new_first]
            first_ep_terminated[fidx] = terminated[fidx]
            first_ep_done[fidx] = True
    if ep_lens:
        all_lens = torch.cat(ep_lens).float()
        result["mean_ep_len"] = float(all_lens.mean())
        result["num_completed_episodes"] = int(all_lens.numel())
        term_done = first_ep_terminated[first_ep_done].float()
        result["failure_termination_frac"] = float(term_done.mean()) if term_done.numel() else None
    done_mask = first_ep_done
    surv = (~done_mask) | (first_ep_len >= K)
    result["first_ep_survival_to_k"] = float(surv.float().mean())
    fail_steps = first_ep_len[(first_ep_len < K) & first_ep_done & first_ep_terminated]
    result["mean_first_fail_step"] = (float(fail_steps.float().mean())
                                      if fail_steps.numel() else None)

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"cell_{args.cell}.json"
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
print(f"wrote {out}")

if hasattr(env.simulator, "shutdown"):
    env.simulator.shutdown()
