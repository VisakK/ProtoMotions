# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect state banks for bank-based resets (``bank_reset.BankResetEnv``).

Two modes (``notes/brainstorm_skill_graph.MD`` §4.2-4.3):

* ``node-hold`` -- roll a **node** policy (mean actions, pushes on by default so
  the distribution covers recovery wobble, reset noise on) and harvest settled
  hold states: max per-body error vs the frozen pose <= --max-err. These are the
  states an *edge leaving this node* actually starts from. Feeds v3 edge training.

* ``edge-arrival`` -- roll an **edge** policy under eval conditions (mean
  actions, no pushes, no reset noise -- matching eval_composition) and harvest
  the states of still-alive envs at each --switch-steps. These are the states a
  *target node* actually receives. Feeds catch training, and doubles as the
  arrival archive for a future capture classifier.

Banks store root_pos in clip-local frame (respawn_root_offset subtracted, which
round-trips exactly because ref_respawn_offset is folded into that offset).

Usage::

    python data/scripts/collect_state_bank.py --mode node-hold \
        --checkpoint results/node_downdog3/epoch_1000.ckpt \
        --out data/smpl/skill_graph_handstand/bank_node_downdog3_hold.pt

    python data/scripts/collect_state_bank.py --mode edge-arrival \
        --checkpoint results/edge_kickup_v3/final.ckpt \
        --switch-steps 263 285 307 \
        --out data/smpl/skill_graph_handstand/bank_arrivals_edge_kickup_v3.pt
"""
import argparse

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["node-hold", "edge-arrival"], required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--num-envs", type=int, default=1024)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--simulator", default="isaaclab")
p.add_argument("--out", required=True)
# node-hold options
p.add_argument("--steps", type=int, default=360, help="node-hold: rollout length")
p.add_argument("--settle-steps", type=int, default=120,
               help="node-hold: skip the first N steps (reset transient + settling)")
p.add_argument("--stride", type=int, default=10, help="node-hold: harvest every N steps")
p.add_argument("--max-err", type=float, default=None,
               help="alive filter: max per-body error vs reference "
                    "(default 0.30 node-hold, 0.50 edge-arrival)")
p.add_argument("--samples", type=int, default=8192, help="node-hold: target bank size")
p.add_argument("--max-rounds", type=int, default=5,
               help="node-hold: extra reset rounds if --samples not reached")
p.add_argument("--pushes", choices=["on", "off"], default=None,
               help="default: on for node-hold (recovery wobble), off for edge-arrival")
p.add_argument("--reset-noise", choices=["on", "off"], default=None,
               help="default: on for node-hold, off for edge-arrival")
# edge-arrival options
p.add_argument("--switch-steps", type=int, nargs="*", default=None)
p.add_argument("--bank-start-prob", type=float, default=None,
               help="edge-arrival: override the edge env's bank_prob "
                    "(default: leave as trained)")
args, _ = p.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch
AppLauncher = import_simulator_before_torch(args.simulator)

import hashlib  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import torch  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "examples/experiments/mimic"))
torch.manual_seed(args.seed)

if args.mode == "node-hold":
    max_err = args.max_err if args.max_err is not None else 0.30
    pushes = (args.pushes or "on") == "on"
    noise = (args.reset_noise or "on") == "on"
else:
    if not args.switch_steps:
        sys.exit("edge-arrival mode needs --switch-steps")
    max_err = args.max_err if args.max_err is not None else 0.50
    pushes = (args.pushes or "off") == "on"
    noise = (args.reset_noise or "off") == "on"

ck = Path(args.checkpoint)
c = torch.load(ck.parent / "resolved_configs.pt", map_location="cpu", weights_only=False)
robot_config = c["robot"]; simulator_config = c["simulator"]; terrain_config = c.get("terrain")
scene_lib_config = c["scene_lib"]; motion_lib_config = c["motion_lib"]
env_config = c["env"]; agent_config = c["agent"]

simulator_config.num_envs = args.num_envs
simulator_config.headless = True
if not pushes and getattr(simulator_config, "domain_randomization", None) is not None:
    simulator_config.domain_randomization.push = None
if not noise:
    robot_config.reset_noise = None
env_config.termination_components = {}
env_config.max_episode_length = 10_000_000
env_config.motion_manager.init_start_prob = 1.0
env_config.reward_components = {k: v for k, v in env_config.reward_components.items()
                                if k not in ("node_value_join_rew", "terminal_goal_rew")}
if hasattr(env_config, "bank_prob") and args.bank_start_prob is not None:
    env_config.bank_prob = args.bank_start_prob

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
from protomotions.simulator.base_simulator.simulator_state import ResetState  # noqa: E402
from bank_reset import BANK_KEYS  # noqa: E402

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


def policy_action(obs):
    with torch.no_grad():
        td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
        return agent.model(td)["mean_action"]


def ref_err(st):
    mm = env.motion_manager
    ref = env.motion_lib.get_motion_state(mm.motion_ids, mm.motion_times)
    rp = ref.rigid_body_pos.clone()
    rp += env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(rp)
    return (rp - st.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]


def harvest(mask, tag, buf):
    """Store clip-local reset states for the masked envs."""
    if not mask.any():
        return 0
    st = env.simulator.get_robot_state()
    rs = ResetState.from_robot_state(st)
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    local_root = rs.root_pos[ids] - env.respawn_root_offset[ids]
    buf["root_pos"].append(local_root.cpu())
    for k in BANK_KEYS[1:]:
        buf[k].append(getattr(rs, k)[ids].cpu())
    buf["tag"].append(torch.full((len(ids),), tag, dtype=torch.long))
    buf["err"].append(ref_err(st)[ids].cpu())
    return int(len(ids))


buf = {k: [] for k in BANK_KEYS}
buf["tag"], buf["err"] = [], []
collected = 0

if args.mode == "node-hold":
    for round_ in range(args.max_rounds):
        obs, _ = env.reset()
        for t in range(args.steps):
            obs, *_ = env.step(policy_action(obs))
            if t >= args.settle_steps and (t - args.settle_steps) % args.stride == 0:
                st = env.simulator.get_robot_state()
                mask = ref_err(st) <= max_err
                collected += harvest(mask, tag=t, buf=buf)
                if collected >= args.samples:
                    break
        print(f"round {round_}: collected {collected}/{args.samples}")
        if collected >= args.samples:
            break
else:  # edge-arrival
    obs, _ = env.reset()
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    switch_set = sorted(set(args.switch_steps))
    last = max(switch_set)
    for t in range(1, last + 1):
        obs, *_ = env.step(policy_action(obs))
        st = env.simulator.get_robot_state()
        alive &= ref_err(st) <= max_err
        if t in switch_set:
            got = harvest(alive, tag=t, buf=buf)
            collected += got
            print(f"switch {t}: alive {float(alive.float().mean()):.3f}, harvested {got}")

if collected == 0:
    sys.exit("collected 0 states -- filter too tight or policy broken; refusing to write a bank")

bank = {k: torch.cat(buf[k])[: args.samples] for k in BANK_KEYS}
tags = torch.cat(buf["tag"])[: args.samples]
errs = torch.cat(buf["err"])[: args.samples]
bank["meta"] = {
    "schema": "state_bank/v1",
    "mode": args.mode,
    "source_checkpoint": str(ck),
    "source_checkpoint_sha256": hashlib.sha256(open(ck, "rb").read()).hexdigest(),
    "motion_file": getattr(motion_lib_config, "motion_file", None),
    "filter_max_err": max_err, "pushes": pushes, "reset_noise": noise,
    "num_samples": int(bank["root_pos"].shape[0]),
    "num_envs": n, "seed": args.seed,
    "switch_steps": args.switch_steps,
    "frame": "clip-local: respawn_root_offset (incl. ref_respawn_offset z) subtracted",
    "err_mean": float(errs.mean()), "err_p95": float(torch.quantile(errs, 0.95)),
    "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
bank["tag"] = tags
bank["err"] = errs
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(bank, args.out)
print(f"wrote {args.out}: {bank['meta']['num_samples']} states "
      f"(err mean {bank['meta']['err_mean']:.3f}, p95 {bank['meta']['err_p95']:.3f})")

if hasattr(env.simulator, "shutdown"):
    env.simulator.shutdown()
