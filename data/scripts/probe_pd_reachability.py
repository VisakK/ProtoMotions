# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PD-reachability probe: how much tracking error is the PLANT, not the policy?

Drives the PD targets directly with the reference joint angles (inverse of the
tanh action mapping: action = atanh((q_ref - offset)/scale)), no policy at all.
Whatever error remains is what the actuation model (gains vs gravity, morphology,
contact) cannot do — the floor beneath any learned policy on this clip. Run the
same script with --action-mode policy for the side-by-side per-body breakdown.

Also decomposes per-body error in the REFERENCE pelvis heading frame, splitting
the lateral (y) component for the spine chain and the legs — a direct measure of
"the torso looks like scoliosis" and "the legs are in the wrong pose".

Usage::

    python data/scripts/probe_pd_reachability.py \
        --checkpoint results/edge_kickup_v3/final.ckpt --steps 320 \
        --action-mode ref-pd --window 4 7 --out results/edge_kickup_v3/pd_reach.json
"""
import argparse

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, required=True)
p.add_argument("--num-envs", type=int, default=64)
p.add_argument("--simulator", default="isaaclab")
p.add_argument("--action-mode", choices=["ref-pd", "policy"], default="ref-pd")
p.add_argument("--window", type=float, nargs=2, default=[4.0, 7.0],
               help="Mid-clip analysis window [s] for the per-body breakdown.")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None)
args, _ = p.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch
AppLauncher = import_simulator_before_torch(args.simulator)

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import torch  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "examples/experiments/mimic"))
torch.manual_seed(args.seed)

ck = Path(args.checkpoint)
c = torch.load(ck.parent / "resolved_configs.pt", map_location="cpu", weights_only=False)
robot_config = c["robot"]; simulator_config = c["simulator"]; terrain_config = c.get("terrain")
scene_lib_config = c["scene_lib"]; motion_lib_config = c["motion_lib"]
env_config = c["env"]; agent_config = c["agent"]

simulator_config.num_envs = args.num_envs
simulator_config.headless = True
if getattr(simulator_config, "domain_randomization", None) is not None:
    simulator_config.domain_randomization.push = None
robot_config.reset_noise = None
env_config.termination_components = {}
env_config.max_episode_length = 10_000_000
env_config.motion_manager.init_start_prob = 1.0
env_config.reward_components = {k: v for k, v in env_config.reward_components.items()
                                if k not in ("node_value_join_rew", "terminal_goal_rew")}
if hasattr(env_config, "bank_prob"):
    env_config.bank_prob = 0.0

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
from protomotions.utils.rotations import calc_heading_quat_inv, quat_rotate  # noqa: E402

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

bn = list(robot_config.kinematic_info.body_names)
n, dt = args.num_envs, env.dt
SPINE = [b for b in ("Torso", "Spine", "Chest", "Neck", "Head") if b in bn]
LEGS = [b for b in ("L_Hip", "L_Knee", "L_Ankle", "L_Toe",
                    "R_Hip", "R_Knee", "R_Ankle", "R_Toe") if b in bn]

# Inverse of the tanh PD mapping (action_functions.normalized_pd_fixed_gains_action)
ac = env.config.action_config
pd_offset = ac["pd_action_offset"].to(fabric.device)
pd_scale = ac["pd_action_scale"].to(fabric.device)


def ref_pd_action(t_next):
    mm = env.motion_manager
    ref = env.motion_lib.get_motion_state(mm.motion_ids, (mm.motion_times + t_next).clamp(max=float(env.motion_lib.motion_lengths[0]) - 1e-4))
    x = ((ref.dof_pos - pd_offset) / pd_scale).clamp(-0.999, 0.999)
    return torch.atanh(x)


obs, _ = env.reset()
IP = bn.index("Pelvis")
w0, w1 = args.window
err_series = []
win_vec = torch.zeros(len(bn), 3, device=fabric.device)  # heading-frame error sum in window
win_abs = torch.zeros(len(bn), device=fabric.device)
win_count = 0
for t in range(args.steps):
    if args.action_mode == "ref-pd":
        a = ref_pd_action(dt)
    else:
        with torch.no_grad():
            td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            a = agent.model(td)["mean_action"]
    obs, *_ = env.step(a)
    st = env.simulator.get_robot_state()
    mm = env.motion_manager
    r = env.motion_lib.get_motion_state(mm.motion_ids, mm.motion_times)
    rp = r.rigid_body_pos.clone()
    rp += env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(rp)
    e = rp - st.rigid_body_pos                                   # [n, B, 3]
    err_series.append(e.norm(dim=-1).max(dim=-1)[0].mean().item())
    if w0 <= t * dt <= w1:
        hq = calc_heading_quat_inv(r.root_rot, w_last=True)      # ref heading frame
        ehf = quat_rotate(hq.unsqueeze(1).expand(-1, len(bn), -1).reshape(-1, 4),
                          e.reshape(-1, 3), w_last=True).reshape(n, len(bn), 3)
        win_vec += ehf.mean(dim=0)
        win_abs += e.norm(dim=-1).mean(dim=0)
        win_count += 1

win_vec /= max(win_count, 1)
win_abs /= max(win_count, 1)

series = torch.tensor(err_series)
result = {
    "schema": "pd_reachability/v1",
    "checkpoint": str(ck), "action_mode": args.action_mode,
    "num_envs": n, "steps": args.steps, "seed": args.seed,
    "window_s": [w0, w1],
    "track_err_mean": float(series.mean()),
    "track_err_at": {f"{i*dt:.1f}s": round(float(series[i]), 3)
                     for i in range(0, args.steps, max(args.steps // 16, 1))},
    "window_per_body_err": {bn[i]: round(float(win_abs[i]), 3)
                            for i in torch.argsort(win_abs, descending=True)[:10]},
    "window_spine_lateral_err": {b: round(float(win_vec[bn.index(b), 1]), 3) for b in SPINE},
    "window_legs_err": {b: round(float(win_abs[bn.index(b)]), 3) for b in LEGS},
    "window_legs_lateral_err": {b: round(float(win_vec[bn.index(b), 1]), 3) for b in LEGS},
    "window_legs_vertical_err": {b: round(float(win_vec[bn.index(b), 2]), 3) for b in LEGS},
}
print(json.dumps(result, indent=2))
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")

if hasattr(env.simulator, "shutdown"):
    env.simulator.shutdown()
