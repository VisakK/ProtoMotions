# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion-quality probe: is the trained policy reproducing the reference's *motion*,
or just arriving somewhere the next node can catch?

Composition rate cannot tell these apart (see
``notes/Skill_graph_handstand_lessons_2.MD`` -- the v2 exit scored 1.000 while
moving its limbs at 1.83x the human's peak speed). This script measures the
speed profile, smoothness, tracking fidelity, terminal pose and leg chirality of
a policy against the reference it imitates, and emits both a human-readable
table and a machine-readable JSON consumed by ``eval_gate.py``.

History: this file originally lived only in a session scratchpad (stdout-only);
it was promoted into the repo 2026-07-29 with JSON output, seeding, checkpoint
hashing, per-env distributions, terminal/structural metrics and a zero-action
calibration mode. See ``notes/brainstorm_skill_graph.MD`` §4.1.

Calibration: run once with ``--action-mode zero`` per experiment. A zero-action
rollout must score terribly (tracking error > 0.5 m); if it does not, the
pipeline itself is broken and eval_gate will mark the whole report INVALID.

Usage::

    python data/scripts/probe_quality.py \
        --checkpoint results/edge_exit_v3/final.ckpt --steps 159 \
        --num-envs 64 --seed 0 --out results/edge_exit_v3/quality.json
"""
import argparse

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", "--edge-checkpoint", dest="checkpoint", required=True)
p.add_argument("--steps", type=int, required=True)
p.add_argument("--num-envs", type=int, default=64)
p.add_argument("--simulator", default="isaaclab")
p.add_argument("--action-mode", choices=["policy", "zero"], default="policy",
               help="'zero' is the pipeline-calibration row: it must FAIL tracking.")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--bank-start-prob", type=float, default=0.0,
               help="Override BankResetEnv bank_prob for this probe (default 0: "
                    "nominal reference starts, comparable across experiments).")
p.add_argument("--out", default=None, help="JSON output path.")
args, _ = p.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch
AppLauncher = import_simulator_before_torch(args.simulator)

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import torch  # noqa: E402
from pathlib import Path  # noqa: E402

# The frozen configs may pickle classes from the experiment dir (edge_terms,
# bank_reset, node_value_join) -- must be importable before torch.load.
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
IP, ILA, IRA = bn.index("Pelvis"), bn.index("L_Ankle"), bn.index("R_Ankle")
n = args.num_envs
num_actions = robot_config.number_of_actions


def ankle_lateral_sep(body_pos, root_rot):
    """Signed L-R ankle lateral separation in the pelvis heading frame [n]."""
    hq = calc_heading_quat_inv(root_rot, w_last=True)
    rel_l = quat_rotate(hq, body_pos[:, ILA] - body_pos[:, IP], w_last=True)
    rel_r = quat_rotate(hq, body_pos[:, IRA] - body_pos[:, IP], w_last=True)
    return rel_l[:, 1] - rel_r[:, 1]


obs, _ = env.reset()
# All per-step buffers are [T, n] so per-env distributions survive to the end.
pol_z, pol_vz, pol_spd, ref_z, ref_vz, ref_spd, err = [], [], [], [], [], [], []
acts = []
prev_act = None
last_st = last_rp = last_ref_rot = None
for t in range(args.steps):
    if args.action_mode == "zero":
        a = torch.zeros(n, num_actions, device=fabric.device)
    else:
        with torch.no_grad():
            td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            a = agent.model(td)["mean_action"]
    if prev_act is not None:
        acts.append((a - prev_act).abs().mean(dim=-1))
    prev_act = a.clone()
    obs, *_ = env.step(a)
    st = env.simulator.get_robot_state()
    mm = env.motion_manager
    r = env.motion_lib.get_motion_state(mm.motion_ids, mm.motion_times)
    rp = r.rigid_body_pos.clone()
    rp += env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(rp)
    pol_z.append(st.rigid_body_pos[:, IP, 2]);   ref_z.append(rp[:, IP, 2])
    pol_vz.append(st.rigid_body_vel[:, IP, 2]);  ref_vz.append(r.rigid_body_vel[:, IP, 2])
    pol_spd.append(st.rigid_body_vel.norm(dim=-1).mean(dim=-1))
    ref_spd.append(r.rigid_body_vel.norm(dim=-1).mean(dim=-1))
    err.append((rp - st.rigid_body_pos).norm(dim=-1).max(dim=-1)[0])
    last_st, last_rp, last_ref_rot = st, rp, r.root_rot

T = torch.stack  # [T, n]
pz, rz, pv, rv, ps, rs, er = map(T, [pol_z, ref_z, pol_vz, ref_vz, pol_spd, ref_spd, err])
da = T(acts)  # [T-1, n]
dt = env.dt


def q(x, p_):
    return float(torch.quantile(x.float(), p_))


# --- terminal metrics (final step) ---
term_per_body = (last_rp - last_st.rigid_body_pos).norm(dim=-1)          # [n, bodies]
term_max_body = term_per_body.max(dim=-1)[0]                             # [n]
term_body_means = term_per_body.mean(dim=0)                              # [bodies]
worst3 = torch.topk(term_body_means, 3)
sep_pol = ankle_lateral_sep(last_st.rigid_body_pos, last_st.root_rot)
sep_ref = ankle_lateral_sep(last_rp, last_ref_rot)
ankle_ok = ((sep_pol.sign() == sep_ref.sign()) | (sep_ref.abs() < 0.01)).float().mean()

# --- speed / smoothness / fidelity ---
env_mean = lambda X: X.mean(dim=1)   # noqa: E731  [T]
pzm, rzm = env_mean(pz), env_mean(rz)
peak_ratio_env = ps.max(dim=0)[0] / rs.max(dim=0)[0].clamp_min(1e-6)     # [n]
early = max(1, int(round(0.5 / dt)))


def drop_time(z):
    hi, lo = float(z.max()), float(z.min())
    a_, b_ = hi - 0.1 * (hi - lo), lo + 0.1 * (hi - lo)
    idx = [i for i, v in enumerate(z) if float(v) <= a_]
    jdx = [i for i, v in enumerate(z) if float(v) <= b_]
    return (jdx[0] - idx[0]) * dt if idx and jdx and jdx[0] > idx[0] else float("nan")


metrics = {
    "peak_pelvis_vz_ratio": float(pv.abs().max() / rv.abs().max().clamp_min(1e-6)),
    "peak_body_speed_ratio": float(ps.max() / rs.max().clamp_min(1e-6)),
    "peak_body_speed_ratio_p95": q(peak_ratio_env, 0.95),
    "mean_body_speed_ratio": float(ps.mean() / rs.mean().clamp_min(1e-6)),
    "descent_time_policy_s": drop_time(pzm),
    "descent_time_ref_s": drop_time(rzm),
    "mean_dact": float(da.mean()),
    "peak_dact": float(da.max()),
    "early_dact_ratio": float(da[:early].mean() / da[early:].mean().clamp_min(1e-8))
                        if da.shape[0] > early else float("nan"),
    "track_err_mean": float(er.mean()),
    "track_err_peak": float(env_mean(er).max()),
    "track_err_peak_time_s": float(env_mean(er).argmax()) * dt,
    "track_err_peak_p95": q(er.max(dim=0)[0], 0.95),
    "terminal_max_body_err_mean": float(term_max_body.mean()),
    "terminal_max_body_err_p95": q(term_max_body, 0.95),
    "terminal_worst_bodies": {bn[int(i)]: float(v) for v, i in zip(*worst3)},
    "ankle_order_correct_frac": float(ankle_ok),
    "ankle_sep_policy_mean": float(sep_pol.mean()),
    "ankle_sep_ref_mean": float(sep_ref.mean()),
    "step0_excess_speed": float(ps[0].mean() - rs[0].mean()),
}
metrics["descent_time_ratio"] = (
    metrics["descent_time_policy_s"] / metrics["descent_time_ref_s"]
    if metrics["descent_time_ref_s"] == metrics["descent_time_ref_s"]
    and metrics["descent_time_ref_s"] > 0 else float("nan"))

# --- stdout table (unchanged format, env-means) ---
psm, rsm, erm = env_mean(ps), env_mean(rs), env_mean(er)
pvm, rvm = env_mean(pv), env_mean(rv)
print(f"\n=== motion quality: {ck.parent.name}, {args.steps} steps "
      f"({args.steps * dt:.2f} s), mode={args.action_mode}, seed={args.seed} ===")
print(f"{'t(s)':>6} {'polPelvZ':>9} {'refPelvZ':>9} {'polVz':>8} {'refVz':>8} "
      f"{'polSpd':>8} {'refSpd':>8} {'maxErr':>8}")
for i in range(0, args.steps, max(args.steps // 20, 1)):
    print(f"{i * dt:6.2f} {pzm[i]:9.3f} {rzm[i]:9.3f} {pvm[i]:8.3f} {rvm[i]:8.3f} "
          f"{psm[i]:8.3f} {rsm[i]:8.3f} {erm[i]:8.3f}")
print()
for k_, v_ in metrics.items():
    print(f"{k_:32s}: {v_}")

# --- machine-readable artifact ---
if args.out:
    sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
    try:
        epoch = torch.load(str(ck), map_location="cpu", weights_only=False).get("epoch")
    except Exception:
        epoch = None
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git = None
    stride = max(args.steps // 40, 1)
    out = {
        "schema": "probe_quality/v1",
        "experiment": ck.parent.name,
        "checkpoint": str(ck), "checkpoint_sha256": sha, "checkpoint_epoch": epoch,
        "git_commit": git,
        "num_envs": n, "steps": args.steps, "dt": dt,
        "action_mode": args.action_mode, "seed": args.seed,
        "bank_start_prob": args.bank_start_prob,
        "metrics": metrics,
        "series": {
            "t_s": [i * dt for i in range(0, args.steps, stride)],
            "pol_pelvis_z": [float(pzm[i]) for i in range(0, args.steps, stride)],
            "ref_pelvis_z": [float(rzm[i]) for i in range(0, args.steps, stride)],
            "pol_spd": [float(psm[i]) for i in range(0, args.steps, stride)],
            "ref_spd": [float(rsm[i]) for i in range(0, args.steps, stride)],
            "max_err": [float(erm[i]) for i in range(0, args.steps, stride)],
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")

if hasattr(env.simulator, "shutdown"):
    env.simulator.shutdown()
