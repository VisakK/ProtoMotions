# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Estimate a skill-graph **node**'s stability region from its value function.

The question
------------
A node policy holds one pose.  For the graph to be usable, an edge has to know
*where it must deliver the character* for the next node to catch it.  That
target is the node's **region of attraction**: the set of states from which the
node policy recovers the pose instead of falling out of it.

Directly characterising that set in a 1027-dimensional observation space is
hopeless.  But PPO already trains a scalar summary of exactly this: the critic.
V(s) is the expected discounted return from s, and under a reward that pays for
holding the pose and an episode that terminates on losing it, V(s) is large
precisely when the policy expects to keep holding.  So the hypothesis is that a
single threshold on V separates recoverable from unrecoverable states::

    hold(s)  <=>  V(s) > V*

This is only well posed because node references are **frozen** (see
``data/scripts/build_skill_graph_clips.py``).  With a constant reference the
``mimic_target_poses`` half of the observation is constant, so V is a function
of the physical state alone -- not of "where am I in the clip".

Method
------
1. Reset a large batch of environments from a **widened** initial-state
   distribution (RSI noise well above the training level).
2. Read V(s_0) from the critic before any action is taken.
3. Roll the **deterministic** policy (``mean_action``) forward ``--horizon``
   control steps with the training termination active (max per-body tracking
   error > 0.5 m).  Label 1 if the node never terminated, 0 otherwise.
4. Fit a 1-D logistic regression on ``(V(s_0), label)``.  Its decision boundary
   is V*.

Two perturbation families are swept, and this separation is the point of the
experiment:

* ``velocity`` (**primary**, used to fit V*) -- configuration noise stays at the
  training level; only DOF / root linear / root angular **velocities** are
  scaled up.  A velocity kick cannot create ground penetration, so the widened
  distribution stays physically clean, and it is the same disturbance channel
  the push randomisation applies during training.
* ``mixed`` (**held out**, used to test V*) -- configuration noise (DOF pose,
  root position and orientation) is scaled too, with caps.  Root rotation noise
  pivots the whole body about the pelvis, which for an inverted pose swings the
  support hands through the floor, so it is capped hard (see ``CAP_*``).

Fitting on one family and testing the threshold on the other is what
distinguishes "V is a stability certificate" from "V just encodes how big the
noise was".

Mid-episode push randomisation is **disabled** here on purpose.  With pushes on,
the label would depend on unobserved future disturbances rather than on s_0
alone, which is not the question being asked.

Usage::

    python data/scripts/node_stability_region.py \
        --checkpoint results/node_handstand/last.ckpt \
        --simulator isaaclab --num-envs 2048 --horizon 300 \
        --out results/node_handstand/stability_region.json
"""


def create_parser():
    import argparse

    p = argparse.ArgumentParser(description="Node stability region from V(s).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--simulator", default="isaaclab")
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument(
        "--horizon",
        type=int,
        default=300,
        help="Control steps a rollout must survive to count as a hold "
        "(300 steps @ 30 Hz = 10 s).",
    )
    p.add_argument(
        "--scales",
        type=float,
        nargs="*",
        default=[0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 24.0, 32.0, 40.0],
        help="Multipliers on the training reset noise. The range must be wide "
        "enough to produce BOTH holds and failures -- a converged node is "
        "hard to break, and degenerate labels leave no boundary to fit.",
    )
    p.add_argument("--repeats", type=int, default=1, help="Batches per scale.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    p.add_argument("--seed", type=int, default=0)
    return p


import argparse  # noqa: E402

parser = create_parser()
args, _unknown = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger("node_stability")


# --------------------------------------------------------------------------- #
# Perturbation families.
#
# Base magnitudes mirror mlp_node_small.RESET_NOISE (the training level), so
# scale = 1.0 reproduces training and scale > 1 widens the distribution.
#
# Caps exist because reset noise perturbs the *root* pose and the simulator does
# FK from there: rotating the root by theta swings a body at lever L by L*theta.
# In a handstand the hands sit ~1.09 m from the pelvis, so 0.03 rad already
# drives them ~3 cm -- past the +1.0 cm spawn clearance and into the floor.
# Penetration triggers PhysX depenetration, which is a sim artifact rather than
# a disturbance, so configuration noise is capped and velocity noise (which
# cannot penetrate anything) carries the sweep.
# --------------------------------------------------------------------------- #
BASE = dict(
    dof_pos_noise=0.02,
    dof_vel_noise=0.15,
    root_pos_xy_noise=0.010,
    root_pos_z_noise=0.003,
    root_rot_noise=0.015,
    root_vel_noise=0.05,
    root_ang_vel_noise=0.15,
)
CAP_CONFIG_SCALE = 4.0   # dof_pos / root_pos_xy never exceed 4x training
CAP_ROT_SCALE = 1.5      # root_rot never exceeds 1.5x training (~0.022 rad)


def noise_for(scale: float, family: str):
    """Build a RobotNoiseConfig kwargs dict for a sweep point."""
    from protomotions.simulator.base_simulator.config import RobotNoiseConfig

    if family in ("velocity", "push_on"):
        # "push_on" is the control batch: training-level noise everywhere, the
        # only difference being that mid-episode push is re-enabled.
        cs = 1.0  # configuration noise pinned at the training level
        rs = 1.0
        vs = scale
    elif family == "mixed":
        cs = min(scale, CAP_CONFIG_SCALE)
        rs = min(scale, CAP_ROT_SCALE)
        vs = scale
    else:
        raise ValueError(family)

    return RobotNoiseConfig(
        dof_pos_noise=BASE["dof_pos_noise"] * cs,
        dof_vel_noise=BASE["dof_vel_noise"] * vs,
        root_pos_noise=[
            BASE["root_pos_xy_noise"] * cs,
            BASE["root_pos_xy_noise"] * cs,
            BASE["root_pos_z_noise"],  # never scaled: penetration guard
        ],
        root_rot_noise=[BASE["root_rot_noise"] * rs] * 3,
        root_vel_noise=BASE["root_vel_noise"] * vs,
        root_ang_vel_noise=BASE["root_ang_vel_noise"] * vs,
    )


# --------------------------------------------------------------------------- #
# 1-D logistic regression + metrics (torch only; no sklearn dependency).
# --------------------------------------------------------------------------- #
def fit_logistic_1d(x: np.ndarray, y: np.ndarray, iters: int = 4000, lr: float = 0.05):
    """Fit sigmoid(w*x + b) to binary y. Returns (w, b, threshold)."""
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    # standardise for conditioning, then map coefficients back
    mu, sd = xt.mean(), xt.std().clamp_min(1e-9)
    xs = (xt - mu) / sd

    w = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        logits = w * xs + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss.backward()
        opt.step()

    w_s, b_s = float(w.item()), float(b.item())
    # logits = w_s*(x-mu)/sd + b_s  =>  w = w_s/sd, b = b_s - w_s*mu/sd
    w_x = w_s / float(sd)
    b_x = b_s - w_s * float(mu) / float(sd)
    thresh = -b_x / w_x if abs(w_x) > 1e-12 else float("nan")
    return w_x, b_x, thresh, float(loss.item())


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), ties averaged."""
    pos, neg = labels == 1, labels == 0
    n_p, n_n = pos.sum(), neg.sum()
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, thresh: float) -> dict:
    pred = (scores > thresh).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    n = len(labels)
    return {
        "threshold": float(thresh),
        "accuracy": (tp + tn) / n if n else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        # A conservative threshold is what an edge planner actually wants: the
        # V above which essentially nothing fails.
        "false_hold_rate": fp / (tp + fp) if (tp + fp) else float("nan"),
    }


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Non-parametric cross-check: threshold maximising (TPR - FPR)."""
    cands = np.unique(scores)
    if len(cands) > 2000:
        cands = np.quantile(scores, np.linspace(0, 1, 2000))
    best, best_j = float("nan"), -np.inf
    n_p = max((labels == 1).sum(), 1)
    n_n = max((labels == 0).sum(), 1)
    for t in cands:
        pred = scores > t
        tpr = ((pred) & (labels == 1)).sum() / n_p
        fpr = ((pred) & (labels == 0)).sum() / n_n
        if tpr - fpr > best_j:
            best_j, best = tpr - fpr, float(t)
    return best


def conservative_threshold(scores: np.ndarray, labels: np.ndarray,
                           target_success: float = 0.99) -> float:
    """Smallest V above which the empirical hold rate is >= target_success."""
    order = np.argsort(scores)
    s, l = scores[order], labels[order]
    # suffix success rate
    n = len(s)
    suffix = np.cumsum(l[::-1])[::-1] / np.arange(n, 0, -1)
    ok = np.where(suffix >= target_success)[0]
    return float(s[ok[0]]) if len(ok) else float("nan")


def main():
    global args
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = Path(args.checkpoint)
    # Load the TRAINING configs, not resolved_configs_inference.pt: the
    # inference hook strips terminations, and without a failure signal there is
    # no boundary to find.
    cfg_path = ckpt.parent / "resolved_configs.pt"
    assert cfg_path.exists(), f"missing {cfg_path}"
    cfgs = torch.load(cfg_path, map_location="cpu", weights_only=False)

    robot_config = cfgs["robot"]
    simulator_config = cfgs["simulator"]
    terrain_config = cfgs.get("terrain")
    scene_lib_config = cfgs["scene_lib"]
    motion_lib_config = cfgs["motion_lib"]
    env_config = cfgs["env"]
    agent_config = cfgs["agent"]

    assert env_config.termination_components, "expected tracking_error termination"

    simulator_config.num_envs = args.num_envs
    simulator_config.headless = True
    # Labels must depend on s_0 alone -> no mid-episode shoves. The training
    # push config is kept so it can be switched back on for the control batch
    # at the end (see "push_control").
    orig_push = None
    if getattr(simulator_config, "domain_randomization", None) is not None:
        orig_push = simulator_config.domain_randomization.push
        simulator_config.domain_randomization.push = None
    # Never let the max-length reset fire inside the horizon.
    env_config.max_episode_length = args.horizon + 10
    env_config.motion_manager.init_start_prob = 1.0

    from protomotions.utils.fabric_config import FabricConfig
    from lightning.fabric import Fabric

    fabric = Fabric(**FabricConfig(accelerator="gpu", devices=1, num_nodes=1,
                                   loggers=[], callbacks=[]).as_kwargs())
    fabric.launch()

    sim_extra = {}
    if args.simulator == "isaaclab":
        sim_extra["simulation_app"] = AppLauncher(
            {"headless": True, "device": str(fabric.device)}
        ).app

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
        **sim_extra,
    )

    from protomotions.utils.hydra_replacement import get_class
    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    from protomotions.agents.base_agent.agent import BaseAgent

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, root_dir=ckpt.parent
    )
    agent.setup()
    agent.load(str(ckpt), load_env=False, load_training_state=False)
    agent.eval()  # freezes the running obs normaliser

    # The critic is trained against reward-normalised returns; recover the
    # value in real discounted-return units so V* is interpretable.
    raw_ckpt = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    rrn = raw_ckpt.get("running_reward_norm", None)
    value_scale = 1.0
    if rrn is not None and "var" in rrn:
        value_scale = float(torch.sqrt(rrn["var"].double() + 1e-5).item())
    log.info(f"value un-normalisation scale = {value_scale:.4f}")
    log.info(f"checkpoint epoch = {raw_ckpt.get('epoch')}")

    # Collision geometry, so the reported perturbations can be checked for the
    # one artifact this method can introduce: reset noise rotates the whole body
    # about the *pelvis*, which in an inverted pose swings the support hands
    # downward. If init_min_geom_z ever goes negative the widened distribution
    # is spawning bodies through the floor and PhysX depenetration -- not the
    # policy -- is deciding some of the failures.
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from clean_yoga_motions import load_char_geoms, compute_min_geom_z

    mjcf_path = _os.path.join(
        robot_config.asset.asset_root, robot_config.asset.asset_file_name
    )
    char_geoms = load_char_geoms(
        mjcf_path, list(robot_config.kinematic_info.body_names), fabric.device
    )
    log.info(f"loaded collision geoms from {mjcf_path}")

    records = {k: [] for k in [
        "value_raw", "value", "success", "survival_steps", "scale", "family",
        "init_max_joint_err", "init_root_speed", "init_root_ang_speed",
        "init_mean_dof_speed", "init_root_tilt_rad", "init_min_geom_z",
        "final_max_joint_err", "final_pelvis_z", "final_mean_toe_z",
    ]}

    # Body indices for the end-of-rollout pose check. "Survived" only means the
    # tracking error never exceeded 0.5 m, which is loose; checking where the
    # pelvis and toes actually ended up confirms the node is holding the pose
    # rather than sitting in some nearby configuration that scrapes past the
    # termination test.
    _bn = list(robot_config.kinematic_info.body_names)
    IDX_PELVIS = _bn.index("Pelvis")
    IDX_TOES = [_bn.index("L_Toe"), _bn.index("R_Toe")]

    def run_batch(scale: float, family: str):
        env.robot_config.reset_noise = noise_for(scale, family)
        obs, _ = env.reset()

        # --- initial-state features, before any action ---
        st = env.simulator.get_robot_state()
        mm = env.motion_manager
        ref = env.motion_lib.get_motion_state(mm.motion_ids, mm.motion_times)
        ref_pos = ref.rigid_body_pos.clone()
        ref_pos += env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(ref_pos)
        max_joint_err = (
            (ref_pos - st.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]
        )
        root_speed = st.rigid_body_vel[:, 0].norm(dim=-1)
        root_ang_speed = st.rigid_body_ang_vel[:, 0].norm(dim=-1)
        dof_speed = st.dof_vel.abs().mean(dim=-1)

        from protomotions.utils.rotations import quat_rotate

        up = torch.tensor([0.0, 0.0, 1.0], device=st.rigid_body_rot.device)
        cur_up = quat_rotate(st.rigid_body_rot[:, 0], up.expand(env.num_envs, 3), w_last=True)
        ref_up = quat_rotate(ref.rigid_body_rot[:, 0].to(cur_up.device),
                             up.expand(env.num_envs, 3), w_last=True)
        tilt = torch.acos((cur_up * ref_up).sum(-1).clamp(-1.0, 1.0))

        min_geom_z = compute_min_geom_z(
            char_geoms, st.rigid_body_pos, st.rigid_body_rot
        )

        # --- V(s_0) ---
        with torch.no_grad():
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            v0 = agent.model(obs_td)["value"].squeeze(-1).clone()

        # --- roll out the deterministic policy ---
        failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        survival = torch.full((env.num_envs,), args.horizon, dtype=torch.long,
                              device=env.device)
        for t in range(args.horizon):
            with torch.no_grad():
                obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
                outs = agent.model(obs_td)
                action = outs["mean_action"] if "mean_action" in outs else outs["action"]
            obs, _rew, _dones, terminated, _extras = env.step(action)
            newly = terminated.bool() & ~failed
            survival[newly] = t + 1
            failed |= terminated.bool()

        success = (~failed).long()

        st_end = env.simulator.get_robot_state()
        final_err = (
            (ref_pos - st_end.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]
        )
        records["final_max_joint_err"].append(final_err.cpu().numpy())
        records["final_pelvis_z"].append(
            st_end.rigid_body_pos[:, IDX_PELVIS, 2].cpu().numpy()
        )
        records["final_mean_toe_z"].append(
            st_end.rigid_body_pos[:, IDX_TOES, 2].mean(dim=-1).cpu().numpy()
        )

        records["value_raw"].append(v0.cpu().numpy())
        records["value"].append((v0 * value_scale).cpu().numpy())
        records["success"].append(success.cpu().numpy())
        records["survival_steps"].append(survival.cpu().numpy())
        records["scale"].append(np.full(env.num_envs, scale))
        records["family"].append(np.array([family] * env.num_envs))
        records["init_max_joint_err"].append(max_joint_err.cpu().numpy())
        records["init_root_speed"].append(root_speed.cpu().numpy())
        records["init_root_ang_speed"].append(root_ang_speed.cpu().numpy())
        records["init_mean_dof_speed"].append(dof_speed.cpu().numpy())
        records["init_root_tilt_rad"].append(tilt.cpu().numpy())
        records["init_min_geom_z"].append(min_geom_z.cpu().numpy())

        sr = float(success.float().mean())
        log.info(
            f"  [{family}] scale={scale:5.1f}  hold_rate={sr:6.2%}  "
            f"V mean={float(v0.mean())*value_scale:8.2f}  "
            f"init_err={float(max_joint_err.mean()):.3f} m  "
            f"root_spd={float(root_speed.mean()):.3f} m/s  "
            f"min_geom_z={float(min_geom_z.min())*100:+.2f} cm"
        )
        return sr

    for family in ("velocity", "mixed"):
        log.info(f"=== sweep family: {family} ===")
        for scale in args.scales:
            for _ in range(args.repeats):
                run_batch(scale, family)

    data = {k: np.concatenate(v) for k, v in records.items()}

    # ------------------------------------------------------------------ #
    # Control batch: nominal initial state, training-level push ON.
    #
    # Two purposes. It confirms the disturbance channel is actually live
    # rather than a silent no-op (a real risk -- push is enabled only if some
    # velocity component is non-zero, and nothing errors if it is not), and it
    # measures what the node's *training* disturbance regime actually costs it,
    # which is the number that matters for holding the pose in a real flow.
    # ------------------------------------------------------------------ #
    # This is a diagnostic, not the deliverable: any failure here must not cost
    # the sweep that has already been paid for in GPU time.
    push_control = None
    if orig_push is not None:
        try:
            log.info("=== control batch: scale 1.0 reset noise + training push ON ===")
            env.simulator.config.domain_randomization.push = orig_push
            env.simulator._init_push_randomization()
            if getattr(env.simulator, "_push_enabled", False):
                run_batch(1.0, "push_on")
                push_control = {
                    "push_enabled": True,
                    "push_interval_range": list(orig_push.push_interval_range),
                    "max_linear_velocity": list(orig_push.max_linear_velocity),
                    "max_angular_velocity": list(orig_push.max_angular_velocity),
                    "hold_rate": float(records["success"][-1].mean()),
                    "V_mean": float(records["value"][-1].mean()),
                    "n": int(len(records["success"][-1])),
                }
                data = {k: np.concatenate(v) for k, v in records.items()}
            else:
                push_control = {"push_enabled": False,
                                "note": "push config present but has_push() was False"}
        except Exception as e:  # noqa: BLE001
            log.exception("push control batch failed; continuing")
            push_control = {"error": repr(e)}
            # Drop any partial batch so the arrays stay rectangular.
            n = min(len(v) for v in records.values())
            for k in records:
                del records[k][n:]
            data = {k: np.concatenate(v) for k, v in records.items()}
        finally:
            env.simulator.config.domain_randomization.push = None

    # ------------------------------------------------------------------ #
    # Fit on the velocity family, test the threshold on the mixed family.
    # ------------------------------------------------------------------ #
    fam = data["family"]
    sweep = fam != "push_on"  # the control batch never enters fit or test
    out = {
        "checkpoint": str(ckpt),
        "epoch": raw_ckpt.get("epoch"),
        "num_envs": args.num_envs,
        "horizon_steps": args.horizon,
        "horizon_seconds": args.horizon / 30.0,
        "scales": list(args.scales),
        "value_scale": value_scale,
        "n_samples": int(sweep.sum()),
        "overall_hold_rate": float(data["success"][sweep].mean()),
        "worst_init_min_geom_z_cm": float(data["init_min_geom_z"][sweep].min() * 100),
        "frac_init_penetrating": float((data["init_min_geom_z"][sweep] < 0).mean()),
        "push_control": push_control,
    }

    # Does a "held" rollout actually end in the node pose? Compare the pelvis
    # and toe heights after the full horizon against the frozen reference.
    _ref0 = env.motion_lib.get_motion_state(
        torch.zeros(1, dtype=torch.long, device=env.device),
        torch.zeros(1, device=env.device),
    )
    held = (data["success"] == 1) & sweep
    out["pose_check"] = {
        "n_held": int(held.sum()),
        "ref_pelvis_z": float(_ref0.rigid_body_pos[0, IDX_PELVIS, 2]),
        "ref_mean_toe_z": float(_ref0.rigid_body_pos[0, IDX_TOES, 2].mean()),
        "held_final_pelvis_z_mean": float(data["final_pelvis_z"][held].mean()) if held.any() else None,
        "held_final_mean_toe_z_mean": float(data["final_mean_toe_z"][held].mean()) if held.any() else None,
        "held_final_max_joint_err_mean": float(data["final_max_joint_err"][held].mean()) if held.any() else None,
        "failed_final_max_joint_err_mean": (
            float(data["final_max_joint_err"][(data["success"] == 0) & sweep].mean())
            if ((data["success"] == 0) & sweep).any() else None
        ),
    }

    for name, mask in (("velocity", fam == "velocity"), ("mixed", fam == "mixed")):
        v = data["value"][mask]
        y = data["success"][mask].astype(np.float64)
        entry = {
            "n": int(mask.sum()),
            "hold_rate": float(y.mean()),
            "auc_V": auc_roc(v, y),
            # Is V better than the obvious geometric feature?
            "auc_init_max_joint_err": auc_roc(-data["init_max_joint_err"][mask], y),
            "auc_init_root_speed": auc_roc(-data["init_root_speed"][mask], y),
            "per_scale": [],
        }
        for s in args.scales:
            m2 = mask & (data["scale"] == s)
            if m2.sum():
                entry["per_scale"].append({
                    "scale": float(s),
                    "n": int(m2.sum()),
                    "hold_rate": float(data["success"][m2].mean()),
                    "V_mean": float(data["value"][m2].mean()),
                    "V_std": float(data["value"][m2].std()),
                    "init_max_joint_err_mean": float(data["init_max_joint_err"][m2].mean()),
                    "init_root_speed_mean": float(data["init_root_speed"][m2].mean()),
                    "init_root_tilt_deg_mean": float(
                        np.degrees(data["init_root_tilt_rad"][m2].mean())
                    ),
                    # < 0 would mean the widened reset is spawning collision
                    # geometry through the floor (see init_min_geom_z note).
                    "init_min_geom_z_cm": float(data["init_min_geom_z"][m2].min() * 100),
                })
        out[name] = entry

    fit_mask = fam == "velocity"
    v_fit, y_fit = data["value"][fit_mask], data["success"][fit_mask].astype(np.float64)
    if 0 < y_fit.mean() < 1:
        w, b, v_star, bce = fit_logistic_1d(v_fit, y_fit)
        out["classifier"] = {
            "fit_on": "velocity",
            "w": w, "b": b, "final_bce": bce,
            "V_star_logistic": v_star,
            "V_star_logistic_raw": v_star / value_scale,
            "V_star_youden": youden_threshold(v_fit, y_fit),
            "V_star_conservative_p99": conservative_threshold(v_fit, y_fit, 0.99),
            "in_sample": threshold_metrics(v_fit, data["success"][fit_mask], v_star),
        }
        test_mask = fam == "mixed"
        if test_mask.sum() and 0 < data["success"][test_mask].mean() < 1:
            out["classifier"]["held_out_mixed"] = threshold_metrics(
                data["value"][test_mask], data["success"][test_mask], v_star
            )
            # The mixed family perturbs the root *pose*, which in an inverted
            # node can swing the support hands below the floor. Re-report the
            # transfer on the subset that never penetrated, so the conclusion
            # cannot be an artifact of PhysX depenetration deciding failures.
            clean = test_mask & (data["init_min_geom_z"] >= 0.0)
            if clean.sum() and 0 < data["success"][clean].mean() < 1:
                m = threshold_metrics(
                    data["value"][clean], data["success"][clean], v_star
                )
                m["auc_V"] = auc_roc(
                    data["value"][clean], data["success"][clean].astype(np.float64)
                )
                m["frac_of_mixed_kept"] = float(clean.sum() / test_mask.sum())
                out["classifier"]["held_out_mixed_nonpenetrating"] = m
        # Empirical calibration: P(hold | V) in deciles of V.
        qs = np.quantile(v_fit, np.linspace(0, 1, 11))
        cal = []
        for i in range(10):
            m = (v_fit >= qs[i]) & (v_fit <= qs[i + 1] if i == 9 else v_fit < qs[i + 1])
            if m.sum():
                cal.append({"V_lo": float(qs[i]), "V_hi": float(qs[i + 1]),
                            "n": int(m.sum()), "hold_rate": float(y_fit[m].mean())})
        out["classifier"]["calibration_deciles"] = cal
    else:
        out["classifier"] = {
            "error": "degenerate labels on the fit family "
            f"(hold rate {float(y_fit.mean()):.3f}); widen or narrow --scales."
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    np.savez(out_path.with_suffix(".npz"), **data)

    print("\n" + "=" * 72)
    print(json.dumps(out, indent=2))
    print("=" * 72)
    log.info(f"wrote {out_path} and {out_path.with_suffix('.npz')}")

    if hasattr(env.simulator, "shutdown"):
        env.simulator.shutdown()


if __name__ == "__main__":
    main()
