# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Does an **edge** hand the character off inside the target **node**'s region?

This is the composition question the skill graph rests on. ``node_stability_region.py``
measures *where* a node can take over (states with V > V\\*); this script asks the
complementary question: does the transition actually **pass through** that set, and
when?

Method — for a target node policy and an edge trajectory:

1. Reset the env from the node's frozen reference. This fixes ``respawn_root_offset``
   = ``spawn_xy − node_ref_root_xy``, i.e. the translation that places the node's
   target pose in the world.
2. Overwrite the **physical** state with the edge reference at time t, translated by
   that *same* offset. This is exactly correct here because the node and edge clips
   were cut from one source clip with no re-centering, so they already share a
   coordinate frame — the relative geometry between "where the edge puts the body"
   and "where the node wants the body" is preserved.
3. Recompute observations and read **V(s_t)** from the node's critic. The node's own
   ``motion_times`` stays 0 and its reference is constant, so the target half of the
   observation is unchanged by the state overwrite.
4. Roll the **node** policy forward ``--horizon`` steps with the training termination
   active, and label whether it held.

That yields, along the whole edge, both the node's value estimate V(t) and the ground
truth hold(t) — so the **hand-off window** can be read off directly, and V\\* can be
checked against it rather than merely assumed.

Note this evaluates the *reference* (kinematic) edge trajectory, not a trained edge
policy. That is the right precondition to establish first: if the human's own motion
never enters the node's region, no edge controller tracking it could hand off either,
and the node needs a wider basin rather than the edge needing better tracking.

Usage::

    python data/scripts/handoff_check.py \\
        --node-checkpoint results/node_handstand/final.ckpt \\
        --edge-motion data/smpl/skill_graph_handstand/edge_kickup.pt \\
        --v-star-from results/node_handstand/stability_region.json \\
        --simulator isaaclab --num-envs 512 --horizon 300 \\
        --out results/node_handstand/handoff_edge_kickup.json
"""


def create_parser():
    import argparse

    p = argparse.ArgumentParser(description="Edge -> node hand-off check.")
    p.add_argument("--node-checkpoint", required=True)
    p.add_argument("--edge-motion", required=True, help="Packaged .pt of the edge clip.")
    p.add_argument(
        "--v-star-from",
        default=None,
        help="stability_region.json to read V* from (optional; used for reporting).",
    )
    p.add_argument("--simulator", default="isaaclab")
    p.add_argument(
        "--num-envs",
        type=int,
        default=512,
        help="One sampled time along the edge per env.",
    )
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--out", required=True)
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
log = logging.getLogger("handoff")


def main():
    global args
    args = parser.parse_args()

    ckpt = Path(args.node_checkpoint)
    cfgs = torch.load(
        ckpt.parent / "resolved_configs.pt", map_location="cpu", weights_only=False
    )
    robot_config = cfgs["robot"]
    simulator_config = cfgs["simulator"]
    terrain_config = cfgs.get("terrain")
    scene_lib_config = cfgs["scene_lib"]
    motion_lib_config = cfgs["motion_lib"]
    env_config = cfgs["env"]
    agent_config = cfgs["agent"]

    simulator_config.num_envs = args.num_envs
    simulator_config.headless = True
    if getattr(simulator_config, "domain_randomization", None) is not None:
        simulator_config.domain_randomization.push = None
    # The hand-off state comes from the edge, not from reset noise.
    robot_config.reset_noise = None
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
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.agents.base_agent.agent import BaseAgent

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,   # the NODE's frozen clip
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
        **sim_extra,
    )

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

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, root_dir=ckpt.parent
    )
    agent.setup()
    agent.load(str(ckpt), load_env=False, load_training_state=False)
    agent.eval()

    raw_ckpt = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    rrn = raw_ckpt.get("running_reward_norm", None)
    value_scale = (
        float(torch.sqrt(rrn["var"].double() + 1e-5).item())
        if rrn is not None and "var" in rrn else 1.0
    )

    # The edge, loaded as a *separate* motion library. The env keeps the node's
    # frozen reference as its tracking target throughout.
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    edge = MotionLib(MotionLibConfig(motion_file=args.edge_motion), device=fabric.device)
    edge_len = float(edge.motion_lengths[0])
    n = args.num_envs
    times = torch.linspace(0.0, edge_len, n, device=fabric.device)
    log.info(f"edge {args.edge_motion}: {edge_len:.3f} s, sampling {n} times")

    # --- 1) reset from the node reference to establish respawn_root_offset ---
    env.reset()
    off = env.respawn_root_offset.clone()  # [n, 3]

    # --- 2) overwrite the physical state with the edge frame at time t ---
    from protomotions.simulator.base_simulator.simulator_state import ResetState

    est = edge.get_motion_state(torch.zeros(n, dtype=torch.long, device=fabric.device), times)
    new_states = ResetState.from_robot_state(est)
    new_states.root_pos = new_states.root_pos + off
    env.simulator.reset_envs(new_states, None,
                             torch.arange(n, device=fabric.device))

    env.progress_buf[:] = 0
    env.reset_buf[:] = False
    env.terminate_buf[:] = False
    env._current_context = None
    env._current_noisy_obs = None
    env.compute_observations(context=env.context)
    obs = env.get_obs()

    # --- 3) V(s_t) under the node's critic ---
    with torch.no_grad():
        v0 = agent.model(
            agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
        )["value"].squeeze(-1).clone()
    v_un = (v0 * value_scale).cpu().numpy()

    # initial distance from the node's target pose, for context
    ref = env.motion_lib.get_motion_state(
        env.motion_manager.motion_ids, env.motion_manager.motion_times
    )
    ref_pos = ref.rigid_body_pos.clone()
    ref_pos += env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(ref_pos)
    st = env.simulator.get_robot_state()
    init_err = (ref_pos - st.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0].cpu().numpy()

    # --- 4) roll the NODE policy and label the hand-off ---
    failed = torch.zeros(n, dtype=torch.bool, device=env.device)
    survival = torch.full((n,), args.horizon, dtype=torch.long, device=env.device)
    for t in range(args.horizon):
        with torch.no_grad():
            outs = agent.model(
                agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            )
            action = outs["mean_action"] if "mean_action" in outs else outs["action"]
        obs, _r, _d, terminated, _e = env.step(action)
        newly = terminated.bool() & ~failed
        survival[newly] = t + 1
        failed |= terminated.bool()
    success = (~failed).long().cpu().numpy()

    ts = times.cpu().numpy()
    v_star = None
    if args.v_star_from and Path(args.v_star_from).exists():
        c = json.load(open(args.v_star_from)).get("classifier", {})
        v_star = c.get("V_star_conservative_p99") or c.get("V_star_logistic")

    # Hand-off window = the largest contiguous run of sampled times that held.
    best_lo = best_hi = -1
    i = 0
    while i < n:
        if success[i]:
            j = i
            while j + 1 < n and success[j + 1]:
                j += 1
            if best_lo < 0 or (j - i) > (best_hi - best_lo):
                best_lo, best_hi = i, j
            i = j + 1
        else:
            i += 1

    out = {
        "node_checkpoint": str(ckpt),
        "node_epoch": raw_ckpt.get("epoch"),
        "edge_motion": args.edge_motion,
        "edge_length_s": edge_len,
        "horizon_steps": args.horizon,
        "n_samples": n,
        "value_scale": value_scale,
        "V_star_used": v_star,
        "overall_handoff_rate": float(success.mean()),
        "handoff_window_s": (
            [float(ts[best_lo]), float(ts[best_hi])] if best_lo >= 0 else None
        ),
        "handoff_window_frac_of_edge": (
            float((ts[best_hi] - ts[best_lo]) / edge_len) if best_lo >= 0 else 0.0
        ),
        "terminal_state": {
            "t_s": float(ts[-1]),
            "V": float(v_un[-1]),
            "held": int(success[-1]),
            "init_max_joint_err": float(init_err[-1]),
            "above_V_star": (bool(v_un[-1] > v_star) if v_star is not None else None),
        },
        "profile": [
            {"t_s": float(ts[i]), "V": float(v_un[i]), "held": int(success[i]),
             "survival_steps": int(survival[i].item()),
             "init_max_joint_err": float(init_err[i])}
            for i in range(0, n, max(n // 60, 1))
        ],
    }
    if v_star is not None:
        pred = v_un > v_star
        out["V_star_agreement"] = {
            "accuracy": float((pred == (success == 1)).mean()),
            "n_predicted_hold": int(pred.sum()),
            "n_actually_held": int(success.sum()),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    np.savez(Path(args.out).with_suffix(".npz"),
             t=ts, V=v_un, success=success,
             survival=survival.cpu().numpy(), init_err=init_err)

    print(json.dumps({k: v for k, v in out.items() if k != "profile"}, indent=2))
    log.info(f"wrote {args.out}")

    if hasattr(env.simulator, "shutdown"):
        env.simulator.shutdown()


if __name__ == "__main__":
    main()
