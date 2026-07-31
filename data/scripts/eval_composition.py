# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed-loop composition: run the **edge** policy, then hand off to the **node**.

This is the end-to-end test of the skill graph, and of the claim that the value
function can join the two. Everything before it was measured on the *reference*
trajectory; here the character is actually driven by the trained edge policy and
then handed to the node policy mid-flight.

Per switch step ``k``:

1. Reset from the start of the edge clip.
2. Run the **edge** policy for ``k`` steps. Failure = max per-body error against
   the *edge* reference exceeding 0.5 m (the edge's own criterion).
3. Read **V_node(s_k)** from the target node's critic.
4. Switch to the **node** policy and run ``--hold-steps``. Failure = max per-body
   error against the *node's frozen pose* exceeding 0.5 m (the node's own
   criterion -- not the edge's, which is meaningless once the edge is over).

Sweeping ``k`` gives, for each candidate switch time, whether the composition
survives and what the node's critic thought at that moment. Two questions fall
out:

* **Is there a switch time that works at all?** If not, the edge does not reach
  the node's basin and the graph is broken at that transition.
* **Does V pick it?** Correlating ``V_node(s_k)`` against composition success is
  the direct test of "use the value classifier to join the edges to the nodes".
  ``notes/Skill_graph_nodes.MD`` §6.2 found V-thresholding anti-predictive on the
  handstand and down-dog nodes, so the honest expectation is that the *band*
  ``|V - V_nom| < delta`` does better than a threshold, and that both are weaker
  than one would like.

The node policy is driven through ``NodeValueJoin.build_node_obs``, which
reconstructs the node's observation for an arbitrary state (validated to 7e-7
relative against the node agent's own path) -- necessary because the env's
reference here is the edge clip, not the node's frozen pose.

Usage::

    python data/scripts/eval_composition.py \\
        --edge-checkpoint results/edge_kickup_join/final.ckpt \\
        --node node_handstand --edge edge_kickup \\
        --switch-steps 240 260 280 300 320 --hold-steps 300 --num-envs 256 \\
        --out results/edge_kickup_join/composition.json
"""


def create_parser():
    import argparse

    p = argparse.ArgumentParser(description="Edge->node closed-loop composition.")
    p.add_argument("--edge-checkpoint", required=True)
    p.add_argument("--node", required=True, help="e.g. node_handstand")
    p.add_argument("--edge", required=True, help="e.g. edge_kickup")
    p.add_argument("--simulator", default="isaaclab")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--switch-steps", type=int, nargs="*", default=None,
                   help="Steps of edge policy before handing off. Default: a "
                        "sweep across the last third of the clip.")
    p.add_argument("--hold-steps", type=int, default=300)
    p.add_argument("--band", type=float, nargs=2, default=None,
                   metavar=("V_LO", "V_HI"),
                   help="Node V band; defaults to value_band.json best_band.")
    p.add_argument("--source-node", default=None,
                   help="Run this node's policy BEFORE the edge, for --prehold-steps. "
                        "Gives the full graph traversal source-node -> edge -> target-node.")
    p.add_argument("--prehold-steps", type=int, default=90,
                   help="Steps the source node holds before the edge starts.")
    p.add_argument("--visualize", action="store_true",
                   help="Open the viewer and loop the traversal instead of sweeping "
                        "switch points. Needs a display.")
    p.add_argument("--zero-actions-at-switch", action="store_true",
                   help="Zero the action history at the hand-off. The observation "
                        "contains previous_actions, so after phase 1 that buffer holds "
                        "the EDGE policy's actions -- an action history no node policy "
                        "would produce. The node was trained with a zeroed buffer at "
                        "reset, so zeroing puts it back in-distribution.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic-edge", action="store_true",
                   help="Drive the EDGE phase with sampled actions (the training "
                        "action distribution) instead of the deterministic mean. "
                        "The parity probe attributed the train/eval survival gap "
                        "almost entirely to this flip; running both protocols "
                        "separates 'cannot do it' from 'cannot do it without "
                        "exploration dither'. The node phase stays deterministic.")
    p.add_argument("--bank-start-prob", type=float, default=0.0,
                   help="If the edge was trained with bank-based starts "
                        "(BankResetEnv), fraction of eval starts drawn from the "
                        "bank. Default 0: nominal reference starts, comparable "
                        "with pre-bank experiments. Set 1.0 for the realistic "
                        "composed-start protocol.")
    p.add_argument("--out", required=True)
    return p


import argparse  # noqa: E402

parser = create_parser()
args, _unknown = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger("composition")


def main():
    global args
    args = parser.parse_args()

    # Must precede torch.load: the edge's resolved config pickles a
    # NodeValueJoin instance (the hand-off reward), so its module has to be
    # importable before unpickling.
    sys.path.insert(0, os.path.join(os.getcwd(), "examples/experiments/mimic"))

    torch.manual_seed(args.seed)

    edge_ck = Path(args.edge_checkpoint)
    cfgs = torch.load(edge_ck.parent / "resolved_configs.pt",
                      map_location="cpu", weights_only=False)
    robot_config = cfgs["robot"]; simulator_config = cfgs["simulator"]
    terrain_config = cfgs.get("terrain"); scene_lib_config = cfgs["scene_lib"]
    motion_lib_config = cfgs["motion_lib"]; env_config = cfgs["env"]
    agent_config = cfgs["agent"]

    simulator_config.num_envs = args.num_envs
    simulator_config.headless = not args.visualize
    if getattr(simulator_config, "domain_randomization", None) is not None:
        simulator_config.domain_randomization.push = None
    robot_config.reset_noise = None
    # Terminations are evaluated manually: the env's tracking_error is measured
    # against the EDGE reference, which is the wrong criterion once the node has
    # taken over.
    env_config.termination_components = {}
    env_config.max_episode_length = 10_000_000
    env_config.motion_manager.init_start_prob = 1.0
    # Bank-trained edges (BankResetEnv): eval starts are nominal by default so
    # numbers stay comparable across experiments; --bank-start-prob 1.0 gives
    # the realistic composed-start protocol.
    if hasattr(env_config, "bank_prob"):
        env_config.bank_prob = args.bank_start_prob
    # The join reward would load a second critic per step for nothing here.
    env_config.reward_components = {
        k: v for k, v in env_config.reward_components.items()
        if k != "node_value_join_rew"
    }

    from protomotions.utils.fabric_config import FabricConfig
    from lightning.fabric import Fabric

    fabric = Fabric(**FabricConfig(accelerator="gpu", devices=1, num_nodes=1,
                                   loggers=[], callbacks=[]).as_kwargs())
    fabric.launch()
    sim_extra = {}
    if args.simulator == "isaaclab":
        sim_extra["simulation_app"] = AppLauncher(
            {"headless": not args.visualize, "device": str(fabric.device)}
        ).app

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config)

    from protomotions.utils.component_builder import build_all_components
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.agents.base_agent.agent import BaseAgent

    comp = build_all_components(
        terrain_config=terrain_config, scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config, simulator_config=simulator_config,
        robot_config=robot_config, device=fabric.device, save_dir=None, **sim_extra)
    env: BaseEnv = get_class(env_config._target_)(
        config=env_config, robot_config=robot_config, device=fabric.device,
        terrain=comp["terrain"], scene_lib=comp["scene_lib"],
        motion_lib=comp["motion_lib"], simulator=comp["simulator"])

    edge_agent: BaseAgent = get_class(agent_config._target_)(
        config=agent_config, env=env, fabric=fabric, root_dir=edge_ck.parent)
    edge_agent.setup()
    edge_agent.load(str(edge_ck), load_env=False, load_training_state=False)
    edge_agent.eval()

    # ---- the node: its own model, driven on a reconstructed observation ----
    from node_value_join import NodeValueJoin

    from protomotions.agents.utils.normalization import (
        materialize_lazy_running_stats_from_state_dict,
    )

    def load_node(name):
        """Node model + an observation builder for it, usable on any state."""
        ck_ = Path(f"results/{name}/final.ckpt")
        cfg_ = torch.load(ck_.parent / "resolved_configs.pt",
                          map_location="cpu", weights_only=False)
        Model = get_class(cfg_["agent"].model._target_)
        m = Model(config=cfg_["agent"].model)
        raw_ = torch.load(str(ck_), map_location="cpu", weights_only=False)
        materialize_lazy_running_stats_from_state_dict(m, raw_["model"])
        if hasattr(m, "materialize_from_state_dict"):
            m.materialize_from_state_dict(raw_["model"])
        m.load_state_dict(raw_["model"]); m.to(fabric.device).eval()
        # Catch-trained variants (e.g. node_handstand_catch) reuse the base
        # node's frozen-pose clip.
        mp = Path(f"data/smpl/skill_graph_handstand/{name}.pt")
        if not mp.exists() and name.endswith("_catch"):
            mp = Path(f"data/smpl/skill_graph_handstand/{name[: -len('_catch')]}.pt")
        jn = NodeValueJoin(node_checkpoint=str(ck_),
                           node_motion_pt=str(mp),
                           v_nominal=0.0, delta=1.0)
        jn._build(fabric.device)
        return m, jn, raw_

    node_model, _joiner_t, raw_node = load_node(args.node)
    node_ck = Path(f"results/{args.node}/final.ckpt")

    band = args.band
    band_path = Path(f"results/{args.node}/value_band.json")
    v_nom = None
    if band is None and band_path.exists():
        vb = json.load(open(band_path))["families"]["velocity"]
        band = [vb["best_band"]["lo"], vb["best_band"]["hi"]]
        v_nom = vb["V_nominal"]
    log.info(f"node band = {band}, V_nominal = {v_nom}")

    joiner = _joiner_t
    node_ref_pos = joiner.ref_pos  # [B,3] in source-clip coords

    src_model = src_joiner = None
    if args.source_node:
        src_model, src_joiner, _ = load_node(args.source_node)
        log.info(f"source node {args.source_node} will hold for {args.prehold_steps} steps")

    n = args.num_envs
    clip_steps = int(round(float(env.motion_lib.motion_lengths[0]) / env.dt))
    switch_steps = args.switch_steps or list(
        range(int(clip_steps * 0.55), clip_steps + 1, max(clip_steps // 14, 1))
    )
    log.info(f"edge clip = {clip_steps} steps; sweeping switches {switch_steps}")

    def node_err(st, offset):
        xy = torch.zeros_like(offset); xy[:, :2] = offset[:, :2]
        ref = node_ref_pos.unsqueeze(0).expand(n, -1, -1) + xy.unsqueeze(1)
        return (ref - st.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]

    def edge_err(st):
        mm = env.motion_manager
        ref = env.motion_lib.get_motion_state(mm.motion_ids, mm.motion_times)
        rp = ref.rigid_body_pos.clone()
        rp += env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(rp)
        return (rp - st.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]

    def drive(model, jn, steps, obs, offset):
        """Run a node policy for `steps` steps on the reconstructed node obs."""
        for _ in range(steps):
            ctx = env.context
            td = jn.build_node_obs(
                body_pos=ctx.current.rigid_body_pos, body_rot=ctx.current.rigid_body_rot,
                body_vel=ctx.current.rigid_body_vel,
                body_ang_vel=ctx.current.rigid_body_ang_vel,
                ground_height=ctx.ground_heights, body_contacts=ctx.body_contacts,
                historical_actions=ctx.historical.actions, respawn_root_offset=offset,
            )
            with torch.no_grad():
                a = model(td)["mean_action"]
            obs, *_ = env.step(a)
        return obs

    # ------------------------------------------------------------------ #
    # Viewer mode: loop the full traversal instead of sweeping switches.
    # ------------------------------------------------------------------ #
    if args.visualize:
        k = args.switch_steps[0] if args.switch_steps else clip_steps
        log.info(f"viewer: {args.source_node or '(none)'} [{args.prehold_steps}] -> "
                 f"edge [{k}] -> {args.node} [{args.hold_steps}], looping. Ctrl-C to stop.")
        try:
            while True:
                obs, _ = env.reset()
                offset = env.respawn_root_offset.clone()

                if src_model is not None:
                    obs = drive(src_model, src_joiner, args.prehold_steps, obs, offset)
                    # The edge must track its clip from the START, but motion time
                    # advanced during the hold -- rewind it (and progress_buf, which
                    # the edge's terminal terms are indexed on).
                    env.motion_manager.motion_times[:] = 0.0
                    env.progress_buf[:] = 0
                    env._current_context = None
                    env.compute_observations(context=env.context)
                    obs = env.get_obs()

                for _ in range(k):
                    with torch.no_grad():
                        td = edge_agent.obs_dict_to_tensordict(
                            edge_agent.add_agent_info_to_obs(obs))
                        a = edge_agent.model(td)["mean_action"]
                    obs, *_ = env.step(a)

                st_ = env.simulator.get_robot_state()
                log.info(f"  hand-off: node err {node_err(st_, offset).mean():.3f} m")
                obs = drive(node_model, joiner, args.hold_steps, obs, offset)
        except KeyboardInterrupt:
            log.info("stopped")
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()
        return

    results = []
    for k in switch_steps:
        obs, _ = env.reset()
        offset = env.respawn_root_offset.clone()
        alive = torch.ones(n, dtype=torch.bool, device=env.device)

        # --- phase 1: the edge policy ---
        edge_action_key = "action" if args.stochastic_edge else "mean_action"
        for _ in range(k):
            with torch.no_grad():
                td = edge_agent.obs_dict_to_tensordict(
                    edge_agent.add_agent_info_to_obs(obs))
                act = edge_agent.model(td)[edge_action_key]
            obs, *_ = env.step(act)
            alive &= edge_err(env.simulator.get_robot_state()) <= 0.5

        if args.zero_actions_at_switch and env.state_history is not None:
            env.state_history.actions[:] = 0.0
            env.state_history.processed_actions[:] = 0.0
            env._current_context = None

        st = env.simulator.get_robot_state()
        ctx = env.context
        node_td = joiner.build_node_obs(
            body_pos=ctx.current.rigid_body_pos, body_rot=ctx.current.rigid_body_rot,
            body_vel=ctx.current.rigid_body_vel, body_ang_vel=ctx.current.rigid_body_ang_vel,
            ground_height=ctx.ground_heights, body_contacts=ctx.body_contacts,
            historical_actions=ctx.historical.actions, respawn_root_offset=offset,
        )
        with torch.no_grad():
            v_switch = node_model(node_td)["value"].squeeze(-1) * joiner._value_scale
        err_switch = node_err(st, offset)
        # Pose proximity is not catchability (notes 6.3): record the momentum too.
        root_spd = st.rigid_body_vel[:, 0].norm(dim=-1)
        dof_spd = st.dof_vel.abs().mean(dim=-1)
        body_spd = st.rigid_body_vel.norm(dim=-1).mean(dim=-1)

        # --- phase 2: the node policy takes over ---
        held = alive.clone()
        for _ in range(args.hold_steps):
            ctx = env.context
            node_td = joiner.build_node_obs(
                body_pos=ctx.current.rigid_body_pos, body_rot=ctx.current.rigid_body_rot,
                body_vel=ctx.current.rigid_body_vel,
                body_ang_vel=ctx.current.rigid_body_ang_vel,
                ground_height=ctx.ground_heights, body_contacts=ctx.body_contacts,
                historical_actions=ctx.historical.actions, respawn_root_offset=offset,
            )
            with torch.no_grad():
                act = node_model(node_td)["mean_action"]
            obs, *_ = env.step(act)
            held &= node_err(env.simulator.get_robot_state(), offset) <= 0.5

        in_band = ((v_switch >= band[0]) & (v_switch <= band[1])) if band else None
        row = {
            "switch_step": int(k),
            "switch_time_s": float(k * env.dt),
            "edge_survival_rate": float(alive.float().mean()),
            "composition_rate": float(held.float().mean()),
            "composition_rate_given_edge_alive": (
                float(held[alive].float().mean()) if alive.any() else None),
            "V_at_switch_mean": float(v_switch.mean()),
            "V_at_switch_std": float(v_switch.std()),
            "node_err_at_switch_mean": float(err_switch.mean()),
            "root_speed_at_switch": float(root_spd.mean()),
            "mean_dof_speed_at_switch": float(dof_spd.mean()),
            "mean_body_speed_at_switch": float(body_spd.mean()),
            "in_band_frac": float(in_band.float().mean()) if in_band is not None else None,
            "band_agreement": (
                float((in_band == held).float().mean()) if in_band is not None else None),
        }
        results.append(row)
        log.info(
            f"  switch@{k:4d} ({k*env.dt:5.2f}s): edge_alive={row['edge_survival_rate']:.3f} "
            f"compose={row['composition_rate']:.3f} V={row['V_at_switch_mean']:.1f}"
            f"±{row['V_at_switch_std']:.1f} node_err={row['node_err_at_switch_mean']:.3f} "
            f"band_agree={row['band_agreement']}"
        )

    import hashlib
    best = max(results, key=lambda r: r["composition_rate"]) if results else None
    out = {
        "edge_checkpoint": str(edge_ck),
        "checkpoint_sha256": hashlib.sha256(open(edge_ck, "rb").read()).hexdigest(),
        "edge_epoch": torch.load(str(edge_ck), map_location="cpu",
                                 weights_only=False).get("epoch"),
        "node": args.node, "edge": args.edge,
        "num_envs": n, "hold_steps": args.hold_steps,
        "seed": args.seed, "bank_start_prob": args.bank_start_prob,
        "stochastic_edge": bool(args.stochastic_edge),
        "zero_actions_at_switch": bool(args.zero_actions_at_switch),
        "clip_steps": clip_steps,
        "node_band": band, "V_nominal": v_nom,
        "best_switch": best,
        "sweep": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "sweep"}, indent=2))
    log.info(f"wrote {args.out}")

    if hasattr(env.simulator, "shutdown"):
        env.simulator.shutdown()


if __name__ == "__main__":
    main()
