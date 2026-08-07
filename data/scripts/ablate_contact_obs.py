# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure how much a trained policy's actions actually depend on each observation block.

Drives the policy normally (real observations, real actions applied) and, at every
step, additionally evaluates the network on counterfactual observations in which one
block has been replaced by that block's running mean.  Because the model concatenates
its ``in_keys`` and normalises the result with a single ``RunningMeanStd``, substituting
the mean makes exactly that slice normalise to zero -- the network's own definition of
"no information" -- without pushing the rest of the input off-distribution the way a
raw-zero substitution would.

The number that matters is not the raw action delta but its size relative to ablating a
block the policy provably must use.  ``mimic_target_poses`` is that yardstick: a mimic
tracker cannot function without its target, so its delta calibrates the scale.

Ablations run per whole key, and per channel within ``contact_obs_v1`` (each channel
substituted across all sensed bodies at once), so a block can be judged as a whole and
channel by channel.

Example::

    python data/scripts/ablate_contact_obs.py \
      --checkpoint results/smpl_yogi_balance_contact_rich_2/last.ckpt \
      --motion-file data/smpl/yoga_yogi_balance_subset_grounded \
      --steps 600 --out results/contact_obs_ablation/ablation.npz
"""

from __future__ import annotations

import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--motion-file", type=str, required=True)
    parser.add_argument("--simulator", type=str, default="isaaclab")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--num-envs", type=int, default=0, help="0 = one per motion.")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--usd-asset",
        type=str,
        default=None,
        help=(
            "Override robot.asset.usd_asset_file_name. Use this to evaluate a "
            "checkpoint on the plant it was actually trained on."
        ),
    )
    return parser


parser = create_parser()
args, _unknown = parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger("ablate")


def key_offsets(in_keys, obs_dict):
    """Start/stop columns of each key inside the concatenated observation."""
    offsets, cursor = {}, 0
    for key in in_keys:
        width = obs_dict[key].shape[-1]
        offsets[key] = (cursor, cursor + width)
        cursor += width
    return offsets, cursor


def main() -> None:
    global args
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    checkpoint = Path(args.checkpoint)
    cfgs = torch.load(
        checkpoint.parent / "resolved_configs_inference.pt",
        map_location="cpu",
        weights_only=False,
    )
    robot_config = cfgs["robot"]
    simulator_config = cfgs["simulator"]
    terrain_config = cfgs.get("terrain")
    scene_lib_config = cfgs["scene_lib"]
    motion_lib_config = cfgs["motion_lib"]
    env_config = cfgs["env"]
    agent_config = cfgs["agent"]

    motion_lib_config.motion_file = args.motion_file
    if args.usd_asset:
        log.info("USD override: %s", args.usd_asset)
        robot_config.asset.usd_asset_file_name = args.usd_asset

    motion_dir = Path(args.motion_file)
    num_motions = (
        len(sorted(motion_dir.rglob("*.motion")))
        if motion_dir.is_dir()
        else len(torch.load(args.motion_file, map_location="cpu", weights_only=False)["motion_files"])
    )
    num_envs = args.num_envs or num_motions
    simulator_config.num_envs = num_envs
    simulator_config.headless = True
    if getattr(simulator_config, "domain_randomization", None) is not None:
        simulator_config.domain_randomization.push = None
    robot_config.reset_noise = None

    env_config.motion_manager.subset_method = list(range(num_envs))
    env_config.motion_manager.init_start_prob = 1.0
    env_config.max_episode_length = max(args.steps * 4, 100_000)

    from lightning.fabric import Fabric  # noqa: E402
    from protomotions.utils.fabric_config import FabricConfig  # noqa: E402

    fabric = Fabric(
        **FabricConfig(
            accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[]
        ).as_kwargs()
    )
    fabric.launch()
    sim_extra = {}
    if args.simulator == "isaaclab":
        sim_extra["simulation_app"] = AppLauncher(
            {"headless": True, "device": str(fabric.device)}
        ).app

    from protomotions.agents.base_agent.agent import BaseAgent  # noqa: E402
    from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
    from protomotions.envs.obs import CONTACT_OBS_V1_LAYOUT  # noqa: E402
    from protomotions.simulator.base_simulator.utils import (  # noqa: E402
        convert_friction_for_simulator,
    )
    from protomotions.utils.component_builder import build_all_components  # noqa: E402
    from protomotions.utils.hydra_replacement import get_class  # noqa: E402

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )
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
    env: BaseEnv = get_class(env_config._target_)(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )
    agent: BaseAgent = get_class(agent_config._target_)(
        config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent
    )
    agent.setup()
    agent.load(str(checkpoint), load_env=False, load_training_state=False)
    agent.eval()

    total_mass = float(
        env.simulator._robot.root_physx_view.get_masses()[0].sum()
    )
    log.info("simulated robot mass: %.2f kg", total_mass)

    actor_norm = agent.model._actor.mu.norm.running_obs_norm
    critic_norm = agent.model._critic.norm.running_obs_norm
    in_keys = list(agent.model._actor.mu.config.in_keys)
    log.info("actor in_keys: %s", in_keys)

    obs, _ = env.reset()
    obs = agent.add_agent_info_to_obs(obs)
    offsets, total_width = key_offsets(in_keys, obs)
    log.info("concatenated width %d; offsets %s", total_width, offsets)

    actor_mean = actor_norm.mean.float().to(fabric.device)
    critic_mean = critic_norm.mean.float().to(fabric.device)
    mean_gap = float((actor_mean - critic_mean).abs().max())
    log.info(
        "max |actor_mean - critic_mean| = %.4g (value deltas assume this is small)",
        mean_gap,
    )

    num_bodies = len(env.contact_observation_body_names)

    # --- what to ablate -----------------------------------------------------
    # Whole keys, plus every channel of contact_obs_v1 across all bodies at once.
    specs: list[tuple[str, list[tuple[int, int]]]] = []
    for key in in_keys:
        specs.append((f"key:{key}", [offsets[key]]))
    specs.append(
        ("key:contact_obs_v1+proximity",
         [offsets["contact_obs_v1"], offsets["contact_proximity_obs"]])
    )

    contact_start = offsets["contact_obs_v1"][0]
    per_body_dim = 17
    for name, sl in CONTACT_OBS_V1_LAYOUT.items():
        cols = []
        for body in range(num_bodies):
            base = contact_start + body * per_body_dim
            cols.append((base + sl.start, base + sl.stop))
        specs.append((f"chan:{name}", cols))
    global_start = contact_start + num_bodies * per_body_dim
    specs.append(("chan:__global__", [(global_start, global_start + 4)]))

    log.info("%d ablations per step", len(specs))

    def ablate(td, spans, mean_vec):
        """Return a copy of td with the given concat columns set to their mean."""
        out = td.clone()
        for key in in_keys:
            start, stop = offsets[key]
            tensor = out[key].clone()
            touched = False
            for a, b in spans:
                lo, hi = max(a, start), min(b, stop)
                if lo < hi:
                    tensor[..., lo - start : hi - start] = mean_vec[lo:hi]
                    touched = True
            if touched:
                out[key] = tensor
        return out

    rec = {
        "delta_action": [],   # [T, S, E]
        "delta_value": [],    # [T, S, E]
        "action_scale": [],   # [T, E]  per-step RMS of the real action
        "track_err": [],      # [T, E]
        "done": [],
    }

    for step in range(args.steps):
        td = agent.obs_dict_to_tensordict(obs)
        with torch.no_grad():
            base_out = agent.model(td.clone())
            a_real = base_out["mean_action"]
            v_real = base_out["value"].squeeze(-1)

            das, dvs = [], []
            for _name, spans in specs:
                oa = agent.model(ablate(td, spans, actor_mean))
                da = (oa["mean_action"] - a_real).pow(2).mean(-1).sqrt()
                ov = agent.model(ablate(td, spans, critic_mean))
                dv = (ov["value"].squeeze(-1) - v_real).abs()
                das.append(da)
                dvs.append(dv)

        rec["delta_action"].append(torch.stack(das).cpu().numpy())
        rec["delta_value"].append(torch.stack(dvs).cpu().numpy())
        rec["action_scale"].append(a_real.pow(2).mean(-1).sqrt().cpu().numpy())

        manager = env.motion_manager
        ref = env.motion_lib.get_motion_state(manager.motion_ids, manager.motion_times)
        ref_pos = ref.rigid_body_pos.clone()
        ref_pos = ref_pos + env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(ref_pos)
        state = env.simulator.get_robot_state()
        err = (ref_pos - state.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]
        rec["track_err"].append(err.cpu().numpy())

        obs, _rew, dones, _term, _extras = env.step(a_real)
        obs = agent.add_agent_info_to_obs(obs)
        rec["done"].append(dones.bool().cpu().numpy())
        if (step + 1) % 100 == 0:
            log.info("step %d/%d", step + 1, args.steps)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **{k: np.stack(v, axis=0) for k, v in rec.items()},
        spec_names=np.array([n for n, _ in specs]),
        motion_names=np.array(
            [Path(f).stem for f in env.motion_lib.motion_files][:num_envs]
        ),
        body_names=np.array(env.contact_observation_body_names),
        total_mass=np.array(total_mass),
        mean_gap=np.array(mean_gap),
        dt=np.array(float(env.dt)),
    )
    log.info("wrote %s", out_path)
    if hasattr(env.simulator, "shutdown"):
        env.simulator.shutdown()


if __name__ == "__main__":
    main()
