# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the contact observation channels a trained policy actually receives.

Runs a contact-rich Stage-1 tracker in closed loop, one environment per
reference motion, and dumps -- for every step -- both the observation the model
consumed and the independent raw simulator state needed to check it.  Nothing
here interprets the numbers; :mod:`data/scripts/plot_contact_obs.py` does that.

The raw side is recorded specifically so the encoded channels can be falsified
rather than merely displayed:

* ``contact_forces`` / ``contact_flags`` invert the log compression,
* ``body_pos`` + ``ground_heights`` check the proximity vectors (on flat
  terrain every sampled terrain point shares one z, so the proximity z channel
  must equal minus the body height above ground),
* ``body_masses`` gives ``m*g``, which the summed upward force must match while
  a pose is actually being held.

Example::

    python data/scripts/record_contact_obs.py \
      --checkpoint results/smpl_yogi_balance_contact_rich_2/last.ckpt \
      --motion-file data/smpl/yoga_yogi_balance_grounded.pt \
      --steps 600 --out results/contact_obs_probe/rollout.npz
"""

from __future__ import annotations

import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--motion-file",
        type=str,
        required=True,
        help="Packaged .pt with N motions; one env is pinned to each motion.",
    )
    parser.add_argument("--simulator", type=str, default="isaaclab")
    parser.add_argument(
        "--steps", type=int, default=600, help="Policy steps to record (30 Hz)."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=0,
        help="0 (default) uses one environment per motion in the file.",
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--usd-asset",
        type=str,
        default=None,
        help="Override robot.asset.usd_asset_file_name (e.g. to match a checkpoint's plant).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the deterministic mean action.",
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
log = logging.getLogger("record_contact_obs")


def _body_masses(simulator) -> np.ndarray:
    """Per-body mass in kg, or an empty array when the backend cannot report it."""
    robot = getattr(simulator, "_robot", None)
    if robot is None:
        return np.zeros(0, dtype=np.float64)
    for getter in (
        lambda: robot.root_physx_view.get_masses()[0],
        lambda: robot.data.default_mass[0],
    ):
        try:
            return np.asarray(getter().cpu().numpy(), dtype=np.float64)
        except Exception:  # pragma: no cover - backend dependent
            continue
    return np.zeros(0, dtype=np.float64)


def main() -> None:
    global args
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    checkpoint = Path(args.checkpoint)
    cfg_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert cfg_path.exists(), f"missing {cfg_path}"
    cfgs = torch.load(cfg_path, map_location="cpu", weights_only=False)

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

    motion_path = Path(args.motion_file)
    if motion_path.is_dir():
        num_motions = len(sorted(motion_path.rglob("*.motion")))
    elif motion_path.suffix == ".motion":
        num_motions = 1
    else:
        num_motions = len(
            torch.load(args.motion_file, map_location="cpu", weights_only=False)[
                "motion_files"
            ]
        )
    num_envs = args.num_envs or num_motions
    assert num_envs <= num_motions, "one env per motion; cannot exceed the motion count"

    simulator_config.num_envs = num_envs
    simulator_config.headless = True
    if getattr(simulator_config, "domain_randomization", None) is not None:
        # Pushes and reset noise would show up as contact-force transients that
        # have nothing to do with the policy or the reference motion.
        simulator_config.domain_randomization.push = None
    robot_config.reset_noise = None

    # Pin env i to motion i and always start at t=0 so every trace is aligned
    # to the beginning of its clip and is reproducible.
    env_config.motion_manager.subset_method = list(range(num_envs))
    env_config.motion_manager.init_start_prob = 1.0
    # Terminations stay ON -- a fall is a real result -- but the episode-length
    # cap must not truncate the window we asked for.
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

    from protomotions.simulator.base_simulator.utils import (  # noqa: E402
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.agents.base_agent.agent import BaseAgent  # noqa: E402
    from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
    from protomotions.envs.obs import (  # noqa: E402
        CONTACT_OBS_V1_GLOBAL_LAYOUT,
        CONTACT_OBS_V1_LAYOUT,
        unflatten_contact_obs_v1,
    )
    from protomotions.utils.component_builder import build_all_components  # noqa: E402
    from protomotions.utils.hydra_replacement import get_class  # noqa: E402

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

    obs_body_names = list(env.contact_observation_body_names)
    obs_body_ids = env.contact_observation_body_ids
    num_obs_bodies = len(obs_body_names)
    all_body_names = list(robot_config.kinematic_info.body_names)
    motion_files = [Path(f).stem for f in env.motion_lib.motion_files]

    contact_params = dict(
        env_config.observation_components["contact_obs_v1"].static_params
    )
    contact_params.pop("body_ids", None)
    contact_params.pop("w_last", None)

    log.info("contact observation bodies (%d): %s", num_obs_bodies, obs_body_names)
    log.info("contact_obs_v1 params: %s", contact_params)

    rec: dict[str, list] = {k: [] for k in (
        "contact_obs",          # [T,E,K,17]
        "contact_global",       # [T,E,4]
        "proximity",            # [T,E,K,3]
        "contact_forces",       # [T,E,K,3]  raw world-frame net force
        "contact_flags",        # [T,E,K]    raw simulator contact booleans
        "body_pos",             # [T,E,K,3]
        "body_vel",             # [T,E,K,3]
        "root_rot",             # [T,E,4]
        "ground_heights",       # [T,E]
        "active_state",         # [T,E,K]    env-side hysteresis state
        "age_steps",            # [T,E,K]
        "air_age_steps",        # [T,E,K]
        "temporal_valid",       # [T,E]
        "motion_times",         # [T,E]
        "progress",             # [T,E]
        "done",                 # [T,E]
        "terminated",           # [T,E]
        "track_err",            # [T,E]      max per-body distance to reference
        "ref_contacts",         # [T,E,K]    stored (unreliable) clip labels
    )}

    def snapshot(obs_dict, done=None, terminated=None) -> None:
        state = env.simulator.get_robot_state()
        per_body, global_feat = unflatten_contact_obs_v1(
            obs_dict["contact_obs_v1"], num_obs_bodies
        )
        prox = obs_dict["contact_proximity_obs"].reshape(num_envs, num_obs_bodies, 3)

        manager = env.motion_manager
        ref = env.motion_lib.get_motion_state(manager.motion_ids, manager.motion_times)
        ref_pos = ref.rigid_body_pos.clone()
        ref_pos = ref_pos + env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_pos
        )
        err = (
            (ref_pos - state.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]
        )
        ref_contacts = getattr(ref, "rigid_body_contacts", None)
        if ref_contacts is None:
            ref_contacts = torch.zeros(
                num_envs, len(all_body_names), device=env.device
            )

        ground = env.terrain.get_ground_heights(state.rigid_body_pos[:, 0]).squeeze(-1)

        zeros_e = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        rec["contact_obs"].append(per_body.detach().cpu().numpy())
        rec["contact_global"].append(global_feat.detach().cpu().numpy())
        rec["proximity"].append(prox.detach().cpu().numpy())
        rec["contact_forces"].append(
            state.rigid_body_contact_forces[:, obs_body_ids].cpu().numpy()
        )
        rec["contact_flags"].append(
            state.rigid_body_contacts[:, obs_body_ids].float().cpu().numpy()
        )
        rec["body_pos"].append(state.rigid_body_pos[:, obs_body_ids].cpu().numpy())
        rec["body_vel"].append(state.rigid_body_vel[:, obs_body_ids].cpu().numpy())
        rec["root_rot"].append(state.rigid_body_rot[:, 0].cpu().numpy())
        rec["ground_heights"].append(ground.cpu().numpy())
        rec["active_state"].append(
            env.contact_active_state[:, obs_body_ids].float().cpu().numpy()
        )
        rec["age_steps"].append(env.contact_age_steps[:, obs_body_ids].cpu().numpy())
        rec["air_age_steps"].append(
            env.contact_air_age_steps[:, obs_body_ids].cpu().numpy()
        )
        rec["temporal_valid"].append(env.contact_temporal_valid.cpu().numpy())
        rec["motion_times"].append(manager.motion_times.cpu().numpy())
        rec["progress"].append(env.progress_buf.cpu().numpy())
        rec["done"].append(
            (zeros_e if done is None else done.bool()).cpu().numpy()
        )
        rec["terminated"].append(
            (zeros_e if terminated is None else terminated.bool()).cpu().numpy()
        )
        rec["track_err"].append(err.detach().cpu().numpy())
        rec["ref_contacts"].append(
            ref_contacts[:, obs_body_ids].float().cpu().numpy()
        )

    action_key = "action" if args.stochastic else "mean_action"

    obs, _ = env.reset()
    log.info(
        "motion ids per env: %s", env.motion_manager.motion_ids.cpu().tolist()
    )
    snapshot(obs)

    for step in range(args.steps):
        with torch.no_grad():
            td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            model_outs = agent.model(td)
            action = model_outs[action_key if action_key in model_outs else "action"]
        obs, _rew, dones, terminated, _extras = env.step(action)
        snapshot(obs, done=dones, terminated=terminated)
        if (step + 1) % 100 == 0:
            log.info("step %d/%d", step + 1, args.steps)

    masses = _body_masses(env.simulator)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **{k: np.stack(v, axis=0) for k, v in rec.items()},
        obs_body_names=np.array(obs_body_names),
        all_body_names=np.array(all_body_names),
        motion_names=np.array(motion_files[:num_envs]),
        motion_ids=env.motion_manager.motion_ids.cpu().numpy(),
        body_masses=masses,
        obs_body_ids=obs_body_ids.cpu().numpy(),
        dt=np.array(float(env.dt)),
        # needed to interpret the proximity channel's horizontal quantisation
        terrain_sample_width=np.array(float(terrain_config.sample_width)),
        terrain_samples_per_axis=np.array(float(terrain_config.num_samples_per_axis)),
        channel_names=np.array(list(CONTACT_OBS_V1_LAYOUT.keys())),
        channel_slices=np.array(
            [[s.start, s.stop] for s in CONTACT_OBS_V1_LAYOUT.values()]
        ),
        global_channel_names=np.array(list(CONTACT_OBS_V1_GLOBAL_LAYOUT.keys())),
        **{f"param_{k}": np.array(float(v)) for k, v in contact_params.items()},
    )
    log.info("wrote %s", out_path)

    if hasattr(env.simulator, "shutdown"):
        env.simulator.shutdown()


if __name__ == "__main__":
    main()
