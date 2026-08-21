# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record only what the contact graph needs, as fast as the simulator allows.

:mod:`record_pressure_rollout` records the same rollout far more richly -- it
also gathers PhysX's per-point ground contact manifold, which is what makes a
bird's-eye pressure map possible.  That gather, plus a device-to-host copy of
six tensors on **every physics substep**, costs roughly twenty synchronisations
per policy step and puts the recorder at ~5 policy steps/s for a single
environment.  Recording 128 clips that way takes hours.

The contact graph needs four channels and no contact points:

    ground_fz      [Ts, B]     per-body vertical force against the terrain
    bb_force       [Ts, B, B]  magnitude of every body-body contact pair force
    body_quat_w    [Ts, B, 4]  for the trunk-orientation bin
    body_com_vel_w [Ts, B, 3]  for picking the most-static frame of a hold

so this recorder gathers those, keeps them **on the GPU** for the whole clip and
transfers once at the end.  The output npz is a strict subset of
``pressure_rollout.npz``'s keys, so
:mod:`build_contact_graph_from_rollouts` reads either without knowing which
produced it.

``num_envs`` is 1 for the same reason as the other recorders: with more than one
environment PhysX's filter patterns resolve to nothing and every body-body
column silently reads zero (see :class:`pair_contact_view.PairContactView`).

Usage::

    PYTHONPATH=. python data/scripts/record_contact_graph_rollouts.py \
      --checkpoint results/smpl_yogi_easy128_contact_rich/last.ckpt \
      --motion-file data/smpl/yoga_yogi_easy128_grounded.pt \
      --overrides env.ref_respawn_offset=0.005 \
      --out-dir results/easy128_contact_rollouts --skip-existing
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_setup import add_common_args  # noqa: E402  (no torch import yet)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--motion-ids", type=int, nargs="*", default=None)
parser.add_argument("--motion-names", type=str, nargs="*", default=None)
parser.add_argument("--skip-existing", action="store_true")
parser.add_argument("--registration-tol", type=float, default=0.05,
                    help="max |sim pelvis xy - offset - reference pelvis xy| at reset")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stochastic", action="store_true")
args = parser.parse_args()

# PhysX filter patterns only resolve for a single environment.
args.num_envs = 1
args.headless = True

from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pair_contact_view import PairContactView  # noqa: E402
from policy_setup import build, motion_names  # noqa: E402


class _Log:
    """Isaac's app launcher reconfigures python logging; print survives it."""

    @staticmethod
    def info(fmt, *fmt_args) -> None:
        print("record_contact_graph: " + (fmt % fmt_args if fmt_args else fmt),
              flush=True)


log = _Log()


class GraphRecorder:
    """One sample per physics substep, accumulated on device.

    Nothing leaves the GPU until :meth:`stacked`. That is the whole point: a
    per-substep ``.to("cpu")`` is a pipeline synchronisation, and there are four
    substeps per policy step.
    """

    def __init__(self, simulator, pair_view: PairContactView):
        self.robot = simulator._robot
        self.pair_view = pair_view
        self.num_bodies = pair_view.num_bodies
        self.num_filters = pair_view.view.filter_count
        self.dt_phys = float(simulator._sim.get_physics_dt())
        self.slot = pair_view.row_to_slot.to(simulator.device)
        self.device = simulator.device
        self.armed = False
        self.reset_buffers()

    def reset_buffers(self) -> None:
        self.ground_fz: list = []
        self.bb_force: list = []
        self.body_quat: list = []
        self.com_vel: list = []
        self.num_substeps = 0

    def _scatter(self, flat: torch.Tensor, trailing: tuple) -> torch.Tensor:
        out = torch.zeros((self.num_bodies, *trailing), device=flat.device,
                          dtype=flat.dtype)
        out[self.slot] = flat.reshape(-1, *trailing)
        return out

    def capture(self) -> None:
        if not self.armed:
            return
        matrix = self._scatter(
            self.pair_view.view.get_contact_force_matrix(dt=self.dt_phys),
            (self.num_filters, 3),
        )
        self.ground_fz.append(matrix[:, 0, 2])
        self.bb_force.append(matrix[:, 1:, :].norm(dim=-1))
        data = self.robot.data
        self.body_quat.append(data.body_link_quat_w[0])
        self.com_vel.append(data.body_com_lin_vel_w[0])
        self.num_substeps += 1

    def stacked(self) -> dict:
        def pack(values):
            return torch.stack(values).to("cpu").numpy().astype(np.float32)

        return {
            "ground_fz": pack(self.ground_fz),
            "bb_force": pack(self.bb_force),
            "body_quat_w": pack(self.body_quat),
            "body_com_vel_w": pack(self.com_vel),
        }


def _per_body_masses(robot) -> np.ndarray:
    for getter in (
        lambda: robot.root_physx_view.get_masses()[0],
        lambda: robot.data.default_mass[0],
    ):
        try:
            return np.asarray(getter().cpu().numpy(), dtype=np.float64)
        except Exception:  # pragma: no cover - backend dependent
            continue
    return np.zeros(0, dtype=np.float64)


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]
    motion_lib, simulator = built["motion_lib"], built["simulator"]
    robot_config = built["configs"]["robot"]

    names = motion_names(motion_lib)
    lengths = motion_lib.get_motion_length(None).detach().cpu()
    ids = args.motion_ids if args.motion_ids else list(range(len(names)))
    if args.motion_names:
        needles = [s.lower() for s in args.motion_names]
        ids = [i for i in ids if any(n in names[i].lower() for n in needles)]

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        before = len(ids)
        ids = [
            i for i in ids
            if not (out_root / names[i] / "pressure_rollout.npz").is_file()
            and not (out_root / names[i] / "contact_rollout.npz").is_file()
        ]
        log.info("--skip-existing: %d/%d clips already recorded", before - len(ids), before)
    if not ids:
        log.info("nothing to do")
        return 0

    sim_body_names = list(simulator._robot.body_names)
    common_body_names = list(robot_config.kinematic_info.body_names)
    body_root_glob = robot_config.asset.usd_bodies_root_prim_path.replace(".*", "*")
    sensor = next(iter(simulator._contact_sensor_map.values()))
    pair_view = PairContactView(
        sensor._physics_sim_view,
        body_root_glob=body_root_glob,
        body_names=sim_body_names,
        num_envs=1,
        # No contact points are gathered, so the manifold buffer can be minimal.
        max_points_per_pair=1,
    )
    log.info("pair contact view: sensors=%d filters=%d",
             pair_view.view.sensor_count, pair_view.view.filter_count)

    recorder = GraphRecorder(simulator, pair_view)
    scene = simulator._scene
    original_update = scene.update

    def patched_update(dt):
        original_update(dt)
        recorder.capture()

    scene.update = patched_update

    masses = _per_body_masses(simulator._robot)
    agent.eval()
    env_ids = torch.arange(env.num_envs, device=env.device)
    action_key = "action" if args.stochastic else "mean_action"
    dt_ctrl = float(env.dt)
    written, skipped = [], []
    started = time.time()

    for k, mid in enumerate(ids):
        name = names[mid]
        clip_len = float(lengths[mid])
        steps = max(int(np.ceil(clip_len / dt_ctrl)), 1)

        recorder.armed = False
        recorder.reset_buffers()

        motion_manager = env.motion_manager
        motion_manager.motion_ids[env_ids] = mid
        motion_manager.motion_times[env_ids] = 0.0
        obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)
        playing = int(motion_manager.motion_ids[0])
        assert playing == mid, (
            f"motion manager is playing clip {playing} ({names[playing]}), "
            f"not the requested {mid} ({name})"
        )

        offset_xy = env.respawn_root_offset[0, :2].detach().cpu().numpy().astype(np.float64)
        ref0 = motion_lib.get_motion_state(
            motion_manager.motion_ids, torch.zeros_like(motion_manager.motion_times)
        )
        ref_pelvis_xy = ref0.rigid_body_pos[0, 0, :2].detach().cpu().numpy()
        sim_pelvis_xy = (
            simulator.get_robot_state().rigid_body_pos[0, 0, :2].detach().cpu().numpy()
        )
        residual = float(np.linalg.norm((sim_pelvis_xy - offset_xy) - ref_pelvis_xy))
        if residual > args.registration_tol:
            log.info("[%d/%d] SKIP %s: registration residual %.4f m",
                     k + 1, len(ids), name, residual)
            skipped.append((name, residual))
            continue

        agent.pre_collect_step(0)
        obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

        # Per-policy-step control channels, kept on device and stacked once. The
        # loop therefore runs without a single host synchronisation.
        substep_index = [0]
        motion_time, track_err, done_flags = [], [], []

        def snapshot(done=None) -> None:
            state = simulator.get_robot_state()
            ref = motion_lib.get_motion_state(
                motion_manager.motion_ids, motion_manager.motion_times
            )
            ref_pos = ref.rigid_body_pos + \
                env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                    ref.rigid_body_pos
                )
            err = (ref_pos - state.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]
            motion_time.append(motion_manager.motion_times[0].clone())
            track_err.append(err[0].clone())
            done_flags.append(
                torch.zeros((), dtype=torch.bool, device=env.device)
                if done is None else done[0].clone()
            )

        recorder.armed = True
        snapshot()
        for _ in range(steps):
            with torch.no_grad():
                model_outs = agent.model(obs_td)
            action = model_outs.get(action_key, model_outs.get("action"))
            obs, _, dones, _terminated, _ = env.step(action)
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            substep_index.append(recorder.num_substeps)
            snapshot(done=dones)
        recorder.armed = False

        dense = recorder.stacked()
        motion_time_np = torch.stack(motion_time).cpu().numpy().astype(np.float64)
        track_err_np = torch.stack(track_err).cpu().numpy().astype(np.float64)
        done_np = torch.stack(done_flags).cpu().numpy()

        # The motion manager resets the track once the clip runs out; everything
        # after that belongs to a different episode and a different spawn.
        if done_np.any():
            end = int(np.argmax(done_np))
            n_ctrl = end + 1
            n_sub = min(int(substep_index[end]), recorder.num_substeps)
        else:
            n_ctrl = len(motion_time_np)
            n_sub = recorder.num_substeps

        motion_dir = out_root / name
        motion_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            motion_dir / "contact_rollout.npz",
            motion_name=np.array(name),
            motion_id=np.array(mid),
            motion_file=np.array(str(motion_lib.motion_files[mid])),
            motion_length_s=np.array(clip_len),
            checkpoint=np.array(str(args.checkpoint)),
            body_names=np.array(sim_body_names),
            common_body_names=np.array(common_body_names),
            body_masses=masses,
            respawn_offset_xy=offset_xy,
            registration_residual_m=np.array(residual),
            dt_phys=np.array(recorder.dt_phys),
            dt_ctrl=np.array(dt_ctrl),
            ctrl_substep_index=np.asarray(substep_index[:n_ctrl], dtype=np.int64),
            ctrl_motion_time=motion_time_np[:n_ctrl],
            ctrl_track_err=track_err_np[:n_ctrl],
            ctrl_done=done_np[:n_ctrl],
            **{key: value[:n_sub] for key, value in dense.items()},
        )
        elapsed = time.time() - started
        log.info("[%d/%d] %s -> %d substeps, %d steps (%.1f s elapsed, %.2f clips/min)",
                 k + 1, len(ids), name, n_sub, n_ctrl, elapsed,
                 60.0 * (k + 1) / max(elapsed, 1e-6))
        written.append(name)

    scene.update = original_update
    if hasattr(simulator, "shutdown"):
        simulator.shutdown()

    print("\n" + "=" * 70)
    print(f"recorded {len(written)}/{len(ids)} clips to {out_root}")
    for name, residual in skipped:
        print(f"  SKIPPED {name}: registration residual {residual:.4f} m")
    print("=" * 70)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
