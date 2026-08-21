# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record what a policy puts on the floor, clip by clip, in the reference clip's frame.

This is the simulated half of the measured-vs-learned pressure comparison in
:mod:`plot_pressure_compare`.  The measured half is the MOYO pressure mat
(``notes/Moyo_pressure_port.MD``); the mat's rectangle is stored **in the clip
frame**, so the only thing that makes the two comparable is getting the
simulated contacts back into that frame.  That is the whole point of this
recorder and it is asserted at every reset.

What it records, per physics substep (120 Hz, by hooking ``scene.update``):

* ``ground_fz`` -- per-body vertical force against the terrain, from
  ``RigidContactView.get_contact_force_matrix`` column 0.  Same channel the
  ``pressure_*`` reward terms read, so the figures and the reward agree.
* the ground **contact manifold** -- every contact point PhysX reported against
  the terrain, with its normal-force magnitude.  A per-body force has no
  location; this is what makes a bird's-eye pressure map possible at all.
* ``bb_force`` -- the magnitude of every body-body pair force, so "crow never
  makes the knee-on-upper-arm contact" is a number rather than an impression.
* body poses, centres of mass and velocities, for COM / XCoM overlays.

Differences from :mod:`record_contact_physics`, which records far more per clip:

* **one process, every clip.**  That recorder refuses more than one motion
  because the motion manager samples ids at ``reset()``; the fix used here is
  the one :mod:`render_policy_videos` uses -- pin ``motion_ids``, reset with
  ``disable_motion_resample=True``, and *assert* the manager is really playing
  the requested clip.  ``num_envs`` is still 1, which is the constraint that
  actually matters (with more envs the PhysX filter patterns silently resolve to
  nothing and every body-body column reads zero).
* it stores the reduced channels above rather than the full
  ``[T, 24, 25, 3]`` pair tensor, so 29 long clips fit in a few hundred MB.

Usage::

    python data/scripts/record_pressure_rollout.py \
      --checkpoint results/smpl_yogi_hard29_pressure_ab_s1/last.ckpt \
      --motion-file data/smpl/yoga_yogi_hard29_pressure.pt \
      --overrides env.ref_respawn_offset=0.005 \
      --out-dir results/hard29_pressure_rollouts
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
parser.add_argument("--motion-ids", type=int, nargs="*", default=None,
                    help="explicit motion ids (default: every clip in the library)")
parser.add_argument("--motion-names", type=str, nargs="*", default=None,
                    help="case-insensitive substrings; keep only clips matching one")
parser.add_argument("--max-seconds", type=float, default=0.0,
                    help="cap per-clip rollout length; 0 = the whole clip")
parser.add_argument("--max-contact-points", type=int, default=16,
                    help="PhysX manifold points kept per (body, filter) pair")
parser.add_argument("--settle-steps", type=int, default=0,
                    help="policy steps to run after reset before recording starts")
parser.add_argument("--registration-tol", type=float, default=0.02,
                    help="max allowed |sim pelvis xy - offset - reference pelvis xy| "
                         "at reset, in metres; above this the clip is skipped rather "
                         "than written into the wrong frame")
parser.add_argument("--skip-existing", action="store_true",
                    help="skip clips that already have a pressure_rollout.npz")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of the deterministic mean action")
args = parser.parse_args()

# The PhysX filter patterns only resolve for a single environment; with more the
# body-body columns come back empty and every pair force silently reads zero.
args.num_envs = 1
args.headless = True

from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pair_contact_view import PairContactView, gather_pair_buffer  # noqa: E402
from policy_setup import build, motion_names  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
_logger = logging.getLogger(__name__)


class _Log:
    """Isaac's app launcher reconfigures python logging; print survives it."""

    @staticmethod
    def info(fmt, *fmt_args) -> None:
        print("record_pressure_rollout: " + (fmt % fmt_args if fmt_args else fmt),
              flush=True)
        _logger.debug(fmt, *fmt_args)


log = _Log()


class GroundRecorder:
    """One sample per physics substep, hooked onto ``scene.update``."""

    def __init__(self, simulator, pair_view: PairContactView):
        self.robot = simulator._robot
        self.pair_view = pair_view
        self.num_bodies = pair_view.num_bodies
        self.num_filters = pair_view.view.filter_count
        self.dt_phys = float(simulator._sim.get_physics_dt())
        self.slot = pair_view.row_to_slot.to(simulator.device)
        self.armed = False
        self.reset_buffers()

    def reset_buffers(self) -> None:
        self.ground_fz: list = []      # [Ts, B]     vertical force vs terrain, N
        self.bb_force: list = []       # [Ts, B, B]  |body-body pair force|, N
        self.body_pos: list = []       # [Ts, B, 3]  link-frame origin, world
        self.body_quat: list = []      # [Ts, B, 4]  wxyz
        self.com_pos: list = []        # [Ts, B, 3]
        self.com_vel: list = []        # [Ts, B, 3]
        self.cp: dict[str, list] = {k: [] for k in ("step", "body", "pos", "force")}
        self.num_substeps = 0

    def _scatter(self, flat: torch.Tensor, trailing: tuple[int, ...]) -> torch.Tensor:
        out = torch.zeros((self.num_bodies, *trailing), device=flat.device,
                          dtype=flat.dtype)
        out[self.slot] = flat.reshape(-1, *trailing)
        return out

    def capture(self) -> None:
        if not self.armed:
            return
        step = self.num_substeps
        view = self.pair_view.view
        dt = self.dt_phys

        matrix = self._scatter(view.get_contact_force_matrix(dt=dt),
                               (self.num_filters, 3))          # [B, F, 3]
        self.ground_fz.append(matrix[:, 0, 2].to("cpu").numpy().astype(np.float32))
        self.bb_force.append(
            matrix[:, 1:, :].norm(dim=-1).to("cpu").numpy().astype(np.float32)
        )

        forces, points, normals, seps, counts, starts = view.get_contact_data(dt=dt)
        counts = counts.view(-1, self.num_filters)
        starts = starts.view(-1, self.num_filters)
        rows, cols, (pt, frc) = gather_pair_buffer(
            counts, starts, points, forces.view(-1, 1)
        )
        if rows.numel():
            # Filter 0 is the terrain; body-body points carry no ground load and
            # would put phantom pressure under a knee resting on a shoulder.
            keep = cols == 0
            if bool(keep.any()):
                rows_np = rows[keep].to("cpu").numpy()
                self.cp["step"].append(np.full(rows_np.shape, step, dtype=np.int32))
                self.cp["body"].append(
                    self.pair_view.row_body[rows_np].astype(np.int16)
                )
                self.cp["pos"].append(pt[keep].to("cpu").numpy().astype(np.float32))
                self.cp["force"].append(
                    frc[keep].to("cpu").numpy().reshape(-1).astype(np.float32)
                )

        data = self.robot.data
        self.body_pos.append(data.body_link_pos_w[0].to("cpu").numpy().astype(np.float32))
        self.body_quat.append(data.body_link_quat_w[0].to("cpu").numpy().astype(np.float32))
        self.com_pos.append(data.body_com_pos_w[0].to("cpu").numpy().astype(np.float32))
        self.com_vel.append(
            data.body_com_lin_vel_w[0].to("cpu").numpy().astype(np.float32)
        )
        self.num_substeps += 1

    def stacked(self) -> dict[str, np.ndarray]:
        out = {
            "ground_fz": np.stack(self.ground_fz),
            "bb_force": np.stack(self.bb_force),
            "body_pos_w": np.stack(self.body_pos),
            "body_quat_w": np.stack(self.body_quat),
            "body_com_pos_w": np.stack(self.com_pos),
            "body_com_vel_w": np.stack(self.com_vel),
        }
        for key, chunks in self.cp.items():
            name = f"cp_{key}"
            if chunks:
                out[name] = np.concatenate(chunks, axis=0)
            else:
                out[name] = np.zeros((0, 3) if key == "pos" else (0,), np.float32)
        return out


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
    if not ids:
        raise SystemExit("no motions selected")

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        before = len(ids)
        ids = [i for i in ids
               if not (out_root / names[i] / "pressure_rollout.npz").is_file()]
        if before != len(ids):
            log.info("--skip-existing: %d/%d clips already recorded",
                     before - len(ids), before)

    sim_body_names = list(simulator._robot.body_names)
    common_body_names = list(robot_config.kinematic_info.body_names)
    body_root_glob = robot_config.asset.usd_bodies_root_prim_path.replace(".*", "*")
    sensor = next(iter(simulator._contact_sensor_map.values()))
    pair_view = PairContactView(
        sensor._physics_sim_view,
        body_root_glob=body_root_glob,
        body_names=sim_body_names,
        num_envs=1,
        max_points_per_pair=args.max_contact_points,
    )
    log.info("pair contact view: %s", pair_view.describe())

    recorder = GroundRecorder(simulator, pair_view)
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

    for k, mid in enumerate(ids):
        name = names[mid]
        clip_len = float(lengths[mid])
        horizon = clip_len if args.max_seconds <= 0 else min(clip_len, args.max_seconds)
        steps = max(int(np.ceil(horizon / dt_ctrl)), 1)
        log.info("[%d/%d] motion %d %s (%.1fs -> %d policy steps)",
                 k + 1, len(ids), mid, name, horizon, steps)

        recorder.armed = False
        recorder.reset_buffers()

        # Pin the clip. Without `disable_motion_resample` the manager resamples
        # at reset and the rollout would be of some *other* clip under this
        # clip's name -- the reason record_contact_physics.py takes one clip per
        # process. Asserting after the reset is what makes one process safe.
        motion_manager = env.motion_manager
        motion_manager.motion_ids[env_ids] = mid
        motion_manager.motion_times[env_ids] = 0.0
        obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)
        playing = int(motion_manager.motion_ids[0])
        assert playing == mid, (
            f"motion manager is playing clip {playing} ({names[playing]}), "
            f"not the requested {mid} ({name})"
        )

        # THE registration. The measured pressure mat lives in the reference
        # clip's own frame; the robot is spawned at a terrain location that has
        # nothing to do with it. `respawn_root_offset[:2]` is the whole of the
        # XY difference (env.py: respawn_offset[:, :2] = spawn_xy - ref_root_xy),
        # so clip_xy = world_xy - offset. Verify it rather than trust it: at
        # t = 0 the simulated pelvis, mapped back, must land on the reference's
        # own frame-0 pelvis.
        offset_xy = env.respawn_root_offset[0, :2].detach().cpu().numpy().astype(
            np.float64
        )
        ref0 = motion_lib.get_motion_state(
            motion_manager.motion_ids, torch.zeros_like(motion_manager.motion_times)
        )
        ref_pelvis_xy = ref0.rigid_body_pos[0, 0, :2].detach().cpu().numpy()
        sim_pelvis_xy = (
            simulator.get_robot_state().rigid_body_pos[0, 0, :2].detach().cpu().numpy()
        )
        residual = float(np.linalg.norm((sim_pelvis_xy - offset_xy) - ref_pelvis_xy))
        if residual > args.registration_tol:
            log.info("    SKIP %s: registration residual %.4f m > %.4f m",
                     name, residual, args.registration_tol)
            skipped.append((name, residual))
            continue
        log.info("    registration residual %.2f mm (offset %.3f, %.3f)",
                 1000 * residual, offset_xy[0], offset_xy[1])

        agent.pre_collect_step(0)
        obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
        for _ in range(max(args.settle_steps, 0)):
            with torch.no_grad():
                model_outs = agent.model(obs_td)
            action = model_outs.get(action_key, model_outs.get("action"))
            obs, _, _, _, _ = env.step(action)
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

        ctrl: dict[str, list] = {k: [] for k in (
            "substep_index", "motion_time", "done", "terminated", "track_err",
            "root_drift",
        )}

        def ctrl_snapshot(done=None, terminated=None) -> None:
            state = simulator.get_robot_state()
            ref = motion_lib.get_motion_state(motion_manager.motion_ids,
                                              motion_manager.motion_times)
            ref_pos = ref.rigid_body_pos.clone()
            ref_pos = ref_pos + \
                env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(ref_pos)
            err = (ref_pos - state.rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]
            drift = (ref_pos[:, 0, :2] - state.rigid_body_pos[:, 0, :2]).norm(dim=-1)
            ctrl["substep_index"].append(recorder.num_substeps)
            ctrl["motion_time"].append(float(motion_manager.motion_times[0]))
            ctrl["done"].append(bool(done[0]) if done is not None else False)
            ctrl["terminated"].append(
                bool(terminated[0]) if terminated is not None else False
            )
            ctrl["track_err"].append(float(err[0]))
            ctrl["root_drift"].append(float(drift[0]))

        recorder.armed = True
        ctrl_snapshot()
        for step in range(steps):
            with torch.no_grad():
                model_outs = agent.model(obs_td)
            action = model_outs.get(action_key, model_outs.get("action"))
            obs, _, dones, terminated, _ = env.step(action)
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            ctrl_snapshot(done=dones, terminated=terminated)
            if bool(dones.any()):
                break
        recorder.armed = False

        dense = recorder.stacked()
        ctrl_arrays = {key: np.asarray(val) for key, val in ctrl.items()}
        n_sub = recorder.num_substeps
        n_ctrl = len(ctrl_arrays["motion_time"])

        motion_dir = out_root / name
        motion_dir.mkdir(parents=True, exist_ok=True)
        payload = {f"ctrl_{key}": val for key, val in ctrl_arrays.items()}
        payload.update(dense)
        np.savez_compressed(
            motion_dir / "pressure_rollout.npz",
            motion_name=np.array(name),
            motion_id=np.array(mid),
            motion_file=np.array(str(motion_lib.motion_files[mid])),
            motion_length_s=np.array(clip_len),
            checkpoint=np.array(str(args.checkpoint)),
            body_names=np.array(sim_body_names),
            common_body_names=np.array(common_body_names),
            body_masses=masses,
            # world_xy = clip_xy + respawn_offset_xy  ->  clip_xy = world - offset
            respawn_offset_xy=offset_xy,
            registration_residual_m=np.array(residual),
            dt_phys=np.array(recorder.dt_phys),
            dt_ctrl=np.array(dt_ctrl),
            # How many steps were burned before recording started. The video
            # renderer burns 2 by default, so this is what lets a composite of
            # the two be aligned instead of assumed aligned.
            settle_steps=np.array(int(args.settle_steps)),
            **payload,
        )
        log.info("    -> %s (%d substeps, %d policy steps, %d ground points)",
                 motion_dir / "pressure_rollout.npz", n_sub, n_ctrl,
                 int(dense["cp_step"].shape[0]))
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
