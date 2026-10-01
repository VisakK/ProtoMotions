# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``.motion`` writer on plant v2 (BodyFix Step 3, part 1): the performer's own MoSh++ fit on her own
skeleton, written as a ProtoMotions reference with nothing changed.

What it writes, per clip (``fit_motion``):

* **Coordinates** from ``mosh_replay.mosh_kinematics(hand="fingers", representative="nearest")`` on the plant
  (``xml_path`` is passed explicitly: the module's default skeleton is bound to ``ids.MJCF`` at import). The
  root is her posed pelvis joint: ``LBS pelvis + trans + (0, 0, +10.17 mm) + capture.VICON_TO_CLIP_XY``; every
  body's world rotation is her SMPL-X joint's times ``retarget.AXES`` and the hand bodies take the chordal mean
  of their four proximal finger joints. The joint coordinates are ``rotvec(R_parent^T R_child)``, PhysX's own
  coordinates; they are written as the fit gives them, so they may lie outside the plant's joint box
  (``retarget_v2`` brings them in).
* **Every field** by the converter's path (``convert_yoga_frames_to_proto.convert_one``): qpos = [root, root
  quaternion wxyz, 69 exp-map coordinates] in float32 -> ``pose_lib.extract_transforms_from_qpos`` ->
  ``fk_from_transforms_with_velocities`` (velocity horizon 3) -> ``dof_vel`` from ``compute_angular_velocity``
  and ``local_rigid_body_rot`` from the joint matrices. BodyFix Step 2 measured this FK against PhysX's to
  1.4e-6 m (``plant_v2_physx.poselib_fk``, whose lines this copies: that module launches IsaacLab at import).
  The converter's ``fix_height`` is **not** applied and nothing is grounded per frame: the root stays hers.
* ``rigid_body_contacts``: geometric, a body whose lowest collision surface is within ``AVATAR_TOUCH_M`` of the
  floor (the retargeted motions' convention since Step 8; the shipped speed-and-height heuristic was 0-16 %
  precise).
* ``plant_sha256``: the plant's MJCF sha256 (``plant_identity``), without which MotionLib on ``smpl_yogi_v2``
  refuses the library.

The corpus (``corpus``) is the source manifest's clips with a readable fit, minus the lotus clips (TODO B1:
the plant's knee range cannot fold a lotus), Standing big toe hold -c (TODO B2: its fit on disk is unreadable)
and Firefly -b (its trunk folds 24.9 deg past the plant's box; BodyFix Step 3): 56 clips.

CLI (writes ``output/reference_curation/fit_writer/<fit_id>/<stem>.motion`` and the record
``data/reference_curation/fit_writer/<fit_id>/fit.json``)::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.fit_writer --all [--workers 8]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import torch

from reference_curation import ids
from reference_curation import mosh_replay as mr

REPO = ids.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from protomotions.utils import plant_identity  # noqa: E402

MODULE = "reference_curation.fit_writer"
SCHEMA_VERSION = 1
WRITER_VERSION = "v1"
PLANT = "v2"
OUT_ROOT = ids.OUTPUT_ROOT / "fit_writer"
RECORD_ROOT = ids.DATA_ROOT / "fit_writer"
AVATAR_TOUCH_M = 0.01          # retarget.AVATAR_TOUCH_M: a body touches at <= 1 cm
VELOCITY_MAX_HORIZON = 3       # convert_yoga_frames_to_proto.py
HAND = "fingers"
ROUND_TRIP_M = 1e-5
# BodyFix §5 "How this relates to the existing plan": B1 and B2 are decided.
DROPPED = {
    "220923_Cockerel_Pose-b": "lotus: the plant's knee range cannot fold a lotus (TODO B1)",
    "220923_Scale_Pose_or_Tolasana_-a": "lotus: the plant's knee range cannot fold a lotus (TODO B1)",
    "220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c": "no readable MoSh fit (TODO B2)",
    "220923_Firefly_Pose_or_Tittibhasana_-b": ("past the plant's joint box: the trunk folds 24.9 deg past Torso_y's 90 deg "
                                               "stop with the thighs wedged into the upper arms; every retarget jerked or "
                                               "kept 14-22 new overlaps (BodyFix Step 3; the user's decision, 2026-09-30)"),
}


# --------------------------------------------------------------------------- #
# The plant
# --------------------------------------------------------------------------- #
def plant_paths(plant: str = PLANT) -> tuple[Path, Path]:
    """``(mjcf, flat)`` of a registered plant (``plant_identity.PLANTS``)."""
    return plant_identity.mjcf_path(plant), plant_identity.flat_path(plant)


@functools.lru_cache(maxsize=4)
def kinematic_info(xml: str):
    from protomotions.components.pose_lib import extract_kinematic_info

    return extract_kinematic_info(xml)


def skeleton(plant: str = PLANT):
    """``mosh_replay.skeleton_for`` the plant: its offsets, joint box and typed colliders."""
    return mr.skeleton_for(plant_paths(plant)[0])


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
def corpus(manifest: Path = ids.DEFAULT_MANIFEST) -> tuple[list[str], dict]:
    """``(stems, dropped)``: the manifest's clips the writer covers, and ``{stem: reason}`` for the rest."""
    from reference_curation import human_mesh as hm

    stems, dropped = [], {}
    for stem in ids.manifest_stems(manifest):
        if stem in DROPPED:
            dropped[stem] = DROPPED[stem]
            continue
        fit, status, err = hm.load_fit(stem)
        if fit is None:
            dropped[stem] = f"{status}: {err}"
            continue
        stems.append(stem)
    return stems, dropped


# --------------------------------------------------------------------------- #
# Plant coordinates -> .motion
# --------------------------------------------------------------------------- #
def fit_coordinates(stem: str, plant: str = PLANT, frames=None) -> dict:
    """``mosh_replay.mosh_kinematics`` of the clip on the plant (fingers hand mode, the nearest exp-map
    representative against the plant's own box)."""
    return mr.mosh_kinematics(stem, hand=HAND, representative="nearest", xml_path=plant_paths(plant)[0],
                              frames=frames)


def body_lowest(pos: np.ndarray, rot_xyzw: np.ndarray, plant: str = PLANT) -> np.ndarray:
    """``[T, B]`` lowest collision surface of every body on the plant (``mosh_replay.body_lowest``)."""
    from scipy.spatial.transform import Rotation

    q = np.asarray(rot_xyzw, np.float64)
    rotm = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(*q.shape[:-1], 3, 3)
    return mr.body_lowest(skeleton(plant), np.asarray(pos, np.float64), rotm)


def write_motion(root_pos, root_rot, dof, fps: int, plant: str = PLANT) -> dict:
    """Every ``.motion`` field from plant coordinates: ``root_pos [T,3]``, ``root_rot [T,3,3]`` (or ``[T,4]``
    xyzw), ``dof [T,69]`` exp-map; the converter's path in float32 (module doc), no height fix, geometric
    contacts and the plant identity."""
    from protomotions.components.pose_lib import (compute_angular_velocity, extract_transforms_from_qpos,
                                                  fk_from_transforms_with_velocities)
    from protomotions.utils.rotations import matrix_to_quaternion

    kin = kinematic_info(str(plant_paths(plant)[0]))
    root_rot = torch.as_tensor(np.asarray(root_rot), dtype=torch.float64)
    q = root_rot if root_rot.shape[-1] == 4 else matrix_to_quaternion(root_rot, w_last=True)
    qpos = torch.cat([torch.as_tensor(np.asarray(root_pos), dtype=torch.float64), q[:, [3, 0, 1, 2]],
                      torch.as_tensor(np.asarray(dof), dtype=torch.float64)], 1).float()
    rp, jrm = extract_transforms_from_qpos(kin, qpos, qpos_is_exp_map_on_3dof_joints=True)
    st = fk_from_transforms_with_velocities(kinematic_info=kin, root_pos=rp, joint_rot_mats=jrm, fps=int(fps),
                                            compute_velocities=True, velocity_max_horizon=VELOCITY_MAX_HORIZON)
    st.dof_pos = qpos[:, 7:].clone()
    st.dof_vel = compute_angular_velocity(jrm[:, 1:], fps=int(fps)).reshape(qpos.shape[0], -1)
    st.local_rigid_body_rot = matrix_to_quaternion(jrm, w_last=True).clone()
    low = body_lowest(st.rigid_body_pos.double().numpy(), st.rigid_body_rot.double().numpy(), plant)
    st.rigid_body_contacts = torch.as_tensor(low <= AVATAR_TOUCH_M)
    out = st.to_dict()
    out["fps"] = int(fps)
    out[plant_identity.KEY] = plant_identity.sha256(plant)
    return out


def fit_motion(stem: str, plant: str = PLANT) -> tuple[dict, dict]:
    """``(motion, kin)``: the clip's fit written on the plant, and its coordinates (``fit_coordinates``)."""
    kin = fit_coordinates(stem, plant)
    fps = int(round(kin["fps"]))
    if abs(kin["fps"] - fps) > 1e-6:
        raise ValueError(f"{stem}: non-integer mocap rate {kin['fps']}")
    return write_motion(kin["root_pos"], kin["root_rot"], kin["dof"], fps, plant), kin


def round_trip(motion: dict, plant: str = PLANT) -> dict:
    """The stored joint coordinates reproduce the stored bodies through the plant's FK (float64,
    ``retarget.fk`` on ``skeleton(plant)``), ``local_rigid_body_rot`` is their exponential map, every field is
    finite and the plant identity is the plant's."""
    from protomotions.utils.rotations import quaternion_to_matrix

    from reference_curation import retarget as rt

    sk = skeleton(plant)
    dof = motion["dof_pos"].double()
    rot = quaternion_to_matrix(motion["rigid_body_rot"].double(), w_last=True)
    pos, rotm = rt.fk(sk, motion["rigid_body_pos"][:, 0].double(), rot[:, 0], dof)
    local = quaternion_to_matrix(motion["local_rigid_body_rot"].double(), w_last=True)
    finite = all(bool(torch.isfinite(v).all()) for v in motion.values() if torch.is_tensor(v) and v.is_floating_point())
    return {"pos_m": float((pos - motion["rigid_body_pos"].double()).abs().max()),
            "rot": float((rotm - rot).abs().max()),
            "local": float((local[:, 1:] - rt.so3_exp(dof.reshape(dof.shape[0], -1, 3))).abs().max()),
            "finite": finite, "plant": motion.get(plant_identity.KEY) == plant_identity.sha256(plant)}


def round_trip_ok(rt_: dict) -> bool:
    return rt_["finite"] and rt_["plant"] and max(rt_["pos_m"], rt_["rot"], rt_["local"]) <= ROUND_TRIP_M


def box_excess_deg(dof: np.ndarray, plant: str = PLANT) -> np.ndarray:
    """``[T, 69]`` degrees past the plant's joint box (its MJCF ranges, PhysX's hard box)."""
    sk = skeleton(plant)
    lo, hi = sk.lower.numpy(), sk.upper.numpy()
    return np.degrees(np.maximum(lo - dof, 0) + np.maximum(dof - hi, 0))


# --------------------------------------------------------------------------- #
# The corpus run and its record
# --------------------------------------------------------------------------- #
def generators() -> list[Path]:
    """The code that decides what the writer writes."""
    from protomotions.components import pose_lib

    from reference_curation import human_mesh as hm

    return [Path(__file__), Path(mr.__file__), Path(hm.__file__), Path(pose_lib.__file__)]


def fit_id(stems: list[str], plant: str = PLANT) -> str:
    from reference_curation import human_mesh as hm

    key = {"schema": SCHEMA_VERSION, "version": WRITER_VERSION, "hand": HAND, "plant": plant_identity.sha256(plant),
           "model": ids.sha256_file(hm.MODEL_PATH), "generators": {p.name: ids.sha256_file(p) for p in generators()},
           "fits": {s: ids.sha256_file(ids.mosh_path(s)) for s in sorted(stems)}}
    return f"fit_{plant}.{WRITER_VERSION}.{ids.sha256_json(key)[:10]}"


def _job(job: tuple) -> tuple[str, dict | None, str | None]:
    stem, out_dir, plant = job
    torch.set_num_threads(1)
    try:
        t0 = time.time()
        motion, kin = fit_motion(stem, plant)
        path = Path(out_dir) / f"{stem}.motion"
        torch.save(motion, path)
        exc = box_excess_deg(motion["dof_pos"].double().numpy(), plant)
        rtp = round_trip(torch.load(path, map_location="cpu", weights_only=False), plant)
        low = body_lowest(motion["rigid_body_pos"].double().numpy(), motion["rigid_body_rot"].double().numpy(), plant)
        rec = {"stem": stem, "frames": int(motion["dof_pos"].shape[0]), "fps": int(motion["fps"]),
               "sha256": ids.sha256_file(path), "fit": ids.display_path(ids.mosh_path(stem)),
               "frames_past_box_1deg": int((exc > 1).any(1).sum()), "frames_past_box_2deg": int((exc > 2).any(1).sum()),
               "box_excess_max_deg": round(float(exc.max()), 2), "lowest_min_cm": round(100 * float(low.min()), 2),
               "lowest_p50_cm": round(100 * float(np.median(low.min(1))), 2), "round_trip": rtp,
               "seconds": round(time.time() - t0, 1)}
        return stem, rec, None if round_trip_ok(rtp) else f"round trip {rtp}"
    except Exception as exc:  # noqa: BLE001 -- report every broken clip, then fail
        import traceback

        return stem, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"


def build(stems: list[str], dropped: dict, plant: str = PLANT, workers: int = 8, out_root: Path = OUT_ROOT,
          record_root: Path = RECORD_ROOT, force: bool = False) -> tuple[Path, dict, list[str]]:
    """Write every clip's fit motion into ``out_root/<fit_id>/`` (kept when its record exists and ``force`` is
    off) and the record into ``record_root/<fit_id>/fit.json``. Returns ``(motion dir, record, failures)``."""
    from reference_curation import human_mesh as hm

    fid = fit_id(stems, plant)
    out, rec_dir = Path(out_root) / fid, Path(record_root) / fid
    rec_path = rec_dir / "fit.json"
    if rec_path.exists() and not force and all((out / f"{s}.motion").exists() for s in stems):
        rec = json.loads(rec_path.read_text())
        bad = [s for s, c in rec["clips"].items() if ids.sha256_file(out / f"{s}.motion") != c["sha256"]]
        if not bad:
            return out, rec, []
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, str(out), plant) for s in stems]
    t0 = time.time()
    if workers > 1:
        with hm._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    failures = [f"{s}: {e}" for s, _, e in results if e]
    clips = {s: r for s, r, _ in results if r is not None}
    rec = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [plant_paths(plant)[0], hm.MODEL_PATH]),
           "fit_id": fid, "plant": plant_identity.identity(plant), "hand": HAND, "out_dir": ids.display_path(out),
           "generators": {ids.display_path(p): ids.sha256_file(p) for p in generators()},
           "dropped": dropped, "clips": clips,
           "totals": {"clips": len(clips), "frames": sum(c["frames"] for c in clips.values()),
                      "frames_past_box_1deg": sum(c["frames_past_box_1deg"] for c in clips.values()),
                      "frames_past_box_2deg": sum(c["frames_past_box_2deg"] for c in clips.values())},
           "seconds": round(time.time() - t0, 1)}
    if not failures:
        rec_dir.mkdir(parents=True, exist_ok=True)
        rec_path.write_text(json.dumps(rec, indent=1) + "\n")
    return out, rec, failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--all", action="store_true", help="every clip of the corpus (``corpus``)")
    what.add_argument("--stem", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rewrite even when the record exists")
    args = ap.parse_args(argv)
    try:
        stems, dropped = corpus()
        if args.stem:
            unknown = [s for s in args.stem if s not in stems]
            if unknown:
                raise ValueError(f"not in the corpus: {unknown} (dropped: {[s for s in unknown if s in dropped]})")
            stems = list(args.stem)
        out, rec, failures = build(stems, dropped, workers=args.workers, force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    t = rec["totals"]
    print(f"fit writer {rec['fit_id']}: {t['clips']} clips, {t['frames']} frames, {len(rec['dropped'])} dropped; "
          f"frames past the joint box > 1 deg {t['frames_past_box_1deg']} (retarget_v2 brings them in); "
          f"{len(failures)} failures -> {ids.display_path(out)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
