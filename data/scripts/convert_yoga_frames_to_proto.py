# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert already-converted "yoga" SMPL motions into ProtoMotions v3 ``.motion``
files for a given SMPL MJCF robot asset (e.g. ``smpl_boxhands_lowtorque.xml``).

Each yoga file is a pickled dict::

    {"loop_mode": int, "fps": int, "frames": (T, 75) list/array}

The 75-dim frame is already in the robot's **Z-up qpos convention**::

    frame = [ root_pos(3), root_orient_axis_angle(3), 69 exp-map DOF ]

i.e. root translation is Z-up (pelvis height ~0.9 m when standing) and the 69
DOF values are exp-map triples for the 23 non-root 3-DOF joints, in the robot's
(MuJoCo) joint order.  So we simply rebuild a MuJoCo ``qpos`` and run the robot's
own FK -- **no** AMASS Y-up rotation and **no** SMPL->MuJoCo joint reindexing
(both would mis-orient the character).

Usage::

    python data/scripts/convert_yoga_frames_to_proto.py \
        --mjcf data/assets/smpl/smpl_boxhands_lowtorque.xml \
        --out-dir data/smpl/yoga_motions_proto \
        data/smpl/yoga_motions/220923_Boat_Pose_or_Paripurna_Navasana_-a  [more...]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
    compute_angular_velocity,
)
from protomotions.utils.rotations import matrix_to_quaternion

# contact_detection lives in data/scripts (added to sys.path by the caller / __file__)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_detection import compute_contact_labels_from_pos_and_vel

NUM_DOF = 69
FOOT_OFFSET = 0.015  # SMPL toe height in T-pose


def load_yoga_frames(path):
    """Return (frames[T,75], fps) or None if the file is not a yoga motion clip."""
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
    except Exception:
        return None  # not a pickle (e.g. the .csv log)
    if not isinstance(d, dict) or "frames" not in d:
        return None  # e.g. yoga.pkl is a {betas,body_pose,...} pose dataset, not a clip
    frames = np.asarray(d["frames"], dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != 75:
        return None
    return frames, int(d.get("fps", 30))


def frames_to_qpos(frames):
    """(T,75) yoga frame -> (T,76) MuJoCo qpos [pos3, quat4(wxyz), dof69]."""
    root_pos = frames[:, :3]
    root_aa = frames[:, 3:6]
    dof = frames[:, 6:75]  # 69 exp-map DOF (already MuJoCo order)
    q_xyzw = sRot.from_rotvec(root_aa).as_quat()          # scipy -> xyzw
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)  # -> wxyz
    qpos = np.concatenate([root_pos, q_wxyz, dof], axis=1)
    return torch.from_numpy(qpos).float()


def convert_one(path, kin, fps_override=None):
    loaded = load_yoga_frames(path)
    if loaded is None:
        return None
    frames, fps = loaded
    if fps_override is not None:
        fps = int(fps_override)  # MOYO yoga dicts mislabel 60 Hz as 30 (MimicKit bug)
    if frames.shape[0] < 2:
        return None
    qpos = frames_to_qpos(frames)
    root_pos, joint_rot_mats = extract_transforms_from_qpos(
        kin, qpos, qpos_is_exp_map_on_3dof_joints=True
    )
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kin,
        root_pos=root_pos,
        joint_rot_mats=joint_rot_mats,
        fps=fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )
    motion.dof_pos = qpos[:, 7:]  # 69 exp-map DOF
    motion.dof_vel = compute_angular_velocity(
        joint_rot_mats[:, 1:, :, :], fps=fps
    ).reshape(-1, NUM_DOF)
    motion.local_rigid_body_rot = matrix_to_quaternion(joint_rot_mats, w_last=True).clone()
    motion.fix_height(height_offset=FOOT_OFFSET)
    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)
    return motion, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motions", nargs="+")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--output-fps", type=int, default=None,
                    help="override clip fps; the MOYO yoga dicts mislabel their "
                         "60 Hz sampling as 30 (MimicKit converter bug), so pass "
                         "60 to fix clip duration and reference velocities")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kin = extract_kinematic_info(args.mjcf)

    written = 0
    for m in args.motions:
        m = Path(m)
        try:
            res = convert_one(m, kin, args.output_fps)
        except Exception as e:
            print(f"SKIP {m.name} (conversion failed: {type(e).__name__}: {e})")
            continue
        if res is None:
            print(f"SKIP {m.name} (not a yoga clip or <2 frames)")
            continue
        motion, fps = res
        out_path = out_dir / (m.name + ".motion")
        torch.save(motion.to_dict(), str(out_path))
        gp = motion.rigid_body_pos
        n = gp.shape[0]
        # head(13) - pelvis(0) z-delta as an orientation sanity signal
        hp_z = float((gp[:, 13, 2] - gp[:, 0, 2]).mean())
        print(
            f"OK   {m.name} -> {out_path.name}  frames={n} fps={fps} "
            f"len={n/fps:.2f}s  mean(head.z-pelvis.z)={hp_z:+.3f}"
        )
        written += 1

    print(f"\nWrote {written} .motion files to {out_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
