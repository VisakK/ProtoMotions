"""What AMP is supposed to change, measured directly: posture against the human, and stillness in the holds.

For each rollout library (the in-training evaluator's: every motion from t = 0, deterministic actions, frame i = clip
time (i + 1) dt), on *tracked* frames only (every body within 0.5 m of the reference after the evaluator's frame-0
XY alignment), against the release reference at the same clip times:

* **posture**: per joint, the angle between the policy's parent-relative rotation and the human's (median over frames),
  and the systematic part of it -- the norm of the mean relative rotation vector, and |mean| / sd. At e15500 these
  were a median 10-15 deg with 3-6 deg droop-like biases in the loaded joints (``e2_amp/README.MD``);
* **stillness in holds** (x0 clips only, so no hold-extension insert enters either side): inside every hold window of
  ``holds_extended.yaml``, the body speed and acceleration (RMS and median) from central differences of body
  positions at the control rate (30 Hz), on the policy and on the reference at the same times.

Also per x0 clip, the median over tracked frames of the mean joint angle (``frame_mean_angle_p50_by_clip_x0``), and
the same over x0 frames inside hold windows and between them (``x0_frame_mean_angle_p50_holds`` / ``_between_holds``);
and, for comparison with the tracking reward's global terms, the global body-orientation error, the root's own and
the bodies' orientations in the root frame (``global_body_rot_err_p50``, ``root_rot_err_p50``,
``in_root_frame_body_rot_err_p50``).

The e15500 run's own late libraries (9,500-15,500) give the no-AMP trend over its last 6,000 epochs; it is not a
matched control (that is ``CONTROL=1``), only the background against which G1's change is read.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ../env_isaaclab/bin/python \\
        expert_revist/graph_growth_2026_10_03/g1_amp_eval/style_shift.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protomotions.components.motion_lib import MotionLib, MotionLibConfig  # noqa: E402
from protomotions.utils import rotations as R  # noqa: E402

HERE = Path(__file__).resolve().parent
RELEASE = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
E15 = "results/smpl_yogi_v2_expert56_a2dda5d2ac/results/predicted_motion_lib_epoch_{}.pt"
G1 = "results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/results/predicted_motion_lib_epoch_{}.pt"
DEFAULT_LIBS = ([f"e15500:{e}" for e in (9500, 11000, 12500, 14000, 15500)]
                + [f"g1:{e}" for e in (1, 1500, 3000, 4500)]
                + ["g1:5000=output/renderings/expert56_v2_amp_g1_e5000/eval_standalone_e5000/results/"
                   "predicted_motion_lib_epoch_0.pt"])
BODIES = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine",
          "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder",
          "R_Elbow", "R_Wrist", "R_Hand"]
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]
CH = list(range(1, 24))
PA = [PARENTS[j] for j in CH]
VARIANT = re.compile(r"_x\d+s$")
TRACK_FAIL_M = 0.5


def local_from_world(rot: torch.Tensor) -> torch.Tensor:          # [N, 24, 4] xyzw -> [N, 23, 4]
    return R.quat_mul(R.quat_conjugate(rot[:, PA], True), rot[:, CH], True)


def rotvec_deg(q: torch.Tensor) -> torch.Tensor:                  # shortest-arc rotation vector, degrees
    q = torch.where(q[..., 3:4] < 0, -q, q)
    return torch.rad2deg(R.quat_to_exp_map(q.reshape(-1, 4), True)).reshape(q.shape[:-1] + (3,))


def angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:  # [..., 4] xyzw pairs -> angle, degrees
    d = R.quat_mul(R.quat_conjugate(q1.reshape(-1, 4), True), q2.reshape(-1, 4), True)
    return torch.rad2deg(2 * torch.atan2(d[:, :3].norm(dim=-1), d[:, 3].abs())).reshape(q1.shape[:-1])


def resolve(spec: str) -> tuple[str, str]:
    """``e15500:<epoch>`` / ``g1:<epoch>`` -> that run's library; ``label=path.pt`` -> that label and path (e.g.
    ``g1:5000=<eval_checkpoint out-dir>/results/predicted_motion_lib_epoch_0.pt``); ``label:path.pt`` likewise."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, path
    label, _, rest = spec.partition(":")
    if label in ("e15500", "g1") and rest.isdigit():
        return spec, (E15 if label == "e15500" else G1).format(rest)
    return (label, rest) if rest else (spec, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--libs", nargs="*", default=DEFAULT_LIBS,
                    help="run:epoch (run in e15500, g1) or label:path/to/library.pt")
    ap.add_argument("--out", default=str(HERE / "data/style_shift.json"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()

    rec = json.loads((REPO / "data/reference_curation/releases" / f"{RELEASE}.json").read_text())
    rdir = REPO / rec["dir"]
    clips = {c["stem"]: c for c in yaml.safe_load(open(rdir / "holds_extended.yaml"))["clips"]}
    ref = MotionLib(MotionLibConfig(motion_file=str(rdir / "motions.pt")), device="cpu")
    ref_stems = [Path(str(f)).name.rsplit(".motion", 1)[0] for f in ref.motion_files]

    out = {"release": RELEASE, "libraries": {}}
    for spec in args.libs:
        label, path = resolve(spec)
        lib = torch.load(REPO / path, map_location="cpu", weights_only=False)
        rel_all, fam_all, clip_angle = [], [], {}
        x0_hold_angle, x0_between_angle = [], []
        glob_err, root_err, inroot_err = [], [], []
        hold = {"agent_v": [], "ref_v": [], "agent_a": [], "ref_a": [], "group": []}
        n_frames = n_tracked = 0
        for i, f in enumerate(lib["motion_files"]):
            stem = Path(str(f)).stem
            a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
            dt = float(lib["motion_dt"][i])
            m = ref_stems.index(stem)
            t = (torch.arange(n, dtype=torch.float32) + 1.0) * dt
            t = t.clamp(max=float(ref.motion_lengths[m]))
            st = ref.get_motion_state(torch.full((n,), m, dtype=torch.long), t)
            pos = lib["gts"][a:a + n].float().clone()
            pos[..., :2] -= pos[0, 0, :2] - st.rigid_body_pos[0, 0, :2]
            ok = (pos - st.rigid_body_pos).norm(dim=-1).max(-1).values < TRACK_FAIL_M
            n_frames += n
            n_tracked += int(ok.sum())
            rel = R.quat_mul(R.quat_conjugate(local_from_world(st.rigid_body_rot.float()).reshape(-1, 4), True),
                             local_from_world(lib["grs"][a:a + n].float()).reshape(-1, 4), True).reshape(n, 23, 4)
            rel_all.append(rotvec_deg(rel)[ok])
            # global body orientation error, the root's own, and body orientations in the root frame (tracked frames)
            qa, qr = lib["grs"][a:a + n].float(), st.rigid_body_rot.float()
            g_err = angle_deg(qa, qr)                                           # [n, 24]
            glob_err.append(g_err[ok])
            root_err.append(g_err[ok][:, 0])
            la = R.quat_mul(R.quat_conjugate(qa[:, :1].expand_as(qa).reshape(-1, 4), True), qa.reshape(-1, 4), True)
            lr = R.quat_mul(R.quat_conjugate(qr[:, :1].expand_as(qr).reshape(-1, 4), True), qr.reshape(-1, 4), True)
            inroot_err.append(angle_deg(la.reshape(n, 24, 4), lr.reshape(n, 24, 4))[ok][:, 1:])
            fam_all += [clips[stem]["group"]] * int(ok.sum())
            if not VARIANT.search(stem) and int(ok.sum()) > 0:
                clip_angle[stem] = float(rel_all[-1].norm(dim=-1).mean(-1).median())

            if VARIANT.search(stem):
                continue                                   # stillness: x0 clips only (no frozen inserts)
            mask = torch.zeros(n, dtype=torch.bool)
            for h in clips[stem]["holds"] or []:
                mask |= (t >= h["t_start"]) & (t <= h["t_end"])
            frame_angle = rotvec_deg(rel).norm(dim=-1).mean(-1)            # [n] mean joint angle per frame
            x0_hold_angle.append(frame_angle[ok & mask])
            x0_between_angle.append(frame_angle[ok & ~mask])
            # central differences need frames i-1, i, i+1 all inside the hold and tracked
            inner = torch.zeros(n, dtype=torch.bool)
            inner[1:-1] = mask[1:-1] & mask[:-2] & mask[2:] & ok[1:-1] & ok[:-2] & ok[2:]
            idx = torch.nonzero(inner).squeeze(-1)
            if len(idx) == 0:
                continue
            rp = st.rigid_body_pos.float()
            for src, key_v, key_a in ((pos, "agent_v", "agent_a"), (rp, "ref_v", "ref_a")):
                v = (src[idx + 1] - src[idx - 1]) / (2 * dt)                       # [k, 24, 3]
                acc = (src[idx + 1] - 2 * src[idx] + src[idx - 1]) / dt ** 2
                hold[key_v].append(v.norm(dim=-1))
                hold[key_a].append(acc.norm(dim=-1))
            hold["group"] += [clips[stem]["group"]] * len(idx)

        d = torch.cat(rel_all)                              # [frames, 23, 3]
        fam = np.array(fam_all)
        ang = d.norm(dim=-1)                                # [frames, 23]
        mean, sd = d.mean(0), d.std(0)
        joints = {BODIES[CH[k]]: dict(median_deg=float(ang[:, k].median()), bias_deg=float(mean[k].norm()),
                                      bias_over_sd=float((mean[k].abs() / sd[k].clamp(min=1e-6)).max()),
                                      mean_vec=[round(float(x), 2) for x in mean[k]])
                  for k in range(23)}
        by_group = {}
        for g in sorted(set(fam)):
            sel = torch.from_numpy(fam == g)
            by_group[g] = float(ang[sel].mean(-1).median())
        res = dict(path=path, frames=n_frames, tracked=n_tracked,
                   posture=dict(median_joint_angle_deg=float(ang.median()),
                                frame_mean_angle_p50=float(ang.mean(-1).median()),
                                frame_mean_angle_by_group_p50=by_group,
                                mean_bias_deg=float(mean.norm(dim=-1).mean()), joints=joints,
                                frame_mean_angle_p50_by_clip_x0=clip_angle,
                                x0_frame_mean_angle_p50_holds=float(torch.cat(x0_hold_angle).median()),
                                x0_frame_mean_angle_p50_between_holds=float(torch.cat(x0_between_angle).median()),
                                global_body_rot_err_p50=float(torch.cat(glob_err).median()),
                                root_rot_err_p50=float(torch.cat(root_err).median()),
                                in_root_frame_body_rot_err_p50=float(torch.cat(inroot_err).median())))
        if hold["group"]:
            av, rv = torch.cat(hold["agent_v"]), torch.cat(hold["ref_v"])
            aa, ra = torch.cat(hold["agent_a"]), torch.cat(hold["ref_a"])
            grp = np.array(hold["group"])
            res["hold_stillness"] = dict(
                frames=int(len(grp)),
                speed_rms=dict(agent=float(av.pow(2).mean().sqrt()), reference=float(rv.pow(2).mean().sqrt())),
                speed_p50=dict(agent=float(av.median()), reference=float(rv.median())),
                accel_rms=dict(agent=float(aa.pow(2).mean().sqrt()), reference=float(ra.pow(2).mean().sqrt())),
                accel_p50=dict(agent=float(aa.median()), reference=float(ra.median())),
                accel_rms_by_body=dict(agent={BODIES[b]: float(aa[:, b].pow(2).mean().sqrt()) for b in range(24)},
                                       reference={BODIES[b]: float(ra[:, b].pow(2).mean().sqrt()) for b in range(24)}),
                by_group={g: dict(speed_rms_agent=float(av[torch.from_numpy(grp == g)].pow(2).mean().sqrt()),
                                  speed_rms_ref=float(rv[torch.from_numpy(grp == g)].pow(2).mean().sqrt()),
                                  accel_rms_agent=float(aa[torch.from_numpy(grp == g)].pow(2).mean().sqrt()),
                                  accel_rms_ref=float(ra[torch.from_numpy(grp == g)].pow(2).mean().sqrt()))
                          for g in sorted(set(grp))})
        out["libraries"][label] = res
        hs = res.get("hold_stillness", {})
        print(f"{label:14s} tracked {n_tracked / n_frames:.3f}  joint angle p50 {res['posture']['median_joint_angle_deg']:5.2f} deg"
              f"  frame-mean p50 {res['posture']['frame_mean_angle_p50']:5.2f}  mean bias {res['posture']['mean_bias_deg']:4.2f}"
              + (f" | holds: speed rms {hs['speed_rms']['agent']:.3f} (ref {hs['speed_rms']['reference']:.3f}) m/s,"
                 f" accel rms {hs['accel_rms']['agent']:.2f} (ref {hs['accel_rms']['reference']:.2f}) m/s2" if hs else "")
              + f"  ({time.time() - t0:.0f} s)", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    labels = list(out["libraries"])
    print("\nper joint: median angle to the human (deg) | systematic bias (deg)")
    print(f"{'joint':11s} " + " ".join(f"{lb:>13s}" for lb in labels))
    for j in [BODIES[c] for c in CH]:
        print(f"{j:11s} " + " ".join(f"{out['libraries'][lb]['posture']['joints'][j]['median_deg']:6.2f}|"
                                     f"{out['libraries'][lb]['posture']['joints'][j]['bias_deg']:5.2f}" for lb in labels))
    print(f"-> {args.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
