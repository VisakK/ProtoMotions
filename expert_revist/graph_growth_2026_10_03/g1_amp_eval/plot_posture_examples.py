"""What the posture change looks like: the human reference, e15500 and G1 at the same clip instant.

For the x0 clips whose mean joint angle to the human improved most (``data/style_shift.json``), take the evaluator's
rollouts (e15500 epoch 15,500 and G1's latest library), find the tracked frame inside a hold window where G1's
improvement over e15500 is largest, and draw the three skeletons pelvis-aligned, each policy skeleton yaw-fitted to
the reference about the pelvis (so heading drift does not masquerade as posture). Two views per clip. Writes
``figures/posture_examples.png``.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ../env_isaaclab/bin/python \\
        expert_revist/graph_growth_2026_10_03/g1_amp_eval/plot_posture_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from protomotions.components.motion_lib import MotionLib, MotionLibConfig  # noqa: E402
from protomotions.utils import rotations as R  # noqa: E402

HERE = Path(__file__).resolve().parent
RELEASE = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
E15 = REPO / "results/smpl_yogi_v2_expert56_a2dda5d2ac/results/predicted_motion_lib_epoch_15500.pt"
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]
CH = list(range(1, 24))
PA = [PARENTS[j] for j in CH]
GREY, BLUE, ORANGE, INK, INK2 = "#9a9893", "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
N_CLIPS = 4


def local_angles(rot: torch.Tensor) -> torch.Tensor:                 # [n, 24, 4] -> [n, 23, 4] parent-relative
    return R.quat_mul(R.quat_conjugate(rot[:, PA], True), rot[:, CH], True)


def angle_to(q_ref: torch.Tensor, q: torch.Tensor) -> torch.Tensor:   # mean joint angle per frame, degrees
    d = R.quat_mul(R.quat_conjugate(q_ref.reshape(-1, 4), True), q.reshape(-1, 4), True)
    a = 2 * torch.atan2(d[:, :3].norm(dim=-1), d[:, 3].abs())
    return torch.rad2deg(a).reshape(q.shape[:-1]).mean(-1)


def yaw_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotate ``src`` [24, 3] about z (pelvis at origin) to best match ``dst`` in the horizontal plane."""
    a, b = src[:, :2], dst[:, :2]
    th = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    c, s = np.cos(th), np.sin(th)
    out = src.copy()
    out[:, 0], out[:, 1] = c * src[:, 0] - s * src[:, 1], s * src[:, 0] + c * src[:, 1]
    return out


def clip_frames(lib, stem):
    names = [Path(str(f)).stem for f in lib["motion_files"]]
    i = names.index(stem)
    a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
    return lib["gts"][a:a + n].float(), lib["grs"][a:a + n].float(), float(lib["motion_dt"][i])


def main() -> None:
    style = json.load(open(HERE / "data/style_shift.json"))["libraries"]
    g1_label = max((lb for lb in style if lb.startswith("g1:")), key=lambda lb: int(lb.split(":")[1]))
    g1_lib_path = REPO / style[g1_label]["path"]           # style_shift records each library's path
    a15 = style["e15500:15500"]["posture"]["frame_mean_angle_p50_by_clip_x0"]
    ag1 = style[g1_label]["posture"]["frame_mean_angle_p50_by_clip_x0"]
    gain = sorted(((a15[k] - ag1[k], k) for k in a15 if k in ag1), reverse=True)
    stems = [k for _, k in gain[:N_CLIPS]]

    rec = json.loads((REPO / "data/reference_curation/releases" / f"{RELEASE}.json").read_text())
    rdir = REPO / rec["dir"]
    clips = {c["stem"]: c for c in yaml.safe_load(open(rdir / "holds_extended.yaml"))["clips"]}
    ref = MotionLib(MotionLibConfig(motion_file=str(rdir / "motions.pt")), device="cpu")
    ref_stems = [Path(str(f)).name.rsplit(".motion", 1)[0] for f in ref.motion_files]
    lib15 = torch.load(E15, map_location="cpu", weights_only=False)
    libg1 = torch.load(g1_lib_path, map_location="cpu", weights_only=False)

    fig = plt.figure(figsize=(4.2 * N_CLIPS, 9.0), facecolor="#fcfcfb")
    for col, stem in enumerate(stems):
        p15, r15, dt = clip_frames(lib15, stem)
        pg1, rg1, _ = clip_frames(libg1, stem)
        n = min(len(p15), len(pg1))
        m = ref_stems.index(stem)
        t = ((torch.arange(n) + 1.0) * dt).clamp(max=float(ref.motion_lengths[m]))
        st = ref.get_motion_state(torch.full((n,), m, dtype=torch.long), t)
        rp, rr = st.rigid_body_pos.float(), st.rigid_body_rot.float()

        def tracked(p):
            q = p[:n].clone()
            q[..., :2] -= q[0, 0, :2] - rp[0, 0, :2]
            return (q - rp).norm(dim=-1).max(-1).values < 0.5

        ok = tracked(p15) & tracked(pg1)
        hold = torch.zeros(n, dtype=torch.bool)
        for h in clips[stem]["holds"] or []:
            hold |= (t >= h["t_start"]) & (t <= h["t_end"])
        e15 = angle_to(local_angles(rr), local_angles(r15[:n]))
        eg1 = angle_to(local_angles(rr), local_angles(rg1[:n]))
        score = torch.where(ok & hold, e15 - eg1, torch.full_like(e15, -1e9))
        k = int(score.argmax())
        skel = {name: (p - p[0]).numpy() for name, p in (("reference", rp[k]), ("e15500", p15[k]), ("G1", pg1[k]))}
        skel["e15500"] = yaw_fit(skel["e15500"], skel["reference"])
        skel["G1"] = yaw_fit(skel["G1"], skel["reference"])
        floor = float(rp[k, :, 2].min() - rp[k, 0, 2])
        title = stem[7:].split("_Pose_or_")[0].split("_pose_or_")[0].replace("_", " ") + " " + stem[-2:]
        for row, (elev, azim) in enumerate(((12, -60), (12, 30))):
            ax = fig.add_axes([col / N_CLIPS, 0.48 - row * 0.46, 1.0 / N_CLIPS, 0.44], projection="3d")
            ax.set_facecolor("#fcfcfb")
            for name, color, lw, z in (("reference", GREY, 6.0, 1), ("e15500", BLUE, 2.4, 2), ("G1", ORANGE, 2.4, 3)):
                sk = skel[name]
                for j, pj in zip(CH, PA):
                    ax.plot([sk[pj, 0], sk[j, 0]], [sk[pj, 1], sk[j, 1]], [sk[pj, 2], sk[j, 2]], color=color, lw=lw,
                            alpha=0.5 if name == "reference" else 0.95, zorder=z, solid_capstyle="round")
            allp = np.concatenate(list(skel.values()))
            c, r = allp.mean(0), (allp.max(0) - allp.min(0)).max() / 2 + 0.05
            gx, gy = np.meshgrid([c[0] - r, c[0] + r], [c[1] - r, c[1] + r])
            ax.plot_surface(gx, gy, np.full_like(gx, floor), color="#e1e0d9", alpha=0.35, zorder=0)
            ax.set_xlim(c[0] - r, c[0] + r)
            ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"{title}, t = {float(t[k]):.1f} s\nmean joint angle to the human\n"
                             f"{float(e15[k]):.1f}° (e15500) → {float(eg1[k]):.1f}° (G1)", fontsize=10, color=INK,
                             fontweight="bold", y=0.98)
    handles = [plt.Line2D([], [], color=GREY, lw=6, alpha=0.5, label="human reference"),
               plt.Line2D([], [], color=BLUE, lw=2.4, label="e15500, epoch 15,500"),
               plt.Line2D([], [], color=ORANGE, lw=2.4, label=f"G1, epoch {g1_label.split(':')[1]}")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.savefig(HERE / "figures/posture_examples.png", dpi=110, facecolor="#fcfcfb")
    print("clips:", ", ".join(stems), "->", HERE / "figures/posture_examples.png")


if __name__ == "__main__":
    main()
