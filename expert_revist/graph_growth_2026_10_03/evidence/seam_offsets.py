"""Where every lane-T variant meets the real clips it would be spliced into (PLAN.MD §1.4, Step 3 build; card T6).

**Moved (card T6, 2026-10-04):** the measure lives in ``data/scripts/edge_synthesis/seams.py`` (``seams.legacy``),
beside T6's seam group (``seams.check``), and ``admit.py`` computes both for every variant. This script is kept to
reproduce ``seam_offsets.json`` (the MuJoCo variants of ``admitted.json``, before T6).

For each variant in ``admitted.json``:
* **start**: its first frame against S's exemplar frame, in S's own clip frame (the export does not move it);
* **end**: its last frame against D's exemplar, placed the way the generator placed it
  (``sketch.hand_anchor_transform``: D's hand midpoint on S's, D's hand line along S's).

Planted limbs are what matters at a seam: the hands at both ends, and the feet at the end of a landing edge (D = plank
or chaturanga). Also reports the hand widths and how far the planted hands drift inside the variant.

    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python \\
        expert_revist/graph_growth_2026_10_03/evidence/seam_offsets.py

Writes ``evidence/seam_offsets.json``. Measured 2026-10-04 (before T6): the quasi-static variants (E1, E3) have the
hands 4.2-5.9 cm off at both ends; the landing variants (E2, B1, E5) land the feet 4.6-15.7 cm from D's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data/scripts"))
from edge_synthesis import sketch as SK  # noqa: E402

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent
RELEASE_MOTIONS = REPO / "data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v2.a2dda5d2ac/motions"
NAMES = ("Pelvis L_Hip L_Knee L_Ankle L_Toe R_Hip R_Knee R_Ankle R_Toe Torso Spine Chest Neck Head L_Thorax "
         "L_Shoulder L_Elbow L_Wrist L_Hand R_Thorax R_Shoulder R_Elbow R_Wrist R_Hand").split()
BI = {n: i for i, n in enumerate(NAMES)}
HANDS = [BI["L_Hand"], BI["R_Hand"]]
FEET = [BI["L_Toe"], BI["R_Toe"], BI["L_Ankle"], BI["R_Ankle"]]
LANDING_DESTINATIONS = ("Plank_Pose", "Four-Limbed_Staff")


def main() -> int:
    edges = {e["id"]: e for e in json.loads((PLAN / "edges.json").read_text())["edges"]}
    admitted = json.loads((PLAN / "admitted.json").read_text())
    clips: dict[str, dict] = {}

    def clip(stem: str) -> dict:
        if stem not in clips:
            clips[stem] = torch.load(RELEASE_MOTIONS / f"{stem}.motion", map_location="cpu", weights_only=False)
        return clips[stem]

    def width(p: np.ndarray) -> float:
        return float(np.linalg.norm(p[BI["L_Hand"], :2] - p[BI["R_Hand"], :2]))

    out = []
    for v in admitted["variants"]:
        motion = torch.load(REPO / v["motion"], map_location="cpu", weights_only=False)
        e = edges[v["edge"]]
        s, d = e["source"], e["destination"]
        sp = clip(s["stem"])["rigid_body_pos"][s["frame_hold"]].double().numpy()
        dp = clip(d["stem"])["rigid_body_pos"][d["frame_hold"]].double().numpy()
        dr = clip(d["stem"])["rigid_body_rot"][d["frame_hold"]].double().numpy()
        yaw, t = SK.hand_anchor_transform(sp, dp, BI)
        dp2, _ = SK.apply_planar(dp, dr, yaw, t)
        pos = motion["rigid_body_pos"].double().numpy()
        landing = any(k in d["stem"] for k in LANDING_DESTINATIONS)
        out.append({
            "variant": v["variant"], "edge": v["edge"], "decision": v["decision"], "failed": v["failed"],
            "hands_start_m": float(np.linalg.norm(pos[0, HANDS] - sp[HANDS], axis=1).max()),
            "hands_end_m": float(np.linalg.norm(pos[-1, HANDS] - dp2[HANDS], axis=1).max()),
            "feet_end_m": float(np.linalg.norm(pos[-1, FEET] - dp2[FEET], axis=1).max()) if landing else None,
            "mean24_start_m": float(np.linalg.norm(pos[0] - sp, axis=1).mean()),
            "hand_width_m": {"synthetic": width(pos[0]), "S": width(sp), "D": width(dp)},
            "hand_drift_m": float(np.linalg.norm(pos[:, HANDS, :2] - pos[:1, HANDS, :2], axis=2).max()),
        })
    (HERE / "seam_offsets.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'variant':34s} {'decision':11s} {'hands S':>8s} {'hands D':>8s} {'feet D':>7s} {'width syn/S/D':>18s}")
    for r in out:
        feet = f"{100 * r['feet_end_m']:6.1f}" if r["feet_end_m"] is not None else "     -"
        w = r["hand_width_m"]
        print(f"{r['variant'][4:]:34s} {r['decision']:11s} {100 * r['hands_start_m']:7.1f}  {100 * r['hands_end_m']:7.1f} "
              f" {feet}  {w['synthetic']:.3f}/{w['S']:.3f}/{w['D']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
