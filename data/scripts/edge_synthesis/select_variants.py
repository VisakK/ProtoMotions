"""The variants Step 3 splices: the user's D1 decision (2026-10-04) applied to the PhysX executions (card T7).

**The decision.** E1 (crow -> handstand press), E3 (handstand -> crow lower), B1 (crow -> plank), E2 (crow ->
chaturanga jump-back) and E5 (handstand -> chaturanga float-down, via the plank) go into Step 3, with their PhysX
variants (``edge_mppi_physx``, ``t_edges_physx/README.MD``). E4 (tripod -> crow) and B2 (firefly -> crow) are
dropped. The user accepted E2 and E5 knowing that they end in a low chaturanga (0.07-0.09 m, 6-body, from the
exemplar; the human chaturanga is not held by MPPI in PhysX).

**The rule** that turns it into a variant list (this module's; the T5 groups are ``admit.py``'s):

====================  ======================================================================================
group                 how it is read
====================  ======================================================================================
contract              hard, every edge (box, floor, new overlaps, acceleration spikes; after de-penetration)
statics               hard, every edge (s* <= 1 on the quasi-static frames, executed contacts)
physx                 hard, every edge (round trip, MotionLib on plant v2; the reset check passed on all clips)
held                  hard, every edge: lane T's criterion over the final 2 s (the root within 10 cm of D's
                      height, tilt < 30 deg) -- a variant that leaves D during its own hold is not a hold
endpoints             hard for E1, E3, B1 (final 6-body error <= 0.05 m, COM speed <= 0.05 m/s, braces);
                      **advisory for E2 and E5** (the user's acceptance of the low chaturanga)
dynamics, naturalness advisory, as D1's default (every PhysX execution saturates some joint; jumps exceed p99)
====================  ======================================================================================

Each selected row carries the motion (path and sha256, checked against the admission record), its recipe (pass,
noise, timing, seed), the metrics R2/G3 read, the advisory failures, and the **seams**: the first frame's hands
against S's exemplar in S's clip frame, and the last frame's hands (and feet, for a landing) against D's exemplar
placed by the generator's hand-anchor transform (``evidence/seam_offsets.py``'s measure, on these motions).

Writes ``expert_revist/graph_growth_2026_10_03/selected_physx.json``.

CLI: ``CUDA_VISIBLE_DEVICES= PYTHONPATH=.:data/scripts python -m edge_synthesis.select_variants``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from edge_synthesis import sketch as SK
from reference_curation import ids

PLAN_DIR = ids.REPO / "expert_revist/graph_growth_2026_10_03"
ADMITTED = PLAN_DIR / "admitted_physx.json"
OUT = PLAN_DIR / "selected_physx.json"
DECISION = {
    "date": "2026-10-04",
    "by": "the user",
    "edges": ["E1", "E3", "B1", "E2", "E5"],
    "dropped": {"E4": "fails statics (as on MuJoCo); the executed head-lift is dynamic",
                "B2": "never reaches crow (as on MuJoCo)"},
    "endpoints_advisory": ["E2", "E5"],
    "note": "E2 and E5 end in a low chaturanga, 0.07-0.09 m (6-body) from the exemplar; accepted by the user",
}
HARD_ALWAYS = ("contract", "statics", "physx_cpu")
LANDING_DESTINATIONS = ("Plank_Pose", "Four-Limbed_Staff")
NAMES = ("Pelvis L_Hip L_Knee L_Ankle L_Toe R_Hip R_Knee R_Ankle R_Toe Torso Spine Chest Neck Head L_Thorax "
         "L_Shoulder L_Elbow L_Wrist L_Hand R_Thorax R_Shoulder R_Elbow R_Wrist R_Hand").split()
BI = {n: i for i, n in enumerate(NAMES)}
HANDS = [BI["L_Hand"], BI["R_Hand"]]
FEET = [BI["L_Toe"], BI["R_Toe"], BI["L_Ankle"], BI["R_Ankle"]]


def seams(motion: dict, e: dict) -> dict:
    """``evidence/seam_offsets.py``'s measure on one motion."""
    from edge_synthesis import plant_mj as pm

    s, d = e["source"], e["destination"]
    sp = pm.load_motion(s["stem"])["rigid_body_pos"][s["frame_hold"]].double().numpy()
    dm = pm.load_motion(d["stem"])
    dp = dm["rigid_body_pos"][d["frame_hold"]].double().numpy()
    dr = dm["rigid_body_rot"][d["frame_hold"]].double().numpy()
    yaw, t = SK.hand_anchor_transform(sp, dp, BI)
    dp2, _ = SK.apply_planar(dp, dr, yaw, t)
    pos = motion["rigid_body_pos"].double().numpy()
    landing = any(k in d["stem"] for k in LANDING_DESTINATIONS)

    def width(p):
        return round(float(np.linalg.norm(p[BI["L_Hand"], :2] - p[BI["R_Hand"], :2])), 4)
    return {"hands_start_cm": round(100 * float(np.linalg.norm(pos[0, HANDS] - sp[HANDS], axis=1).max()), 2),
            "hands_end_cm": round(100 * float(np.linalg.norm(pos[-1, HANDS] - dp2[HANDS], axis=1).max()), 2),
            "feet_end_cm": (round(100 * float(np.linalg.norm(pos[-1, FEET] - dp2[FEET], axis=1).max()), 2)
                            if landing else None),
            "hand_width_m": {"synthetic": width(pos[0]), "S": width(sp), "D": width(dp)},
            "hand_drift_cm": round(100 * float(np.linalg.norm(pos[:, HANDS, :2] - pos[:1, HANDS, :2], axis=2).max()), 2)}


def select(admitted: Path = ADMITTED) -> dict:
    rec = json.load(open(admitted))
    edges = {e["id"]: e for e in SK.load_edges()["edges"]}
    rows, rejected = [], []
    for v in rec["variants"]:
        if v["edge"] not in DECISION["edges"]:
            continue
        run_dir = ids.REPO / v["source_run"]
        run = json.load(open(run_dir / "run.json"))
        m = run["metrics"]
        hard = set(HARD_ALWAYS) | ({"endpoints"} if v["edge"] not in DECISION["endpoints_advisory"] else set())
        failed_hard = sorted(set(v["failed"]) & hard)
        if not m["held"]:
            failed_hard.append("held")
        if failed_hard:
            rejected.append({"variant": v["variant"], "edge": v["edge"], "failed_hard": failed_hard})
            continue
        mpath = ids.REPO / v["motion"]
        sha = ids.sha256_file(mpath)
        if sha != v["sha256"]:
            raise RuntimeError(f"{v['variant']}: the motion changed since admission ({sha} != {v['sha256']})")
        motion = torch.load(mpath, map_location="cpu", weights_only=False)
        rows.append({
            "variant": v["variant"], "edge": v["edge"], "label": edges[v["edge"]]["label"],
            "motion": v["motion"], "sha256": sha, "record": v["motion"].replace(".motion", ".json"),
            "source_run": v["source_run"], "frames": int(motion["dof_pos"].shape[0]), "fps": int(motion["fps"]),
            "recipe": {"pass": "px12" if v["variant"].endswith("_px12") else "px", "noise": run["mppi"]["noise"],
                       "timing": run["timing"], "durations_s": run["durations_s"], "T": run["T"],
                       "seed": run["seed"], "reference": run.get("reference"), "via": run.get("via"),
                       "start": run["start"]},
            "metrics": {"final_d6_m": m["final_d6_m"], "com_speed_end": m["com_speed_end"], "held": m["held"],
                        "landing_peak_bw": max(m["landing_peak_bw"].values(), default=0.0),
                        "hand_drift_cm": m["hand_drift_cm"],
                        "torque_saturated_share_max": v["dynamics"]["torque_saturated_share_max"],
                        "speed_over_p99_max": max(v["naturalness"]["speed_over_p99"].values(), default=0.0),
                        "statics_s_star_max": v["statics"]["s_star_max"], "free_violations": m["free_violations"]},
            "advisory_failed": sorted(set(v["failed"]) - hard),
            "seams": seams(motion, edges[v["edge"]]),
        })
    by = defaultdict(list)
    for r in rows:
        by[r["edge"]].append(r)
    summary = {}
    for edge in DECISION["edges"]:
        rs = by.get(edge, [])
        summary[edge] = {"variants": len(rs), "names": [r["variant"] for r in rs],
                         "final_d6_m": [r["metrics"]["final_d6_m"] for r in rs],
                         "hands_start_cm_max": max((r["seams"]["hands_start_cm"] for r in rs), default=None),
                         "hands_end_cm_max": max((r["seams"]["hands_end_cm"] for r in rs), default=None),
                         "feet_end_cm_max": max((r["seams"]["feet_end_cm"] or 0 for r in rs), default=None)}
    return {**ids.provenance(1, "edge_synthesis.select_variants", __file__, [admitted]),
            "kind": "step3_variant_selection", "decision": DECISION,
            "rule": {"hard_every_edge": list(HARD_ALWAYS) + ["held"],
                     "hard_unless_endpoints_advisory": ["endpoints"],
                     "advisory": ["dynamics", "naturalness_p99"],
                     "physx_reset_check": rec.get("physx_launch_check")},
            "summary": summary, "variants": rows, "rejected": rejected}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--admitted", type=Path, default=ADMITTED)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    out = select(args.admitted)
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    for edge, s in out["summary"].items():
        print(f"{edge}: {s['variants']} variants, final 6-body error {s['final_d6_m']}, seams: hands start <= "
              f"{s['hands_start_cm_max']} cm, hands end <= {s['hands_end_cm_max']} cm, feet end <= {s['feet_end_cm_max']} cm")
    print("wrote", ids.display_path(args.out), "--", len(out["variants"]), "variants;", len(out["rejected"]), "rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
