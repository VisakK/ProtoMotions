"""What the panel's replicas actually stand on, per goal: the support check the panel's pose and IoU metrics miss.

The panel's crow goal reads contact IoU ~0.2 even when it starts in a real crow (its body-body braces are
under-reported), so IoU cannot tell a crow from G1's one-foot forward fold (``g1_amp_eval/README.MD`` §7), and the
pose error is blind to support (memory ``pose-metric-blind-to-support``). This reads the first K replicas of every
plan dumped by ``run_panel_bestyaw.py --dump-positions K`` and, over each goal's hold window, reports per replica:

* which zones are down: hands (``L_Hand``/``R_Hand`` origin), feet (lowest of ``*_Ankle``/``*_Toe``), head, knees,
  each below 8 cm (the evaluator's v1 rule; flat floor, so z is height);
* the pelvis height;
* the best-yaw 6-body error to the goal's pose (``score_hold_attainment.best_yaw_dist``).

A crow is hands down with both feet up on >= 90 % of the window; a handstand the same with the pelvis above 0.9 m.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/panel_supports.py \\
        output/renderings/expert56_v2_g3_e3420/panel_e3420_bestyaw_dump/positions.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "data" / "scripts", REPO / "expert_revist" / "run1_gap_analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from propose_hold_manifest import COMMON_BODY_ORDER, GOAL_BODIES  # noqa: E402
from score_hold_attainment import best_yaw_dist  # noqa: E402

RELEASE_DIR = REPO / "data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v3.2f132f4299"
DOWN = 0.08
B = {n: i for i, n in enumerate(COMMON_BODY_ORDER)}
GID = [B[b] for b in GOAL_BODIES]


def down_shares(pos: np.ndarray) -> dict:
    """pos [T, 24, 3] -> per-frame booleans of the zones the classification uses."""
    z = pos[..., 2]
    feet_l = np.minimum(z[:, B["L_Ankle"]], z[:, B["L_Toe"]]) < DOWN
    feet_r = np.minimum(z[:, B["R_Ankle"]], z[:, B["R_Toe"]]) < DOWN
    return {"L_HAND": z[:, B["L_Hand"]] < DOWN, "R_HAND": z[:, B["R_Hand"]] < DOWN, "L_FOOT": feet_l,
            "R_FOOT": feet_r, "HEAD": z[:, B["Head"]] < DOWN, "L_KNEE": z[:, B["L_Knee"]] < DOWN,
            "R_KNEE": z[:, B["R_Knee"]] < DOWN}


def classify(d: dict, pelvis: float) -> str:
    hands = d["L_HAND"] >= 0.9 and d["R_HAND"] >= 0.9
    feet_up = d["L_FOOT"] <= 0.1 and d["R_FOOT"] <= 0.1
    feet_down = d["L_FOOT"] >= 0.9 and d["R_FOOT"] >= 0.9
    if d["HEAD"] >= 0.5:
        return "head down"
    if hands and feet_up:
        return "hands only (handstand)" if pelvis > 0.9 else "hands only (arm balance)"
    if hands and feet_down:
        return "hands + feet (plank family)" if pelvis < 0.75 else "hands + feet (fold / dog)"
    if hands:
        return "hands + one foot"
    if feet_down and not (d["L_HAND"] > 0.1 or d["R_HAND"] > 0.1):
        return "feet only (upright)" if pelvis > 0.7 else "feet only (squat / fold)"
    if (d["L_FOOT"] >= 0.9) != (d["R_FOOT"] >= 0.9) and d["L_HAND"] <= 0.1 and d["R_HAND"] <= 0.1:
        return "one foot"
    return "other"


def main(path: str) -> int:
    z = np.load(path, allow_pickle=False)
    pos, owner, ft = z["positions"], z["owner"], z["frame_times"]
    seqs, goals, poses, timing = list(z["sequences"]), list(z["goals"]), list(z["goal_pose"]), list(z["goal_timing"])
    names = [Path(m["file"]).name[: -len(".motion")] for m in yaml.safe_load(open(RELEASE_DIR / "motions.yaml"))["motions"]]
    cache: dict = {}

    def goal_rel(spec: str) -> np.ndarray:
        mid, t = spec.split("@")
        stem = names[int(mid)]
        if stem not in cache:
            cache[stem] = torch.load(RELEASE_DIR / "motions" / f"{stem}.motion", map_location="cpu", weights_only=False)
        m = cache[stem]
        rp = m["rigid_body_pos"].float().numpy()
        f = min(int(round(float(t) * int(m["fps"]))), len(rp) - 1)
        return rp[f][GID] - rp[f][0]

    out = {}
    for s, name in enumerate(seqs):
        cols = np.nonzero(owner == s)[0]
        g_names, g_pose = goals[s].split(";"), poses[s].split(";")
        g_time = [tuple(map(float, x.split(","))) for x in timing[s].split(";")]
        ends = np.cumsum([r + h for r, h in g_time])
        rows = []
        for gi, (gname, spec, (reach, hold)) in enumerate(zip(g_names, g_pose, g_time)):
            win = (ft >= ends[gi] - hold) & (ft < ends[gi])
            if not win.any():
                continue
            ex = goal_rel(spec)
            reps = []
            for c in cols:
                p = pos[win][:, c]
                d = {k: float(v.mean()) for k, v in down_shares(p).items()}
                pel = float(np.median(p[:, B["Pelvis"], 2]))
                err = float(np.median(best_yaw_dist(p[:, GID] - p[:, :1], ex)))
                reps.append({"down": d, "pelvis_z": round(pel, 3), "err6": round(err, 3), "class": classify(d, pel)})
            classes = {}
            for r in reps:
                classes[r["class"]] = classes.get(r["class"], 0) + 1
            rows.append({"goal": gname, "window_s": [round(float(ends[gi] - hold), 2), round(float(ends[gi]), 2)],
                         "classes": classes, "err6_p50": round(float(np.median([r["err6"] for r in reps])), 3),
                         "pelvis_z_p50": round(float(np.median([r["pelvis_z"] for r in reps])), 3),
                         "down_mean": {k: round(float(np.mean([r["down"][k] for r in reps])), 2) for k in reps[0]["down"]},
                         "replicas": reps})
        out[name] = rows
    dst = HERE / "data/panel_supports_e3420.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    for name, rows in out.items():
        for r in rows:
            dm = r["down_mean"]
            print(f"{name[:30]:30s} {r['goal'][:22]:22s} err {r['err6_p50']:.3f} pelvis {r['pelvis_z_p50']:.2f} "
                  f"hands {dm['L_HAND']:.2f}/{dm['R_HAND']:.2f} feet {dm['L_FOOT']:.2f}/{dm['R_FOOT']:.2f} "
                  f"head {dm['HEAD']:.2f} | {r['classes']}")
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
