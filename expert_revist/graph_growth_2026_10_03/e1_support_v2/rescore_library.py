"""Card E1 acceptance: re-score a saved evaluator library with support rules v1 and v2, on the CPU.

The evaluator's own path, offline: the same packaged ``MotionLib`` interpolation, the same
``hold_curriculum.score_clip`` (v1, and v2 through its ``hold_violation`` masks), the same
``support_v2_holds`` / ``zone_lowest_points`` kernels and the same sidecar / manifest / graph mapping
(``HoldCurriculumEvaluator._setup_support_v2``).

What decides "loaded" on a known-free zone:

* ``force`` -- the runtime rule: terrain-filtered vertical load >= 3 % BW. Needs the library's
  ``sim_rigid_body_ground_forces``, which evaluators save from card E1 on (G1's libraries).
* ``proxy`` -- for libraries saved before that (e15500): the simulator's per-body contact flag on any of
  the zone's bodies AND the zone's lowest collider point within 2 cm of the floor. The flag alone
  includes body-body contact; the geometry alone counts a hover. A proxy, not the force rule.
* ``flag`` and ``geometry`` -- each half of the proxy alone, reported for comparison.

    PYTHONPATH=. python expert_revist/graph_growth_2026_10_03/e1_support_v2/rescore_library.py \\
        --library results/smpl_yogi_v2_expert56_a2dda5d2ac/results/predicted_motion_lib_epoch_15500.pt \\
        --csv results/smpl_yogi_v2_expert56_a2dda5d2ac/curriculum/eval_epoch_015500.csv \\
        --out expert_revist/graph_growth_2026_10_03/e1_support_v2/e15500.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protomotions.agents.evaluators.hold_curriculum import (  # noqa: E402
    GOAL_BODY_NAMES,
    SUPPORT_ZONES,
    HoldWindow,
    ScoreParams,
    SupportV2Params,
    score_clip,
    support_v2_holds,
    tracked_frames,
    zone_lowest_points,
    zone_vertical_load,
)
from protomotions.agents.evaluators.hold_curriculum_evaluator import HoldCurriculumEvaluator  # noqa: E402
from protomotions.components.contact_graph import ContactGraph  # noqa: E402
from protomotions.components.motion_lib import MotionLib, MotionLibConfig  # noqa: E402
from protomotions.envs.control.contact_targets import ContactTargets  # noqa: E402
from protomotions.envs.control.physics_terms import PhysicsTables  # noqa: E402

RELEASE = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"

# expert_revist/expert56_v2_e15500/README.MD §4: the ten substitution holds (stem fragment, t_hold, zones the
# policy puts down) and the one hover that must not count.
EXPECTED = [
    ("Eagle_Pose_or_Garudasana_-a", 13.33, "R_FOOT"),
    ("Eagle_Pose_or_Garudasana_-a", 11.67, "R_FOOT"),
    ("Eagle_Pose_or_Garudasana_-a", 7.23, "R_FOOT"),
    ("Shoulder-Pressing_Pose_or_Bhujapidasana_-a", 9.05, "L_FOOT"),
    ("Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c", 22.5, "R_FOOT"),
    ("Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a", 14.5, "R_FOOT"),
    ("Koundinyanasana_I_and_II-b", 19.63, "R_FOOT"),
    ("Koundinyanasana_I_and_II-b", 17.67, "R_FOOT"),
    ("Peacock_Pose_or_Mayurasana_-a", 13.13, "HEAD"),
    ("Upward_Plank_Pose_or_Purvottanasana_-a", 23.48, "HEAD"),
]
NOT_EXPECTED = [("Koundinyanasana_I_and_II-a", 20.35, "HEAD")]
CSV_V1 = ("p_track", "p_hold", "p_family", "support_violation", "score", "p_family_event")


def matches(stem: str, t_hold: float, spec) -> bool:
    return stem.endswith(spec[0]) and abs(t_hold - spec[1]) < 0.06


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", required=True)
    ap.add_argument("--csv", help="the evaluator's curriculum/eval_epoch_<N>.csv to check v1 against")
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synth-forces", action="store_true",
                    help="no saved forces: synthesise them as check_evaluator_path.py does (100 N on a body whose "
                         "contact flag is set while its own lowest collider point is within 2 cm) and run the "
                         "'force' mode on them -- a wiring-parity check against the evaluator path, not a proxy")
    args = ap.parse_args()

    rec = json.loads((REPO / "data/reference_curation/releases" / f"{args.release}.json").read_text())
    art = {k: REPO / v["path"] for k, v in rec["artifacts"].items()}
    manifest = yaml.safe_load(open(art["holds_extended"]))
    by_stem = {c["stem"]: c for c in manifest["clips"]}
    lib = torch.load(args.library, map_location="cpu", weights_only=False)
    stems = [Path(str(f)).stem for f in lib["motion_files"]]
    motion_lib = MotionLib(MotionLibConfig(motion_file=str(art["package"])), device="cpu")
    assert [Path(f).stem for f in motion_lib.motion_files] == stems, "library and package motion order differ"
    graph = ContactGraph.from_file(str(art["graph"]), device="cpu")
    payload = torch.load(art["physics_tables"], map_location="cpu", weights_only=False)
    body_names = list(payload["body_names"])
    tables = PhysicsTables(str(art["physics_tables"]), stems, body_names, "cpu")
    targets = ContactTargets(str(art["contact_targets"]), graph, stems, "cpu")

    # the evaluator's tables, built by the evaluator's own code
    zone_order = list(tables.zone_order)
    zone_body_ids = [[body_names.index(b) for b in tables.zone_bodies[z]] for z in zone_order]
    ground_pairs = [targets.pair_names.index(f"{z}:G") for z in zone_order]
    goal_ids = [body_names.index(b) for b in GOAL_BODY_NAMES]
    v1_zone_ids = {z: [body_names.index(b) for b in bodies] for z, bodies in SUPPORT_ZONES.items()}
    sp = ScoreParams()
    v2p = SupportV2Params(body_weight_n=float(tables.body_mass.sum()) * 9.81)
    forces = lib.get("sim_rigid_body_ground_forces")
    synth = forces is None and args.synth_forces
    modes = (["force"] if forces is not None or synth else []) + ["proxy", "flag", "geometry"]

    csv_rows = {}
    if args.csv:
        with open(args.csv) as f:
            csv_rows = {r["motion"]: r for r in csv.DictReader(f)}

    v1_mismatch, v1_cells, clips, totals = [], 0, [], {}
    for m, stem in enumerate(stems):
        clip = by_stem[stem]
        a, frames = int(lib["length_starts"][m]), int(lib["motion_num_frames"][m])
        if frames < 2:
            continue
        dt = float(lib["motion_dt"][m])
        sim = lib["gts"][a:a + frames].float()
        rot = lib["grs"][a:a + frames].float()
        flag = lib["contacts"][a:a + frames].bool()
        times = (torch.arange(frames, dtype=torch.float32) + 1.0) * dt
        ref = motion_lib.get_motion_state(torch.full((frames,), m, dtype=torch.long), times).rigid_body_pos.float()
        raw = clip["holds"]
        holds = [HoldWindow(float(h["t_hold"]), float(h["t_end"]), bool(h.get("extend", False)),
                            float(h.get("t_start", h["t_hold"]))) for h in raw]
        exemplars = None
        if holds:
            exemplars = motion_lib.get_motion_state(
                torch.full((len(holds),), m, dtype=torch.long),
                torch.tensor([h.t_hold for h in holds], dtype=torch.float32)).rigid_body_pos.float()

        # --- v1, against the evaluator's CSV
        s1 = score_clip(sim, ref, times, holds, exemplars, goal_ids, v1_zone_ids, sp)
        if stem in csv_rows:
            for k in CSV_V1:
                v1_cells += 1
                if f"{s1[k]:.4f}" != csv_rows[stem][k]:
                    v1_mismatch.append((stem, k, f"{s1[k]:.4f}", csv_rows[stem][k]))

        # --- v2
        commanded = torch.zeros(len(raw), len(zone_order), dtype=torch.bool)
        known_free = torch.zeros(len(raw), len(zone_order), dtype=torch.bool)
        for k, h in enumerate(raw):
            for pair in h.get("pairs_ground") or []:
                commanded[k, zone_order.index(pair.split(":")[0])] = True
            seg = HoldCurriculumEvaluator._hold_segment(graph, m, k, h, stem)
            known_free[k] = targets.ground_free[m, seg] & ~targets.masked[m, seg][ground_pairs]
        zone_low = zone_lowest_points(sim, rot, tables, zone_body_ids)
        tracked = tracked_frames(sim, ref, sp.track_fail_m)
        down = zone_low <= v2p.down_m
        zone_flag = torch.stack([flag[:, ids].any(dim=-1) for ids in zone_body_ids], dim=-1)
        loaded = {"proxy": zone_flag & down, "flag": zone_flag, "geometry": down}
        if forces is not None:
            f = forces[a:a + frames].float()
            loaded["force"] = zone_vertical_load(f, zone_body_ids) >= v2p.load_threshold_n
        elif synth:
            body_low = zone_lowest_points(sim, rot, tables, [[b] for b in range(len(body_names))])
            f = torch.zeros(frames, len(body_names), 3)
            f[..., 2] = 100.0 * (flag & (body_low <= 0.02)).float()
            loaded["force"] = zone_vertical_load(f, zone_body_ids) >= v2p.load_threshold_n
        per_mode = {}
        for mode in modes:
            rows, masks = support_v2_holds(times, loaded[mode], zone_low, holds, commanded, known_free, tracked, v2p)
            s2 = score_clip(sim, ref, times, holds, exemplars, goal_ids, v1_zone_ids, sp, hold_violation=masks)
            per_mode[mode] = (rows, s2)
        hold_out = []
        rows0 = per_mode[modes[0]][0]
        for k, h in enumerate(raw):
            r = rows0[k]
            if r is None:
                hold_out.append({"t_hold": h["t_hold"], "recorded": False})
                continue
            entry = {
                "t_hold": h["t_hold"], "hold_id": h.get("hold_id"), "recorded": True,
                "pose_role": (h.get("labels") or {}).get("pose_role"),
                "commanded": [p.split(":")[0] for p in h.get("pairs_ground") or []],
                "tracked_share": round(r["tracked_share"], 3), "realised": r["realised"],
                "support_share": {zone_order[z]: round(v, 3) for z, v in r["support_share"].items()},
            }
            for mode in modes:
                rm = per_mode[mode][0][k]
                entry[f"flagged_{mode}"] = [zone_order[z] for z in rm["flagged"]]
                entry[f"share_{mode}"] = {zone_order[z]: round(float(rm["zone_share"][z]), 3)
                                          for z in range(len(zone_order)) if float(rm["zone_share"][z]) >= 0.05}
            hold_out.append(entry)
        clips.append({
            "stem": stem, "group": clip["group"], "x0": float(clip.get("variant_s", 0.0) or 0.0) == 0.0,
            "v1": {k: round(float(s1[k]), 4) for k in CSV_V1},
            "v2": {mode: {k: round(float(per_mode[mode][1][k]), 4) for k in CSV_V1} for mode in modes},
            "holds": hold_out,
        })

    # --- rollups over x0 clips, tracked holds (README §4's population)
    report = {"library": args.library, "release": args.release, "modes": modes,
              "v1_csv": {"cells": v1_cells, "mismatches": len(v1_mismatch), "examples": v1_mismatch[:20]}}
    x0 = [c for c in clips if c["x0"]]
    tracked = [(c, h) for c in x0 for h in c["holds"] if h.get("recorded") and h["tracked_share"] >= v2p.tracked_share]
    with_cmd = [(c, h) for c, h in tracked if h["realised"] is not None]
    report["x0_holds"] = sum(len(c["holds"]) for c in x0)
    report["x0_holds_tracked"] = len(tracked)
    report["x0_supports_realised"] = sum(1 for _, h in with_cmd if h["realised"])
    report["x0_supports_scored"] = len(with_cmd)
    for mode in modes:
        flagged = [(c["stem"], h["t_hold"], h[f"flagged_{mode}"]) for c, h in tracked if h[f"flagged_{mode}"]]
        hit = [spec for spec in EXPECTED if any(matches(s, t, spec) and spec[2] in z for s, t, z in flagged)]
        miss = [spec for spec in EXPECTED if spec not in hit]
        false_pos = [x for x in flagged if not any(matches(x[0], x[1], spec) for spec in EXPECTED)]
        hover = [spec for spec in NOT_EXPECTED if any(matches(s, t, spec) and spec[2] in z for s, t, z in flagged)]
        report[f"mode_{mode}"] = {
            "flagged_tracked_holds": len(flagged), "expected_hit": len(hit), "expected_missed": miss,
            "not_in_readme": [(s[7:], t, z) for s, t, z in false_pos], "koundinya_a_hover_flagged": bool(hover),
            "flagged": [(s[7:], t, z) for s, t, z in flagged],
        }
    report["clips"] = clips
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")

    print(f"v1 vs CSV: {len(v1_mismatch)} / {v1_cells} cells differ at 4 decimals")
    for row in v1_mismatch[:10]:
        print("   ", row)
    print(f"x0: {report['x0_holds']} holds, {report['x0_holds_tracked']} tracked on >= 90 % of the window; "
          f"commanded supports realised on {report['x0_supports_realised']} / {report['x0_supports_scored']} "
          f"(README: 232 / 260)")
    for mode in modes:
        r = report[f"mode_{mode}"]
        print(f"[{mode}] flagged {r['flagged_tracked_holds']} tracked x0 holds; README's 10: hit "
              f"{r['expected_hit']}, missed {[(s, t) for s, t, _ in r['expected_missed']]}; "
              f"Koundinya -a hover flagged: {r['koundinya_a_hover_flagged']}")
        for x in r["not_in_readme"]:
            print(f"      not in README: {x}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
