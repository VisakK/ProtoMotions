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
seams                 card T6, on records that carry it (``admit``'s ``seams`` group): hard, every edge. Its end
                      seam is read ``attainable`` by default -- a planted body whose end offset is the exemplars'
                      own (S's palm kept, S's and D's palms 1.8-3.0 cm apart) passes and is listed -- or ``strict``
                      (``--seam-end strict``: every planted body within 1.5 cm of D's exemplar, which no variant
                      that keeps S's palms can meet on E1, E3, B1 or E2)
physx (reset)         hard where the record carries the reset check (``admit --physx-launch-check``)
====================  ======================================================================================

Each selected row carries the motion (path and sha256, checked against the admission record), its recipe (pass,
noise, timing, seed, reference, start), the metrics R2/G3 read, the advisory failures, the T6 seam group
(``seams_t6``) and the **seams** in the measure PLAN.MD's tables use (``seams.legacy``: the first frame's hands
against S's exemplar in S's clip frame, the last frame's hands -- and feet, for a landing -- against D's exemplar
placed by the generator's hand-anchor transform).

Writes ``expert_revist/graph_growth_2026_10_03/selected_physx.json``.

CLI: ``CUDA_VISIBLE_DEVICES= PYTHONPATH=.:data/scripts python -m edge_synthesis.select_variants [--admitted <json>]``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

from edge_synthesis import costs as C
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
T6_DECISION = {
    "date": "2026-10-04",
    "by": "the user",
    "drift_cm": 1.0,
    "seam_end": "attainable",
    "landing_slide": "accept",
    "note": "after card T6: 1) the planted-hand drift is read at 1.0 cm (the press's palms rock 0.7-1.3 cm "
            "transiently); 2) the palms' end offset is the exemplars' own 12 deg of yaw, read 'attainable': the "
            "policy learns around it; 3) the landings' slide is accepted (drift and end seams advisory on B1, E2, "
            "E5): the question is whether the policy can pick up the dynamic edges at all",
}
HARD_ALWAYS = ("contract", "statics", "physx_cpu")
SEAM_END = ("attainable", "strict")
LANDING_SLIDE = ("accept", "reject")


def seams(motion: dict, e: dict) -> dict:
    """``evidence/seam_offsets.py``'s measure on one motion (``seams.legacy``)."""
    from edge_synthesis import seams as SM

    return SM.legacy(motion, e)


def policy(seam_end: str = T6_DECISION["seam_end"], drift_cm: float | None = T6_DECISION["drift_cm"],
           landing_slide: str = T6_DECISION["landing_slide"]) -> dict:
    """The D1 rule as data (what ``verdict`` applies), with the user's T6 reading of the seam group."""
    if seam_end not in SEAM_END or landing_slide not in LANDING_SLIDE:
        raise ValueError((seam_end, landing_slide))
    from edge_synthesis import seams as SM
    return {"decision": DECISION, "t6_decision": T6_DECISION,
            "hard_every_edge": list(HARD_ALWAYS) + ["held", "seams", "physx_reset"],
            "hard_unless_endpoints_advisory": ["endpoints"], "advisory": ["dynamics", "naturalness_p99"],
            "seams_end": seam_end, "seams_drift_cm": SM.DRIFT_CM if drift_cm is None else drift_cm,
            "landing_slide": landing_slide,
            "note": "seams and physx_reset apply to records that carry them (T6 on); the end seam read 'attainable' "
                    "passes planted bodies whose offset to D's exemplar is the exemplars' own (seams.py); "
                    "seams_drift_cm re-reads the recorded drift (the card's gate is 0.5 cm); landing_slide 'accept' "
                    "keeps only the start seam hard on a landing edge (one that makes a support), its drift and "
                    "end seams advisory"}


def landing_edge(edge_id: str) -> bool:
    """An edge that makes a ground support (D's ground set holds a zone S's does not): B1, E2, E5."""
    e = SK.edge(SK.load_edges(), edge_id)
    return bool(set(e["destination"]["ground"]) - set(e["source"]["ground"]))


def seams_pass(sm: dict, seam_end: str = "attainable", drift_cm: float | None = None) -> bool:
    """The seam group as D1 reads it: the end seam literal or attainable; the drift at the card's 0.5 cm or at
    ``drift_cm`` (the recorded per-zone maxima, read again)."""
    end_ok = sm["end"]["pass"] if seam_end == "strict" else sm["end"]["pass_attainable"]
    drift_ok = sm["drift"]["pass"] if drift_cm is None else (
        not sm["drift"]["never_touched"] and sm["drift"]["max_cm"] <= drift_cm)
    if not end_ok and seam_end == "attainable" and drift_cm is not None:
        # a palm kept at S's pose is classified as the exemplars' own only within START_PLANTED_CM of S; with the
        # drift re-read, that allowance follows the drift threshold
        by = sm["end"]["by_body"]
        end_ok = all((by[b]["to_S_cm"] is not None and by[b]["to_S_cm"] <= drift_cm
                      and by[b]["exemplars_S_to_D_cm"] > sm["thresholds_cm"]["end_planted"])
                     for b in sm["end"]["beyond"])
    return sm["start"]["pass"] and end_ok and drift_ok


def verdict(v: dict, seam_end: str = T6_DECISION["seam_end"], held: bool | None = None,
            drift_cm: float | None = T6_DECISION["drift_cm"],
            landing_slide: str = T6_DECISION["landing_slide"]) -> dict:
    """The D1 decision on one admission row: ``{selected, failed_hard, advisory_failed}``."""
    hard = set(HARD_ALWAYS) | ({"endpoints"} if v["edge"] not in DECISION["endpoints_advisory"] else set())
    failed = set(v["failed"]) - {"seams"}
    failed_hard = sorted(failed & hard)
    advisory = sorted(failed - hard)
    held = v.get("held") if held is None else held
    if not held:
        failed_hard.append("held")
    sm = v.get("seams")
    if sm is not None:
        if landing_slide == "accept" and landing_edge(v["edge"]):
            if not sm["start"]["pass"]:
                failed_hard.append("seams")
            advisory += [f"seams_{k}" for k in ("drift", "end") if not sm[k]["pass"]]
        elif not seams_pass(sm, seam_end, drift_cm):
            failed_hard.append("seams")
    if (v.get("physx") or {}).get("pass_reset") is False:
        failed_hard.append("physx_reset")
    return {"selected": v["edge"] in DECISION["edges"] and not failed_hard, "failed_hard": failed_hard,
            "advisory_failed": advisory}


def select(admitted: Path = ADMITTED, seam_end: str = T6_DECISION["seam_end"],
           drift_cm: float | None = T6_DECISION["drift_cm"], landing_slide: str = T6_DECISION["landing_slide"]) -> dict:
    rec = json.load(open(admitted))
    edges = {e["id"]: e for e in SK.load_edges()["edges"]}
    rows, rejected = [], []
    for v in rec["variants"]:
        if v["edge"] not in DECISION["edges"]:
            continue
        run_dir = ids.REPO / v["source_run"]
        run = json.load(open(run_dir / "run.json"))
        m = run["metrics"]
        vd = verdict(v, seam_end, held=m["held"], drift_cm=drift_cm, landing_slide=landing_slide)
        if not vd["selected"]:
            row = {"variant": v["variant"], "edge": v["edge"], "failed_hard": vd["failed_hard"]}
            if "seams" in vd["failed_hard"]:
                row["seams"] = {k: v["seams"][k] for k in ("failed", "start", "drift")} | {
                    "end": {k: v["seams"]["end"].get(k) for k in ("planted_max_cm", "beyond", "end_exemplar_bodies",
                                                                  "reason")}}
            rejected.append(row)
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
            "recipe": {"pass": "px12" if v["variant"].endswith("px12") else "px", "noise": run["mppi"]["noise"],
                       "timing": run["timing"], "durations_s": run["durations_s"], "T": run["T"],
                       "seed": run["seed"], "reference": run.get("reference"), "via": run.get("via"),
                       "start": run["start"], "weights": {k: x for k, x in run["cost"]["weights"].items()
                                                          if x != getattr(C.EdgeWeights(), k)}},
            "metrics": {"final_d6_m": m["final_d6_m"], "com_speed_end": m["com_speed_end"], "held": m["held"],
                        "landing_peak_bw": max(m["landing_peak_bw"].values(), default=0.0),
                        "hand_drift_cm": m["hand_drift_cm"],
                        "torque_saturated_share_max": v["dynamics"]["torque_saturated_share_max"],
                        "speed_over_p99_max": max(v["naturalness"]["speed_over_p99"].values(), default=0.0),
                        "statics_s_star_max": v["statics"]["s_star_max"], "free_violations": m["free_violations"]},
            "advisory_failed": vd["advisory_failed"],
            "seams": seams(motion, edges[v["edge"]]),
        })
        if "seams" in v:
            sm = v["seams"]
            rows[-1]["seams_t6"] = {"pass": sm["pass"], "pass_attainable": sm["pass_attainable"],
                                    "start_all_max_cm": sm["start"]["all_max_cm"],
                                    "start_planted_max_cm": sm["start"]["planted_max_cm"],
                                    "end_planted_max_cm": sm["end"]["planted_max_cm"],
                                    "end_beyond": sm["end"]["beyond"],
                                    "end_exemplar_bodies": sm["end"]["end_exemplar_bodies"],
                                    "end_by_body": sm["end"]["by_body"], "drift_max_cm": sm["drift"]["max_cm"],
                                    "drift_zones": {z: r.get("max_cm") for z, r in sm["drift"]["zones"].items()}}
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
    t6 = any("seams" in v for v in rec["variants"])
    rule = policy(seam_end, drift_cm, landing_slide) if t6 else {"hard_every_edge": list(HARD_ALWAYS) + ["held"],
                                        "hard_unless_endpoints_advisory": ["endpoints"],
                                        "advisory": ["dynamics", "naturalness_p99"]}
    return {**ids.provenance(1, "edge_synthesis.select_variants", __file__, [admitted]),
            "kind": "step3_variant_selection", "decision": DECISION,
            "rule": {**rule, "physx_reset_check": rec.get("physx_launch_check")},
            "summary": summary, "variants": rows, "rejected": rejected}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--admitted", type=Path, default=ADMITTED)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seam-end", choices=SEAM_END, default=T6_DECISION["seam_end"])
    ap.add_argument("--drift-cm", type=float, default=T6_DECISION["drift_cm"],
                    help="re-read the recorded planted-body drift at this threshold (the user's 1.0 cm; the card's 0.5)")
    ap.add_argument("--landing-slide", choices=LANDING_SLIDE, default=T6_DECISION["landing_slide"],
                    help="accept (the user's): on B1, E2, E5 only the start seam is hard")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    out = select(args.admitted, args.seam_end, args.drift_cm, args.landing_slide)
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    for edge, s in out["summary"].items():
        print(f"{edge}: {s['variants']} variants, final 6-body error {s['final_d6_m']}, seams: hands start <= "
              f"{s['hands_start_cm_max']} cm, hands end <= {s['hands_end_cm_max']} cm, feet end <= {s['feet_end_cm_max']} cm")
    print("wrote", ids.display_path(args.out), "--", len(out["variants"]), "variants;", len(out["rejected"]), "rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
