# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realisability of policy rollouts on a release (BodyFix Step 5 / BUILD_PLAN Step 10): slide, float and COM vs COP.

The in-training evaluator (``HoldCurriculumEvaluator``) scores tracking and holds and logs drag and support violation.
Step 5's evaluation adds three quantities its references were curated for, measured on a policy's own rollouts
the way Steps 3-4 measured them on the references:

* **slide** (TODO A1's metric): on feet and hands the human keeps on the floor at that moment (her evidence,
  ``sources.ground_source``, at the x0 source frame) and the policy has planted (its lowest patch point within
  ``PLANTED_M``), the slowest patch point's horizontal speed (``build_physics_tables.zone_patch_speed``: a pivot
  reads 0): p50 / p90 in cm/s and the share above 5 cm/s;
* **float**: inside a commanded hold, the commanded ground supports (the release's commanded set) whose lowest patch
  point is more than ``FLOAT_M`` up: the share of zone-frames;
* **COM vs COP**: inside a commanded hold, where the mat's COP is valid at that frame (column 0 or 2 >= 0.9), the
  distance between the policy's COM relative to the centroid of its own commanded-support patches and the human's
  measured COP relative to the reference's: p50 / p90 in cm (``diag_lean_error_m``'s quantity, on every hold).

All three count only *tracked* frames (every body within the evaluator's 0.5 m of the reference, XY aligned at the
first frame as the evaluator does), so a fall does not read as a float. Rollouts come from the evaluator's
``predicted_motion_lib_epoch_<N>.pt`` (every motion from t = 0, deterministic, at the control rate; frame ``i`` is
clip time ``(i + 1) dt``). ``--reference`` scores the reference itself at the same times: the numbers a perfect
imitator gets under these definitions, which is the baseline a policy is compared with (Step 3's own statistics were
taken at 60 Hz on exemplars or every touch frame, so they agree in size, not digit for digit).

Run in a plant-v1 process (no ``REFERENCE_PLANT``): the human evidence is read from the capture stores.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.realisability_v2 \\
        --release <id> --rollouts results/<run>/results/predicted_motion_lib_epoch_<N>.pt --out <json>
    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.realisability_v2 \\
        --release <id> --reference --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from build_physics_tables import patch_points, zone_patch_speed
from extract_contact_configs import ZONE_ORDER, ZONES
from reference_curation import capture_v4, ids, sources

REPO = ids.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODULE = "reference_curation.realisability_v2"
SCHEMA_VERSION = 1
TRACK_FAIL_M = 0.5           # the evaluator's per-body tracking gate (HoldCurriculumConfig.track_fail_m)
PLANTED_M = 0.02
FLOAT_M = 0.02
SLIDE_MPS = 0.05
SLIDE_ZONES = ("L_FOOT", "R_FOOT", "L_HAND", "R_HAND")
VALID = 0.9
CONTROL_DT = 1.0 / 30.0
CONST_KEYS = ("body_mass", "body_com_local", "geom_type", "box_center", "box_half", "box_quat", "cap_a", "cap_b",
              "radius", "sph_center")


def release_paths(release_id: str) -> dict:
    rec = json.loads((ids.DATA_ROOT / "releases" / f"{release_id}.json").read_text())
    out = REPO / rec["dir"]
    return {"record": rec, "dir": out, "ext": out / "holds_extended.yaml", "lineage": out / "lineage.npz",
            "tables": out / "physics_tables.pt", "targets": out / "contact_targets.pt", "graph": out / "contact_graph.pt",
            "motions": out / "motions"}


def load_rollouts(path: Path) -> dict:
    """``{stem: (pos [n, B, 3], rot [n, B, 4] xyzw, dt)}`` from an evaluator's predicted motion library."""
    lib = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for i, f in enumerate(lib["motion_files"]):
        a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
        out[Path(str(f)).stem] = (lib["gts"][a:a + n].float(), lib["grs"][a:a + n].float(), float(lib["motion_dt"][i]))
    return out


def reference_rollouts(motions_dir: Path, stems: list[str], dt: float = CONTROL_DT) -> dict:
    """The reference sampled at the evaluator's times ``(i + 1) dt``: the perfect imitator."""
    out = {}
    for stem in stems:
        m = torch.load(motions_dir / f"{stem}.motion", map_location="cpu", weights_only=False)
        T = m["rigid_body_pos"].shape[0]
        n = int((T - 1) / (dt * int(m["fps"])))
        frames = torch.clamp(torch.round((torch.arange(n) + 1) * dt * int(m["fps"])).long(), max=T - 1)
        out[stem] = (m["rigid_body_pos"][frames].float(), m["rigid_body_rot"][frames].float(), dt)
    return out


def com_xy(pos: torch.Tensor, rot: torch.Tensor, consts: dict) -> torch.Tensor:
    from protomotions.utils.rotations import quat_rotate

    n, B = pos.shape[:2]
    offset = quat_rotate(rot.reshape(-1, 4), consts["body_com_local"].expand(n, B, 3).reshape(-1, 3), w_last=True)
    body = pos + offset.view(n, B, 3)
    m = consts["body_mass"]
    return ((body * m.view(1, B, 1)).sum(1) / m.sum())[:, :2]


def score_motion(stem: str, sim_pos, sim_rot, dt: float, ctx: dict) -> dict:
    """Per-frame quantities of one rollout, gated to tracked frames; pooled later."""
    e = ctx["ext"][stem]
    ref = ctx["ref"][stem]
    fps = int(ref["fps"])
    T = ref["rigid_body_pos"].shape[0]
    n = sim_pos.shape[0]
    frames = torch.clamp(torch.round((torch.arange(n) + 1) * dt * fps).long(), max=T - 1)
    times = frames.double() / fps
    ref_pos, ref_rot = ref["rigid_body_pos"][frames].float(), ref["rigid_body_rot"][frames].float()
    sim = sim_pos.clone()
    sim[..., :2] -= sim[0, 0, :2] - ref_pos[0, 0, :2]
    tracked = ((sim - ref_pos).norm(dim=-1).max(dim=-1).values < TRACK_FAIL_M).numpy()
    source = ctx["lineage"][f"{stem}.index"][frames.numpy()]
    human = ctx["human"][e["source_stem"]][source]                    # [n, Z] 1 contact, 0 separated, -1 unknown
    consts, zb = ctx["consts"], ctx["zone_bodies"]
    zmin, speed = {}, {}
    for z in ZONE_ORDER:
        s, lowest = zone_patch_speed(sim, sim_rot, consts, zb[z], 1.0 / dt)
        zmin[z], speed[z] = lowest.numpy(), s.numpy()
    # every key, so a rollout with no tracked hold frame still pools (as zero frames)
    out = {"slide_mps": [], "float": [], "com_cop_m": [], "tracked": []}
    for z in SLIDE_ZONES:
        zi = ZONE_ORDER.index(z)
        sel = tracked & (human[:, zi] == 1) & (zmin[z] < PLANTED_M)
        out["slide_mps"].append(speed[z][sel])
    # holds: the commanded segment each frame lies in
    g, t = ctx["graph"], ctx["targets"]
    m = ctx["motion_index"][stem]
    seg = np.full(n, -1)
    for k in range(int(g["seg_count"][m])):
        inside = (times.numpy() >= float(g["seg_start"][m, k])) & (times.numpy() <= float(g["seg_end"][m, k]))
        seg[inside] = k
    gr, valid = ref["ground_reaction"][frames].double(), ref["ground_reaction_valid"][frames].double()
    cop_ok = ((valid[:, 0] >= VALID) | (valid[:, 2] >= VALID)).numpy()
    ground_slots = [ctx["pair_names"].index(f"{z}:G") for z in ZONE_ORDER]
    for k in range(int(g["seg_count"][m])):
        rows = np.nonzero((seg == k) & tracked)[0]
        if not len(rows):
            continue
        commanded = [z for z, slot in zip(ZONE_ORDER, ground_slots) if bool(t["seg_commanded"][m, k, slot])]
        for z in commanded:
            out["float"].append(zmin[z][rows] > FLOAT_M)
        bodies = [b for z in commanded for b in zb[z]]
        rows_cop = rows[cop_ok[rows]]
        if not bodies or not len(rows_cop):
            continue
        r = torch.as_tensor(rows_cop)
        sim_c = patch_points(sim[r], sim_rot[r], consts, bodies)[0][..., :2].mean(1)
        ref_c = patch_points(ref_pos[r], ref_rot[r], consts, bodies)[0][..., :2].mean(1)
        policy = com_xy(sim[r], sim_rot[r], consts) - sim_c
        human_cop = gr[r, 1:3].float() - ref_c
        out["com_cop_m"].append((policy - human_cop).norm(dim=-1).numpy())
    out["tracked"] = [tracked]
    return {k: np.concatenate(v) if v else np.zeros(0, dtype=bool) for k, v in out.items()}


def summarise(per: dict) -> dict:
    def pooled(key):
        return np.concatenate([p[key] for p in per.values()]) if per else np.zeros(0)

    s, f, c, tr = pooled("slide_mps"), pooled("float"), pooled("com_cop_m"), pooled("tracked")

    def pct(x, q, scale=100.0):
        return round(scale * float(np.percentile(x, q)), 2) if len(x) else None

    # TODO A1's statistic: each clip's median slide, then the 90th percentile over clips (clips with >= 10 frames)
    medians = [float(np.median(p["slide_mps"])) for p in per.values() if len(p["slide_mps"]) >= 10]
    # the evaluator's eval/perf/track averages per motion; tracked_share pools frames (motions differ in length)
    per_motion = [float(p["tracked"].mean()) for p in per.values() if len(p["tracked"])]
    return {"frames": int(len(tr)), "tracked_share": round(float(tr.mean()), 4) if len(tr) else None,
            "tracked_motion_mean": round(float(np.mean(per_motion)), 4) if per_motion else None,
            "slide_zone_frames": int(len(s)), "slide_cm_s_p50": pct(s, 50), "slide_cm_s_p90": pct(s, 90),
            "slide_clip_median_cm_s_p90": pct(np.array(medians), 90) if medians else None,
            "slide_share_over_5cm_s": round(float((s > SLIDE_MPS).mean()), 4) if len(s) else None,
            "float_zone_frames": int(len(f)), "float_share_over_2cm": round(float(f.mean()), 4) if len(f) else None,
            "com_cop_frames": int(len(c)), "com_cop_cm_p50": pct(c, 50), "com_cop_cm_p90": pct(c, 90)}


def evaluate(release_id: str, rollouts: dict) -> dict:
    """Score ``{stem: (pos, rot, dt)}`` against the release; per group and pooled."""
    capture_v4.require_v1_process()
    p = release_paths(release_id)
    ext = {c["stem"]: c for c in yaml.safe_load(open(p["ext"]))["clips"]}
    tables = torch.load(p["tables"], map_location="cpu", weights_only=False)
    graph = torch.load(p["graph"], map_location="cpu", weights_only=False)
    names = list(graph["motion_names"])
    missing = [s for s in rollouts if s not in ext]
    if missing:
        raise ValueError(f"rollouts of motions the release does not have: {missing[:3]}")
    body_names = list(tables["body_names"])
    ctx = {"ext": ext, "graph": graph, "pair_names": list(graph["pair_names"]),
           "targets": torch.load(p["targets"], map_location="cpu", weights_only=False),
           "lineage": np.load(p["lineage"]), "consts": {k: tables[k] for k in CONST_KEYS},
           "zone_bodies": {z: [body_names.index(b) for b in ZONES[z]] for z in ZONE_ORDER},
           "motion_index": {s: i for i, s in enumerate(names)},
           "ref": {s: torch.load(p["motions"] / f"{s}.motion", map_location="cpu", weights_only=False) for s in rollouts},
           "human": {}}
    for src in sorted({ext[s]["source_stem"] for s in rollouts}):
        ctx["human"][src], _ = sources.ground_source(capture_v4.load(src, rebuild=False))
    per = {s: score_motion(s, pos, rot, dt, ctx) for s, (pos, rot, dt) in rollouts.items()}
    groups = defaultdict(dict)
    for s, v in per.items():
        groups[ext[s].get("group", "all")][s] = v
    return {"pooled": summarise(per), "groups": {g: summarise(v) for g, v in sorted(groups.items())},
            "per_motion": {s: summarise({s: v}) for s, v in per.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rollouts", help="an evaluator's predicted_motion_lib_epoch_<N>.pt")
    src.add_argument("--reference", action="store_true", help="score the reference itself (the control)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    start = time.time()
    try:
        p = release_paths(args.release)
        if args.reference:
            stems = [c["stem"] for c in yaml.safe_load(open(p["ext"]))["clips"]]
            rollouts, what = reference_rollouts(p["motions"], stems), "reference"
        else:
            rollouts, what = load_rollouts(Path(args.rollouts)), ids.display_path(args.rollouts)
        result = evaluate(args.release, rollouts)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    inputs = [p["ext"], p["lineage"], p["tables"], p["targets"], p["graph"]]
    if not args.reference:
        inputs.append(Path(args.rollouts))
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), "release_id": args.release,
              "rollouts": what, "thresholds": {"track_fail_m": TRACK_FAIL_M, "planted_m": PLANTED_M, "float_m": FLOAT_M,
                                               "slide_mps": SLIDE_MPS, "cop_valid": VALID},
              "seconds": round(time.time() - start, 1), **result}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    q = result["pooled"]
    print(f"realisability of {what} on {args.release}: tracked {q['tracked_share']} of frames "
          f"({q['tracked_motion_mean']} per motion, the evaluator's eval/perf/track), slide p50/p90 "
          f"{q['slide_cm_s_p50']}/{q['slide_cm_s_p90']} cm/s ({q['slide_share_over_5cm_s']} > 5 cm/s), float "
          f"{q['float_share_over_2cm']} of commanded supports, COM-COP p50/p90 {q['com_cop_cm_p50']}/{q['com_cop_cm_p90']} cm "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
