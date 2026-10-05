"""S1's fork-battery collation: the commanded family hold (or edge destination) per plan, across battery runs.

Each run is ``name=<goal_causality_x0.py --parts forks output dir>@<plan dir>[,<plan dir>...]``. A plan's scored
goal is its *target*: route plans (``make_route_probe_plans.py``) name it (node key and hold window), the release's
family forks command it first (S0's rule: the family hold, not the closing standing), ``edge_*`` / ``nohijack_*``
second and ``fork_edge_*`` last. Scored goals are matched by node key and window end, because the battery drops a
goal whose hold window is empty, which shifts positions.

Two scores per target:

* ``success`` -- the battery's own, over the goal's hold window ``[reach end, end]``. For a route plan that is the
  segment's ``[t_hold, t_end]``: under training's semantics the hold frame can sit at the segment's end (Peacock's
  does: its window is empty and the goal is not scored), and is often short (Warrior III -b: 1.2 s of a 10.8 s
  segment);
* ``segment_supports`` (route plans only) -- the battery's support rule over the target segment's whole span
  ``[t_start, t_end]`` on the clip's clock, from its saved per-frame traces (``forks/zone_loads.npz``): every
  commanded zone down (>= 5 N) on >= 90 % of the frames and no other zone loaded (>= 3 % of 74 kg) on >= 20 % of
  them. A segment is a contact configuration, constant over the span by definition, while the reference's *pose*
  moves inside it (Scorpion -b's 18.2 s hands-only segment passes through a handstand before the scorpion
  exemplar at its end), so the pose test stays on the hold window. ``segment_success`` adds the pose test over the
  whole span too; it is kept in the JSON and is meaningful only where the reference holds one pose.

    python expert_revist/graph_growth_2026_10_03/s1_port/forks_compare.py --graph <release>/contact_graph.json \\
        S0=output/goal_causality_x0/g3_e3420_forks@<release plans> route=output/s1_port/forks_route_training@<route plans>,<release plans>

writes ``data/forks_compare.json`` and prints the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
KEYS = ("success", "hold_rate24", "supports_realised", "no_substitution", "err24_p50", "held_s_p50", "pelvis_z_p50",
        "reach_rate24")
TOUCH_N = 5.0
LOAD_N = 0.03 * 74.0 * 9.81
POSE_M, REALISED, UNWANTED = 0.15, 0.90, 0.20


def target_of(plan: dict, name: str):
    """``(node key, window end or None)`` of a plan's scored goal."""
    if "target" in plan:
        return plan["target"]["config"], float(plan["target"]["window_s"][1])
    if name.startswith(("edge_", "nohijack_")):
        return plan["goals"][1]["config"], None              # the commanded continuation from a held S
    if name.startswith("fork_edge_"):
        return plan["goals"][-1]["config"], None             # D, after S from standing
    return next(g["config"] for g in plan["goals"] if g["name"] != "standing"), None


def pick(goals: list, key: str, end):
    hits = [g for g in goals if g["node"] == key]
    if end is not None:
        hits = [g for g in hits if abs(float(g["window_s"][1]) - end) < 0.06]
    return hits[0] if hits else None


def segment_of(graph: dict, plan: dict):
    """``(t_start, t_end, commanded ground zones)`` of a route plan's target segment, on the clip's clock."""
    stem = plan["start"]["clip"]
    goal = plan["goals"][plan["target"]["goal"]]
    for seg in graph["clips"][stem]["segments"]:
        if abs(float(seg["t_hold"]) - float(goal["pose_time"])) < 1e-4 and seg["config"] == goal["config"]:
            return float(seg["t_start"]), float(seg["t_end"]), [p[:-2] for p in seg["pairs"] if p.endswith(":G")]
    raise KeyError(f"no segment of {stem} holds {goal['config']} at {goal['pose_time']}")


def segment_success(run_dir: Path, name: str, span) -> dict:
    """The battery's success rule over ``span``, every replica of ``name``, from the saved per-frame traces."""
    z = np.load(run_dir / "forks/zone_loads.npz")
    p = np.load(run_dir / "forks/pose_error_traces.npz")
    seqs = list(z["sequences"])
    if name not in seqs:
        return None
    S, s = int(z["num_sequences"]), seqs.index(name)
    t = z["frame_times"]
    frames = (t >= span[0]) & (t < span[1])
    if not frames.any():
        return None
    zones = list(z["zone_names"])
    want = np.array([zn in span[2] for zn in zones])
    load = z["zone_load_n"][frames][:, s::S].astype(np.float32)            # [F, R, Z]
    err = p["pose_errors"][frames][:, s::S]                                  # [F, R]
    held = np.nanmean(err, axis=0) < POSE_M
    touch = (load >= TOUCH_N).mean(axis=0)                                   # [R, Z]
    loaded = (load >= LOAD_N).mean(axis=0)
    realised = (touch[:, want] >= REALISED).all(axis=1) if want.any() else np.ones(len(held), bool)
    clean = ~(loaded[:, ~want] >= UNWANTED).any(axis=1)
    return {"segment_supports": float((realised & clean).mean()), "segment_success": float((held & realised & clean).mean()),
            "segment_hold": float(held.mean()), "segment_s": [round(span[0], 3), round(span[1], 3)],
            "replicas": int(len(held))}


def load(spec: str, graph: dict) -> dict:
    out_dir, plan_dirs = spec.split("@")
    run_dir = REPO / out_dir
    scored = json.loads((run_dir / "forks_scored.json").read_text())
    names = set(scored)
    if (run_dir / "forks/zone_loads.npz").exists():
        names |= set(np.load(run_dir / "forks/zone_loads.npz")["sequences"].tolist())
    rows = {}
    for name in sorted(names):
        path = next((REPO / d / f"{name}.json" for d in plan_dirs.split(",") if (REPO / d / f"{name}.json").exists()),
                    None)
        if path is None:
            continue
        plan = json.loads(path.read_text())
        key, end = target_of(plan, name)
        goal = pick(scored.get(name, []), key, end)
        row = {k: goal.get(k) for k in KEYS} if goal is not None else {k: None for k in KEYS}
        row["node"] = key
        if "target" in plan:
            row.update(segment_success(run_dir, name, segment_of(graph, plan)) or {})
        rows[name] = row
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--graph", default="data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v3.2f132f4299/"
                                       "contact_graph.json")
    ap.add_argument("runs", nargs="+", help="name=<battery output dir>@<plan dir>[,<plan dir>...]")
    a = ap.parse_args()
    graph = json.loads((REPO / a.graph).read_text())
    runs = dict(arg.split("=", 1) for arg in a.runs)
    data = {name: load(spec, graph) for name, spec in runs.items()}
    names = sorted({p for rows in data.values() for p in rows})
    out = {"runs": runs, "plans": {p: {r: data[r].get(p) for r in data} for p in names}}
    fam = [p for p in names if p.startswith("fork_") and not p.startswith("fork_edge_")]
    for r in data:
        got = [data[r][p]["success"] for p in fam if p in data[r] and data[r][p]["success"] is not None]
        seg = [data[r][p].get("segment_supports") for p in fam if p in data[r] and data[r][p].get("segment_supports")
               is not None]
        out.setdefault("family_success", {})[r] = {
            "scored": len(got), "succeeded": sum(v >= 0.5 for v in got), "mean": sum(got) / len(got) if got else None,
            "segment_scored": len(seg), "segment_succeeded": sum(v >= 0.5 for v in seg)}
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/forks_compare.json").write_text(json.dumps(out, indent=1))
    width = max(len(p) for p in names)
    print(f"{'plan':{width}s} " + " ".join(f"{r[:22]:>22s}" for r in data))
    for p in names:
        cells = []
        for r in data:
            row = data[r].get(p)
            if row is None:
                cells.append(f"{'-':>22s}")
                continue
            s = "  -  " if row["success"] is None else f"{row['success']:5.2f}"
            seg = f" sup {row['segment_supports']:4.2f}" if row.get("segment_supports") is not None else ""
            z = "" if row["pelvis_z_p50"] is None else f" z{row['pelvis_z_p50']:4.2f}"
            cells.append(f"{s}{z}{seg}")
        print(f"{p:{width}s} " + " ".join(f"{c:>22s}" for c in cells))
    for r, v in out["family_success"].items():
        print(f"{r}: family forks succeeding (>= 0.5 of replicas) {v['succeeded']} of {v['scored']} scored; "
              f"supports held over the whole target segment {v['segment_succeeded']} of {v['segment_scored']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
