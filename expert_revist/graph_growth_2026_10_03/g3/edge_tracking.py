"""Per-variant execution of the synthesised edges, from an evaluator rollout library (PLAN.MD card R4b).

For every ``SYN_`` motion of release v3 (x0, x3s, x7s) in a ``predicted_motion_lib_epoch_<N>.pt``:

* the **transition window** is ``[departure, arrival)``: the S hold's ``t_end`` (the departure, the generator's frame
  ``k_d``) to the D hold's ``t_start`` (the arrival, ``k_a``), read from the release manifest, so the x3s / x7s shifts
  come for free. ``trans_tracked`` is the share of its frames on which every body is within 0.5 m of the reference
  (``analyze_rollouts_v2``'s rule, the training termination's threshold). **G2's gate is ``trans_tracked >= 0.6``.**
* **arrival at D**: the D hold's window ``[arrival, t_end]`` covers the 2 s synthetic hold at D and D's own real
  window in the lead-out. ``d_err6_p50`` / ``d_err6_ok`` are the 6 goal bodies' best-yaw error to D's exemplar over
  ``[t_hold, t_end]`` (``analyze_rollouts_v2``), ``d_support_geom`` the worst commanded zone's share of window frames
  with its lowest collider point within 2 cm, ``d_support_load`` the same share by load (terrain-filtered vertical
  force >= 3 % BW, E1's runtime threshold), and ``d_subst_load`` every un-commanded zone that carries >= 3 % BW on
  >= 20 % of the window (E1's force rule, here over every free zone rather than the sidecar's known-free ones).
* ``reached_D``: the transition is tracked (>= 0.6), D's pose is held (``d_err6_ok >= 0.5``) and D's commanded
  supports are down by geometry on >= 90 % of the window (E1's "realised" rule).
* naturalness of the executed transition: per-body peak speed and peak jerk over ``[departure, arrival + 0.5 s]``
  (30 Hz differences), as a multiple of the per-body p99 of the human x0 *reference* clips and of the policy's own
  rollouts of those clips (same library, same 30 Hz rate).

The S hold before the departure is scored the same way, so a variant that fails can be told apart from one that never
reached S.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/edge_tracking.py \\
        --library e1=results/smpl_yogi_v2_expert56_g3_2f132f4299/results/predicted_motion_lib_epoch_1.pt \\
        --library e3420=output/renderings/expert56_v2_g3_e3420/eval_standalone_e3420/results/predicted_motion_lib_epoch_0.pt \\
        --out expert_revist/graph_growth_2026_10_03/g3/data/edge_tracking.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "data" / "scripts", REPO / "expert_revist" / "expert56_v2_e15500",
          REPO / "expert_revist" / "run1_gap_analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import analyze_rollouts_v2 as ar  # noqa: E402

RELEASE = "holds_repaired_ftC_posefix.release_v3.2f132f4299"
SYN_DIR = REPO / "data/smpl/reference_curation/synthetic_v3/synthetic_v3.d034199f23"
EDGE_LABEL = {"E1": "crow -> handstand (press)", "E3": "handstand -> crow (lower)", "B1": "crow -> plank (jump)",
              "E2": "crow -> chaturanga (jump-back)", "E5": "handstand -> chaturanga (float-down)"}
EDGE_ORDER = ("E1", "E3", "B1", "E2", "E5")
G2_GATE = 0.6
LOAD_FRAC = 0.03          # E1: SupportV2Params.load_threshold (3 % BW)
SUBST_SHARE = 0.2         # E1: a free zone loaded on >= 20 % of the window
REALISED = 0.9            # E1: commanded zone down on >= 90 % of the window
POSE_OK_SHARE = 0.5       # D's pose held: 6-body error < 0.15 m on at least half of [t_hold, t_end]
LANDING_S = 0.5           # naturalness window runs past the arrival by this much (landings)
TRACE_PRE_S, TRACE_POST_S = 1.5, 3.0


def base_stem(stem: str) -> tuple[str, int]:
    m = re.match(r"^(.*)_x(\d+)s$", stem)
    return (m.group(1), int(m.group(2))) if m else (stem, 0)


def parse_variant(base: str) -> dict:
    m = re.match(r"^SYN_(E\d|B\d)_([a-z]+)_(high|mid)_s(\d+)_(t6\w+)$", base)
    assert m, base
    return {"edge": m.group(1), "kind": m.group(2), "timing": m.group(3), "seed": int(m.group(4)), "tag": m.group(5)}


def speed_jerk(pos: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """[T-1, B] speed and [T-3, B] jerk magnitude from body positions [T, B, 3]."""
    v = np.linalg.norm(np.diff(pos, axis=0), axis=-1) / dt
    j = np.linalg.norm(np.diff(pos, 3, axis=0), axis=-1) / dt ** 3
    return v, j


def corpus_p99(lib: dict, stems: list[str], ctx: dict) -> dict:
    """Per-body p99 speed and jerk at 30 Hz: the human x0 references, and the policy's rollouts of them."""
    vr, jr, vp, jp = [], [], [], []
    for i, stem in enumerate(stems):
        if stem.startswith("SYN_") or ctx["ext"][stem]["variant_s"] != 0.0:
            continue
        a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
        dt = float(lib["motion_dt"][i])
        sim = lib["gts"][a:a + n].double().numpy()
        m = ctx["ref"](stem)
        fps = int(m["fps"])
        step = int(round(fps * dt))
        ref = m["rigid_body_pos"].double().numpy()[::step]
        for out_v, out_j, p in ((vr, jr, ref), (vp, jp, sim)):
            v, j = speed_jerk(p, dt)
            out_v.append(v)
            out_j.append(j)
    q = lambda xs: np.percentile(np.concatenate(xs, 0), 99, axis=0)  # noqa: E731
    return {"ref_speed": q(vr), "ref_jerk": q(jr), "policy_speed": q(vp), "policy_jerk": q(jp)}


def score_library(path: Path, ctx: dict, bw_n: float) -> dict:
    lib = torch.load(path, map_location="cpu", weights_only=False)
    stems = [Path(str(f)).stem for f in lib["motion_files"]]
    forces = lib.get("sim_rigid_body_ground_forces")
    p99 = corpus_p99(lib, stems, ctx)
    names = list(ar.COMMON_BODY_ORDER)
    rows = []
    for i, stem in enumerate(stems):
        if not stem.startswith("SYN_"):
            continue
        base, xs = base_stem(stem)
        clip = ctx["ext"][stem]
        holds = clip["holds"]
        a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
        dt = float(lib["motion_dt"][i])
        sc = ar.score_rollout(stem, lib["gts"][a:a + n], lib["grs"][a:a + n], dt, 1, ctx)
        tr = sc["_trace"]
        t, max_err, mean_err = tr["t"], tr["max_err"], tr["mean_err"]
        tracked = max_err < ar.TRACK_FAIL_M
        s_k = next(k for k, h in enumerate(holds) if h.get("extend"))
        info = parse_variant(base)
        spec = json.loads((SYN_DIR / f"{base}.json").read_text())
        d_id = spec["D"]["hold_id"]
        d_k = next(k for k, h in enumerate(holds) if h.get("inherits") == d_id)
        S, D = holds[s_k], holds[d_k]
        t_dep, t_arr, t_lo, t_dend = float(S["t_end"]), float(D["t_start"]), float(D["t_hold"]), float(D["t_end"])
        # the generator's clock, cross-checked against the manifest (60 fps; x-variants shift by the inserted frames)
        shift = int(clip.get("inserted_frames", 0))
        assert abs(t_dep - (spec["layout"]["departure"] + shift) / 60) < 1e-3, (stem, t_dep)
        assert abs(t_arr - (spec["layout"]["arrival"] + shift) / 60) < 1e-3, (stem, t_arr)
        trans = (t >= t_dep) & (t < t_arr)
        lead_in = t < t_dep
        lead_out = t > t_dend
        first_loss = None
        lost = np.nonzero(trans & ~tracked)[0]
        if len(lost):
            first_loss = round(float(t[lost[0]] - t_dep), 3)
        sr, dr = sc["holds"][s_k], sc["holds"][d_k]

        row = {"stem": stem, "base": base, "x": xs, **info, "T": spec["T"],
               "t_departure": round(t_dep, 3), "t_arrival": round(t_arr, 3), "t_lead_out": round(t_lo, 3),
               "clip_s": clip["length_s"], "recorded_s": sc["recorded_s"],
               "tracked_clip": sc["tracked_share"], "first_termination_s": sc["first_termination_s"],
               "lead_in_tracked": round(float(tracked[lead_in].mean()), 3) if lead_in.any() else None,
               "trans_tracked": round(float(tracked[trans].mean()), 3) if trans.any() else None,
               "trans_first_loss_s": first_loss,
               "trans_mean_err_p50": round(float(np.median(mean_err[trans])), 3) if trans.any() else None,
               "trans_max_err_peak": round(float(max_err[trans].max()), 3) if trans.any() else None,
               "lead_out_tracked": round(float(tracked[lead_out].mean()), 3) if lead_out.any() else None,
               "s_err6_p50": sr.get("err6_p50"), "s_support_geom": sr.get("support_min"),
               "s_subst_geom": sr.get("substitutions"),
               "d_tracked": dr.get("tracked_share"), "d_err6_p50": dr.get("err6_p50"), "d_err6_ok": dr.get("err6_share_ok"),
               "d_support_geom": dr.get("support_min"), "d_support_geom_zones": dr.get("support"),
               "d_subst_geom": dr.get("substitutions"), "d_pelvis_z": dr.get("pelvis_z"),
               "d_pelvis_z_ref": dr.get("pelvis_z_ref"), "d_ground": D["pairs_ground"]}

        if forces is not None:
            f = forces[a:a + n].float()
            load = ar_zone_load(f, ctx).numpy()                     # [T, Z] N
            loaded = load >= LOAD_FRAC * bw_n
            win = (t >= t_arr) & (t <= t_dend)
            commanded = [p.split(":")[0] for p in D["pairs_ground"]]
            ci = [ar.ZONE_ORDER.index(z) for z in commanded]
            free = [z for z in range(len(ar.ZONE_ORDER)) if z not in ci]
            if win.any():
                share = {ar.ZONE_ORDER[z]: round(float(loaded[win, z].mean()), 3) for z in ci}
                row["d_support_load"] = min(share.values()) if share else None
                row["d_support_load_zones"] = share
                row["d_subst_load"] = {ar.ZONE_ORDER[z]: round(float(loaded[win, z].mean()), 3) for z in free
                                       if loaded[win, z].mean() >= SUBST_SHARE}
                row["d_load_bw"] = {ar.ZONE_ORDER[z]: round(float(np.median(load[win, z])) / bw_n, 3)
                                    for z in range(len(ar.ZONE_ORDER)) if np.median(load[win, z]) >= LOAD_FRAC * bw_n}
            # contacts during the transition the reference does not make (e.g. a knee down in a jump-back)
            row["trans_free_load_share"] = {ar.ZONE_ORDER[z]: round(float(loaded[trans, z].mean()), 3)
                                            for z in range(len(ar.ZONE_ORDER))
                                            if loaded[trans, z].mean() >= 0.05}

        # naturalness of the executed transition (and the reference's, at the same rate)
        nat_win = (t >= t_dep) & (t <= t_arr + LANDING_S)
        if nat_win.sum() >= 4:
            sim = lib["gts"][a:a + n].double().numpy()[nat_win]
            m = ctx["ref"](stem)
            step = int(round(int(m["fps"]) * dt))
            idx = np.minimum(np.round(t[nat_win] * int(m["fps"])).astype(int), m["rigid_body_pos"].shape[0] - 1)
            ref = m["rigid_body_pos"].double().numpy()[idx]
            assert step == 2
            for side, p in (("policy", sim), ("ref", ref)):
                v, j = speed_jerk(p, dt)
                vmax, jmax = v.max(0), j.max(0)
                row[f"nat_{side}"] = {
                    "speed_over_ref_p99_max": round(float((vmax / p99["ref_speed"]).max()), 2),
                    "speed_over_ref_p99_body": names[int((vmax / p99["ref_speed"]).argmax())],
                    "jerk_over_ref_p99_max": round(float((jmax / p99["ref_jerk"]).max()), 2),
                    "jerk_over_ref_p99_body": names[int((jmax / p99["ref_jerk"]).argmax())],
                    "speed_over_policy_p99_max": round(float((vmax / p99["policy_speed"]).max()), 2),
                    "jerk_over_policy_p99_max": round(float((jmax / p99["policy_jerk"]).max()), 2),
                    "bodies_speed_over_ref_p99": int((vmax > p99["ref_speed"]).sum()),
                    "bodies_jerk_over_ref_p99": int((jmax > p99["ref_jerk"]).sum()),
                }

        rel = (t - t_dep)
        keep = (rel >= -TRACE_PRE_S) & (rel <= (t_arr - t_dep) + TRACE_POST_S)
        row["_trace"] = {"t_rel": np.round(rel[keep], 4).tolist(), "max_err": np.round(max_err[keep], 4).tolist(),
                         "mean_err": np.round(mean_err[keep], 4).tolist()}
        row["reached_D"] = bool((row["trans_tracked"] or 0) >= G2_GATE and (row["d_err6_ok"] or 0) >= POSE_OK_SHARE
                                and (row["d_support_geom"] or 0) >= REALISED)
        row["g2_pass"] = bool((row["trans_tracked"] or 0) >= G2_GATE)
        rows.append(row)
    return {"library": str(path), "rows": rows,
            "p99": {k: np.round(v, 3).tolist() for k, v in p99.items()}, "body_names": names}


def ar_zone_load(f: torch.Tensor, ctx: dict) -> torch.Tensor:
    """[T, Z] terrain-filtered vertical load per zone (N): hold_curriculum.zone_vertical_load on the release's zones."""
    from protomotions.agents.evaluators.hold_curriculum import zone_vertical_load
    return zone_vertical_load(f, [ctx["zone_bodies"][z] for z in ar.ZONE_ORDER])


def summarise(rows: list[dict]) -> dict:
    out = {}
    for e in EDGE_ORDER:
        for scope in ("x0", "all"):
            sel = [r for r in rows if r["edge"] == e and (scope == "all" or r["x"] == 0)]
            if not sel:
                continue
            tt = [r["trans_tracked"] for r in sel if r["trans_tracked"] is not None]
            out[f"{e}_{scope}"] = {
                "edge": e, "label": EDGE_LABEL[e], "scope": scope, "n": len(sel),
                "g2_pass": sum(r["g2_pass"] for r in sel), "reached_D": sum(r["reached_D"] for r in sel),
                "trans_tracked_p50": round(float(np.median(tt)), 3), "trans_tracked_min": round(float(min(tt)), 3),
                "trans_tracked_max": round(float(max(tt)), 3),
                "d_err6_p50_median": round(float(np.median([r["d_err6_p50"] for r in sel if r["d_err6_p50"] is not None])), 3),
                "d_support_geom_median": round(float(np.median([r["d_support_geom"] for r in sel
                                                                 if r["d_support_geom"] is not None])), 3),
                "lead_in_tracked_p50": round(float(np.median([r["lead_in_tracked"] for r in sel])), 3),
                "lead_out_tracked_p50": round(float(np.median([r["lead_out_tracked"] for r in sel
                                                                if r["lead_out_tracked"] is not None])), 3),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", action="append", required=True, help="LABEL=path (repeatable)")
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctx = ar.load_context(args.release)
    bw_n = float(ctx["consts"]["body_mass"].sum()) * 9.81
    result = {"release": args.release, "body_weight_n": round(bw_n, 2), "g2_gate": G2_GATE, "libraries": {}}
    for spec in args.library:
        label, path = spec.split("=", 1)
        scored = score_library(Path(path), ctx, bw_n)
        scored["summary"] = summarise(scored["rows"])
        result["libraries"][label] = scored
        print(f"\n=== {label}: {path}")
        print(f"{'edge':4s} {'scope':5s} {'n':>3s} {'G2 pass':>8s} {'reached D':>9s} {'trans p50':>9s} "
              f"{'min':>6s} {'max':>6s} {'D err6':>7s} {'D supp':>7s} {'lead-in':>7s} {'lead-out':>8s}")
        for v in scored["summary"].values():
            print(f"{v['edge']:4s} {v['scope']:5s} {v['n']:3d} {v['g2_pass']:4d}/{v['n']:<3d} {v['reached_D']:5d}/{v['n']:<3d} "
                  f"{v['trans_tracked_p50']:9.3f} {v['trans_tracked_min']:6.3f} {v['trans_tracked_max']:6.3f} "
                  f"{v['d_err6_p50_median']:7.3f} {v['d_support_geom_median']:7.3f} {v['lead_in_tracked_p50']:7.3f} "
                  f"{v['lead_out_tracked_p50']:8.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result) + "\n")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
