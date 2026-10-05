"""G3 against its pre-registered gates (PLAN.MD card G3), on human clips only, at every evaluation.

The run's own ``eval/*`` totals pool all 252 motions, so the 84 synthetic ones move every corpus-wide number
(``eval/success_rate``, ``eval/perf/score``, ``eval/drag/all_J``, ``eval/normalized_jerk_mean``,
``eval/perf/substitution_holds_v2_x0``). This script restates each gate over the 168 human motions:

* groups: ``eval/perf_group/<group>_score`` (the four human groups never contain a ``SYN_`` clip);
* drag: the mean of the CSV's ``drag_J`` over human motions (``eval/drag/all_J``'s pooling, restricted);
* substitutions: the sum of ``substitution_holds_v2`` over human x0 motions (``..._v2_x0``'s count, restricted);
* success: 1 - human motions in ``failed_motions_epoch_<N>`` / 168 (``eval/success_rate`` = 1 - failed / all);
* jerk: recomputed per motion from the evaluator's rollout libraries with ``SmoothnessCalculator``'s own kernel
  (0.4 s windows, high-jerk threshold 6,500), validated against the logged all-motion mean; only epochs with a
  saved library (1, 1,500, 3,000; the standalone 3,420; G1's standalone 5,000).

Inputs: G3's TensorBoard dump (``dump_tb_scalars.py``), its CSVs and failed-motion lists, the standalone evaluation
of epoch 3,420, and G1's standalone evaluation of epoch 5,000 (the warm start) for reference.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/g3_gates.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from protomotions.agents.evaluators.smoothness_calculator import SmoothnessCalculator  # noqa: E402

RUN = REPO / "results/smpl_yogi_v2_expert56_g3_2f132f4299"
OUT = REPO / "output/renderings/expert56_v2_g3_e3420"
STANDALONE = OUT / "eval_standalone_e3420"
G1_RUN = REPO / "results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac"
G1_STANDALONE = REPO / "output/renderings/expert56_v2_amp_g1_e5000/eval_standalone_e5000"
G1_GATES = HERE.parent / "g1_amp_eval/data/gates.json"
LAST3 = (2000, 2500, 3000)
GROUPS = ("single_leg", "inversion", "arm_balance", "connective")
# PLAN.MD, card G3: floors = G1's last-three mean - 0.035; drag <= G1's 51.9 J; jerk < 72.2; high-jerk < 1.30 %;
# v2 substitutions on human x0 holds <= 6.
FLOORS = {"single_leg": 0.936, "inversion": 0.897, "arm_balance": 0.951, "connective": 0.929}
DRAG_MAX, JERK_MAX, HIGHJERK_MAX, SUBST_MAX = 51.9, 72.2, 1.30, 6
TRAINING = ("info/episode_length", "env/terminate_mean", "env/raw_r/gt_rew_mean", "env/raw_r/gr_rew_mean",
            "env/raw_r/unwanted_support_rew_mean", "env/raw_r/diag_known_free_load_n_mean",
            "env/raw_r/diag_swing_load_n_mean", "env/raw_r/diag_lean_error_m_mean", "actor/clip_frac",
            "actor/approx_kl", "critic/explained_variance", "amp/reward_w", "amp/reward_mean_human",
            "amp/reward_mean_syn", "amp/syn_sample_share", "advantages/style_to_task_std",
            "discriminator/agent_acc", "discriminator/pos_acc", "eval/curriculum/ess", "times/last_epoch_seconds")
BIN = 500


def base(stem: str) -> str:
    return re.sub(r"_x\d+s$", "", stem)


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def failed_ids(path: Path) -> set[int]:
    return {int(x) for x in path.read_text().split()} if path.exists() else set()


def csv_metrics(rows: list[dict], failed: set[int]) -> dict:
    human = [r for r in rows if not r["motion"].startswith("SYN_")]
    syn = [r for r in rows if r["motion"].startswith("SYN_")]
    hx0 = [r for r in human if base(r["motion"]) == r["motion"]]
    f = lambda rs, k: [float(r[k]) for r in rs if r[k] not in ("", "nan")]  # noqa: E731
    out = {
        "n_human": len(human), "n_syn": len(syn),
        "human_score_mean": float(np.mean(f(human, "score"))),
        "human_drag_J": float(np.nanmean(f(human, "drag_J"))) if f(human, "drag_J") else None,
        "human_subst_v2_x0": int(sum(float(r["substitution_holds_v2"] or 0) for r in hx0
                                     if r.get("substitution_holds_v2") not in (None, "", "nan"))),
        "human_failed": sorted(int(r["motion_id"]) for r in human if int(r["motion_id"]) in failed),
        "human_success": 1.0 - sum(int(r["motion_id"]) in failed for r in human) / max(len(human), 1),
        "syn_score_mean": float(np.mean(f(syn, "score"))) if syn else None,
        "syn_p_track_mean": float(np.mean(f(syn, "p_track"))) if syn else None,
        "syn_failed": sorted(int(r["motion_id"]) for r in syn if int(r["motion_id"]) in failed),
    }
    for g in GROUPS:
        out[f"{g}_score_csv"] = float(np.mean(f([r for r in human if r["group"] == g], "score")))
    edge = {}
    for r in syn:
        e = r["motion"].split("_")[1]
        edge.setdefault(e, []).append(float(r["p_track"]))
    out["syn_p_track_by_edge"] = {e: round(float(np.mean(v)), 4) for e, v in sorted(edge.items())}
    return out


def library_jerk(path: Path) -> dict:
    lib = torch.load(path, map_location="cpu", weights_only=False)
    calc = SmoothnessCalculator(device="cpu", dt=float(lib["motion_dt"][0]))
    window = max(4, int(round(0.4 / calc.dt)))
    per = []
    for i, f in enumerate(lib["motion_files"]):
        stem = Path(str(f)).stem
        a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
        if n < window:
            continue
        nj = calc._compute_windowed_normalized_jerk(lib["gts"][a:a + n].float(), window)
        if nj.numel() == 0:
            continue
        per.append((stem, float(nj.mean()), float((nj > calc.high_jerk_threshold).any(dim=1).float().mean() * 100)))
    sel = lambda pred: [p for p in per if pred(p[0])]  # noqa: E731
    out = {}
    for name, rows in (("all", per), ("human", sel(lambda s: not s.startswith("SYN_"))),
                       ("syn", sel(lambda s: s.startswith("SYN_")))):
        if rows:
            out[name] = {"n": len(rows), "jerk": float(np.mean([r[1] for r in rows])),
                         "high_jerk_pct": float(np.mean([r[2] for r in rows]))}
    out["per_motion"] = {s: round(j, 2) for s, j, _ in per}
    return out


def main() -> int:
    tb = json.load(open(OUT / "tb_scalars.json"))
    evals = {t: v for t, v in tb.items() if t.startswith("eval/")}
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/tb_eval_scalars.json").write_text(json.dumps(evals) + "\n")
    binned = {}
    for t in TRAINING:
        bins: dict[int, list[float]] = {}
        for step, value in tb.get(t, []):
            bins.setdefault(int(step) // BIN * BIN, []).append(value)
        binned[t] = {str(k): st.mean(v) for k, v in sorted(bins.items())}
    for t in [k for k in tb if k.startswith("env/anchor/")]:
        bins = {}
        for step, value in tb.get(t, []):
            bins.setdefault(int(step) // BIN * BIN, []).append(value)
        binned[t] = {str(k): st.mean(v) for k, v in sorted(bins.items())}
    (HERE / "data/tb_training_binned.json").write_text(json.dumps({"bin_epochs": BIN, "tags": binned}, indent=1) + "\n")

    epochs = sorted(int(a) for a, _ in evals["eval/success_rate"])
    series = {}
    for e in epochs:
        rows = read_csv(RUN / f"curriculum/eval_epoch_{e:06d}.csv")
        m = csv_metrics(rows, failed_ids(RUN / f"failed_motions/failed_motions_epoch_{e}_rank_0.txt"))
        for g in GROUPS + ("edge",):
            v = dict((int(a), b) for a, b in evals.get(f"eval/perf_group/{g}_score", []))
            m[f"{g}_score"] = v.get(e)
        for tag, key in (("eval/normalized_jerk_mean", "jerk_all_tb"), ("eval/high_jerk_frame_percentage_mean",
                         "high_jerk_all_tb"), ("eval/success_rate", "success_all_tb"), ("eval/perf/score", "perf_all_tb"),
                         ("eval/drag/all_J", "drag_all_tb"), ("eval/perf/substitution_holds_v2_x0", "subst_all_tb"),
                         ("eval/perf/worst10_score", "worst10_tb"), ("eval/curriculum/ess", "ess_tb")):
            v = dict((int(a), b) for a, b in evals.get(tag, []))
            m[key] = v.get(e)
        series[e] = m

    # the standalone evaluation of epoch 3,420 (the analysed checkpoint)
    agg = json.load(open(STANDALONE / "aggregate.json"))["log"]
    m = csv_metrics(read_csv(STANDALONE / "curriculum/eval_epoch_000000.csv"),
                    failed_ids(STANDALONE / "failed_motions/failed_motions_epoch_0_rank_0.txt"))
    for g in GROUPS + ("edge",):
        m[f"{g}_score"] = agg.get(f"eval/perf_group/{g}_score")
    m.update(jerk_all_tb=agg.get("eval/normalized_jerk_mean"), high_jerk_all_tb=agg.get("eval/high_jerk_frame_percentage_mean"),
             success_all_tb=agg.get("eval/success_rate"), perf_all_tb=agg.get("eval/perf/score"),
             drag_all_tb=agg.get("eval/drag/all_J"), subst_all_tb=agg.get("eval/perf/substitution_holds_v2_x0"),
             worst10_tb=agg.get("eval/perf/worst10_score"))
    series["3420s"] = m

    # G1 epoch 5,000 (the warm start): its standalone evaluation on release v2 (human only by construction)
    g1agg = json.load(open(G1_STANDALONE / "aggregate.json"))["log"]
    g1csv = sorted((G1_STANDALONE / "curriculum").glob("eval_epoch_*.csv"))[-1]
    g1 = csv_metrics(read_csv(g1csv), failed_ids(next((G1_STANDALONE / "failed_motions").glob("*.txt"))))
    for g in GROUPS:
        g1[f"{g}_score"] = g1agg.get(f"eval/perf_group/{g}_score")
    g1.update(jerk_all_tb=g1agg.get("eval/normalized_jerk_mean"), high_jerk_all_tb=g1agg.get("eval/high_jerk_frame_percentage_mean"),
              success_all_tb=g1agg.get("eval/success_rate"), drag_all_tb=g1agg.get("eval/drag/all_J"),
              subst_all_tb=g1agg.get("eval/perf/substitution_holds_v2_x0"))

    # jerk from the libraries (human only), validated against the logged all-motion mean
    libs = {1: RUN / "results/predicted_motion_lib_epoch_1.pt", 1500: RUN / "results/predicted_motion_lib_epoch_1500.pt",
            3000: RUN / "results/predicted_motion_lib_epoch_3000.pt",
            "3420s": STANDALONE / "results/predicted_motion_lib_epoch_0.pt"}
    jerk = {}
    for k, p in libs.items():
        jerk[str(k)] = library_jerk(p)
        series[k]["jerk_human_lib"] = jerk[str(k)]["human"]["jerk"]
        series[k]["high_jerk_human_lib"] = jerk[str(k)]["human"]["high_jerk_pct"]
        series[k]["jerk_syn_lib"] = jerk[str(k)]["syn"]["jerk"]
        series[k]["jerk_all_lib"] = jerk[str(k)]["all"]["jerk"]
    g1lib = G1_STANDALONE / "results/predicted_motion_lib_epoch_0.pt"
    jerk["g1_5000s"] = library_jerk(g1lib)
    g1["jerk_human_lib"] = jerk["g1_5000s"]["human"]["jerk"]
    g1["high_jerk_human_lib"] = jerk["g1_5000s"]["human"]["high_jerk_pct"]
    g1["jerk_all_lib"] = jerk["g1_5000s"]["all"]["jerk"]

    # ---- print
    cols = epochs + ["3420s"]
    print(f"evaluations: {epochs} + the standalone 3420 (3420s); G1 = G1's standalone epoch 5,000\n")
    head = f"{'metric':26s} {'G1 5000':>8s} | " + " ".join(f"{str(c):>7s}" for c in cols) + " | last3   gate"
    print(head)

    def line(name, key, gate=None, kind=None, fmt="7.3f"):
        vals = [series[c].get(key) for c in cols]
        last3 = [series[c].get(key) for c in LAST3 if series[c].get(key) is not None]
        mean3 = float(np.mean(last3)) if last3 else float("nan")
        ok = ""
        if gate is not None and last3:
            v = mean3
            ok = {"ge": v >= gate, "le": v <= gate, "lt": v < gate}[kind]
            ok = f"{'pass' if ok else 'FAIL'} ({kind} {gate})"
        g1v = g1.get(key)
        s = " ".join(f"{v:{fmt}}" if isinstance(v, (int, float)) and v is not None else f"{'-':>7s}" for v in vals)
        g1s = f"{g1v:8.3f}" if isinstance(g1v, (int, float)) else f"{'-':>8s}"
        print(f"{name:26s} {g1s} | {s} | {mean3:6.3f} {ok}")
        return {"name": name, "key": key, "series": dict(zip(map(str, cols), vals)), "g1_5000": g1v,
                "last3_mean": mean3, "gate": gate, "kind": kind}

    table = []
    for g in GROUPS:
        table.append(line(f"{g} score", f"{g}_score", FLOORS[g], "ge"))
    table.append(line("edge score", "edge_score"))
    table.append(line("human drag J", "human_drag_J", DRAG_MAX, "le", "7.1f"))
    table.append(line("human subst v2 x0", "human_subst_v2_x0", SUBST_MAX, "le", "7.0f"))
    table.append(line("human success", "human_success"))
    table.append(line("human score mean", "human_score_mean"))
    table.append(line("jerk all (logged)", "jerk_all_tb", fmt="7.1f"))
    table.append(line("jerk all (library)", "jerk_all_lib", fmt="7.1f"))
    table.append(line("jerk human (library)", "jerk_human_lib", JERK_MAX, "lt", "7.1f"))
    table.append(line("jerk SYN (library)", "jerk_syn_lib", fmt="7.1f"))
    table.append(line("high-jerk % all (logged)", "high_jerk_all_tb", fmt="7.2f"))
    table.append(line("high-jerk % human (lib)", "high_jerk_human_lib", HIGHJERK_MAX, "lt", "7.2f"))
    table.append(line("success all (logged)", "success_all_tb"))
    table.append(line("drag all (logged)", "drag_all_tb", fmt="7.1f"))
    table.append(line("subst v2 x0 all (logged)", "subst_all_tb", fmt="7.0f"))
    table.append(line("worst-10 (logged)", "worst10_tb"))
    table.append(line("SYN p_track mean", "syn_p_track_mean"))
    print("\nhuman motions failed per evaluation:")
    for c in cols:
        print(f"  {c}: {series[c]['human_failed']}  (SYN failed: {series[c]['syn_failed']})")
    print("\nSYN p_track by edge:")
    for c in cols:
        print(f"  {c}: {series[c]['syn_p_track_by_edge']}")

    out = {"epochs": [str(c) for c in cols], "last3": list(LAST3), "floors": FLOORS,
           "gates": {"drag_max": DRAG_MAX, "jerk_max": JERK_MAX, "high_jerk_max": HIGHJERK_MAX, "subst_max": SUBST_MAX},
           "series": {str(k): v for k, v in series.items()}, "g1_5000_standalone": g1, "table": table,
           "jerk": {k: {kk: vv for kk, vv in v.items() if kk != "per_motion"} for k, v in jerk.items()},
           "jerk_per_motion": {k: v["per_motion"] for k, v in jerk.items()}}
    (HERE / "data/gates.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\n-> {HERE / 'data/gates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
