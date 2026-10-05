"""S0's collation: X0 at h = 1, the paired horizons and the fork battery, side by side for several checkpoints.

Reads the output directories of ``data/scripts/goal_causality_x0.py`` (one per checkpoint) and writes
``data/s0_compare.json`` and the figures in ``figures/``:

* ``x0_h1_curves.png`` -- per hub, the median ``|da| / sigma_a`` at h = 1 against the dwell remaining, for the
  different-goal and the same-pose partners, one colour per checkpoint;
* ``x0_paired.png`` -- per hub and split, the same-goal floor and the different-goal response at h = 1 / 8 / 24, as
  an action distance and as a body-position divergence;
* ``forks.png`` -- per fork-battery goal, the success rate (pose held, commanded supports loaded, no substitution).

Every ``|da| / sigma_a`` is in that run's own sigma_a (the policy's own per-DOF action spread over the corpus):
X0 asks whether a goal moves the action by a meaningful share of what the policy itself spans. The RMS ratio of each
run's sigma_a to the first run's is reported beside it.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/s0_x0/compare.py \\
        G3_e3420=output/goal_causality_x0/g3_e3420+output/goal_causality_x0/g3_e3420_forks \\
        e15500=output/goal_causality_x0/e15500+output/goal_causality_x0/e15500_forks \\
        G1_e5000=output/goal_causality_x0/g1_e5000_forks G3_e2000=output/goal_causality_x0/g3_e2000_forks
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPH_JSON = REPO / "data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v3.2f132f4299/contact_graph.json"
NEAR = {"standing": 0.2, "crow": 1.0, "handstand": 1.0}     # "near the end of the dwell", s
FAR = {"standing": 0.45, "crow": 3.0, "handstand": 3.0}


def node_names():
    nodes = json.loads(GRAPH_JSON.read_text())["nodes"]
    out = {}
    for i, n in enumerate(nodes):
        base = re.split(r"_pose_or_|_Pose_or_", n["name"])[0]
        sub = re.search(r"_(h\d)$", n["name"])
        out[i] = base + (f"_{sub.group(1)}" if sub else "")
    return out


def short(stem: str) -> str:
    s = re.sub(r"^2209\d\d_", "", stem)
    s = re.sub(r"_pose_or_[A-Za-z_]+?(_-[a-d])", r"\1", s, flags=re.I)
    s = re.sub(r"_Pose_or_[A-Za-z_]+?(_-[a-d])", r"\1", s)
    return s


def load(spec: str):
    """``dirA[+dirB...]``: each file is read from the LAST listed directory that has it (a later fork-only run
    overrides an earlier run's fork battery)."""
    dirs = [Path(d) for d in spec.split("+")]

    def find(name):
        hits = [d / name for d in dirs if (d / name).exists()]
        return hits[-1] if hits else None

    rec = dict(path=spec, meta=json.loads(find("meta.json").read_text()))
    if find("corpus.json"):
        rec["corpus"] = json.loads(find("corpus.json").read_text())
    if find("x0_summary.json"):
        rec["summary"] = json.loads(find("x0_summary.json").read_text())
        rows = []
        with open(find("x0_h1_rows.jsonl")) as f:
            for line in f:
                rows.append(json.loads(line))
        rec["rows"] = rows
        rec["blocks"] = json.loads(find("x0_paired_blocks.json").read_text())
    if find("forks_scored.json"):
        rec["forks"] = json.loads(find("forks_scored.json").read_text())
        rec["forks_from"] = str(find("forks_scored.json"))
    return rec


def rescale(rows, own_sigma, ref_sigma):
    """|da|/sigma_a is an RMS over DOF of da_j / sigma_j; re-express it against a common sigma.

    Rows store only the RMS, so the exact re-scaling needs da per DOF, which the rows do not keep; the ratio of the
    two sigmas' RMS is used instead and reported. With one reference run (the default) the factor is 1.
    """
    f = float(np.sqrt(np.mean((np.asarray(own_sigma) / np.asarray(ref_sigma)) ** 2)))
    return f


def h1_tables(rec, names, factor):
    rows = rec["rows"]
    out = {}
    for hub in sorted({r["hub"] for r in rows}):
        hrows = [r for r in rows if r["hub"] == hub]
        curve = {}
        for rn in sorted({round(r["r_nominal"], 3) for r in hrows}, reverse=True):
            at = [r for r in hrows if round(r["r_nominal"], 3) == rn]
            item = {}
            for st in ("different", "same_pose", "identical", "between"):
                v = np.array([r["da_rms_z"] for r in at if r["stratum"] == st]) * factor
                item[st] = dict(n=int(len(v)), p50=float(np.median(v)) if len(v) else None,
                                p90=float(np.percentile(v, 90)) if len(v) else None)
            if item["different"]["p50"] is not None and item["same_pose"]["p50"] is not None:
                item["excess_p50"] = item["different"]["p50"] - item["same_pose"]["p50"]
            curve[str(rn)] = item
        # per own clip x partner slot-1 goal, near and far
        groups = defaultdict(list)
        for r in hrows:
            if r["stratum"] not in ("different", "same_pose", "identical"):
                continue
            when = "near" if r["r_nominal"] <= NEAR[hub] + 1e-6 else ("far" if r["r_nominal"] >= FAR[hub] - 1e-6
                                                                       else "mid")
            slot1 = names.get(r["partner_nodes"][-1], "?") if r["partner_valid"][-1] else "none"
            if hub == "standing":       # 54 own clips x their first holds: pooled, or the table is megabytes
                groups[("all 54 clips", r["stratum"], "any", when)].append(r["da_rms_z"] * factor)
            else:
                groups[(short(r["own"]), r["stratum"], slot1, when)].append(r["da_rms_z"] * factor)
        by_goal = [dict(own=k[0], stratum=k[1], partner_slot1=k[2], when=k[3], n=len(v), p50=float(np.median(v)))
                   for k, v in sorted(groups.items())]
        out[hub] = dict(curve=curve, by_partner_goal=by_goal)
    return out


def paired_tables(rec, factor):
    out = defaultdict(list)
    for hub, blocks in rec["summary"]["paired"].items():
        for b in blocks:
            for hh, v in b["horizons"].items():
                diff = [a for a in v["arms"].get("different", [])]
                same = [a for a in v["arms"].get("same_pose", [])] + [a for a in v["arms"].get("identical", [])]
                out[hub].append(dict(
                    own=short(b["own"]), r=b["r"], h=int(hh),
                    floor_da=None if v["same_goal_floor"] is None else v["same_goal_floor"] * factor,
                    floor_dpos_cm=None if v["same_goal_dpos_m"] is None else v["same_goal_dpos_m"] * 100,
                    same_da=float(np.median([a["da"] for a in same])) * factor if same else None,
                    same_dpos_cm=float(np.median([a["dpos_m"] for a in same])) * 100 if same else None,
                    diff_da=float(np.median([a["da"] for a in diff])) * factor if diff else None,
                    diff_dpos_cm=float(np.median([a["dpos_m"] for a in diff])) * 100 if diff else None,
                    n_diff=len(diff)))
    return dict(out)


def fork_table(rec):
    out = {}
    for plan, goals in rec.get("forks", {}).items():
        out[plan] = [{k: g.get(k) for k in ("goal", "n", "hold_rate24", "hold_rate6", "reach_rate24", "err24_p50",
                                            "err6_p50", "held_s_p50", "supports_realised", "supports_loaded",
                                            "missing_zones", "no_substitution", "success", "pelvis_z_p50",
                                            "loaded_sets", "commanded")} for g in goals]
    return out


def main(argv) -> int:
    runs = {}
    for arg in argv:
        label, path = arg.split("=", 1)
        runs[label] = load(path)
    names = node_names()
    ref = next((r for r in runs.values() if "corpus" in r), None)
    ref_sigma = ref["corpus"]["sigma_a"] if ref else None
    data = {}
    for label, rec in runs.items():
        item = dict(checkpoint=rec["meta"]["checkpoint"], config=rec["meta"]["config"])
        factor = 1.0
        if "corpus" in rec:
            c = rec["corpus"]
            item["corpus"] = {k: c[k] for k in ("sigma_a_mean", "sigma_a_min", "sigma_a_max", "samples",
                                                "replica_spread")}
            if ref_sigma is not None:
                # reported, not applied: each run's |da| is in its own sigma_a (the policy's own action spread)
                item["sigma_rms_ratio_to_reference"] = rescale(None, c["sigma_a"], ref_sigma)
        item["sigma_factor_to_reference"] = factor
        if "summary" in rec:
            item["checks"] = rec["summary"]["checks"]
            item["acceptance"] = rec["summary"]["acceptance"]
            item["h1"] = h1_tables(rec, names, factor)
            item["paired"] = paired_tables(rec, factor)
            item["h1_rows"] = len(rec["rows"])
        if "forks" in rec:
            item["forks"] = fork_table(rec)
            item["forks_from"] = rec["forks_from"]
        data[label] = item
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/s0_compare.json").write_text(json.dumps(dict(runs={k: v["path"] for k, v in runs.items()},
                                                               reference_sigma_run=next(iter(runs)) if ref else None,
                                                               data=data), indent=1) + "\n")
    _figures(data)
    _print(data)
    return 0


def _print(data):
    f = lambda x: "  -  " if x is None else f"{x:5.3f}"  # noqa: E731
    for label, d in data.items():
        print(f"\n=== {label}: {d['checkpoint']}  (sigma factor {d['sigma_factor_to_reference']:.3f})")
        if "corpus" in d:
            c = d["corpus"]
            print(f"  sigma_a mean {c['sigma_a_mean']:.3f} (min {c['sigma_a_min']:.3f}, max {c['sigma_a_max']:.3f})")
        if "checks" in d:
            print(f"  checks {json.dumps({k: v for k, v in d['checks'].items() if k != 'manual_path'})}")
            if d["checks"].get("manual_path"):
                mp = d["checks"]["manual_path"]
                print(f"  manual path differs on {[k for k, v in mp['max_abs_by_key'].items() if v > 0]}")
            print(f"  acceptance {json.dumps(d['acceptance'])}")
            for hub, t in d["h1"].items():
                print(f"  [{hub}] r: different p50 | same-pose p50 | excess  (n)")
                for rn, it in t["curve"].items():
                    print(f"     {float(rn):5.2f}: {f(it['different']['p50'])} | {f(it['same_pose']['p50'])} | "
                          f"{f(it.get('excess_p50'))}  ({it['different']['n']}/{it['same_pose']['n']})")
        if "forks" in d:
            print("  forks: plan / goal: success (hold24, supports, no-subst), loaded")
            for plan, goals in d["forks"].items():
                for g in goals:
                    top = next(iter(g["loaded_sets"]), "-")
                    print(f"     {plan[:34]:34s} {g['goal'][:22]:22s} {g['success']:.2f} ({g['hold_rate24']:.2f}, "
                          f"{g['supports_realised']:.2f}, {g['no_substitution']:.2f})  {top}")


# Reference palette (dataviz skill, references/palette.md): categorical slots 1-2 for the checkpoints (validated
# all-pairs, light surface), the sequential blue ramp for the fork heatmap, ink tokens for every text.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"G3_e3420": "#2a78d6", "e15500": "#eb6834"}
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]
DWELL_TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _figures(data):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import FixedLocator, NullLocator

    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    (HERE / "figures").mkdir(exist_ok=True)
    series = [k for k in SERIES if k in data and "h1" in data[k]]
    hubs = [h for h in ("standing", "crow", "handstand") if any(h in data[k]["h1"] for k in series)]

    # ---- 1. h = 1 response against the dwell remaining ------------------------------------------------- #
    if hubs:
        fig, axes = plt.subplots(1, len(hubs), figsize=(4.4 * len(hubs), 3.6), dpi=150, squeeze=False)
        for ax, hub in zip(axes[0], hubs):
            _style(ax)
            for label in series:
                t = data[label]["h1"].get(hub)
                if not t:
                    continue
                for st, ls, mk in (("different", "-", "o"), ("same_pose", "--", "s")):
                    pts = sorted((float(r), it[st]["p50"]) for r, it in t["curve"].items() if it[st]["p50"] is not None)
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker=mk, ms=4, lw=2,
                                color=SERIES[label], label=f"{label}: {'different goal' if st == 'different' else 'same pose'}")
            ax.axhline(0.05, color=INK2, lw=1, ls=":")
            ax.text(0.98, 0.05, "0.05 sigma_a", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=7, color=INK2)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(DWELL_TICKS))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels([f"{v:g}" for v in DWELL_TICKS])
            rs = [float(r) for label in series for r in data[label]["h1"].get(hub, {}).get("curve", {})]
            ax.set_xlim(max(rs) * 1.3, min(rs) / 1.3)          # time runs toward the departure
            ax.set_ylim(0, None)
            ax.set_xlabel("dwell remaining at the probe (s)", fontsize=8)
            ax.set_title(f"{hub} hub", fontsize=10)
        axes[0][0].set_ylabel("median |da| / sigma_a, h = 1", fontsize=8)
        axes[0][0].legend(fontsize=7, frameon=False, loc="center left")
        fig.suptitle("X0 at a fixed state: action change when only the goal window changes", fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(HERE / "figures/x0_h1_curves.png", bbox_inches="tight")
        plt.close(fig)

    # ---- 2. paired rollouts: response and body divergence against h ------------------------------------ #
    if hubs:
        fig, axes = plt.subplots(2, len(hubs), figsize=(4.4 * len(hubs), 6.2), dpi=150, squeeze=False)
        for col, hub in enumerate(hubs):
            for row, (k_diff, k_floor, ylab) in enumerate((("diff_da", "floor_da", "|da| / sigma_a"),
                                                          ("diff_dpos_cm", "floor_dpos_cm", "body divergence (cm)"))):
                ax = axes[row][col]
                _style(ax)
                floors = []
                for label in series:
                    rows = data[label].get("paired", {}).get(hub, [])
                    splits = sorted({r["r"] for r in rows})
                    if not splits:
                        continue
                    for r_split, ls, when in ((splits[0], "-", "near departure"), (splits[-1], "--", "far from it")):
                        sel = [r for r in rows if r["r"] == r_split and r[k_diff] is not None]
                        hs = sorted({r["h"] for r in sel})
                        ys = [float(np.median([r[k_diff] for r in sel if r["h"] == hh])) for hh in hs]
                        ax.plot(hs, ys, ls=ls, marker="o", ms=4, lw=2, color=SERIES[label],
                                label=f"{label}, split {when}")
                    floors += [r[k_floor] for r in rows if r[k_floor] is not None]
                if floors:
                    ax.axhspan(min(floors), max(floors), color=GRID, alpha=0.9, lw=0)
                    ax.text(24, max(floors), " same-goal floor (range)", fontsize=7, color=INK2, ha="right",
                            va="bottom")
                ax.set_yscale("log")
                ax.set_xticks([1, 8, 24])
                ax.set_xlabel("steps after the split (h, 30 Hz)", fontsize=8)
                if col == 0:
                    ax.set_ylabel(ylab, fontsize=8)
                if row == 0:
                    ax.set_title(f"{hub} hub: a different goal", fontsize=10)
        axes[0][0].legend(fontsize=6.5, frameon=False, loc="lower right")
        fig.text(0.5, -0.01, "Medians over blocks. Split near departure: 0.05 s (standing), 0.2 s (crow, handstand) before "
                 "the own clip leaves the hold; far: 0.5 s / 5 s. Grey band: the same-goal floor (own-arm replicas from "
                 "the same start), min to max over every block and horizon.", ha="center", va="top", fontsize=7,
                 color=INK2, wrap=True)
        fig.tight_layout()
        fig.savefig(HERE / "figures/x0_paired.png", bbox_inches="tight")
        plt.close(fig)

    # ---- 3. fork battery: success per commanded goal and checkpoint ----------------------------------- #
    order = [k for k in ("e15500", "G1_e5000", "G3_e2000", "G3_e3420") if "forks" in data.get(k, {})]
    if order:
        first = data[order[-1]]["forks"]

        def keep(plan, gi, goal):
            if plan.startswith("edge_") or plan.startswith("nohijack_"):
                return gi == 1                                   # the commanded continuation from a held S
            if plan.startswith("fork_edge_"):
                return True                                      # S from standing, then D
            return goal != "standing"                            # the family hold, not the return

        def label(plan, goal):
            if plan.startswith("fork_edge_"):
                return f"standing -> {plan[10:]}: {goal.split('_Pose')[0].split('_pose')[0]}"
            if plan.startswith("edge_") or plan.startswith("nohijack_"):
                kind = "edge" if plan.startswith("edge_") else "no-hijack"
                return f"{kind} {plan.split('_')[-1]} from held S: {goal.split('_Pose')[0].split('_pose')[0]}"
            return "fork: " + re.split(r"_Pose_or_|_pose_or_", plan[5:])[0].replace("_", " ")

        rows = [(p, gi, label(p, g["goal"])) for p, gl in first.items() for gi, g in enumerate(gl) if keep(p, gi, g["goal"])]
        groups = lambda r: (0 if r[2].startswith("edge") else 1 if r[2].startswith("no-hijack") else 2
                            if r[2].startswith("standing") else 3, r[2])
        rows.sort(key=groups)
        M = np.full((len(rows), len(order)), np.nan)
        for j, k in enumerate(order):
            fk = data[k]["forks"]
            for i, (p, gi, _) in enumerate(rows):
                if p in fk and gi < len(fk[p]):
                    M[i, j] = fk[p][gi]["success"]
        fig, ax = plt.subplots(figsize=(5.6, 0.2 * len(rows) + 1.4), dpi=150)
        cmap = ListedColormap(BLUES)
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(order)):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="#ffffff" if v >= 0.6 else INK)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([k.replace("_", " ") for k in order], fontsize=7, color=INK)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[2] for r in rows], fontsize=6, color=INK)
        for edge in ("top", "right", "bottom", "left"):
            ax.spines[edge].set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(order)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=1.5)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(length=0)
        ax.set_title("fork battery: share of replicas that hold the commanded pose (best yaw, < 0.15 m),\n"
                     "with every commanded zone down and no other zone loaded", fontsize=8, color=INK, pad=24)
        fig.tight_layout()
        fig.savefig(HERE / "figures/forks.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
