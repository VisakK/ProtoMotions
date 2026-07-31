# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc: the node's stability region is a V **band**, not a V **half-line**.

``node_stability_region.py`` tests the natural hypothesis ``hold(s) <=> V(s) > V*``.
On a *converged* node that hypothesis is **false**, and instructively so.

What actually happens (measured on `node_handstand` @ epoch 1870, 49k rollouts):
V sits at a nominal value on-distribution -- 109.2, which is exactly the
infinite-horizon return ``1.09 / (1 - 0.99)`` for a policy that holds forever, so
the critic is *well calibrated where it was trained*. Off-distribution it does not
fall towards the true truncated return; it **drifts upward** (109.2 -> 118.8 as the
velocity kick grows) with rapidly growing spread (std 0.42 -> 9.76), while the
actual hold rate collapses 0.96 -> 0.01. So the ranking by V is *anti*-predictive
(AUC 0.298) even though the empirical hold rate is a clean inverted-U in V.

The reason is not a bug, it is the training distribution. The node was trained with
mild disturbance (reset scale 1) and became near-perfect, so the critic almost never
observed a failure and never had to learn a value gradient across the stability
boundary. Off that distribution its output is unconstrained extrapolation.

What *is* predictive is the **deviation from the on-distribution value**,
``-|V - V_nominal|`` (AUC 0.887): large deviation in *either* direction means "this
state is unlike anything I mastered". So the usable certificate is

    hold(s)  <=>  |V(s) - V_nominal| < delta

This script recomputes that from the ``.npz`` written by
``node_stability_region.py`` and reports, per perturbation family: AUC of V, of
``-|V - V_nom|`` and of the geometric control (root speed); the best two-sided band;
and a conservative band whose interior holds at ``--target``.

Usage::

    python data/scripts/node_value_band.py --node node_handstand
    python data/scripts/node_value_band.py --node node_handstand --out results/node_handstand/value_band.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels == 1, labels == 0
    n_p, n_n = pos.sum(), neg.sum()
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def best_band(v: np.ndarray, y: np.ndarray, grid: int = 160):
    """Two-sided threshold on V maximising accuracy."""
    los = np.quantile(v, np.linspace(0.0, 0.6, grid))
    his = np.quantile(v, np.linspace(0.4, 1.0, grid))
    best = (-1.0, float("nan"), float("nan"))
    for lo in los:
        m1 = v >= lo
        for hi in his:
            if hi <= lo:
                continue
            pred = m1 & (v <= hi)
            acc = float((pred == (y == 1)).mean())
            if acc > best[0]:
                best = (acc, float(lo), float(hi))
    return best


def conservative_delta(v: np.ndarray, y: np.ndarray, v_nom: float, target: float):
    """Widest |V - V_nom| <= delta whose interior still holds at >= target."""
    dev = np.abs(v - v_nom)
    order = np.argsort(dev)
    dev_s, y_s = dev[order], y[order]
    running = np.cumsum(y_s) / np.arange(1, len(y_s) + 1)
    ok = np.where(running >= target)[0]
    if len(ok) == 0:
        return None
    k = int(ok[-1])
    return {
        "delta": float(dev_s[k]),
        "hold_rate_inside": float(running[k]),
        "n_inside": k + 1,
        "frac_of_family": float((k + 1) / len(y_s)),
    }


def metrics_for_band(v, y, lo, hi):
    pred = (v >= lo) & (v <= hi)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    return {"lo": float(lo), "hi": float(hi),
            "accuracy": (tp + tn) / len(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node", required=True, help="e.g. node_handstand")
    p.add_argument("--npz", default=None, help="defaults to results/<node>/stability_region.npz")
    p.add_argument("--target", type=float, default=0.95,
                   help="Hold rate the conservative band must sustain.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    npz = Path(args.npz or f"results/{args.node}/stability_region.npz")
    d = np.load(npz, allow_pickle=True)
    fam = d["family"].astype(str)
    V, y, scale, spd = d["value"], d["success"].astype(int), d["scale"], d["init_root_speed"]

    out = {"node": args.node, "npz": str(npz), "target": args.target, "families": {}}

    fit = None
    for f in ("velocity", "mixed"):
        m = fam == f
        if not m.any():
            continue
        v, yy, s = V[m], y[m], spd[m]
        zero = (scale[m] == 0)
        v_nom = float(np.median(v[zero])) if zero.any() else float(np.median(v))
        acc, lo, hi = best_band(v, yy)
        entry = {
            "n": int(m.sum()),
            "hold_rate": float(yy.mean()),
            "V_nominal": v_nom,
            "auc_V": auc_roc(v, yy),
            "auc_neg_abs_dev": auc_roc(-np.abs(v - v_nom), yy),
            "auc_neg_root_speed": auc_roc(-s, yy),
            "best_band": metrics_for_band(v, yy, lo, hi),
            "conservative": conservative_delta(v, yy, v_nom, args.target),
        }
        out["families"][f] = entry
        if f == "velocity":
            fit = (lo, hi, v_nom)

        print(f"\n=== {args.node} / {f}   n={entry['n']}  hold={entry['hold_rate']:.3f} ===")
        print(f"  V_nominal (scale-0 median)   {v_nom:9.2f}")
        print(f"  AUC  V                       {entry['auc_V']:9.4f}   <- the proposed feature")
        print(f"  AUC -|V - V_nom|             {entry['auc_neg_abs_dev']:9.4f}   <- the band feature")
        print(f"  AUC -root_speed              {entry['auc_neg_root_speed']:9.4f}   <- geometric control")
        b = entry["best_band"]
        print(f"  best band [{b['lo']:.2f}, {b['hi']:.2f}]  acc={b['accuracy']:.4f} "
              f"prec={b['precision']:.4f} rec={b['recall']:.4f}")
        c = entry["conservative"]
        if c:
            print(f"  conservative |V-V_nom| <= {c['delta']:.2f}: hold {c['hold_rate_inside']:.4f} "
                  f"over {c['frac_of_family']*100:.1f}% of the family")

    # transfer: band fitted on velocity, applied to mixed
    if fit is not None and (fam == "mixed").any():
        lo, hi, _ = fit
        m = fam == "mixed"
        out["transfer_velocity_band_on_mixed"] = metrics_for_band(V[m], y[m], lo, hi)
        t = out["transfer_velocity_band_on_mixed"]
        print(f"\n  transfer: velocity band on mixed -> accuracy {t['accuracy']:.4f} "
              f"(prec {t['precision']:.4f}, rec {t['recall']:.4f})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
