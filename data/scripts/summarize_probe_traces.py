# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate a probe suite's per-sequence trace JSONs into one feasibility table.

Reads the ``<label>.json`` files ``render_contact_goal_sequence.py`` writes
(each carries a per-goal ``summary`` computed over that goal's hold window and
the full per-step ``trace``) and prints one row per sequence:

* ``goals`` / ``exact`` — how many hold windows ended in exactly the wanted
  ground-contact set. Standing-family goals are counted separately (``std``):
  their contact set is degenerate (any stand satisfies it), so their "exact"
  is necessary, never sufficient — judge them on the video.
* ``iou`` — mean ground IoU over all non-standing hold windows.
* ``min_z`` / ``fell`` — the lowest root height on any frame and whether it
  dropped below ``--fall-z`` (default 0.30 m) outside goals whose own target
  is a floor pose (supine/prone family with root that low is often the pose,
  not a fall — the flag is a triage signal, not a verdict).

Verdict triage: ``ok`` (every non-standing goal exact, no fall),
``partial`` (some exact, or IoU >= 0.5 everywhere, no fall), ``fell`` /
``failed`` otherwise. The verdict column exists to sort twenty videos by
which to watch first — the video stays the ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STANDING_ZONES = ("L_FOOT", "R_FOOT")


def is_standing_goal(row) -> bool:
    return sorted(row.get("wanted_ground_zones", [])) == sorted(STANDING_ZONES)


def summarize_file(path: Path, fall_z: float):
    payload = json.loads(path.read_text())
    summary = payload.get("summary", [])
    trace = payload.get("trace", {})
    root_z = trace.get("root_z", [])

    rows_std = [r for r in summary if is_standing_goal(r)]
    rows_other = [r for r in summary if not is_standing_goal(r)]

    exact_other = sum(bool(r["reached_exactly"]) for r in rows_other)
    exact_std = sum(bool(r["reached_exactly"]) for r in rows_std)
    ious = [r["ground_iou_mean_over_hold"] for r in rows_other]
    mean_iou = sum(ious) / len(ious) if ious else float("nan")
    min_z = min(root_z) if root_z else float("nan")
    fell = bool(root_z) and min_z < fall_z

    if rows_other and exact_other == len(rows_other) and not fell:
        verdict = "ok"
    elif not rows_other and not fell:
        verdict = "ok(pose)"  # standing-family only: contact says nothing
    elif fell:
        verdict = "fell?"
    elif exact_other > 0 or (ious and min(ious) >= 0.5):
        verdict = "partial"
    else:
        verdict = "failed"

    return {
        "label": payload.get("label", path.stem),
        "goals": len(summary),
        "exact_other": f"{exact_other}/{len(rows_other)}",
        "exact_std": f"{exact_std}/{len(rows_std)}",
        "mean_iou": mean_iou,
        "min_root_z": min_z,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--fall-z", type=float, default=0.30)
    args = parser.parse_args()

    files = sorted(Path(args.dir).glob(f"{args.prefix}*.json"))
    if not files:
        raise SystemExit(f"no {args.prefix}*.json traces in {args.dir}")

    rows = [summarize_file(f, args.fall_z) for f in files]
    width = max(len(r["label"]) for r in rows) + 2
    header = (
        f"{'sequence':<{width}}{'goals':>6}{'exact':>8}{'std':>7}"
        f"{'iou':>7}{'min_z':>7}  verdict"
    )
    print(header)
    print("-" * len(header))
    order = {"failed": 0, "fell?": 1, "partial": 2, "ok(pose)": 3, "ok": 4}
    for r in sorted(rows, key=lambda r: (order.get(r["verdict"], 9), r["label"])):
        print(
            f"{r['label']:<{width}}{r['goals']:>6}{r['exact_other']:>8}"
            f"{r['exact_std']:>7}{r['mean_iou']:>7.2f}{r['min_root_z']:>7.2f}"
            f"  {r['verdict']}"
        )
    print(
        "\nNOTE: 'exact'/'iou' cover non-standing goals only; standing-family "
        "goals (std column) and warrior II/III are pose-judged - watch the "
        "video. 'fell?' uses a root-height triage threshold and misflags "
        "genuine floor poses; it orders viewing, it is not a verdict."
    )


if __name__ == "__main__":
    main()
