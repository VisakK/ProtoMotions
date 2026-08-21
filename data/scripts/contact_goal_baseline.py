# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Is the contact half of a goal redundant with the pose half?

The Stage-2 student is conditioned on two things that describe the same instant:
the contact configuration being held, and the pose being held.  If the contact
set were recoverable from the pose, the contact channel would be decoration --
and the run would produce a saturated diagnostic that looks like success.
``notes/Singleleg14_pressure_experiment.MD`` §3.2 is the cautionary tale: a
reward term whose target turned out to be a *constant*, discovered offline in
twenty seconds by a script like this one, after a GPU-night had already been
spent looking at it.

So this asks the question directly, on CPU, before training.  For every hold
frame in the contact graph it predicts the ground-contact zone set from the
*reference kinematics alone* -- a zone is in contact when its lowest collision
geom is within ``eps`` of the floor -- and scores that against the force-annotated
truth.  The threshold is swept and the **best** value reported, so a poor score
cannot be blamed on a badly chosen one.

Read the output as: the closer the best F1 is to 1.0, the more redundant the
contact channel; the gap is what the channel adds.  Compare it against the
trivial predictors, because "hard to beat a constant" is the failure mode that
matters.

Usage::

    PYTHONPATH=. python data/scripts/contact_goal_baseline.py \
      --graph data/smpl/yoga_contact_graph/contact_graph.json \
      --motion-file data/smpl/yoga_yogi_student171.pt \
      --mjcf data/assets/smpl/smpl_yogi03596_lowtorque.xml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_geometry import (  # noqa: E402
    geom_ground_distance,
    geom_to_world,
    parse_typed_geoms,
)
from extract_contact_configs import ZONE_ORDER, ZONES, mjcf_body_names  # noqa: E402


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=str, required=True,
                        help="contact_graph.json from the rollout graph builder")
    parser.add_argument("--motion-file", type=str, required=True,
                        help="the packaged .pt the graph is keyed to")
    parser.add_argument("--mjcf", type=str,
                        default="data/assets/smpl/smpl_yogi03596_lowtorque.xml")
    parser.add_argument("--eps", type=float, nargs="*", default=None,
                        help="ground-gap thresholds to sweep, metres")
    parser.add_argument("--max-holds", type=int, default=0,
                        help="cap the number of hold frames scored (0 = all)")
    return parser


def zone_ground_gaps(
    body_pos: torch.Tensor, body_rot: torch.Tensor, typed_geoms, body_names
) -> torch.Tensor:
    """Smallest collision-geom-to-floor gap per zone. ``[T, num_zones]``."""
    gaps = torch.full((body_pos.shape[0], len(ZONE_ORDER)), float("inf"))
    index = {name: i for i, name in enumerate(body_names)}
    for zone_index, zone in enumerate(ZONE_ORDER):
        for body in ZONES[zone]:
            body_id = index[body]
            for geom in typed_geoms[body]:
                world = geom_to_world(
                    geom, body_pos[:, body_id], body_rot[:, body_id]
                )
                gap, _ = geom_ground_distance(world)
                gaps[:, zone_index] = torch.minimum(gaps[:, zone_index], gap)
    return gaps


def score(prediction: np.ndarray, truth: np.ndarray) -> dict:
    tp = float((prediction & truth).sum())
    fp = float((prediction & ~truth).sum())
    fn = float((~prediction & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = float((prediction == truth).all(axis=1).mean())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_match": exact,
        "per_zone_accuracy": float((prediction == truth).mean()),
    }


def main() -> int:
    args = create_parser().parse_args()
    graph = json.loads(Path(args.graph).read_text())
    packaged = torch.load(args.motion_file, map_location="cpu", weights_only=False)
    motion_files = [Path(f).stem for f in packaged["motion_files"]]

    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    lib = MotionLib(config=MotionLibConfig(motion_file=args.motion_file), device="cpu")

    body_names = mjcf_body_names(args.mjcf)
    typed_geoms = parse_typed_geoms(args.mjcf, body_names)

    # Collect every trusted hold: (motion id, hold time, ground-zone truth).
    motion_ids, hold_times, truth_rows = [], [], []
    for clip, record in graph["clips"].items():
        if clip not in motion_files:
            continue
        motion_id = motion_files.index(clip)
        for segment in record["segments"]:
            if not segment["trusted"]:
                continue
            ground = {p[:-2] for p in segment["pairs"] if p.endswith(":G")}
            motion_ids.append(motion_id)
            hold_times.append(segment["t_hold"])
            truth_rows.append([zone in ground for zone in ZONE_ORDER])

    if not motion_ids:
        raise SystemExit("no trusted holds in the graph")
    if args.max_holds and len(motion_ids) > args.max_holds:
        keep = np.linspace(0, len(motion_ids) - 1, args.max_holds).astype(int)
        motion_ids = [motion_ids[i] for i in keep]
        hold_times = [hold_times[i] for i in keep]
        truth_rows = [truth_rows[i] for i in keep]

    truth = np.asarray(truth_rows, dtype=bool)
    print(f"{len(motion_ids)} trusted hold frames over "
          f"{len(set(motion_ids))} clips, {len(ZONE_ORDER)} zones")

    state = lib.get_motion_state(
        torch.tensor(motion_ids, dtype=torch.long), torch.tensor(hold_times)
    )
    gaps = zone_ground_gaps(
        state.rigid_body_pos, state.rigid_body_rot, typed_geoms, body_names
    ).numpy()

    thresholds = args.eps or [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
    print("\nkinematic predictor: a zone is in contact when its lowest geom is "
          "within eps of the floor")
    print(f"  {'eps (m)':>9} {'F1':>7} {'prec':>7} {'recall':>7} "
          f"{'exact set':>10} {'per-zone acc':>13}")
    best = None
    for eps in thresholds:
        result = score(gaps <= eps, truth)
        marker = ""
        if best is None or result["f1"] > best[1]["f1"]:
            best = (eps, result)
            marker = "  <-"
        print(f"  {eps:>9.3f} {result['f1']:>7.3f} {result['precision']:>7.3f} "
              f"{result['recall']:>7.3f} {result['exact_set_match']:>10.3f} "
              f"{result['per_zone_accuracy']:>13.3f}{marker}")

    print("\ntrivial predictors, for scale:")
    for name, prediction in (
        ("always feet", np.tile(
            [[z in ("L_FOOT", "R_FOOT") for z in ZONE_ORDER]], (len(truth), 1))),
        ("most common set", np.tile(
            [list(Counter(map(tuple, truth.tolist())).most_common(1)[0][0])],
            (len(truth), 1))),
        ("never", np.zeros_like(truth)),
    ):
        result = score(np.asarray(prediction, dtype=bool), truth)
        print(f"  {name:>16}: F1 {result['f1']:.3f}  exact set "
              f"{result['exact_set_match']:.3f}")

    eps, result = best
    print(f"\nbest kinematic threshold {eps:.3f} m -> F1 {result['f1']:.3f}, "
          f"exact configuration recovered on {100 * result['exact_set_match']:.1f} % "
          f"of holds")
    print("The residual is what the force-annotated contact channel carries that "
          "the goal pose does not.")

    # Where the kinematic predictor goes wrong is more informative than how often.
    prediction = gaps <= eps
    false_negative = (~prediction & truth).sum(axis=0)
    false_positive = (prediction & ~truth).sum(axis=0)
    print(f"\n{'zone':>14} {'holds in contact':>17} {'missed':>8} {'invented':>10}")
    order = np.argsort(-(false_negative + false_positive))
    for zone_index in order:
        if false_negative[zone_index] + false_positive[zone_index] == 0:
            continue
        print(f"  {ZONE_ORDER[zone_index]:>12} {int(truth[:, zone_index].sum()):>17} "
              f"{int(false_negative[zone_index]):>8} {int(false_positive[zone_index]):>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
