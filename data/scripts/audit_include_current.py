# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How much of training is spent commanded to LEAVE the pose you are in?

The v10_1 gate for ``--include-current-segment``.  The round-9 diagnosis
measured that **42.1 % of all trusted dwell** is spent inside a segment while
already commanded to the *next* hold -- because ``next_goal_indices`` returns
the first hold beyond ``min_lead_s``, and once the current segment's own hold
frame is behind you that is always a different segment.  That is where
"anticipatory departure at a commanded pose" is learned.

This calls ``ContactGraph.next_goal_indices`` itself rather than re-deriving
the schedule, so it measures the shipped implementation, not a model of it.
It sweeps clip time on a fixed stride and reports, over frames that lie inside
a trusted segment, the share whose commanded slot-0 goal is some *other*
segment.

Usage::

    PYTHONPATH=. python data/scripts/audit_include_current.py \
      --graph data/smpl/yoga_contact_graph_student44h_v101/contact_graph.pt
"""

from __future__ import annotations

import argparse

import torch

from protomotions.components.contact_graph import ContactGraph


def audit(graph: ContactGraph, stride_s: float, num_steps: int) -> dict:
    """Fraction of in-segment time whose slot-0 goal is a different segment."""
    motion_ids, times, seg_of = [], [], []
    starts, ends, counts = graph.seg_start, graph.seg_end, graph.seg_count
    for m in range(starts.shape[0]):
        for s in range(int(counts[m])):
            t0, t1 = float(starts[m, s]), float(ends[m, s])
            t = t0
            while t <= t1:
                motion_ids.append(m); times.append(t); seg_of.append(s)
                t += stride_s
    motion_ids = torch.tensor(motion_ids, dtype=torch.long, device=graph.device)
    times = torch.tensor(times, dtype=torch.float32, device=graph.device)
    seg_of = torch.tensor(seg_of, dtype=torch.long, device=graph.device)

    out = {"frames": int(motion_ids.numel())}
    for label, flag in (("legacy", False), ("include_current", True)):
        idx, valid = graph.next_goal_indices(
            motion_ids, times, num_steps, include_current=flag
        )
        slot0, ok = idx[:, 0], valid[:, 0]
        elsewhere = (slot0 != seg_of) & ok
        out[label] = float(elsewhere.float().mean())
        out[label + "_invalid"] = float((~ok).float().mean())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True, help="path to contact_graph.pt")
    p.add_argument("--stride-s", type=float, default=1.0 / 30.0)
    p.add_argument("--num-steps", type=int, default=5)
    args = p.parse_args()

    payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    graph = ContactGraph(payload, device="cpu")
    r = audit(graph, args.stride_s, args.num_steps)
    print(f"{args.graph}")
    print(f"  in-segment frames sampled : {r['frames']}")
    print(f"  commanded ELSEWHERE, legacy          : {100*r['legacy']:.1f} %")
    print(f"  commanded ELSEWHERE, include_current : {100*r['include_current']:.1f} %")
    print(f"  slot 0 invalid (legacy / new)        : "
          f"{100*r['legacy_invalid']:.1f} % / {100*r['include_current_invalid']:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
