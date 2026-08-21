# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure selection / horizon logic for :mod:`render_policy_videos`.

Separated so it can be unit-tested without a simulator: the renderer itself
imports IsaacLab at module scope, which makes every function in it untestable
on CPU.  Nothing here touches torch, Isaac, or the filesystem.
"""

from __future__ import annotations

import csv
from typing import List, Optional, Sequence, Tuple


def select_motions(
    names: Sequence[str],
    rank_csv: Optional[str],
    worst: Optional[int],
    explicit: Optional[Sequence[int]],
    name_filters: Optional[Sequence[str]] = None,
) -> List[Tuple[int, str, Optional[float]]]:
    """-> list of (motion_id, label_prefix, gt_error or None), in render order."""
    if explicit:
        sel = [(i, f"{k:03d}", None) for k, i in enumerate(explicit)]
    elif rank_csv:
        with open(rank_csv) as f:
            rows = list(csv.DictReader(f))
        # eval_per_motion.py writes worst-first, so file order is the ranking.
        sel = []
        for rank, r in enumerate(rows, start=1):
            mid = int(r["motion_id"])
            err = float(r.get("gt_error_mean", "nan"))
            failed = r.get("failed", "False") == "True"
            sel.append((mid, f"{rank:03d}{'_FAIL' if failed else ''}", err))
        if worst:
            sel = sel[:worst]
    else:
        sel = [(i, f"{i:03d}", None) for i in range(len(names))]

    if name_filters:
        needles = [s.lower() for s in name_filters]
        sel = [e for e in sel if any(n in names[e[0]].lower() for n in needles)]
    return sel


def plan_rollout(
    clip_len: float, dt: float, max_seconds: float, min_seconds: float
) -> Tuple[int, float, bool]:
    """How many policy steps to roll, and whether short clips may be replayed.

    ``max_seconds <= 0`` means the whole clip -- the old 12 s default silently
    truncated every hard-pose clip (they run 16-59 s) so the video ended before
    the pose was entered.  ``min_seconds`` is the opposite guard: a clip shorter
    than it is *replayed* from the top rather than padded, because padding would
    record the policy holding a finished reference, which reads as a hang.

    **An explicit cap outranks the floor.**  ``--max-seconds 10 --min-seconds 15``
    records 10 s: a cap the caller typed is a hard limit, and silently recording
    half again as much would be the same class of surprise as the 12 s default
    this function exists to remove.

    Returns ``(steps, target_seconds, allow_replay)``.  ``allow_replay`` is true
    only when the target genuinely exceeds the clip, so a replay can never be
    armed for a clip that is already long enough.
    """
    cap = float("inf") if max_seconds <= 0 else max_seconds
    target = min(clip_len, cap)
    if min_seconds > target:
        target = min(min_seconds, cap)
    allow_replay = target > clip_len + 1e-6
    return max(int(round(target / dt)), 1), target, allow_replay


def outcome_suffix(seconds: float, reason: str, fail_time: Optional[float]) -> str:
    """Compact, sortable tag appended to every file name.

    A 6-second video is ambiguous on its own -- truncated, or did the policy fall
    over?  This makes the answer part of the file name.
    """
    tag = f"_{seconds:04.1f}s"
    if fail_time is not None:
        tag += f"_FELL@{fail_time:04.1f}s"
    elif reason != "full":
        tag += f"_{reason}"
    return tag
