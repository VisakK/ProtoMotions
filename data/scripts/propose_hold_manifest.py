# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Propose the entry / hold / exit demarcation of every clip in a corpus.

``expert_revist/expert_revisit.MD`` §3–§5: the goal-conditioned expert's graph is
built from *holds* found on the kinematic reference, not from contact segments
found in expert rollouts. A hold is a window in which the reference is still;
the pose held there is a node; the motion between two holds is an edge. This
script proposes those windows and everything the graph needs about each one,
and writes a **reviewable manifest** -- the manifest, not the script, is the
durable artefact. Rename holds, move boundaries, drop or add windows in the
YAML and rebuild; the proposal is only a starting point.

Per clip:

1. **Stillness.** Mean body speed (``‖rigid_body_vel‖`` averaged over the 24
   bodies), median-filtered over ``--smooth-s``. Frames below ``--v-still`` form
   candidate windows; gaps shorter than ``--merge-s`` are bridged.
2. **Support split.** A still window is cut wherever the *ground* contact set
   (or the orientation bin) changes and stays changed: a crow clip is slow from
   the squat with the hands down all the way through the balance, and those are
   two holds, not one. Runs shorter than ``--min-hold-s`` are transitions and
   are dropped.
3. **Exemplar.** The frame of minimum smoothed speed inside the window is the
   hold frame (``t_hold``) -- the pose the goal names.
4. **Contact set.** The geometric detector
   (``extract_contact_configs.compute_active_pairs``: typed-geom surface
   distances, calibrated hysteresis, the stillness-gated body-body tier and the
   COM/support-polygon static model) is run once per clip, and a pair belongs to
   the hold if it is active on at least ``--pair-vote`` of the window's frames.
   This is the regime the detector was validated in: static frames, where the
   float-biased supports are recoverable. Body-body pairs whose two zones are
   both grounded are demoted (the load-path rule the shipped graphs use), so a
   standing hold does not fork on an incidental feet-touch flag. The manifest
   records both the full voted set (``pairs`` -- the goal vector) and its
   ground subset (``pairs_ground`` -- node identity), the same split the
   shipped graphs make with ``--node-identity ground``: a light feet-on-shins
   flag while standing must not fragment the standing hub, while crow's
   shin-on-upper-arm pairs must stay in the goal. One more narrowly scoped
   rule: leg-leg pairs while standing upright on both feet (feet together,
   ankles brushing shins) are dropped from the goal vector -- they are
   proximity, not load, and they appeared on every standing hold of the corpus.
5. **Orientation.** Majority trunk-orientation bin over the window.
6. **Name.** A window is ``standing`` when its ground set is both feet, the
   trunk is upright and its exemplar is within ``--rest-pose-m`` (6-body
   heading-normalised metric) of the clip's first or last frame -- the rest
   pose the capture protocol starts and ends in. The remaining windows are
   grouped by (ground set, orientation) *and* pose -- a window joins a group
   when its exemplar is within ``--same-hold-m`` of the group's longest
   member, so a wide-stance preparation and the warrior II it precedes stay
   apart while a clip that holds the handstand twice gets one handstand node
   with two segments. The family name goes to the longest group whose ground
   set matches what the family's pose is held on (``EXPECTED_SUPPORT``: an
   arm balance is held on the hands, a single-leg balance on one foot, a
   forearm balance on forearms and hands ...), falling back to the longest
   group when nothing matches -- because in several clips the longest still
   window is *not* the named pose (side crow -b and both Koundinyasana clips
   dwell longest in the tripod headstand they enter through). The other groups
   are ``<family>_h<k>`` in order of first appearance. Non-``standing`` holds are flagged
   ``extend: true`` so :mod:`make_hold_extended_clips` tiles them.
7. **Rest ends.** When no hold covers the first (last) ``--rest-lead-s`` of the
   clip -- the subject was settling, or walking into position -- a hold is added
   over the initial (final) run of the frame's support id, so the entry from
   the rest pose is an edge in the graph and the motion manager can anchor
   there. It is ``standing`` under the rule above, else ``rest_start`` /
   ``rest_end``, and never extended.

Usage::

    PYTHONPATH=. python data/scripts/propose_hold_manifest.py \
      --corpus data/smpl/expert60/corpus.yaml \
      --out data/smpl/expert60/holds.yaml --review data/smpl/expert60/review.md
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_contact_configs import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    ORIENT_BINS,
    compute_active_pairs,
)

DEFAULT_MJCF = "data/assets/smpl/smpl_yogi03596_lowtorque.xml"
STANDING_PAIRS = frozenset({"L_FOOT:G", "R_FOOT:G"})
LEG_ZONES = frozenset({"L_FOOT", "L_SHANK", "L_THIGH", "R_FOOT", "R_SHANK", "R_THIGH"})
FEET = frozenset({"L_FOOT", "R_FOOT"})
HANDS = frozenset({"L_HAND", "R_HAND"})
FOREARMS = frozenset({"L_FOREARM", "R_FOREARM"})


def _hands_only(ground: frozenset) -> bool:
    return ground == HANDS


def _forearms_and_hands(ground: frozenset) -> bool:
    return HANDS <= ground and FOREARMS <= ground and not (ground & FEET)


def _one_foot(ground: frozenset) -> bool:
    return len(ground & FEET) == 1


def _both_feet(ground: frozenset) -> bool:
    return FEET <= ground


def _head_and_hands(ground: frozenset) -> bool:
    return "HEAD" in ground and HANDS <= ground and not (ground & FEET)


def _head_on_shoulders(ground: frozenset) -> bool:
    return "HEAD" in ground and "TRUNK" in ground and not (ground & HANDS)


# Which ground support the named pose of a family is held on. Matched by
# substring of the family name (the clip stem minus its date and take). A
# family absent here falls back to "the longest hold".
EXPECTED_SUPPORT = [
    ("Crane_Crow", _hands_only),
    ("Firefly", _hands_only),
    ("Peacock_Pose", _hands_only),          # Mayurasana; not Feathered_Peacock
    ("Koundinya", _hands_only),
    ("Scale_Pose", _hands_only),
    ("Shoulder-Pressing", _hands_only),
    ("Eight-Angle", _hands_only),
    ("Cockerel", _hands_only),
    ("Handstand", _hands_only),
    ("Feathered_Peacock", _forearms_and_hands),
    ("Scorpion", lambda g: _forearms_and_hands(g) or _hands_only(g)),
    ("Supported_Headstand", _head_and_hands),
    ("Supported_Shoulderstand", _head_on_shoulders),
    ("Plow", _head_on_shoulders),
    ("Tree_Pose", _one_foot),
    ("Eagle_Pose", _one_foot),
    ("big_toe_hold", _one_foot),
    ("Lord_of_the_Dance", _one_foot),
    ("Warrior_III", _one_foot),
    ("Half_Moon", _one_foot),
    ("Standing_Split", _one_foot),
    ("Downward-Facing_Dog", lambda g: _both_feet(g) and HANDS <= g),
    ("Plank_Pose_or_Kumbhakasana", lambda g: HANDS <= g and (g & FEET)),
    ("Chaturanga", lambda g: HANDS <= g and (g & FEET)),
    ("Dolphin", lambda g: FOREARMS <= g and (g & FEET)),
    ("Uttanasana", _both_feet),
    ("Garland", _both_feet),
    ("Low_Lunge", _both_feet),
    ("Warrior_II_", _both_feet),
    ("Triangle", lambda g: (g & FEET)),
    ("Side_Angle", lambda g: (g & FEET)),
    ("Upward_Plank", lambda g: _both_feet(g) and HANDS <= g),
    ("Side_Plank", lambda g: (g & FEET) and (g & HANDS)),
]


def expected_support(family: str):
    for token, predicate in EXPECTED_SUPPORT:
        if token in family:
            return predicate
    return None


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without a corpus)
# --------------------------------------------------------------------------- #
def median_filter(x: np.ndarray, window: int) -> np.ndarray:
    """Centered running median with edge replication; ``window`` is made odd."""
    window = max(int(window), 1)
    if window % 2 == 0:
        window += 1
    if window == 1 or len(x) == 0:
        return np.asarray(x, dtype=np.float64).copy()
    half = window // 2
    padded = np.pad(np.asarray(x, dtype=np.float64), half, mode="edge")
    out = np.empty(len(x), dtype=np.float64)
    for i in range(len(x)):
        out[i] = np.median(padded[i : i + window])
    return out


def stillness_windows(
    speed: np.ndarray, v_still: float, min_frames: int, merge_frames: int
) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` frame windows where ``speed < v_still``.

    Gaps of fewer than ``merge_frames`` frames between consecutive windows are
    bridged first, then windows shorter than ``min_frames`` are dropped -- in
    that order, so two short still stretches separated by a blip can together
    make one hold.
    """
    still = np.asarray(speed) < v_still
    runs: list[list[int]] = []
    start = None
    for t, flag in enumerate(still):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            runs.append([start, t - 1])
            start = None
    if start is not None:
        runs.append([start, len(still) - 1])

    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 < merge_frames:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [(s, e) for s, e in merged if e - s + 1 >= min_frames]


def demote_supported_pairs(pairs: set[str]) -> set[str]:
    """Drop ``A+B`` when both ``A:G`` and ``B:G`` are present (load-path rule)."""
    grounded = {p[:-2] for p in pairs if p.endswith(":G")}
    kept = set()
    for pair in pairs:
        if "+" in pair and not pair.endswith(":G"):
            a, b = pair.split("+", 1)
            if a in grounded and b in grounded:
                continue
        kept.add(pair)
    return kept


def drop_standing_leg_flags(pairs: set[str], orientation: str) -> set[str]:
    """Remove leg-leg pairs when standing upright on both feet."""
    ground = {p for p in pairs if p.endswith(":G")}
    if ground != set(STANDING_PAIRS) or orientation != "upright":
        return set(pairs)
    kept = set()
    for pair in pairs:
        if "+" in pair and not pair.endswith(":G"):
            a, b = pair.split("+", 1)
            if a in LEG_ZONES and b in LEG_ZONES:
                continue
        kept.add(pair)
    return kept


def vote_pairs(
    active: dict, start: int, end: int, vote: float, orientation: str = ""
) -> list[str]:
    """Pairs active on at least ``vote`` of the frames in ``[start, end]``."""
    n = end - start + 1
    chosen = set()
    for pair, mask in active.items():
        if n > 0 and float(np.asarray(mask[start : end + 1]).sum()) / n >= vote:
            chosen.add(pair)
    chosen = demote_supported_pairs(chosen)
    chosen = drop_standing_leg_flags(chosen, orientation)
    return sorted(chosen, key=lambda p: (":G" not in p, p))


def split_by_config(
    window: tuple[int, int], config_ids: np.ndarray, min_frames: int
) -> list[tuple[int, int]]:
    """Cut ``window`` into maximal runs of one ``config_ids`` value.

    Runs shorter than ``min_frames`` are transitions between holds (or detector
    chatter) and are dropped rather than merged, so a hold never spans a
    support change.
    """
    start, end = window
    out = []
    run_start = start
    for t in range(start + 1, end + 2):
        if t > end or config_ids[t] != config_ids[run_start]:
            if t - run_start >= min_frames:
                out.append((run_start, t - 1))
            run_start = t
    return out


# The student's goal metric (score_probe_pose.py / ContactGraphControl
# ._goal_pose_error): the 6 conditionable bodies, pelvis-relative, rotated into
# the heading frame of the root. Body order is the .motion (COMMON) order.
GOAL_BODIES = ("Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head")
COMMON_BODY_ORDER = (
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle",
    "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder",
    "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Hand",
)


def goal_frame_pose(rigid_body_pos: np.ndarray, rigid_body_rot: np.ndarray, frame: int) -> np.ndarray:
    """``[6, 3]`` heading-normalised pelvis-relative goal-body positions."""
    pos = np.asarray(rigid_body_pos[frame])
    x, y, z, w = np.asarray(rigid_body_rot[frame][0])
    yaw = np.arctan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    ids = [COMMON_BODY_ORDER.index(b) for b in GOAL_BODIES]
    rel = pos[ids] - pos[0]
    xy = rel[:, :2] @ np.array([[c, -s_], [s_, c]]).T
    return np.concatenate([xy, rel[:, 2:3]], axis=1)


def pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=1).mean())


def name_holds(windows: list[dict], family: str) -> None:
    """Assign names in place; see the module docstring for the rule.

    Each window must carry ``pairs_ground``, ``orientation`` and
    ``rest_pose_m`` (distance of its exemplar to the nearer of the clip's first
    and last frames) and the caller's ``rest_pose_threshold_m``.
    """
    if not windows:
        return
    for hold in windows:
        standing = (
            set(hold["pairs_ground"]) == STANDING_PAIRS
            and hold["orientation"] == "upright"
            and hold["rest_pose_m"] <= hold["rest_pose_threshold_m"]
        )
        hold["name"] = "standing" if standing else None
    rest = [h for h in windows if h["name"] is None]
    if rest:
        def identity(hold):
            return (tuple(hold["pairs_ground"]), hold["orientation"])

        # Longest first, so every group's representative is its longest hold.
        groups: list[dict] = []
        for hold in sorted(rest, key=lambda h: -h["duration_s"]):
            for group in groups:
                if identity(group["rep"]) == identity(hold) and pose_distance(
                    group["rep"]["_pose"], hold["_pose"]
                ) <= hold["same_hold_m"]:
                    group["members"].append(hold)
                    break
            else:
                groups.append({"rep": hold, "members": [hold]})
        # The family name goes to the longest group whose support matches the
        # family's expected support; else to the longest group outright.
        predicate = expected_support(family)
        family_group = groups[0]
        if predicate is not None:
            for group in groups:
                ground = frozenset(p[:-2] for p in group["rep"]["pairs_ground"])
                if predicate(ground):
                    family_group = group
                    break
        ordered = sorted(groups, key=lambda g: min(m["frame_start"] for m in g["members"]))
        k = 1
        for group in ordered:
            if group is family_group:
                name = family
            else:
                name = f"{family}_h{k}"
                k += 1
            for member in group["members"]:
                member["name"] = name
    for hold in windows:
        if hold.get("auto_rest") and hold["name"] != "standing":
            hold["name"] = "rest_start" if hold["frame_start"] == 0 else "rest_end"
        hold["extend"] = hold["name"] == family


# --------------------------------------------------------------------------- #
def propose_clip(task: dict) -> dict:
    """Worker: one clip -> its proposed holds (plus a speed summary)."""
    path = Path(task["source"])
    thresholds = dict(DEFAULT_THRESHOLDS)
    resolved = compute_active_pairs(path, task["mjcf"], thresholds)
    if resolved is None:
        raise ValueError(f"{path} is not a motion dict")
    motion = resolved["motion"]
    fps = int(resolved["fps"])
    T = int(resolved["T"])

    vel = motion["rigid_body_vel"]
    speed_raw = vel.norm(dim=-1).mean(dim=-1).numpy()
    speed = median_filter(speed_raw, round(task["smooth_s"] * fps))
    windows = stillness_windows(
        speed,
        v_still=task["v_still"],
        min_frames=round(task["min_hold_s"] * fps),
        merge_frames=round(task["merge_s"] * fps),
    )
    pelvis_z = motion["rigid_body_pos"][:, 0, 2].numpy()
    bins = resolved["bins"]
    active = resolved["active"]

    # Per-frame support id: the ground pairs (cleaned by the detector's own
    # hysteresis and dwell) plus the orientation bin. Same id <=> same node
    # identity, which is what a hold may not change inside.
    ground_pairs = [p for p in active if p.endswith(":G")]
    ground_matrix = np.stack([np.asarray(active[p], dtype=np.int64) for p in ground_pairs], axis=1)
    weights = 1 << np.arange(len(ground_pairs), dtype=np.int64)
    config_ids = ground_matrix @ weights + (bins.astype(np.int64) << len(ground_pairs))

    min_frames = round(task["min_hold_s"] * fps)
    rest_a = goal_frame_pose(motion["rigid_body_pos"].numpy(), motion["rigid_body_rot"].numpy(), 0)
    rest_b = goal_frame_pose(motion["rigid_body_pos"].numpy(), motion["rigid_body_rot"].numpy(), T - 1)

    pos_np = motion["rigid_body_pos"].numpy()
    rot_np = motion["rigid_body_rot"].numpy()

    def make_hold(start: int, end: int, auto_rest: bool = False) -> dict:
        exemplar = start + int(np.argmin(speed[start : end + 1]))
        orientation = ORIENT_BINS[int(bins[exemplar])]
        pairs = vote_pairs(active, start, end, task["pair_vote"], orientation)
        pose = goal_frame_pose(pos_np, rot_np, exemplar)
        return {
            "name": None,
            "t_start": round(start / fps, 4),
            "t_end": round(end / fps, 4),
            "t_hold": round(exemplar / fps, 4),
            "duration_s": round((end - start + 1) / fps, 3),
            "frame_start": int(start),
            "frame_end": int(end),
            "frame_hold": int(exemplar),
            "pairs": pairs,
            "pairs_ground": [p for p in pairs if p.endswith(":G")],
            "orientation": orientation,
            "speed_at_hold": round(float(speed[exemplar]), 4),
            "pelvis_z": round(float(pelvis_z[exemplar]), 3),
            "rest_pose_m": round(min(pose_distance(pose, rest_a), pose_distance(pose, rest_b)), 3),
            "rest_pose_threshold_m": task["rest_pose_m"],
            "same_hold_m": task["same_hold_m"],
            "_pose": pose,
            "auto": True,
            "auto_rest": auto_rest,
        }

    holds = []
    for window in windows:
        for start, end in split_by_config(window, config_ids, min_frames):
            holds.append(make_hold(start, end))

    # Rest ends (docstring step 7).
    rest_lead = round(task["rest_lead_s"] * fps)
    first_covered = holds[0]["frame_start"] if holds else T
    if first_covered > rest_lead:
        run_end = 0
        while run_end + 1 < first_covered and config_ids[run_end + 1] == config_ids[0]:
            run_end += 1
        if run_end + 1 >= min_frames:
            holds.insert(0, make_hold(0, run_end, auto_rest=True))
    last_covered = holds[-1]["frame_end"] if holds else -1
    if last_covered < T - 1 - rest_lead:
        run_start = T - 1
        while run_start - 1 > last_covered and config_ids[run_start - 1] == config_ids[T - 1]:
            run_start -= 1
        if T - run_start >= min_frames:
            holds.append(make_hold(run_start, T - 1, auto_rest=True))

    name_holds(holds, task["family"])
    for hold in holds:
        hold.pop("rest_pose_threshold_m")
        hold.pop("same_hold_m")
        hold.pop("_pose")
    return {
        "stem": task["stem"],
        "group": task["group"],
        "family": task["family"],
        "source": str(path),
        "fps": fps,
        "num_frames": T,
        "length_s": round(T / fps, 3),
        "speed_p50": round(float(np.median(speed)), 4),
        "speed_p90": round(float(np.percentile(speed, 90)), 4),
        "still_fraction": round(float((speed < task["v_still"]).mean()), 3),
        "holds": holds,
    }


def write_review(clips: list[dict], path: Path, v_still: float) -> None:
    lines = [
        "# Proposed holds — review table",
        "",
        f"`v_still` = {v_still} m/s (mean body speed). Edit `holds.yaml`, not this file.",
        "",
    ]
    for clip in clips:
        lines.append(
            f"## {clip['stem']}  ({clip['group']}, {clip['length_s']} s, "
            f"still {clip['still_fraction']:.0%}, speed p50/p90 "
            f"{clip['speed_p50']}/{clip['speed_p90']})"
        )
        lines.append("")
        lines.append("| # | name | t_start | t_hold | t_end | dur | orient | pelvis z | v_hold | rest m | ground | body-body | extend |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, hold in enumerate(clip["holds"]):
            body_body = [p for p in hold["pairs"] if not p.endswith(":G")]
            flag = " (rest)" if hold.get("auto_rest") else ""
            lines.append(
                f"| {k} | {hold['name']}{flag} | {hold['t_start']:.2f} | {hold['t_hold']:.2f} | "
                f"{hold['t_end']:.2f} | {hold['duration_s']:.2f} | {hold['orientation']} | "
                f"{hold['pelvis_z']:.2f} | {hold['speed_at_hold']:.3f} | {hold['rest_pose_m']:.2f} | "
                f"{' '.join(hold['pairs_ground']) or 'NONE'} | {' '.join(body_body) or '–'} | {hold['extend']} |"
            )
        if not clip["holds"]:
            lines.append("| – | (no still window found — lower v_still or min_hold_s) | | | | | | | | | | | |")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", default="data/smpl/expert60/corpus.yaml")
    parser.add_argument("--mjcf", default=DEFAULT_MJCF)
    parser.add_argument("--out", default="data/smpl/expert60/holds.yaml")
    parser.add_argument("--review", default="data/smpl/expert60/review.md")
    parser.add_argument("--v-still", type=float, default=0.15, help="m/s, mean body speed")
    parser.add_argument("--min-hold-s", type=float, default=0.8)
    parser.add_argument("--merge-s", type=float, default=0.3)
    parser.add_argument("--smooth-s", type=float, default=0.25)
    parser.add_argument("--pair-vote", type=float, default=0.5)
    parser.add_argument(
        "--same-hold-m", type=float, default=0.25,
        help="Two holds of one clip with the same support are the same node when "
             "their exemplars are within this 6-body distance.",
    )
    parser.add_argument(
        "--rest-lead-s", type=float, default=1.0,
        help="Add a rest hold when no hold covers this much of the clip's start/end.",
    )
    parser.add_argument(
        "--rest-pose-m", type=float, default=0.25,
        help="A both-feet upright hold within this 6-body distance of the clip's "
             "first or last frame is the 'standing' rest pose.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="*", default=None, help="Clip stems to process (debug).")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = REPO_ROOT / corpus_path
    corpus = yaml.safe_load(open(corpus_path))
    mjcf = str(Path(args.mjcf) if Path(args.mjcf).is_absolute() else REPO_ROOT / args.mjcf)

    tasks = []
    for entry in corpus["motions"]:
        if args.only and entry["stem"] not in args.only:
            continue
        tasks.append(
            {
                **entry,
                "mjcf": mjcf,
                "v_still": args.v_still,
                "min_hold_s": args.min_hold_s,
                "merge_s": args.merge_s,
                "smooth_s": args.smooth_s,
                "pair_vote": args.pair_vote,
                "rest_pose_m": args.rest_pose_m,
                "rest_lead_s": args.rest_lead_s,
                "same_hold_m": args.same_hold_m,
            }
        )
    torch.set_num_threads(1)
    if args.workers > 1 and len(tasks) > 1:
        with Pool(args.workers) as pool:
            clips = pool.map(propose_clip, tasks)
    else:
        clips = [propose_clip(t) for t in tasks]

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "corpus": str(corpus_path),
        "mjcf": mjcf,
        "detector_thresholds": dict(DEFAULT_THRESHOLDS),
        "stillness": {
            "v_still": args.v_still,
            "min_hold_s": args.min_hold_s,
            "merge_s": args.merge_s,
            "smooth_s": args.smooth_s,
            "pair_vote": args.pair_vote,
            "rest_pose_m": args.rest_pose_m,
            "rest_lead_s": args.rest_lead_s,
            "same_hold_m": args.same_hold_m,
        },
        "clips": clips,
    }
    with open(out, "w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, width=120)
    review = Path(args.review)
    if not review.is_absolute():
        review = REPO_ROOT / review
    write_review(clips, review, args.v_still)

    num_holds = sum(len(c["holds"]) for c in clips)
    empty = [c["stem"] for c in clips if not c["holds"]]
    print(f"wrote {out}: {len(clips)} clips, {num_holds} holds; review at {review}")
    if empty:
        print(f"WARNING: {len(empty)} clips have no still window: {empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
