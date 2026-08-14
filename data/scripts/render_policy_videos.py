# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render one MP4 per motion so a policy can be reviewed by eye, in bulk.

The viewer's ``L`` key already records video -- ``RecordingMixin`` captures the
viewport each step and compiles it with moviepy.  This drives that same
machinery from a script, one clip at a time, so instead of scrubbing 128 poses
by hand you get a directory of videos named by rank and tracking error.

Pair it with ``eval_per_motion.py``: pass its CSV to ``--rank-csv`` and the file
names are prefixed with the ranking, so the worst clips sort to the top of the
folder.  ``--worst N`` renders only those.

**Must run with a viewer** (no ``--headless``).  ``RecordingMixin.render()`` is
gated on ``not headless`` and IsaacLab's frame grab reads the *active viewport*,
which does not exist without a window -- headless would silently produce empty
videos.  On this box that means a display (``DISPLAY=:0``); over SSH use
X-forwarding or Isaac's livestream.

Usage::

    python data/scripts/render_policy_videos.py \
        --checkpoint results/smpl_yogi_easy128_contact_rich/inspect_snapshot.ckpt \
        --simulator isaaclab \
        --overrides env.ref_respawn_offset=0.005 \
        --rank-csv results/smpl_yogi_easy128_contact_rich/per_motion_eval.csv \
        --worst 20 \
        --out-dir output/renderings/easy128_review
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_setup import add_common_args  # noqa: E402  (no torch import yet)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--out-dir", type=str, default="output/renderings/policy_review")
parser.add_argument("--rank-csv", type=str, default=None,
                    help="eval_per_motion.py CSV; orders and labels the videos")
parser.add_argument("--worst", type=int, default=None,
                    help="render only the N worst clips from --rank-csv")
parser.add_argument("--motion-ids", type=int, nargs="*", default=None,
                    help="explicit motion ids (overrides --rank-csv selection)")
parser.add_argument("--max-seconds", type=float, default=12.0,
                    help="cap per-clip rollout length; 0 = full clip")
parser.add_argument("--settle-steps", type=int, default=2,
                    help="steps to run after reset before recording starts")
parser.add_argument("--keep-frames", action="store_true",
                    help="keep the PNG frames next to each video")
parser.add_argument("--flush-seconds", type=float, default=1.0,
                    help="wait after stopping capture, before compiling, so the "
                         "async viewport writer can drain")
args = parser.parse_args()

if args.headless:
    raise SystemExit(
        "render_policy_videos.py needs a viewer: RecordingMixin.render() is gated on "
        "`not headless` and IsaacLab captures the active viewport, so --headless would "
        "produce empty videos. Drop --headless (needs a display)."
    )
# One env: the recorder captures a single viewport and serialises one env's
# motion, and extra envs would only add physics cost per rendered frame.
args.num_envs = 1

from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

import csv  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from policy_setup import build, motion_names  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


def select_motions(names, rank_csv, worst, explicit):
    """-> list of (motion_id, label_prefix, gt_error or None), in render order."""
    if explicit:
        return [(i, f"{k:03d}", None) for k, i in enumerate(explicit)]
    if rank_csv:
        with open(rank_csv) as f:
            rows = list(csv.DictReader(f))
        # eval_per_motion.py writes worst-first, so file order is the ranking.
        sel = []
        for rank, r in enumerate(rows, start=1):
            mid = int(r["motion_id"])
            err = float(r.get("gt_error_mean", "nan"))
            failed = r.get("failed", "False") == "True"
            sel.append((mid, f"{rank:03d}{'_FAIL' if failed else ''}", err))
        return sel[:worst] if worst else sel
    return [(i, f"{i:03d}", None) for i in range(len(names))]


def main() -> int:
    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]
    motion_lib, sim = built["motion_lib"], built["simulator"]

    if getattr(sim, "headless", True):
        raise SystemExit("simulator came up headless; cannot capture a viewport")

    names = motion_names(motion_lib)
    lengths = motion_lib.get_motion_length(None).detach().cpu()
    selection = select_motions(names, args.rank_csv, args.worst, args.motion_ids)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Rendering %d/%d motions to %s", len(selection), len(names), out_dir)

    motion_manager = env.motion_manager
    agent.eval()
    written = []

    for k, (mid, prefix, err) in enumerate(selection):
        name = names[mid]
        label = f"{prefix}_{name}" if err is None else f"{prefix}_gt{err:.3f}_{name}"
        clip_len = float(lengths[mid])
        horizon = clip_len if args.max_seconds <= 0 else min(clip_len, args.max_seconds)
        steps = max(int(horizon / env.dt), 1)

        log.info("[%d/%d] motion %d %s  (%.1fs -> %d steps)",
                 k + 1, len(selection), mid, name, horizon, steps)

        # Point every env at this clip and restart it from the top.
        # `disable_motion_resample=True` is load-bearing: without it reset()
        # resamples and the rollout would be of some *other* clip while the file
        # keeps this clip's name. `sample_flat=True` matches MimicEvaluator, so
        # the video shows the same conditions the reported gt_error was measured
        # under.
        env_ids = torch.arange(env.num_envs, device=env.device)
        motion_manager.motion_ids[env_ids] = mid
        motion_manager.motion_times[env_ids] = 0.0
        obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)
        agent.pre_collect_step(0)
        obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

        # Let the reset transient pass before the recorder starts, otherwise
        # every video opens on the spawn pop.
        for _ in range(max(args.settle_steps, 0)):
            model_outs = agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))
            obs, _, _, _, _ = env.step(actions)
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

        # RecordingMixin names the folder `<path> % <timestamp>`; steer it per clip.
        sim._camera_target["env"] = 0
        sim._user_recording_video_path = str(out_dir / f"{label}-%s")
        sim._toggle_video_record()  # start (takes effect on the next render())

        for _ in range(steps):
            model_outs = agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))
            obs, _, _, _, _ = env.step(actions)
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

        recording_dir = getattr(sim, "_curr_user_recording_name", None)

        # Omniverse's capture_viewport_to_file is asynchronous, so the last
        # frames are still queued when the step loop ends. Drain *after*
        # flipping the recording flag off, not before: `_toggle_video_record`
        # clears `_user_is_recording` immediately while deferring the compile to
        # the next render(), so this window lets pending writes land without
        # queueing new ones. (Draining with extra render() calls while still
        # recording is self-defeating -- each one queues another frame, which is
        # why an earlier version left ~7 stray PNGs per clip.)
        sim._toggle_video_record()  # stop requested; capture stops now
        time.sleep(args.flush_seconds)
        env.simulator.render()  # compiles the mp4, then clears the PNG folder

        mp4 = Path(f"{recording_dir}.mp4") if recording_dir else None
        if mp4 is not None and mp4.exists():
            final = out_dir / f"{label}.mp4"
            mp4.replace(final)
            for suffix in (".motion", ".markers.pt", ".objects.pt"):
                side = Path(f"{recording_dir}{suffix}")
                if side.exists():
                    side.replace(out_dir / f"{label}{suffix}")
            if args.keep_frames and Path(recording_dir).exists():
                Path(recording_dir).replace(out_dir / f"{label}_frames")
            elif Path(recording_dir).exists():
                shutil.rmtree(recording_dir, ignore_errors=True)
            written.append(final)
            log.info("    -> %s", final.name)
        else:
            log.warning("    no video produced for %s (expected %s)", name, mp4)

    # Final sweep: a frame that lands after its clip's cleanup recreates the
    # directory, so remove any recording folders left in the output dir.
    if not args.keep_frames:
        for leftover in out_dir.iterdir():
            if leftover.is_dir() and not leftover.name.endswith("_frames"):
                shutil.rmtree(leftover, ignore_errors=True)

    if hasattr(env.simulator, "shutdown"):
        env.simulator.shutdown()

    print("\n" + "=" * 70)
    print(f"wrote {len(written)}/{len(selection)} videos to {out_dir}")
    for p in written[:15]:
        print(f"  {p.name}")
    if len(written) > 15:
        print(f"  ... {len(written) - 15} more")
    print("=" * 70)
    return 0 if written else 1


if __name__ == "__main__":
    with torch.no_grad():
        raise SystemExit(main())
