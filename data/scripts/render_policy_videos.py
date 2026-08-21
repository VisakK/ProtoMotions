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

How long a video is
-------------------
``Simulator.step`` calls ``render()`` exactly once, so one policy step is one
captured PNG and the mp4 is compiled at 30 fps -- i.e. the video runs in real
time and ``N`` policy steps give ``N * env.dt`` seconds.

The default is the **whole clip**.  ``--max-seconds`` used to default to 12 s,
which silently truncated every hard-pose clip (they run 16-59 s), so the video
stopped before the pose was even entered.  ``0`` means "no cap"; ``--min-seconds``
is the opposite guard, replaying a short clip from the top until the video is at
least that long.

Where a video *ends* is decided by the simulator rather than by a stopwatch:

* ``--stop-on-reset`` (default) ends at the clip-end reset, so nothing after the
  reference runs out is recorded;
* ``--fail-threshold`` replays the training-time ``tracking_error`` termination
  (max per-body error > 0.5 m) that inference strips out, keeps ``--tail-seconds``
  of the failure and then stops, tagging the file ``FELL@<t>s``.

Every file name carries the recorded duration and how the rollout ended, so a
short video is legible as a failure rather than as a truncation.

Usage::

    python data/scripts/render_policy_videos.py \
        --checkpoint results/smpl_yogi_hard29_pressure_ab_s1/last.ckpt \
        --simulator isaaclab \
        --motion-file data/smpl/yoga_yogi_hard29_pressure.pt \
        --overrides env.ref_respawn_offset=0.005 \
        --fail-threshold 0.5 \
        --out-dir output/renderings/hard29_pressure
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_setup import add_common_args  # noqa: E402  (no torch import yet)
from render_planning import (  # noqa: E402  (pure python, unit-tested on CPU)
    outcome_suffix,
    plan_rollout,
    select_motions,
)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--out-dir", type=str, default="output/renderings/policy_review")
parser.add_argument("--rank-csv", type=str, default=None,
                    help="eval_per_motion.py CSV; orders and labels the videos")
parser.add_argument("--worst", type=int, default=None,
                    help="render only the N worst clips from --rank-csv")
parser.add_argument("--motion-ids", type=int, nargs="*", default=None,
                    help="explicit motion ids (overrides --rank-csv selection)")
parser.add_argument("--motion-names", type=str, nargs="*", default=None,
                    help="case-insensitive substrings; keep only clips matching one "
                         "of them (applied after --rank-csv / --motion-ids)")
parser.add_argument("--max-seconds", type=float, default=0.0,
                    help="cap per-clip rollout length in seconds; 0 (default) = the "
                         "whole clip. The old 12 s default truncated every hard pose.")
parser.add_argument("--min-seconds", type=float, default=15.0,
                    help="floor on video length: a clip shorter than this is replayed "
                         "from the top until reached. 0 disables.")
parser.add_argument("--stop-on-reset", dest="stop_on_reset", action="store_true",
                    default=True,
                    help="end the recording when the simulator resets the env (default)")
parser.add_argument("--no-stop-on-reset", dest="stop_on_reset", action="store_false",
                    help="keep stepping through resets until the time cap")
parser.add_argument("--fail-threshold", type=float, default=0.5,
                    help="max per-body tracking error (m) that counts as a failure, "
                         "matching the training-time tracking_error termination that "
                         "inference strips out. 0 disables.")
parser.add_argument("--tail-seconds", type=float, default=1.0,
                    help="keep recording this long after a stop trigger, so the "
                         "failure itself is on screen")
parser.add_argument("--settle-steps", type=int, default=2,
                    help="steps to run after reset before recording starts")
parser.add_argument("--skip-existing", action="store_true",
                    help="skip clips that already have an mp4 in --out-dir (resume a "
                         "batch without re-rendering)")
parser.add_argument("--keep-frames", action="store_true",
                    help="keep the PNG frames next to each video")
parser.add_argument("--flush-seconds", type=float, default=2.0,
                    help="wait after stopping capture, before compiling, so the "
                         "async viewport writer can drain. At 1.0 s the last ~5 "
                         "frames of a clip were still queued and were lost.")
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

import logging  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from policy_setup import build, motion_names  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
_logger = logging.getLogger(__name__)


class _Log:
    """Progress goes through ``print``, because Isaac silences ``logging.info``.

    IsaacLab's app launcher reconfigures the root logger and raises its level, so
    every ``log.info`` in this script vanishes while ``log.warning`` survives --
    which on a 29-clip, 90-minute batch means the run looks identical whether it
    is rendering or wedged.  That cost real time once; ``record_contact_physics``
    already carries the same class for the same reason.
    """

    @staticmethod
    def info(fmt, *fmt_args) -> None:
        print("render_policy_videos: " + (fmt % fmt_args if fmt_args else fmt),
              flush=True)
        _logger.debug(fmt, *fmt_args)

    @staticmethod
    def warning(fmt, *fmt_args) -> None:
        print("render_policy_videos: WARNING " + (fmt % fmt_args if fmt_args else fmt),
              flush=True)
        _logger.debug(fmt, *fmt_args)


log = _Log()


def main() -> int:
    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]
    motion_lib, sim = built["motion_lib"], built["simulator"]

    if getattr(sim, "headless", True):
        raise SystemExit("simulator came up headless; cannot capture a viewport")

    names = motion_names(motion_lib)
    lengths = motion_lib.get_motion_length(None).detach().cpu()
    selection = select_motions(names, args.rank_csv, args.worst, args.motion_ids,
                               args.motion_names)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        done_names = {p.stem for p in out_dir.glob("*.mp4")}
        before = len(selection)
        selection = [
            e for e in selection
            if not any(names[e[0]] in stem for stem in done_names)
        ]
        if before != len(selection):
            log.info("--skip-existing: skipping %d already-rendered clips",
                     before - len(selection))
    log.info("Rendering %d/%d motions to %s", len(selection), len(names), out_dir)

    motion_manager = env.motion_manager
    agent.eval()
    written = []
    env_ids = torch.arange(env.num_envs, device=env.device)

    # One render() per policy step, and RecordingMixin hard-codes the mp4 at
    # 30 fps -- so the video only runs in real time while env.dt is 1/30.
    if abs(env.dt - 1.0 / 30.0) > 1e-6:
        log.warning(
            "env.dt is %.5f s but videos are compiled at a fixed 30 fps: playback "
            "will run at %.2fx real time and the durations in the file names are "
            "simulated seconds, not wall-clock.", env.dt, env.dt * 30.0
        )

    def max_tracking_error() -> float:
        """The training-time ``tracking_error`` termination, recomputed by hand.

        Inference freezes ``termination_components`` to ``{}``, so nothing stops
        a rollout when the policy loses the pose -- the video would just show it
        lying on the floor for the rest of the clip.  This is
        ``compute_tracking_error`` (protomotions/envs/terminations/tracking.py)
        evaluated on the same two quantities the trainer used: the simulator's
        body positions against the reference with the spawn offset applied.
        """
        ref = motion_lib.get_motion_state(motion_manager.motion_ids,
                                          motion_manager.motion_times)
        ref_pos = ref.rigid_body_pos
        ref_pos = ref_pos + env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_pos
        )
        cur = env.simulator.get_robot_state().rigid_body_pos
        return float((ref_pos - cur).pow(2).sum(-1).sqrt().max(-1)[0].max())

    for k, (mid, prefix, err) in enumerate(selection):
        name = names[mid]
        label = f"{prefix}_{name}" if err is None else f"{prefix}_gt{err:.3f}_{name}"
        clip_len = float(lengths[mid])
        steps, target, allow_replay = plan_rollout(
            clip_len, env.dt, args.max_seconds, args.min_seconds
        )
        tail_steps = max(int(round(args.tail_seconds / env.dt)), 0)

        log.info("[%d/%d] motion %d %s  (clip %.1fs -> record %.1fs / %d steps%s)",
                 k + 1, len(selection), mid, name, clip_len, target, steps,
                 ", replaying short clip" if allow_replay else "")

        # Point every env at this clip and restart it from the top.
        # `disable_motion_resample=True` is load-bearing: without it reset()
        # resamples and the rollout would be of some *other* clip while the file
        # keeps this clip's name. `sample_flat=True` matches MimicEvaluator, so
        # the video shows the same conditions the reported gt_error was measured
        # under.
        def restart():
            motion_manager.motion_ids[env_ids] = mid
            motion_manager.motion_times[env_ids] = 0.0
            fresh, _ = env.reset(env_ids, sample_flat=True,
                                 disable_motion_resample=True)
            return fresh

        obs = restart()
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

        end_reason, fail_time, replays = "full", None, 0
        stop_at = None
        recorded = 0
        for i in range(steps):
            model_outs = agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))
            obs, _, dones, _terminated, _ = env.step(actions)
            obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
            recorded = i + 1
            elapsed = recorded * env.dt

            # The failure the aggregate metrics hide. Recorded once, with a tail,
            # so the fall itself is in the video rather than just its aftermath.
            if args.fail_threshold > 0 and fail_time is None and stop_at is None:
                if max_tracking_error() > args.fail_threshold:
                    fail_time = elapsed
                    stop_at = i + tail_steps
                    log.info("    tracking error > %.2f m at %.1fs -> ending video",
                             args.fail_threshold, elapsed)

            if stop_at is not None:
                if i >= stop_at:
                    end_reason = "fell"
                    break
                continue

            # `env.step` never auto-resets: it reports `reset_buf` and leaves the
            # decision here. For a mimic env at inference the only thing that
            # raises it is the clip running out, so this is the natural cut --
            # and it also keeps `motion_times` from running past the reference.
            if bool(dones.any()):
                if allow_replay and elapsed + 1e-6 < args.min_seconds:
                    replays += 1
                    obs = restart()
                    obs_td = agent.obs_dict_to_tensordict(
                        agent.add_agent_info_to_obs(obs)
                    )
                    continue
                if args.stop_on_reset:
                    end_reason = "clipend"
                    break

        duration = recorded * env.dt
        if replays:
            log.info("    replayed the clip %dx to reach the %.0fs floor",
                     replays, args.min_seconds)

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

        # Count what actually landed before the compile deletes the folder. A
        # video that is a second short of the clip is the exact complaint this
        # script exists to answer, so it gets said out loud rather than being
        # absorbed into a rounded duration.
        if recording_dir and Path(recording_dir).is_dir():
            landed = len(list(Path(recording_dir).glob("*.png")))
            if landed < recorded:
                log.warning("    %d/%d frames landed (%.2fs lost to the async "
                            "viewport writer); raise --flush-seconds",
                            landed, recorded, (recorded - landed) * env.dt)
                duration = landed * env.dt

        # Named only now, so the duration in the file name is the duration of the
        # file rather than of the rollout that produced it.
        label += outcome_suffix(duration, end_reason, fail_time)

        # `render()` compiles the mp4 and then deletes the PNG folder in the same
        # call, so anything that wants the frames has to take them first --
        # checking for the folder afterwards silently keeps nothing.
        if args.keep_frames and recording_dir and Path(recording_dir).is_dir():
            shutil.copytree(recording_dir, out_dir / f"{label}_frames",
                            dirs_exist_ok=True)

        env.simulator.render()  # compiles the mp4, then clears the PNG folder

        mp4 = Path(f"{recording_dir}.mp4") if recording_dir else None
        if mp4 is not None and mp4.exists():
            final = out_dir / f"{label}.mp4"
            mp4.replace(final)
            for suffix in (".motion", ".markers.pt", ".objects.pt"):
                side = Path(f"{recording_dir}{suffix}")
                if side.exists():
                    side.replace(out_dir / f"{label}{suffix}")
            if Path(recording_dir).exists():
                shutil.rmtree(recording_dir, ignore_errors=True)
            written.append(final)
            log.info("    -> %s", final.name)
        else:
            log.warning("    no video produced for %s (expected %s)", name, mp4)

    # Final sweep: a frame that lands after its clip's cleanup recreates the
    # directory, so remove any recording folders left in the output dir. Matched
    # on RecordingMixin's own `-%Y-%m-%d-%H-%M-%S` suffix rather than on "any
    # directory that is not ours" -- --out-dir is a user-chosen path and an
    # unqualified rmtree of its subdirectories is a footgun.
    stamped = re.compile(r"-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")
    for leftover in out_dir.iterdir():
        if leftover.is_dir() and stamped.search(leftover.name):
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
