"""Keep lane T's CPU work out of a running GPU training job's way (PLAN.MD §2 Operations).

Measured 2026-10-03 beside G1 (``smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac``): its epochs take 10.0-10.1 s (10.7-10.8 s
every 10th, the checkpoint write); 16-20 unpinned MuJoCo rollout threads at normal priority pushed three epochs to
13.0-13.8 s (+30 %), and 14 threads at nice 19 on the E-cores plus three P-cores still raised the median to 10.9 s
(+8 %). The machine is an i9-12900KS: logical CPUs 0-15 are 8 hyper-threaded P-cores, 16-23 eight E-cores, and the
training job's feeding thread runs on a P-core at ~85 %. Measured against a window with this lane's jobs paused
(G1 at 10.12 s; its own time had drifted from 9.72 s for other reasons), the eight E-cores at nice 19 cost +3.5 % and
one 4-core E-core cluster +0.2 %. So every CPU-heavy CLI of this package calls ``be_polite()`` first: nice 19 and
affinity to **one 4-core E-core cluster** (``POLITE_CPUS``), which every thread started afterwards (MuJoCo's rollout
pool) inherits. ``epoch_seconds()`` reads the job's own per-epoch wall
time from its TensorBoard event file, to check the rule "if it rises by more than 5 %, cut threads".

CLI: ``PYTHONPATH=.:data/scripts python -m edge_synthesis.gpu_guard [--results results/<run>] [--last 30]``
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time
from pathlib import Path

from reference_curation import ids

POLITE_CPUS = tuple(range(16, 20))     # one 4-core E-core cluster (CPUs 16-19 share an L2)
POLITE_THREADS = len(POLITE_CPUS)
DEFAULT_RESULTS = ids.REPO / "results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac"


def be_polite(cpus=POLITE_CPUS, nice: int = 19) -> dict:
    """Lower this process's priority and pin it (and every thread it starts later) to ``cpus``."""
    try:
        os.nice(max(0, nice - os.nice(0)))
    except OSError:
        pass
    avail = os.sched_getaffinity(0)
    want = set(cpus) & avail if avail else set(cpus)
    if want:
        os.sched_setaffinity(0, want)
    return {"nice": os.nice(0), "cpus": sorted(os.sched_getaffinity(0))}


def epoch_seconds(results: Path = DEFAULT_RESULTS, last: int = 30) -> list[tuple[int, float, str]]:
    """``[(epoch, seconds, wall clock)]`` of the job's last ``last`` epochs (``times/last_epoch_seconds``)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(glob.glob(str(Path(results) / "lightning_logs/version_*/events.out.tfevents.*")))
    if not files:
        return []
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    if "times/last_epoch_seconds" not in ea.Tags()["scalars"]:
        return []
    ev = ea.Scalars("times/last_epoch_seconds")[-last:]
    return [(e.step, round(e.value, 2), time.strftime("%H:%M:%S", time.localtime(e.wall_time))) for e in ev]


def summary(rows) -> dict:
    """Median epoch time without the checkpoint epochs (every 10th), and the slowest of those."""
    plain = [s for e, s, _ in rows if e % 10]
    return {"epochs": [rows[0][0], rows[-1][0]] if rows else None,
            "median_s": round(statistics.median(plain), 2) if plain else None,
            "max_s": max(plain) if plain else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--last", type=int, default=30)
    args = ap.parse_args(argv)
    rows = epoch_seconds(args.results, args.last)
    for r in rows:
        print(*r)
    print(summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
