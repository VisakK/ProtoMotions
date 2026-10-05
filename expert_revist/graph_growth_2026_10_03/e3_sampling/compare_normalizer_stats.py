#!/usr/bin/env python3
"""Card E3: compare every observation normaliser's statistics between two checkpoints (CPU).

For the integration stage's freeze check: the warm-start source (e.g. G1's ``epoch_5000.ckpt``)
against the fine-tune's checkpoint after N epochs with ``--freeze-obs-normalizers True``. The
actor's and the critic's (``_actor.mu.norm``, ``_critic.norm``) must be bit-identical; an AMP
discriminator's and its critic's should have moved. Exit status 1 when a frozen one moved.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/e3_sampling/compare_normalizer_stats.py \
        results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/epoch_5000.ckpt results/<G3>/last.ckpt [--out x.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

FROZEN = ("_actor.mu.norm", "_critic.norm")
SUFFIX = ".running_obs_norm."


def statistics(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model = state["model"]
    return {k: v for k, v in model.items() if SUFFIX in k}, state.get("epoch")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    a, epoch_a = statistics(args.before)
    b, epoch_b = statistics(args.after)
    rows = {}
    for key in sorted(set(a) | set(b)):
        norm = key.split(SUFFIX)[0]
        if key not in a or key not in b:
            rows.setdefault(norm, {})[key.split(SUFFIX)[1]] = "missing in one checkpoint"
            continue
        same = torch.equal(a[key], b[key])
        diff = float((a[key].double() - b[key].double()).abs().max()) if a[key].numel() else 0.0
        rows.setdefault(norm, {})[key.split(SUFFIX)[1]] = {"bit_identical": same, "max_abs_diff": diff}
    frozen_ok = all(
        isinstance(v, dict) and v["bit_identical"]
        for norm in FROZEN if norm in rows for v in rows[norm].values()
    ) and all(norm in rows for norm in FROZEN)
    result = {"before": str(args.before), "before_epoch": epoch_a, "after": str(args.after),
              "after_epoch": epoch_b, "normalizers": rows, "frozen_bit_identical": frozen_ok}
    for norm, stats in rows.items():
        tag = "frozen" if norm in FROZEN else "free"
        print(f"{norm:32s} ({tag}): " + ", ".join(
            f"{k} {'identical' if isinstance(v, dict) and v['bit_identical'] else v if isinstance(v, str) else 'max |d| %.3g' % v['max_abs_diff']}"
            for k, v in stats.items()))
    print(f"actor and critic statistics bit-identical (epoch {epoch_a} -> {epoch_b}): {frozen_ok}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    sys.exit(0 if frozen_ok else 1)


if __name__ == "__main__":
    main()
