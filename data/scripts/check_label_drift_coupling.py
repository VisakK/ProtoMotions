# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Is the imitation error larger exactly where the label depends on something the student cannot see?

``check_student_xy_invariance.py`` §17.3 measures the *sensitivity*: with the
student's inputs made floor-free and the frozen teachers left world-anchored, the
expert action moves by ~10 % of its own spread per 10 cm of robot-vs-reference
drift.  Whether that matters during training is a different question, and the
aggregate ``supervised/loss`` is a poor instrument for it: the implied noise floor
is order 1e-5-1e-4 against a loss of 1.4e-3 at epoch 300, so a short run cannot
resolve it in the mean.

It *is* resolvable in the conditional.  If the label carries a floor dependence
the student cannot observe, the imitation error must be systematically larger on
**high-drift frames**, because those are exactly the frames where the unobserved
quantity is large.  A student whose inputs still contain the drift (original v9)
can explain it and should show no such slope.

So this rolls a trained checkpoint forward under its own privileged action -- what
75 % of training environments do -- and records, per frame:

* ``drift``  = ``|ref_root.xy - cur_root.xy|``, the quantity re-anchoring removes;
* ``err``    = the per-row imitation MSE, ``mean_j (privileged_action - expert_action)^2``,
  i.e. exactly the quantity ``supervised/loss`` averages.

and reports ``err`` binned by ``drift`` plus the rank correlation.  The comparison
that carries the result is **arm A (inputs see the drift) against arm B (inputs
floor-free, labels not)**; under re-anchoring drift is identically zero, so arm C
is evidence by construction rather than by regression.

Usage::

    PYTHONPATH=. python data/scripts/check_label_drift_coupling.py \\
      --checkpoint results/smoke_xy_B_on_norea/last.ckpt --headless \\
      --num-envs 512 --seconds 6 --out-dir output/label_drift
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_setup import add_common_args  # noqa: E402  (no torch import yet)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--out-dir", type=str, default=None)
parser.add_argument("--seconds", type=float, default=6.0)
parser.add_argument("--warmup", type=float, default=1.0,
                    help="seconds discarded so drift is not dominated by the reset")
parser.add_argument("--bins", type=float, nargs="+",
                    default=[0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 10.0])
parser.add_argument("--label", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

args.headless = True
args.resolved_configs = "resolved_configs.pt"

from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

import logging  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from policy_setup import build  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("check_label_drift_coupling")


def main() -> int:
    torch.manual_seed(args.seed)
    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]
    if not getattr(agent, "expert_actors", None):
        raise SystemExit("no frozen experts; this needs resolved_configs.pt")

    device = env.device
    env_ids = torch.arange(env.num_envs, device=device)
    steps = int(round(args.seconds / float(env.dt)))
    skip = int(round(args.warmup / float(env.dt)))

    flags = {}
    for key in ("masked_mimic_target_poses", "mimic_target_poses"):
        component = env.config.observation_components.get(key)
        if component is not None:
            flags[key] = bool(component.static_params.get("root_relative_xy", False))
    realign = bool(
        env.motion_manager.config.realign_motion_with_humanoid_on_each_step
    )
    log.info("root_relative_xy=%s  realign=%s", flags, realign)

    agent.eval()
    env.reset(env_ids, sample_flat=True)
    agent.pre_collect_step(0)
    obs_td = agent.obs_dict_to_tensordict(
        agent.add_agent_info_to_obs(env.get_obs())
    )

    drift, err = [], []
    with torch.no_grad():
        for step in range(steps):
            expert = agent._collect_external_expert_action(obs_td)
            out = agent.model(obs_td)
            action = out["privileged_action"]
            row_err = (action - expert).square().mean(dim=-1)

            ctx = env._current_context
            ref = ctx.mimic.ref_state.rigid_body_pos
            cur = ctx.current.rigid_body_pos
            row_drift = (ref[:, 0, :2] - cur[:, 0, :2]).norm(dim=-1)
            if step >= skip:
                drift.append(row_drift.cpu().numpy())
                err.append(row_err.cpu().numpy())

            obs, *_ = env.step(action)
            agent.pre_collect_step(step + 1)
            obs_td = agent.obs_dict_to_tensordict(
                agent.add_agent_info_to_obs(obs)
            )

    drift = np.concatenate(drift)
    err = np.concatenate(err)
    from scipy.stats import spearmanr

    rho, p = spearmanr(drift, err)
    name = args.label or Path(args.checkpoint).parent.name
    print(f"\n=== {name} ===")
    print(f"root_relative_xy={flags}  realign={realign}")
    print(f"{len(err)} (frame, env) rows; overall imitation MSE {err.mean():.3e}")
    print(f"drift: p50 {np.median(drift):.4f}  p90 {np.percentile(drift, 90):.4f}  "
          f"max {drift.max():.4f} m")
    print(f"\n{'drift bin (m)':<18}{'rows':>9}{'share':>8}{'imitation MSE':>16}"
          f"{'vs bin 0':>10}")
    rows = []
    base = None
    for lo, hi in zip(args.bins[:-1], args.bins[1:]):
        mask = (drift >= lo) & (drift < hi)
        if mask.sum() == 0:
            continue
        value = float(err[mask].mean())
        if base is None:
            base = value
        rows.append(dict(lo=lo, hi=hi, n=int(mask.sum()),
                         share=float(mask.mean()), mse=value, ratio=value / base))
        print(f"[{lo:.2f}, {hi:.2f})".ljust(18)
              + f"{int(mask.sum()):>9}{mask.mean():>8.3f}{value:>16.3e}"
              + f"{value / base:>10.2f}x")
    print(f"\nSpearman(drift, imitation MSE) = {rho:+.4f}   p = {p:.2e}")
    print("A positive slope on an XY-free student is label noise it cannot explain; "
          "on a world-anchored one it is a state it can.")

    report = dict(name=name, checkpoint=args.checkpoint, flags=flags,
                  realign=realign, rows=int(len(err)),
                  overall_mse=float(err.mean()), spearman_rho=float(rho),
                  spearman_p=float(p), drift_p50=float(np.median(drift)),
                  drift_p90=float(np.percentile(drift, 90)), bins=rows)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(report, indent=1))
        print(f"\nwrote {out / (name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
