# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-check the online body-body contact channel against the offline recorder.

Two independent paths now report the same physical quantity:

* **offline** -- ``PairContactView``, one PhysX ``RigidContactView`` over every
  body filtered against every other body. This is what
  ``record_contact_graph_rollouts.py`` used to build the contact graph, so it is
  the definition the goals are written in. It only works at ``num_envs == 1``.
* **online** -- ``RobotConfig.contact_pair_bodies``, which adds one filter column
  per body to each per-body ``ContactSensor``. This one scales to 1024
  environments and is what the student actually observes.

They read different PhysX views with different filter layouts, so agreeing is
evidence and disagreeing localises the bug. The failure this is written to catch
is a column-order mistake: reading the pair matrix one filter out is silent, and
would attribute the crow shin-on-upper-arm force to some other pair entirely --
the same class of error as the simulator-vs-common body-order trap recorded in
``notes/Pressure_supervision_design.MD`` §4.

Usage::

    DISPLAY= PYTHONPATH=. python data/scripts/validate_body_pair_contacts.py \\
      --checkpoint results/smpl_yogi_hard29_pressure_ab_s1/last.ckpt \\
      --clip Koundinyanasana --steps 400
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_setup import add_common_args  # noqa: E402  (no torch import yet)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--clip", type=str, required=True,
                    help="case-insensitive substring of the clip to roll out")
parser.add_argument("--start-time", type=float, default=0.0)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--tolerance-n", type=float, default=1.0,
                    help="newtons of disagreement tolerated per pair per frame")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# PhysX filter patterns in a single RigidContactView only resolve one-to-one for
# a single environment, which is the whole reason the online path exists.
args.num_envs = 1
args.headless = True
# The checkpoint under test was almost certainly trained without pair sensing;
# turn it on for this process.
args.overrides = list(args.overrides or []) + ["robot.contact_pair_bodies=all"]

from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pair_contact_view import PairContactView  # noqa: E402
from policy_setup import build, motion_names  # noqa: E402


def log(fmt, *fmt_args) -> None:
    print("validate_body_pair_contacts: " + (fmt % fmt_args if fmt_args else fmt),
          flush=True)


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    built = build(args, AppLauncher)
    env, agent = built["env"], built["agent"]
    motion_lib, simulator = built["motion_lib"], built["simulator"]
    robot_config = env.robot_config

    pair_bodies = list(robot_config.contact_pair_bodies or [])
    if not pair_bodies:
        raise SystemExit(
            "contact_pair_bodies is empty: the override did not take. Body-body "
            "contact is not being sensed, so there is nothing to validate."
        )
    log("online path senses %d pair bodies", len(pair_bodies))

    sim_body_names = list(simulator._robot.body_names)
    sensor = next(iter(simulator._contact_sensor_map.values()))
    view = PairContactView(
        sensor._physics_sim_view,
        # glob, not the regex the ContactSensorCfg takes
        body_root_glob=robot_config.asset.usd_bodies_root_prim_path.replace(".*", "*"),
        body_names=sim_body_names,
        num_envs=1,
        max_points_per_pair=1,
    )
    log("offline view: %s", view.describe().splitlines()[0])

    names = motion_names(motion_lib)
    matches = [i for i, n in enumerate(names) if args.clip.lower() in n.lower()]
    if len(matches) != 1:
        raise SystemExit(
            f"--clip '{args.clip}' matched {len(matches)} clips: "
            f"{[names[i] for i in matches[:5]]}"
        )
    motion_id = matches[0]
    log("clip: %s", names[motion_id])

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.motion_manager.motion_ids[env_ids] = motion_id
    env.motion_manager.motion_times[env_ids] = args.start_time
    obs, _ = env.reset(env_ids, sample_flat=True, disable_motion_resample=True)
    agent.eval()
    agent.pre_collect_step(0)
    obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

    # Body order differs between the two paths: the online channel is converted
    # to COMMON order by the env, the offline view is in SIMULATOR order. Remap
    # by NAME rather than assuming they agree -- they do not.
    common_names = list(robot_config.kinematic_info.body_names)
    sim_to_common = torch.tensor(
        [common_names.index(n) for n in sim_body_names], dtype=torch.long
    )
    pair_of_common = torch.tensor(
        [pair_bodies.index(n) if n in pair_bodies else -1 for n in common_names],
        dtype=torch.long,
    )

    worst = 0.0
    worst_where = None
    compared = 0
    nonzero_pairs = 0
    dt_phys = float(simulator._sim.get_physics_dt())

    for step in range(args.steps):
        with torch.no_grad():
            # PPO experts expose plain __call__; MaskedMimic students also have
            # forward_inference, which is the deployable prior-only path.
            forward = getattr(agent.model, "forward_inference", agent.model)
            outputs = forward(obs_td)
        action = outputs.get("mean_action", outputs.get("action"))
        obs, _, dones, _, _ = env.step(action)
        obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

        # Offline: rows are (env, body) in PhysX's own order, which is NOT
        # promised to be env-major, so scatter by the view's resolved map exactly
        # as the graph recorder does. Filter 0 is the ground.
        raw = view.view.get_contact_force_matrix(dt=dt_phys)
        offline = torch.zeros(
            len(sim_body_names), view.view.filter_count, 3, device=raw.device
        )
        offline[view.row_to_slot.to(raw.device)] = raw.reshape(
            -1, view.view.filter_count, 3
        )
        offline_bb = offline[:, 1:, :].norm(dim=-1).cpu()  # [B_sim, B_sim]

        state = simulator.get_robot_state()
        online = state.rigid_body_pair_contact_forces
        if online is None:
            raise SystemExit(
                "rigid_body_pair_contact_forces is None with contact_pair_bodies "
                "set -- the simulator is not filling the pair columns."
            )
        online_bb = online[0].norm(dim=-1).cpu()  # [B_common, P]

        for si, sj in zip(*(offline_bb > 1e-6).nonzero(as_tuple=True)):
            ci, cj = int(sim_to_common[si]), int(sim_to_common[sj])
            slot = int(pair_of_common[cj])
            if slot < 0:
                continue
            a = float(offline_bb[si, sj])
            b = float(online_bb[ci, slot])
            compared += 1
            nonzero_pairs += 1
            if abs(a - b) > worst:
                worst = abs(a - b)
                worst_where = (step, common_names[ci], common_names[cj], a, b)

        if bool(dones.any()):
            log("clip ended at step %d", step)
            break

    log("compared %d non-zero body-body entries over %d steps", compared, step + 1)
    if compared == 0:
        log("NO body-body contact occurred; this clip cannot validate the channel")
        return 2
    log("worst disagreement: %.4f N", worst)
    if worst_where:
        s, bi, bj, a, b = worst_where
        log("  at step %d, %s <- %s: offline %.3f N vs online %.3f N", s, bi, bj, a, b)
    if worst > args.tolerance_n:
        log("FAIL: the two paths disagree by more than %.3f N", args.tolerance_n)
        return 1
    log("PASS: the online per-sensor pair columns match the offline RigidContactView")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
