# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 FSQ-C student, round 8: give the intent code back its authority.

This is :mod:`contact_graph_fsq_v7` with the three changes of
``notes/Student_improvement_round7_1.MD`` §6.1-§6.3, which are one hypothesis
rather than three: **the code should be free to move the action at the moment it
is issued, the goal should carry the pose, and the trunk should know where it is
inside a chunk.** Everything else -- corpus, graph, experts, routing, DAgger,
chunk length, vocabulary, sampling, learning rates, fixed sampling -- is held at
v7/v7_1, so v7_1 stays the clean baseline.

**The measurement this executes.** Round 7_1 §2.1 normalised round 7's
code-influence proxy by how much the codes actually disagree:

    code gain = (prior_mse / privileged_mse - 1) / latent_residual_l2

    v6    0.378  0.346  0.326  0.327  0.321  0.331   (e2k .. e15.4k)
    v7    0.071  0.089  0.067  0.058
    v7_1  0.081  0.077  0.069  0.080  0.061

Flat within every run, and v7 and v7_1 are indistinguishable at ~5x below v6 --
so the goal-conditioned trunk is not the cause, and the two anti-jerk changes
are what is left. Both act at the refresh and almost nowhere else, which is the
same instant a commitment happens: **suppressing v6's 3.75 Hz chunk-clock
artifact and suppressing commitment were the same operation.** They are
separable in time, and that is what §6.1 does.

**1. Selective smoothing (§6.1).** ``--latent-ema-alpha 1.0`` restores v6's hard
substitution -- the ramp's only action was to delay a newly committed code by
~2-3 steps, and at alpha 0.5 it also took the trunk's input *off the FSQ
lattice* for roughly the first three steps of every chunk, dissolving the
discreteness the head exists for exactly where a decision is made.
``--action-rate-free-steps 1`` keeps the teacher-rate loss (v7's measured, large
anti-jerk win: standing-hold joint-accel RMS 12.47 -> 0.56) but stops charging
it on the **refresh row**, the only row whose action increment spans a code
change. Everywhere else the term still charges jitter, which is what it is for.

**2. Chunk phase into the trunk (§6.2).** ``--chunk-phase-to-trunk`` feeds a
one-hot of steps-since-refresh (width ``chunk_steps``). Without it the trunk has
no clock, so a held code can only mean "shift the action by a constant" -- never
"over the next 0.27 s, do this". Note this is the *opposite* call from v7 §1.3,
which correctly declined to give the **AR head** a phase input: the head is only
ever queried at a refresh, where the phase is constant, while the trunk runs at
every phase.

**3. The trunk reads the pose half of the goal only (§6.3).** ``--trunk-goal
pose``. Measured over every training frame, the goal's support set already
equals the current one on **72 %** of them and only **13.8 %** ask the body to
unload a grounded zone -- so the contact half is redundant with
``contact_state_obs`` on most rows and decision-bearing on the rest, which is
the intent code's one remaining job. The pose half is not: it is what fixed the
Warrior III substitution in v7 (round 7 §10.2, 0.153 m with
``score_probe_pose`` naming Warrior III itself, better than v6 at its own
optimum). ``full`` reproduces v7, ``none`` reproduces v6/v7_1.

**Deliberately NOT in this round**, so the attribution stays clean -- they are
§6.4 and the v8_1/v8_2 arms: the encoder freeze, the posterior/prior alignment
term, the DAgger loss on the sampled action, and the round-7_1 rebuilt graph
(``yoga_contact_graph_student44h_bbdemoted``). This run uses the **old** graph
so the training changes are the only variable against v7_1.

**Guard.** ``model/fsq_code_perplexity`` remains the nominal guard but has the
blind spot round 7 §5 recorded (it measures the encoder, not the decoder). Read
**code gain** instead -- both terms are already logged under
``env/action_gap/`` -- and expect it to climb back toward v6's ~0.33 while
``eval/action_rate_mean_rad_s`` stays near v7's level. If the gain returns and
the jerk returns with it, §6.1 has failed and the two are genuinely inseparable.

Train::

    bash data/scripts/run_student_distill_v8.sh
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.supervised.multi_expert import (
    MultiExpertMaskedMimicAgentConfig,
)


def _load_base_module():
    """Load the v7 experiment by file path (same pattern v7 uses for v6)."""
    base_path = Path(__file__).resolve().parent / "contact_graph_fsq_v7.py"
    spec = importlib.util.spec_from_file_location("contact_graph_fsq_v7_base", base_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

NUM_GOAL_STEPS = base.NUM_GOAL_STEPS
STATE_KEYS = base.STATE_KEYS
UNNORMALIZED_STATE_KEYS = base.UNNORMALIZED_STATE_KEYS
CONTACT_EVENT_FLAG_KEY = base.CONTACT_EVENT_FLAG_KEY
ENCODER_FUTURE_STEPS = base.ENCODER_FUTURE_STEPS
GOAL_TRUNK_KEYS = base.GOAL_TRUNK_KEYS
TRUNK_GOAL_POSES_NORM = base.TRUNK_GOAL_POSES_NORM
TRUNK_GOAL_TIMES_NORM = base.TRUNK_GOAL_TIMES_NORM

# The pose half of the goal: where to be and when, without the contact set that
# says which supports to be on. §6.3.
POSE_GOAL_TRUNK_KEYS = [key for key in GOAL_TRUNK_KEYS if key != "contact_goal_obs"]
TRUNK_GOAL_CHOICES = {
    "none": None,
    "pose": POSE_GOAL_TRUNK_KEYS,
    "full": GOAL_TRUNK_KEYS,
}


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Everything v7 takes, plus the round-8 knobs."""
    base.additional_experiment_arguments(parser)

    def _flag(value) -> bool:
        return str(value).lower() not in ("0", "false", "no")

    parser.add_argument(
        "--trunk-goal", choices=sorted(TRUNK_GOAL_CHOICES), default="pose",
        help="Which half of the goal the action trunk reads. 'pose' (round 8) "
             "gives it the target frame, masks and deadlines but NOT "
             "contact_goal_obs, so the support decision stays the code's job. "
             "'full' reproduces v7, 'none' reproduces v6/v7_1. Supersedes "
             "--goal-conditioned-trunk, which is ignored here.",
    )
    parser.add_argument(
        "--chunk-phase-to-trunk", type=_flag, default=True,
        help="Feed the trunk a one-hot of steps-since-refresh, so a held code "
             "can express a short program rather than a constant offset. "
             "False reproduces v6/v7.",
    )
    parser.add_argument(
        "--action-rate-free-steps", type=int, default=1,
        help="Steps at the start of each chunk on which the teacher-rate term "
             "is not charged. 1 exempts the refresh row -- the only row whose "
             "action increment spans a code change. 0 reproduces v7.",
    )
    # Round-8 values for knobs v7 already owns, so `--help` shows what this
    # round runs at rather than what v7 did.
    parser.set_defaults(latent_ema_alpha=1.0)


terrain_config = base.terrain_config
scene_lib_config = base.scene_lib_config
motion_lib_config = base.motion_lib_config
configure_robot_and_simulator = base.configure_robot_and_simulator
apply_inference_overrides = base.apply_inference_overrides
env_config = base.env_config


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> MultiExpertMaskedMimicAgentConfig:
    """The v7 agent with the round-8 changes applied on top.

    Built by mutating v7's configuration rather than restating it, so anything
    v7 changes later is inherited and the diff between the rounds stays exactly
    the list in this module's docstring.
    """
    # v7 applies its own full-goal trunk conditioning when the flag is on; turn
    # it off there and apply the requested subset here, so exactly one of the
    # two runs and the resulting trunk config has no duplicated normalizers.
    forwarded = argparse.Namespace(**vars(args))
    forwarded.goal_conditioned_trunk = False
    cfg = base.agent_config(robot_config, env_config, forwarded)

    trunk_goal = getattr(args, "trunk_goal", "pose")
    if trunk_goal not in TRUNK_GOAL_CHOICES:
        raise ValueError(
            f"--trunk-goal must be one of {sorted(TRUNK_GOAL_CHOICES)}, got {trunk_goal!r}"
        )
    keys = TRUNK_GOAL_CHOICES[trunk_goal]
    if keys is not None:
        base._goal_condition_trunk(cfg.model.trunk, keys=keys)

    cfg.model.fsq.chunk_phase_to_trunk = bool(
        getattr(args, "chunk_phase_to_trunk", True)
    )
    if cfg.model.fsq.chunk_phase_to_trunk:
        _add_chunk_phase_to_trunk(cfg.model.trunk)

    cfg.action_rate_free_steps = int(getattr(args, "action_rate_free_steps", 1))
    return cfg


def _add_chunk_phase_to_trunk(trunk_config) -> None:
    """Declare the model-produced chunk-phase one-hot as a trunk input.

    Unnormalised and appended raw: it is already a one-hot, and a running
    normaliser would divide the rarely-visited phases by a near-zero standard
    deviation -- the same reason ``contact_goal_obs`` is never normalised.
    ``FSQMaskedMimicModel`` writes the tensor inside ``forward`` and keeps the
    key out of ``in_keys``, so the environment is never asked for it.
    """
    from protomotions.agents.supervised.fsq_masked_mimic_model import (
        FSQ_CHUNK_PHASE_KEY,
    )

    trunk_config.in_keys = list(
        dict.fromkeys(trunk_config.in_keys + [FSQ_CHUNK_PHASE_KEY])
    )
    _, head = base._trunk_head(trunk_config)
    head.in_keys = list(dict.fromkeys(head.in_keys + [FSQ_CHUNK_PHASE_KEY]))
