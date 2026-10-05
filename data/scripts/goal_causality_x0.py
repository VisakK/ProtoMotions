# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S0 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``: the X0 goal-causality falsifier and the fork battery.

**The question.** Does a frozen expert's action depend on its *goal* when its *state* is held fixed? A goal-blind
tracker (every expert before Design B) cannot be distilled into a goal-conditioned student: its labels carry no goal
at any horizon (``notes/V9_crucial_investigations/v11_and_other_failures.md`` §8). X0 measures it directly, with the
same frozen expert at a byte-identical robot state, once under its own goal window and once under a *matched
partner's* -- a clip that passes through the same state with a different commanded goal.

**Three parts** (``--parts``, all by default; one IsaacLab process):

``corpus``
    Every human x0 clip from t = 0 under its own schedule, deterministic actions. Gives ``sigma_a`` (the per-DOF
    action standard deviation every ``|da|`` below is divided by) and a determinism check: how far replicas that
    start from the same reference state drift apart (they sit at different floor positions, so float rounding and
    GPU PhysX both enter).

``x0``
    *Hubs* are hold nodes where the corpus itself presents one state with several commanded goals:

    * ``standing`` -- every human clip opens with a standing segment, and slot 1 (the next hold) is that clip's
      first move. This is the card's acceptance case: different slot-1 goals must change the action near the fork.
    * ``crow`` (node of ``Crane_Crow_Pose_or_Bakasana|L_HAND:G|R_HAND:G@prone``) -- release v3's spliced clips of
      E1 (press), E2 (jump-back) and B1 (to plank) share Crow -a's real lead-in, so the crow state is identical and
      slot 1 is a handstand, a chaturanga, a plank, or (Crow -a itself, E3's D) standing.
    * ``handstand`` (``Handstand_pose_or_Adho_Mukha_Vrksasana|L_HAND:G|R_HAND:G@inverted``) -- E3 (lower to crow)
      and E5 (float down to chaturanga) leave Handstand -a's handstand; Handstand -a itself steps down.

    *Partners* of an own probe ``(A, t_a)``: every segment of the hub's node in another clip, at the clip time with
    the **same dwell remaining** (``t_b = t_end_B - (t_end_A - t_a)``), kept only if its reference frame matches A's
    (24-body mean distance after the segment anchor below, ``--state-match-m``). The partner's goal window is
    installed exactly as training serves it -- the schedule is switched to ``(B, t_b)`` and refreshed
    (``ContactGraphControl._refresh_goal_indices``), so deadlines, dwell channels and contact sets are training's --
    and placed in A's frame by the rigid yaw + XY transform that maps B's hold exemplar onto A's (identity for the
    spliced clips, which are built in their source's frame). Partners are stratified by the commanded *pose*: the
    6-body pelvis-relative best-yaw distance between the two windows, slot by slot -- ``identical`` (the same goal
    window, e.g. A's own hold-extension variants), ``same_pose`` (every slot < ``--same-pose-m``), ``different``
    (some slot > ``--diff-goal-m``, or a slot valid in one window only).

    ``h = 1`` (exact): during A's own rollout from t = 0 -- the expert's own arrival, with its real history -- the
    observation is rebuilt under each partner window **without stepping**, and ``|da| / sigma_a`` is recorded. The
    own window is then restored and its rebuilt observation is asserted bit-identical to the one ``env.step``
    produced. (A rebuild after ``env.step`` must first put back the pre-step contact sample:
    ``_finalize_contact_state`` overwrites ``previous_contact_forces`` after the step's observation, so a naive
    rebuild sees a zero force rate in ``contact_obs_v1``.)

    ``h = 8 / 24`` (paired): blocks of envs reset to the same reference state ``--paired-warmup-s`` before the split,
    run the own schedule to the split, then the partner arms switch to their partner's window for good. Own-arm
    replicas give the same-goal divergence (PhysX and float rounding); ``|a_own(h) - a_arm(h)| / sigma_a`` at
    h = 1 .. ``--horizon`` gives the response, per stratum.

``forks``
    The fork battery: every ``fork_*``, ``edge_*`` and ``nohijack_*`` plan of the release, driven by
    ``SequenceVizRunner`` exactly as the training panel drives them (``set_manual_goal``, 10 settle steps, deadlines
    re-armed every 0.5 s, hold lead 1.2 s), every env scored. Per replica and per goal: the pose error after the best
    yaw (24 conditionable bodies, the panel's own set, and the project's 6 goal bodies), time held, and **supports
    from the terrain-filtered ground force**: every commanded zone down (>= 5 N) on >= 90 % of the hold window, and
    no other zone loaded (>= 3 % body weight, E1's rule) on >= 20 % of it -- a substitution. "Success" is the pose
    held (< 0.15 m) with both. Per-frame zone loads are saved (``forks/zone_loads.npz``) for re-scoring.

**Outputs** (``--out-dir``): ``meta.json``, ``corpus.json``, ``x0_h1.npz`` + ``x0_h1_rows.jsonl`` (one row per own
probe x partner), ``x0_paired.npz``, ``x0_summary.json``, ``forks/summary.json`` (the runner's, best-yaw metric),
``forks_scored.json`` (per goal, per replica) and ``report.md``.

Usage (G3's teacher; the "before" is e15500 run under G3's training config, so the release-v3 partners exist)::

    PYTHONPATH=. ../env_isaaclab/bin/python data/scripts/goal_causality_x0.py \\
        --checkpoint results/smpl_yogi_v2_expert56_g3_2f132f4299/epoch_3420.ckpt \\
        --out-dir output/goal_causality_x0/g3_e3420
    PYTHONPATH=. ../env_isaaclab/bin/python data/scripts/goal_causality_x0.py \\
        --checkpoint results/smpl_yogi_v2_expert56_a2dda5d2ac/epoch_15500.ckpt \\
        --config results/smpl_yogi_v2_expert56_g3_2f132f4299/resolved_configs.pt \\
        --out-dir output/goal_causality_x0/e15500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from policy_setup import add_common_args  # noqa: E402  (no torch before the simulator)

PARTS = ("corpus", "x0", "forks")

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
add_common_args(parser)
parser.add_argument("--config", default=None,
                    help="resolved TRAINING config to run under (default <checkpoint dir>/resolved_configs.pt). Its "
                         "four inference fields are overridden: no terminations, long episodes, starts not anchored.")
parser.add_argument("--out-dir", required=True)
parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
parser.add_argument("--hubs", nargs="+", default=["standing", "crow", "handstand"],
                    choices=["standing", "crow", "handstand"])
parser.add_argument("--h1-replicas", type=int, default=4, help="replicas of each own clip in the h=1 rollouts")
parser.add_argument("--state-match-m", type=float, default=0.10,
                    help="partner kept if its anchored reference frame is within this of the own frame (24-body mean)")
parser.add_argument("--same-pose-m", type=float, default=0.10)
parser.add_argument("--diff-goal-m", type=float, default=0.25)
parser.add_argument("--horizon", type=int, default=24, help="paired rollouts: steps recorded after the split")
parser.add_argument("--paired-warmup-s", type=float, default=1.0)
parser.add_argument("--paired-own-replicas", type=int, default=4)
parser.add_argument("--paired-arm-replicas", type=int, default=2)
parser.add_argument("--paired-max-partners", type=int, default=6)
parser.add_argument("--corpus-steps", type=int, default=600)
parser.add_argument("--fork-plans", nargs="*", default=None,
                    help="default: the release's fork_*, edge_* and nohijack_* plans")
parser.add_argument("--fork-max-seconds", type=float, default=24.0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
args.headless = True

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import numpy as np  # noqa: E402
import torch  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from policy_setup import build, motion_names  # noqa: E402

GOAL6 = ("Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head")
HUB_KEYS = {
    "standing": "standing|L_FOOT:G|R_FOOT:G@upright",
    "crow": "Crane_Crow_Pose_or_Bakasana|L_HAND:G|R_HAND:G@prone",
    "handstand": "Handstand_pose_or_Adho_Mukha_Vrksasana|L_HAND:G|R_HAND:G@inverted",
}
# Own clips per hub (x7s: the longest frozen dwell). Standing uses every human x0 clip that opens standing.
HUB_OWN = {
    "crow": ["220923_Crane_Crow_Pose_or_Bakasana_-a_x7s", "SYN_E1_press_high_s0_t6px_x7s",
             "SYN_E2_jumpback_mid_s0_t6px12_x7s", "SYN_B1_jumpplank_high_s1_t6rpx_x7s"],
    "handstand": ["220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a_x7s", "SYN_E3_lower_high_s3_t6px12_x7s",
                  "SYN_E5_floatdown_mid_s0_t6rpx12_x7s"],
}
# Dwell remaining (s) at which each own clip is probed; snapped to the control grid.
HUB_DWELL = {
    "standing": [0.6, 0.45, 0.3, 0.2, 0.1, 0.05],
    "crow": [7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.75, 0.5, 0.3, 0.2, 0.1],
    "handstand": [7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.75, 0.5, 0.3, 0.2, 0.1],
}
HUB_SPLIT = {"standing": [0.5, 0.25, 0.05], "crow": [5.0, 1.0, 0.2], "handstand": [5.0, 1.0, 0.2]}
# Paired blocks at the standing hub: the training panel's eight fork families.
STANDING_PAIRED = ["Warrior_II_Pose_or_Virabhadrasana_II_-a", "Tree_Pose_or_Vrksasana_-a",
                   "Crane_Crow_Pose_or_Bakasana_-a", "Handstand_pose_or_Adho_Mukha_Vrksasana_-a",
                   "Feathered_Peacock_Pose_or_Pincha_Mayurasana_-a", "Supported_Headstand_pose_or_Salamba_Sirsasana_-a",
                   "Side_Plank_Pose_or_Vasisthasana_-a", "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a"]
GOAL_OBS_KEYS = ("masked_mimic_target_poses", "masked_mimic_target_masks", "masked_mimic_target_times",
                 "masked_mimic_target_poses_masks", "contact_goal_obs", "contact_goal_masks")
CONTACT_SAMPLE = ("previous_contact_forces", "contact_temporal_valid", "prev_contact_force_magnitudes")
LOAD_FRAC = 0.03          # E1's support rule: a zone carries load at >= 3 % body weight (substitutions)
TOUCH_N = 5.0             # a commanded zone is down when its terrain-filtered ground force is >= 5 N
REALISED_SHARE = 0.90     # commanded zone down on >= 90 % of the hold window
UNWANTED_SHARE = 0.20     # any other zone loaded (>= 3 % BW) on >= 20 % of it


def say(msg: str) -> None:
    print(f"x0: {msg}", flush=True)


def is_x0(stem: str) -> bool:
    return not re.search(r"_x\d+s$", stem)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------- #
# Geometry helpers (torch, batched)
# ---------------------------------------------------------------------------------------------------------------- #
def kabsch_yaw_xy(P: torch.Tensor, Q: torch.Tensor):
    """Rigid yaw-about-z + XY shift taking Q onto P (least squares over bodies). P, Q: [N, B, 3].

    Returns ``(yaw [N], shift [N, 2], residual [N])``: ``R(yaw) Q_xy + shift ~ P_xy``; the residual is the mean 3-D
    body distance after the map (z untouched).
    """
    pc, qc = P[..., :2].mean(1, keepdim=True), Q[..., :2].mean(1, keepdim=True)
    a, b = Q[..., :2] - qc, P[..., :2] - pc
    num = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).sum(-1)
    den = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(-1)
    yaw = torch.atan2(num, den)
    shift = pc.squeeze(1) - rotate_xy(qc.squeeze(1), yaw)
    mapped = apply_yaw_xy(Q, yaw, shift)
    return yaw, shift, (mapped - P).norm(dim=-1).mean(-1)


def rotate_xy(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate [..., 2] vectors by yaw [...] (broadcast over the leading dims of v beyond yaw's)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    while c.dim() < v.dim() - 1:
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def apply_yaw_xy(X: torch.Tensor, yaw: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """X [N, ..., 3] -> R(yaw) X_xy + shift, z unchanged."""
    xy = rotate_xy(X[..., :2], yaw)
    sh = shift
    while sh.dim() < xy.dim():
        sh = sh.unsqueeze(1)
    return torch.cat([xy + sh, X[..., 2:]], -1)


def best_yaw_shape_dist(X: torch.Tensor, Y: torch.Tensor, ids, pelvis: int) -> torch.Tensor:
    """Mean distance over bodies ``ids`` of pelvis-relative X and Y after the yaw that best aligns X to Y. [N]."""
    a = X[:, ids] - X[:, pelvis:pelvis + 1]
    b = Y[:, ids] - Y[:, pelvis:pelvis + 1]
    num = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).sum(-1)
    den = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(-1)
    th = torch.atan2(num, den)
    rot = torch.cat([rotate_xy(a[..., :2], th), a[..., 2:]], -1)
    return (rot - b).norm(dim=-1).mean(-1)


def fdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Observation difference; some observation entries are bool tensors."""
    return a.float() - b.float()


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion of a rotation by yaw about +z."""
    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)


# ---------------------------------------------------------------------------------------------------------------- #
# The harness: env access, exact observation rebuilds, partner windows
# ---------------------------------------------------------------------------------------------------------------- #
class Harness:
    def __init__(self, built):
        self.env, self.agent = built["env"], built["agent"]
        self.lib = built["motion_lib"]
        self.device = self.env.device
        self.names = motion_names(self.lib)
        self.name_to_id = {n: i for i, n in enumerate(self.names)}
        self.ctrl = self.env.control_manager.components["contact_graph"]
        self.graph = self.ctrl.graph
        self.cfg = self.ctrl.config
        self.K = int(self.cfg.num_goal_steps)
        bodies = list(self.env.robot_config.kinematic_info.body_names)
        self.body_names = bodies
        self.pelvis = bodies.index("Pelvis")
        self.goal6 = [bodies.index(b) for b in GOAL6]
        self.dt = float(self.env.dt)
        self.E = self.env.num_envs
        self.lengths = self.lib.motion_lengths.float()
        self.pre_contact = None
        self.weight_n = float(getattr(self.cfg, "support_body_weight_n", 74.0 * 9.81))
        # The deployable inputs: the non-goal ones must not move when only the goal window does.
        self.actor_keys = list(self.agent.config.model.actor.in_keys)
        # Per-env rigid placement of the goal window (yaw about the reference frame's origin, then XY shift);
        # inactive rows are untouched. Applied after populate_context builds ctx.masked_mimic.
        self.t_active = torch.zeros(self.E, dtype=torch.bool, device=self.device)
        self.t_yaw = torch.zeros(self.E, device=self.device)
        self.t_shift = torch.zeros(self.E, 2, device=self.device)
        self._install_transform_hook()

    # ---- goal placement hook ------------------------------------------------------------------------------- #
    def _install_transform_hook(self):
        ctrl, h = self.ctrl, self
        original = ctrl.populate_context

        def populate_context(ctx):
            original(ctx)
            if not bool(h.t_active.any()):
                return
            mm = ctx.masked_mimic
            rows = h.t_active.nonzero(as_tuple=True)[0]
            offset = h.env.respawn_root_offset[rows, :2]                          # the playing clip's XY offset
            pos = mm.ref_pos[rows].clone()                                        # [n, K, B, 3] world
            local = pos.clone()
            local[..., :2] = local[..., :2] - offset[:, None, None, :]            # back to reference coordinates
            moved = apply_yaw_xy(local, h.t_yaw[rows], h.t_shift[rows])
            moved[..., :2] = moved[..., :2] + offset[:, None, None, :]
            ref_pos = mm.ref_pos.clone()
            ref_pos[rows] = moved
            q = yaw_quat(h.t_yaw[rows])[:, None, None, :].expand_as(mm.ref_rot[rows]).contiguous()
            from protomotions.utils.rotations import quat_mul
            ref_rot = mm.ref_rot.clone()
            ref_rot[rows] = quat_mul(q, mm.ref_rot[rows].contiguous(), w_last=True)
            mm.ref_pos = ref_pos
            mm.ref_rot = ref_rot

        ctrl.populate_context = populate_context

    # ---- stepping and exact rebuilds -------------------------------------------------------------------------- #
    def act(self, obs) -> torch.Tensor:
        td = self.agent.obs_dict_to_tensordict(self.agent.add_agent_info_to_obs(obs))
        return self.agent.model.forward_inference(td)["mean_action"]

    def step(self, action):
        # The contact sample the step's observation is computed against; _finalize_contact_state replaces it after.
        self.pre_contact = {k: getattr(self.env, k).clone() for k in CONTACT_SAMPLE}
        obs, *_ = self.env.step(action)
        return obs

    def rebuild(self):
        """Recompute every observation from the current sim state, as the last env.step did."""
        env = self.env
        post = {k: getattr(env, k) for k in CONTACT_SAMPLE}
        if self.pre_contact is not None:
            for k, v in self.pre_contact.items():
                setattr(env, k, v)
        try:
            env._current_context = env._build_global_context(env.simulator.get_robot_state())
            env.compute_observations(context=env._current_context)
            return env.get_obs()
        finally:
            for k, v in post.items():
                setattr(env, k, v)

    def reset_to(self, motion_ids: torch.Tensor, times: torch.Tensor):
        env = self.env
        ids = torch.arange(self.E, device=self.device)
        env.motion_manager.motion_ids[ids] = motion_ids.to(self.device).long()
        env.motion_manager.motion_times[ids] = times.to(self.device).float()
        self.t_active.zero_()
        obs, _ = env.reset(ids, sample_flat=True, disable_motion_resample=True)
        self.pre_contact = None
        return obs

    # ---- schedule swaps ------------------------------------------------------------------------------------- #
    def swap(self, rows, motion, times, yaw, shift):
        mm = self.env.motion_manager
        mm.motion_ids[rows] = motion
        mm.motion_times[rows] = times
        self.t_yaw[rows] = yaw
        self.t_shift[rows] = shift
        self.t_active[rows] = (yaw.abs() > 0) | (shift.abs().amax(-1) > 0)
        self.ctrl._refresh_goal_indices()

    def snapshot_schedule(self):
        mm = self.env.motion_manager
        return (mm.motion_ids.clone(), mm.motion_times.clone(), self.t_active.clone(), self.t_yaw.clone(),
                self.t_shift.clone())

    def restore_schedule(self, snap):
        mm = self.env.motion_manager
        mm.motion_ids.copy_(snap[0])
        mm.motion_times.copy_(snap[1])
        self.t_active.copy_(snap[2])
        self.t_yaw.copy_(snap[3])
        self.t_shift.copy_(snap[4])
        self.ctrl._refresh_goal_indices()

    def window_record(self, rows):
        """The goal window the policy sees now, for ``rows`` (after a rebuild)."""
        ctx = self.env._current_context
        c = self.ctrl
        return dict(
            pose=ctx.masked_mimic.ref_pos[rows].clone(),                      # [n, K, 24, 3] world, placed
            valid=c.goal_valid[rows].clone(),
            node=c._gathered["node"][rows].clone(),
            deadline=c._time_offsets[rows].clone(),
            dwell=c._dwell_features[rows].clone() if c._dwell_features.shape[-1] else
            torch.zeros(len(rows), self.K, 0, device=self.device),
        )

    # ---- the goal window as a pure function of (motion, time) ----------------------------------------------- #
    def segment_table(self, node_key):
        node = self.graph.node_id_for_key(node_key)
        hits = (self.graph.seg_node == node).nonzero(as_tuple=False).cpu().tolist()
        return node, [(m, k) for m, k in hits if k < int(self.graph.seg_count[m])]

    def seg(self, m, k):
        g = self.graph
        return float(g.seg_start[m, k]), float(g.seg_end[m, k]), float(g.seg_hold[m, k])

    def frames(self, mids, times):
        """Reference body positions [N, 24, 3] (reference coordinates)."""
        st = self.lib.get_motion_state(mids.to(self.device).long(), times.to(self.device).float())
        return st.rigid_body_pos

    def window(self, mids, times):
        g, c = self.graph, self.cfg
        mids = mids.to(self.device).long()
        times = times.to(self.device).float()
        idx, valid = g.next_goal_indices(mids, times, self.K, min_lead_s=c.min_lead_s,
                                         include_current=c.include_current_segment,
                                         interval=bool(getattr(c, "interval_schedule", False)))
        gath = g.gather(mids, idx)
        lengths = self.lib.get_motion_length(mids).unsqueeze(-1)
        t_hold = torch.where(torch.isfinite(gath["t_hold"]), gath["t_hold"], lengths).minimum(lengths)
        pose = self.frames(mids.unsqueeze(-1).expand_as(idx).reshape(-1), t_hold.reshape(-1))
        return dict(node=gath["node"], valid=valid, t_hold=t_hold, pose=pose.view(len(mids), self.K, -1, 3),
                    deadline=torch.where(valid, (t_hold - times.unsqueeze(-1)).clamp(min=0.0),
                                         torch.zeros_like(t_hold)))


def goal_gaps(h: Harness, own_pose, own_valid, own_node, par_pose, par_valid, par_node):
    """Per slot: 6-body pelvis-relative best-yaw shape gap and 24-body world gap; validity mismatch -> inf."""
    n, K = own_valid.shape
    shape = torch.full((n, K), float("inf"), device=own_pose.device)
    world = torch.full((n, K), float("inf"), device=own_pose.device)
    for k in range(K):
        both = own_valid[:, k] & par_valid[:, k]
        none = ~own_valid[:, k] & ~par_valid[:, k]
        d6 = best_yaw_shape_dist(own_pose[:, k], par_pose[:, k], h.goal6, h.pelvis)
        dw = (own_pose[:, k] - par_pose[:, k]).norm(dim=-1).mean(-1)
        shape[:, k] = torch.where(both, d6, torch.where(none, torch.zeros_like(d6), shape[:, k]))
        world[:, k] = torch.where(both, dw, torch.where(none, torch.zeros_like(dw), world[:, k]))
    return shape, world


def stratum_of(shape_gap, world_gap, same_nodes, same_timing, same_m, diff_m):
    worst = np.max(shape_gap, axis=-1)
    identical = same_nodes & same_timing & (np.max(world_gap, axis=-1) < 1e-3)
    return np.where(identical, "identical",
                    np.where(worst < same_m, "same_pose", np.where(worst > diff_m, "different", "between")))


# ---------------------------------------------------------------------------------------------------------------- #
# Part 1: corpus -- sigma_a and the replica determinism check
# ---------------------------------------------------------------------------------------------------------------- #
def run_corpus(h: Harness, out: Path):
    human = [i for i, n in enumerate(h.names) if is_x0(n) and not n.startswith("SYN_")]
    E = h.E
    clip_of_env = torch.tensor([human[e % len(human)] for e in range(E)])
    steps = int(args.corpus_steps)
    say(f"corpus: {len(human)} human x0 clips x ~{E // len(human)} replicas, {steps} steps from t = 0")
    obs = h.reset_to(clip_of_env, torch.zeros(E))
    lengths = h.lengths[clip_of_env.to(h.device)]
    D = None
    s1 = s2 = None
    count = None
    marks = sorted({m for m in (1, 10, 30, 100, 300, steps) if m <= steps})
    spread = {}
    first_of_clip = {}
    for e in range(E):
        first_of_clip.setdefault(int(clip_of_env[e]), e)
    ref_env = torch.tensor([first_of_clip[int(c)] for c in clip_of_env], device=h.device)
    for s in range(steps):
        a = h.act(obs)
        if D is None:
            D = a.shape[-1]
            s1 = torch.zeros(D, dtype=torch.float64, device=h.device)
            s2 = torch.zeros(D, dtype=torch.float64, device=h.device)
            count = torch.zeros((), dtype=torch.float64, device=h.device)
        live = (h.env.motion_manager.motion_times + 0.5 < lengths)
        al = a[live].double()
        s1 += al.sum(0)
        s2 += (al * al).sum(0)
        count += live.sum()
        obs = h.step(a)
        if s + 1 in marks:
            st = h.env.simulator.get_robot_state()
            local = st.rigid_body_pos - h.env.respawn_root_offset[:, None, :]
            dpos = (local - local[ref_env]).norm(dim=-1).mean(-1)             # mean body drift vs replica 0
            dact = (a - a[ref_env]).abs().max(-1).values
            others = torch.arange(E, device=h.device) != ref_env
            spread[str(s + 1)] = dict(
                body_drift_m_p50=float(dpos[others].median()), body_drift_m_p90=float(dpos[others].quantile(0.9)),
                body_drift_m_max=float(dpos[others].max()), exactly_equal_share=float((dpos[others] == 0).float().mean()),
                action_maxabs_p50=float(dact[others].median()), action_maxabs_max=float(dact[others].max()))
    mean = s1 / count
    sigma = (s2 / count - mean * mean).clamp(min=0).sqrt().float()
    rec = dict(clips=len(human), envs=E, steps=steps, samples=int(count), sigma_a=sigma.cpu().tolist(),
               sigma_a_mean=float(sigma.mean()), sigma_a_min=float(sigma.min()), sigma_a_max=float(sigma.max()),
               replica_spread=spread,
               note="replicas start from the same reference state at different floor positions; drift is "
                    "clip-local (respawn offset removed), mean over 24 bodies, against each clip's first replica")
    (out / "corpus.json").write_text(json.dumps(rec, indent=1))
    say(f"corpus: sigma_a mean {rec['sigma_a_mean']:.3f} rad (min {rec['sigma_a_min']:.3f}, max "
        f"{rec['sigma_a_max']:.3f}) over {rec['samples']} samples")
    for k, v in spread.items():
        say(f"  step {k:>4}: replica drift p50 {v['body_drift_m_p50']:.2e} m, max {v['body_drift_m_max']:.2e} m, "
            f"bit-equal share {v['exactly_equal_share']:.3f}; action max|d| p50 {v['action_maxabs_p50']:.2e}")
    return sigma


# ---------------------------------------------------------------------------------------------------------------- #
# Part 2: X0
# ---------------------------------------------------------------------------------------------------------------- #
def plan_hubs(h: Harness):
    """Own probes and partner pools per hub."""
    plans = []
    for hub in args.hubs:
        node, segs = h.segment_table(HUB_KEYS[hub])
        if hub == "standing":
            segs = [(m, k) for m, k in segs if k == 0 and not h.names[m].startswith("SYN_")]
            own = [(m, k) for m, k in segs if is_x0(h.names[m])]
            pool = segs                                   # all variants; cross-clip ones are thinned below
        else:
            first = {}
            for m, k in segs:
                first.setdefault(m, min(kk for mm, kk in segs if mm == m))
            missing = [s for s in HUB_OWN[hub] if s not in h.name_to_id]
            if missing:
                raise SystemExit(f"hub {hub}: own clips not in the library: {missing}")
            own = [(h.name_to_id[s], first[h.name_to_id[s]]) for s in HUB_OWN[hub]]
            pool = segs
        plans.append(dict(hub=hub, node=node, own=own, pool=pool))
        say(f"hub {hub}: node {node}, {len(own)} own clips, {len(pool)} segments of the node in the library")
    return plans


def base_stem(stem: str) -> str:
    return re.sub(r"_x\d+s$", "", stem)


def partner_pool_for(h: Harness, plan, own_m):
    """Candidate partner segments for one own clip (cross-clip x0 only at the standing hub, plus own variants)."""
    own_base = base_stem(h.names[own_m])
    out = []
    for m, k in plan["pool"]:
        if m == own_m:
            continue
        stem = h.names[m]
        if plan["hub"] == "standing" and not is_x0(stem) and base_stem(stem) != own_base:
            continue
        out.append((m, k))
    return out


def anchor(h: Harness, own_m, own_k, par):
    """Rigid map (yaw, shift) placing each partner segment's hold exemplar on the own one's; identity if exact."""
    _, _, th_a = h.seg(own_m, own_k)
    P = h.frames(torch.tensor([own_m]), torch.tensor([th_a])).expand(len(par), -1, -1)
    Q = h.frames(torch.tensor([m for m, _ in par]), torch.tensor([h.seg(m, k)[2] for m, k in par]))
    yaw, shift, res = kabsch_yaw_xy(P, Q)
    raw = (Q - P).norm(dim=-1).mean(-1)
    exact = raw < 1e-4
    yaw = torch.where(exact, torch.zeros_like(yaw), yaw)
    shift = torch.where(exact.unsqueeze(-1), torch.zeros_like(shift), shift)
    return yaw, shift, torch.where(exact, raw, res)


def run_x0(h: Harness, out: Path, sigma: torch.Tensor):
    plans = plan_hubs(h)
    dt, E, dev = h.dt, h.E, h.device
    R1 = int(args.h1_replicas)

    # ---- env layout: h=1 blocks, then paired blocks ---------------------------------------------------------- #
    start_motion, start_time = [], []
    h1_env = []        # (env, plan index, own (m, k), [probe steps], [nominal r])
    for pi, plan in enumerate(plans):
        for own_m, own_k in plan["own"]:
            t0, t1, th = h.seg(own_m, own_k)
            steps, rs = [], []
            for r in HUB_DWELL[plan["hub"]]:
                t = t1 - r
                lo = max(th, 0.1) if plan["hub"] != "standing" else 0.1
                if t < lo:
                    continue
                s = int(round(t / dt))
                if s not in steps:
                    steps.append(s)
                    rs.append(r)
            for _ in range(R1):
                h1_env.append((len(start_motion), pi, (own_m, own_k), steps, rs))
                start_motion.append(own_m)
                start_time.append(0.0)
    paired = []        # blocks: dict(plan, own, split_r, start, split_step, envs=[(env, arm, partner (m,k) or None)])
    warm = float(args.paired_warmup_s)
    for pi, plan in enumerate(plans):
        owns = plan["own"]
        if plan["hub"] == "standing":
            owns = [(m, k) for m, k in owns if any(h.names[m].endswith(s) for s in STANDING_PAIRED)]
        for own_m, own_k in owns:
            t0, t1, th = h.seg(own_m, own_k)
            for r in HUB_SPLIT[plan["hub"]]:
                t_a = t1 - r
                if t_a < (0.1 if plan["hub"] == "standing" else th):
                    continue
                start = max(0.0, t_a - warm)
                split_step = int(round((t_a - start) / dt))
                paired.append(dict(plan=pi, own=(own_m, own_k), r=r, start=start, split_step=split_step, arms=[]))
    say(f"x0 layout: {len(h1_env)} h=1 envs, {len(paired)} paired blocks (arms chosen after the partner screen)")

    # ---- partner screen at nominal probe times (pure functions of the library and the graph) ---------------- #
    screen = {}
    for pi, plan in enumerate(plans):
        for own_m, own_k in plan["own"]:
            par = partner_pool_for(h, plan, own_m)
            if not par:
                continue
            yaw, shift, res = anchor(h, own_m, own_k, par)
            screen[(own_m, own_k)] = dict(partners=par, yaw=yaw, shift=shift, anchor_res=res)
    say(f"x0 screen: {sum(len(v['partners']) for v in screen.values())} (own, partner) segment pairs")

    def partner_times(own_m, own_k, t_a: torch.Tensor, par):
        """t_b with the own dwell remaining, and whether it lies inside the partner segment."""
        _, t1, _ = h.seg(own_m, own_k)
        r = t1 - t_a
        tb, ok = [], []
        for m, k in par:
            s0, s1, _ = h.seg(m, k)
            t = s1 - r
            tb.append(t)
            ok.append((t >= s0 - 1e-6) & (t <= s1 + 1e-6))
        return torch.stack(tb, -1), torch.stack(ok, -1)

    # choose paired arms at the nominal split time: identical first, then same-pose, then different (distinct slot 1)
    for blk in paired:
        own_m, own_k = blk["own"]
        sc = screen.get((own_m, own_k))
        if sc is None:
            continue
        _, t1, _ = h.seg(own_m, own_k)
        t_a = torch.tensor([t1 - blk["r"]], device=dev)
        tb, ok = partner_times(own_m, own_k, t_a, sc["partners"])
        tb, ok = tb[0], ok[0]
        n = len(sc["partners"])
        pm = torch.tensor([m for m, _ in sc["partners"]], device=dev)
        own_frame = h.frames(torch.tensor([own_m]), t_a.cpu())
        par_frames = apply_yaw_xy(h.frames(pm, tb.clamp(min=0)), sc["yaw"], sc["shift"])
        state_gap = (par_frames - own_frame).norm(dim=-1).mean(-1)
        wo = h.window(torch.tensor([own_m]), t_a)
        wp = h.window(pm, tb.clamp(min=0))
        wp_pose = apply_yaw_xy(wp["pose"].view(n, -1, 3), sc["yaw"], sc["shift"]).view_as(wp["pose"])
        shape, world = goal_gaps(h, wo["pose"].expand(n, -1, -1, -1), wo["valid"].expand(n, -1),
                                 wo["node"].expand(n, -1), wp_pose, wp["valid"], wp["node"])
        same_nodes = ((wp["node"] == wo["node"]) | ~(wp["valid"] | wo["valid"])).all(-1)
        same_timing = (wp["deadline"] - wo["deadline"]).abs().amax(-1) < 1e-3
        strata = stratum_of(shape.cpu().numpy(), world.cpu().numpy(), same_nodes.cpu().numpy(),
                            same_timing.cpu().numpy(), args.same_pose_m, args.diff_goal_m)
        usable = (ok & (state_gap < args.state_match_m)).cpu().numpy()
        chosen, seen_slot1 = [], set()
        order = list(np.argsort(state_gap.cpu().numpy()))
        for want, cap in (("identical", 1), ("same_pose", 2), ("different", args.paired_max_partners)):
            for j in order:
                if len(chosen) >= args.paired_max_partners:
                    break
                if not usable[j] or strata[j] != want or j in chosen:
                    continue
                if want == "different":
                    key = int(wp["node"][j, -1]) if bool(wp["valid"][j, -1]) else -1
                    if key in seen_slot1:
                        continue
                    seen_slot1.add(key)
                chosen.append(j)
                if sum(1 for c in chosen if strata[c] == want) >= cap:
                    break
        blk["arms"] = [(int(j), str(strata[j])) for j in chosen]

    # assign paired envs
    for blk in paired:
        envs = []
        for _ in range(args.paired_own_replicas):
            envs.append((len(start_motion), -1))
            start_motion.append(blk["own"][0])
            start_time.append(blk["start"])
        for j, _st in blk["arms"]:
            for _ in range(args.paired_arm_replicas):
                envs.append((len(start_motion), j))
                start_motion.append(blk["own"][0])
                start_time.append(blk["start"])
        blk["envs"] = envs
    used = len(start_motion)
    if used > E:
        raise SystemExit(f"x0 needs {used} envs, have {E}: lower --h1-replicas or the paired replicas")
    filler = h.name_to_id[h.names[plans[0]["own"][0][0]]] if plans else 0
    while len(start_motion) < E:
        start_motion.append(filler)
        start_time.append(0.0)
    say(f"x0: {used} of {E} envs used ({E - used} idle fillers)")

    # ---- the lockstep rollout ------------------------------------------------------------------------------- #
    probes_at = {}
    for env_i, pi, own, steps, rs in h1_env:
        for s, r in zip(steps, rs):
            probes_at.setdefault(s, []).append((env_i, pi, own, r))
    splits_at = {}
    for bi, blk in enumerate(paired):
        splits_at.setdefault(blk["split_step"], []).append(bi)
    last = max([max(probes_at, default=0)] + [b["split_step"] + args.horizon for b in paired])
    H = int(args.horizon)
    pair_env = np.array([e for b in paired for e, _ in b["envs"]], dtype=np.int64)
    pair_split = {e: b["split_step"] for b in paired for e, _ in b["envs"]}
    act_rec = np.full((H, max(len(pair_env), 1), 0), np.nan, dtype=np.float32)
    pos_rec = None
    col_of = {e: i for i, e in enumerate(pair_env)}

    sig = sigma.to(dev).clamp(min=1e-6)
    rows = []
    checks = dict(own_rebuild_max_abs=0.0, own_rebuild_checked=0, nongoal_max_abs=0.0, identical_max_da=0.0,
                  naive_rebuild_contact_obs_max_abs=None, manual_path=None)
    t_start = time.time()
    obs = h.reset_to(torch.tensor(start_motion), torch.tensor(start_time))
    for s in range(last + 1):
        own_obs = obs
        # -- paired splits: switch the arms for good, before this step's action ---------------------------------- #
        if s in splits_at:
            for bi in splits_at[s]:
                blk = paired[bi]
                own_m, own_k = blk["own"]
                sc = screen[(own_m, own_k)]
                arm_envs = [(e, j) for e, j in blk["envs"] if j >= 0]
                if not arm_envs:
                    continue
                rows_t = torch.tensor([e for e, _ in arm_envs], device=dev)
                js = [j for _, j in arm_envs]
                par = [sc["partners"][j] for j in js]
                t_a = h.env.motion_manager.motion_times[rows_t]
                tb, ok = partner_times(own_m, own_k, t_a, par)
                tb = torch.stack([tb[i, i] for i in range(len(js))])
                h.swap(rows_t, torch.tensor([m for m, _ in par], device=dev), tb.clamp(min=0),
                       sc["yaw"][js], sc["shift"][js])
            h.ctrl.prev_first_index = h.ctrl.goal_index[:, 0].clone()
            own_obs = h.rebuild()
            obs = own_obs
        a_own = h.act(own_obs)
        # -- h=1 probes ------------------------------------------------------------------------------------------ #
        if s in probes_at:
            if h.pre_contact is None:
                raise RuntimeError("an h=1 probe at the reset step: its observation used a cleared contact sample")
            group = probes_at[s]
            env_rows = torch.tensor([g[0] for g in group], device=dev)
            # the naive rebuild (no contact-sample restore) once, to document the trap
            if checks["naive_rebuild_contact_obs_max_abs"] is None and h.pre_contact is not None:
                saved = h.pre_contact
                h.pre_contact = None
                naive = h.rebuild()
                h.pre_contact = saved
                checks["naive_rebuild_contact_obs_max_abs"] = float(
                    fdiff(naive["contact_obs_v1"], own_obs["contact_obs_v1"]).abs().max())
            if checks["manual_path"] is None:
                checks["manual_path"] = manual_path_check(h, own_obs)
            ref_obs = h.rebuild()
            dmax = max(float(fdiff(ref_obs[k], own_obs[k]).abs().max()) for k in own_obs)
            checks["own_rebuild_max_abs"] = max(checks["own_rebuild_max_abs"], dmax)
            checks["own_rebuild_checked"] += 1
            own_win = h.window_record(env_rows)
            snap = h.snapshot_schedule()
            per_env = []
            for (env_i, pi, (own_m, own_k), r) in group:
                sc = screen.get((own_m, own_k))
                per_env.append(sc)
            K = max([len(sc["partners"]) for sc in per_env if sc is not None], default=0)
            t_a_all = h.env.motion_manager.motion_times[env_rows].clone()
            for k in range(K):
                sel, mids, tbs, yaws, shifts, meta = [], [], [], [], [], []
                for gi, (env_i, pi, (own_m, own_k), r) in enumerate(group):
                    sc = per_env[gi]
                    if sc is None or k >= len(sc["partners"]):
                        continue
                    m, kk = sc["partners"][k]
                    s0, s1, _ = h.seg(m, kk)
                    _, t1, _ = h.seg(own_m, own_k)
                    tb = s1 - (t1 - float(t_a_all[gi]))
                    if tb < s0 - 1e-6 or tb > s1 + 1e-6:
                        continue
                    sel.append(gi)
                    mids.append(m)
                    tbs.append(tb)
                    yaws.append(sc["yaw"][k])
                    shifts.append(sc["shift"][k])
                    meta.append((env_i, pi, own_m, own_k, r, m, kk, float(sc["anchor_res"][k])))
                if not sel:
                    continue
                sel_t = torch.tensor(sel, device=dev)
                rows_t = env_rows[sel_t]
                mids_t = torch.tensor(mids, device=dev)
                tbs_t = torch.tensor(tbs, device=dev, dtype=torch.float32)
                yaw_t, shift_t = torch.stack(yaws), torch.stack(shifts)
                # the matched-state screen at the actual times
                own_frame = h.frames(h.env.motion_manager.motion_ids[rows_t], t_a_all[sel_t])
                par_frame = apply_yaw_xy(h.frames(mids_t, tbs_t), yaw_t, shift_t)
                state_gap = (par_frame - own_frame).norm(dim=-1).mean(-1)
                keep = state_gap < args.state_match_m
                if not bool(keep.any()):
                    continue
                rows_t, mids_t, tbs_t = rows_t[keep], mids_t[keep], tbs_t[keep]
                yaw_t, shift_t, state_gap = yaw_t[keep], shift_t[keep], state_gap[keep]
                sel_t = sel_t[keep]
                meta = [mt for mt, kp in zip(meta, keep.cpu().tolist()) if kp]
                h.swap(rows_t, mids_t, tbs_t, yaw_t, shift_t)
                par_obs = h.rebuild()
                a_par = h.act(par_obs)
                par_win = h.window_record(rows_t)
                nongoal = max(float(fdiff(par_obs[key][rows_t], own_obs[key][rows_t]).abs().max())
                              for key in h.actor_keys if key not in GOAL_OBS_KEYS)
                checks["nongoal_max_abs"] = max(checks["nongoal_max_abs"], nongoal)
                goal_dobs = {key: fdiff(par_obs[key][rows_t], own_obs[key][rows_t]).norm(dim=-1)
                             for key in GOAL_OBS_KEYS if key in own_obs}
                da = a_par[rows_t] - a_own[rows_t]
                rms_z = ((da / sig) ** 2).mean(-1).sqrt()
                ow = {kk_: v[sel_t] for kk_, v in own_win.items()}
                shape, world = goal_gaps(h, ow["pose"], ow["valid"], ow["node"], par_win["pose"], par_win["valid"],
                                         par_win["node"])
                same_nodes = ((par_win["node"] == ow["node"]) | ~(par_win["valid"] | ow["valid"])).all(-1)
                same_timing = (par_win["deadline"] - ow["deadline"]).abs().amax(-1) < 1e-3
                if ow["dwell"].shape[-1]:
                    same_timing &= (par_win["dwell"] - ow["dwell"]).abs().amax(-1).amax(-1) < 1e-4
                strata = stratum_of(shape.cpu().numpy(), world.cpu().numpy(), same_nodes.cpu().numpy(),
                                    same_timing.cpu().numpy(), args.same_pose_m, args.diff_goal_m)
                rms_np, l2_np = rms_z.cpu().numpy(), da.norm(dim=-1).cpu().numpy()
                mx_np = da.abs().amax(-1).cpu().numpy()
                for i, (env_i, pi, own_m, own_k, r, m, kk, ares) in enumerate(meta):
                    if strata[i] == "identical":
                        checks["identical_max_da"] = max(checks["identical_max_da"], float(mx_np[i]))
                    rows.append(dict(
                        hub=plans[pi]["hub"], env=int(env_i), own=h.names[own_m], own_motion=int(own_m),
                        t_a=float(t_a_all[int(sel_t[i])]),
                        r=float(h.seg(own_m, own_k)[1] - float(t_a_all[int(sel_t[i])])),
                        r_nominal=float(r), partner=h.names[m], partner_motion=int(m), t_b=float(tbs_t[i]),
                        state_gap_m=float(state_gap[i]), anchor_residual_m=ares,
                        anchor_yaw_deg=float(torch.rad2deg(yaw_t[i])), anchor_shift_m=float(shift_t[i].norm()),
                        own_nodes=[int(x) for x in ow["node"][i].tolist()],
                        own_valid=[bool(x) for x in ow["valid"][i].tolist()],
                        partner_nodes=[int(x) for x in par_win["node"][i].tolist()],
                        partner_valid=[bool(x) for x in par_win["valid"][i].tolist()],
                        own_deadline=[round(float(x), 4) for x in ow["deadline"][i].tolist()],
                        partner_deadline=[round(float(x), 4) for x in par_win["deadline"][i].tolist()],
                        own_dwell=[[round(float(y), 4) for y in x] for x in ow["dwell"][i].tolist()],
                        partner_dwell=[[round(float(y), 4) for y in x] for x in par_win["dwell"][i].tolist()],
                        shape_gap_m=[float(x) for x in shape[i].tolist()],
                        world_gap_m=[float(x) for x in world[i].tolist()],
                        stratum=str(strata[i]), da_rms_z=float(rms_np[i]), da_l2=float(l2_np[i]),
                        da_maxabs=float(mx_np[i]),
                        dobs={key: float(v[i]) for key, v in goal_dobs.items()}))
                h.restore_schedule(snap)
            back = h.rebuild()
            dmax = max(float(fdiff(back[k], own_obs[k]).abs().max()) for k in own_obs)
            checks["own_rebuild_max_abs"] = max(checks["own_rebuild_max_abs"], dmax)
            checks["own_rebuild_checked"] += 1
        # -- record paired actions ------------------------------------------------------------------------------- #
        if len(pair_env):
            if act_rec.shape[-1] == 0:
                act_rec = np.full((H, len(pair_env), a_own.shape[-1]), np.nan, dtype=np.float32)
                pos_rec = np.full((H, len(pair_env), len(h.body_names), 3), np.nan, dtype=np.float32)
            rel = np.array([s - pair_split[e] for e in pair_env])
            live = (rel >= 0) & (rel < H)
            if live.any():
                cols = np.nonzero(live)[0]
                act_np = a_own[torch.tensor(pair_env[cols], device=dev)].cpu().numpy()
                for c, a_row in zip(cols, act_np):
                    act_rec[rel[c], c] = a_row
                st = h.env.simulator.get_robot_state()
                e_t = torch.tensor(pair_env[cols], device=dev)
                local = (st.rigid_body_pos[e_t] - h.env.respawn_root_offset[e_t][:, None, :]).cpu().numpy()
                for c, p in zip(cols, local):
                    pos_rec[rel[c], c] = p
        if s == last:
            break
        obs = h.step(a_own)
        if s % 100 == 0:
            say(f"x0 rollout step {s}/{last} ({time.time() - t_start:.0f} s, {len(rows)} h=1 rows)")
    say(f"x0 rollout done in {time.time() - t_start:.0f} s: {len(rows)} h=1 rows; checks {checks}")

    # ---- write the raw records ------------------------------------------------------------------------------ #
    with open(out / "x0_h1_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    blocks_meta = []
    for b in paired:
        own_m, own_k = b["own"]
        sc = screen.get((own_m, own_k))
        blocks_meta.append(dict(
            hub=plans[b["plan"]]["hub"], own=h.names[own_m], r=b["r"], start=b["start"], split_step=b["split_step"],
            arms=[dict(partner=h.names[sc["partners"][j][0]], stratum=st) for j, st in b["arms"]] if sc else [],
            envs=[[int(e), int(j)] for e, j in b["envs"]]))
    np.savez_compressed(out / "x0_paired.npz", actions=act_rec, positions=pos_rec if pos_rec is not None else
                        np.zeros(0), env_cols=pair_env, sigma_a=sigma.cpu().numpy())
    (out / "x0_paired_blocks.json").write_text(json.dumps(blocks_meta, indent=1))
    return rows, paired, blocks_meta, checks, act_rec, pos_rec, pair_env


def manual_path_check(h: Harness, own_obs):
    """The card's literal installer, measured: every env's OWN window re-issued through ``set_manual_goal``.

    ``set_manual_goal`` takes one ``hold_seconds`` per slot and uses it for both dwell channels (duration and
    remaining), while a scheduled segment has ``duration = t_end - t_hold`` and ``remaining = t_end - now``; its
    ``step()`` also floors the deadline at ``min_lead_s`` where training's floor is 0. This reports which actor
    inputs differ from training's when the same window goes through the manual path (passing the remaining dwell,
    as the panel does), then clears it and checks the schedule comes back exactly.
    """
    c = h.ctrl
    snap = h.snapshot_schedule()
    now = h.env.motion_manager.motion_times
    nodes = torch.where(c.goal_valid, c._gathered["node"], torch.full_like(c._gathered["node"], -1))
    remaining = (c._gathered["t_end"] - now.unsqueeze(-1)).clamp(min=0.0)
    remaining = torch.where(torch.isfinite(remaining), remaining, torch.zeros_like(remaining))
    ones = torch.ones_like(c.goal_valid)
    c.set_manual_goal(node_ids=nodes.clone(), pose_motion_ids=c._goal_motion_ids.clone(),
                      pose_times=c.target_times.clone(), time_offsets=c._time_offsets.clone(),
                      pose_visible=ones, contact_visible=ones, hold_seconds=remaining)
    try:
        man = h.rebuild()
        diffs = {k: float(fdiff(man[k], own_obs[k]).abs().max()) for k in h.actor_keys}
        a_man, a_own = h.act(man), h.act(own_obs)
        rms = float(((a_man - a_own) ** 2).mean(-1).sqrt().median())
    finally:
        c.clear_manual_goal()
        h.restore_schedule(snap)
    back = h.rebuild()
    restored = max(float(fdiff(back[k], own_obs[k]).abs().max()) for k in own_obs)
    say(f"manual-path check: actor inputs that differ from training's window: "
        f"{ {k: round(v, 4) for k, v in diffs.items() if v > 0} }; restored exactly: {restored == 0.0}")
    return dict(max_abs_by_key=diffs, action_rms_p50_raw=rms, schedule_restored_max_abs=restored)


def summarise_x0(rows, blocks_meta, act_rec, pos_rec, pair_env, sigma, checks, out: Path):
    """Medians of |da| / sigma_a by hub, own clip, dwell remaining and stratum; the paired horizons."""
    summary = dict(checks=checks, h1={}, paired={}, acceptance={})
    by = {}
    for r in rows:
        key = (r["hub"], r["own"], round(r["r_nominal"], 3), r["stratum"])
        by.setdefault(key, []).append(r)
    for (hub, own, rn, st), rs in sorted(by.items()):
        v = np.array([x["da_rms_z"] for x in rs])
        summary["h1"].setdefault(hub, {}).setdefault(own, {}).setdefault(str(rn), {})[st] = dict(
            n=len(v), p50=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
            mean=float(v.mean()), partners=len({x["partner"] for x in rs}))
    # hub-level curves: median over own clips of (different - same_pose) and of different alone
    for hub in sorted({r["hub"] for r in rows}):
        curve = {}
        for rn in sorted({round(r["r_nominal"], 3) for r in rows if r["hub"] == hub}, reverse=True):
            sel = [r for r in rows if r["hub"] == hub and round(r["r_nominal"], 3) == rn]
            dif = np.array([r["da_rms_z"] for r in sel if r["stratum"] == "different"])
            same = np.array([r["da_rms_z"] for r in sel if r["stratum"] == "same_pose"])
            ident = np.array([r["da_rms_z"] for r in sel if r["stratum"] == "identical"])
            curve[str(rn)] = dict(
                different_p50=float(np.median(dif)) if len(dif) else None, n_different=len(dif),
                same_pose_p50=float(np.median(same)) if len(same) else None, n_same_pose=len(same),
                identical_max=float(ident.max()) if len(ident) else None, n_identical=len(ident),
                excess_p50=(float(np.median(dif) - np.median(same)) if len(dif) and len(same) else None))
        summary["h1"].setdefault(hub, {})["_curve"] = curve
    # paired horizons: per block, same-goal floor (own vs own) and per arm stratum
    sig = sigma.cpu().numpy().clip(min=1e-6)
    for bi, b in enumerate(blocks_meta):
        envs = b["envs"]
        cols = {e: i for i, e in enumerate(pair_env.tolist())}
        own_cols = [cols[e] for e, j in envs if j < 0]
        res = {}
        for hh in sorted({1, 8, args.horizon}):
            if hh > act_rec.shape[0]:
                continue
            A = act_rec[hh - 1]
            P = pos_rec[hh - 1] if pos_rec is not None else None
            def d(i, j):
                return float(np.sqrt((((A[i] - A[j]) / sig) ** 2).mean()))
            def dp(i, j):
                return float(np.linalg.norm(P[i] - P[j], axis=-1).mean()) if P is not None else float("nan")
            floor = [d(i, j) for ii, i in enumerate(own_cols) for j in own_cols[ii + 1:]]
            floor_p = [dp(i, j) for ii, i in enumerate(own_cols) for j in own_cols[ii + 1:]]
            arms = {}
            for ai, arm in enumerate(b["arms"]):
                jcols = [cols[e] for e, j in envs if j >= 0][ai * args.paired_arm_replicas:(ai + 1) * args.paired_arm_replicas]
                vals = [d(i, j) for i in own_cols for j in jcols]
                pvals = [dp(i, j) for i in own_cols for j in jcols]
                arms.setdefault(arm["stratum"], []).append(dict(partner=arm["partner"], da=float(np.mean(vals)),
                                                               dpos_m=float(np.mean(pvals))))
            res[str(hh)] = dict(same_goal_floor=float(np.mean(floor)) if floor else None,
                                same_goal_dpos_m=float(np.mean(floor_p)) if floor_p else None, arms=arms)
        summary["paired"].setdefault(b["hub"], []).append(dict(own=b["own"], r=b["r"], horizons=res))
    # acceptance: the card's known case and G3's hub gate
    acc = {}
    st = summary["h1"].get("standing", {}).get("_curve", {})
    near = [v for k, v in st.items() if float(k) <= 0.2 and v["excess_p50"] is not None]
    if near:
        acc["standing_near_fork_excess_p50"] = float(np.median([v["excess_p50"] for v in near]))
        acc["standing_near_fork_different_p50"] = float(np.median([v["different_p50"] for v in near]))
        far = [v for k, v in st.items() if float(k) >= 0.45 and v["different_p50"] is not None]
        acc["standing_far_different_p50"] = float(np.median([v["different_p50"] for v in far])) if far else None
        acc["known_case_reproduced"] = bool(acc["standing_near_fork_excess_p50"] > 0.05)
    for hub in ("crow", "handstand"):
        cv = summary["h1"].get(hub, {}).get("_curve", {})
        near = [v for k, v in cv.items() if float(k) <= 1.0 and v["different_p50"] is not None]
        if near:
            exc = [v["excess_p50"] for v in near if v["excess_p50"] is not None]
            acc[f"{hub}_near_end_different_p50"] = float(np.median([v["different_p50"] for v in near]))
            acc[f"{hub}_near_end_excess_p50"] = float(np.median(exc)) if exc else None
            base = acc[f"{hub}_near_end_excess_p50"] if exc else acc[f"{hub}_near_end_different_p50"]
            acc[f"{hub}_near_end_excess_above_0.05"] = bool(base > 0.05)
    summary["acceptance"] = acc
    (out / "x0_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------------------------------------------------------- #
# Part 3: the fork battery
# ---------------------------------------------------------------------------------------------------------------- #
def run_forks(h: Harness, out: Path):
    from protomotions.agents.evaluators import sequence_viz as sv
    from protomotions.agents.evaluators.sequence_viz import SequenceVizConfig, SequenceVizRunner

    h.t_active.zero_()          # X0's partner placements must not leak into the panel's episodes
    plans = args.fork_plans
    if not plans:
        rel_dir = Path(str(h.cfg.graph_file)).parent / "plans"
        plans = sorted(str(p) for pat in ("fork_*.json", "edge_*.json", "nohijack_*.json")
                       for p in rel_dir.glob(pat))
    say(f"forks: {len(plans)} plans")
    cfg = SequenceVizConfig(viz_every=1, plan_files=list(plans), num_sequences=len(plans), num_hold_sequences=0,
                            max_seconds=args.fork_max_seconds, settle_steps=10, reissue_every_s=0.5,
                            hold_lead_s=1.2, hold_lead_mode="clamp", max_replicas=0, pose_arrive_m=0.15,
                            pose_depart_m=0.30, log_scalars=False, dump_traces=True)
    runner = SequenceVizRunner(h.agent, cfg)
    if len(runner.sequences) != len(plans):
        say(f"forks: WARNING only {len(runner.sequences)} of {len(plans)} plans resolved")
    stash = dict(fz=[])
    orig_zones = SequenceVizRunner._measured_ground_zones

    def zones(self, state, limit):
        gf = getattr(state, "rigid_body_ground_forces", None)
        stash["fz"].append(gf[:limit, :, 2].clamp(min=0).half().cpu() if gf is not None else None)
        return orig_zones(self, state, limit)

    def goal_pose_errors(self, sequences, positions, root_rots, frame_times):
        stash["positions"], stash["frame_times"] = positions, list(frame_times)
        errs = {}
        for name, ids in (("err24", self.control.conditionable_body_ids.cpu()), ("err6", torch.tensor(h.goal6))):
            flat, offsets = [], []
            for seq in sequences:
                offsets.append(len(flat))
                flat.extend(seq.goals)
            ref = self.env.motion_lib.get_motion_state(
                torch.tensor([g.pose_motion for g in flat], device=self.device),
                torch.tensor([g.pose_time for g in flat], device=self.device, dtype=torch.float32))
            rp = ref.rigid_body_pos.cpu()
            goal_local = rp[:, ids] - rp[:, self.pelvis_index].unsqueeze(1)
            T, C = positions.shape[0], positions.shape[1]
            out_e = np.full((T, C), np.nan, dtype=np.float32)
            S = len(sequences)
            for si, seq in enumerate(sequences):
                cols = torch.arange(si, C, S)
                if cols.numel() == 0:
                    continue
                pos = positions[:, cols].float()
                local = pos[:, :, ids] - pos[:, :, self.pelvis_index].unsqueeze(2)
                goals = goal_local[torch.tensor([offsets[si] + seq.active_index(t) for t in frame_times])]
                g = goals.unsqueeze(1).expand_as(local)
                num = (local[..., 0] * g[..., 1] - local[..., 1] * g[..., 0]).sum(-1)
                den = (local[..., 0] * g[..., 0] + local[..., 1] * g[..., 1]).sum(-1)
                th = torch.atan2(num, den).unsqueeze(-1)
                c, sn = torch.cos(th), torch.sin(th)
                rot = torch.stack([c * local[..., 0] - sn * local[..., 1], sn * local[..., 0] + c * local[..., 1],
                                   local[..., 2]], -1)
                out_e[:, cols.numpy()] = (rot - g).norm(dim=-1).mean(-1).numpy()
            errs[name] = out_e
        stash["err6"] = errs["err6"]
        return errs["err24"]

    SequenceVizRunner._measured_ground_zones = zones
    SequenceVizRunner._goal_pose_errors = goal_pose_errors
    sv.render_stick_video = lambda *a, **k: None
    fork_dir = out / "forks"
    fork_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runner.run(0, out_dir=fork_dir)
    say(f"forks: rollout done in {time.time() - t0:.0f} s")
    SequenceVizRunner._measured_ground_zones = orig_zones

    # ---- per replica, per goal: pose, time held, supports by load --------------------------------------------- #
    seqs = runner.sequences
    S = len(seqs)
    positions = stash["positions"].float().numpy()                                 # [T, C, 24, 3]
    fz = torch.stack(stash["fz"]).float().numpy()                                  # [T, C, 24]
    err6 = stash["err6"]
    ft = np.array(stash["frame_times"])
    ctrl = h.ctrl
    zone_names = list(ctrl._ground_zone_names)
    zmat = np.zeros((len(zone_names), len(h.body_names)), dtype=np.float32)
    zmat[ctrl._ground_zone_rows.cpu().numpy(), ctrl._ground_zone_cols.cpu().numpy()] = 1.0
    zone_load = fz @ zmat.T                                                        # [T, C, Z]
    loaded = zone_load >= LOAD_FRAC * h.weight_n
    touching = zone_load >= TOUCH_N
    ground_ids = ctrl._ground_pair_ids.cpu()
    traces = np.load(fork_dir / "pose_error_traces.npz")
    e24 = traces["pose_errors"]
    C = positions.shape[1]
    scored = {}
    for si, seq in enumerate(seqs):
        cols = list(range(si, C, S))
        ends = seq.ends
        goals_out = []
        for gi, goal in enumerate(seq.goals):
            hold = (ft >= ends[gi] - goal.hold_s) & (ft < ends[gi])
            window = (ft >= ends[gi] - goal.hold_s - goal.reach_s) & (ft < ends[gi])
            if not hold.any():
                continue
            contact, _ = h.graph.manual_contact(torch.tensor([[goal.node]], device=h.device),
                                                torch.tensor([[goal.pose_motion]], device=h.device),
                                                torch.tensor([[goal.pose_time]], device=h.device))
            want = (contact[0, 0].cpu()[ground_ids] > 0.5).numpy()
            reps = []
            for c in cols:
                share = loaded[hold, c].mean(0)                                   # [Z] carrying >= 3 % BW
                touch = touching[hold, c].mean(0)                                 # [Z] down (>= 5 N)
                realised = bool((touch[want] >= REALISED_SHARE).all()) if want.any() else True
                realised_load = bool((share[want] >= REALISED_SHARE).all()) if want.any() else True
                missing = [zone_names[z] for z in np.nonzero(want & (touch < REALISED_SHARE))[0]]
                unwanted = [zone_names[z] for z in np.nonzero(~want & (share >= UNWANTED_SHARE))[0]]
                held24 = float(np.nanmean(e24[hold, c]))
                held6 = float(np.nanmean(err6[hold, c]))
                best24 = float(np.nanmin(e24[window, c]))
                run, best_run, t_in = 0.0, 0.0, None
                for f in np.nonzero(window)[0]:
                    e = e24[f, c]
                    if t_in is None and e <= 0.15:
                        t_in = ft[f]
                    elif t_in is not None and e > 0.30:
                        best_run = max(best_run, ft[f] - t_in)
                        t_in = None
                if t_in is not None:
                    best_run = max(best_run, ft[np.nonzero(window)[0][-1]] - t_in)
                reps.append(dict(err24=round(held24, 4), err6=round(held6, 4), best24=round(best24, 4),
                                 held_s=round(float(best_run), 3), realised=realised, realised_load=realised_load,
                                 missing=missing, unwanted=unwanted,
                                 loaded="+".join(zone_names[z] for z in np.nonzero(share >= 0.5)[0]) or "-",
                                 pelvis_z=round(float(np.median(positions[hold, c, h.pelvis, 2])), 3)))
            n = len(reps)
            hold_ok = np.array([r["err24"] < 0.15 for r in reps])
            real = np.array([r["realised"] for r in reps])
            clean = np.array([not r["unwanted"] for r in reps])
            hist = {}
            for r in reps:
                hist[r["loaded"]] = hist.get(r["loaded"], 0) + 1
            goals_out.append(dict(
                goal=goal.name, node=h.graph.node_keys[goal.node], window_s=[round(ends[gi] - goal.hold_s, 2),
                                                                            round(ends[gi], 2)],
                commanded=[zone_names[z] for z in np.nonzero(want)[0]], n=n,
                hold_rate24=float(hold_ok.mean()), hold_rate6=float(np.mean([r["err6"] < 0.15 for r in reps])),
                reach_rate24=float(np.mean([r["best24"] < 0.15 for r in reps])),
                err24_p50=float(np.median([r["err24"] for r in reps])), err6_p50=float(np.median([r["err6"] for r in reps])),
                held_s_p50=float(np.median([r["held_s"] for r in reps])),
                supports_realised=float(real.mean()),
                supports_loaded=float(np.mean([r["realised_load"] for r in reps])),
                missing_zones={z: sum(z in r["missing"] for r in reps) for z in sorted({m for r in reps
                                                                                       for m in r["missing"]})},
                no_substitution=float(clean.mean()),
                success=float((hold_ok & real & clean).mean()),
                pelvis_z_p50=float(np.median([r["pelvis_z"] for r in reps])),
                loaded_sets=dict(sorted(hist.items(), key=lambda kv: -kv[1])[:4]), replicas=reps))
        scored[seq.name] = goals_out
    (out / "forks_scored.json").write_text(json.dumps(scored, indent=1))
    # the raw material, so any support rule can be re-scored offline
    np.savez_compressed(fork_dir / "zone_loads.npz", zone_load_n=zone_load.astype(np.float16),
                        pelvis_z=positions[:, :, h.pelvis, 2].astype(np.float16), err6=err6.astype(np.float16),
                        frame_times=ft.astype(np.float32), zone_names=np.array(zone_names),
                        sequences=np.array([sq.name for sq in seqs]), num_sequences=np.int32(S))
    return scored


# ---------------------------------------------------------------------------------------------------------------- #
def write_report(out: Path, meta, corpus_sigma, x0_summary, forks):
    lines = [f"# S0 X0 + fork battery: {meta['checkpoint']}", "",
             f"config `{meta['config']}`; {meta['num_envs']} envs; parts {meta['parts']}", ""]
    if x0_summary:
        lines += ["## X0, h = 1 (exact, in-env rebuild)", "",
                  f"Checks: {json.dumps(x0_summary['checks'])}", "",
                  "| hub | r (s) | different p50 (n) | same pose p50 (n) | excess | identical max |",
                  "|---|---|---|---|---|---|"]
        for hub, d in x0_summary["h1"].items():
            for rn, v in d.get("_curve", {}).items():
                f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
                lines.append(f"| {hub} | {rn} | {f(v['different_p50'])} ({v['n_different']}) | "
                             f"{f(v['same_pose_p50'])} ({v['n_same_pose']}) | {f(v['excess_p50'])} | "
                             f"{f(v['identical_max'])} |")
        lines += ["", f"Acceptance: {json.dumps(x0_summary['acceptance'])}", "", "## X0, paired", ""]
        for hub, blocks in x0_summary["paired"].items():
            for b in blocks:
                for hh, v in b["horizons"].items():
                    arms = "; ".join(f"{st}: " + ", ".join(f"{a['da']:.3f}" for a in al) for st, al in v["arms"].items())
                    fl = v["same_goal_floor"]
                    lines.append(f"- {hub} {b['own'][7:47]} r={b['r']} h={hh}: same-goal floor "
                                 f"{'-' if fl is None else f'{fl:.3f}'}; {arms}")
    if forks:
        lines += ["", "## Fork battery", "", "| plan | goal | n | hold (24/6) | supports | no subst. | success | "
                  "held s | pelvis z | loaded |", "|---|---|---|---|---|---|---|---|---|---|"]
        for plan, goals in forks.items():
            for g in goals:
                top = ", ".join(f"{k} {v}" for k, v in list(g["loaded_sets"].items())[:2])
                lines.append(f"| {plan} | {g['goal'][:24]} | {g['n']} | {g['hold_rate24']:.2f} / "
                             f"{g['hold_rate6']:.2f} | {g['supports_realised']:.2f} | {g['no_substitution']:.2f} | "
                             f"{g['success']:.2f} | {g['held_s_p50']:.1f} | {g['pelvis_z_p50']:.2f} | {top} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    torch.manual_seed(args.seed)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else ckpt.parent / "resolved_configs.pt"
    cfg = torch.load(config_path, map_location="cpu", weights_only=False)
    env_cfg = cfg["env"]
    # The four fields the run's inference config changes (resolved_configs_inference.yaml): nothing ends an
    # episode, and every reset lands exactly where this script puts it.
    env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1_000_000
    env_cfg.motion_manager.init_start_prob = 1.0
    if hasattr(env_cfg.motion_manager, "segment_start_prob"):
        env_cfg.motion_manager.segment_start_prob = 0.0
    if hasattr(env_cfg.motion_manager, "segment_end_prob"):
        env_cfg.motion_manager.segment_end_prob = 0.0
    patched = out / "resolved_configs_s0.pt"
    torch.save(cfg, patched)
    args.resolved_configs = str(patched)
    built = build(args, AppLauncher)
    h = Harness(built)
    h.agent.eval()
    # An AMP agent loads a plain PPO checkpoint with strict=False (amp/component.py), which would skip a mismatched
    # actor tensor silently. The actor is the thing under test: prove every one of its tensors is the checkpoint's.
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    live = h.agent.model.state_dict()
    actor_live = [k for k in live if k.startswith("_actor.")]
    bad = [k for k in actor_live if k not in saved or not torch.equal(live[k].cpu(), saved[k].cpu())]
    if not actor_live or bad:
        raise SystemExit(f"the loaded actor is not the checkpoint's: {len(bad)} of {len(actor_live)} tensors differ "
                         f"or are missing, e.g. {bad[:3]}")
    say(f"actor verified: {len(actor_live)} tensors equal the checkpoint's")
    del saved, live
    meta = dict(checkpoint=str(ckpt), checkpoint_sha256=sha256_of(ckpt), config=str(config_path),
                config_sha256=sha256_of(config_path), num_envs=h.E, parts=args.parts, hubs=args.hubs,
                args={k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
                      for k, v in vars(args).items()},
                schedule=dict(include_current_segment=bool(h.cfg.include_current_segment),
                              interval_schedule=bool(getattr(h.cfg, "interval_schedule", False)),
                              dwell_channels=bool(h.cfg.dwell_channels), num_goal_steps=h.K,
                              min_lead_s=float(h.cfg.min_lead_s)),
                motions=h.lib.num_motions(), dt=h.dt, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    x0_summary, forks = None, None
    try:
        with torch.no_grad():
            if "corpus" in args.parts:
                sigma = run_corpus(h, out)
            else:
                sigma = None
            if "x0" in args.parts:
                if sigma is None:
                    prev = out / "corpus.json"
                    if not prev.exists():
                        raise SystemExit("x0 needs sigma_a: run the corpus part first (or keep it in --parts)")
                    sigma = torch.tensor(json.loads(prev.read_text())["sigma_a"])
                res = run_x0(h, out, sigma)
                x0_summary = summarise_x0(res[0], res[2], res[4], res[5], res[6], sigma, res[3], out)
                say(f"acceptance: {json.dumps(x0_summary['acceptance'])}")
            if "forks" in args.parts:
                forks = run_forks(h, out)
    finally:
        write_report(out, meta, None, x0_summary, forks)
        meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        if hasattr(h.env.simulator, "shutdown"):
            h.env.simulator.shutdown()
    say(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
