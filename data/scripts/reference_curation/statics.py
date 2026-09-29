# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated statics (BUILD_PLAN Step 7): what the plant must do to hold a reference pose on the contacts
the reference actually realises, and which of those contacts it cannot do without.

Built on ``data/scripts/static_hold_lp.py``: its MuJoCo plant (74 kg, the MJCF torque limits), its
pose -> qpos decomposition, friction pyramids and point Jacobians. ``solve`` there is replaced, because
it answers a different question (README §3.5).

The problem
-----------
For a pose q (qvel = qacc = 0) and a contact set, find contact forces inside linearised friction cones
(mu 0.75 on the ground, 0.5 between bodies: PhysX's effective values) that balance gravity, and
minimise the peak joint-torque utilisation

    s* = min max_i |tau_i| / tau_max_i,   tau = g(q) - sum_c J_c(q)^T f_c,   root rows of tau = 0.

Body-body forces are internal (f on A, -f on B at one point), so they change joint torques, never the
whole-body balance.

Gates
-----
* **A ground point carries load only inside the contact band**: height <= ``GROUND_BAND_M`` (2 cm, the
  labels' float threshold). The candidates are every box corner, capsule end-cap bottom and sphere
  bottom of the zone (``contact_geometry.geom_ground_patch``); the old solver took the four lowest
  corners whatever their height, balanced Side Crow -c on a right hand 3.9 cm up, and did not change
  when the pose was lifted 0.5 m. A zone with no point in the band is *not realised*.
* **A pair carries load only inside the gap band**: avatar gap <= ``PAIR_BAND_M`` (1 cm, the labels'
  ``realised``). An open pair gets no force columns, so it carries exactly zero load.

Statuses
--------
``support_not_realised``  a requested ground support has no point in the band. It wins over the LP,
                          whose own status and s* over the realised contacts are kept as ``lp_status``
                          and ``realised_s_star``; no contact's necessity is decided on that subset.
``optimal``               the LP solved and passed the finite and residual checks; ``s_star`` may exceed 1
                          (``beyond_plant``: the plant cannot hold the pose with these contacts).
``infeasible``            no forces balance gravity (the COM is outside what the contacts can support).
``iteration_limit``, ``numerical``   the solver stopped, or its answer failed the checks. HiGHS's
                          simplex returns "model status unknown" on a COM that sits on the polygon's
                          edge (Plow -b at its start: 0.4 mm outside); such a solve is retried once with
                          the interior-point method before it is called numerical.

Necessity (the min/max load test)
---------------------------------
For every contact that can carry load: its normal-load interval [min, max] over all static solutions
within the plant's torque limits (s <= 1; for a pose beyond them, min over all balanced solutions and
max within ``PEAK_SLACK`` of s*), and s* with the contact removed (``s_without``):

``required``   removing it leaves no solution within the limits (s_without > 1, equivalently min load > 0:
               the two are LP duals and ``check`` asserts they agree). For a pose beyond the plant
               (s* > 1), removing it leaves no balanced solution at all. The necessity is strict:
               ``load_min_n`` says how much it rests on (Plank -a's left foot at the clip start is
               required with a 0.1 N minimum: without it the peak reaches 1.018).
``useful``     not required, and it relieves the plant's effort: the least summed utilisation
               sum_i |tau_i| / tau_max_i (at a peak within ``PEAK_SLACK`` of the worse of s* and s_without)
               falls by at least ``RELIEF_MIN`` (0.02) when the contact is available. The peak alone
               cannot say this: on Side Crow -c the peak joint is a wrist (20 N m), which no shelf
               relieves, so the peak relief ``relief`` is reported beside it.
``redundant``  neither: the other contacts do the job as well.
``open`` / ``not_realised``   outside its band: no load (a pair) or no support (a ground zone).
``undecided``  the LP was not optimal, or the pose is beyond the plant and the contact is not required.

The loads reported per contact (``load_n``) and the torques the witness feeds forward come from one
representative solution: the least summed utilisation among those whose peak is within
``PEAK_SLACK`` of s*. The min-max optimum alone is degenerate, with arbitrary internal forces.

Joint stops (optional)
----------------------
With ``stops=True`` every hinge within ``STOP_TOL_RAD`` of a range bound (or beyond it) gets a free,
unilateral torque pushing it back into range, and the result reports ``s_star`` with stops, the
relief and the stop torques: how much a solution relies on the stops. Two caveats keep this
diagnostic: 290 of 303 exemplars violate some hinge range of the MuJoCo XYZ decomposition by > 2 deg
(elbow twist most often), and the training plant's limits are soft (the USD's
``physxLimit:rot*:stiffness`` is 300-500).

Counterfactual closure
----------------------
``closed`` names contacts the capture says the human makes. A closed ground zone adds its resting face
(each box's four lowest corners, whatever their height: the old solver's patch) to its in-band points;
a closed open pair gets its columns.
Every such contact and the result are marked ``counterfactual``: what the plant would need if the
reference realised the human's contact (Step 8's retarget), not evidence that the reference does.

The corpus
----------
``audit_holds`` runs every hold of a labels v1 folder at its corrected exemplar: the gated verdict on
the configured contacts (ground set, ``required_touch`` pairs, carried labels), the counterfactual one
when a contact the human makes is not realised or the gated LP cannot hold the pose (and closing adds
something), and the statue witness (``witness.py``) when the gated LP is optimal within the limits.
A hold's ``verdict``: ``held`` (the witness passed), ``feasible`` (the LP holds it, the statue did
not), ``beyond_plant``, ``infeasible``, ``support_not_realised``, or the solver's failure. A contact's
``necessity`` is the gated one when that decides it, else the counterfactual one (``necessity_basis``). ``write`` puts ``holds.jsonl``, ``contacts.jsonl``, ``statics.json`` and
``summary.md`` in ``data/reference_curation/statics/<statics_id>/``.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.statics
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog

import static_hold_lp as S
from contact_geometry import geom_ground_patch, geom_to_world
from reference_curation import audit, ids, verdicts

MODULE = "reference_curation.statics"
SCHEMA_VERSION = 1
STATICS_VERSION = "v1"
STATICS_DIR = ids.DATA_ROOT / "statics"
LABELS_DIR = ids.DATA_ROOT / "labels"

GROUND_BAND_M = audit.HOVER_M                   # 0.02: the labels' float threshold
PAIR_BAND_M = verdicts.PAIR_TOUCH_CM / 100.0    # 0.01: the labels' avatar pair ``realised``
S_CAP = 1.0                                     # the plant's torque limits
RELIEF_MIN = 0.02                               # summed-utilisation relief that makes a contact useful
LOAD_EPS_N = 1.0                                # a stop torque below this is unused
DUAL_TOL_N = 1e-3                               # the necessity tests agree to the solver's tolerance
PEAK_SLACK = 1e-3                               # the representative solution's peak is within this of s*
STOP_TOL_RAD = math.radians(2.0)
LIMIT_REPORT_DEG = 2.0
STATUSES = ("optimal", "infeasible", "numerical", "iteration_limit", "support_not_realised")
NECESSITY = ("required", "useful", "redundant", "open", "not_realised", "undecided")
_HIGHS = {0: "optimal", 1: "iteration_limit", 2: "infeasible", 3: "numerical", 4: "numerical"}
CONFIG = {"ground_band_m": GROUND_BAND_M, "pair_band_m": PAIR_BAND_M, "s_cap": S_CAP, "relief_min": RELIEF_MIN,
          "load_eps_n": LOAD_EPS_N, "peak_slack": PEAK_SLACK, "stop_tol_deg": math.degrees(STOP_TOL_RAD),
          "mu_ground": S.MU_GROUND, "mu_body": S.MU_BODY, "pyramid_rays": 4, "plant": ids.display_path(S.FLAT),
          "mass_kg": round(float(S.M.body_mass.sum()), 3)}

N_HINGE = S.M.njnt - 1
JOINT_RANGE = S.M.jnt_range[1:].copy()          # [69, 2] rad, hinge order = dof order
GROUPS = {"shoulders": ["Shoulder"], "hips": ["Hip"], "spine": ["Torso", "Spine", "Chest"],
          "wrists": ["Wrist", "Hand"], "elbows": ["Elbow"], "knees": ["Knee"], "ankles": ["Ankle", "Toe"]}


def _num(x, nd: int = 3):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


# --------------------------------------------------------------------------- #
# Geometry of the candidate contacts
# --------------------------------------------------------------------------- #
def ground_candidates(pos: np.ndarray, rot: np.ndarray, zone: str) -> list[tuple[str, np.ndarray]]:
    """Every candidate ground point of a zone, ``(body, xyz)``: box corners, capsule end-cap bottoms
    and sphere bottoms of all its bodies' collision geoms."""
    out = []
    for b in S.ZONES[zone]:
        i = S.BODY.index(b)
        for g in S.TYPED[b]:
            wg = geom_to_world(g, torch.as_tensor(pos[i:i + 1]), torch.as_tensor(rot[i:i + 1]))
            out += [(b, p) for p in geom_ground_patch(wg)[0].numpy()]
    return out


def pair_geometry(pos: np.ndarray, rot: np.ndarray, za: str, zb: str) -> dict:
    """The closest body pair of two zones: bodies, witness points, gap (negative = penetration) and the
    unit normal of the force on A (pushing A away from B)."""
    a, pa, b, pb, gap = S.body_pair_point(pos, rot, za, zb)
    fallback = abs(gap) <= 1e-4
    u = (pb - pa) / gap if not fallback else S.D.xpos[S.M.body(b).id] - S.D.xpos[S.M.body(a).id]
    return {"a": a, "b": b, "point": 0.5 * (pa + pb), "gap": float(gap), "normal": -u / np.linalg.norm(u),
            "normal_fallback": bool(fallback)}


def limit_violations(qpos: np.ndarray) -> dict:
    """``{hinge: degrees beyond its MJCF range}`` for violations above ``LIMIT_REPORT_DEG``."""
    q = qpos[7:]
    v = np.degrees(np.maximum(JOINT_RANGE[:, 0] - q, 0) + np.maximum(q - JOINT_RANGE[:, 1], 0))
    return {S.JNAMES[j]: round(float(v[j]), 1) for j in np.nonzero(v > LIMIT_REPORT_DEG)[0]}


# --------------------------------------------------------------------------- #
# The linear programme (pure given a Problem)
# --------------------------------------------------------------------------- #
class Problem:
    """The columns of one pose: ``A [nv, K]`` generalised force of each pyramid ray, ``owner [K]`` the
    contact each ray belongs to, ``bias [nv]`` g(q), and the joint-stop columns ``E [69, n]``."""

    def __init__(self, bias: np.ndarray, cols: list, owner: list, names: list[str], stop_cols=(), stop_names=()):
        self.bias = np.asarray(bias, float)
        self.A = np.array(cols, float).T if cols else np.zeros((len(self.bias), 0))
        self.owner = np.asarray(owner, int)
        self.names = list(names)
        self.E = np.array(stop_cols, float).T if len(stop_cols) else np.zeros((N_HINGE, 0))
        self.stop_names = list(stop_names)
        self.tau_max = S.TAU_MAX.astype(float)
        if not (np.isfinite(self.bias).all() and np.isfinite(self.A).all() and (self.tau_max > 0).all()):
            raise FloatingPointError("non-finite plant quantities")


def solve_lp(prob: Problem, *, drop: frozenset = frozenset(), stops: bool = False, s_cap: float | None = None,
             load_of: int | None = None, sense: int = 1, l1: bool = False) -> tuple[str, dict]:
    """One LP. Variables: ray magnitudes lam >= 0, stop torques sig >= 0 (``stops``), the peak
    utilisation s >= 0, and with ``l1`` the per-joint utilisations u. Objective: min s; or, with
    ``load_of``, min (``sense`` 1) / max (-1) that contact's normal load; or, with ``l1``, min sum u.
    ``s_cap`` bounds s. Contacts in ``drop`` get no columns. Returns ``(status, solution)``; the status
    is ``numerical`` if the answer is not finite or breaks the constraints."""
    keep = np.array([o not in drop for o in prob.owner], bool)
    A = prob.A[:, keep]
    owner = prob.owner[keep]
    K, J = A.shape[1], N_HINGE
    E = prob.E if stops else np.zeros((J, 0))
    n_sig, tmax = E.shape[1], prob.tau_max
    Ar, br, Aj, bj = A[:6], prob.bias[:6], A[6:], prob.bias[6:]
    n_u = J if l1 else 0
    n = K + n_sig + 1 + n_u
    i_s = K + n_sig
    # joint rows: tau = bj - Aj lam - E sig ;  |tau| <= s * tmax  (and <= u * tmax with l1)
    rows, rhs = [], []
    for sign in (1.0, -1.0):
        r = np.zeros((J, n))
        r[:, :K] = -sign * Aj
        r[:, K:K + n_sig] = -sign * E
        r[:, i_s] = -tmax
        rows.append(r)
        rhs.append(-sign * bj)
        if l1:
            r = np.zeros((J, n))
            r[:, :K] = -sign * Aj
            r[:, K:K + n_sig] = -sign * E
            r[:, i_s + 1:] = -np.diag(tmax)
            rows.append(r)
            rhs.append(-sign * bj)
    A_ub, b_ub = np.vstack(rows), np.concatenate(rhs)
    A_eq = np.zeros((6, n))
    A_eq[:, :K] = Ar
    c = np.zeros(n)
    if load_of is not None:
        c[:K] = sense * (owner == load_of)
    elif l1:
        c[i_s + 1:] = 1.0
    else:
        c[i_s] = 1.0
    bounds = [(0, None)] * n
    if s_cap is not None:
        bounds[i_s] = (0, s_cap)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=br, bounds=bounds, method="highs")
    status = _HIGHS.get(res.status, "numerical")
    if status == "numerical":   # simplex gave up (a COM on the polygon's edge): the interior point decides
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=br, bounds=bounds, method="highs-ipm")
        status = _HIGHS.get(res.status, "numerical")
    if status != "optimal":
        return status, {}
    x = res.x
    if not np.isfinite(x).all():
        return "numerical", {}
    lam, sig, s = x[:K], x[K:K + n_sig], float(x[i_s])
    tau = bj - Aj @ lam - E @ sig
    scale = 1.0 + np.abs(prob.bias).max()
    if (np.abs(Ar @ lam - br).max() > 1e-6 * scale or (np.abs(tau) / tmax).max() > s + 1e-6
            or lam.min() < -1e-7 or (s_cap is not None and s > s_cap + 1e-7)):
        return "numerical", {}
    loads = np.zeros(len(prob.names))
    np.add.at(loads, owner, lam)   # each pyramid ray has unit normal component
    return "optimal", {"s": s, "tau": tau, "loads": loads, "stop_torque": sig, "util": np.abs(tau) / tmax}


# --------------------------------------------------------------------------- #
# One pose
# --------------------------------------------------------------------------- #
def bottom_face(pos: np.ndarray, rot: np.ndarray, zone: str) -> list[tuple[str, np.ndarray]]:
    """The zone's resting face whatever its height: each box's four lowest corners, capsule end-cap
    bottoms, sphere bottoms (``static_hold_lp.ground_points``, the old solver's what-if-closed patch)."""
    return S.ground_points(pos, rot, zone)


def build_problem(pos: np.ndarray, rot: np.ndarray, ground=(), pairs=(), closed=(), stops: bool = False):
    """``(problem, contacts, qpos)``: sets the plant to the pose, gates every requested contact and
    builds the columns of those that carry load."""
    pos, rot = np.asarray(pos, float), np.asarray(rot, float)
    if not (np.isfinite(pos).all() and np.isfinite(rot).all()):
        raise FloatingPointError("non-finite pose")
    if np.abs(np.linalg.norm(rot, axis=-1) - 1).max() > 1e-3:
        raise ValueError("rotations are not unit quaternions")
    bias = S.set_pose(pos, rot)
    qpos = S.D.qpos.copy()
    closed = set(closed)
    contacts, cols, owner, names = {}, [], [], []

    def add(name, columns):
        owner.extend([len(names)] * len(columns))
        cols.extend(columns)
        names.append(name)

    for z in ground:
        name = f"{z}:G"
        cand = ground_candidates(pos, rot, z)
        low = min(float(p[2]) for _, p in cand)
        take = [(b, p) for b, p in cand if p[2] <= GROUND_BAND_M]
        cf = False
        if name in closed:   # counterfactual: the resting face as if the reference realised the touch
            face = [(b, p) for b, p in bottom_face(pos, rot, z) if p[2] > GROUND_BAND_M]
            cf = bool(face)
            take += face
        realised = low <= GROUND_BAND_M
        contacts[name] = {"kind": "ground", "zones": [z], "realised": bool(realised), "counterfactual": cf,
                          "loaded": bool(take), "height_cm": _num(100 * low, 2), "points": len(take)}
        if take:
            e_n = S.pyramid(np.array([0, 0, 1.0]), S.MU_GROUND)
            add(name, [S.jac_point(b, p).T @ e for b, p in take for e in e_n])
    for name in pairs:
        za, zb = name.split("+")
        g = pair_geometry(pos, rot, za, zb)
        realised = g["gap"] <= PAIR_BAND_M
        cf = not realised and name in closed
        contacts[name] = {"kind": "pair", "zones": [za, zb], "realised": bool(realised), "counterfactual": bool(cf),
                          "loaded": bool(realised or cf), "gap_cm": _num(100 * g["gap"], 2), "bodies": [g["a"], g["b"]],
                          "normal_fallback": g["normal_fallback"]}
        if realised or cf:
            Ja, Jb = S.jac_point(g["a"], g["point"]), S.jac_point(g["b"], g["point"])
            add(name, [Ja.T @ e - Jb.T @ e for e in S.pyramid(g["normal"], S.MU_BODY)])
    stop_cols, stop_names = [], []
    if stops:
        q = qpos[7:]
        for j in range(N_HINGE):
            for bound, sign in ((0, 1.0), (1, -1.0)):
                if sign * (q[j] - JOINT_RANGE[j, bound]) <= STOP_TOL_RAD:   # at or beyond this bound
                    col = np.zeros(N_HINGE)
                    col[j] = sign
                    stop_cols.append(col)
                    stop_names.append(f"{S.JNAMES[j]}:{'low' if bound == 0 else 'high'}")
    return Problem(bias, cols, owner, names, stop_cols, stop_names), contacts, qpos


def _top_joints(util: np.ndarray, tau: np.ndarray, k: int = 6) -> list:
    order = np.argsort(-util)[:k]
    return [[S.JNAMES[i], round(float(util[i]), 3), round(float(tau[i]), 1)] for i in order]


def _group_util(util: np.ndarray) -> dict:
    return {g: round(float(max(util[i] for i, n in enumerate(S.JNAMES) if any(k in n for k in ks))), 3)
            for g, ks in GROUPS.items()}


def effort(prob: Problem, cap: float, drop: frozenset = frozenset()) -> float | None:
    """The least summed utilisation sum_i |tau_i| / tau_max_i among solutions with peak <= ``cap``."""
    st, sol = solve_lp(prob, drop=drop, s_cap=cap, l1=True)
    return float(sol["util"].sum()) if st == "optimal" else None


def necessity(prob: Problem, s_star: float, i: int) -> dict:
    """The min/max load test of loaded contact ``i`` (see the module docstring)."""
    st_wo, wo = solve_lp(prob, drop=frozenset({i}))
    out = {"s_without": _num(wo["s"], 4) if st_wo == "optimal" else None, "infeasible_without": st_wo == "infeasible",
           "relief": None, "effort_relief": None, "load_min_n": None, "load_max_n": None}
    if st_wo not in ("optimal", "infeasible"):
        out["necessity"] = "undecided"
        return out
    s_without = wo["s"] if st_wo == "optimal" else math.inf
    within = s_star <= S_CAP
    required = s_without > S_CAP if within else math.isinf(s_without)
    st_lo, lo = solve_lp(prob, s_cap=S_CAP if within else None, load_of=i, sense=1)
    st_hi, hi = solve_lp(prob, s_cap=max(S_CAP, s_star * (1 + PEAK_SLACK)), load_of=i, sense=-1)
    if st_lo == "optimal":
        out["load_min_n"] = _num(lo["loads"][i], 1)
        if required != (lo["loads"][i] > DUAL_TOL_N):   # LP duals: a disagreement is solver tolerance
            out["dual_mismatch"] = True
    if st_hi == "optimal":
        out["load_max_n"] = _num(hi["loads"][i], 1)
    if required:
        out["necessity"] = "required"
        return out
    out["relief"] = _num(s_without - s_star, 4)
    e_cap = max(s_star, s_without) * (1 + PEAK_SLACK) + 1e-9
    e_with, e_without = effort(prob, e_cap), effort(prob, e_cap, frozenset({i}))
    if e_with is None or e_without is None:
        out["necessity"] = "undecided"
        return out
    out["effort_relief"] = _num(e_without - e_with, 4)
    out["necessity"] = "useful" if e_without - e_with >= RELIEF_MIN else "redundant"
    return out


def analyse(pos, rot, ground=(), pairs=(), closed=(), stops: bool = False, with_necessity: bool = True) -> dict:
    """The gated statics of one pose (see the module docstring). ``tau`` (the representative joint
    torques, [69]) is returned for the witness and is not JSON."""
    prob, contacts, qpos = build_problem(pos, rot, ground, pairs, closed, stops=stops)
    unrealised = [n for n, c in contacts.items() if c["kind"] == "ground" and not c["realised"]]
    missing = [n for n in unrealised if not contacts[n]["counterfactual"]]
    open_pairs = [n for n, c in contacts.items() if c["kind"] == "pair" and not c["realised"]
                  and not c["counterfactual"]]
    status, sol = solve_lp(prob)
    decided = status == "optimal" and not missing
    out = {"status": "support_not_realised" if missing else status, "lp_status": status,
           "s_star": _num(sol["s"], 4) if decided else None, "beyond_plant": bool(decided and sol["s"] > S_CAP),
           "realised_s_star": _num(sol["s"], 4) if missing and sol else None,
           "counterfactual": any(c["counterfactual"] for c in contacts.values()),
           "unrealised": unrealised, "open": open_pairs, "contacts": contacts,
           "limit_violations_deg": limit_violations(qpos), "tau": None}
    for name, c in contacts.items():
        c.update(load_n=0.0 if c["kind"] == "pair" and not c["loaded"] else None, necessity=None)
        if not c["loaded"]:
            c["necessity"] = "open" if c["kind"] == "pair" else "not_realised"
    if not decided:   # a partial contact set says nothing about any one contact's necessity
        for c in contacts.values():
            if c["loaded"]:
                c["necessity"] = "undecided"
        return out
    s_star = sol["s"]
    st_rep, rep = solve_lp(prob, s_cap=s_star * (1 + PEAK_SLACK) + 1e-9, l1=True)
    rep = rep if st_rep == "optimal" else sol
    out.update(tau=rep["tau"], top_joints=_top_joints(rep["util"], rep["tau"]), group_util=_group_util(rep["util"]),
               representative=st_rep)
    for i, name in enumerate(prob.names):
        c = contacts[name]
        c["load_n"] = _num(rep["loads"][i], 1)
        if with_necessity:
            c.update(necessity(prob, s_star, i))
    if stops:
        st_st, st = solve_lp(prob, stops=True)
        used = []
        if st_st == "optimal":
            used = sorted(([prob.stop_names[k], round(float(t), 1)] for k, t in enumerate(st["stop_torque"])
                           if t > LOAD_EPS_N), key=lambda x: -x[1])
        out["stops"] = {"status": st_st, "candidates": len(prob.stop_names),
                        "s_star": _num(st["s"], 4) if st_st == "optimal" else None,
                        "relief": _num(s_star - st["s"], 4) if st_st == "optimal" else None, "torques": used}
    return out


def load_pose(stem: str, frame: int, motion_dir: Path = ids.SHIPPED_DIR) -> tuple[np.ndarray, np.ndarray]:
    mot = _load_motion(str(ids.motion_path(stem, motion_dir)))
    T = mot["rigid_body_pos"].shape[0]
    if not 0 <= frame < T:
        raise IndexError(f"{stem}: frame {frame} outside 0-{T - 1}")
    return mot["rigid_body_pos"][frame].double().numpy(), mot["rigid_body_rot"][frame].double().numpy()


_MOTIONS: dict = {}


def _load_motion(path: str) -> dict:
    if path not in _MOTIONS:
        if len(_MOTIONS) > 4:
            _MOTIONS.clear()
        _MOTIONS[path] = torch.load(path, map_location="cpu", weights_only=False)
    return _MOTIONS[path]


# --------------------------------------------------------------------------- #
# The corpus: every hold of a labels v1 folder
# --------------------------------------------------------------------------- #
def default_labels_dir(manifest: Path = ids.DEFAULT_MANIFEST) -> Path:
    """The labels v1 folder of ``manifest`` under ``data/reference_curation/labels`` (exactly one)."""
    dirs = sorted(LABELS_DIR.glob(f"{Path(manifest).stem}.labels_v1.*"))
    if len(dirs) != 1:
        raise FileNotFoundError(f"expected one labels v1 folder for {Path(manifest).stem}, found {len(dirs)}")
    return dirs[0]


def load_labels(labels_dir: Path) -> dict:
    """``{"dir", "id", "clips": [(stem, [hold])], "anns": {hold_id: [annotation]}}``."""
    labels_dir = Path(labels_dir)
    manifest = ids.load_manifest(labels_dir / "holds.yaml")
    anns = collections.defaultdict(list)
    with open(labels_dir / "annotations.jsonl") as f:
        for line in f:
            a = json.loads(line)
            anns[a["hold_id"]].append(a)
    return {"dir": labels_dir, "id": manifest["labels"]["labels_id"],
            "clips": [(c["stem"], c["holds"]) for c in manifest["clips"]], "anns": dict(anns)}


def hold_request(anns: list[dict]) -> dict:
    """What one hold asks of the plant: its configured ground zones and pairs (ground set,
    ``required_touch`` pairs, carried labels), and those the human is seen making (``closed``)."""
    conf = [a for a in anns if a["in_configuration"]]
    return {"ground": [a["zones"][0] for a in conf if a["kind"] == "ground"],
            "pairs": [a["contact"] for a in conf if a["kind"] == "pair"],
            "closed": [a["contact"] for a in conf if a["source_state"] == "observed_contact"]}


def _public(result: dict | None) -> dict | None:
    return None if result is None else {k: v for k, v in result.items() if k != "tau"}


def audit_hold(stem: str, hold: dict, anns: list[dict], motion_dir: Path = ids.SHIPPED_DIR,
               run_witness: bool = True) -> dict:
    """One hold at its corrected exemplar: the gated verdict, the counterfactual one when a contact the
    human makes is not realised, and the statue witness when the gated LP is optimal within the limits."""
    from reference_curation import witness

    req = hold_request(anns)
    frame = int(hold["frame_hold"])
    pos, rot = load_pose(stem, frame, motion_dir)
    gated = analyse(pos, rot, req["ground"], req["pairs"], stops=True)
    missing = [n for n in req["closed"] if not gated["contacts"][n]["realised"]]
    cf = None
    if missing or gated["status"] != "optimal" or gated["beyond_plant"]:
        cf = analyse(pos, rot, req["ground"], req["pairs"], closed=req["closed"])
        cf = cf if cf["counterfactual"] else None   # closing added nothing: it is the gated verdict
    wit = None
    if run_witness and gated["status"] == "optimal" and not gated["beyond_plant"]:
        wit = witness.run(pos, rot, gated["tau"], [n for n in req["pairs"] if gated["contacts"][n]["realised"]])
    status = gated["status"]
    verdict = ("held" if wit and wit["passed"] else "feasible") if status == "optimal" and not gated["beyond_plant"] \
        else ("beyond_plant" if status == "optimal" else status)
    return {"hold_id": hold["hold_id"], "stem": stem, "hold_name": hold["name"], "family_hold": bool(hold.get("extend")),
            "frame": frame, "t": hold["t_hold"],
            "request": req, "not_realised_human_contacts": missing, "verdict": verdict,
            "gated": _public(gated), "counterfactual": _public(cf), "witness": wit}


def _worker(job: tuple) -> tuple[list[dict], list[str]]:
    stem, holds, anns, motion_dir, run_witness = job
    out, failures = [], []
    for h in holds:
        try:
            out.append(audit_hold(stem, h, anns[h["hold_id"]], Path(motion_dir), run_witness))
        except Exception as exc:  # noqa: BLE001 -- report every broken hold, then fail
            failures.append(f"{h['hold_id']}: {type(exc).__name__}: {exc}")
    return out, failures


def audit_holds(labels: dict, motion_dir: Path = ids.SHIPPED_DIR, stems: list[str] | None = None,
                workers: int = 1, run_witness: bool = True) -> tuple[list[dict], list[str]]:
    """Every hold of ``labels`` (of ``stems``, if given), in the labels' order."""
    jobs = [(stem, holds, {h["hold_id"]: labels["anns"][h["hold_id"]] for h in holds}, str(motion_dir), run_witness)
            for stem, holds in labels["clips"] if stems is None or stem in stems]
    if workers > 1:
        from reference_curation.human_mesh import _single_threaded_children
        with _single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            parts = list(ex.map(_worker, jobs))
    else:
        parts = [_worker(j) for j in jobs]
    return [r for rs, _ in parts for r in rs], [f for _, fs in parts for f in fs]


def contact_rows(records: list[dict], labels: dict) -> list[dict]:
    """One row per (hold, configured contact): the labels' role and evidence beside the gated and the
    counterfactual statics."""
    keys = ("realised", "counterfactual", "height_cm", "gap_cm", "points", "load_n", "load_min_n", "load_max_n",
            "s_without", "relief", "effort_relief", "necessity")
    rows = []
    for r in records:
        anns = {a["contact"]: a for a in labels["anns"][r["hold_id"]]}
        for name, g in r["gated"]["contacts"].items():
            a = anns[name]
            cf = (r["counterfactual"] or {}).get("contacts", {}).get(name)
            rows.append({"hold_id": r["hold_id"], "contact": name, "kind": g["kind"], "zones": g["zones"],
                         "family_hold": r["family_hold"], "target_role": a["target_role"],
                         "review_role": (a["review"] or {}).get("role"), "load_path_class": a["load_path_class"],
                         "source_state": a["source_state"], "label_action": a["label_action"],
                         "hold_status": r["gated"]["status"], "gated": {k: g.get(k) for k in keys},
                         "counterfactual": None if cf is None else {k: cf.get(k) for k in keys},
                         "necessity": g["necessity"] if g["necessity"] not in ("not_realised", "open", "undecided")
                         or cf is None else cf["necessity"],
                         "necessity_basis": "gated" if g["necessity"] not in ("not_realised", "open", "undecided")
                         or cf is None else "counterfactual"})
    return rows


# --------------------------------------------------------------------------- #
# Checks, metrics and output
# --------------------------------------------------------------------------- #
VERDICTS = ("held", "feasible", "beyond_plant", "infeasible", "support_not_realised", "numerical", "iteration_limit")


def check(records: list[dict]) -> list[str]:
    """Every violation of the statics contract: unknown statuses or classes, a load outside its band,
    an unrealised support without its status, a necessity whose two LP tests disagree, non-JSON numbers."""
    problems = []
    for r in records:
        hid, g = r["hold_id"], r["gated"]
        if g["status"] not in STATUSES or r["verdict"] not in VERDICTS:
            problems.append(f"{hid}: unknown status {g['status']} / verdict {r['verdict']}")
        missing = [n for n, c in g["contacts"].items() if c["kind"] == "ground" and not c["realised"]]
        if bool(missing) != (g["status"] == "support_not_realised"):
            problems.append(f"{hid}: support_not_realised must mean an unrealised ground support")
        for which in ("gated", "counterfactual"):
            res = r[which]
            if res is None:
                continue
            for n, c in res["contacts"].items():
                if c["necessity"] not in NECESSITY:
                    problems.append(f"{hid} {which} {n}: unknown necessity {c['necessity']}")
                if c.get("dual_mismatch"):
                    problems.append(f"{hid} {which} {n}: min load and s_without disagree on necessity")
                if not c["realised"] and not c["counterfactual"] and c["loaded"]:
                    problems.append(f"{hid} {which} {n}: loaded outside its band")
                if c["kind"] == "pair" and not c["loaded"] and c["load_n"] != 0.0:
                    problems.append(f"{hid} {which} {n}: an open pair must carry zero load")
            if which == "gated" and res["counterfactual"]:
                problems.append(f"{hid}: the gated verdict is counterfactual")
            if which == "counterfactual" and not res["counterfactual"]:
                problems.append(f"{hid}: a counterfactual verdict that closes nothing")
        try:
            json.dumps(r, allow_nan=False)
        except ValueError as exc:
            problems.append(f"{hid}: {exc}")
    return problems


def metrics(records: list[dict], rows: list[dict]) -> dict:
    def block(rs):
        hid = {r["hold_id"] for r in rs}
        mine = [x for x in rows if x["hold_id"] in hid]
        wit = [r for r in rs if r["witness"]]
        pairs = [x for x in mine if x["kind"] == "pair"]
        ground = [x for x in mine if x["kind"] == "ground"]
        stops = [r["gated"]["stops"]["relief"] for r in rs if (r["gated"].get("stops") or {}).get("relief") is not None]
        return {
            "holds": len(rs), "verdict": dict(collections.Counter(r["verdict"] for r in rs)),
            "counterfactual": dict(collections.Counter(r["counterfactual"]["status"] + (
                "_beyond_plant" if r["counterfactual"]["beyond_plant"] else "") for r in rs if r["counterfactual"])),
            "witness_run": len(wit), "witness_passed": sum(r["witness"]["passed"] for r in wit),
            "pairs_by_role": {role: dict(collections.Counter(f"{x['necessity']}:{x['necessity_basis']}"
                                                             for x in pairs if x["target_role"] == role))
                              for role in sorted({x["target_role"] for x in pairs})},
            "ground_by_role": {role: dict(collections.Counter(f"{x['necessity']}:{x['necessity_basis']}"
                                                              for x in ground if x["target_role"] == role))
                               for role in sorted({x["target_role"] for x in ground})},
            "stops_relief_ge_0.02": sum(x >= RELIEF_MIN for x in stops), "stops_evaluated": len(stops),
            "limit_violations_holds": sum(bool(r["gated"]["limit_violations_deg"]) for r in rs),
        }
    return {"all": block(records), "family": block([r for r in records if r["family_hold"]])}


def statics_id(labels: dict, records: list[dict], motion_dir: Path) -> str:
    from reference_curation import witness

    stems = sorted({r["stem"] for r in records})
    key = {"schema": SCHEMA_VERSION, "config": CONFIG, "witness": witness.CONFIG, "labels": labels["id"],
           "generators": {Path(f).name: ids.sha256_file(f) for f in _generator_files()},
           "motions": {s: ids.sha256_file(ids.motion_path(s, motion_dir)) for s in stems},
           "holds": sorted(r["hold_id"] for r in records)}
    return f"{labels['id']}.statics_{STATICS_VERSION}.{ids.sha256_json(key)[:10]}"


def _generator_files() -> list[Path]:
    from reference_curation import witness
    import protomotions.robot_configs.smpl_yogi as yogi

    return [Path(__file__), Path(witness.__file__), Path(S.__file__), Path(yogi.__file__), S.FLAT, ids.MJCF]


def summary_markdown(sid: str, m: dict, records: list[dict], rows: list[dict]) -> str:
    a, f = m["all"], m["family"]
    lines = [f"# Statics `{sid}`", "",
             "Generated by `reference_curation.statics` (BUILD_PLAN Step 7); the rules are in its docstring and in "
             "`witness.py`'s. Gated = the reference as it is; counterfactual = with the human's contacts closed "
             "(what Step 8's retarget would need), never evidence about the reference.", "",
             "| Metric | All holds | Family holds |", "|---|---|---|", f"| Holds | {a['holds']} | {f['holds']} |"]
    lines += [f"| Verdict `{v}` | {a['verdict'].get(v, 0)} | {f['verdict'].get(v, 0)} |" for v in VERDICTS]
    lines += [f"| Counterfactual {k} | {v} | {f['counterfactual'].get(k, 0)} |" for k, v in sorted(a["counterfactual"].items())]
    lines += [f"| Witness passed / run | {a['witness_passed']} / {a['witness_run']} | "
              f"{f['witness_passed']} / {f['witness_run']} |",
              f"| Holds whose stops relieve s* by >= {RELIEF_MIN} | {a['stops_relief_ge_0.02']} / {a['stops_evaluated']} | "
              f"{f['stops_relief_ge_0.02']} / {f['stops_evaluated']} |",
              f"| Holds violating a hinge range by > {LIMIT_REPORT_DEG:g} deg | {a['limit_violations_holds']} | "
              f"{f['limit_violations_holds']} |", ""]
    for title, key in (("Body-body contacts", "pairs_by_role"), ("Ground supports", "ground_by_role")):
        lines += [f"## {title}: necessity by labels role (`class:basis`)", "", "| Role | Necessity |", "|---|---|"]
        lines += [f"| `{role}` | {', '.join(f'{k} {v}' for k, v in sorted(c.items()))} |" for role, c in a[key].items()]
        lines.append("")
    lines += ["## Held by the statue witness", ""]
    lines += [f"- `{r['hold_id']}` {r['hold_name']}: settle {r['witness']['settle_cm']} cm, drift "
              f"{r['witness']['drift_cm']} cm, s* {r['gated']['s_star']}" for r in records if r["verdict"] == "held"]
    lines += ["", "## Family holds", "", "| Hold | Name | Verdict | s* gated | s* counterfactual | Not realised | Witness |",
              "|---|---|---|---|---|---|---|"]
    for r in records:
        if not r["family_hold"]:
            continue
        w = r["witness"]
        wtxt = "" if w is None else ("pass" if w["passed"] else f"fail (settle {w['settle_cm']}, drift {w['drift_cm']}, "
                                     f"lifted {w['lifted'] or '-'}, landed {w['landed'] or '-'})")
        cf = r["counterfactual"]
        lines.append(f"| `{r['hold_id']}` | {r['hold_name']} | {r['verdict']} | {r['gated']['s_star']} | "
                     f"{'' if cf is None else cf['s_star'] if cf['status'] == 'optimal' else cf['status']} | "
                     f"{', '.join(r['gated']['unrealised'] + r['gated']['open']) or '-'} | {wtxt} |")
    worklist = [x for x in rows if x["kind"] == "pair" and x["target_role"] == "required_touch"]
    lines += ["", f"## Worklist: the {len(worklist)} `required_touch` pairs", "",
              "| Hold | Pair | Statics | Basis | Min-max load (N) | Relief (peak / effort) | Reviewer role |",
              "|---|---|---|---|---|---|---|"]
    for x in worklist:
        v = x[x["necessity_basis"]] or {}
        lines.append(f"| `{x['hold_id']}` | {x['contact']} | {x['necessity']} | {x['necessity_basis']} | "
                     f"{v.get('load_min_n')}-{v.get('load_max_n')} | {v.get('relief')} / {v.get('effort_relief')} | "
                     f"{x['review_role']} |")
    return "\n".join(lines) + "\n"


def write(labels: dict, records: list[dict], motion_dir: Path, out_root: Path = STATICS_DIR) -> Path:
    from reference_curation import witness

    rows = contact_rows(records, labels)
    sid = statics_id(labels, records, motion_dir)
    out = Path(out_root) / sid
    out.mkdir(parents=True, exist_ok=True)
    head = {"schema_version": SCHEMA_VERSION, "statics_id": sid}
    (out / "holds.jsonl").write_text("".join(json.dumps({**head, **r}, allow_nan=False) + "\n" for r in records))
    (out / "contacts.jsonl").write_text("".join(json.dumps({**head, **x}, allow_nan=False) + "\n" for x in rows))
    m = metrics(records, rows)
    inputs = [labels["dir"] / "holds.yaml", labels["dir"] / "annotations.jsonl", S.FLAT, ids.MJCF]
    inputs += [ids.motion_path(s, motion_dir) for s in sorted({r["stem"] for r in records})]
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), "statics_id": sid, "labels_id": labels["id"],
              "motion_dir": ids.display_path(motion_dir), "config": CONFIG, "witness": witness.CONFIG,
              "generators": {ids.display_path(p): ids.sha256_file(p) for p in _generator_files()}, "metrics": m}
    (out / "statics.json").write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    (out / "summary.md").write_text(summary_markdown(sid, m, records, rows))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, help="a labels v1 folder (default: the manifest's one)")
    ap.add_argument("--motion-dir", type=Path, default=ids.SHIPPED_DIR, help="the reference motions to judge")
    ap.add_argument("--stem", nargs="*", help="only these clips")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--no-witness", action="store_true")
    ap.add_argument("--out-root", type=Path, default=STATICS_DIR)
    args = ap.parse_args(argv)
    start = time.time()
    try:
        labels = load_labels(args.labels or default_labels_dir())
        records, failures = audit_holds(labels, args.motion_dir, args.stem or None, args.workers, not args.no_witness)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    failures += check(records)
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if failures or not records:
        print(f"statics: {len(failures)} failures, {len(records)} holds; nothing written", file=sys.stderr)
        return 1
    out = write(labels, records, args.motion_dir, args.out_root)
    a = metrics(records, contact_rows(records, labels))["all"]
    print(f"statics {out.name}: {a['holds']} holds, verdicts {a['verdict']}; counterfactual {a['counterfactual']}; "
          f"witness {a['witness_passed']}/{a['witness_run']} held; in {time.time() - start:.0f} s -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
