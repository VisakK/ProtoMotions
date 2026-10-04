"""The kinematic side of an edge: its endpoints on one floor, a sketch between them, and its contact schedule.

* **Endpoints.** S and D are release v2 exemplars (``edges.json``), each in its own clip's world frame. D is moved
  rigidly -- a yaw about z and a horizontal shift, never a tilt or a lift (both stand on the same floor) -- so that
  its two hands land on S's (``hand_anchor_transform``: the 2-point Procrustes fit of D's ``L_Hand``/``R_Hand``
  origins onto S's). The hand widths differ by at most 3.5 cm between the endpoints of edges.json, so each hand ends
  within ~2 cm of its source position; the planted-hand cost keeps the hands where S has them, and T2's IK edit
  closes the rest.
* **Sketch.** Keyframes ``(t, qpos)`` from S (t = 0) through any intermediate keyframes to D (t = T) are joined with
  minimum-jerk timing per segment: the root position by the quintic smoothstep, the root rotation and every joint's
  local rotation by slerp on the same clock. Hinge angles are taken branch-continuous along the sketch
  (``Plant.hinge_from_local`` with ``prev``), so PD targets never jump by pi. Before 0 the sketch is S, after T it is D.
* **Schedule.** The edge's phases with chosen durations (``Timing``: ``low``/``mid``/``high`` of each phase's
  admissible range, or explicit seconds) give, at any time, the planted zones, the closed braces, the zones that must
  stay off the floor and the event windows in which a making zone may touch down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from extract_contact_configs import ZONE_ORDER
from edge_synthesis import plant_mj as pm
from reference_curation import ids

EDGES_JSON = ids.REPO / "expert_revist/graph_growth_2026_10_03/edges.json"
EVENT_WINDOW_S = 0.15          # a making zone may touch down this long either side of its scheduled event


def load_edges(path: Path = EDGES_JSON) -> dict:
    return json.load(open(path))


def edge(spec: dict, edge_id: str) -> dict:
    for e in spec["edges"]:
        if e["id"] == edge_id:
            return e
    raise KeyError(edge_id)


def exemplar(ep: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(pos [24,3], rot [24,4] xyzw, dof [69])`` of an endpoint's exemplar frame."""
    return pm.release_frame(ep["stem"], int(ep["frame_hold"]))


def hand_anchor_transform(src_pos: np.ndarray, dst_pos: np.ndarray, body_index: dict) -> tuple[float, np.ndarray]:
    """``(yaw, t_xy)``: rotating D by ``yaw`` about z and shifting by ``t_xy`` puts its hand midpoint on S's and its
    hand line along S's."""
    a1, a2 = src_pos[body_index["L_Hand"], :2], src_pos[body_index["R_Hand"], :2]
    b1, b2 = dst_pos[body_index["L_Hand"], :2], dst_pos[body_index["R_Hand"], :2]
    ea, eb = a2 - a1, b2 - b1
    yaw = float(np.arctan2(eb[0] * ea[1] - eb[1] * ea[0], eb @ ea))
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    t = 0.5 * (a1 + a2) - R @ (0.5 * (b1 + b2))
    return yaw, t


def apply_planar(pos: np.ndarray, rot_xyzw: np.ndarray, yaw: float, t_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Rz = Rotation.from_euler("z", yaw)
    p = Rz.apply(pos.reshape(-1, 3)).reshape(pos.shape)
    p[..., :2] += t_xy
    r = (Rz * Rotation.from_quat(rot_xyzw.reshape(-1, 4))).as_quat().reshape(rot_xyzw.shape)
    return p, r


@dataclass
class Endpoints:
    """S and the re-anchored D, as bodies and hinge ``qpos``."""
    src_pos: np.ndarray
    src_rot: np.ndarray
    src_qpos: np.ndarray
    dst_pos: np.ndarray
    dst_rot: np.ndarray
    dst_qpos: np.ndarray
    yaw: float
    t_xy: np.ndarray
    hand_residual_m: dict


def endpoints(plant: pm.Plant, e: dict) -> Endpoints:
    sp, sr, _ = exemplar(e["source"])
    dp, dr, _ = exemplar(e["destination"])
    yaw, t = hand_anchor_transform(sp, dp, plant.body_index)
    dp2, dr2 = apply_planar(dp, dr, yaw, t)
    bi = plant.body_index
    res = {b: round(float(np.linalg.norm(dp2[bi[b], :2] - sp[bi[b], :2])), 4) for b in ("L_Hand", "R_Hand", "L_Wrist",
                                                                                        "R_Wrist")}
    sq = plant.qpos_from_bodies(sp, sr)
    dq = plant.qpos_from_bodies(dp2, dr2)
    return Endpoints(sp, sr, sq, dp2, dr2, dq, yaw, t, res)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.clip(x * x * x * (10 - 15 * x + 6 * x * x), 0.0, 1.0)


class Sketch:
    """Keyframes ``[(t, qpos [76])]`` (t ascending) joined by minimum-jerk slerp; ``qpos(t)`` for any times."""

    def __init__(self, plant: pm.Plant, keyframes: list[tuple[float, np.ndarray]], anchor_bodies=("L_Hand", "R_Hand"),
                 anchor_point: np.ndarray | None = None):
        """``anchor_bodies``: between keyframes the root is placed so these bodies' midpoint stays at
        ``anchor_point`` (default: their midpoint in the first keyframe) -- a slerp of the joints alone would drag
        planted hands across the floor. ``None`` interpolates the root position instead."""
        self.plant = plant
        self.anchor_idx = None if anchor_bodies is None else [plant.body_index[b] for b in anchor_bodies]
        self.times = np.array([k[0] for k in keyframes], float)
        self.q = np.stack([np.asarray(k[1], float) for k in keyframes])
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("keyframe times must increase")
        self.root_rot = [Rotation.from_quat(np.r_[q[4:7], q[3]]) for q in self.q]
        self.local = [plant.local_rotations(q) for q in self.q]          # [23, 3, 3] each
        if self.anchor_idx is not None:
            p0, _ = plant.fk(self.q[0][None])
            self.anchor_point = p0[0, self.anchor_idx].mean(0) if anchor_point is None else np.asarray(anchor_point)
        self._grid_t = None

    @property
    def T(self) -> float:
        return float(self.times[-1])

    def _eval(self, t: float, prev: np.ndarray | None) -> np.ndarray:
        """The sketch at ``t``; outside the keyframes it holds the end keyframe's rotations, and the hinge branch is
        always the one continuous with ``prev`` (the end keyframes' own range-based branch may differ)."""
        if len(self.times) == 1:
            return self.q[0].copy()
        if t <= self.times[0]:
            k, s = 0, 0.0
        elif t >= self.times[-1]:
            k, s = len(self.times) - 2, 1.0
        else:
            k = int(np.searchsorted(self.times, t, side="right") - 1)
            s = float(_smoothstep(np.array((t - self.times[k]) / (self.times[k + 1] - self.times[k]))))
        q0, q1 = self.q[k], self.q[k + 1]
        out = np.empty_like(q0)
        out[:3] = (1 - s) * q0[:3] + s * q1[:3]
        rr = Slerp([0, 1], Rotation.concatenate([self.root_rot[k], self.root_rot[k + 1]]))([s])[0].as_quat()
        out[3:7] = [rr[3], rr[0], rr[1], rr[2]]
        L0, L1 = Rotation.from_matrix(self.local[k]), Rotation.from_matrix(self.local[k + 1])
        rel = (L0.inv() * L1).as_rotvec()
        Ls = (L0 * Rotation.from_rotvec(s * rel)).as_matrix()
        ref = prev[7:] if prev is not None else (1 - s) * q0[7:] + s * q1[7:]
        out[7:] = self.plant.hinge_from_local(Ls[None], ref[None])[0]
        if self.anchor_idx is not None:
            z = out.copy()
            z[:3] = 0.0
            p, _ = self.plant.fk(z[None])
            out[:3] = self.anchor_point - p[0, self.anchor_idx].mean(0)
        return out

    def build_grid(self, fps: float = 240.0, t0: float = -2.0, t1: float | None = None):
        """Precompute the sketch on a dense grid (branch-continuous); ``qpos`` interpolates it."""
        t1 = self.T + 5.0 if t1 is None else t1
        ts = np.arange(t0, t1 + 1e-9, 1.0 / fps)
        qs = np.empty((len(ts), self.q.shape[1]))
        prev = None
        for i, t in enumerate(ts):
            qs[i] = self._eval(t, prev)
            prev = qs[i]
        self._grid_t, self._grid_q = ts, qs

    def qpos(self, t: np.ndarray) -> np.ndarray:
        if self._grid_t is None:
            self.build_grid()
        t = np.clip(np.asarray(t, float), self._grid_t[0], self._grid_t[-1])
        i = np.clip(np.searchsorted(self._grid_t, t) - 1, 0, len(self._grid_t) - 2)
        a = ((t - self._grid_t[i]) / (self._grid_t[i + 1] - self._grid_t[i]))[..., None]
        out = (1 - a) * self._grid_q[i] + a * self._grid_q[i + 1]
        q = out[..., 3:7]
        out[..., 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
        return out

    def pd_targets(self, t: np.ndarray) -> np.ndarray:
        return self.qpos(t)[..., 7:]


@dataclass
class Schedule:
    """The contact schedule of an edge at a chosen timing (seconds since the departure)."""
    phase_names: list
    bounds: np.ndarray                      # [P + 1] phase boundaries, bounds[0] = 0, bounds[-1] = T
    ground: list                            # per phase (and S before 0, D after T): zone sets
    braces: list
    src_ground: list
    dst_ground: list
    src_braces: list
    dst_braces: list
    known: set                              # zones whose state is known at both ends (planted or known free)
    quasi_static: list
    events: list = field(default_factory=list)   # [(t, zone, "make"|"break")]

    @property
    def T(self) -> float:
        return float(self.bounds[-1])

    def phase_at(self, t: np.ndarray) -> np.ndarray:
        """-1 before the departure, P after the arrival, else the phase index."""
        t = np.asarray(t, float)
        p = np.searchsorted(self.bounds, t, side="right") - 1
        return np.where(t < 0, -1, np.where(t >= self.T, len(self.phase_names), p))

    def config_at(self, t: float) -> tuple[set, set, bool]:
        p = int(self.phase_at(np.array([t]))[0])
        if p < 0:
            return set(self.src_ground), set(self.src_braces), True
        if p >= len(self.phase_names):
            return set(self.dst_ground), set(self.dst_braces), True
        return set(self.ground[p]), set(self.braces[p]), bool(self.quasi_static[p])

    def masks(self, t: np.ndarray, window: float = EVENT_WINDOW_S) -> dict:
        """Per time: ``ground [n, Z]``, ``free [n, Z]`` (outside every event window of the zone), ``window [n, Z]``
        (inside a make/break window), ``quasi_static [n]``, ``phase [n]``."""
        t = np.asarray(t, float)
        Z = len(ZONE_ORDER)
        ground = np.zeros((len(t), Z), bool)
        free = np.zeros((len(t), Z), bool)
        win = np.zeros((len(t), Z), bool)
        qs = np.zeros(len(t), bool)
        for i, ti in enumerate(t):
            g, _, q = self.config_at(float(ti))
            qs[i] = q
            for z, name in enumerate(ZONE_ORDER):
                ground[i, z] = name in g
                free[i, z] = (name not in g) and (name in self.known)
        for te, zone, _ in self.events:
            z = ZONE_ORDER.index(zone)
            inside = np.abs(t - te) <= window
            win[inside, z] = True
            free[inside, z] = False
            ground[inside, z] = False          # no planted-support demand inside its own make/break window
        return {"ground": ground, "free": free, "window": win, "quasi_static": qs, "phase": self.phase_at(t)}


def timing(e: dict, which: str | list = "mid") -> list:
    """Phase durations: ``low``/``mid``/``high`` of each admissible range, or explicit seconds."""
    if isinstance(which, (list, tuple)):
        if len(which) != len(e["phases"]):
            raise ValueError("one duration per phase")
        return [float(x) for x in which]
    f = {"low": 0.0, "mid": 0.5, "high": 1.0}[which]
    return [p["duration_s"][0] + f * (p["duration_s"][1] - p["duration_s"][0]) for p in e["phases"]]


def schedule(e: dict, durations: list) -> Schedule:
    b = np.r_[0.0, np.cumsum(durations)]
    src = e["source"]
    dst = e["destination"]
    src_ground = sorted(src["ground"])
    dst_ground = sorted(dst["ground"])
    src_braces = sorted(x["pair"] for x in src["braces"])
    dst_braces = sorted(x["pair"] for x in dst["braces"])
    known = (set(src_ground) | set(src["known_free"])) & (set(dst_ground) | set(dst["known_free"]))
    configs = [(src_ground, src_braces)] + [(p["ground"], p["braces"]) for p in e["phases"]] + [(dst_ground, dst_braces)]
    times = [0.0] + list(b[1:-1]) + [float(b[-1])]
    events = []
    for k in range(len(configs) - 1):
        g0, g1 = set(configs[k][0]), set(configs[k + 1][0])
        for z in sorted(g1 - g0):
            events.append((times[k], z, "make"))
        for z in sorted(g0 - g1):
            events.append((times[k], z, "break"))
    return Schedule([p["name"] for p in e["phases"]], b, [p["ground"] for p in e["phases"]],
                    [p["braces"] for p in e["phases"]], src_ground, dst_ground, src_braces, dst_braces, known,
                    [p["quasi_static"] for p in e["phases"]], events)


class RefSketch:
    """A dense reference (``quasistatic``'s ``reference.npz``, 60 fps exp-map coordinates) as a sketch: hinge
    ``qpos`` per frame, branch-continuous along the reference, linear in between, held outside it. ``T`` is the
    edge's arrival time (the reference runs on past it)."""

    def __init__(self, plant: pm.Plant, ref_dir: Path, T: float):
        z = np.load(Path(ref_dir) / "reference.npz")
        self.plant, self.times, self._T = plant, z["times"], float(T)
        quat = Rotation.from_matrix(z["root_rot"]).as_quat()
        q = np.empty((len(self.times), plant.nq))
        prev = None
        for i in range(len(q)):
            q[i] = plant.qpos_from_expmap(z["root_pos"][i], quat[i], z["dof"][i], prev=prev)
            prev = q[i]
        self.q = q

    @property
    def T(self) -> float:
        return self._T

    def qpos(self, t: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(t, float), self.times[0], self.times[-1])
        i = np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times) - 2)
        a = ((t - self.times[i]) / (self.times[i + 1] - self.times[i]))[..., None]
        out = (1 - a) * self.q[i] + a * self.q[i + 1]
        qq = out[..., 3:7]
        out[..., 3:7] = qq / np.linalg.norm(qq, axis=-1, keepdims=True)
        return out

    def pd_targets(self, t: np.ndarray) -> np.ndarray:
        return self.qpos(t)[..., 7:]
