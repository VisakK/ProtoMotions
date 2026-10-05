# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MimicEvaluator with a hold-aware performance score driving a mixture curriculum.

The rollouts are the stock ones (every motion from t = 0, deterministic actions),
but ``max_eval_steps`` is expected to cover the longest clip so exits are scored.
After each evaluation every motion gets a continuous score from
``hold_curriculum.score_clip`` (tracking over the whole clip + attainment of every
annotated hold, pose *and* support), smoothed across evaluations with an EMA, and
the sampling weights become ``uniform_fraction`` uniform + the rest in proportion to
``1 - score``. The legacy binary verdict is still computed and logged as
``eval/success_rate`` (over the full window now) and ``failed_motions/*.txt``.

The evaluated score returned to the agent (``score_based.ckpt``) is the mean
performance score, not the legacy success rate.

**Support rule v2** (``hold_curriculum.support_v2_holds``, card E1 of
``expert_revist/graph_growth_2026_10_03/PLAN.MD``). When the contact-graph control
has the physics tables and the release's contact-target sidecar loaded and the
simulator reports terrain-filtered ground forces, every evaluation also scores
each hold by load on the zones the human keeps free, plus commanded supports
realised by collider geometry. Every pre-existing key (``eval/perf/*``,
``eval/perf_group/*``, the CSV's columns, the score returned for
``score_based.ckpt``) stays v1. The v2 numbers go to new keys
(``eval/perf_v2/*``, ``eval/perf_group_v2/*``, ``eval/perf/support_*_v2``,
``eval/perf/substitution_holds_v2[_x0]``) and to columns appended to the CSV.
``HoldCurriculumConfig.support_rule = "v2"`` makes only the curriculum's sampling
(``score_ema`` -> mixture probabilities) run on the v2 score.

**Package prior** (card E3). The package's per-motion yaml ``weight`` used to apply
only until the first evaluation replaced the manager's weights. With
``HoldCurriculumConfig.motion_prior = "package"`` it is captured once, when the
evaluator is built, and every evaluation's uniform share becomes
``uniform_fraction * w_m / sum(w)`` (``hold_curriculum.mixture_sampling_probs``);
``eval/curriculum/prior_share/<group>`` logs each group's share of it.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import yaml

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.hold_curriculum import (
    GOAL_BODY_NAMES,
    SUPPORT_ZONES,
    HoldWindow,
    ScoreParams,
    SupportV2Params,
    mixture_sampling_probs,
    score_clip,
    support_v2_holds,
    tracked_frames,
    uniform_share,
    update_score_ema,
    zone_lowest_points,
    zone_vertical_load,
)
from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator

logger = logging.getLogger(__name__)


class HoldCurriculumEvaluator(MimicEvaluator):
    """See module docstring."""

    def __init__(self, agent, fabric, config):
        super().__init__(agent, fabric, config)
        self._score_ema: Optional[torch.Tensor] = None
        self._holds = None  # per-motion list[HoldWindow], lazily from the manifest
        self._groups = None  # per-motion group name
        self._body_ids = None
        self._drag_tables = None  # the control's PhysicsTables, when loaded (drag metrics)
        self._support_rule = "v1"
        self._v2 = None  # support rule v2 tables (_setup_support_v2), None when unavailable
        self._v2_text = None  # per-motion flagged holds of the last evaluation (CSV)
        # Captured now, before the first evaluation overwrites the manager's weights;
        # never saved -- every launch re-reads it from the same package.
        self._motion_prior = self._capture_motion_prior()

    def _capture_motion_prior(self) -> Optional[torch.Tensor]:
        """The package's per-motion weights (``[M]``, CPU), or None for ``motion_prior='none'``.

        They are the library's ``motion_weights`` -- motions.yaml's ``weight`` as packed. When the
        library is a packed ``X.pt`` with an ``X.yaml`` beside it, the yaml is read too and must
        agree motion for motion: a package whose yaml says one thing and whose .pt another would
        otherwise train on a prior nobody wrote down.
        """
        mode = str(getattr(self.config.curriculum, "motion_prior", "none") or "none")
        if mode not in ("none", "package"):
            raise ValueError(f"HoldCurriculumConfig.motion_prior must be 'none' or 'package', got {mode!r}")
        if mode == "none":
            return None
        def stem(path) -> str:
            # 'X.motion' -> 'X' (a stem may itself contain dots); anything else -> Path.stem
            name = Path(str(path)).name
            return name[: -len(".motion")] if name.endswith(".motion") else Path(name).stem

        weights = self.motion_lib.motion_weights.detach().to(device="cpu", dtype=torch.float32).clone()
        stems = [stem(f) for f in self.motion_lib.motion_files]
        if len(stems) != weights.numel():
            raise ValueError(f"motion library has {len(stems)} files and {weights.numel()} weights")
        library = Path(str(getattr(self.motion_lib, "motion_file", "") or ""))
        sidecar = library.with_suffix(".yaml")
        if library.suffix == ".pt" and sidecar.is_file():
            with open(sidecar) as handle:
                entries = (yaml.safe_load(handle) or {}).get("motions") or []
            by_stem = {stem(e["file"]): float(e.get("weight", 1.0)) for e in entries}
            missing = [s for s in stems if s not in by_stem]
            if missing:
                raise ValueError(f"motion prior: {len(missing)} motions missing from {sidecar}: {missing[:3]}")
            # compared as the library stores them (float32: a yaml 0.6 is packed as 0.6000000238)
            declared = torch.tensor([by_stem[s] for s in stems], dtype=torch.float32)
            wrong = [s for m, s in enumerate(stems) if bool(declared[m] != weights[m])]
            if wrong:
                raise ValueError(
                    f"motion prior: {sidecar} and the packed library disagree on {len(wrong)} weights "
                    f"(e.g. {wrong[0]}: yaml {by_stem[wrong[0]]}, library {float(weights[stems.index(wrong[0])])})"
                )
        if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()) or float(weights.sum()) <= 0.0:
            raise ValueError("motion prior: the package weights must be finite, non-negative and not all zero")
        values, counts = torch.unique(weights, return_counts=True)
        histogram = {float(v): int(c) for v, c in zip(values, counts)}
        print(f"HoldCurriculumEvaluator: package motion prior over {weights.numel()} motions "
              f"(weight: motions) {histogram}")
        return weights

    # ------------------------------------------------------------------ #
    def _setup_tables(self) -> None:
        if self._holds is not None:
            return
        if not self.config.hold_manifest:
            raise ValueError("HoldCurriculumEvaluatorConfig.hold_manifest is required")
        # A run that names its release (ContactGraphControlConfig.release_file) scores the release's
        # own hold manifest, checked by content like every other artifact it loads.
        ctrl = getattr(self.env, "control_manager", None)
        comps = getattr(ctrl, "components", None) or {}
        graph_ctrl = comps.get("contact_graph") if isinstance(comps, dict) else None
        release = getattr(graph_ctrl, "release", None)
        if release is not None:
            from protomotions.utils.release_identity import require_artifact

            require_artifact(release, "holds_extended", self.config.hold_manifest, "evaluator hold manifest")
        manifest = yaml.safe_load(open(self.config.hold_manifest))
        by_stem = {c["stem"]: c for c in manifest["clips"]}
        stems = [Path(f).name[: -len(".motion")] for f in self.motion_lib.motion_files]
        missing = [s for s in stems if s not in by_stem]
        if missing:
            raise ValueError(f"{len(missing)} motions missing from {self.config.hold_manifest}: {missing[:3]}")
        self._stems = stems
        self._holds = [
            [HoldWindow(float(h["t_hold"]), float(h["t_end"]), bool(h.get("extend", False)),
                        float(h.get("t_start", h["t_hold"])))
             for h in by_stem[s]["holds"]]
            for s in stems
        ]
        self._groups = [by_stem[s].get("group", "all") for s in stems]
        # x0 = the unextended clips, the population the e15500 review counted on
        self._x0 = [float(by_stem[s].get("variant_s", 0.0) or 0.0) == 0.0 for s in stems]
        names = list(self.env.robot_config.kinematic_info.body_names)
        self._goal_ids = [names.index(b) for b in GOAL_BODY_NAMES]
        self._zone_ids = {z: [names.index(b) for b in bodies] for z, bodies in SUPPORT_ZONES.items()}
        c = self.config.curriculum
        self._params = ScoreParams(
            track_fail_m=c.track_fail_m, pose_threshold_m=c.pose_threshold_m,
            foot_down_z=c.foot_down_z, unloaded_ref_min_z=c.unloaded_ref_min_z,
            track_weight=c.track_weight,
            event_dilate_frames=int(getattr(c, "event_dilate_frames", 7)),
        )
        patterns = list(getattr(c, "report_exclude_motions", None) or [])
        self._report_excluded = [any(p in s for p in patterns) for s in stems]
        self._setup_drag(stems, names)
        self._setup_support_v2(stems, names, by_stem)

    def _setup_support_v2(self, stems, body_names, by_stem) -> None:
        """Tables for support rule v2: the control's physics tables (collider geometry, zones, masses)
        and the release's contact-target sidecar (known-free zones per hold).

        Without either, v2 is off and its metrics are simply absent -- unless the curriculum was
        asked to run on it (``support_rule = "v2"``), which is then a configuration error.
        """
        c = self.config.curriculum
        rule = str(getattr(c, "support_rule", "v1") or "v1")
        if rule not in ("v1", "v2"):
            raise ValueError(f"HoldCurriculumConfig.support_rule must be 'v1' or 'v2', got {rule!r}")
        self._support_rule = rule
        self._v2 = None
        ctrl = getattr(self.env, "control_manager", None)
        comps = getattr(ctrl, "components", None) or {}
        graph_ctrl = comps.get("contact_graph") if isinstance(comps, dict) else None
        tables = getattr(graph_ctrl, "_physics", None)
        targets = getattr(graph_ctrl, "_targets", None)
        graph = getattr(graph_ctrl, "graph", None)
        if tables is None or targets is None or graph is None:
            if rule == "v2":
                raise ValueError(
                    "support_rule 'v2' needs the contact-graph control's physics tables (--physics-tables) "
                    "and the release's contact-target sidecar (--contact-targets)"
                )
            return
        zone_order = list(tables.zone_order)
        if list(targets.zone_order) != zone_order:
            raise ValueError(f"sidecar zone order {targets.zone_order} != physics tables {zone_order}")
        zone_body_ids = [[body_names.index(b) for b in tables.zone_bodies[z]] for z in zone_order]
        ground_pairs = [targets.pair_names.index(f"{z}:G") for z in zone_order]
        commanded, known_free = [], []
        for m, stem in enumerate(stems):
            holds = by_stem[stem]["holds"]
            cmd = torch.zeros(len(holds), len(zone_order), dtype=torch.bool)
            free = torch.zeros(len(holds), len(zone_order), dtype=torch.bool)
            for k, h in enumerate(holds):
                for pair in h.get("pairs_ground") or []:
                    cmd[k, zone_order.index(pair.split(":")[0])] = True
                seg = self._hold_segment(graph, m, k, h, stem)
                free[k] = (targets.ground_free[m, seg] & ~targets.masked[m, seg][ground_pairs]).cpu()
            commanded.append(cmd)
            known_free.append(free)
        params = SupportV2Params(
            load_frac_bw=float(getattr(c, "support_v2_load_frac_bw", 0.03)),
            body_weight_n=float(tables.body_mass.sum()) * 9.81,
            min_share=float(getattr(c, "support_v2_min_share", 0.2)),
            dilate_frames=int(getattr(c, "event_dilate_frames", 7)),
            down_m=float(getattr(c, "support_v2_down_m", 0.02)),
            realised_share=float(getattr(c, "support_v2_realised_share", 0.9)),
            tracked_share=float(getattr(c, "support_v2_tracked_share", 0.9)),
        )
        self._v2 = dict(tables=tables, zone_order=zone_order, zone_body_ids=zone_body_ids,
                        commanded=commanded, known_free=known_free, params=params)
        logger.info(
            "HoldCurriculumEvaluator: support rule v2 on (curriculum runs on %s); %d known-free zone-holds, "
            "load threshold %.1f N", rule, int(sum(int(f.sum()) for f in known_free)), params.load_threshold_n,
        )

    @staticmethod
    def _hold_segment(graph, m: int, k: int, hold: dict, stem: str) -> int:
        """The graph segment of manifest hold ``k`` of motion ``m`` (the sidecar is indexed by segment).

        The release builds them in the same order; checked by hold id, or by ``t_hold`` for graphs
        without hold ids.
        """
        if k >= int(graph.seg_count[m]):
            raise ValueError(f"{stem}: hold {k} has no graph segment")
        index = getattr(graph, "seg_hold_index", None)
        ids = getattr(graph, "hold_ids", None)
        if index is not None and ids and hold.get("hold_id") is not None:
            got = ids[int(index[m, k])] if int(index[m, k]) >= 0 else None
            if got != hold["hold_id"]:
                raise ValueError(f"{stem}: segment {k} is hold {got}, the manifest says {hold['hold_id']}")
        elif abs(float(graph.seg_hold[m, k]) - float(torch.tensor(float(hold["t_hold"])))) > 1e-4:
            raise ValueError(f"{stem}: segment {k} holds at {float(graph.seg_hold[m, k]):.4f} s, "
                             f"the manifest at {hold['t_hold']}")
        return k

    # Feet and hands, the zones the drag audit measured (contact_balance_investigation §2).
    _DRAG_ZONES = (("L_FOOT", ("L_Ankle", "L_Toe")), ("R_FOOT", ("R_Ankle", "R_Toe")),
                   ("L_HAND", ("L_Wrist", "L_Hand")), ("R_HAND", ("R_Wrist", "R_Hand")))

    def _setup_drag(self, stems, body_names) -> None:
        """Drag needs the box geometry and swing labels of the control's physics tables."""
        c = self.config.curriculum
        self._drag_tables = None
        ctrl = getattr(self.env, "control_manager", None)
        comps = getattr(ctrl, "components", None) or {}
        graph_ctrl = comps.get("contact_graph") if isinstance(comps, dict) else None
        tables = getattr(graph_ctrl, "_physics", None)
        needed = ("rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "rigid_body_ground_forces")
        if tables is None:
            return
        self._drag_tables = tables
        self._drag_needed = needed
        self._drag_bodies = [[body_names.index(b) for b in bodies] for _, bodies in self._DRAG_ZONES]
        self._drag_zone_cols = [tables.zone_order.index(z) for z, _ in self._DRAG_ZONES]
        self._drag_patterns = list(getattr(c, "drag_report_motions", None) or [])
        unmatched = [p for p in self._drag_patterns if not any(p in s for s in stems)]
        if unmatched:
            raise ValueError(f"drag_report_motions entries match no motion: {unmatched}")

    # ------------------------------------------------------------------ #
    def _score_all_motions(self) -> Dict[str, torch.Tensor]:
        self._setup_tables()
        pos_metric = self._metrics["rigid_body_pos"]
        num_motions = self.motion_lib.num_motions()
        num_bodies = self.env.robot_config.kinematic_info.num_bodies
        dt = float(self.env.dt)
        keys = ("p_track", "p_hold", "p_family", "p_family_event", "support_violation",
                "track_fail_frac", "score")
        drag_keys = ("drag_s", "drag_J", "drag_J_swing")
        out = {k: torch.full((num_motions,), float("nan")) for k in keys + drag_keys}
        drag = self._drag_tables is not None and all(k in self._metrics for k in self._drag_needed)
        v2 = self._support_v2_inputs(num_bodies)
        if v2 is not None:
            out.update({k: torch.full((num_motions,), float("nan")) for k in self._V2_KEYS})
            self._v2_text = [""] * num_motions
        for m in range(num_motions):
            frames = int(pos_metric.frame_counts[m].item())
            if frames < 2:
                continue
            if drag:
                out_drag = self._drag_clip(m, frames, num_bodies, dt)
                for k in drag_keys:
                    out[k][m] = out_drag[k]
            sim = pos_metric.data[m, :frames].view(frames, num_bodies, 3).float()
            times = (torch.arange(frames, device=sim.device, dtype=torch.float32) + 1.0) * dt
            ids = torch.full((frames,), m, device=sim.device, dtype=torch.long)
            ref = self.motion_lib.get_motion_state(ids, times).rigid_body_pos.float()
            holds = self._holds[m]
            exemplars = None
            if holds:
                h_ids = torch.full((len(holds),), m, device=sim.device, dtype=torch.long)
                h_t = torch.tensor([h.t_hold for h in holds], device=sim.device, dtype=torch.float32)
                exemplars = self.motion_lib.get_motion_state(h_ids, h_t).rigid_body_pos.float()
            s = score_clip(sim, ref, times, holds, exemplars, self._goal_ids, self._zone_ids, self._params)
            for k in keys:
                out[k][m] = s[k]
            if v2 is not None and int(v2["recorded"][m]) >= frames:
                self._score_motion_v2(m, frames, sim, ref, times, holds, exemplars, v2, out)
        return out

    # per-motion v2 outputs (NaN where unscored); counts are over the motion's tracked holds
    _V2_KEYS = ("score_v2", "p_hold_v2", "p_family_v2", "p_family_event_v2", "support_violation_v2",
                "holds_tracked_v2", "supports_scored_v2", "supports_realised_v2", "substitution_holds_v2")

    def _support_v2_inputs(self, num_bodies: int) -> Optional[Dict[str, torch.Tensor]]:
        """``[M, L, Z]`` lowest collider point and loaded flag of every zone on every recorded frame
        (one batched pass), or None when v2 is off or the simulator recorded no ground forces."""
        if self._v2 is None:
            return None
        need = ("rigid_body_pos", "rigid_body_rot", "rigid_body_ground_forces")
        forces = self._metrics.get("rigid_body_ground_forces")
        if not all(k in self._metrics for k in need) or not bool((forces.frame_counts > 0).any()):
            if self._support_rule == "v2":
                raise RuntimeError("support_rule 'v2': the evaluation recorded no rigid_body_ground_forces")
            return None
        v2 = self._v2
        pos_all = self._metrics["rigid_body_pos"].data
        rot_all = self._metrics["rigid_body_rot"].data
        num_motions, length = pos_all.shape[:2]
        zone_low = torch.empty(num_motions, length, len(v2["zone_order"]), device=pos_all.device)
        loaded = torch.empty(num_motions, length, len(v2["zone_order"]), dtype=torch.bool, device=pos_all.device)
        chunk = 16  # motions per pass: bounds the box-corner temporaries on a GPU shared with training
        with torch.no_grad():
            for i in range(0, num_motions, chunk):
                j = min(i + chunk, num_motions)
                n = (j - i) * length
                pos = pos_all[i:j].reshape(n, num_bodies, 3).float()
                rot = rot_all[i:j].reshape(n, num_bodies, 4).float()
                ground = forces.data[i:j].reshape(n, num_bodies, 3).float()
                zone_low[i:j] = zone_lowest_points(pos, rot, v2["tables"], v2["zone_body_ids"]).view(j - i, length, -1)
                loaded[i:j] = (zone_vertical_load(ground, v2["zone_body_ids"])
                               >= v2["params"].load_threshold_n).view(j - i, length, -1)
        return dict(zone_low=zone_low, zone_loaded=loaded, recorded=forces.frame_counts)

    def _score_motion_v2(self, m, frames, sim, ref, times, holds, exemplars, v2, out) -> None:
        params = self._v2["params"]
        tracked = tracked_frames(sim, ref, self._params.track_fail_m)
        rows, masks = support_v2_holds(
            times, v2["zone_loaded"][m, :frames], v2["zone_low"][m, :frames], holds,
            self._v2["commanded"][m], self._v2["known_free"][m], tracked, params,
        )
        s = score_clip(sim, ref, times, holds, exemplars, self._goal_ids, self._zone_ids, self._params,
                       hold_violation=masks)
        for k in ("score", "p_hold", "p_family", "p_family_event", "support_violation"):
            out[f"{k}_v2"][m] = s[k]
        tracked_rows = [r for r in rows if r is not None and r["tracked_share"] >= params.tracked_share]
        with_cmd = [r for r in tracked_rows if r["realised"] is not None]
        out["holds_tracked_v2"][m] = len(tracked_rows)
        out["supports_scored_v2"][m] = len(with_cmd)
        out["supports_realised_v2"][m] = sum(1 for r in with_cmd if r["realised"])
        out["substitution_holds_v2"][m] = sum(1 for r in tracked_rows if r["substitution"])
        zones = self._v2["zone_order"]
        self._v2_text[m] = ";".join(
            f"{holds[k].t_hold:.2f}:{'+'.join(zones[z] for z in r['flagged'])}"
            + ("" if r["tracked_share"] >= params.tracked_share else "(untracked)")
            for k, r in enumerate(rows) if r is not None and r["substitution"]
        )

    def _drag_clip(self, m: int, frames: int, num_bodies: int, dt: float) -> Dict[str, float]:
        from protomotions.envs.control.physics_terms import clip_drag

        def rec(key, width):
            return self._metrics[key].data[m, :frames].view(frames, num_bodies, width).float()

        tables = self._drag_tables
        times = (torch.arange(frames, device=tables.swing.device, dtype=torch.float32) + 1.0) * dt
        ids = torch.full((frames,), m, device=tables.swing.device, dtype=torch.long)
        swing = tables.swing_at(ids, times)[:, self._drag_zone_cols]
        c = self.config.curriculum
        return clip_drag(
            rec("rigid_body_pos", 3), rec("rigid_body_rot", 4), rec("rigid_body_vel", 3),
            rec("rigid_body_ang_vel", 3), rec("rigid_body_ground_forces", 3), tables,
            self._drag_bodies, swing.to(self._metrics["rigid_body_pos"].data.device), dt,
            load_n=float(getattr(c, "drag_load_n", 50.0)),
            slip_mps=float(getattr(c, "drag_slip_mps", 0.10)),
            mu=float(getattr(c, "drag_mu", 0.75)),
        )

    def _update_curriculum(self, scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        c = self.config.curriculum
        # support rule v2 steers only the sampling; every logged v1 key is untouched
        key = "score_v2" if self._support_rule == "v2" else "score"
        self._score_ema = update_score_ema(self._score_ema, scores[key], c.score_ema_keep)
        probs = mixture_sampling_probs(
            self._score_ema, c.uniform_fraction, power=c.priority_power, eps=c.priority_eps,
            prior=getattr(self, "_motion_prior", None),
        )
        self.env.motion_manager.update_sampling_weights(probs.to(self.env.motion_manager.motion_weights.device))
        return probs

    def _write_table(self, scores, probs) -> None:
        out_dir = self.root_dir / "curriculum"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"eval_epoch_{self.agent.current_epoch:06d}.csv"
        v2 = "score_v2" in scores
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["motion_id", "motion", "group", "p_track", "p_hold", "p_family",
                        "support_violation", "score", "score_ema", "sampling_prob", "p_family_event",
                        "drag_s", "drag_J", "drag_J_swing"]
                       + (["score_v2", "p_hold_v2", "p_family_v2", "support_violation_v2",
                           "support_realised_v2", "holds_tracked_v2", "substitution_holds_v2",
                           "substitutions_v2"] if v2 else []))
            for m, stem in enumerate(self._stems):
                row = ([m, stem, self._groups[m]]
                       + [f"{float(scores[k][m]):.4f}" for k in
                          ("p_track", "p_hold", "p_family", "support_violation", "score")]
                       + [f"{float(self._score_ema[m]):.4f}", f"{float(probs[m]):.6f}",
                          f"{float(scores['p_family_event'][m]):.4f}"]
                       + [f"{float(scores[k][m]):.2f}" for k in ("drag_s", "drag_J", "drag_J_swing")])
                if v2:
                    scored = float(scores["supports_scored_v2"][m])
                    realised = (float(scores["supports_realised_v2"][m]) / scored
                                if scored == scored and scored > 0 else float("nan"))
                    row += ([f"{float(scores[k][m]):.4f}" for k in
                             ("score_v2", "p_hold_v2", "p_family_v2", "support_violation_v2")]
                            + [f"{realised:.4f}"]
                            + [f"{float(scores[k][m]):.0f}" for k in ("holds_tracked_v2", "substitution_holds_v2")]
                            + [(self._v2_text or [""] * len(self._stems))[m]])
                w.writerow(row)

    @staticmethod
    def _nanmean(x: torch.Tensor) -> float:
        x = x[torch.isfinite(x)]
        return float(x.mean()) if x.numel() else float("nan")

    def _curriculum_logs(self, scores, probs) -> Dict[str, float]:
        logs = {
            "eval/perf/score": self._nanmean(scores["score"]),
            "eval/perf/track": self._nanmean(scores["p_track"]),
            "eval/perf/hold": self._nanmean(scores["p_hold"]),
            "eval/perf/family_hold": self._nanmean(scores["p_family"]),
            "eval/perf/support_violation": self._nanmean(scores["support_violation"]),
            "eval/perf/track_fail_frac": self._nanmean(scores["track_fail_frac"]),
        }
        finite = scores["score"][torch.isfinite(scores["score"])]
        if finite.numel():
            k = min(10, finite.numel())
            logs["eval/perf/worst10_score"] = float(torch.topk(finite, k, largest=False).values.mean())
        for g in sorted(set(self._groups)):
            idx = torch.tensor([i for i, x in enumerate(self._groups) if x == g])
            logs[f"eval/perf_group/{g}_score"] = self._nanmean(scores["score"][idx])
            logs[f"eval/perf_group/{g}_hold"] = self._nanmean(scores["p_hold"][idx])
            logs[f"eval/perf_group/{g}_family"] = self._nanmean(scores["p_family"][idx])
            logs[f"eval/perf_group/{g}_family_event"] = self._nanmean(scores["p_family_event"][idx])
        # the support-term reports' "penalised arm balances": the group minus the motions the
        # support terms do not charge (report_exclude_motions)
        pen = [i for i, g in enumerate(self._groups)
               if g == "arm_balance" and not self._report_excluded[i]]
        if pen:
            idx = torch.tensor(pen)
            logs["eval/perf/arm_balance_family_penalised"] = self._nanmean(scores["p_family"][idx])
            logs["eval/perf/arm_balance_family_event_penalised"] = self._nanmean(scores["p_family_event"][idx])
        logs.update(self._drag_logs(scores))
        logs.update(self._support_v2_logs(scores))
        n = probs.numel()
        prior = getattr(self, "_motion_prior", None)
        if prior is not None:
            prior = prior.to(probs.device)
        prioritized = probs - uniform_share(n, self.config.curriculum.uniform_fraction, prior)
        logs["eval/curriculum/ess"] = float(1.0 / (probs ** 2).sum())
        logs["eval/curriculum/max_prob_x_n"] = float(probs.max() * n)
        if prioritized.sum() > 0:
            top = torch.topk(prioritized, min(10, n)).values.sum() / prioritized.sum()
            logs["eval/curriculum/prioritized_top10_share"] = float(top)
        if prior is not None:
            # the prior's share of the uniform mass, and the group's whole sampling probability
            share = prior / prior.sum()
            for g in sorted(set(self._groups)):
                idx = torch.tensor([i for i, x in enumerate(self._groups) if x == g], device=probs.device)
                logs[f"eval/curriculum/prior_share/{g}"] = float(share[idx].sum())
                logs[f"eval/curriculum/group_prob/{g}"] = float(probs[idx].sum())
        return {k: v for k, v in logs.items() if not math.isnan(v)}

    def _support_v2_logs(self, scores) -> Dict[str, float]:
        """Support rule v2 under new keys (module docstring); empty when v2 did not run.

        ``*_x0`` keys count the unextended clips only -- the population of the e15500 review
        (README §4: supports realised on 232 of 260 tracked holds, 10 substitutions).
        """
        if "score_v2" not in scores:
            return {}
        logs = {
            "eval/perf_v2/score": self._nanmean(scores["score_v2"]),
            "eval/perf_v2/hold": self._nanmean(scores["p_hold_v2"]),
            "eval/perf_v2/family_hold": self._nanmean(scores["p_family_v2"]),
            "eval/perf_v2/family_hold_event": self._nanmean(scores["p_family_event_v2"]),
            "eval/perf/support_violation_v2": self._nanmean(scores["support_violation_v2"]),
        }
        finite = scores["score_v2"][torch.isfinite(scores["score_v2"])]
        if finite.numel():
            k = min(10, finite.numel())
            logs["eval/perf_v2/worst10_score"] = float(torch.topk(finite, k, largest=False).values.mean())
        for g in sorted(set(self._groups)):
            idx = torch.tensor([i for i, x in enumerate(self._groups) if x == g])
            logs[f"eval/perf_group_v2/{g}_score"] = self._nanmean(scores["score_v2"][idx])
            logs[f"eval/perf_group_v2/{g}_hold"] = self._nanmean(scores["p_hold_v2"][idx])
            logs[f"eval/perf_group_v2/{g}_family"] = self._nanmean(scores["p_family_v2"][idx])

        def total(key, ids):
            x = scores[key][torch.tensor(ids, dtype=torch.long)] if ids else torch.zeros(0)
            return float(torch.nansum(x)) if x.numel() else float("nan")

        everything = list(range(len(self._stems)))
        x0 = [i for i in everything if self._x0[i]]
        for suffix, ids in (("", everything), ("_x0", x0)):
            scored = total("supports_scored_v2", ids)
            if scored == scored and scored > 0:
                logs[f"eval/perf/support_realised_v2{suffix}"] = total("supports_realised_v2", ids) / scored
            logs[f"eval/perf/substitution_holds_v2{suffix}"] = total("substitution_holds_v2", ids)
            logs[f"eval/perf/holds_tracked_v2{suffix}"] = total("holds_tracked_v2", ids)
        return logs

    def _drag_logs(self, scores) -> Dict[str, float]:
        """Mean drag per rollout (J, s): corpus, per group, the gate set and each gate clip.

        Motions in ``report_exclude_motions`` (the terms' exclusions) are left out everywhere.
        """
        if self._drag_tables is None:
            return {}
        kept = [i for i in range(len(self._stems)) if not self._report_excluded[i]]
        logs = {}

        def pool(prefix, ids):
            if not ids:
                return
            idx = torch.tensor(ids)
            logs[f"{prefix}_J"] = self._nanmean(scores["drag_J"][idx])
            logs[f"{prefix}_swing_J"] = self._nanmean(scores["drag_J_swing"][idx])
            logs[f"{prefix}_s"] = self._nanmean(scores["drag_s"][idx])

        pool("eval/drag/all", kept)
        for g in sorted(set(self._groups)):
            pool(f"eval/drag_group/{g}", [i for i in kept if self._groups[i] == g])
        top = []
        for p in self._drag_patterns:
            ids = [i for i in kept if p in self._stems[i]]
            top += [i for i in ids if i not in top]
            name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in p)
            if ids:
                logs[f"eval/drag/{name}_J"] = self._nanmean(scores["drag_J"][torch.tensor(ids)])
        pool("eval/drag/top", top)
        return logs

    # ------------------------------------------------------------------ #
    def process_eval_results(self) -> Tuple[Dict, Optional[float], int]:
        # Legacy scalar logs (eval/success_rate, per-component means) -- skip the
        # MimicEvaluator's weight update, which this class replaces.
        to_log, _legacy_success, num_eval_items = BaseEvaluator.process_eval_results(self)
        if self._motion_failed is not None:
            failed = torch.nonzero(self._motion_failed).flatten().tolist()
            self._save_failed_motions(failed, self.agent.current_epoch)

        scores = self._score_all_motions()
        probs = self._update_curriculum(scores)
        if self.fabric.global_rank == 0:
            self._write_table(scores, probs)
        to_log.update(self._curriculum_logs(scores, probs))
        to_log.update(self._compute_additional_metrics(self._metrics))

        if self.fabric.global_rank == 0:
            if (
                self.config.save_predicted_motion_lib_every is not None
                and self.eval_count % self.config.save_predicted_motion_lib_every == 0
            ):
                self._save_predicted_motion_lib(self._metrics, epoch=self.agent.current_epoch)

        score = to_log.get("eval/perf/score")
        return to_log, score, num_eval_items

    # ------------------------------------------------------------------ #
    def get_state_dict(self) -> Dict:
        state = super().get_state_dict()
        if self._score_ema is not None:
            state["score_ema"] = self._score_ema.cpu()
        return state

    def load_state_dict(self, state_dict: Dict) -> None:
        super().load_state_dict(state_dict)
        if "score_ema" in state_dict:
            self._score_ema = state_dict["score_ema"].clone()
