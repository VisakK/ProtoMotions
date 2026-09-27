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
    mixture_sampling_probs,
    score_clip,
    update_score_ema,
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

    # ------------------------------------------------------------------ #
    def _setup_tables(self) -> None:
        if self._holds is not None:
            return
        if not self.config.hold_manifest:
            raise ValueError("HoldCurriculumEvaluatorConfig.hold_manifest is required")
        manifest = yaml.safe_load(open(self.config.hold_manifest))
        by_stem = {c["stem"]: c for c in manifest["clips"]}
        stems = [Path(f).name[: -len(".motion")] for f in self.motion_lib.motion_files]
        missing = [s for s in stems if s not in by_stem]
        if missing:
            raise ValueError(f"{len(missing)} motions missing from {self.config.hold_manifest}: {missing[:3]}")
        self._stems = stems
        self._holds = [
            [HoldWindow(float(h["t_hold"]), float(h["t_end"]), bool(h.get("extend", False)))
             for h in by_stem[s]["holds"]]
            for s in stems
        ]
        self._groups = [by_stem[s].get("group", "all") for s in stems]
        names = list(self.env.robot_config.kinematic_info.body_names)
        self._goal_ids = [names.index(b) for b in GOAL_BODY_NAMES]
        self._zone_ids = {z: [names.index(b) for b in bodies] for z, bodies in SUPPORT_ZONES.items()}
        c = self.config.curriculum
        self._params = ScoreParams(
            track_fail_m=c.track_fail_m, pose_threshold_m=c.pose_threshold_m,
            foot_down_z=c.foot_down_z, unloaded_ref_min_z=c.unloaded_ref_min_z,
            track_weight=c.track_weight,
        )

    # ------------------------------------------------------------------ #
    def _score_all_motions(self) -> Dict[str, torch.Tensor]:
        self._setup_tables()
        pos_metric = self._metrics["rigid_body_pos"]
        num_motions = self.motion_lib.num_motions()
        num_bodies = self.env.robot_config.kinematic_info.num_bodies
        dt = float(self.env.dt)
        keys = ("p_track", "p_hold", "p_family", "support_violation", "track_fail_frac", "score")
        out = {k: torch.full((num_motions,), float("nan")) for k in keys}
        for m in range(num_motions):
            frames = int(pos_metric.frame_counts[m].item())
            if frames < 2:
                continue
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
        return out

    def _update_curriculum(self, scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        c = self.config.curriculum
        self._score_ema = update_score_ema(self._score_ema, scores["score"], c.score_ema_keep)
        probs = mixture_sampling_probs(
            self._score_ema, c.uniform_fraction, power=c.priority_power, eps=c.priority_eps
        )
        self.env.motion_manager.update_sampling_weights(probs.to(self.env.motion_manager.motion_weights.device))
        return probs

    def _write_table(self, scores, probs) -> None:
        out_dir = self.root_dir / "curriculum"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"eval_epoch_{self.agent.current_epoch:06d}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["motion_id", "motion", "group", "p_track", "p_hold", "p_family",
                        "support_violation", "score", "score_ema", "sampling_prob"])
            for m, stem in enumerate(self._stems):
                w.writerow([m, stem, self._groups[m]]
                           + [f"{float(scores[k][m]):.4f}" for k in
                              ("p_track", "p_hold", "p_family", "support_violation", "score")]
                           + [f"{float(self._score_ema[m]):.4f}", f"{float(probs[m]):.6f}"])

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
        n = probs.numel()
        prioritized = probs - self.config.curriculum.uniform_fraction / n
        logs["eval/curriculum/ess"] = float(1.0 / (probs ** 2).sum())
        logs["eval/curriculum/max_prob_x_n"] = float(probs.max() * n)
        if prioritized.sum() > 0:
            top = torch.topk(prioritized, min(10, n)).values.sum() / prioritized.sum()
            logs["eval/curriculum/prioritized_top10_share"] = float(top)
        return {k: v for k, v in logs.items() if not math.isnan(v)}

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
