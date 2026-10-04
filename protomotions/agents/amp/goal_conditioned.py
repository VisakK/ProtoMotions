# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMP for the goal-conditioned expert (graph-growth round, card E2).

``expert_revist/graph_growth_2026_10_03/PLAN.MD`` §1.2 / card E2. The stock
``AMP`` agent, with four changes the plan asks for:

* **Demonstrations from the x0 clips only, uniform per frame.** The stock
  sampler draws through the curriculum's motion weights, which move at every
  evaluation, and 13.6 % of the packaged frames are hold-extension inserts with
  exactly zero velocity -- an unreachable, non-physical cluster no human
  produces. The x0 variants carry no inserts and lose nothing (the variants'
  real frames duplicate them). Motions named in ``demo_exclude_motions``
  (Scorpion -b, whose kick-up gate v2 certified beyond the plant) never serve
  as demonstrations, and a window never starts before ``demo_min_time_steps``
  control steps so no demonstration window is clamped at t = 0.
* **A weight schedule.** ``discriminator_reward_w`` is 0 for epochs
  ``[0, amp_reward_w_start_epoch)`` -- the discriminator trains against the
  warm-started policy without paying it anything -- then ramps linearly to its
  target at ``amp_reward_w_full_epoch``. It is set inside ``add_advantages``,
  which is where the stock component reads it. With
  ``amp_calibrate_style_ratio`` > 0 the target is not guessed: at the start
  epoch it is frozen at that ratio divided by the median raw style/task
  advantage-std ratio of the preceding ``amp_calibrate_window`` (w = 0) epochs,
  i.e. the weight at which the style advantage's std is that fraction of the
  task advantage's (PLAN.MD: 0.2-0.35), clamped to [w_min, w_max] and kept in
  the checkpoint so a resume does not re-calibrate.
* **Unweighted style-advantage diagnostics.** The plan calibrates the weight by
  advantage scale (style std 0.2-0.35 of task std). The stock log only has the
  *weighted* style advantage, which is identically 0 while w = 0; this logs the
  unweighted one too, so ``advantages/style_to_task_std_at_target`` predicts the
  ratio the target weight will produce before the ramp starts.
* **A per-lineage AMP weight hook** (``amp_lineage_weights``: motion-name
  substring -> weight, default 1.0 for every motion; Step 3 will weight
  synthesised clips by it). It scales each env's AMP reward by its motion's weight.

The discriminator reward threshold is expected to be 0 (no discriminator
termination): an early, sharp discriminator would otherwise end episodes en
masse. ``-log(1 - D)`` is never negative, so a 0 threshold never fires.

At the first rollout step the agent checks its own feature plumbing on the GPU
(``amp_parity_check``): every env has just been reset from the reference, so its
history window *is* the reference's, and its ``amp_obs`` must equal the
demonstration features at its (motion, time). The result is logged once as
``amp/parity_max_abs`` (and printed); it is the runtime twin of
``protomotions/tests/test_amp_features_v1.py``, through ``torch.compile``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor

from protomotions.agents.amp.agent import AMP
from protomotions.agents.amp.component import AMPTrainingComponent
from protomotions.agents.amp.config import AMPAgentConfig

log = logging.getLogger(__name__)


@dataclass
class GoalConditionedAMPAgentConfig(AMPAgentConfig):
    """``AMPAgentConfig`` plus the schedule, the demonstration set and the lineage hook."""

    _target_: str = "protomotions.agents.amp.goal_conditioned.GoalConditionedAMP"

    amp_reward_w_target: float = field(
        default=0.1, metadata={"help": "discriminator_reward_w once the ramp is complete (when not calibrated)."})
    amp_reward_w_start_epoch: int = field(
        default=200, metadata={"help": "Epochs before this pay no style reward (w = 0)."})
    amp_reward_w_full_epoch: int = field(
        default=500, metadata={"help": "w reaches its target here (linear ramp from the start epoch)."})
    amp_calibrate_style_ratio: float = field(
        default=0.0,
        metadata={"help": "> 0: at the start epoch, replace amp_reward_w_target by this ratio divided by the median "
                          "raw style/task advantage-std ratio of the preceding amp_calibrate_window epochs (PLAN.MD "
                          "§1.2: style advantage std 0.2-0.35 of the task's), clamped to [amp_reward_w_min, "
                          "amp_reward_w_max]. 0 = use amp_reward_w_target as given."})
    amp_calibrate_window: int = field(
        default=100, metadata={"help": "Epochs before the start epoch whose ratios the calibration takes the median of."})
    amp_reward_w_min: float = field(default=0.05, metadata={"help": "Lower clamp of a calibrated target."})
    amp_reward_w_max: float = field(default=0.5, metadata={"help": "Upper clamp of a calibrated target."})
    demo_exclude_regex: str = field(
        default=r"_x\d+s$",
        metadata={"help": "Motion stems matching this never serve as demonstrations (default: the "
                          "hold-extended x3s/x7s variants, so only x0 clips do)."})
    demo_exclude_motions: List[str] = field(
        default_factory=lambda: ["Scorpion_pose_or_vrischikasana-b"],
        metadata={"help": "Motion-stem substrings never used as demonstrations."})
    demo_min_time_steps: int = field(
        default=20, metadata={"help": "Demonstration windows end no earlier than this many control steps in."})
    amp_lineage_weights: Dict[str, float] = field(
        default_factory=dict,
        metadata={"help": "Motion-stem substring -> AMP reward weight (first match wins; default 1.0)."})
    amp_parity_check: bool = field(
        default=True, metadata={"help": "Check agent vs demonstration features once at the first rollout step."})


def amp_reward_weight(epoch: int, target: float, start_epoch: int, full_epoch: int) -> float:
    """0 before ``start_epoch``, linear to ``target`` at ``full_epoch``, ``target`` after."""
    if epoch < start_epoch:
        return 0.0
    if full_epoch <= start_epoch or epoch >= full_epoch:
        return float(target)
    return float(target) * (epoch - start_epoch) / float(full_epoch - start_epoch)


def demo_motion_mask(stems: List[str], exclude_regex: str, exclude_motions: List[str]) -> List[bool]:
    """Which motions may serve as demonstrations."""
    pattern = re.compile(exclude_regex) if exclude_regex else None
    return [not (pattern is not None and pattern.search(s)) and not any(x in s for x in exclude_motions)
            for s in stems]


class GoalConditionedAMPComponent(AMPTrainingComponent):
    """AMPTrainingComponent with the x0 demonstration sampler, the schedule and the lineage hook."""

    def __init__(self, agent):
        super().__init__(agent)
        self._demo_ids: Optional[Tensor] = None
        self._demo_lo: Optional[Tensor] = None
        self._demo_len: Optional[Tensor] = None
        self._demo_p: Optional[Tensor] = None
        self._motion_amp_weight: Optional[Tensor] = None
        self._lineage_ready = False
        self.current_w = 0.0
        # one-shot weight calibration (amp_calibrate_style_ratio): per-epoch raw ratios, then the frozen target
        self.ratio_history: List[List[float]] = []
        self.calibrated_target: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Demonstrations
    # ------------------------------------------------------------------ #
    def _stems(self) -> List[str]:
        return [Path(str(f)).name.rsplit(".motion", 1)[0] for f in self.agent.motion_lib.motion_files]

    def _setup_demo_sampler(self) -> None:
        if self._demo_ids is not None:
            return
        cfg = self.config
        stems = self._stems()
        keep = demo_motion_mask(stems, cfg.demo_exclude_regex, list(cfg.demo_exclude_motions))
        lengths = self.agent.motion_lib.motion_lengths.to(self.device).float()
        lo = float(cfg.demo_min_time_steps) * float(self.agent.env.simulator.dt)
        ids = torch.tensor([i for i, k in enumerate(keep) if k and float(lengths[i]) > lo],
                           device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            raise ValueError("GoalConditionedAMP: no motion qualifies as a demonstration")
        span = lengths[ids] - lo
        self._demo_ids, self._demo_lo, self._demo_len = ids, lo, span
        self._demo_p = span / span.sum()          # uniform per frame: a motion's share is its usable length
        excluded = [s for s, k in zip(stems, keep) if not k]
        log.info("AMP demonstrations: %d of %d motions, %.1f s usable (t >= %.3f s); %d excluded, e.g. %s",
                 ids.numel(), len(stems), float(span.sum()), lo, len(excluded), excluded[:3])

    def sample_demo(self, num_samples: int):
        self._setup_demo_sampler()
        pick = torch.multinomial(self._demo_p, num_samples, replacement=True)
        motion_ids = self._demo_ids[pick]
        times = self._demo_lo + torch.rand(num_samples, device=self.device) * self._demo_len[pick]
        return motion_ids, times

    def get_expert_disc_obs(self, num_samples: int):
        motion_ids, motion_times = self.sample_demo(num_samples)
        return self.reference_obs(motion_ids, motion_times)

    def reference_obs(self, motion_ids: Tensor, motion_times: Tensor) -> Dict[str, Tensor]:
        """The discriminator's inputs read from the motion library at (motion, time), in env-sized chunks."""
        ref_obs_components = self.config.reference_obs_components or {}
        if not ref_obs_components:
            raise ValueError("AMP requires reference_obs_components to be defined in AMPAgentConfig.")
        disc_in_keys = set(self.discriminator.module.in_keys) if hasattr(self, "discriminator") \
            else set(self.agent.model._discriminator.in_keys)
        chunk = self.agent.num_envs
        parts: Dict[str, List[Tensor]] = {}
        for start in range(0, motion_ids.shape[0], chunk):
            ids, times = motion_ids[start:start + chunk], motion_times[start:start + chunk]
            context = self._build_expert_obs_context(ids.shape[0], ids, times)
            for name, router in ref_obs_components.items():
                if name not in disc_in_keys:
                    continue
                obs = self._call_ref_obs_fn(router.get_compute_func(), context, router.get_params().copy())
                parts.setdefault(name, []).append(obs.view(ids.shape[0], -1))
        return {k: torch.cat(v, dim=0) for k, v in parts.items()}

    # ------------------------------------------------------------------ #
    # Lineage hook
    # ------------------------------------------------------------------ #
    def _env_amp_weight(self) -> Optional[Tensor]:
        if not self._lineage_ready:
            self._lineage_ready = True
            rules = dict(self.config.amp_lineage_weights or {})
            if rules:
                weights = []
                for stem in self._stems():
                    hit = next((w for pat, w in rules.items() if pat in stem), 1.0)
                    weights.append(float(hit))
                self._motion_amp_weight = torch.tensor(weights, device=self.device)
                log.info("AMP lineage weights: %s (motions not matched: 1.0)", rules)
        if self._motion_amp_weight is None:
            return None
        return self._motion_amp_weight[self.agent.motion_manager.motion_ids]

    @torch.no_grad()
    def record_rollout_step(self, next_obs_td, rewards, terminated, done_indices, extras, step) -> None:
        # AMPTrainingComponent.record_rollout_step, with the per-env lineage weight on the reward.
        disc_logits = self.discriminator(next_obs_td)[self.discriminator.module.config.out_keys[0]]
        extras["disc_logits"] = disc_logits.detach()
        amp_rewards = self.discriminator.module.compute_disc_reward(disc_logits).flatten()
        bad_transition = amp_rewards < self.config.amp_parameters.discriminator_reward_threshold
        self.num_cumulative_bad_transitions[bad_transition] += 1
        self.num_cumulative_bad_transitions[~bad_transition] = 0
        if len(done_indices) > 0:
            self.num_cumulative_bad_transitions[done_indices] = 0

        weight = self._env_amp_weight()
        if weight is not None:
            amp_rewards = amp_rewards * weight

        if self.use_disc_critic:
            next_disc_value = self.disc_critic(next_obs_td)[self.disc_critic.module.config.out_keys[0]]
            next_disc_value = next_disc_value * (1 - terminated.float()).unsqueeze(-1)
            self.agent.experience_buffer.update_data("next_disc_value", step, next_disc_value)

        if self.config.normalize_rewards:
            self.running_reward_norm.record_reward(amp_rewards, terminated)
        self.agent.experience_buffer.update_data("amp_rewards", step, amp_rewards)

    # ------------------------------------------------------------------ #
    # Schedule + diagnostics
    # ------------------------------------------------------------------ #
    def target_w(self) -> float:
        """The weight the ramp heads for: the calibrated one once frozen, else the configured one."""
        if self.calibrated_target is not None:
            return self.calibrated_target
        return float(self.config.amp_reward_w_target)

    def _maybe_calibrate(self, epoch: int) -> None:
        """At the start epoch, freeze target = ratio / median(raw style std / task std) over the w = 0 window."""
        cfg = self.config
        ratio = float(getattr(cfg, "amp_calibrate_style_ratio", 0.0) or 0.0)
        if ratio <= 0.0 or self.calibrated_target is not None or epoch < cfg.amp_reward_w_start_epoch:
            return
        lo = cfg.amp_reward_w_start_epoch - int(cfg.amp_calibrate_window)
        window = [r for e, r in self.ratio_history if lo <= e < cfg.amp_reward_w_start_epoch]
        if not window:          # e.g. resumed past the window without history: fall back to the configured target
            log.warning("AMP calibration: no ratios recorded in epochs [%d, %d); using amp_reward_w_target %.3f",
                        lo, cfg.amp_reward_w_start_epoch, cfg.amp_reward_w_target)
            self.calibrated_target = float(cfg.amp_reward_w_target)
            return
        median = float(torch.tensor(window).median())
        raw = ratio / max(median, 1e-8)
        self.calibrated_target = float(min(max(raw, cfg.amp_reward_w_min), cfg.amp_reward_w_max))
        msg = (f"AMP calibration at epoch {epoch}: median raw style/task advantage-std ratio {median:.4f} over "
               f"{len(window)} epochs -> target w {raw:.4f} for ratio {ratio:.3f}, clamped to {self.calibrated_target:.4f}")
        log.info(msg)
        print(f"[amp] {msg}")

    @torch.no_grad()
    def add_advantages(self, advantages_dict):
        cfg = self.config
        epoch = int(self.agent.current_epoch)
        self._maybe_calibrate(epoch)
        w = amp_reward_weight(epoch, self.target_w(), cfg.amp_reward_w_start_epoch, cfg.amp_reward_w_full_epoch)
        task = advantages_dict["advantages"]
        params = cfg.amp_parameters
        params.discriminator_reward_w = 1.0       # unit weight: the stock code returns task + raw style
        try:
            out = super().add_advantages(dict(advantages_dict))
        finally:
            params.discriminator_reward_w = w
        style = out["advantages"] - task
        out["advantages"] = task + w * style
        self.current_w = w
        self.agent._diag_task_advantages = task.detach()
        self.agent._diag_disc_advantages = (w * style).detach()
        self.agent._diag_disc_raw_advantages = style.detach()
        raw_ratio = float(style.std() / task.std().clamp(min=1e-8))
        self.ratio_history.append([epoch, raw_ratio])
        self.ratio_history = self.ratio_history[-1000:]
        return out

    def add_epoch_logging(self, training_log_dict) -> None:
        super().add_epoch_logging(training_log_dict)
        training_log_dict["amp/reward_w"] = torch.tensor(self.current_w)
        training_log_dict["amp/reward_w_target"] = torch.tensor(self.target_w())
        raw = getattr(self.agent, "_diag_disc_raw_advantages", None)
        task = getattr(self.agent, "_diag_task_advantages", None)
        if raw is not None and task is not None:
            task_std = task.std().clamp(min=1e-8)
            raw_std = raw.std()
            training_log_dict["advantages/disc_raw_std"] = raw_std
            training_log_dict["advantages/style_raw_to_task_std"] = raw_std / task_std
            training_log_dict["advantages/style_to_task_std"] = self.current_w * raw_std / task_std
            training_log_dict["advantages/style_to_task_std_at_target"] = self.target_w() * raw_std / task_std
        parity = getattr(self.agent, "_amp_parity_log", None)
        if parity:
            training_log_dict.update(parity)
            self.agent._amp_parity_log = None

    # The calibration survives a resume (a checkpoint after the start epoch carries the frozen target).
    def add_state_dict(self, state_dict) -> dict:
        state_dict = super().add_state_dict(state_dict)
        state_dict["amp_weight_calibration"] = {"ratio_history": list(self.ratio_history),
                                                "calibrated_target": self.calibrated_target}
        return state_dict

    def load_training_state(self, state_dict) -> None:
        super().load_training_state(state_dict)
        cal = state_dict.get("amp_weight_calibration")
        if cal:
            self.ratio_history = [list(x) for x in cal.get("ratio_history", [])]
            self.calibrated_target = cal.get("calibrated_target")


class GoalConditionedAMP(AMP):
    """``AMP`` with ``GoalConditionedAMPComponent`` (see the module docstring)."""

    config: GoalConditionedAMPAgentConfig

    def __init__(self, fabric, env, config, root_dir=None):
        super().__init__(fabric, env, config, root_dir=root_dir)
        # Replace the stock component before anything has used it (no optimizer, buffer or
        # replay entry exists yet; the stock one's replay buffer is empty and discarded).
        self.amp_component = GoalConditionedAMPComponent(self)
        self._amp_parity_done = not bool(getattr(config, "amp_parity_check", True))
        self._amp_parity_log = None

    @torch.no_grad()
    def collect_rollout_step(self, obs_td, step):
        if not self._amp_parity_done:
            self._amp_parity_done = True
            self._run_amp_parity_check(obs_td)
        return super().collect_rollout_step(obs_td, step)

    @torch.no_grad()
    def _run_amp_parity_check(self, obs_td) -> None:
        """Every env was just reset from the reference: its window must equal the demonstration's."""
        try:
            keys = [k for k in self.model._discriminator.in_keys if k in (self.config.reference_obs_components or {})]
            fresh = (self.env.progress_buf == 0).nonzero(as_tuple=False).flatten()
            if not keys or fresh.numel() == 0:
                return
            mm = self.motion_manager
            ref = AMPTrainingComponent.for_agent(self).reference_obs(mm.motion_ids[fresh], mm.motion_times[fresh])
            logs = {}
            for key in keys:
                diff = (obs_td[key][fresh].float() - ref[key].float()).abs().amax(dim=-1)
                logs[f"amp/parity_max_abs_{key}"] = diff.max()
                logs[f"amp/parity_share_ok_{key}"] = (diff <= 1e-4).float().mean()
                log.info("AMP parity (%s): %d fresh envs, max |agent - demo| = %.3g, median %.3g, "
                         "share <= 1e-4: %.4f", key, fresh.numel(), float(diff.max()), float(diff.median()),
                         float((diff <= 1e-4).float().mean()))
                print(f"[amp parity] {key}: {fresh.numel()} fresh envs, max {float(diff.max()):.3g}, "
                      f"median {float(diff.median()):.3g}, share<=1e-4 {float((diff <= 1e-4).float().mean()):.4f}")
            self._amp_parity_log = logs
        except Exception:
            log.exception("AMP parity check failed to run (training continues)")
