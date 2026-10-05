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

Card E6 (the Step 3 fine-tune, G3) adds three things:

* **The lineage rules from the command line** (``--amp-lineage-weights
  PATTERN=W ...``, parsed by ``parse_amp_lineage_weights``) and, once rules are
  set, **per-lineage diagnostics**: every rollout step splits the unweighted
  style reward by the env's motion at reward time into "syn" (a motion some
  rule matches) and "human" (the rest), and each epoch logs
  ``amp/reward_mean_syn``, ``amp/reward_mean_human``,
  ``amp/reward_mean_syn_weighted`` and ``amp/syn_sample_share``. A group with
  no samples in an epoch is left out of that epoch's log (never NaN). With no
  rules nothing is accumulated or logged.
* **The AMP training state on a warm start**
  (``GoalConditionedAMP._load_optimization_state``, under
  ``--warm-start-optimization-state``): the discriminator and disc-critic
  optimisers, the AMP reward normaliser and the weight calibration load after
  PPO's state. A checkpoint without them (a PPO checkpoint) is skipped with a
  printed note. The calibration's ratio history is dropped (it counts epochs
  on the old run's clock), and with calibration off the configured
  ``amp_reward_w_target`` wins over a restored calibrated target.
* **Printed setup lines** the smoke greps for: ``[amp lineage]`` (what each
  rule matched) and ``[amp demos]`` (the demonstration set). The package's
  ``log.info`` lines do not reach a run's log.

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
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


# ---------------------------------------------------------------------- #
# Lineage rules (card E6)
# ---------------------------------------------------------------------- #
def parse_amp_lineage_rule(item: str) -> Tuple[str, float]:
    """One ``PATTERN=W`` rule: a non-empty stem substring and a finite weight >= 0. Raises ValueError."""
    text = str(item)
    if text.count("=") != 1:
        raise ValueError(f"lineage rule {text!r}: expected PATTERN=W with exactly one '='")
    pattern, _, weight = text.partition("=")
    if not pattern or pattern != pattern.strip():
        raise ValueError(f"lineage rule {text!r}: PATTERN must be a non-empty stem substring without "
                         f"surrounding whitespace")
    try:
        w = float(weight)
    except ValueError:
        raise ValueError(f"lineage rule {text!r}: W {weight!r} is not a number") from None
    if not math.isfinite(w) or w < 0.0:
        raise ValueError(f"lineage rule {text!r}: W must be finite and >= 0, got {w}")
    return pattern, w


def parse_amp_lineage_weights(items: Optional[Iterable[str]]) -> Dict[str, float]:
    """``["SYN_=0.5", ...]`` -> ``{"SYN_": 0.5, ...}`` in the order given (first match wins).

    A pattern given twice raises: the second could never match, so it is a typo.
    """
    rules: Dict[str, float] = {}
    for item in list(items or []):
        pattern, w = parse_amp_lineage_rule(item)
        if pattern in rules:
            raise ValueError(f"lineage rule {item!r}: pattern {pattern!r} given twice")
        rules[pattern] = w
    return rules


def lineage_match(stems: List[str], rules: Dict[str, float]) -> Tuple[List[float], List[int]]:
    """Per stem: its AMP weight and the index of the rule that matched it (-1: none, weight 1.0).

    A rule matches when its pattern is a substring of the stem; the first rule in ``rules``' order wins.
    """
    items = list(rules.items())
    weights, which = [], []
    for stem in stems:
        hit = next((k for k, (pat, _) in enumerate(items) if pat in stem), -1)
        which.append(hit)
        weights.append(float(items[hit][1]) if hit >= 0 else 1.0)
    return weights, which


class LineageRewardMeter:
    """Per-epoch sums of the style reward over env-steps, split by lineage.

    "syn" is an env-step whose motion some lineage rule matched; "human" is every other env-step. The
    *unweighted* discriminator reward is the comparison the plan asks for (does the style term fight
    tracking on the edges?); the weighted mean is kept for syn only, since a human motion's weight is 1.0
    and its weighted mean is its unweighted one. Everything stays on the device until ``pop_log``.
    """

    # n_all, n_syn, raw_all, raw_syn, weighted_syn
    _N = 5

    def __init__(self, device):
        self._sums = torch.zeros(self._N, dtype=torch.float64, device=device)

    @torch.no_grad()
    def record(self, raw: Tensor, weighted: Tensor, is_syn: Tensor) -> None:
        raw64 = raw.flatten().to(torch.float64)
        syn = is_syn.flatten().to(torch.float64)
        self._sums += torch.stack([
            torch.ones_like(raw64).sum(), syn.sum(), raw64.sum(), (raw64 * syn).sum(),
            (weighted.flatten().to(torch.float64) * syn).sum(),
        ])

    def pop_log(self) -> Dict[str, Tensor]:
        """This epoch's log entries; resets the sums. A group without samples is left out."""
        n_all, n_syn, raw_all, raw_syn, w_syn = (float(x) for x in self._sums.tolist())
        self._sums.zero_()
        out: Dict[str, Tensor] = {}
        if n_all <= 0:
            return out
        out["amp/syn_sample_share"] = torch.tensor(n_syn / n_all)
        if n_syn > 0:
            out["amp/reward_mean_syn"] = torch.tensor(raw_syn / n_syn)
            out["amp/reward_mean_syn_weighted"] = torch.tensor(w_syn / n_syn)
        if n_all - n_syn > 0:
            out["amp/reward_mean_human"] = torch.tensor((raw_all - raw_syn) / (n_all - n_syn))
        return out


# ---------------------------------------------------------------------- #
# Warm start (card E6)
# ---------------------------------------------------------------------- #
def amp_training_state_keys(state_dict: dict, use_disc_critic: bool,
                            normalize_rewards: bool) -> Tuple[List[str], List[str]]:
    """The AMP training-state entries a checkpoint must carry, split into (present, missing)."""
    need = ["discriminator_optimizer"]
    if use_disc_critic:
        need.append("disc_critic_optimizer")
    if normalize_rewards:
        need.append("running_amp_reward_norm")
    present = [k for k in need if k in state_dict]
    return present, [k for k in need if k not in state_dict]


def warm_start_calibration(restored_target: Optional[float], configured_target: float,
                           calibrate_ratio: float) -> Tuple[Optional[float], str]:
    """The calibrated target a warm start keeps, and the line saying which weight is in force.

    * Calibration off (``calibrate_ratio`` <= 0): the configured target is the weight. A restored calibrated
      target is dropped, so ``target_w()`` reads the configured one, and the note says whether it differed.
    * Calibration on: a restored target is kept, frozen (no re-calibration); without one the run calibrates
      at its own start epoch.
    """
    configured = float(configured_target)
    if float(calibrate_ratio or 0.0) <= 0.0:
        if restored_target is None:
            return None, f"AMP weight target {configured:g} (configured; calibration off, none in the checkpoint)"
        if abs(float(restored_target) - configured) <= 1e-12:
            return None, (f"AMP weight target {configured:g} (configured; calibration off; equals the "
                          f"checkpoint's calibrated target)")
        return None, (f"AMP weight target {configured:g} (configured) replaces the checkpoint's calibrated "
                      f"target {float(restored_target):g} (calibration off)")
    if restored_target is not None:
        return float(restored_target), (f"AMP weight target {float(restored_target):g} (the checkpoint's "
                                         f"calibration, kept frozen; configured {configured:g} unused)")
    return None, (f"AMP weight target: calibrated at this run's start epoch (calibration on, none in the "
                  f"checkpoint; configured {configured:g} is the fallback)")


def optimizer_step_count(optimizer) -> int:
    """The largest per-parameter ``step`` in an optimiser's state (0 when it has never stepped)."""
    state = optimizer.state_dict().get("state", {})
    steps = [float(s["step"]) for s in state.values() if isinstance(s, dict) and "step" in s]
    return int(max(steps)) if steps else 0


class GoalConditionedAMPComponent(AMPTrainingComponent):
    """AMPTrainingComponent with the x0 demonstration sampler, the schedule and the lineage hook."""

    def __init__(self, agent):
        super().__init__(agent)
        self._demo_ids: Optional[Tensor] = None
        self._demo_lo: Optional[Tensor] = None
        self._demo_len: Optional[Tensor] = None
        self._demo_p: Optional[Tensor] = None
        self._motion_amp_weight: Optional[Tensor] = None
        self._motion_is_syn: Optional[Tensor] = None
        self._lineage_meter: Optional[LineageRewardMeter] = None
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
        # Printed (log.info does not reach a run's log): the smoke checks that no SYN_ stem is a demonstration.
        self.setup_lineage()
        regex = re.compile(cfg.demo_exclude_regex) if cfg.demo_exclude_regex else None
        by_regex = [s for s in excluded if regex is not None and regex.search(s)]
        by_name = [s for s in excluded if not (regex is not None and regex.search(s))]
        short = [s for i, (s, k) in enumerate(zip(stems, keep)) if k and float(lengths[i]) <= lo]
        demos = [stems[i] for i in ids.tolist()]
        print(f"[amp demos] {len(demos)} of {len(stems)} motions serve as demonstrations, {float(span.sum()):.1f} s "
              f"usable (t >= {lo:.3f} s); excluded {len(by_regex)} by regex {cfg.demo_exclude_regex!r}, "
              f"{len(by_name)} by name {list(cfg.demo_exclude_motions)}, {len(short)} too short")
        print(f"[amp demos] excluded by name: {', '.join(by_name) if by_name else '-'}")
        print(f"[amp demos] demonstrations: {', '.join(demos)}")
        if self._motion_is_syn is not None:
            syn = self._motion_is_syn.tolist()
            syn_demos = [s for i, s in zip(ids.tolist(), demos) if syn[i]]
            print(f"[amp demos] lineage-matched motions among the demonstrations: {len(syn_demos)}"
                  + (f" ({', '.join(syn_demos)})" if syn_demos else ""))

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
    def setup_lineage(self) -> None:
        """Build the per-motion lineage tables once; print what each rule matched (nothing without rules)."""
        if self._lineage_ready:
            return
        self._lineage_ready = True
        rules = dict(self.config.amp_lineage_weights or {})
        if not rules:
            return
        stems = self._stems()
        weights, which = lineage_match(stems, rules)
        self._motion_amp_weight = torch.tensor(weights, device=self.device)
        self._motion_is_syn = torch.tensor([k >= 0 for k in which], device=self.device, dtype=torch.bool)
        self._lineage_meter = LineageRewardMeter(self.device)
        log.info("AMP lineage weights: %s (motions not matched: 1.0)", rules)
        for k, (pattern, w) in enumerate(rules.items()):
            hits = [s for s, j in zip(stems, which) if j == k]
            if hits:
                print(f"[amp lineage] rule {pattern!r} (w {w:g}) matched {len(hits)} of {len(stems)} motions: "
                      f"{', '.join(hits)}")
            else:
                # Say why: either no stem contains the pattern, or an earlier rule took every one that does.
                shadowed = sum(1 for s, j in zip(stems, which) if pattern in s and 0 <= j < k)
                why = (f"{shadowed} stem(s) contain it, all taken by an earlier rule" if shadowed
                       else "no stem contains it")
                print(f"[amp lineage] WARNING: rule {pattern!r} (w {w:g}) matched no motion "
                      f"(of {len(stems)}; {why})")
        n_syn = sum(j >= 0 for j in which)
        print(f"[amp lineage] {n_syn} of {len(stems)} motions are 'syn' (matched); {len(stems) - n_syn} are "
              f"'human' (w 1.0)")

    def _env_amp_weight(self) -> Optional[Tensor]:
        self.setup_lineage()
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
            raw_rewards = amp_rewards
            amp_rewards = amp_rewards * weight
            # The env's motion at reward time: resets happen at the next step's start, so this is the motion
            # the transition tracked -- the same ids the weight was gathered with.
            if self._lineage_meter is not None:
                self._lineage_meter.record(raw_rewards, amp_rewards,
                                           self._motion_is_syn[self.agent.motion_manager.motion_ids])

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
        if self._lineage_meter is not None:
            training_log_dict.update(self._lineage_meter.pop_log())

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

    def load_warm_start_state(self, state_dict) -> List[str]:
        """A warm start's AMP state (card E6); returns the lines to print.

        ``load_training_state`` restores exactly the discriminator and disc-critic optimisers (Adam moments,
        step counts and the checkpoint's learning rates), the AMP reward normaliser (mean/var/count) and the
        weight calibration. The networks themselves, their observation normalisers included, came with the
        model state dict. Nothing else the component owns is checkpointed, so these start fresh: the replay
        buffer (the first epoch's own samples stand in), the per-env bad-transition counters, the
        demonstration sampler and lineage tables (built from this run's library and config), and
        ``current_w`` (recomputed every epoch). Two things the resume path keeps are wrong here: the
        calibration's ratio history counts epochs on the old run's clock (this run restarts at epoch 0, so a
        calibration window would read the old run's ratios), and the calibrated target must yield to the
        configured one when calibration is off (``warm_start_calibration``).
        """
        self.load_training_state(state_dict)
        cfg = self.config
        dropped = len(self.ratio_history)
        self.ratio_history = []
        self.calibrated_target, weight_note = warm_start_calibration(
            self.calibrated_target, cfg.amp_reward_w_target, getattr(cfg, "amp_calibrate_style_ratio", 0.0))
        parts = [f"discriminator optimizer (step {optimizer_step_count(self.discriminator_optimizer)}, "
                 f"lr {self.discriminator_optimizer.param_groups[0]['lr']:g})"]
        if self.use_disc_critic:
            parts.append(f"disc-critic optimizer (step {optimizer_step_count(self.disc_critic_optimizer)}, "
                         f"lr {self.disc_critic_optimizer.param_groups[0]['lr']:g})")
        if cfg.normalize_rewards:
            norm = self.running_reward_norm
            parts.append(f"AMP reward normaliser (count {int(norm.count)}, var {float(norm.var.flatten()[0]):.4g})")
        return [f"Warm start: restored AMP training state from checkpoint: {', '.join(parts)}",
                f"Warm start: {weight_note}; {dropped} calibration ratio-history entries from the old run's "
                f"epoch clock dropped"]


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

    def setup(self):
        super().setup()
        # The lineage tables need only the library: build them (and print what each rule matched) now, not
        # at the first rollout step. A no-op without rules.
        AMPTrainingComponent.for_agent(self).setup_lineage()

    def _load_optimization_state(self, state_dict):
        """Warm start (``--warm-start-optimization-state``): PPO's state, then the AMP component's (card E6).

        PPO's restores the task reward normaliser, the actor/critic optimisers and the advantage EMA. A
        checkpoint without a discriminator, or without its training state (a PPO checkpoint), leaves the AMP
        state fresh with a printed note. One that carries only part of it raises.
        """
        super()._load_optimization_state(state_dict)
        component = AMPTrainingComponent.for_agent(self)
        fresh = ("the discriminator/disc-critic optimisers, the AMP reward normaliser and the weight "
                 "calibration start fresh")
        if getattr(self, "_warm_start_from_non_amp_checkpoint", False):
            print(f"Warm start: no AMP training state loaded -- the checkpoint has no discriminator (a PPO "
                  f"checkpoint); {fresh}")
            return
        present, missing = amp_training_state_keys(state_dict, component.use_disc_critic,
                                                   bool(self.config.normalize_rewards))
        if not present:
            print(f"Warm start: no AMP training state loaded -- the checkpoint carries none; {fresh}")
            return
        if missing:
            raise KeyError(f"Warm start: the checkpoint carries part of the AMP training state ({present}) "
                           f"but not {missing}")
        for line in component.load_warm_start_state(state_dict):
            print(line)

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
