# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distillation from several Stage-1 experts, routed by motion clip.

``SupervisedAgent`` labels every rollout state with a single expert.  That is the
right thing when one tracker covers the whole corpus, and the wrong thing here:
the yoga corpus is covered by three trackers trained on disjoint slices of it
(non-balance poses, inversions + arm balances, single-leg standing balances), and
no one of them is competent on another's clips.

This agent holds all of them and labels each environment with the expert that
*owns the clip that environment is playing*.  The routing table is a per-motion-id
expert index, produced next to the packaged library by
``data/scripts/package_student_corpus.py`` and validated against the library's
clip names at construction -- a table built for a different packaging order would
silently label every action with the wrong expert.

Two properties make this cheap and safe:

* The experts are structurally identical (same observation contract, same actor
  architecture), so the environment computes **one** set of ``expert_*``
  observations and all three read it.  That is asserted, not assumed.
* PPO actors declare no per-environment rollout state, so an expert can be run on
  a *subset* of environments.  Routing therefore costs about one expert forward
  in total rather than one per expert.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from tensordict import TensorDict

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.common import weight_init_trainable
from protomotions.agents.supervised.agent import SupervisedAgent
from protomotions.agents.supervised.masked_mimic_config import (
    MaskedMimicSupervisedAgentConfig,
)
from protomotions.utils.hydra_replacement import get_class

log = logging.getLogger(__name__)


@dataclass
class MultiExpertMaskedMimicAgentConfig(MaskedMimicSupervisedAgentConfig):
    """MaskedMimic distillation with one expert per motion group.

    Attributes:
        expert_model_paths: Checkpoints, in the order the routing table indexes.
        motion_expert_file: JSON written by ``package_student_corpus.py`` holding
            ``motion_expert`` (expert index per motion id) and ``motion_names``.
    """

    _target_: str = "protomotions.agents.supervised.multi_expert.MultiExpertSupervisedAgent"

    expert_model_paths: List[str] = field(default_factory=list)
    motion_expert_file: Optional[str] = None


class MultiExpertSupervisedAgent(SupervisedAgent):
    """Supervised distillation that routes expert labelling by motion clip."""

    config: MultiExpertMaskedMimicAgentConfig

    def create_model(self):
        model_cls = get_class(self.config.model._target_)
        model: BaseModel = model_cls(config=self.config.model)
        if not getattr(model, "skip_default_weight_init", False):
            model.apply(weight_init_trainable)

        paths = list(self.config.expert_model_paths or [])
        if self.config.expert_model_path is not None:
            paths = [self.config.expert_model_path] + paths
        if not paths:
            # Inference: apply_inference_overrides clears the expert paths and
            # strips their observation components, and the student runs on its
            # prior alone. Training without experts is caught by fit().
            log.info("No expert checkpoints configured; running student-only.")
            self.expert_models = []
            self.expert_actors = []
            self.expert_actor_in_keys = []
            self.expert_model = None
            self.expert_actor = None
            self._motion_expert = None
            return model

        self.expert_models = []
        self.expert_actors = []
        self.expert_actor_in_keys = []
        for path in paths:
            expert_model, expert_actor, in_keys = self._build_external_expert(path)
            if self.expert_actor_in_keys and in_keys != self.expert_actor_in_keys:
                raise ValueError(
                    "experts disagree on their observation contract:\n"
                    f"  {paths[0]}: {self.expert_actor_in_keys}\n"
                    f"  {path}: {in_keys}\n"
                    "The environment computes one set of expert_* observations, so "
                    "every expert must consume the same keys."
                )
            self.expert_actor_in_keys = in_keys
            self.expert_models.append(expert_model)
            self.expert_actors.append(expert_actor)

        # SupervisedAgent branches on `expert_model is not None` to decide whether
        # to record expert labels at all; point it at the first expert so those
        # paths behave, while routing happens in _collect_external_expert_action.
        self.expert_model = self.expert_models[0]
        self.expert_actor = self.expert_actors[0]

        self._motion_expert = self._load_routing_table(len(self.expert_models))
        counts = torch.bincount(self._motion_expert, minlength=len(self.expert_models))
        log.info(
            "Multi-expert distillation over %d experts; clips per expert: %s",
            len(self.expert_models),
            counts.tolist(),
        )
        return model

    # ------------------------------------------------------------------ #
    def _load_routing_table(self, num_experts: int) -> torch.Tensor:
        """Per-motion-id expert index, checked against this library's clip names."""
        if not self.config.motion_expert_file:
            raise ValueError(
                "MultiExpertSupervisedAgent requires motion_expert_file so each "
                "clip can be labelled by the expert that owns it"
            )
        payload = json.loads(Path(self.config.motion_expert_file).read_text())
        table = payload["motion_expert"]
        names = payload.get("motion_names")

        library = [Path(f).stem for f in self.env.motion_lib.motion_files]
        if len(table) != len(library):
            raise ValueError(
                f"routing table covers {len(table)} motions but the library holds "
                f"{len(library)}; rebuild it for this motion file"
            )
        if names is not None and names != library:
            mismatch = next(
                (i for i, (a, b) in enumerate(zip(names, library)) if a != b), None
            )
            raise ValueError(
                f"routing table was built for a different clip order (first "
                f"mismatch at motion {mismatch}: '{names[mismatch]}' vs "
                f"'{library[mismatch]}')"
            )
        tensor = torch.tensor(table, dtype=torch.long, device=self.device)
        if int(tensor.max()) >= num_experts or int(tensor.min()) < 0:
            raise ValueError(
                f"routing table indexes experts [{int(tensor.min())}, "
                f"{int(tensor.max())}] but only {num_experts} were loaded"
            )
        return tensor

    def _collect_external_expert_action(self, obs_td: TensorDict) -> torch.Tensor:
        """Label each environment with the expert that owns its clip."""
        expert_obs_td = self._build_expert_obs_td(obs_td, self.expert_actor_in_keys)
        motion_ids = self.env.motion_manager.motion_ids
        assignment = self._motion_expert[motion_ids]

        actions: Optional[torch.Tensor] = None
        for index, actor in enumerate(self.expert_actors):
            selected = (assignment == index).nonzero(as_tuple=True)[0]
            if selected.numel() == 0:
                continue
            output = actor(expert_obs_td[selected])
            if "mean_action" in output.keys():
                sub_actions = output["mean_action"]
            elif "action" in output.keys():
                sub_actions = output["action"]
            else:
                raise KeyError(
                    "External expert actor must produce either 'mean_action' or "
                    f"'action'. Available keys: {list(output.keys())}"
                )
            if actions is None:
                actions = torch.zeros(
                    obs_td.batch_size[0],
                    sub_actions.shape[-1],
                    device=sub_actions.device,
                    dtype=sub_actions.dtype,
                )
            actions[selected] = sub_actions

        if actions is None:  # pragma: no cover - only if num_envs == 0
            raise RuntimeError("no environments were assigned to any expert")
        return actions

    def eval(self):
        super().eval()
        for expert in getattr(self, "expert_models", []):
            expert.eval()

    def train(self):
        super().train()
        # Experts stay frozen in eval so their observation normalizers never
        # record moments from the student's rollout distribution.
        for expert in getattr(self, "expert_models", []):
            expert.eval()


__all__ = [
    "MultiExpertMaskedMimicAgentConfig",
    "MultiExpertSupervisedAgent",
]
