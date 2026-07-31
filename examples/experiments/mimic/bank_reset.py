# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bank-based resets: start training episodes from *real* states, not the reference.

Why (``notes/brainstorm_skill_graph.MD`` §4.2-4.3): in composed operation an
edge never starts from its reference's frame 0 -- it starts from wherever the
source node actually is (a drooped, wandering hold state), and a target node
never receives its frozen pose -- it receives the edge's arrival. Training
either policy only from its reference start leaves exactly that boundary
distribution unseen. This module makes a configurable fraction of resets draw
the *physical* initial state from a pre-collected bank
(``data/scripts/collect_state_bank.py``) while the reference clock still starts
where the motion manager put it -- so ``progress_buf`` and every
progress-indexed term in ``edge_terms.py`` remain valid.

Coordinates: banks store root_pos in clip-local frame (the collecting env's
``respawn_root_offset`` -- which includes the ``ref_respawn_offset`` z-lift --
subtracted). At injection the current env's own offset is added back, exactly
mirroring ``move_reset_robot_obj_states_to_respawn_position``. Node and edge
clips are cut from one source with no re-centering, so they share this frame
(``notes/Skill_graph_handstand_lessons.MD`` §2.1).

Reset noise, when configured, is applied by ``BaseEnv.reset`` *after*
``compute_ref_reset_state`` returns -- i.e. on top of bank states too, which is
intended (the bank is a distribution, not a set of shrines).
"""
from dataclasses import dataclass
from typing import Optional

import torch

from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.base_env.env import BaseEnv

BANK_KEYS = ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel")


@dataclass
class BankResetEnvConfig(EnvConfig):
    """EnvConfig + state-bank reset fields. Picklable from this module like
    NodeValueJoin -- eval scripts already put examples/experiments/mimic on
    sys.path before torch.load."""

    _target_: str = "bank_reset.BankResetEnv"
    bank_file: Optional[str] = None
    bank_prob: float = 0.0


class BankResetEnv(BaseEnv):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._bank = None
        self._bank_size = 0

    def _load_bank(self):
        d = torch.load(self.config.bank_file, map_location="cpu", weights_only=False)
        missing = [k for k in BANK_KEYS if k not in d]
        if missing:
            raise ValueError(f"state bank {self.config.bank_file} missing keys {missing}")
        self._bank = {k: d[k].to(self.device) for k in BANK_KEYS}
        self._bank_size = self._bank["root_pos"].shape[0]
        meta = d.get("meta", {})
        print(f"[bank_reset] loaded {self._bank_size} states from {self.config.bank_file} "
              f"(source: {meta.get('source_checkpoint', '?')}, "
              f"filter max_err: {meta.get('filter_max_err', '?')})")

    def compute_ref_reset_state(self, env_ids, motion_ids, motion_times, sample_flat=False):
        new_states, new_object_states = super().compute_ref_reset_state(
            env_ids, motion_ids, motion_times, sample_flat
        )
        p = float(getattr(self.config, "bank_prob", 0.0) or 0.0)
        if p <= 0.0 or not getattr(self.config, "bank_file", None):
            return new_states, new_object_states
        if self._bank is None:
            self._load_bank()

        m = torch.rand(len(env_ids), device=self.device) < p
        if m.any():
            idx = torch.randint(0, self._bank_size, (int(m.sum()),), device=self.device)
            # Same transform the parent applied to the reference state: clip-local
            # frame + this env's full respawn offset.
            offs = self.respawn_root_offset[env_ids[m]]
            new_states.root_pos[m] = self._bank["root_pos"][idx] + offs
            new_states.root_rot[m] = self._bank["root_rot"][idx]
            new_states.root_vel[m] = self._bank["root_vel"][idx]
            new_states.root_ang_vel[m] = self._bank["root_ang_vel"][idx]
            new_states.dof_pos[m] = self._bank["dof_pos"][idx]
            new_states.dof_vel[m] = self._bank["dof_vel"][idx]
        return new_states, new_object_states
