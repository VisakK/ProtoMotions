# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A PhysX ``RigidContactView`` over every robot body, filtered against the ground
*and* every other robot body.

Extracted from :mod:`record_contact_physics` so more than one recorder can use it.
That module parses its arguments and imports the simulator at module scope, so it
cannot be imported as a library; this one deliberately imports nothing beyond
numpy/torch and is safe to import from any recorder *after* the simulator has been
brought up.

The contact sensors ProtoMotions builds are per-body and filtered against the
terrain only, with ``track_contact_points`` off, so they can report neither
body-body forces (crow pose: knee on upper arm) nor contact points.  This view
adds both without touching the trained configuration.
"""

from __future__ import annotations

import re

import numpy as np
import torch

GROUND_FILTER_PATH = "/World/ground/terrain/mesh"
GROUND_NAME = "ground"


class PairContactView:
    """``RigidContactView`` with one filter per robot body plus the terrain mesh.

    Filter 0 is always the ground; filters ``1..B`` are the robot's own bodies in
    ``body_names`` order.
    """

    def __init__(self, sim_view, body_root_glob: str, body_names: list[str],
                 num_envs: int, max_points_per_pair: int):
        self.body_names = list(body_names)
        self.num_envs = num_envs
        num_bodies = len(body_names)
        pattern = f"{body_root_glob}({'|'.join(body_names)})"
        filters = [GROUND_FILTER_PATH] + [f"{body_root_glob}{b}" for b in body_names]
        self.view = sim_view.create_rigid_contact_view(
            pattern,
            filter_patterns=filters,
            max_contact_data_count=max_points_per_pair * num_bodies * num_envs,
        )
        self.filter_names = [GROUND_NAME] + list(body_names)

        sensor_paths = list(self.view.sensor_paths)
        if len(sensor_paths) != num_envs * num_bodies:
            raise RuntimeError(
                f"pair contact view matched {len(sensor_paths)} sensors, "
                f"expected {num_envs * num_bodies}"
            )
        if self.view.filter_count != len(filters):
            raise RuntimeError(
                f"pair contact view has filter_count={self.view.filter_count}, "
                f"expected {len(filters)} (1 ground + {num_bodies} bodies)"
            )
        # PhysX pairs each sensor with its own copy of every filter pattern. When
        # the pattern expands to a different number of prims than the sensor set
        # can absorb it silently hands back empty paths and every body-body
        # column stays zero, so refuse to record rather than log a lie.
        resolved = list(self.view.filter_paths)
        if resolved and isinstance(resolved[0], (list, tuple)):
            unresolved = [i for i, p in enumerate(resolved[0]) if not str(p)]
            if unresolved:
                missing = [filters[i] for i in unresolved[:3]]
                raise RuntimeError(
                    f"{len(unresolved)}/{len(filters)} contact filters did not resolve "
                    f"to a prim (e.g. {missing}). This happens with num_envs > 1; "
                    "record one motion per process."
                )

        # Row -> (env, body).  Resolved from the prim paths rather than assumed,
        # because PhysX does not promise env-major ordering.
        body_index = {b: i for i, b in enumerate(body_names)}
        row_env = np.zeros(len(sensor_paths), dtype=np.int64)
        row_body = np.zeros(len(sensor_paths), dtype=np.int64)
        for row, path in enumerate(sensor_paths):
            match = re.search(r"/env_(\d+)/", path)
            if match is None:
                raise RuntimeError(f"cannot parse env index from sensor path '{path}'")
            row_env[row] = int(match.group(1))
            leaf = path.rsplit("/", 1)[-1]
            if leaf not in body_index:
                raise RuntimeError(f"unexpected sensor leaf '{leaf}' in '{path}'")
            row_body[row] = body_index[leaf]
        self.row_env = row_env
        self.row_body = row_body
        # Flat scatter index: row -> env * num_bodies + body.
        self.row_to_slot = torch.as_tensor(row_env * num_bodies + row_body)
        self.num_bodies = num_bodies
        self.sensor_paths = sensor_paths
        try:
            self.filter_paths = list(self.view.filter_paths)
        except Exception:  # pragma: no cover - backend dependent
            self.filter_paths = []

    def describe(self) -> str:
        return (
            f"sensors={self.view.sensor_count} filters={self.view.filter_count} "
            f"max_contact_data={self.view.max_contact_data_count}\n"
            f"  first sensor paths: {self.sensor_paths[:3]}\n"
            f"  filter paths[:3]  : {self.filter_paths[:3]}"
        )


def gather_pair_buffer(counts, starts, *columns):
    """Flatten PhysX's ``(count, start_index)`` per-pair contact buffers.

    Returns ``(rows, cols, gathered_columns)`` where ``rows``/``cols`` index the
    (sensor, filter) pair each contact point belongs to.
    """
    device = counts.device
    mask = counts > 0
    if not bool(mask.any()):
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, [c[:0] for c in columns]
    rows, cols = mask.nonzero(as_tuple=True)
    count = counts[rows, cols].to(torch.long)
    start = starts[rows, cols].to(torch.long)
    total = int(count.sum())
    pair_ids = torch.repeat_interleave(torch.arange(rows.numel(), device=device), count)
    block_starts = count.cumsum(0) - count
    deltas = torch.arange(total, device=device) - block_starts.repeat_interleave(count)
    flat = start[pair_ids] + deltas
    gathered = [c.index_select(0, flat) for c in columns]
    return rows[pair_ids], cols[pair_ids], gathered
