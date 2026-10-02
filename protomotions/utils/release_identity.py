# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which curation release a training run consumes, and a loud error when an artifact is not that release's.

A release (``reference_curation.release_v2``; BUILD_PLAN Step 10, BodyFix Step 5) is one immutable set of
training artifacts -- the packaged motion library, the hold graph, the physics tables, the contact-target
sidecar and the hold manifest the evaluator scores -- built together from pinned inputs. Its record
(``data/reference_curation/releases/<release_id>.json``) lists every artifact's sha256. The artifacts also
cross-reference each other (the graph names its package, the tables and the sidecar name their graph), but a
run assembled from paths could still pair a graph with a library rebuilt under the same names. So a run that
names its release (``ContactGraphControlConfig.release_file``) has every file it loads checked against the
record by content, and refuses to start on a mismatch.

The check reads whole files (the package is ~1 GB): sha256s are cached per (path, size, mtime) for the
process.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Optional

REPO = Path(__file__).resolve().parents[2]
RECORD_KIND = "reference_release"
ROLES = ("package", "graph", "physics_tables", "contact_targets", "holds_extended")


class ReleaseMismatchError(ValueError):
    """A file a run loads is not the artifact its release records."""


@functools.lru_cache(maxsize=256)
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _abs(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def file_sha256(path) -> str:
    """The sha256 of a file's bytes (cached per path, size and modification time)."""
    p = _abs(path).resolve()
    st = os.stat(p)
    return _sha256(str(p), st.st_size, st.st_mtime_ns)


def load_release(path) -> dict:
    """A release record, checked to be one."""
    record = json.loads(_abs(path).read_text())
    if record.get("kind") != RECORD_KIND or "artifacts" not in record or "release_id" not in record:
        raise ReleaseMismatchError(f"{path} is not a release record ({RECORD_KIND})")
    return record


def require_artifact(release: Mapping, role: str, path, what: Optional[str] = None) -> str:
    """Raise ``ReleaseMismatchError`` unless ``path``'s bytes are the release's ``role`` artifact. Returns its sha."""
    if role not in ROLES:
        raise ValueError(f"unknown artifact role {role!r}; expected one of {ROLES}")
    entry = release["artifacts"].get(role)
    if entry is None:
        raise ReleaseMismatchError(f"release {release['release_id']} has no {role} artifact")
    actual = file_sha256(path)
    if actual != entry["sha256"]:
        raise ReleaseMismatchError(
            f"{what or role} {path} is not release {release['release_id']}'s {role} "
            f"(sha256 {actual[:12]}, the release records {entry['sha256'][:12]} at {entry.get('path')}). "
            "Point the run at the release's own artifacts, or build a new release."
        )
    return actual
