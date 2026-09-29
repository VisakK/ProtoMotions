# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures of the reference-curation tests.

``stores`` builds capture store v1, v2 and v3 of every clip of the manifest, and the Step 2 audit,
into one temp dir per session, with the committed calibrations: Step 6's tests read labels and
replay the Pass-B calibration over the whole corpus. Nothing under ``output/`` or
``data/reference_curation/`` is written. About two minutes, on 8 spawned workers.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from reference_curation import audit, capture, human_mesh as hm, ids, labels, packets, sources

WORKERS = min(8, os.cpu_count() or 1)


@pytest.fixture(scope="session")
def stores(tmp_path_factory):
    if not (ids.MOYO_DATA / "mosh").exists() or not hm.MODEL_PATH.exists() or not ids.SHIPPED_DIR.exists():
        pytest.skip("the MOYO MoSh fits, the SMPL-X model or the shipped ftC clips are not on disk")
    out = tmp_path_factory.mktemp("stores")
    stems = ids.manifest_stems()
    cal_v1, _, failures = capture.build_all(stems, out / "v1", out / "capture_v1.json")
    assert failures == []
    records, context, failures = audit.audit(store_dir=out / "v1", calibration=cal_v1)
    assert failures == []
    aud = packets.load_audit(audit.write(records, context, out / "audits"))
    st = labels.Stores(out / "v1", out / "v2", out / "v3")
    measured, failures = hm.measure_all(stems, WORKERS)
    assert failures == []
    cal_v2 = hm.load_calibration()
    v2 = {m["meta"]["stem"]: hm.build(m["meta"]["stem"], cal_v2, st.v2, measured=m,
                                      base=capture.load(m["meta"]["stem"], st.v1, cal_v1)) for m in measured}
    v3, failures = sources.build_all(stems, st.v3, WORKERS, bases=v2)
    assert failures == []
    return SimpleNamespace(out=out, cal_v1=cal_v1, aud=aud, stores=st, v3={r.meta["stem"]: r for r in v3})
