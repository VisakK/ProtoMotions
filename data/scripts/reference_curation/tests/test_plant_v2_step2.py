# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for plant v2 as the training plant (BodyFix Step 2).

They pin what ``data/scripts/build_plant_v2_usd.py`` (the USD package, checked without PhysX) and
``reference_curation.plant_v2_physx`` (the robot as IsaacLab builds it) measured, and re-derive what can be
re-derived without a simulator: the package on disk is the one the records describe, it has exactly one
articulation root, and its physics layer is the MJCF. The offline scripts' plant selection is checked in
subprocesses (the modules read ``REFERENCE_PLANT`` at import). The deliberate-mismatch tests are
``protomotions/tests/test_plant_identity.py``. Nothing is written.

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests/test_plant_v2_step2.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import build_plant_v2_usd as U
from protomotions.utils import plant_identity as pi
from reference_curation import ids

PHYSX = ids.DATA_ROOT / "plant_v2" / "physx_smpl_yogi_v2.json"
pytestmark = pytest.mark.skipif(not (U.RECORD.exists() and PHYSX.exists() and U.USDA.exists()),
                                reason="plant v2's USD package or its records are not built")


@pytest.fixture(scope="module")
def usd():
    return json.loads(U.RECORD.read_text())


@pytest.fixture(scope="module")
def physx():
    return json.loads(PHYSX.read_text())


def test_records_describe_the_package_on_disk(usd, physx):
    assert usd["failures"] == [] and physx["failures"] == []
    assert usd["layer_sha256"] == {k: ids.sha256_file(p) for k, p in {"usda": U.USDA, **U.LAYERS}.items()}
    for rel, sha in {**usd["inputs"], **physx["inputs"]}.items():
        if rel.startswith("data/assets/"):
            assert ids.sha256_file(ids.REPO / rel) == sha, rel
    assert physx["plant"] == pi.identity("v2") and physx["robot"] == "smpl_yogi_v2"


def test_conversion_ran_every_post_processing_step(usd):
    c = usd["convert"]
    assert c["returncode"] == 0 and c["masses_returncode"] == 0 and not c["cleaned_xml_left_behind"]
    assert c["flat_issues"] == [] and c["visual_mesh_patch"].startswith("not needed")
    # Isaac Sim printed its output (every layer saved); on 2026-09-30 it then hung on exit and the watchdog killed it
    assert c["importer"]["reported_output"] and c["importer"]["exit"] in ("exited", "killed: hung on exit after writing")
    assert usd["flatten"]["flattened_main_vs_flat"] == [] and usd["flatten"]["main_vs_flat_beyond_motor_ctrlrange"] == []


def test_one_articulation_root_on_the_pelvis():
    roots = U.composed_roots()
    assert roots["active_articulation_roots"] == ["/smpl_yogi03596_v2_flat/Pelvis/Pelvis"]
    assert roots["usda_has_worldbody_override"] and not roots["usda_has_cleaned_suffix"] and not roots["worldbody_active"]
    assert U._layer_roots(U.LAYERS["physics"]) == ["/smpl_yogi03596_v2_flat_cleaned/Pelvis/Pelvis"]


def test_physics_layer_is_the_mjcf():
    c = U.check_physics_layer()
    assert (c["bodies"], c["joints"]) == (24, 23)
    assert not (c["missing_bodies"] or c["extra_bodies"] or c["collider_errors"] or c["joint_errors"] or c["authored_masses"])
    assert c["density_max_rel_err"] < 1e-7 and abs(c["mjcf_total_mass_kg"] - 74.0) < 1e-5
    assert c["collider_pos_max_err_mm"] < 1e-5 and c["collider_rot_max_err_deg"] < 1e-5 and c["collider_size_max_err_mm"] < 1e-4
    assert c["body_rest_pos_max_err_mm"] < 1e-4 and c["joint_frame_max_err_mm"] < 1e-4
    assert c["limit_max_err_deg"] < 1e-10 and c["drive_max_force_max_err"] == 0.0


def test_the_adoption_collider_check_passes_on_v2(usd):
    a = usd["adoption_collider_check"]
    if "skipped" in a:
        pytest.skip("output/.../adoption_risks/usd_collider_check.py is not on disk")
    w = a["worst"]
    assert a["returncode"] == 0 and max(w["pos_err_mm"], w["box_extent_err_mm"], w["radius_err_mm"]) < 1e-5
    assert max(w["box_rot_err_deg"], w["capsule_axis_err_deg"]) < 1e-5


def test_physx_simulates_the_mjcf_masses_and_actuation(physx):
    p = physx["properties"]
    assert abs(p["total_physx_kg"] - 74.0) < 1e-5 and p["mass_max_rel_err"] < 1e-6          # card: 74.00, 0.1 %
    assert p["com_max_err_m"] < 1e-8 and p["inertia_max_rel_err"] < 1e-6
    e = p["dof_max_err"]
    assert e["limit_rad"] < 1e-7 and e["max_force_vs_mjcf"] == 0.0
    assert all(v < 1e-7 for k, v in e.items() if k.endswith("_vs_config_rel"))
    assert p["neck_gains"] == {f"Neck_{a}": [1158.0, 116.0] for a in "xyz"}
    assert {k: v for k, v in p["effort_by_joint"].items() if k[2:] in ("Wrist", "Hand")} == \
        {"L_Wrist": 30.0, "R_Wrist": 30.0, "L_Hand": 15.0, "R_Hand": 15.0}


def test_physx_fk_is_the_reference_writers_fk(physx):
    fk = physx["fk"]
    assert (fk["frames"], fk["holds"]) == (418, 300)
    assert fk["pos_max_err_m"] < 2e-6                     # card: 1e-5 m; measured 1.4e-6 (float32)
    assert fk["rot_max_err_deg"] < 1e-3 and fk["dof_readback_max_err_rad"] == 0.0
    assert fk["writer_float32_vs_float64_m"] < 1e-6


def test_reset_does_not_launch(physx):
    r = physx["reset"]
    s = r["summary"]
    d = s["default"]
    assert d["launched"] == 0 and d["rise_max_cm"] == 0.0 and d["lowest_collider_at_reset_cm"] == 0.05
    assert r["standing_holds"] == 136 and s["standing"]["launched_cases"] == 0 and s["standing"]["overshoot_max_cm"] == 0.0
    # the positive control is seen; shallow sinks are restored to the surface, not thrown; 5 cm falls through
    assert s["thrown_1ms"]["launched"] == s["thrown_1ms"]["replicas"] > 0
    for k, depth in (("sunk_1cm", 1.0), ("sunk_2cm", 2.0)):
        assert s["sunk"][k]["launched"] == 0 and s["sunk"][k]["rise_min_cm"] >= 0.8 * depth
    assert s["sunk"]["sunk_5cm"]["rise_max_cm"] == 0.0 and s["sunk"]["sunk_5cm"]["root_drop_at_end_max_cm"] > 5.0


def _plant_state(plant: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != pi.ENV_VAR}
    if plant:
        env[pi.ENV_VAR] = plant
    code = ("import json; from reference_curation import ids; import static_hold_lp as S; "
            "print(json.dumps({'ids': str(ids.MJCF.relative_to(ids.REPO)), 'flat': str(S.FLAT.relative_to(ids.REPO)), "
            "'mjcf': str(S.MJCF.relative_to(ids.REPO)), 'kg': round(float(S.M.body_mass.sum()), 4), "
            "'wrist': float(S.TAU_MAX[S.JNAMES.index('L_Wrist_x')])}))")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(ids.REPO), capture_output=True, text=True,
                         check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_offline_scripts_follow_reference_plant():
    assert _plant_state(None) == {"ids": pi.PLANTS["v1"][0], "flat": pi.PLANTS["v1"][1], "mjcf": pi.PLANTS["v1"][0],
                                  "kg": 74.0, "wrist": 20.0}
    assert _plant_state("v2") == {"ids": pi.PLANTS["v2"][0], "flat": pi.PLANTS["v2"][1], "mjcf": pi.PLANTS["v2"][0],
                                  "kg": 74.0, "wrist": 30.0}


def test_capture_and_statics_records_are_refused_on_plant_v2():
    """The finished stores were built on v1 and say so (their provenance inputs hash the MJCF); Step 4's new store
    versions load them through ``ids.require_plant``, which refuses them on v2. ``capture.py`` is not edited: its
    source hash is the records' generator id, and a new one would rebuild every record and move the audit id."""
    capture = sorted((ids.OUTPUT_ROOT / "capture" / "v1").glob("*.json"))[:3]
    statics = sorted((ids.DATA_ROOT / "statics").glob("*/statics.json"))[:3]
    if not (capture and statics):
        pytest.skip("the capture / statics stores are not on disk")
    for path in capture + statics:
        record = json.loads(path.read_text())
        assert ids.plant_of(record) == pi.sha256("v1")
        assert ids.require_plant(record, path.name, mjcf=pi.mjcf_path("v1")) == "v1"
        with pytest.raises(pi.PlantMismatchError, match="built for plant v1"):
            ids.require_plant(record, path.name, mjcf=pi.mjcf_path("v2"))
