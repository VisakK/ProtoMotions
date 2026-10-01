# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plant identity (BodyFix Step 2): data built for one plant raises when it meets another.

A plant is its MJCF's bytes. Motions, motion libraries and physics tables carry the sha256 of the MJCF they were
built on; data without one predates plant v2 and was built on v1. The deliberate mismatches here are the card's
acceptance test: v1 or legacy data on the v2 robot, v2 data on the v1 robot, a rebuilt XML, and a library that
mixes plants all raise ``PlantMismatchError``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.envs.control.physics_terms import PhysicsTables
from protomotions.tests.test_motion_lib_helpers import _motion_file_payload
from protomotions.utils import plant_identity as pi
from protomotions.utils.component_builder import require_motion_plant

V1, V2 = (pi.mjcf_path(name) for name in ("v1", "v2"))
pytestmark = pytest.mark.skipif(not (V1.is_file() and V2.is_file()), reason="plant v1/v2 MJCFs are not on disk")


def _robot(plant: str) -> SimpleNamespace:
    return SimpleNamespace(asset=SimpleNamespace(asset_root="data/assets",
                                                 asset_file_name=pi.PLANTS[plant][0].split("data/assets/", 1)[1]))


def test_the_two_plants_are_registered_and_distinct():
    assert pi.sha256("v1") != pi.sha256("v2")
    assert pi.sha256("v2") == pi.sha256(V2) == pi.sha256(str(V2.relative_to(pi.REPO)))
    assert pi.name_of(pi.sha256("v1")) == "v1" and pi.name_of(pi.sha256("v2")) == "v2" and pi.name_of("0" * 64) is None
    assert pi.identity("v2") == pi.identity(V2) == {"plant": "v2", "mjcf": pi.PLANTS["v2"][0], pi.KEY: pi.sha256("v2")}
    assert pi.flat_path("v2").name == "smpl_yogi03596_v2_flat.xml" and pi.flat_path(V2) == pi.flat_path("v2")


def test_offline_scripts_select_the_plant_with_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(pi.ENV_VAR, raising=False)
    assert pi.selected() == "v1" and pi.mjcf_path() == V1          # finished records were all built on v1
    monkeypatch.setenv(pi.ENV_VAR, "v2")
    assert pi.mjcf_path() == V2 and pi.flat_path().name == "smpl_yogi03596_v2_flat.xml"
    assert pi.default_mjcf() == pi.PLANTS["v2"][0]
    monkeypatch.setenv(pi.ENV_VAR, "v3")
    with pytest.raises(ValueError):
        pi.selected()
    other = tmp_path / "robot.xml"
    other.write_bytes(V2.read_bytes())
    monkeypatch.setenv(pi.ENV_VAR, str(other))
    assert pi.mjcf_path() == other and pi.flat_path() == tmp_path / "robot_flat.xml"


def test_legacy_data_is_plant_v1():
    assert pi.require(None, V1, "legacy") == "v1"
    with pytest.raises(pi.PlantMismatchError, match="predates plant v2"):
        pi.require(None, V2, "legacy motion")


def test_data_of_one_plant_is_refused_by_the_other():
    assert pi.require(pi.sha256("v2"), V2, "v2 data") == "v2"
    assert pi.require(pi.sha256("v1"), V1, "v1 data") == "v1"
    with pytest.raises(pi.PlantMismatchError, match="built for plant v1"):
        pi.require(pi.sha256("v1"), V2, "v1 data")
    with pytest.raises(pi.PlantMismatchError, match="built for plant v2"):
        pi.require(pi.sha256("v2"), V1, "v2 data")


def test_a_rebuilt_xml_is_another_plant(tmp_path):
    rebuilt = tmp_path / "smpl_yogi03596_v2.xml"
    rebuilt.write_bytes(V2.read_bytes() + b"\n")
    assert pi.name_of(pi.sha256(rebuilt)) is None
    with pytest.raises(pi.PlantMismatchError):
        pi.require(pi.sha256("v2"), rebuilt, "v2 data")
    assert pi.require(None, rebuilt, "legacy data on an unregistered robot") is None      # no plant rule applies


def test_recorded_sha_reads_every_carrier():
    sha = pi.sha256("v2")
    assert pi.recorded_sha({pi.KEY: sha}) == sha
    assert pi.recorded_sha({"plant": pi.identity("v2")}) == sha
    assert pi.recorded_sha({"inputs": {pi.PLANTS["v1"][0]: pi.sha256("v1"), "other.json": "x"}}) == pi.sha256("v1")
    assert pi.recorded_sha({"inputs": {"other.json": "x"}}) is None and pi.recorded_sha(None) is None


def test_physics_tables_refuse_another_plant(tmp_path):
    for recorded, mjcf in ((None, V2), (pi.sha256("v1"), V2), (pi.sha256("v2"), V1)):
        path = tmp_path / "tables.pt"
        torch.save({} if recorded is None else {pi.KEY: recorded}, path)
        with pytest.raises(pi.PlantMismatchError):
            PhysicsTables(str(path), [], [], "cpu", plant_mjcf=str(mjcf))
    torch.save({pi.KEY: pi.sha256("v2")}, path)
    with pytest.raises(KeyError, match="motion_names"):       # the plant check passed; the payload is a stub
        PhysicsTables(str(path), [], [], "cpu", plant_mjcf=str(V2))


def _library(tmp_path, plants):
    for k, plant in enumerate(plants):
        payload = _motion_file_payload(4, offset=float(k))
        if plant is not None:
            payload[pi.KEY] = pi.sha256(plant)
        torch.save(payload, tmp_path / f"clip_{k}.motion")
    config = MotionLibConfig(motion_file=str(tmp_path))
    return MotionLib(config, device="cpu"), config


def test_motion_library_carries_its_plant(tmp_path):
    lib, config = _library(tmp_path, ["v2", "v2"])
    assert lib.plant_sha256 == pi.sha256("v2") and lib.num_motions() == 2
    require_motion_plant(lib, config, _robot("v2"))
    with pytest.raises(pi.PlantMismatchError, match="built for plant v2"):
        require_motion_plant(lib, config, _robot("v1"))
    packed = tmp_path / "packed" / "lib.pt"
    lib.save_to_file(packed)
    assert MotionLib(MotionLibConfig(motion_file=str(packed)), device="cpu").plant_sha256 == pi.sha256("v2")


def test_legacy_motion_library_is_refused_on_plant_v2(tmp_path):
    lib, config = _library(tmp_path, [None, None])
    assert lib.plant_sha256 is None
    require_motion_plant(lib, config, _robot("v1"))
    with pytest.raises(pi.PlantMismatchError, match="predates plant v2"):
        require_motion_plant(lib, config, _robot("v2"))
    packed = tmp_path / "packed" / "legacy.pt"
    lib.save_to_file(packed)
    old = torch.load(packed, weights_only=False)
    assert "plant_sha256" not in old                                  # a packed library from before plant identity
    assert MotionLib(MotionLibConfig(motion_file=str(packed)), device="cpu").plant_sha256 is None


def test_a_library_mixing_plants_is_refused(tmp_path):
    with pytest.raises(pi.PlantMismatchError, match="different plants"):
        _library(tmp_path, ["v2", None])


def test_empty_library_and_unregistered_robots_are_not_checked(tmp_path):
    empty_config = MotionLibConfig(motion_file=None)
    require_motion_plant(MotionLib(empty_config, device="cpu"), empty_config, _robot("v2"))
    lib, config = _library(tmp_path, [None])
    require_motion_plant(lib, config, SimpleNamespace(asset=SimpleNamespace(
        asset_root="protomotions/data/assets", asset_file_name="mjcf/smpl_humanoid.xml")))


def test_smpl_yogi_v2_robot_config_is_plant_v2():
    import mujoco
    import numpy as np

    from protomotions.robot_configs.factory import robot_config

    v1, v2 = robot_config("smpl_yogi"), robot_config("smpl_yogi_v2")
    assert pi.robot_mjcf(v2) == "data/assets/smpl/smpl_yogi03596_v2.xml" and pi.robot_mjcf(SimpleNamespace()) is None
    assert pi.name_of(pi.sha256(f"{v2.asset.asset_root}/{v2.asset.asset_file_name}")) == "v2"
    assert pi.name_of(pi.sha256(f"{v1.asset.asset_root}/{v1.asset.asset_file_name}")) == "v1"
    assert (pi.REPO / v2.asset.asset_root / v2.asset.usd_asset_file_name).is_file()
    assert v2.default_root_height == 0.975 and v1.default_root_height == 0.95
    m = mujoco.MjModel.from_xml_path(str(V2))
    lo = np.array([m.jnt_range[m.joint(n).id][0] for n in v2.kinematic_info.dof_names])
    assert np.allclose(v2.kinematic_info.dof_limits_lower.numpy(), lo)
    ci = v2.control.control_info
    assert (ci["Neck_x"].stiffness, ci["Neck_x"].damping) == (1158, 116)
    assert (ci["Head_x"].stiffness, ci["L_Wrist_x"].stiffness, ci["L_Knee_y"].stiffness) == (500, 300, 800)
    assert (ci["L_Wrist_x"].effort_limit, ci["R_Hand_z"].effort_limit, ci["L_Hip_x"].effort_limit) == (30, 15, 300)
    assert {k: (c.stiffness, c.damping) for k, c in ci.items() if not k.startswith("Neck_")} == \
        {k: (c.stiffness, c.damping) for k, c in v1.control.control_info.items() if not k.startswith("Neck_")}
