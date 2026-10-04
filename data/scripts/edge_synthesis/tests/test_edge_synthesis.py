"""Tests of lane T's building blocks (CPU only, ~1 min).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=.:data/scripts python -m pytest data/scripts/edge_synthesis/tests -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from edge_synthesis import plant_mj as pm
from edge_synthesis import sketch as SK


@pytest.fixture(scope="module")
def plant():
    return pm.Plant(nthread=1)


def test_edges_json_reproduces_and_checks():
    """T0: the spec re-derives to the written file, every hold resolves and §1.3's contact edits hold."""
    from edge_synthesis import edge_spec

    spec = edge_spec.build()
    assert edge_spec.check(spec) == []
    old = json.load(open(edge_spec.OUT))
    strip = lambda d: {k: v for k, v in d.items() if k not in ("generator", "inputs")}  # noqa: E731
    assert json.dumps(strip(old), sort_keys=True) == json.dumps(strip(json.loads(json.dumps(spec))), sort_keys=True)


def test_fk_matches_poselib(plant):
    r = pm.fk_test(plant, n=20, seed=1)
    assert r["mj_kinematics_vs_poselib_m"] <= 1e-6 and r["numpy_fk_vs_poselib_m"] <= 1e-6
    assert r["expmap_round_trip_rot"] <= 1e-9


def test_hinge_orders_round_trip(plant):
    """Every body's hinge order (HINGE_ORDER) reproduces any local rotation, on both Euler branches."""
    rng = np.random.default_rng(0)
    R = Rotation.random(23 * 50, random_state=rng.integers(1 << 31)).as_matrix().reshape(50, 23, 3, 3)
    h = plant.hinge_from_local(R)
    q = np.zeros((50, plant.nq))
    q[:, 3] = 1.0
    q[:, 7:] = h
    assert np.abs(plant.local_rotations(q) - R).max() < 1e-9


def test_knee_sweep_is_branch_continuous(plant):
    """A knee folding through 90 deg with a little abduction never flips its hinge angles (the reason for
    HINGE_ORDER: the MJCF's x-y-z order puts knee flexion in the singular middle slot)."""
    ki = plant.body_names.index("L_Knee") - 1
    prev, worst = None, 0.0
    for th in np.linspace(0.0, 2.6, 200):
        L = np.tile(np.eye(3), (23, 1, 1))
        L[ki] = (Rotation.from_euler("y", th) * Rotation.from_euler("x", 0.05)).as_matrix()
        h = plant.hinge_from_local(L[None], None if prev is None else prev[None])[0]
        if prev is not None:
            worst = max(worst, float(np.abs(h - prev).max()))
        prev = h
    assert worst < np.radians(1.0)


def test_box_excess_takes_the_better_representative(plant):
    """A rotation stored the long way round is not called a box violation (retarget.nearest_representative)."""
    i = plant.dof_names.index("L_Knee_y")
    dof = np.zeros(69)
    dof[i] = np.radians(150.0)
    assert plant.box_excess(dof).max() == 0.0
    v = dof.copy()
    v[i] = np.radians(150.0) - 2 * np.pi                       # the same rotation as a -210 deg turn
    assert plant.box_excess(v).max() == 0.0


def test_friction_is_isotropic_0p75():
    r = pm.slide_test(tilts=(0.72, 0.78), azimuths_deg=(0.0, 45.0), seconds=0.6)
    assert r["pass"], r


def test_sketch_keeps_hands_and_ends_on_d(plant):
    e = SK.edge(SK.load_edges(), "B1")
    ep = SK.endpoints(plant, e)
    sched = SK.schedule(e, SK.timing(e, "mid"))
    sk = SK.Sketch(plant, [(0.0, ep.src_qpos), (sched.T, ep.dst_qpos)])
    sk.build_grid(t0=-0.5, t1=sched.T + 1.0)
    t = np.linspace(-0.2, sched.T + 0.5, 40)
    pos, _ = plant.fk(sk.qpos(t))
    bi = plant.body_index
    mid = pos[:, [bi["L_Hand"], bi["R_Hand"]]].mean(1)
    assert np.abs(mid - mid[0]).max() < 1e-4          # exact on the 240 Hz grid; ~2 um between its points
    end, _ = plant.fk(sk.qpos(np.array([sched.T + 0.4])))
    assert np.abs(end[0] - ep.dst_pos).max() < 1e-3


def test_schedule_masks_jump_back():
    """E2: the feet are free while airborne, may touch down only inside their event window, then are planted."""
    e = SK.edge(SK.load_edges(), "E2")
    sched = SK.schedule(e, SK.timing(e, "mid"))
    te = [t for t, z, k in sched.events if z == "L_FOOT" and k == "make"][0]
    m = sched.masks(np.array([te - 0.3, te, te + 0.3]))
    lf = SK.ZONE_ORDER.index("L_FOOT")
    assert m["free"][0, lf] and not m["ground"][0, lf]
    assert not m["free"][1, lf] and m["window"][1, lf]
    assert m["ground"][2, lf]
