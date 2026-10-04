"""Tests of the PhysX port's CPU parts: the torch cost and MPPI pieces against lane T's numpy originals.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=.:data/scripts python -m pytest data/scripts/edge_synthesis/tests -q

The PhysX plant itself needs IsaacLab (the GPU); its measured facts are in ``physx_plant``'s module doc and the
record (``t_edges_physx/README.MD``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from edge_synthesis import costs as C
from edge_synthesis import costs_torch as CT
from edge_synthesis import mppi as M
from edge_synthesis import plant_mj as pm
from edge_synthesis import sketch as SK


@pytest.fixture(scope="module")
def plant():
    return pm.Plant(nthread=2)


def _edge_setup(plant, edge_id: str, timing: str = "mid", via: str | None = None):
    from edge_synthesis import edge_mppi as EM

    e = SK.edge(SK.load_edges(), edge_id)
    ep = SK.endpoints(plant, e)
    sched = SK.schedule(e, SK.timing(e, timing))
    makes = sorted({te for te, _, kind in sched.events if kind == "make" and te > 0})
    land = ep.dst_qpos if via is None else EM.via_qpos(plant, ep, via)
    kfs = [(0.0, ep.src_qpos)] + ([(makes[0], land)] if makes and makes[0] < sched.T else []) + [(sched.T, ep.dst_qpos)]
    sk = SK.Sketch(plant, kfs)
    sk.build_grid(t0=-1.5, t1=sched.T + 4.0)
    return e, ep, sched, sk


@pytest.mark.parametrize("edge_id,t0", [("B1", 0.2), ("B1", 0.9), ("E2", 0.4)])
def test_edge_cost_torch_matches_numpy(plant, edge_id, t0):
    """Every term of ``EdgeCostTorch`` equals ``costs.EdgeCost``'s on the same MuJoCo rollouts (the plant-specific
    inputs -- torque utilisation in hinge coordinates, the box on the exp-map -- fed from MuJoCo's own)."""
    e, ep, sched, sk = _edge_setup(plant, edge_id)
    npc = C.EdgeCost(plant, sk, sched, ep.src_qpos, ep.dst_qpos)
    H, n, dt = 24, 6, 1.0 / pm.CTRL_HZ
    rng = np.random.default_rng(3)
    base = sk.pd_targets(t0 + dt * np.arange(H))
    ctrl = base[None] + 0.08 * rng.standard_normal((n, H, plant.nu))
    ctrl[0] = base
    s0, _ = plant.make_state(sk.qpos(np.array([t0]))[0])
    st, sd = plant.rollout(s0, ctrl)
    r = M.Rollouts(plant, st, sd, ctrl, t0)
    c_np, terms_np = npc(r)
    # a grid on which window j covers exactly r.t
    j = 5
    grid = CT.Grid(t0 - j * dt, dt, j + H + 3)
    assert np.allclose(grid.t[j + 1:j + 1 + H], r.t, atol=1e-12)
    ctc = CT.EdgeCostTorch(plant, sk, sched, ep.src_qpos, ep.dst_qpos, C.EdgeWeights(), grid, "cpu", H)
    pos, rot = r.fk()
    tau = plant.kp * (r.ctrl - r.qpos[..., 7:]) - plant.kd * r.qvel[..., 6:]
    _, _, dof = plant.expmap_from_qpos(r.qpos)
    f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)   # noqa: E731
    view = CT.View(pos=f32(pos), R=f32(rot), root_z=f32(r.qpos[..., 2]), com=f32(r.com), com_vel=f32(r.com_vel),
                   ang_mom=f32(r.ang_mom), fz=f32(r.force[..., 2]), fz_max=f32(r.force_z_max), ctrl=f32(r.ctrl),
                   util=f32(np.abs(tau) / plant.torque_limit), box=f32(plant.box_excess(dof)))
    c_t, terms_t = ctc(view, j)
    assert set(terms_t) == set(terms_np)
    for k in terms_np:
        a, b = np.asarray(terms_np[k], float), terms_t[k].double().numpy()
        assert np.allclose(a, b, rtol=2e-3, atol=1e-4 * max(1.0, np.abs(a).max())), (k, a, b)
    assert np.allclose(c_np, c_t.double().numpy(), rtol=2e-3)


def test_zone_lowest_torch(plant):
    from reference_curation import mosh_replay as mr

    sk = mr.skeleton_for(mr.V2_XML)
    pos, rot, _ = pm.release_frame("220923_Crane_Crow_Pose_or_Bakasana_-a", 651)
    q = plant.qpos_from_bodies(pos, rot)
    p, R = plant.fk(q[None])
    low_np, _ = C.zone_lowest(sk, p, R)
    low_t = CT.ZoneLowest(sk, "cpu")(torch.as_tensor(p, dtype=torch.float32), torch.as_tensor(R, dtype=torch.float32))
    assert np.abs(low_np - low_t.double().numpy()).max() < 1e-5


def test_mppi_basis_and_shift_match_numpy():
    """``MPPITorch``'s spline basis and the shifted warm start are ``mppi.MPPI``'s."""
    cfg = M.MPPIConfig()
    kt = np.linspace(0, cfg.horizon - 1, cfg.knots)
    W = M.basis(np.arange(cfg.horizon), kt, cfg.interp)
    Ws = M.basis(np.arange(cfg.horizon) + cfg.replan, kt, cfg.interp)
    rng = np.random.default_rng(0)
    off = rng.standard_normal((cfg.knots, 69))
    want = (Ws @ off)[np.round(kt).astype(int)]
    got = (torch.as_tensor(Ws, dtype=torch.float64) @ torch.as_tensor(off)[None])[:, np.round(kt).astype(int)][0]
    assert np.allclose(want, got.numpy())
    assert np.allclose(W.sum(1), 1.0)
