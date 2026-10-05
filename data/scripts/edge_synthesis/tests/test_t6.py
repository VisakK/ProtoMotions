"""Card T6's building blocks (CPU, ~1 min): the pin and landing-cone rows against finite differences, the frozen
frames, and the seam group on a clip that holds S's exemplar.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=.:data/scripts python -m pytest data/scripts/edge_synthesis/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from extract_contact_configs import ZONE_ORDER
from edge_synthesis import exact as X
from edge_synthesis import quasistatic as Q
from edge_synthesis import sketch as SK


@pytest.fixture(scope="module")
def sk():
    from reference_curation import retarget_v2 as rv2

    torch.set_num_threads(1)
    with rv2.on_plant("v2") as s:
        X._install()
        yield s


def _static_problem(sk, pose: Q.Pose, T: int = 3):
    con = Q.Contacts(np.zeros((T, len(ZONE_ORDER)), bool), [set()] * T, np.zeros(len(ZONE_ORDER), bool),
                     np.zeros(T, bool))
    return Q.build_problem(sk, "t6_test", np.zeros(T), np.repeat(pose.root_pos[None], T, 0),
                           np.repeat(pose.root_rot[None], T, 0), np.repeat(pose.dof[None], T, 0), con)


def test_pin_and_cone_rows_match_finite_differences(sk):
    """``pin_pos``, ``pin_rot`` and ``cone`` rows: the analytic Jacobian equals central differences of the residual."""
    from reference_curation import retarget as rt

    e = SK.edge(SK.load_edges(), "B1")
    S, D, _ = Q.endpoint_poses(sk, e)
    T = 3
    prob = _static_problem(sk, S, T)
    pn = X.pins(sk, np.arange(T), X.body_poses(sk, D, ("L_Hand", "R_Toe", "R_Wrist")))   # S pinned to D: residuals
    lp, lr = Q.bodies(sk, D)
    pts = rt.candidate_points(sk, torch.as_tensor(lp)[None], torch.as_tensor(lr)[None])[0].numpy()
    cands = np.asarray(sk.zone_cands[ZONE_ORDER.index("L_FOOT")])[:6]
    cone = {"frames": np.repeat(np.arange(T), len(cands)), "cand": np.tile(cands, T),
            "xy": np.tile(pts[cands, :2], (T, 1)), "h0": np.full(T * len(cands), 1.5),   # h0 high: every row active
            "w": np.ones(T * len(cands))}
    rng = np.random.default_rng(0)
    x = rt.initial(prob) + 0.02 * torch.as_tensor(rng.standard_normal((T, rt.NV)))

    def blocks(xx, jac):
        st = rt.kinematics(prob, xx)
        b = X.pin_rows(pn, st, jac)
        b["cone"] = X.cone_rows(cone, st, jac)
        return b
    b0 = blocks(x, True)
    assert all(len(b0[k]["r"]) for k in ("pin_pos", "pin_rot", "cone"))
    eps = 1e-6
    for col in [0, 4, 6 + 3 * 15, 6 + 3 * 17 + 1, 6 + 3 * 22 + 2, 6 + 3 * 7]:
        for f in range(T):
            dx = torch.zeros_like(x)
            dx[f, col] = eps
            bp, bm = blocks(x + dx, False), blocks(x - dx, False)
            for k in ("pin_pos", "pin_rot", "cone"):
                num = (bp[k]["r"] - bm[k]["r"]) / (2 * eps)
                sel = torch.as_tensor(b0[k]["frames"] == f)
                ana = b0[k]["J"][:, col]
                assert torch.allclose(num[sel], ana[sel], rtol=1e-4, atol=1e-4 * float(ana.abs().max() + 1)), (k, col, f)
                assert torch.allclose(num[~sel], torch.zeros_like(num[~sel]), atol=1e-6), (k, col, f)


def test_frozen_frames_are_exact(sk):
    """``freeze_x`` gives the variables whose ``unpack`` is the pose, and the wrapped bounds pin them (lo = hi)."""
    from reference_curation import retarget as rt
    from reference_curation import retarget_v2 as rv2

    e = SK.edge(SK.load_edges(), "E1")
    S, D, _ = Q.endpoint_poses(sk, e)
    T = 4
    prob = _static_problem(sk, D, T)
    mask = np.array([True, False, False, True])
    fz = X.freeze_x(prob, mask, lambda f: S)
    prob.plan["t6"] = {"pins": None, "cone": None, "freeze": fz}
    lo, hi = rv2.bounds(prob)
    assert torch.equal(lo[0], hi[0]) and torch.equal(lo[3], hi[3]) and not torch.equal(lo[1], hi[1])
    rp, rr, dof = rt.unpack(prob, fz["x"])
    with torch.no_grad():
        p, _ = rt.fk(sk, rp[mask], rr[mask], dof[mask])
    ps, _ = Q.bodies(sk, S)
    assert float((p - torch.as_tensor(ps)[None]).norm(dim=-1).max()) < 1e-9


def test_seam_group_on_a_held_exemplar():
    """A clip that holds S's exemplar: start and drift pass at 0; on E1 the end fails at the palms by exactly the
    exemplars' own offset (S's crow palms are turned ~12 deg from the handstand's), which ``end_attainable`` passes."""
    from reference_curation import fit_writer as fw
    from reference_curation import retarget_v2 as rv2
    from edge_synthesis import seams as SM

    e = SK.edge(SK.load_edges(), "E1")
    with rv2.on_plant("v2") as sk_:
        S, _, _ = Q.endpoint_poses(sk_, e)
    T = 30
    mot = fw.write_motion(np.repeat(S.root_pos[None], T, 0), np.repeat(S.root_rot[None], T, 0),
                          np.repeat(S.dof[None], T, 0), 60, "v2")
    sm = SM.check(mot, e)
    assert sm["start"]["pass"] and sm["start"]["all_max_cm"] < 0.01
    assert sm["drift"]["pass"] and sm["drift"]["max_cm"] < 0.01
    assert not sm["end"]["pass"] and sm["end"]["pass_attainable"] and sm["pass_attainable"]
    assert set(sm["end"]["beyond"]) == set(sm["end"]["end_exemplar_bodies"]) == {"L_Wrist", "R_Wrist"}
    assert 2.5 < sm["end"]["by_body"]["L_Wrist"]["to_D_cm"] < 3.2
    assert sm["end"]["by_body"]["L_Hand"]["to_D_cm"] < 1.0
    leg = SM.legacy(mot, e)
    assert leg["hands_start_cm"] < 0.01 and leg["hands_end_cm"] < 1.0
