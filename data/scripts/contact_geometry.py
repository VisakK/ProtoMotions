# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pairwise surface distances between MJCF collision geoms, for kinematic contact detection.

Unlike :mod:`clean_yoga_motions` (which flattens every geom to witness points +
radius, enough for ground clearance), this module keeps the geom *type* so that
body-body distances are exact: sphere-sphere / sphere-capsule / capsule-capsule
are closed-form, anything-vs-box goes through the box's exact signed distance
(ternary search along a capsule axis -- the SDF of a convex set is convex, so
the 1-D restriction is unimodal).  Box-box is the one approximate kernel: each
box is sampled with a 3x3 grid per face (54 points) against the other box's
exact SDF, both directions; error is well under the ~2 cm contact thresholds
for the skeleton's small boxes (feet 16x9x5 cm, hands 8x8.4x2.4 cm).

All kernels are torch, batched over frames, quaternions xyzw (w_last=True)
matching ``rigid_body_rot`` in ``.motion`` files.  Every pair kernel returns
``(gap, witness_a, witness_b)``: signed surface separation in meters
(negative = penetration) and the closest surface points in world coordinates.

Ground kernels assume the flat z=0 plane of the grounded corpora.
"""

import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F

from protomotions.utils.rotations import quat_mul, quat_rotate, quat_rotate_inverse


# --------------------------------------------------------------------------- #
# MJCF parsing (typed -- keeps sphere/capsule/box identity).
# --------------------------------------------------------------------------- #
def _mjcf_quat_to_xyzw(q_wxyz):
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)


def parse_typed_geoms(mjcf_path, body_names):
    """Dict body_name -> list of typed geom dicts (body-local frame).

    sphere: {type, center[3], radius}
    capsule: {type, seg[2,3], radius}
    box: {type, center[3], half[3], quat[4] xyzw}
    """
    tree = ET.parse(mjcf_path)
    worldbody = tree.getroot().find("worldbody")
    geoms = {name: [] for name in body_names}

    def _parse(geom_xml):
        gtype = geom_xml.attrib.get("type", "sphere")
        pos = geom_xml.attrib.get("pos")
        pos = np.fromstring(pos, dtype=np.float32, sep=" ") if pos else np.zeros(3, np.float32)
        quat = geom_xml.attrib.get("quat")
        if quat:
            quat = _mjcf_quat_to_xyzw(np.fromstring(quat, dtype=np.float32, sep=" "))
        else:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        size = geom_xml.attrib.get("size")
        size = np.fromstring(size, dtype=np.float32, sep=" ") if size else np.zeros(1, np.float32)
        fromto = geom_xml.attrib.get("fromto")

        density = float(geom_xml.attrib.get("density", 1000.0))
        if gtype == "sphere":
            r = float(size[0])
            vol = 4.0 / 3.0 * np.pi * r**3
            return {"type": "sphere", "center": pos, "radius": r, "mass": density * vol}
        if gtype == "capsule":
            r = float(size[0])
            if fromto:
                ft = np.fromstring(fromto, dtype=np.float32, sep=" ")
                seg = np.stack([ft[0:3], ft[3:6]])
            else:
                q = torch.tensor(quat).unsqueeze(0)
                axis = quat_rotate(q, torch.tensor([[0.0, 0.0, float(size[1])]]), w_last=True)[0].numpy()
                seg = np.stack([pos + axis, pos - axis])
            length = float(np.linalg.norm(seg[1] - seg[0]))
            vol = np.pi * r**2 * length + 4.0 / 3.0 * np.pi * r**3
            return {"type": "capsule", "seg": seg.astype(np.float32), "radius": r, "mass": density * vol}
        if gtype == "box":
            half = size[:3].astype(np.float32)
            vol = 8.0 * float(half[0] * half[1] * half[2])
            return {"type": "box", "center": pos, "half": half, "quat": quat, "mass": density * vol}
        raise ValueError(f"unsupported geom type '{gtype}'")

    def _recurse(body_xml):
        name = body_xml.attrib.get("name")
        if name in geoms:
            for geom_xml in body_xml.findall("geom"):
                geoms[name].append(_parse(geom_xml))
        for child in body_xml.findall("body"):
            _recurse(child)

    for root_body in worldbody.findall("body"):
        _recurse(root_body)
    return geoms


def geom_to_world(geom, body_pos, body_rot):
    """Body-local typed geom -> world-frame typed geom for a [T] batch of frames.

    body_pos [T,3], body_rot [T,4] xyzw.  Returned tensors are [T,...].
    """
    T = body_pos.shape[0]
    dev, dt = body_pos.device, body_pos.dtype
    mass = geom.get("mass", 0.0)
    if geom["type"] == "sphere":
        c = torch.as_tensor(geom["center"], device=dev, dtype=dt).expand(T, 3)
        return {
            "type": "sphere",
            "center": quat_rotate(body_rot, c, w_last=True) + body_pos,
            "radius": geom["radius"],
            "mass": mass,
        }
    if geom["type"] == "capsule":
        seg = torch.as_tensor(geom["seg"], device=dev, dtype=dt)
        a = quat_rotate(body_rot, seg[0].expand(T, 3), w_last=True) + body_pos
        b = quat_rotate(body_rot, seg[1].expand(T, 3), w_last=True) + body_pos
        return {"type": "capsule", "a": a, "b": b, "radius": geom["radius"], "mass": mass}
    if geom["type"] == "box":
        c = torch.as_tensor(geom["center"], device=dev, dtype=dt).expand(T, 3)
        q = torch.as_tensor(geom["quat"], device=dev, dtype=dt).expand(T, 4)
        return {
            "type": "box",
            "center": quat_rotate(body_rot, c, w_last=True) + body_pos,
            "quat": quat_mul(body_rot, q, w_last=True),
            "half": torch.as_tensor(geom["half"], device=dev, dtype=dt),
            "mass": mass,
        }
    raise ValueError(geom["type"])


def world_geom_center(g):
    """Volumetric center of a world-frame typed geom, [T,3]."""
    if g["type"] == "capsule":
        return (g["a"] + g["b"]) / 2
    return g["center"]


# --------------------------------------------------------------------------- #
# Low-level kernels.
# --------------------------------------------------------------------------- #
def _pt_seg(p, a, b):
    """Closest point on segment ab to p; all [T,3]."""
    ab = b - a
    t = ((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)
    return a + t.clamp(0.0, 1.0).unsqueeze(-1) * ab


def _seg_seg(p1, q1, p2, q2):
    """Closest points between segments p1q1 and p2q2 (Ericson 5.1.9); all [T,3]."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    denom = a * e - b * b
    s = torch.where(
        denom > 1e-12,
        ((b * f - c * e) / denom.clamp_min(1e-12)).clamp(0.0, 1.0),
        torch.zeros_like(a),
    )
    t = (b * s + f) / e.clamp_min(1e-12)
    t_cl = t.clamp(0.0, 1.0)
    # Recompute s when t was clamped -- or when segment 2 is degenerate (e ~ 0),
    # where t evaluates to exactly 0 and the t==t_cl shortcut would wrongly skip
    # the projection of the closest point onto segment 1.
    recompute = (t != t_cl) | (e <= 1e-12)
    s = torch.where(recompute, ((t_cl * b - c) / a.clamp_min(1e-12)).clamp(0.0, 1.0), s)
    return p1 + s.unsqueeze(-1) * d1, p2 + t_cl.unsqueeze(-1) * d2


def _pt_box(p, box):
    """Signed distance from world points p [T,3] to a world box; (sdf [T], surface witness [T,3]).

    Outside: Euclidean distance to the box, witness = closest boundary point.
    Inside: minus the distance to the boundary (exact for a box), witness on the
    nearest face.
    """
    half = box["half"]
    pl = quat_rotate_inverse(box["quat"], p - box["center"], w_last=True)
    clamped = torch.clamp(pl, -half, half)
    out_vec = pl - clamped
    out_dist = out_vec.norm(dim=-1)
    inside = out_dist == 0.0

    face_gap = half - pl.abs()  # >= 0 inside, per axis
    ax = face_gap.argmin(-1)
    pen = face_gap.gather(-1, ax.unsqueeze(-1)).squeeze(-1)
    onehot = F.one_hot(ax, 3).to(pl.dtype)
    sign = torch.where(pl >= 0, 1.0, -1.0)
    face_pt = pl * (1.0 - onehot) + onehot * sign * half

    surf_local = torch.where(inside.unsqueeze(-1), face_pt, clamped)
    sdf = torch.where(inside, -pen, out_dist)
    witness = quat_rotate(box["quat"], surf_local, w_last=True) + box["center"]
    return sdf, witness


def _seg_box(a, b, box, iters=48):
    """Closest point param t* on segment ab to a box via ternary search on the
    (convex) box SDF restricted to the segment.  Returns (sdf [T], seg_pt [T,3],
    box_witness [T,3])."""
    lo = torch.zeros(a.shape[0], device=a.device, dtype=a.dtype)
    hi = torch.ones_like(lo)
    ab = b - a
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        g1, _ = _pt_box(a + m1.unsqueeze(-1) * ab, box)
        g2, _ = _pt_box(a + m2.unsqueeze(-1) * ab, box)
        left = g1 <= g2
        hi = torch.where(left, m2, hi)
        lo = torch.where(left, lo, m1)
    t = (lo + hi) / 2.0
    seg_pt = a + t.unsqueeze(-1) * ab
    sdf, witness = _pt_box(seg_pt, box)
    return sdf, seg_pt, witness


_BOX_GRID = None


def _box_surface_grid():
    """54 points: 3x3 grid on each face of the unit box [-1,1]^3 (local units of half-extents)."""
    global _BOX_GRID
    if _BOX_GRID is None:
        lin = torch.tensor([-1.0, 0.0, 1.0])
        pts = []
        for axis in range(3):
            for side in (-1.0, 1.0):
                for u in lin:
                    for v in lin:
                        p = [0.0, 0.0, 0.0]
                        p[axis] = side
                        p[(axis + 1) % 3] = float(u)
                        p[(axis + 2) % 3] = float(v)
                        pts.append(p)
        _BOX_GRID = torch.unique(torch.tensor(pts), dim=0)  # 54 -> 26 unique
    return _BOX_GRID


def _surface_from_core(core_a, core_b, r_a, r_b):
    """Push core witness points (centers / axis points) out to the surfaces along
    the connecting line; returns (gap, surf_a, surf_b)."""
    d_vec = core_b - core_a
    d = d_vec.norm(dim=-1)
    n = d_vec / d.clamp_min(1e-9).unsqueeze(-1)
    return d - r_a - r_b, core_a + n * r_a, core_b - n * r_b


# --------------------------------------------------------------------------- #
# Pair dispatch.
# --------------------------------------------------------------------------- #
def _round_vs_box(g, box):
    """Sphere or capsule vs box."""
    if g["type"] == "sphere":
        sdf, w_box = _pt_box(g["center"], box)
        core = g["center"]
    else:
        sdf, core, w_box = _seg_box(g["a"], g["b"], box)
    n = w_box - core
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    n = torch.where((sdf >= 0).unsqueeze(-1), n, -n)  # inside the box: surface is the other way
    return sdf - g["radius"], core + n * g["radius"], w_box


def _box_vs_box(ga, gb):
    grid = _box_surface_grid().to(ga["center"].device, ga["center"].dtype)
    best = None
    for src, dst, flip in ((ga, gb, False), (gb, ga, True)):
        pts_local = grid * src["half"]
        for i in range(pts_local.shape[0]):
            p = quat_rotate(src["quat"], pts_local[i].expand_as(src["center"]), w_last=True) + src["center"]
            sdf, w = _pt_box(p, dst)
            wa, wb = (w, p) if flip else (p, w)
            if best is None:
                best = (sdf, wa, wb)
            else:
                closer = sdf < best[0]
                cl3 = closer.unsqueeze(-1)
                best = (
                    torch.where(closer, sdf, best[0]),
                    torch.where(cl3, wa, best[1]),
                    torch.where(cl3, wb, best[2]),
                )
    return best


def geom_pair_distance(ga, gb):
    """(gap [T], witness_a [T,3], witness_b [T,3]) between two world-frame typed geoms."""
    ta, tb = ga["type"], gb["type"]
    if ta == "sphere" and tb == "sphere":
        return _surface_from_core(ga["center"], gb["center"], ga["radius"], gb["radius"])
    if ta == "sphere" and tb == "capsule":
        core_b = _pt_seg(ga["center"], gb["a"], gb["b"])
        return _surface_from_core(ga["center"], core_b, ga["radius"], gb["radius"])
    if ta == "capsule" and tb == "sphere":
        gap, wb, wa = geom_pair_distance(gb, ga)
        return gap, wa, wb
    if ta == "capsule" and tb == "capsule":
        c1, c2 = _seg_seg(ga["a"], ga["b"], gb["a"], gb["b"])
        return _surface_from_core(c1, c2, ga["radius"], gb["radius"])
    if ta in ("sphere", "capsule") and tb == "box":
        return _round_vs_box(ga, gb)
    if ta == "box" and tb in ("sphere", "capsule"):
        gap, wb, wa = _round_vs_box(gb, ga)
        return gap, wa, wb
    if ta == "box" and tb == "box":
        return _box_vs_box(ga, gb)
    raise ValueError(f"{ta} vs {tb}")


def geom_ground_distance(g):
    """(gap [T], witness [T,3]) of a world-frame typed geom to the z=0 plane."""
    if g["type"] == "sphere":
        gap = g["center"][:, 2] - g["radius"]
        w = g["center"].clone()
        w[:, 2] -= g["radius"]
        return gap, w
    if g["type"] == "capsule":
        za, zb = g["a"][:, 2], g["b"][:, 2]
        gap = torch.minimum(za, zb) - g["radius"]
        # Contact-patch centroid: a horizontal capsule (kneeling shank) rests on
        # its whole length, so blend endpoints by how close each is to the low
        # point rather than snapping to one end.
        near_a = (za - torch.minimum(za, zb)) < 0.01
        near_b = (zb - torch.minimum(za, zb)) < 0.01
        wgt_a = near_a.float()
        wgt_b = near_b.float()
        w = (g["a"] * wgt_a.unsqueeze(-1) + g["b"] * wgt_b.unsqueeze(-1)) / (
            (wgt_a + wgt_b).unsqueeze(-1)
        )
        w[:, 2] = torch.minimum(za, zb) - g["radius"]
        return gap, w
    if g["type"] == "box":
        corners = _box_corners_world(g)  # [T,8,3]
        z = corners[..., 2]
        min_z = z.min(dim=1, keepdim=True)[0]
        # Contact-patch centroid: a flat-lying box face has 4 corners at equal
        # height; averaging the near-ground corners puts the witness mid-patch
        # (a single argmin corner picks an arbitrary heel/edge corner and
        # misplaces the support point by up to the box half-extent).
        near = (z - min_z) < 0.01  # [T,8]
        wgt = near.float().unsqueeze(-1)
        w = (corners * wgt).sum(dim=1) / wgt.sum(dim=1)
        gap = min_z.squeeze(1)
        w[:, 2] = gap
        return gap, w
    raise ValueError(g["type"])


def _box_corners_world(g):
    signs = torch.tensor(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        device=g["center"].device,
        dtype=g["center"].dtype,
    )  # [8,3]
    local = signs * g["half"]
    T = g["center"].shape[0]
    q = g["quat"].unsqueeze(1).expand(T, 8, 4).reshape(-1, 4)
    p = local.unsqueeze(0).expand(T, 8, 3).reshape(-1, 3)
    return (quat_rotate(q, p, w_last=True).reshape(T, 8, 3)) + g["center"].unsqueeze(1)


def geom_ground_patch(g):
    """Candidate ground-contact points of a world geom: [T,K,3] with z already
    lowered to the surface (radius subtracted).  The caller decides which points
    are near enough to the ground to count as part of the support patch."""
    if g["type"] == "sphere":
        w = g["center"].clone()
        w[:, 2] -= g["radius"]
        return w.unsqueeze(1)
    if g["type"] == "capsule":
        a, b = g["a"].clone(), g["b"].clone()
        a[:, 2] -= g["radius"]
        b[:, 2] -= g["radius"]
        return torch.stack([a, b], dim=1)
    if g["type"] == "box":
        return _box_corners_world(g)
    raise ValueError(g["type"])


def witness_velocity(witness, body_pos, body_vel, body_ang_vel):
    """Rigid-body velocity of a world point attached to the body: v + w x r; all [T,3]."""
    return body_vel + torch.linalg.cross(body_ang_vel, witness - body_pos, dim=-1)


# --------------------------------------------------------------------------- #
# Self-test: Monte-Carlo brute force against dense surface sampling.
# --------------------------------------------------------------------------- #
def _random_quat(n, gen):
    q = torch.randn(n, 4, generator=gen)
    return q / q.norm(dim=-1, keepdim=True)


def _core_cloud(g, seg_n=400, box_lin=21):
    """Dense point cloud [T, P, 3] + radius r such that the geom's surface is
    exactly {p : dist(p, cloud) = r} up to sampling pitch.  Round geoms sample
    the *core* (center / axis segment) so the radius carries no sampling error;
    boxes sample the surface with a box_lin x box_lin grid per face."""
    if g["type"] == "sphere":
        return g["center"].unsqueeze(1), g["radius"]
    if g["type"] == "capsule":
        t = torch.linspace(0, 1, seg_n)
        core = g["a"].unsqueeze(1) + t.view(1, -1, 1) * (g["b"] - g["a"]).unsqueeze(1)
        return core, g["radius"]
    if g["type"] == "box":
        lin = torch.linspace(-1, 1, box_lin)
        faces = []
        for axis in range(3):
            for side in (-1.0, 1.0):
                u, v = torch.meshgrid(lin, lin, indexing="ij")
                f = torch.zeros(u.numel(), 3)
                f[:, axis] = side
                f[:, (axis + 1) % 3] = u.reshape(-1)
                f[:, (axis + 2) % 3] = v.reshape(-1)
                faces.append(f)
        local = torch.cat(faces) * g["half"]  # [P,3]
        T, P = g["center"].shape[0], local.shape[0]
        q = g["quat"].unsqueeze(1).expand(T, P, 4).reshape(-1, 4)
        p = local.unsqueeze(0).expand(T, P, 3).reshape(-1, 3)
        return quat_rotate(q, p, w_last=True).reshape(T, P, 3) + g["center"].unsqueeze(1), 0.0
    raise ValueError(g["type"])


def _brute_gap(ga, gb):
    pa, ra = _core_cloud(ga)
    pb, rb = _core_cloud(gb)
    T = pa.shape[0]
    chunk = max(1, int(2e7 / (pa.shape[1] * pb.shape[1])))
    mins = []
    for i in range(0, T, chunk):
        d = torch.cdist(pa[i : i + chunk], pb[i : i + chunk])
        mins.append(d.flatten(1).min(dim=1)[0])
    return torch.cat(mins) - ra - rb


def _make_random_geom(gtype, n, gen):
    if gtype == "sphere":
        return {
            "type": "sphere",
            "center": torch.rand(n, 3, generator=gen) * 0.6 - 0.3,
            "radius": 0.04 + 0.07 * torch.rand(1, generator=gen).item(),
        }
    if gtype == "capsule":
        a = torch.rand(n, 3, generator=gen) * 0.6 - 0.3
        return {
            "type": "capsule",
            "a": a,
            "b": a + torch.randn(n, 3, generator=gen) * 0.15,
            "radius": 0.04 + 0.02 * torch.rand(1, generator=gen).item(),
        }
    return {
        "type": "box",
        "center": torch.rand(n, 3, generator=gen) * 0.6 - 0.3,
        "quat": _random_quat(n, gen),
        "half": torch.tensor([0.08, 0.045, 0.025]) * (0.5 + torch.rand(1, generator=gen).item()),
    }


def _self_test():
    gen = torch.Generator().manual_seed(0)
    n = 80
    # Brute-force sampling only *over*-estimates the true gap; round cores are
    # sampled densely (sub-mm) so only box faces contribute sampling pitch.
    tol = {"exact": 2e-3, "roundbox": 5e-3, "boxbox": 2.5e-2}
    print(f"{'pair':20s} {'max |err|':>10s} {'tolerance':>10s}")
    for ta in ("sphere", "capsule", "box"):
        for tb in ("sphere", "capsule", "box"):
            ga = _make_random_geom(ta, n, gen)
            gb = _make_random_geom(tb, n, gen)
            gap, wa, wb = geom_pair_distance(ga, gb)
            brute = _brute_gap(ga, gb)
            sep = brute > 5e-3  # brute sampling is meaningless under penetration
            err = (gap[sep] - brute[sep]).abs().max().item()
            # witness consistency: |wa-wb| must equal the gap where separated
            werr = ((wa - wb).norm(dim=-1)[sep] - gap[sep]).abs().max().item()
            n_box = (ta == "box") + (tb == "box")
            t = tol["boxbox"] if n_box == 2 else (tol["roundbox"] if n_box == 1 else tol["exact"])
            status = "ok" if err < t and werr < t else "FAIL"
            print(f"{ta+'-'+tb:20s} {err:10.5f} {t:10.5f}  witness {werr:.5f}  {status}")
            assert err < t and werr < t, f"{ta}-{tb} exceeded tolerance"
    # ground: sphere resting on plane
    g = {"type": "sphere", "center": torch.tensor([[0.3, -0.2, 0.05]]), "radius": 0.05}
    gap, w = geom_ground_distance(g)
    assert abs(gap.item()) < 1e-6 and abs(w[0, 2].item()) < 1e-6
    print("ground kernels ok")
    # degenerate segment 2 (point): closest pair must project onto segment 1
    c1, c2 = _seg_seg(
        torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.5, 2.0, 0.0]]), torch.tensor([[0.5, 2.0, 0.0]]),
    )
    d = (c1 - c2).norm().item()
    assert abs(d - 2.0) < 1e-6, f"degenerate seg-seg: {d} != 2.0"
    print("degenerate segment kernel ok")


if __name__ == "__main__":
    with torch.no_grad():
        _self_test()
