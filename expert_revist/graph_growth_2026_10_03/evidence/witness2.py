"""Support-consistent corpus witness: frames i < j (<= W s apart) whose ground set (and capsule braces)
equal the source / target hold's, scored by max of the 6-body best-yaw distances."""
import json
import sys

import mujoco
import numpy as np
import torch

REL = "data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
d = json.load(open(sys.argv[1]))
H = d["H"]
g = json.load(open(f"{REL}/contact_graph.json"))
lib = torch.load(f"{REL}/motions.pt", map_location="cpu", weights_only=False)
POS = lib["gts"].numpy().astype(np.float64)
ROT = lib["grs"].numpy().astype(np.float64)
CON = lib["contacts"].numpy() > 0.5
STARTS = lib["length_starts"].numpy()
NF = lib["motion_num_frames"].numpy()
DT = lib["motion_dt"].numpy()
m = mujoco.MjModel.from_xml_path("data/assets/smpl/smpl_yogi03596_v2_flat.xml")
NAMES = [m.body(i).name for i in range(1, m.nbody)]
B = {n: i for i, n in enumerate(NAMES)}
GOAL = [B[n] for n in ["Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head"]]
ZB = {"L_FOOT": ["L_Ankle", "L_Toe"], "R_FOOT": ["R_Ankle", "R_Toe"], "L_SHANK": ["L_Knee"], "R_SHANK": ["R_Knee"],
      "L_THIGH": ["L_Hip"], "R_THIGH": ["R_Hip"], "PELVIS": ["Pelvis"], "TRUNK": ["Torso", "Spine", "Chest", "L_Thorax", "R_Thorax"],
      "HEAD": ["Neck", "Head"], "L_UPPER_ARM": ["L_Shoulder"], "R_UPPER_ARM": ["R_Shoulder"], "L_FOREARM": ["L_Elbow"],
      "R_FOREARM": ["R_Elbow"], "L_HAND": ["L_Wrist", "L_Hand"], "R_HAND": ["R_Wrist", "R_Hand"]}
ZONES = list(ZB)
ZI = np.array([[B[b] in [B[x] for x in ZB[z]] for b in NAMES] for z in ZONES])  # [15, 24]
CAP = {"L_SHANK": "L_Knee", "R_SHANK": "R_Knee", "L_THIGH": "L_Hip", "R_THIGH": "R_Hip", "L_UPPER_ARM": "L_Shoulder",
       "R_UPPER_ARM": "R_Shoulder", "L_FOREARM": "L_Elbow", "R_FOREARM": "R_Elbow"}
GID = {m.body(m.geom_bodyid[i]).name: i for i in range(m.ngeom) if m.geom_bodyid[i] > 0}


def qrot(q, v):
    u, w = q[..., :3], q[..., 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def capsule(bname, p, q):
    gi = GID[bname]
    bi = B[bname]
    gq = m.geom_quat[gi]
    gq = np.array([gq[1], gq[2], gq[3], gq[0]])
    r, hl = m.geom_size[gi][0], m.geom_size[gi][1]
    c = p[:, bi] + qrot(q[:, bi], np.broadcast_to(m.geom_pos[gi], p[:, bi].shape))
    ax = qrot(q[:, bi], np.broadcast_to(qrot(gq, np.array([0, 0, 1.0])), p[:, bi].shape))
    return c - hl * ax, c + hl * ax, r


def capgap(z1, z2, p, q):
    a0, a1, ra = capsule(CAP[z1], p, q)
    b0, b1, rb = capsule(CAP[z2], p, q)
    best = np.full(len(p), 1e9)
    for s in np.linspace(0, 1, 21):
        pa = a0 + (a1 - a0) * s
        dd = b1 - b0
        t = np.clip(((pa - b0) * dd).sum(-1) / (dd * dd).sum(-1), 0, 1)[:, None]
        best = np.minimum(best, np.linalg.norm(pa - (b0 + t * dd), axis=-1))
    return best - ra - rb


def best_yaw(a, b):
    num = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).sum(-1)
    den = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(-1)
    th = np.arctan2(num, den)[..., None]
    c, s = np.cos(th), np.sin(th)
    x = c * a[..., 0] - s * a[..., 1]
    y = s * a[..., 0] + c * a[..., 1]
    rot = np.stack([x, y, np.broadcast_to(a[..., 2], x.shape)], -1)
    return np.linalg.norm(rot - b, axis=-1).mean(-1)


hold = {}
x0 = {n: c for n, c in g["clips"].items() if c["variant_s"] == 0.0}
for n, c in x0.items():
    for s in c["segments"]:
        hold[s["hold_id"]] = dict(mid=c["motion_id"], clip=n, **s)

seqs = {}
for n, c in x0.items():
    mid = c["motion_id"]
    idx = np.arange(0, int(NF[mid]), 6)  # 10 Hz
    rows = STARTS[mid] + idx
    p, q = POS[rows], ROT[rows]
    zc = (CON[rows][:, None, :] & ZI[None]).any(-1)  # [T, 15]
    seqs[n] = dict(rel=(p - p[:, :1])[:, GOAL], t=idx * float(DT[mid]), zc=zc, p=p, q=q)


def spec(hid):
    h = hold[hid]
    ground = {x[:-2] for x in h["pairs"] if x.endswith(":G")}
    bb = [tuple(x.split("+")) for x in h["pairs"] if "+" in x and all(z in CAP for z in x.split("+"))]
    r = int(STARTS[h["mid"]] + round(h["t_hold"] / float(DT[h["mid"]])))
    pose = (POS[r] - POS[r, :1])[GOAL]
    return ground, bb, pose


def ok(seq, ground, bb, tol=0.03):
    gmask = np.array([z in ground for z in ZONES])
    good = (seq["zc"] == gmask[None]).all(-1)
    for z1, z2 in bb:
        good &= capgap(z1, z2, seq["p"], seq["q"]) <= tol
    return good


def witness(a, b, W=10.0, exclude=()):
    ga, bba, pa = spec(a)
    gb, bbb, pb = spec(b)
    best = None
    for n, s in seqs.items():
        if n in exclude:
            continue
        oka, okb = ok(s, ga, bba), ok(s, gb, bbb)
        da = np.where(oka, best_yaw(s["rel"], pa[None]), np.inf)
        db = np.where(okb, best_yaw(s["rel"], pb[None]), np.inf)
        w = int(W * 10)
        for i in np.argsort(da)[:40]:
            if not np.isfinite(da[i]):
                break
            j0, j1 = i + 1, min(len(db), i + w + 1)
            if j0 >= j1:
                continue
            j = j0 + int(np.argmin(db[j0:j1]))
            if not np.isfinite(db[j]):
                continue
            sc = max(da[i], db[j])
            if best is None or sc < best[0]:
                best = (round(float(sc), 3), n.replace("220923_", "")[:45], round(float(s["t"][i]), 1), round(float(s["t"][j]), 1),
                        round(float(da[i]), 3), round(float(db[j]), 3))
    return best


PAIRS = [("crow", "hs"), ("crow", "chat"), ("hs", "crow"), ("crow", "tripod_k_b"), ("tripod_k_b", "crow"),
         ("tripod_sh_b", "crow"), ("crow", "tripod_sh_b"), ("firefly", "crow"), ("crow", "firefly"), ("hs", "chat"),
         ("hs", "plank"), ("crow", "plank"), ("downdog", "hs"), ("pincha", "crow"), ("hs", "tripod_k_b"),
         ("chat", "crow"), ("bhuja", "crow"), ("crow", "bhuja"), ("koun_a", "chat"), ("sidecrow_a", "koun_a"),
         ("crow", "garland")]
out = {}
for a, b in PAIRS:
    w = witness(H[a], H[b])
    wo = witness(H[a], H[b], exclude=(hold[H[a]]["clip"], hold[H[b]]["clip"]))
    out[f"{a}->{b}"] = dict(any=w, other_clips=wo)
    print(f"{a:>12s} -> {b:<12s} any: {w}   other clips: {wo}")
json.dump(out, open(sys.argv[2], "w"), indent=1)
