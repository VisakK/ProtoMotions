"""Graph v2 census + candidate-edge evidence for the workshop edge selection (read-only, CPU)."""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import mujoco

sys.path.insert(0, "data/scripts")
from goal_pose_separation import heading_quat_inv, quat_rotate  # noqa: E402

REL = "data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
STATICS = "data/reference_curation/statics_v2/holds_repaired_ftC_posefix.labels_v2.46024365fa.statics_v2.aab60189f4/holds.jsonl"
LIB = "expert_revist/expert56_v2_e15500/data/library_e15500.json"
MJCF_FLAT = "data/assets/smpl/smpl_yogi03596_v2_flat.xml"
OUT = Path(sys.argv[1])

g = json.load(open(f"{REL}/contact_graph.json"))
rel = yaml.safe_load(open(f"{REL}/release_holds.yaml"))
statics = {json.loads(l)["hold_id"]: json.loads(l) for l in open(STATICS)}
libj = json.load(open(LIB))
comp = {h["hold_id"]: h for c in libj["clips"] for h in c["holds"]}
lib = torch.load(f"{REL}/motions.pt", map_location="cpu", weights_only=False)
POS = lib["gts"].numpy()
ROT = lib["grs"].numpy()
STARTS = lib["length_starts"].numpy()
NF = lib["motion_num_frames"].numpy()
DT = lib["motion_dt"].numpy()

model = mujoco.MjModel.from_xml_path(MJCF_FLAT)
MASS = model.body_mass[1:].copy()
IPOS = model.body_ipos[1:].copy()
BODY = [model.body(i).name for i in range(1, model.nbody)]
B = {n: i for i, n in enumerate(BODY)}
GOAL = [B[n] for n in ["Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head"]]

clips = g["clips"]
x0 = {name: c for name, c in clips.items() if c["variant_s"] == 0.0}
hold_info = {}  # hold_id -> dict from the x0 clip
for name, c in x0.items():
    for s in c["segments"]:
        hold_info[s["hold_id"]] = dict(motion_id=c["motion_id"], clip=name, **s)
relh = {h["hold_id"]: h for c in rel["clips"] for h in c["holds"]}


def row(mid, t):
    f = int(round(t / float(DT[mid])))
    f = max(0, min(f, int(NF[mid]) - 1))
    return int(STARTS[mid]) + f


def goal_heading(mid, t):
    r = row(mid, t)
    p = POS[r]
    h = heading_quat_inv(ROT[r, 0])
    return np.stack([quat_rotate(h, q - p[0]) for q in p[GOAL]])


def goal_rel(mid, t):
    r = row(mid, t)
    p = POS[r]
    return (p - p[:1])[GOAL]


def best_yaw(a, b):
    num = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).sum(-1)
    den = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(-1)
    th = np.arctan2(num, den)[..., None]
    c, s = np.cos(th), np.sin(th)
    x = c * a[..., 0] - s * a[..., 1]
    y = s * a[..., 0] + c * a[..., 1]
    rot = np.stack([x, y, np.broadcast_to(a[..., 2], x.shape)], -1)
    return np.linalg.norm(rot - b, axis=-1).mean(-1)


def quat_apply_xyzw(q, v):
    u, w = q[..., :3], q[..., 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def com(mid, t):
    r = row(mid, t)
    p, q = POS[r], ROT[r]
    c = p + quat_apply_xyzw(q, IPOS)
    return (MASS[:, None] * c).sum(0) / MASS.sum(), p


def hd(a_hold, b_hold):
    A, Bh = hold_info[a_hold], hold_info[b_hold]
    dh = float(np.linalg.norm(goal_heading(A["motion_id"], A["t_hold"]) - goal_heading(Bh["motion_id"], Bh["t_hold"]), axis=-1).mean())
    dy = float(best_yaw(goal_rel(A["motion_id"], A["t_hold"]), goal_rel(Bh["motion_id"], Bh["t_hold"])))
    return dh, dy


# ---------------------------------------------------------------- census
nodes = g["nodes"]
edges = g["edges"]
x0_names = set(x0)
ecount = {}
eocc = {}
for e in edges:
    occ = [o for o in e["occurrences"] if o["motion"] in x0_names]
    ecount[(e["src"], e["dst"])] = len(occ)
    eocc[(e["src"], e["dst"])] = occ
hist = collections.Counter(ecount.values())
no_rev = sum(1 for (a, b) in ecount if (b, a) not in ecount)
# x0 dwell and segments per node
dwell = collections.defaultdict(float)
nseg = collections.Counter()
for name, c in x0.items():
    for s in c["segments"]:
        dwell[s["node"]] += s["duration_s"]
        nseg[s["node"]] += 1
top = [n for n, _ in sorted(dwell.items(), key=lambda kv: -kv[1]) if nseg[n] >= 2][:25]
adj = collections.defaultdict(set)
for (a, b) in ecount:
    adj[a].add(b)
direct = two = neither = 0
for a in top:
    for b in top:
        if a == b:
            continue
        if b in adj[a]:
            direct += 1
        elif any(b in adj[m] for m in adj[a]):
            two += 1
        else:
            neither += 1
# per-edge exemplar distances (x0 occurrences: the passage's own endpoints)
edge_d = []
edge_d_yaw = []
for (a, b), occ in eocc.items():
    if not occ:
        continue
    ds = [hd(o["hold_src"], o["hold_dst"]) for o in occ]
    edge_d.append(float(np.median([d[0] for d in ds])))
    edge_d_yaw.append(float(np.median([d[1] for d in ds])))
# excluding edges that touch standing (node 120's star)
STAND = [i for i, n in enumerate(nodes) if n["name"] == "standing"]
edge_d_ns = []
for (a, b), occ in eocc.items():
    if occ and a not in STAND and b not in STAND:
        ds = [hd(o["hold_src"], o["hold_dst"])[1] for o in occ]
        edge_d_ns.append(float(np.median(ds)))


def pct(x):
    x = np.asarray(x)
    return dict(n=int(len(x)), p10=round(float(np.percentile(x, 10)), 3), p50=round(float(np.percentile(x, 50)), 3),
                p90=round(float(np.percentile(x, 90)), 3), max=round(float(x.max()), 3))


census = dict(
    nodes=len(nodes), edges=len(edges), standing_nodes=STAND,
    edge_x0_count_hist={int(k): int(v) for k, v in sorted(hist.items())},
    edges_without_reverse=no_rev,
    top25=[dict(node=n, name=nodes[n]["name"], dwell_x0_s=round(dwell[n], 2), segs=nseg[n], key=nodes[n]["key"]) for n in top],
    top25_pairs=dict(direct=direct, two_hop_only=two, neither=neither),
    edge_dist_heading=pct(edge_d), edge_dist_bestyaw=pct(edge_d_yaw), edge_dist_bestyaw_nonstanding=pct(edge_d_ns),
    nodes_with_degree=dict(
        standing_out=len(adj[STAND[0]]),
        standing_in=sum(1 for (a, b) in ecount if b == STAND[0]),
    ),
)

# ---------------------------------------------------------------- hold table
FAM = ["Crane_Crow", "Side_Crane", "Firefly", "Koundinya", "Bhujapidasana", "Peacock_Pose_or_Mayurasana", "Handstand",
       "Feathered_Peacock", "Headstand", "Scorpion", "Plank_Pose_or_Kumbhakasana", "Chaturanga", "Dolphin_Plank",
       "Downward", "Garland"]
holds = []
for hid, h in sorted(hold_info.items(), key=lambda kv: (kv[1]["clip"], kv[1]["t_hold"])):
    if not any(f in h["clip"] for f in FAM):
        continue
    if h["name"] == "standing":
        continue
    st = statics.get(hid, {})
    gs = st.get("gated", {}) or {}
    wit = st.get("witness", {}) or {}
    cp = comp.get(hid, {})
    rh = relh.get(hid, {})
    ground = [p.replace(":G", "") for p in h["pairs"] if p.endswith(":G")]
    bb = [p for p in h["pairs"] if "+" in p]
    c, p = com(h["motion_id"], h["t_hold"])
    holds.append(dict(
        hold=hid.replace("220923_", "").replace("220926_", ""), hold_id=hid, node=h["node"], t_hold=h["t_hold"],
        dur=round(h["duration_s"], 2), orient=h["orientation_bin"], ground=ground, bb=bb,
        role=(rh.get("labels") or {}).get("pose_role"), gate=(rh.get("gate") or {}).get("decision"),
        flags=(rh.get("gate") or {}).get("flags"),
        statics=st.get("verdict"), s_star=gs.get("s_star"), util=gs.get("group_util"),
        top_joint=(gs.get("top_joints") or [[None]])[0][:2], statue=wit.get("passed"),
        tracked=cp.get("tracked_share"), support_min=cp.get("support_min"), err6=cp.get("err6_p50"),
        subst=cp.get("substitutions"), pelvis_z=round(float(p[0, 2]), 3), com_z=round(float(c[2]), 3),
    ))

# ---------------------------------------------------------------- candidates
def find(stem_part, at):
    hits = [hid for hid, h in hold_info.items() if stem_part in h["clip"] and hid.endswith(f"@{at}")]
    assert len(hits) == 1, (stem_part, at, hits)
    return hits[0]


H = dict(
    crow=find("Crane_Crow", 439), crow_prep=find("Crane_Crow", 424),
    hs=find("Handstand", 1193), hs2=find("Handstand", 2314), hs3leg=find("Handstand", 1632),
    chat=find("Chaturanga", 485), plank=find("Plank_Pose_or_Kumbhakasana", 541),
    tripod_k_a=find("Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-a", 920), tripod_k_b=find("Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-b", 578),
    tripod_sh_b=find("Supported_Headstand_pose_or_Salamba_Sirsasana_-b", 1093), tripod_sc_b=find("Side_Crane_Crow_Pose_or_Parsva_Bakasana_-b", 702),
    firefly=find("Firefly", 776), sidecrow_a=find("Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a", 871),
    sidecrow_b=find("Side_Crane_Crow_Pose_or_Parsva_Bakasana_-b", 1493), sidecrow_c=find("Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c", 606),
    koun_a=find("Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-a", 1199), koun_b=find("Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-b", 1060),
    pincha=find("Feathered_Peacock_Pose_or_Pincha_Mayurasana_-c", 1554), downdog=find("Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a", 448),
    bhuja=find("Bhujapidasana", 543), garland=find("Garland", 334), koun_a_feet=find("Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II-a", 1602),
    sh_a=find("Supported_Headstand_pose_or_Salamba_Sirsasana_-a", 1051), scorpion=find("Scorpion_pose_or_vrischikasana-a", 2684),
    dolphin_plank=find("Dolphin_Plank", 677),
)
CANDS = [
    ("crow", "hs", "user: crow -> handstand (press)"),
    ("crow", "chat", "user: crow -> chaturanga (jump back)"),
    ("hs", "crow", "user: handstand -> crow (lower)"),
    ("crow", "tripod_k_b", "crow -> tripod headstand"),
    ("tripod_k_b", "crow", "tripod headstand -> crow"),
    ("crow", "tripod_sh_b", "crow -> tripod headstand (Sirsasana -b)"),
    ("tripod_sh_b", "crow", "tripod headstand (Sirsasana -b) -> crow"),
    ("firefly", "crow", "firefly -> crow"),
    ("crow", "firefly", "crow -> firefly"),
    ("hs", "chat", "handstand -> chaturanga"),
    ("hs", "plank", "handstand -> plank"),
    ("crow", "plank", "crow -> plank (jump back, straight arms)"),
    ("downdog", "hs", "downdog -> handstand (float)"),
    ("crow", "sidecrow_c", "crow -> side crow -c"),
    ("crow", "sidecrow_a", "crow -> side crow -a"),
    ("sidecrow_a", "koun_a", "side crow -a -> Koundinya -a"),
    ("pincha", "crow", "pincha -> crow"),
    ("hs", "tripod_k_b", "handstand -> tripod headstand"),
    ("crow", "garland", "crow -> garland (bail-out)"),
    ("chat", "crow", "chaturanga -> crow"),
    ("bhuja", "crow", "bhujapidasana -> crow"),
    ("crow", "bhuja", "crow -> bhujapidasana"),
    ("koun_a", "chat", "Koundinya -a -> chaturanga"),
]

# corpus witness (best-yaw 6-body, 10 Hz, window 10 s)
seqs = {}
for name, c in x0.items():
    mid = c["motion_id"]
    n = int(NF[mid])
    step = max(1, int(round(1.0 / float(DT[mid]) / 10)))
    idx = np.arange(0, n, step)
    rows_ = STARTS[mid] + idx
    p = POS[rows_]
    seqs[name] = ((p - p[:, :1])[:, GOAL], idx * float(DT[mid]))


def witness(a_hold, b_hold, window_s=10.0, exclude_clip=None):
    A, Bh = hold_info[a_hold], hold_info[b_hold]
    sa = goal_rel(A["motion_id"], A["t_hold"])
    sb = goal_rel(Bh["motion_id"], Bh["t_hold"])
    best = None
    for name, (relp, times) in seqs.items():
        if exclude_clip and exclude_clip(name):
            continue
        ds = best_yaw(relp, sa[None])
        dt_ = best_yaw(relp, sb[None])
        W = int(round(window_s * 10))
        n = len(relp)
        for i in np.argsort(ds)[:60]:
            j0, j1 = i + 1, min(n, i + W + 1)
            if j0 >= j1:
                continue
            j = j0 + int(np.argmin(dt_[j0:j1]))
            score = max(ds[i], dt_[j])
            if best is None or score < best[0]:
                best = (float(score), name, float(times[i]), float(times[j]), float(ds[i]), float(dt_[j]))
    s, name, ti, tj, di, dj = best
    return dict(witness_m=round(s, 3), clip=name.replace("220923_", "").replace("220926_", ""), t_from=round(ti, 2),
                t_to=round(tj, 2), d_from=round(di, 3), d_to=round(dj, 3))


def bfs(src, dst, maxd=6):
    prev = {src: None}
    q = [src]
    while q:
        nq = []
        for u in q:
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    nq.append(v)
        q = nq
        if dst in prev:
            break
    if dst not in prev:
        return None
    path = [dst]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path = path[::-1]
    hops = []
    for a, b in zip(path, path[1:]):
        occ = eocc[(a, b)]
        hops.append(dict(src=a, dst=b, n=len(occ), via=sorted({o["motion"].replace("220923_", "")[:28] for o in occ})[:3]))
    return hops


cands = []
for a, b, label in CANDS:
    A, Bh = hold_info[H[a]], hold_info[H[b]]
    ga = {p for p in A["pairs"] if p.endswith(":G")}
    gb = {p for p in Bh["pairs"] if p.endswith(":G")}
    ba = {p for p in A["pairs"] if "+" in p}
    bb = {p for p in Bh["pairs"] if "+" in p}
    dh, dy = hd(H[a], H[b])
    ca, pa = com(A["motion_id"], A["t_hold"])
    cb, pb = com(Bh["motion_id"], Bh["t_hold"])
    # hands centroid of each pose (wrist+hand origins) -> COM horizontal offset from the hands, per pose
    def hands_off(c, p):
        hc = p[[B["L_Hand"], B["R_Hand"]]].mean(0)
        return round(float(np.linalg.norm(c[:2] - hc[:2])), 3)
    def feet_hand_span(p):
        hc = p[[B["L_Hand"], B["R_Hand"]]].mean(0)
        fc = p[[B["L_Toe"], B["R_Toe"]]].mean(0)
        return round(float(np.linalg.norm(fc[:2] - hc[:2])), 3)
    cands.append(dict(
        label=label, src=H[a].split("@")[0].replace("220923_", "").replace("220926_", "")[:40] + "@" + H[a].split("@")[1],
        dst=H[b].split("@")[0].replace("220923_", "").replace("220926_", "")[:40] + "@" + H[b].split("@")[1],
        src_node=A["node"], dst_node=Bh["node"], orient=f"{A['orientation_bin']}->{Bh['orientation_bin']}",
        ground_break=sorted(p.replace(":G", "") for p in ga - gb), ground_make=sorted(p.replace(":G", "") for p in gb - ga),
        bb_break=sorted(ba - bb), bb_make=sorted(bb - ba),
        d6_heading=round(dh, 3), d6_bestyaw=round(dy, 3),
        direct_edge=(A["node"], Bh["node"]) in ecount, path=bfs(A["node"], Bh["node"]),
        witness=witness(H[a], H[b]),
        witness_other_clip=witness(H[a], H[b], exclude_clip=lambda n, s=(A["clip"], Bh["clip"]): n in s),
        com_hands_off_src=hands_off(ca, pa), com_hands_off_dst=hands_off(cb, pb),
        feet_hands_span_src=feet_hand_span(pa), feet_hands_span_dst=feet_hand_span(pb),
        pelvis_dz=round(float(pb[0, 2] - pa[0, 2]), 3), com_dz=round(float(cb[2] - ca[2]), 3),
    ))

OUT.write_text(json.dumps(dict(census=census, holds=holds, cands=cands, H=H), indent=1, default=str))
print("written", OUT)
