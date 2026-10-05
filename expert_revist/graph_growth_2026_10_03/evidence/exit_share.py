"""Offline: share of training resets that land in the get-up (exit) windows, under the run's manager config."""
import csv, yaml, torch, numpy as np
R = "data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
g = torch.load(f"{R}/contact_graph.pt", map_location="cpu", weights_only=False)
ext = yaml.safe_load(open(f"{R}/holds_extended.yaml"))
clips = {c["stem"]: c for c in ext["clips"]}
names = list(g["motion_names"])
fps = int(g["fps"]); dt = 1.0 / 30
L = (g["motion_num_frames"].double() / fps).numpy()
M = len(names)
SEG_P, INIT_P, PRE = 0.6, 0.2, 0.5
# sampling probs at e15500 (the curriculum's) and uniform
probs = np.zeros(M)
with open("results/smpl_yogi_v2_expert56_a2dda5d2ac/curriculum/eval_epoch_015500.csv") as f:
    for row in csv.DictReader(f):
        probs[int(row["motion_id"])] = float(row["sampling_prob"])
assert abs(probs.sum() - 1) < 1e-3, probs.sum()
uni = np.full(M, 1.0 / M)
STANDING = "standing"   # pose_role
targets = ["Plow_Pose_or_Halasana_-b", "Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_-a",
           "Upward_Plank_Pose_or_Purvottanasana_-a", "Handstand_pose_or_Adho_Mukha_Vrksasana_-a"]
def p_in_window(m, w0, w1):
    """P(start time in [w0, w1] | motion m) under the manager: 0.6 anchored, else (0.2 t=0, 0.8 uniform)."""
    n = int(g["seg_count"][m]); Lm = L[m]
    upper = Lm - dt
    # anchored: seg k uniform, start = clamp(seg_start - U(0, PRE), 0, upper)
    pa = 0.0
    if n > 0:
        for k in range(n):
            s = float(g["seg_start"][m, k])
            lo, hi = s - PRE, s       # uniform support before clamping
            # overlap of [lo, hi] with [w0, w1] (ignore clamp edge effects except at 0)
            ov = max(0.0, min(hi, w1) - max(lo, w0))
            frac = ov / PRE
            if lo < 0 and w0 <= 0 <= w1:     # mass clamped to 0
                frac += (0 - lo) / PRE if hi > 0 else 1.0
            pa += frac / n
    pu = max(0.0, min(w1, upper) - max(w0, 0.0)) / upper
    p0 = 1.0 if w0 <= 0 <= w1 else 0.0
    anch = SEG_P if n > 0 else 0.0
    return anch * pa + (1 - anch) * (INIT_P * p0 + (1 - INIT_P) * pu)
rows = []
tot_u = tot_c = 0.0
for stem in names:
    base = clips[stem]["source_stem"]
    if not any(t in base for t in targets):
        continue
    m = names.index(stem)
    holds = sorted(clips[stem]["holds"], key=lambda h: float(h["t_hold"]))
    roles = [h["labels"]["pose_role"] for h in holds]
    # exit window: from the end of the last non-standing hold to the start of the final standing hold
    last_ns = max(i for i, r in enumerate(roles) if r != STANDING)
    nxt = [i for i in range(last_ns + 1, len(holds)) if roles[i] == STANDING]
    w0 = float(holds[last_ns]["t_end"])
    w1 = float(holds[nxt[0]]["t_start"]) if nxt else L[m]
    p = p_in_window(m, w0, w1)
    rows.append((stem, len(holds), roles, round(w0, 2), round(w1, 2), round(w1 - w0, 2), round(L[m], 1), p,
                 p * uni[m], p * probs[m], probs[m]))
    tot_u += p * uni[m]; tot_c += p * probs[m]
for r in rows:
    print(f"{r[0][7:60]:55s} holds {r[1]:2d} last_ns->stand window {r[3]:6.2f}-{r[4]:6.2f} ({r[5]:5.2f} s of {r[6]:5.1f} s)"
          f"  P(start in exit|clip) {r[7]:.4f}  share(uniform) {r[8]*100:.3f}%  share(e15500 curriculum) {r[9]*100:.3f}%  p_clip {r[10]:.4f}")
print(f"TOTAL share of all resets landing in these exit windows: uniform clip sampling {tot_u*100:.3f}%, e15500 curriculum {tot_c*100:.3f}%")
# compare: exit window share of these motions' own time
for r in rows[:3]:
    pass
# What an anchored start in the FINAL standing segment gives: starts <= 0.5 s before the standing hold
print("roles of one clip:", rows[0][2])

# decomposition: anchored vs uniform contributions, and the first half of the exit (the get-up proper)
print("\n--- decomposition (x0 variants) ---")
tot_time = float(L.sum())
exit_time = 0.0
for stem in names:
    base = clips[stem]["source_stem"]
    if not any(t in base for t in targets):
        continue
    m = names.index(stem)
    holds = sorted(clips[stem]["holds"], key=lambda h: float(h["t_hold"]))
    roles = [h["labels"]["pose_role"] for h in holds]
    last_ns = max(i for i, r in enumerate(roles) if r != STANDING)
    nxt = [i for i in range(last_ns + 1, len(holds)) if roles[i] == STANDING]
    w0 = float(holds[last_ns]["t_end"]); w1 = float(holds[nxt[0]]["t_start"]) if nxt else L[m]
    exit_time += w1 - w0
    if stem != base:
        continue
    n = int(g["seg_count"][m]); upper = L[m] - dt
    pa = 0.0
    for k in range(n):
        s = float(g["seg_start"][m, k]); lo, hi = s - PRE, s
        pa += max(0.0, min(hi, w1) - max(lo, w0)) / PRE / n
    pu = (w1 - w0) / upper
    first_half = (w0, (w0 + w1) / 2)
    pa_h = 0.0
    for k in range(n):
        s = float(g["seg_start"][m, k]); lo, hi = s - PRE, s
        pa_h += max(0.0, min(hi, first_half[1]) - max(lo, first_half[0])) / PRE / n
    pu_h = (first_half[1] - first_half[0]) / upper
    print(f"{base[7:50]:45s} anchored {SEG_P*pa:.4f}  uniform {(1-SEG_P)*(1-INIT_P)*pu:.4f}  | first half of exit: anchored {SEG_P*pa_h:.4f} uniform {(1-SEG_P)*(1-INIT_P)*pu_h:.4f}")
print(f"exit windows = {exit_time:.1f} s of {tot_time:.0f} s corpus time ({exit_time/tot_time*100:.2f} %)")
