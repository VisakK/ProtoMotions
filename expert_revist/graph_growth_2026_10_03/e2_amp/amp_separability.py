"""Offline AMP separability check (PLAN.MD card E2, item 5).

Can a discriminator on ``amp_features_v1`` tell the epoch-15,500 expert's motion from the human reference?

* **Agent windows**: the evaluator's own rollouts of every motion (``predicted_motion_lib_epoch_15500.pt``: t = 0,
  deterministic actions, never reset; frame i = clip time (i + 1) dt at the control rate). Frame i's window is frames
  ``i - k`` for ``k`` in ``AMP_FEATURES_V1_STEPS`` -- exactly what the env's state history buffer holds.
* **Reference windows**: the release library's x0 clips (never Scorpion -b), read at ``t - k dt`` by the very function
  the training discriminator's demonstrations use (``compute_amp_features_v1_from_motion_lib``), at the same clip
  times as the agent frames.

Two classifiers, both MLPs on standardised, clamped (+-5) inputs like the discriminator's:
* **in-sample** (random 80/20 split of windows): the separation a co-trained discriminator can reach;
* **leave-clips-out** (5 folds over the 56 base clips): separation that generalises to unseen poses, i.e. a systematic
  difference -- a feature mismatch or a real artefact (jitter, stillness, collapse).

Per family (the manifest's group), per feature block (posture / root velocities / local rotations / key bodies), for
the newest frame alone vs the whole window, and on tracked windows only (every body within 0.5 m of the reference
across the window: a fall is trivially separable and says nothing about style).

    PYTHONPATH=. python expert_revist/graph_growth_2026_10_03/e2_amp/amp_separability.py \
        --out expert_revist/graph_growth_2026_10_03/e2_amp/separability_e15500.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protomotions.components.motion_lib import MotionLib, MotionLibConfig  # noqa: E402
from protomotions.envs.obs.amp_features import (  # noqa: E402
    AMP_FEATURES_V1_STEPS,
    amp_features_v1_params,
    amp_frame_features_v1,
    compute_amp_features_v1_from_motion_lib,
)

RELEASE = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
LIBRARY = "results/smpl_yogi_v2_expert56_a2dda5d2ac/results/predicted_motion_lib_epoch_15500.pt"
SMPL_BODIES = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
               "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
               "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
SMPL_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]
DEMO_EXCLUDE = ("Scorpion_pose_or_vrischikasana-b",)
VARIANT = re.compile(r"_x\d+s$")
TRACK_FAIL_M = 0.5
F = 163
# per-frame column blocks of amp_features_v1 (10 + 23 x 6 + 5 x 3)
BLOCKS = {"posture (height, gravity)": list(range(0, 4)), "root velocities": list(range(4, 10)),
          "local rotations": list(range(10, 148)), "key bodies": list(range(148, 163))}


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties averaged)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = label == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_mlp(x: torch.Tensor, y: torch.Tensor, epochs: int, seed: int, hidden=(512, 256)) -> torch.nn.Module:
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0).clamp(min=1e-4)
    layers, d = [], x.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
        d = h
    layers += [torch.nn.Linear(d, 1)]
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_w = (y == 0).sum() / (y == 1).sum().clamp(min=1)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, 4096):
            b = perm[s:s + 4096]
            xb = ((x[b] - mean) / std).clamp(-5, 5)
            loss = lossf(net(xb).squeeze(-1), y[b].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    net.norm = (mean, std)
    return net


@torch.no_grad()
def predict(net, x: torch.Tensor) -> np.ndarray:
    mean, std = net.norm
    out = [net(((x[s:s + 65536] - mean) / std).clamp(-5, 5)).squeeze(-1) for s in range(0, x.shape[0], 65536)]
    return torch.cat(out).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", default=LIBRARY)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--out", default=str(Path(__file__).with_name("separability_e15500.json")))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()

    rec = json.loads((REPO / "data/reference_curation/releases" / f"{args.release}.json").read_text())
    rdir = REPO / rec["dir"]
    ext = {c["stem"]: c for c in yaml.safe_load(open(rdir / "holds_extended.yaml"))["clips"]}
    ref_lib = MotionLib(MotionLibConfig(motion_file=str(rdir / "motions.pt")), device="cpu")
    ref_stems = [Path(str(f)).name.rsplit(".motion", 1)[0] for f in ref_lib.motion_files]
    params = amp_features_v1_params(SMPL_BODIES, SMPL_PARENTS)
    steps = list(AMP_FEATURES_V1_STEPS)
    kmax = max(steps)

    lib = torch.load(REPO / args.library, map_location="cpu", weights_only=False)
    dt = float(lib["motion_dt"][0])
    rows_x, rows_y, rows_clip, rows_family, rows_tracked, rows_variant, rows_stem = [], [], [], [], [], [], []

    for i, f in enumerate(lib["motion_files"]):
        stem = Path(str(f)).stem
        base = VARIANT.sub("", stem)
        fam = ext[stem]["group"]
        a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
        if n <= kmax:
            continue
        pos, rot = lib["gts"][a:a + n].float(), lib["grs"][a:a + n].float()
        vel, ang = lib["gvs"][a:a + n].float(), lib["gavs"][a:a + n].float()
        frames = amp_frame_features_v1(pos, rot, vel, ang, torch.zeros(n), **params)          # [n, F]
        idx = torch.arange(kmax, n)
        win = torch.stack([frames[idx - k] for k in steps], dim=1).reshape(len(idx), -1)       # [n - kmax, 8F]
        # tracking: the evaluator's frame-0 XY alignment, every body within 0.5 m on every frame of the window
        m = ref_stems.index(stem)
        times = (torch.arange(n, dtype=torch.float32) + 1.0) * dt
        ref = ref_lib.get_motion_state(torch.full((n,), m, dtype=torch.long), times).rigid_body_pos
        sim = pos.clone()
        sim[..., :2] -= sim[0, 0, :2] - ref[0, 0, :2]
        ok = ((sim - ref).norm(dim=-1).max(-1).values < TRACK_FAIL_M)
        ok_win = torch.stack([ok[idx - k] for k in steps], dim=1).all(dim=1)
        rows_x.append(win)
        rows_y.append(torch.zeros(len(idx), dtype=torch.long))
        rows_clip += [base] * len(idx)
        rows_family += [fam] * len(idx)
        rows_tracked.append(ok_win)
        rows_variant += ["x0" if base == stem else stem[len(base) + 1:]] * len(idx)
        rows_stem += [stem] * len(idx)

        # reference windows of the x0 clip at the same clip times (positives), never Scorpion -b
        if base == stem and not any(x in stem for x in DEMO_EXCLUDE):
            t_ref = times[idx]
            t_ref = t_ref[t_ref <= float(ref_lib.motion_lengths[m])]
            demo = compute_amp_features_v1_from_motion_lib(ref_lib, torch.full((len(t_ref),), m, dtype=torch.long),
                                                           t_ref, dt, steps, **params)
            rows_x.append(demo)
            rows_y.append(torch.ones(len(t_ref), dtype=torch.long))
            rows_clip += [base] * len(t_ref)
            rows_family += [fam] * len(t_ref)
            rows_tracked.append(torch.ones(len(t_ref), dtype=torch.bool))
            rows_variant += ["reference"] * len(t_ref)
            rows_stem += [stem] * len(t_ref)

    x = torch.cat(rows_x)
    y = torch.cat(rows_y)
    tracked = torch.cat(rows_tracked).numpy()
    clip = np.array(rows_clip)
    family = np.array(rows_family)
    variant = np.array(rows_variant)
    stems = np.array(rows_stem)
    ynp = y.numpy()
    print(f"windows: {int((ynp == 0).sum())} agent ({int(((ynp == 0) & tracked).sum())} tracked), "
          f"{int((ynp == 1).sum())} reference; {len(set(clip))} base clips; features {x.shape[1]} "
          f"({time.time() - t0:.0f} s)")

    rng = np.random.default_rng(args.seed)
    results: dict = {"release": args.release, "library": args.library, "dt": dt, "steps": steps,
                     "n_agent": int((ynp == 0).sum()), "n_agent_tracked": int(((ynp == 0) & tracked).sum()),
                     "n_reference": int((ynp == 1).sum())}

    def report(score, mask, label):
        out = {"all": auroc(score[mask], ynp[mask]),
               "tracked": auroc(score[mask & ((ynp == 1) | tracked)], ynp[mask & ((ynp == 1) | tracked)])}
        for g in sorted(set(family)):
            sel = mask & (family == g) & ((ynp == 1) | tracked)
            out[f"tracked/{g}"] = auroc(score[sel], ynp[sel])
        for v in ("x0", "x3s", "x7s"):
            sel = mask & (((variant == v) & tracked) | (ynp == 1))
            out[f"tracked/agent_{v}"] = auroc(score[sel], ynp[sel])
        print(f"{label:<44} " + "  ".join(f"{k}={v:.3f}" for k, v in out.items()))
        return out

    # 1. in-sample: random 80/20 split of windows (what a co-trained discriminator can reach)
    test = rng.random(len(ynp)) < 0.2
    net = train_mlp(x[~test], y[~test], args.epochs, args.seed)
    s_in = predict(net, x)
    results["in_sample"] = report(s_in, test, "in-sample (random 20 % held out)")
    agent_test = test & (ynp == 0) & tracked
    p_agent = 1.0 / (1.0 + np.exp(-s_in[agent_test]))
    results["in_sample_agent_reward_proxy"] = {
        "mean_D_agent_tracked": float(p_agent.mean()),
        "mean_-log(1-D)_agent_tracked": float(np.mean(-np.log(np.clip(1 - p_agent, 1e-4, 1)))),
    }
    # worst clips: the agent windows the in-sample classifier finds least human (lowest D), per stem
    per_stem = {}
    for s_ in sorted(set(stems[agent_test])):
        sel = agent_test & (stems == s_)
        per_stem[s_] = float((1.0 / (1.0 + np.exp(-s_in[sel]))).mean())
    results["in_sample_mean_D_by_agent_stem"] = dict(sorted(per_stem.items(), key=lambda kv: kv[1]))

    # 2. leave-clips-out: folds over base clips
    bases = np.array(sorted(set(clip)))
    rng.shuffle(bases)
    fold_of = {b: k % args.folds for k, b in enumerate(bases)}
    folds = np.array([fold_of[c] for c in clip])
    s_out = np.zeros(len(ynp))
    for k in range(args.folds):
        tr = folds != k
        net_k = train_mlp(x[tr], y[tr], args.epochs, args.seed + 1 + k)
        s_out[folds == k] = predict(net_k, x[folds == k])
    results["leave_clips_out"] = report(s_out, np.ones(len(ynp), dtype=bool), f"leave-clips-out ({args.folds} folds)")

    # 3. diagnostics (in-sample split): newest frame only, and each feature block over the window
    newest = list(range(F))                                      # frame at step 1
    net_new = train_mlp(x[~test][:, newest], y[~test], args.epochs, args.seed)
    results["block/newest_frame_only"] = report(predict(net_new, x[:, newest]), test, "newest frame only")
    for name, cols in BLOCKS.items():
        idx_cols = [k * F + c for k in range(len(steps)) for c in cols]
        net_b = train_mlp(x[~test][:, idx_cols], y[~test], args.epochs, args.seed, hidden=(256,))
        results[f"block/{name}"] = report(predict(net_b, x[:, idx_cols]), test, f"block: {name}")

    # 4. which single features differ most (standardised mean difference, tracked agent vs reference)
    a_x, r_x = x[(ynp == 0) & tracked], x[ynp == 1]
    sd = torch.sqrt(0.5 * (a_x.var(0) + r_x.var(0))).clamp(min=1e-6)
    d = ((a_x.mean(0) - r_x.mean(0)) / sd).abs()
    names = []
    blocks_per_frame = (["root_h"] + [f"grav_{c}" for c in "xyz"] + [f"vel_{c}" for c in "xyz"]
                        + [f"angvel_{c}" for c in "xyz"]
                        + [f"{SMPL_BODIES[j]}_rot{c}" for j in params["child_ids"] for c in range(6)]
                        + [f"{SMPL_BODIES[j]}_pos_{c}" for j in params["key_body_ids"] for c in "xyz"])
    for k in steps:
        names += [f"t-{k}:{nm}" for nm in blocks_per_frame]
    top = torch.topk(d, 15)
    results["top_standardised_mean_differences"] = {names[i]: round(float(v), 3)
                                                    for v, i in zip(top.values, top.indices)}
    # speed/jerk proxy: per-window std of the root velocity across the 4 dense frames (control-rate jitter)
    vcols = [k * F + c for k in range(4) for c in range(4, 10)]
    jit_a = x[(ynp == 0) & tracked][:, vcols].view(-1, 4, 6).std(1).mean(-1)
    jit_r = x[ynp == 1][:, vcols].view(-1, 4, 6).std(1).mean(-1)
    results["root_velocity_jitter_p50"] = {"agent_tracked": float(jit_a.median()), "reference": float(jit_r.median())}
    results["seconds"] = round(time.time() - t0, 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
    print("top |d|:", list(results["top_standardised_mean_differences"].items())[:8])
    print("root-velocity jitter p50 (agent tracked / reference):", results["root_velocity_jitter_p50"])
    print(f"-> {args.out} ({results['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
