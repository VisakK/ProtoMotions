"""Where G1's extra drag happens: per clip, zone and time, on the evaluator's own rollouts.

Drag (``physics_terms.clip_drag``, the evaluator's metric) is friction work while a foot or hand carries more than
50 N and its slowest bottom corner slides faster than 0.10 m/s. This re-runs the same kernel frame by frame on a
rollout library that carries the simulator's ground forces (E1 added them; e15500's own library predates that), so
**G1's epoch-1 library stands in for e15500** (one PPO update from it; its drag is 38.8 J against 38.4 J). The
per-clip totals are checked against the evaluator's CSV. Writes ``data/drag_locate.json``.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ../env_isaaclab/bin/python \\
        expert_revist/graph_growth_2026_10_03/g1_amp_eval/drag_locate.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protomotions.envs.control.physics_terms import PhysicsTables, corner_slip_speed  # noqa: E402

HERE = Path(__file__).resolve().parent
RELEASE = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
RUN = REPO / "results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac"
ZONES = (("L_FOOT", ("L_Ankle", "L_Toe")), ("R_FOOT", ("R_Ankle", "R_Toe")),
         ("L_HAND", ("L_Wrist", "L_Hand")), ("R_HAND", ("R_Wrist", "R_Hand")))
LOAD_N, SLIP, MU = 50.0, 0.10, 0.75
VARIANT = re.compile(r"_x\d+s$")


def per_frame_drag(lib, i, tables, body_idx):
    a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
    pos, rot = lib["gts"][a:a + n].float(), lib["grs"][a:a + n].float()
    vel, ang = lib["gvs"][a:a + n].float(), lib["gavs"][a:a + n].float()
    fz = lib["sim_rigid_body_ground_forces"][a:a + n].float()[..., 2].clamp_min(0.0)
    dt = float(lib["motion_dt"][i])
    work = torch.zeros(n, len(ZONES))
    for z, bodies in enumerate(body_idx):
        load = fz[:, bodies].sum(-1)
        speed = corner_slip_speed(pos, rot, vel, ang, tables, bodies)
        mask = (load > LOAD_N) & (speed > SLIP)
        work[:, z] = MU * load * speed * mask.float() * dt
    return work, dt


def segments(work_t: torch.Tensor, dt: float, min_gap: int = 6):
    """Merge drag frames (any zone) into time segments: (t0, t1, J)."""
    on = torch.nonzero(work_t > 0).squeeze(-1).tolist()
    out, cur = [], None
    for f in on:
        if cur and f - cur[1] <= min_gap:
            cur[1] = f
        else:
            if cur:
                out.append(cur)
            cur = [f, f]
    if cur:
        out.append(cur)
    return [((s + 1) * dt, (e + 1) * dt, float(work_t[s:e + 1].sum())) for s, e in out]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--epochs", nargs="*", type=int, default=[1, 4500])
    ap.add_argument("--standalone", default=None,
                    help="an eval_checkpoint.py --out-dir: its results/predicted_motion_lib_epoch_0.pt and its CSV are "
                         "added as the label 'standalone'")
    ap.add_argument("--clips", nargs="*", default=["Side_Plank_Pose_or_Vasisthasana_-a", "Dolphin_Plank",
                                                   "Plank_Pose_or_Kumbhakasana_-a", "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a",
                                                   "Plow_Pose_or_Halasana_-b", "Half_Moon_Pose_or_Ardha_Chandrasana_-b",
                                                   "Side_Plank_Pose_or_Vasisthasana_-b", "Supported_Shoulderstand",
                                                   "Scorpion_pose_or_vrischikasana-b", "Intense_Side_Stretch_Pose_or_Parsvottanasana_-b"])
    ap.add_argument("--out", default=str(HERE / "data/drag_locate.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    rec = json.loads((REPO / "data/reference_curation/releases" / f"{RELEASE}.json").read_text())
    rdir = REPO / rec["dir"]
    holds = {c["stem"]: c["holds"] or [] for c in yaml.safe_load(open(rdir / "holds_extended.yaml"))["clips"]}
    out = {}
    tables = None
    sources = [(str(e), RUN / f"results/predicted_motion_lib_epoch_{e}.pt", RUN / f"curriculum/eval_epoch_{e:06d}.csv")
               for e in args.epochs]
    if args.standalone:
        sd = Path(args.standalone)
        sources.append(("standalone", sd / "results/predicted_motion_lib_epoch_0.pt",
                        sorted((sd / "curriculum").glob("eval_epoch_*.csv"))[-1]))
    for epoch, lib_path, csv_path in sources:
        lib = torch.load(lib_path, map_location="cpu", weights_only=False)
        names = [Path(str(f)).stem for f in lib["motion_files"]]
        if tables is None:
            payload = torch.load(rdir / "physics_tables.pt", map_location="cpu", weights_only=False)
            body_names = list(payload["body_names"])
            tables = PhysicsTables(str(rdir / "physics_tables.pt"), names, body_names, "cpu")
            body_idx = [[body_names.index(b) for b in bodies] for _, bodies in ZONES]
        csv_drag = {r["motion"]: float(r["drag_J"]) for r in csv.DictReader(open(csv_path))}
        diffs = []
        res = {}
        for i, stem in enumerate(names):
            work, dt = per_frame_drag(lib, i, tables, body_idx)
            total = float(work.sum())
            diffs.append(abs(total - csv_drag[stem]))
            if VARIANT.search(stem) or not any(c in stem for c in args.clips):
                continue
            segs = segments(work.sum(-1), dt)
            segs = sorted(segs, key=lambda s: -s[2])[:6]
            def where(t):
                for h in holds[stem]:
                    if h["t_start"] - 0.25 <= t <= h["t_end"] + 0.25:
                        return f"hold {h['name'][:24]} [{h['t_start']:.1f}-{h['t_end']:.1f}]"
                return "between holds"
            res[stem] = dict(total_J=total, by_zone={z: float(work[:, k].sum()) for k, (z, _) in enumerate(ZONES)},
                             segments=[dict(t0=round(s[0], 2), t1=round(s[1], 2), J=round(s[2], 1),
                                            where=where(0.5 * (s[0] + s[1]))) for s in segs])
        out[epoch] = res
        print(f"epoch {epoch}: |recomputed - CSV drag_J| max {max(diffs):.3f} J over {len(diffs)} motions")
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for stem in out[sources[0][0]]:
        print(f"\n{stem[7:]}")
        for e in out:
            r = out[e].get(stem)
            if not r:
                continue
            zones = " ".join(f"{z} {v:.0f}" for z, v in r["by_zone"].items() if v > 0.5)
            print(f"  {e:>10s}: {r['total_J']:6.1f} J ({zones})")
            for s in r["segments"][:4]:
                print(f"      {s['t0']:6.2f}-{s['t1']:6.2f} s  {s['J']:6.1f} J  {s['where']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
