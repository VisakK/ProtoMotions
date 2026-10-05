"""Where G3's drag happens, per clip and time: G1's ``drag_locate.py`` kernel on G3's libraries (release v3).

Drag is the evaluator's friction work while a foot or hand carries more than 50 N and slides faster than 0.10 m/s
(``physics_terms.clip_drag``). The per-frame kernel is G1's (``../g1_amp_eval/drag_locate.py``), validated against the
evaluator's CSV to the same tolerance here; the clips are the largest drag contributors at epoch 3,420 plus G1's
stance-closing clips (Side Plank -a/-b, Plank -a, Downward Dog -a, Dolphin Plank). Writes ``data/drag_locate.json``.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/drag_locate.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, HERE.parent / "g1_amp_eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import drag_locate as g1dl  # noqa: E402  (G1's kernel: per_frame_drag, segments, ZONES)
from protomotions.envs.control.physics_terms import PhysicsTables  # noqa: E402

RELEASE = "holds_repaired_ftC_posefix.release_v3.2f132f4299"
RUN = REPO / "results/smpl_yogi_v2_expert56_g3_2f132f4299"
STANDALONE = REPO / "output/renderings/expert56_v2_g3_e3420/eval_standalone_e3420"
CLIPS = ("Side_Plank_Pose_or_Vasisthasana_-a", "Side_Plank_Pose_or_Vasisthasana_-b", "Plank_Pose_or_Kumbhakasana_-a",
         "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a", "Dolphin_Plank", "Upward_Plank_Pose_or_Purvottanasana_-a",
         "Half_Moon_Pose_or_Ardha_Chandrasana_-b", "Scorpion_pose_or_vrischikasana-a", "Koundinyanasana_I_and_II-b",
         "Low_Lunge_pose_or_Anjaneyasana_-a", "Shoulder-Pressing_Pose_or_Bhujapidasana_-a", "Plow_Pose_or_Halasana_-b")
VARIANT = re.compile(r"_x\d+s$")


def main() -> int:
    torch.set_num_threads(2)
    rec = json.loads((REPO / "data/reference_curation/releases" / f"{RELEASE}.json").read_text())
    rdir = REPO / rec["dir"]
    holds = {c["stem"]: c["holds"] or [] for c in yaml.safe_load(open(rdir / "holds_extended.yaml"))["clips"]}
    sources = [("3000", RUN / "results/predicted_motion_lib_epoch_3000.pt", RUN / "curriculum/eval_epoch_003000.csv"),
               ("3420s", STANDALONE / "results/predicted_motion_lib_epoch_0.pt",
                STANDALONE / "curriculum/eval_epoch_000000.csv")]
    out, tables = {}, None
    for label, lib_path, csv_path in sources:
        lib = torch.load(lib_path, map_location="cpu", weights_only=False)
        names = [Path(str(f)).stem for f in lib["motion_files"]]
        if tables is None:
            payload = torch.load(rdir / "physics_tables.pt", map_location="cpu", weights_only=False)
            body_names = list(payload["body_names"])
            tables = PhysicsTables(str(rdir / "physics_tables.pt"), names, body_names, "cpu")
            body_idx = [[body_names.index(b) for b in bodies] for _, bodies in g1dl.ZONES]
        csv_drag = {r["motion"]: float(r["drag_J"]) for r in csv.DictReader(open(csv_path))}
        diffs, res = [], {}
        for i, stem in enumerate(names):
            work, dt = g1dl.per_frame_drag(lib, i, tables, body_idx)
            total = float(work.sum())
            diffs.append(abs(total - csv_drag[stem]))
            if VARIANT.search(stem) or stem.startswith("SYN_") or not any(c in stem for c in CLIPS):
                continue
            segs = sorted(g1dl.segments(work.sum(-1), dt), key=lambda s: -s[2])[:6]

            def where(t, stem=stem):
                for h in holds[stem]:
                    if h["t_start"] - 0.25 <= t <= h["t_end"] + 0.25:
                        return f"hold {h['name'][:24]} [{h['t_start']:.1f}-{h['t_end']:.1f}]"
                return "between holds"
            a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
            pos = lib["gts"][a:a + n].float()
            rows = []
            for s in segs:
                f0, f1 = int(round(s[0] / dt)) - 1, int(round(s[1] / dt)) - 1
                feet = (pos[f0:f1 + 1, body_names.index("L_Ankle"), :2] - pos[f0:f1 + 1, body_names.index("R_Ankle"), :2])
                stance = feet.norm(dim=-1)
                rows.append(dict(t0=round(s[0], 2), t1=round(s[1], 2), J=round(s[2], 1), where=where(0.5 * (s[0] + s[1])),
                                 pelvis_z=round(float(pos[f0:f1 + 1, 0, 2].median()), 2),
                                 stance_m=[round(float(stance[0]), 2), round(float(stance[-1]), 2)]))
            res[stem] = dict(total_J=total, by_zone={z: float(work[:, k].sum()) for k, (z, _) in enumerate(g1dl.ZONES)},
                             segments=rows)
        out[label] = res
        print(f"{label}: |recomputed - CSV drag_J| max {max(diffs):.3f} J over {len(diffs)} motions")
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/drag_locate.json").write_text(json.dumps(out, indent=1) + "\n")
    for stem in sorted(out["3420s"], key=lambda s: -out["3420s"][s]["total_J"]):
        print(f"\n{stem[7:]}")
        for label in out:
            r = out[label].get(stem)
            if not r:
                continue
            zones = " ".join(f"{z} {v:.0f}" for z, v in r["by_zone"].items() if v > 0.5)
            print(f"  {label:>6s}: {r['total_J']:6.1f} J ({zones})")
            for s in r["segments"][:3]:
                print(f"      {s['t0']:6.2f}-{s['t1']:6.2f} s  {s['J']:6.1f} J  {s['where']}  pelvis {s['pelvis_z']} m, "
                      f"stance {s['stance_m'][0]} -> {s['stance_m'][1]} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
