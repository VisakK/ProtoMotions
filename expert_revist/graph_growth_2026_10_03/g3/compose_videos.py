"""Caption and assemble the G3 review renders (``render_review.py``) into the deliverable videos.

Inputs: the two render folders (``part_a``, ``part_b``: ``NNN_<stem>_<dur>s_clipend.mp4``, 1280x720, 30 fps, video
frame 0 = clip time 0.1 s), ``data/per_clip.json``, ``data/edge_tracking.json`` and the synthetic clips' layouts
(R2's per-clip JSON). Outputs, under ``output/renderings/expert56_v2_g3_e3420/``:

* ``human/H<NN>_<group>_<clip>.mp4``: the 56 human clips, each captioned (clip, group, G3 score against G1's, tracked
  share, mean body error; clip time; which figure is which);
* ``edges/<edge>_<variant>.mp4``: the 28 synthetic clips, captioned the same way plus a phase banner from the clip's
  lineage (human lead-in, the S hold, SYNTHESISED TRANSITION, hold at D, human lead-out);
* ``G3_e3420_new_edges.mp4``: the new-edges video. Per edge: a title card, the edge's showcase variant (R4a's plan
  clip) from 3 s before the departure to 4 s after the arrival, then every variant of that edge in a grid, all
  synchronised at their departure;
* ``G3_e3420_new_edges_all_variants.mp4``: all 28 captioned synthetic clips in full, edge by edge;
* ``G3_e3420_human_<group>.mp4``: the human clips of each group in full, with a title card per group.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/compose_videos.py [--only human|edges]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "output/renderings/expert56_v2_g3_e3420"
PARTS = (OUT / "part_a", OUT / "part_b")
SYN_DIR = REPO / "data/smpl/reference_curation/synthetic_v3/synthetic_v3.d034199f23"
RELEASE_DIR = REPO / "data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v3.2f132f4299"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
VIDEO_T0 = 0.1                       # clip time of video frame 0 (reset + 2 settle steps, 30 Hz)
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30"]
GROUPS = ("single_leg", "inversion", "arm_balance", "connective")
GROUP_NAME = {"single_leg": "Single-leg balances", "inversion": "Inversions", "arm_balance": "Arm balances",
              "connective": "Connective poses"}
EDGES = ("E1", "E3", "B1", "E2", "E5")
EDGE_NAME = {"E1": "E1 press: crow -> handstand", "E3": "E3 lower: handstand -> crow",
             "B1": "B1 jump: crow -> plank", "E2": "E2 jump-back: crow -> chaturanga",
             "E5": "E5 float-down: handstand -> chaturanga"}
SHOWCASE = {"E1": "SYN_E1_press_high_s0_t6px", "E3": "SYN_E3_lower_high_s3_t6px12",
            "B1": "SYN_B1_jumpplank_high_s1_t6rpx", "E2": "SYN_E2_jumpback_mid_s0_t6px12",
            "E5": "SYN_E5_floatdown_mid_s0_t6rpx12"}           # R4a's plan clips
BEFORE = {"E1": "before G3 (epoch 1): a foot comes back down 1.1-1.7 s into the press; transition tracked 0.36-0.39",
          "E3": "before G3 (epoch 1): lowered onto the head instead of into crow; transition tracked 0.00-0.50",
          "B1": "before G3 (epoch 1): landed off the plank pose (D error 0.28-0.34 m); 0 of 5 reached D",
          "E2": "before G3 (epoch 1): lost the jump 0.4 s in (tracked 0.42-0.52); 0 of 11 reached D",
          "E5": "before G3 (epoch 1): fell out of the handstand before departing; transition tracked 0.32"}
SHORT_NAMES = {"Pose Dedicated to the Sage Koundinya": "Koundinya", "Feathered Peacock Pose": "Pincha (Feathered Peacock)",
               "Shoulder-Pressing Pose": "Shoulder-Pressing", "viparita virabhadrasana": "Reverse Warrior"}


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:8])} ...\n{r.stderr[-2000:]}")


def nice(clip: str) -> str:
    for long, short in SHORT_NAMES.items():
        if clip.startswith(long):
            return short + clip[len(long):]
    return clip


def renders() -> dict[str, Path]:
    out = {}
    for d in PARTS:
        for p in sorted(d.glob("*.mp4")):
            m = re.match(r"^\d+_(.+?)_\d+\.\ds(_.*)?$", p.stem)
            if m:
                out[m.group(1)] = p
    return out


def esc(path: Path) -> str:
    return str(path).replace(":", r"\:")


def text_file(tmp: Path, name: str, text: str) -> Path:
    p = tmp / f"{name}.txt"
    p.write_text(text)
    return p


def caption_filters(tmp: Path, key: str, line1: str, lines: list[str], extra: list[str] | None = None) -> str:
    """drawtext chain: a header box (one bold line, then smaller lines), the clip time, the figure legend."""
    f1 = text_file(tmp, f"{key}_l1", line1)
    leg = text_file(tmp, f"{key}_leg", "grey: policy (PhysX, G3 epoch 3,420)     green: reference clip (shown 1.8 m to the side)")
    box = "box=1:boxcolor=0x0b0b0b@0.55:boxborderw=8"
    parts = [f"drawtext=fontfile={FONT_B}:textfile={esc(f1)}:x=24:y=20:fontsize=24:fontcolor=white:{box}"]
    for i, line in enumerate(lines):
        f = text_file(tmp, f"{key}_l{i + 2}", line)
        parts.append(f"drawtext=fontfile={FONT}:textfile={esc(f)}:x=24:y={62 + 32 * i}:fontsize=17:"
                     f"fontcolor=0xe1e0d9:{box}")
    parts += [
        f"drawtext=fontfile={FONT}:text='clip time %{{pts\\:hms\\:{VIDEO_T0}}}':x=w-tw-24:y=20:fontsize=18:"
        f"fontcolor=white:{box}",
        f"drawtext=fontfile={FONT}:textfile={esc(leg)}:x=24:y=h-th-20:fontsize=16:fontcolor=white:{box}",
    ]
    return ",".join(parts + (extra or []))


def phase_banners(tmp: Path, key: str, spec: dict, s_name: str, d_name: str) -> list[str]:
    lay = spec["layout"]
    t = lambda f: f / 60.0 - VIDEO_T0  # noqa: E731  (60 fps clip frame -> video time)
    phases = [(0.0, t(lay["s_exemplar"]) - 1.0, f"human lead-in ({spec['S']['stem'][7:].split('_or_')[0].replace('_', ' ')})", "0x2a78d6"),
              (t(lay["s_exemplar"]) - 1.0, t(lay["departure"]), f"S: {s_name} (held)", "0x52514e"),
              (t(lay["departure"]), t(lay["arrival"]), "SYNTHESISED TRANSITION", "0xeb6834"),
              (t(lay["arrival"]), t(lay["lead_out"][0]), f"D: {d_name} (2 s hold, blended onto the human exemplar)", "0x52514e"),
              (t(lay["lead_out"][0]), 1e9, f"human lead-out ({spec['D']['stem'][7:].split('_or_')[0].replace('_', ' ')})", "0x2a78d6")]
    out = []
    for i, (a, b, text, color) in enumerate(phases):
        f = text_file(tmp, f"{key}_ph{i}", text)
        out.append(f"drawtext=fontfile={FONT_B}:textfile={esc(f)}:x=(w-tw)/2:y=h-th-64:fontsize=26:fontcolor=white:"
                   f"box=1:boxcolor={color}@0.85:boxborderw=10:enable='between(t,{max(a, 0):.3f},{b:.3f})'")
    return out


def encode(src: Path, dst: Path, vf: str, ss: float | None = None, dur: float | None = None) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.3f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    run(cmd + ["-vf", vf] + ENC + ["-an", str(dst)])


def title_card(dst: Path, tmp: Path, key: str, lines: list[tuple[str, int, str]], dur: float = 3.0) -> None:
    draws, y = [], 250
    for i, (text, size, color) in enumerate(lines):
        f = text_file(tmp, f"{key}_t{i}", text)
        draws.append(f"drawtext=fontfile={FONT_B if i == 0 else FONT}:textfile={esc(f)}:x=(w-tw)/2:y={y}:"
                     f"fontsize={size}:fontcolor={color}")
        y += size + 26
    run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x1a1a19:s=1280x720:d={dur}:r=30",
         "-vf", ",".join(draws)] + ENC + [str(dst)])


def concat(parts: list[Path], dst: Path, tmp: Path) -> None:
    lst = tmp / f"{dst.stem}_concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(dst)])


def human(rend: dict, tmp: Path) -> None:
    clips = json.load(open(HERE / "data/per_clip.json"))["clips"]
    shown = {r["stem"]: r for r in json.load(open(HERE / "data/render_compare.json"))["rows"]}   # the rendered rollouts
    order = sorted(clips, key=lambda c: (GROUPS.index(c["group"]), c["clip"]))
    dst_dir = OUT / "human"
    dst_dir.mkdir(exist_ok=True)
    per_group: dict[str, list[Path]] = {g: [] for g in GROUPS}
    for n, c in enumerate(order, 1):
        src = rend.get(c["stem"])
        if src is None:
            print(f"  missing render: {c['stem']}")
            continue
        name = nice(c["clip"])
        dst = dst_dir / f"H{n:02d}_{c['group']}_{re.sub(r'[^A-Za-z0-9-]+', '_', name).strip('_')}.mp4"
        k = sum(1 for x in order[:n] if x["group"] == c["group"])
        line1 = f"{name}   |   {GROUP_NAME[c['group']].lower()}  {k}/{sum(1 for x in order if x['group'] == c['group'])}"
        r = shown[c["stem"]]
        line2 = (f"evaluator score: G3 epoch 3,420 {c['g3_3420s']:.2f}, G1 epoch 5,000 {c['g1_5000s']:.2f}   |   "
                 f"this rollout: tracked {r['render_tracked']:.2f}, mean body error {r['render_mean_err']:.3f} m")
        if not dst.exists():
            encode(src, dst, caption_filters(tmp, f"h{n}", line1, [line2]))
        per_group[c["group"]].append(dst)
        print(f"  {dst.name}")
    for g, files in per_group.items():
        card = tmp / f"card_{g}.mp4"
        title_card(card, tmp, f"card_{g}", [(GROUP_NAME[g], 54, "white"),
                                            (f"{len(files)} clips  |  G3 expert, epoch 3,420  |  every clip from t = 0, "
                                             "deterministic actions, no early termination", 22, "0xc3c2b7"),
                                            ("grey = policy (PhysX)      green = reference clip", 22, "0xc3c2b7")])
        concat([card] + files, OUT / f"G3_e3420_human_{g}.mp4", tmp)
        print(f"-> G3_e3420_human_{g}.mp4 ({len(files)} clips)")


def edges(rend: dict, tmp: Path) -> None:
    rows = {r["stem"]: r for r in json.load(open(HERE / "data/edge_tracking.json"))["libraries"]["e3420"]["rows"]}
    dst_dir = OUT / "edges"
    dst_dir.mkdir(exist_ok=True)
    s_name = {"E1": "crow", "E3": "handstand", "B1": "crow", "E2": "crow", "E5": "handstand"}
    d_name = {"E1": "handstand", "E3": "crow", "B1": "plank", "E2": "chaturanga", "E5": "chaturanga"}
    captioned: dict[str, list[tuple[str, Path, dict]]] = {e: [] for e in EDGES}
    for stem, r in sorted(rows.items()):
        if r["x"] != 0:
            continue
        src = rend.get(stem)
        if src is None:
            print(f"  missing render: {stem}")
            continue
        spec = json.loads((SYN_DIR / f"{stem}.json").read_text())
        e = r["edge"]
        line1 = f"{EDGE_NAME[e]}   |   {stem[4:]}"
        line2 = (f"T = {spec['T']:.2f} s   |   evaluator, epoch 3,420: transition tracked {r['trans_tracked']:.2f}, "
                 f"D error {r['d_err6_p50']:.3f} m, D supports down {r['d_support_geom']:.2f}")
        dst = dst_dir / f"{stem[4:]}.mp4"
        if not dst.exists():
            encode(src, dst, caption_filters(tmp, stem, line1, [line2, BEFORE[e]],
                                             phase_banners(tmp, stem, spec, s_name[e], d_name[e])))
        captioned[e].append((stem, dst, spec, src))
        print(f"  {dst.name}")

    # all variants in full
    parts = []
    for e in EDGES:
        card = tmp / f"card_all_{e}.mp4"
        title_card(card, tmp, f"card_all_{e}", [(EDGE_NAME[e], 48, "white"),
                                                (f"{len(captioned[e])} variants, each spliced between real human clips",
                                                 24, "0xc3c2b7"), (BEFORE[e], 20, "0xc3c2b7")])
        parts += [card] + [p for _, p, _, _ in captioned[e]]
    concat(parts, OUT / "G3_e3420_new_edges_all_variants.mp4", tmp)

    # the showcase video: title, showcase variant around its transition, grid of every variant synced at departure
    parts = []
    intro = tmp / "card_intro.mp4"
    title_card(intro, tmp, "card_intro", [("Five new edges, learnt by the G3 expert", 50, "white"),
                                          ("transitions no human clip demonstrates, synthesised by MPPI in PhysX and "
                                           "spliced between real clips (release v3)", 21, "0xc3c2b7"),
                                          ("grey = policy (G3 epoch 3,420)    green = reference    orange banner = "
                                           "the synthesised part", 21, "0xc3c2b7")], dur=4.0)
    parts.append(intro)
    for e in EDGES:
        items = captioned[e]
        if not items:
            continue
        rows_e = [rows[s] for s, _, _, _ in items]
        n_ok = sum(r["reached_D"] for r in rows_e)
        card = tmp / f"card_{e}.mp4"
        title_card(card, tmp, f"card_{e}", [(EDGE_NAME[e], 50, "white"),
                                            (f"G3 epoch 3,420: {n_ok} of {len(items)} variants execute it "
                                             f"(transition tracked, D held with its supports)", 24, "0xc3c2b7"),
                                            (BEFORE[e], 20, "0xc3c2b7")])
        parts.append(card)
        show = next((it for it in items if it[0] == SHOWCASE[e]), items[0])
        _, p, sp, _ = show
        dep, arr = sp["layout"]["departure"] / 60 - VIDEO_T0, sp["layout"]["arrival"] / 60 - VIDEO_T0
        seg = tmp / f"show_{e}.mp4"
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(dep - 3.0, 0):.3f}", "-i", str(p), "-t",
             f"{arr - dep + 7.0:.3f}"] + ENC + ["-an", str(seg)])
        parts.append(seg)
        if len(items) > 1:
            parts.append(grid(e, items, tmp))
    concat(parts, OUT / "G3_e3420_new_edges.mp4", tmp)
    print("-> G3_e3420_new_edges.mp4, G3_e3420_new_edges_all_variants.mp4")


def grid(e: str, items: list, tmp: Path) -> Path:
    """Every variant of edge ``e`` tiled from the raw renders, each trimmed to start 2 s before its own departure
    (synchronised), labelled with its variant and an orange tag while its own transition runs."""
    n = len(items)
    cols = 2 if n <= 4 else (3 if n <= 9 else 4)
    rows = math.ceil(n / cols)
    w, h = 1280 // cols, int(round(720 / cols / 2)) * 2
    T_max = max(sp["T"] for _, _, sp, _ in items)
    dur = 2.0 + T_max + 3.0
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for _, _, sp, raw in items:
        dep = sp["layout"]["departure"] / 60 - VIDEO_T0
        cmd += ["-ss", f"{max(dep - 2.0, 0):.3f}", "-t", f"{dur:.3f}", "-i", str(raw)]
    labels, filters = [], []
    tag = text_file(tmp, f"grid_{e}_tag", "transition")
    for i, (stem, _, sp, _) in enumerate(items):
        f = text_file(tmp, f"grid_{e}_{i}", f"{stem[4:]}  (T {sp['T']:.2f} s)")
        filters.append(f"[{i}:v]scale={w}:{h},drawtext=fontfile={FONT}:textfile={esc(f)}:x=8:y=8:fontsize=15:"
                       f"fontcolor=white:box=1:boxcolor=0x0b0b0b@0.55:boxborderw=4,"
                       f"drawtext=fontfile={FONT_B}:textfile={esc(tag)}:x=(w-tw)/2:y=h-th-8:fontsize=15:fontcolor=white:"
                       f"box=1:boxcolor=0xeb6834@0.9:boxborderw=4:enable='between(t,2.0,{2.0 + sp['T']:.3f})'[v{i}]")
        labels.append(f"[v{i}]")
    for i in range(n, cols * rows):                  # pad the grid with blank tiles
        filters.append(f"color=c=0x1a1a19:s={w}x{h}:d={dur:.3f}:r=30[v{i}]")
        labels.append(f"[v{i}]")
    layout = "|".join(f"{(i % cols) * w}_{(i // cols) * h}" for i in range(cols * rows))
    note = text_file(tmp, f"grid_{e}_note", f"{EDGE_NAME[e]}: all {n} variants, synchronised 2 s before departure")
    filters.append("".join(labels) + f"xstack=inputs={cols * rows}:layout={layout}:fill=0x1a1a19,"
                   f"pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x1a1a19,"
                   f"drawtext=fontfile={FONT_B}:textfile={esc(note)}:x=(w-tw)/2:y=h-th-12:fontsize=20:fontcolor=white:"
                   f"box=1:boxcolor=0x0b0b0b@0.6:boxborderw=6[out]")
    dst = tmp / f"grid_{e}.mp4"
    run(cmd + ["-filter_complex", ";".join(filters), "-map", "[out]", "-t", f"{dur:.3f}"] + ENC + ["-an", str(dst)])
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=("human", "edges"), default=None)
    args = ap.parse_args()
    tmp = OUT / "_compose_tmp"
    tmp.mkdir(exist_ok=True)
    rend = renders()
    print(f"{len(rend)} renders found")
    if args.only in (None, "edges"):
        edges(rend, tmp)
    if args.only in (None, "human"):
        human(rend, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
