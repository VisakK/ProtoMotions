"""Side-by-side review videos: e15500 (left) and G1 epoch 5,000 (right), same clip, same clock.

Both renders come from ``render_policy_videos.py`` with identical flags (one env, full clip from t = 0, deterministic
actions, no termination, ghost on), so frame k of either video is the same clip time. Each panel is scaled to
960 x 540 and labelled; a clip-time counter runs across the top (the video runs about 0.1 s behind clip time, README
§7). The shorter video holds its last frame. Also stacks each clip's contact sheets and timelines (e15500 above
G1), so the contact detail the follow camera cannot show is compared on the same clip clock.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g1_amp_eval/side_by_side.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEFT = ROOT / "output/renderings/expert56_v2_e15500"
RIGHT = ROOT / "output/renderings/expert56_v2_amp_g1_e5000"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def label(text: str) -> str:
    return (f"drawtext=fontfile={FONT}:text='{text}':x=14:y=12:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.55:boxborderw=8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--left", default=str(LEFT))
    ap.add_argument("--right", default=str(RIGHT))
    ap.add_argument("--out", default=str(RIGHT / "side_by_side"))
    ap.add_argument("--left-label", default="e15500  (no AMP)")
    ap.add_argument("--right-label", default="G1  AMP fine-tune, epoch 5000")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    left, right, out = Path(args.left), Path(args.right), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"^(\d\d)_([a-z_]+?)_(\d{6}_.+?)_[\d.]+s_[a-z]+")
    made = 0
    for lv in sorted(left.glob("[0-9][0-9]_*.mp4")):
        m = pat.match(lv.stem)
        if not m:
            continue
        nn, group, clip = m.groups()
        rv = sorted(right.glob(f"{nn}_{group}_{clip}_*.mp4"))
        if not rv:
            print(f"missing right video for {lv.name}")
            continue
        rv = rv[0]
        dst = out / f"{nn}_{group}_{clip}_e15500_vs_g1.mp4"
        if dst.exists() and not args.overwrite:
            continue
        # frame i of a render is clip time (i + 3) dt (reset + 2 settle steps), so clip t = video t + 0.1 s
        timer = (f"drawtext=fontfile={FONT}:text='clip t = %{{eif\\:t+0.1\\:d}}.%{{eif\\:mod((t+0.1)*10\\,10)\\:d}} s':"
                 "x=(w-text_w)/2:y=h-text_h-16:fontsize=30:fontcolor=yellow:box=1:boxcolor=black@0.55:boxborderw=8")
        fc = (f"[0:v]scale=960:540,{label(args.left_label)},tpad=stop_mode=clone:stop_duration=60[a];"
              f"[1:v]scale=960:540,{label(args.right_label)},tpad=stop_mode=clone:stop_duration=60[b];"
              f"[a][b]hstack=inputs=2,{timer}[v]")
        dur = max(float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                        "csv=p=0", str(p)], capture_output=True, text=True).stdout or 0)
                  for p in (lv, rv))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(lv), "-i", str(rv), "-filter_complex", fc,
                        "-map", "[v]", "-t", f"{dur:.2f}", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", str(dst)], check=True)
        for kind in ("sheet", "timeline"):
            ls, rs = lv.with_name(f"{lv.stem}_{kind}.png"), rv.with_name(f"{rv.stem}_{kind}.png")
            if ls.exists() and rs.exists():
                from PIL import Image
                top, bottom = Image.open(ls).convert("RGB"), Image.open(rs).convert("RGB")
                bottom = bottom.resize((top.width, round(bottom.height * top.width / bottom.width)))
                both = Image.new("RGB", (top.width, top.height + bottom.height), "white")
                both.paste(top, (0, 0))
                both.paste(bottom, (0, top.height))
                both.save(out / f"{nn}_{group}_{clip}_{kind}s_e15500_top_g1_bottom.png")
        made += 1
        print(f"  {dst.name}  ({dur:.1f} s)")
    print(f"{made} side-by-side videos -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
