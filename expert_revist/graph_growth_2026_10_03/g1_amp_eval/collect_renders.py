"""Move the two render parts into one folder, numbered like the e15500 review (``NN_<group>_<clip>...``).

``render_policy_videos.py`` names its outputs ``<k>_<clip>_<duration>s_<end>.<ext>`` with ``k`` the position in its own
``--motion-ids`` list; the e15500 review renamed them into its §7 review order. This applies the same numbering to
G1's batch, taking each clip's ``NN_<group>`` from the e15500 folder, so the two folders line up file for file.

    ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g1_amp_eval/collect_renders.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E15 = ROOT / "output/renderings/expert56_v2_e15500"
G1 = ROOT / "output/renderings/expert56_v2_amp_g1_e5000"
CLIP = re.compile(r"(\d{6}_.+?)_[\d.]+s_[a-z]+")


def main() -> int:
    order = {}
    for mp4 in E15.glob("[0-9][0-9]_*.mp4"):
        m = re.match(r"(\d\d_[a-z_]+?)_(\d{6}_.+?)_[\d.]+s_[a-z]+$", mp4.stem)
        order[m.group(2)] = m.group(1)
    moved = 0
    for part in ("part_a", "part_b"):
        for f in sorted((G1 / part).glob("*")):
            if f.is_dir():
                continue
            m = re.match(r"\d{3}_(.+)$", f.name)
            if not m:
                continue
            clip = CLIP.match(m.group(1)).group(1)
            if clip not in order:
                print(f"  no e15500 slot for {f.name}")
                continue
            dst = G1 / f"{order[clip]}_{m.group(1)}"
            shutil.move(str(f), dst)
            moved += 1
    print(f"moved {moved} files; {len(list(G1.glob('[0-9][0-9]_*.mp4')))} numbered videos in {G1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
