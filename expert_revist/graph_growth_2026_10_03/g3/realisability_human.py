"""``reference_curation.realisability_v2`` on the human motions of a release-v3 rollout library.

The scorer reads each rollout stem's human evidence (capture v4), which synthetic clips do not have, so it raises on
a v3 library. Release v3's 168 human motions are release v2's byte for byte (R3's check), so this scores them under
release v2, which makes the numbers directly comparable with e15500's and G1's (``g1_amp_eval/data/realisability/``).

    PYTHONPATH=.:data/scripts OMP_NUM_THREADS=1 ../env_isaaclab/bin/python \\
        expert_revist/graph_growth_2026_10_03/g3/realisability_human.py --library <lib.pt> --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "data" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from reference_curation import realisability_v2 as rv  # noqa: E402

RELEASE_V2 = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    start = time.time()
    rollouts = {s: v for s, v in rv.load_rollouts(Path(args.library)).items() if not s.startswith("SYN_")}
    result = rv.evaluate(RELEASE_V2, rollouts)
    record = {"release_id": RELEASE_V2, "rollouts": str(args.library), "motions": len(rollouts),
              "note": "human motions of a release-v3 library, scored under release v2 (identical motions)",
              "seconds": round(time.time() - start, 1), **result}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    q = result["pooled"]
    print(f"{args.library}: {len(rollouts)} human motions; tracked {q['tracked_share']} ({q['tracked_motion_mean']} per "
          f"motion), slide p50/p90 {q['slide_cm_s_p50']}/{q['slide_cm_s_p90']} cm/s, float {q['float_share_over_2cm']}, "
          f"COM-COP p50/p90 {q['com_cop_cm_p50']}/{q['com_cop_cm_p90']} cm -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
