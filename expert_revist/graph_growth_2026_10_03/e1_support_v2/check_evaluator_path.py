"""Card E1: run ``HoldCurriculumEvaluator``'s own scoring path (setup, batched v2 inputs, per-motion v2, curriculum,
CSV, logs) on a saved library, on the CPU, with no simulator.

The evaluator's ``_metrics`` are filled from the library. Libraries saved before card E1 carry no ground forces, so
unless the library has ``sim_rigid_body_ground_forces`` the forces are synthesised from the contact-flag proxy of
``rescore_library.py``: 100 N vertical on every body whose contact flag is set while its own lowest collider point is
within 2 cm of the floor. The evaluator's counts must then match the offline proxy mode.

    PYTHONPATH=. python expert_revist/graph_growth_2026_10_03/e1_support_v2/check_evaluator_path.py \\
        --library results/smpl_yogi_v2_expert56_a2dda5d2ac/results/predicted_motion_lib_epoch_15500.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protomotions.agents.evaluators.config import HoldCurriculumConfig, HoldCurriculumEvaluatorConfig  # noqa: E402
from protomotions.agents.evaluators.hold_curriculum import zone_lowest_points  # noqa: E402
from protomotions.agents.evaluators.hold_curriculum_evaluator import HoldCurriculumEvaluator  # noqa: E402
from protomotions.agents.evaluators.metrics import MotionMetrics  # noqa: E402
from protomotions.components.contact_graph import ContactGraph  # noqa: E402
from protomotions.components.motion_lib import MotionLib, MotionLibConfig  # noqa: E402
from protomotions.envs.control.contact_targets import ContactTargets  # noqa: E402
from protomotions.envs.control.physics_terms import PhysicsTables  # noqa: E402

RELEASE = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"


class _MotionManager:
    def __init__(self, n):
        self.motion_weights = torch.zeros(n)

    def update_sampling_weights(self, w):
        self.motion_weights[:] = w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--release", default=RELEASE)
    ap.add_argument("--rule", default="v2", choices=["v1", "v2"])
    ap.add_argument("--out", help="write the logs as JSON here")
    args = ap.parse_args()

    rec = json.loads((REPO / "data/reference_curation/releases" / f"{args.release}.json").read_text())
    art = {k: REPO / v["path"] for k, v in rec["artifacts"].items()}
    lib = torch.load(args.library, map_location="cpu", weights_only=False)
    stems = [Path(str(f)).stem for f in lib["motion_files"]]
    motion_lib = MotionLib(MotionLibConfig(motion_file=str(art["package"])), device="cpu")
    graph = ContactGraph.from_file(str(art["graph"]), device="cpu")
    body_names = list(torch.load(art["physics_tables"], map_location="cpu", weights_only=False)["body_names"])
    tables = PhysicsTables(str(art["physics_tables"]), stems, body_names, "cpu")
    targets = ContactTargets(str(art["contact_targets"]), graph, stems, "cpu")
    B, M = len(body_names), len(stems)

    frames = lib["motion_num_frames"].long()
    L = int(frames.max())
    metrics = {k: MotionMetrics(M, frames, L, num_sub_features=w * B, device=torch.device("cpu"))
               for k, w in (("rigid_body_pos", 3), ("rigid_body_rot", 4), ("rigid_body_ground_forces", 3))}
    forces = lib.get("sim_rigid_body_ground_forces")
    source = "library" if forces is not None else "synthesised from the contact-flag proxy"
    for m in range(M):
        a, n = int(lib["length_starts"][m]), int(frames[m])
        pos, rot = lib["gts"][a:a + n].float(), lib["grs"][a:a + n].float()
        if forces is not None:
            f = forces[a:a + n].float()
        else:
            low = zone_lowest_points(pos, rot, tables, [[b] for b in range(B)])      # per body
            f = torch.zeros(n, B, 3)
            f[..., 2] = 100.0 * (lib["contacts"][a:a + n].bool() & (low <= 0.02)).float()
        for key, x in (("rigid_body_pos", pos), ("rigid_body_rot", rot), ("rigid_body_ground_forces", f)):
            metrics[key].data[m, :n] = x.reshape(n, -1)
            metrics[key].frame_counts[m] = n

    control = SimpleNamespace(_physics=tables, _targets=targets, graph=graph, release=None)
    env = SimpleNamespace(
        control_manager=SimpleNamespace(components={"contact_graph": control}),
        robot_config=SimpleNamespace(kinematic_info=SimpleNamespace(body_names=body_names, num_bodies=B)),
        dt=float(lib["motion_dt"][0]), motion_manager=_MotionManager(M),
    )
    root = Path(tempfile.mkdtemp(prefix="e1_eval_"))
    ev = object.__new__(HoldCurriculumEvaluator)
    ev.config = HoldCurriculumEvaluatorConfig(
        hold_manifest=str(art["holds_extended"]),
        curriculum=HoldCurriculumConfig(support_rule=args.rule),
    )
    ev.agent = SimpleNamespace(env=env, motion_lib=motion_lib, root_dir=root, current_epoch=15500)
    ev.fabric = SimpleNamespace(global_rank=0, device=torch.device("cpu"))
    ev._metrics = metrics
    ev._score_ema, ev._holds, ev._groups, ev._body_ids, ev._drag_tables = None, None, None, None, None
    ev._support_rule, ev._v2, ev._v2_text = "v1", None, None

    t0 = time.time()
    scores = ev._score_all_motions()
    t_score = time.time() - t0
    t0 = time.time()
    v2_inputs = ev._support_v2_inputs(B)
    t_batch = time.time() - t0
    probs = ev._update_curriculum(scores)
    ev._write_table(scores, probs)
    logs = ev._curriculum_logs(scores, probs)
    print(f"forces: {source}; curriculum rule {args.rule}; scoring all {M} motions took {t_score:.2f} s on the CPU "
          f"(of which the batched v2 geometry/load pass {t_batch:.2f} s)")
    for k in sorted(logs):
        if "_v2" in k or k in ("eval/perf/score", "eval/perf/support_violation", "eval/perf/hold"):
            print(f"  {k} = {logs[k]:.4f}")
    flagged = [(stems[m][7:], ev._v2_text[m]) for m in range(M) if ev._x0[m] and ev._v2_text[m]]
    print("x0 clips with flagged holds:")
    for s, t in flagged:
        print(f"  {s}: {t}")
    csv_path = root / "curriculum" / "eval_epoch_015500.csv"
    print(f"CSV header: {csv_path.read_text().splitlines()[0]}")
    if args.out:
        Path(args.out).write_text(json.dumps({"forces": source, "rule": args.rule, "logs": logs,
                                              "flagged_x0": flagged, "seconds": t_score}, indent=1) + "\n")
    assert v2_inputs is not None
    return 0


if __name__ == "__main__":
    sys.exit(main())
