"""``run_sequence_panel.py`` with the goal-pose error measured after the best yaw, not in the heading chart.

``SequenceVizRunner._goal_pose_errors`` puts both the policy's and the goal's pelvis-relative bodies in each row's own
heading frame (``calc_heading_quat_inv``: the root's x axis projected on the floor). That chart is ill-conditioned
when the pelvis is near horizontal, which is exactly where E2 and E5 land (chaturanga): on the evaluator's own clip
rollouts at epoch 3,000 the D-window error reads **0.40 m in the heading chart against 0.03 m after the best yaw**, and
even the reference against its own exemplar reads 0.12-0.21 m (``g3/README.MD``, §4). With ``--metric best_yaw`` the
error is ``score_hold_attainment.best_yaw_dist``'s closed form (the yaw that best aligns the policy's pelvis-relative
bodies to the goal's, per frame), on the same conditionable bodies; everything else is ``run_sequence_panel.py``.

    PYTHONPATH=. ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/run_panel_bestyaw.py \\
        --metric best_yaw --checkpoint results/<run>/epoch_3420.ckpt --num-envs 4096 --no-video \\
        --out-dir output/panel --plans <plan files>
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCRIPTS = REPO / "data" / "scripts"
for p in (str(REPO), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

def _pop(flag: str, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        value = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return type(default)(value)
    return default


metric = _pop("--metric", "heading")
assert metric in ("heading", "best_yaw"), metric
# --dump-positions K: also save the first K replicas of every sequence (body positions, root rotations, the frame
# times and each frame's active goal) to <out-dir>/positions.npz, for offline support and pose checks.
dump_k = _pop("--dump-positions", 0)
out_dir = sys.argv[sys.argv.index("--out-dir") + 1] if "--out-dir" in sys.argv else None

import policy_setup  # noqa: E402  (no torch import before the simulator)

_original_build = policy_setup.build


def _patched_build(args, app_launcher_cls=None):
    built = _original_build(args, app_launcher_cls)
    if metric == "best_yaw":
        import numpy as np
        import torch

        from protomotions.agents.evaluators import sequence_viz as sv

        def _goal_pose_errors(self, sequences, positions, root_rots, frame_times):
            body_ids = self.control.conditionable_body_ids.cpu()
            offsets, flat = [], []
            for sequence in sequences:
                offsets.append(len(flat))
                flat.extend(sequence.goals)
            if not flat:
                return None
            reference = self.env.motion_lib.get_motion_state(
                torch.tensor([g.pose_motion for g in flat], device=self.device),
                torch.tensor([g.pose_time for g in flat], device=self.device, dtype=torch.float32))
            ref_pos = reference.rigid_body_pos.cpu()
            goal_local = ref_pos[:, body_ids] - ref_pos[:, self.pelvis_index].unsqueeze(1)    # [G, nb, 3]
            num_frames, num_cols = positions.shape[0], positions.shape[1]
            errors = np.full((num_frames, num_cols), np.nan, dtype=np.float32)
            num_seq = len(sequences)
            for s, sequence in enumerate(sequences):
                cols = torch.arange(s, num_cols, num_seq)
                if cols.numel() == 0:
                    continue
                pos = positions[:, cols].float()                                              # [T, R, B, 3]
                local = pos[:, :, body_ids] - pos[:, :, self.pelvis_index].unsqueeze(2)       # [T, R, nb, 3]
                goals = goal_local[torch.tensor([offsets[s] + sequence.active_index(t) for t in frame_times])]
                g = goals.unsqueeze(1)                                                        # [T, 1, nb, 3]
                num = (local[..., 0] * g[..., 1] - local[..., 1] * g[..., 0]).sum(-1)
                den = (local[..., 0] * g[..., 0] + local[..., 1] * g[..., 1]).sum(-1)
                th = torch.atan2(num, den).unsqueeze(-1)
                c, sn = torch.cos(th), torch.sin(th)
                x = c * local[..., 0] - sn * local[..., 1]
                y = sn * local[..., 0] + c * local[..., 1]
                rot = torch.stack([x, y, local[..., 2]], -1)
                errors[:, cols.numpy()] = (rot - g).norm(dim=-1).mean(-1).numpy()
            return errors

        sv.SequenceVizRunner._goal_pose_errors = _goal_pose_errors
        print("run_panel_bestyaw: goal-pose error = best-yaw (score_hold_attainment.best_yaw_dist)", flush=True)
    else:
        print("run_panel_bestyaw: goal-pose error = heading chart (stock)", flush=True)

    if dump_k > 0:
        import numpy as np
        import torch

        from protomotions.agents.evaluators import sequence_viz as sv

        scorer = sv.SequenceVizRunner._goal_pose_errors

        def _dumping(self, sequences, positions, root_rots, frame_times):
            num_cols, num_seq = positions.shape[1], len(sequences)
            keep, owner = [], []
            for s in range(num_seq):
                cols = list(range(s, num_cols, num_seq))[:dump_k]
                keep += cols
                owner += [s] * len(cols)
            active = np.array([[seq.active_index(t) for t in frame_times] for seq in sequences], dtype=np.int16)
            np.savez_compressed(
                Path(out_dir) / "positions.npz",
                positions=positions[:, keep].float().cpu().numpy().astype(np.float32),
                root_rots=root_rots[:, keep].float().cpu().numpy().astype(np.float32),
                columns=np.array(keep), owner=np.array(owner), frame_times=np.array(frame_times, dtype=np.float32),
                active_goal=active, sequences=np.array([seq.name for seq in sequences]),
                goals=np.array([";".join(g.name for g in seq.goals) for seq in sequences]),
                goal_pose=np.array([";".join(f"{g.pose_motion}@{g.pose_time:.4f}" for g in seq.goals)
                                    for seq in sequences]),
                goal_timing=np.array([";".join(f"{g.reach_s:.4f},{g.hold_s:.4f}" for g in seq.goals)
                                      for seq in sequences]))
            print(f"run_panel_bestyaw: dumped {len(keep)} replica columns x {positions.shape[0]} frames", flush=True)
            return scorer(self, sequences, positions, root_rots, frame_times)

        sv.SequenceVizRunner._goal_pose_errors = _dumping
    return built


policy_setup.build = _patched_build

if __name__ == "__main__":
    sys.argv[0] = str(SCRIPTS / "run_sequence_panel.py")
    runpy.run_path(str(SCRIPTS / "run_sequence_panel.py"), run_name="__main__")
