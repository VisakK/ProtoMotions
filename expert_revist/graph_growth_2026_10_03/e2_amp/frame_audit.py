"""Why amp_features_v1 separates the e15500 rollouts from the reference perfectly (card E2 item 5 follow-up).

Per joint, three local-rotation comparisons on the evaluator's epoch-15,500 rollouts (frame i = clip time (i+1) dt):
  (a) simulator body frames vs the simulator's own joint coordinates: conj(R_parent) R_child from ``grs`` against
      exp(``dps``) -- disagreement = PhysX body frames differ from pose_lib's (a feature-convention mismatch);
  (b) the same check on the reference library (its stored rotations against its stored dof_pos);
  (c) the policy's local rotation against the reference's at the same clip time (tracked frames): the systematic
      (mean) offset per joint and its spread -- a real posture bias if (a) and (b) agree.
"""
import sys, json, re
from pathlib import Path
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.utils import rotations as R

BODIES = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine",
          "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder",
          "R_Elbow", "R_Wrist", "R_Hand"]
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]
CH = list(range(1, 24)); PA = [PARENTS[j] for j in CH]

def local_from_world(rot):            # [N, 24, 4] xyzw -> [N, 23, 4]
    return R.quat_mul(R.quat_conjugate(rot[:, PA], True), rot[:, CH], True)

def local_from_dof(dof):              # [N, 69] exp-map -> [N, 23, 4] xyzw
    return R.exp_map_to_quat(dof.reshape(-1, 23, 3).reshape(-1, 3), True).reshape(-1, 23, 4)

def angle(q1, q2):                    # [.., 4] -> degrees
    d = R.quat_mul(R.quat_conjugate(q1.reshape(-1, 4), True), q2.reshape(-1, 4), True)
    return torch.rad2deg(2 * torch.atan2(d[:, :3].norm(dim=-1), d[:, 3].abs())).reshape(q1.shape[:-1])

def rotvec(q):                        # relative rotation as a rotation vector (deg), shortest arc
    q = torch.where(q[..., 3:4] < 0, -q, q)
    return torch.rad2deg(R.quat_to_exp_map(q.reshape(-1, 4), True)).reshape(q.shape[:-1] + (3,))

rec = json.loads((REPO / "data/reference_curation/releases/holds_repaired_ftC_posefix.release_v2.a2dda5d2ac.json").read_text())
ref = MotionLib(MotionLibConfig(motion_file=str(REPO / rec["dir"] / "motions.pt")), device="cpu")
stems = [Path(str(f)).name.rsplit(".motion", 1)[0] for f in ref.motion_files]
lib = torch.load(REPO / "results/smpl_yogi_v2_expert56_a2dda5d2ac/results/predicted_motion_lib_epoch_15500.pt",
                 map_location="cpu", weights_only=False)

# (a) simulator: body frames vs its own joints
grs, dps = lib["grs"].float(), lib["dps"].float()
a = angle(local_from_world(grs), local_from_dof(dps))
# (b) reference library: stored rotations vs stored dofs
b = angle(local_from_world(ref.grs.float()), local_from_dof(ref.dps.float()))
print("joint        (a) sim body vs sim dof   (b) ref body vs ref dof    deg p50 / p99")
for k, j in enumerate(CH):
    print(f"  {BODIES[j]:<11}  {a[:, k].median():8.3f} / {torch.quantile(a[:, k], 0.99):8.3f}"
          f"        {b[:, k].median():8.3f} / {torch.quantile(b[:, k], 0.99):8.3f}")

# (c) policy vs reference at matched clip times, tracked frames only
diffs, ok_all = [], []
for i, f in enumerate(lib["motion_files"]):
    stem = Path(str(f)).stem
    s, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
    m = stems.index(stem)
    t = (torch.arange(n, dtype=torch.float32) + 1.0) * float(lib["motion_dt"][i])
    st = ref.get_motion_state(torch.full((n,), m, dtype=torch.long), t)
    sim_pos = lib["gts"][s:s + n].float().clone()
    sim_pos[..., :2] -= sim_pos[0, 0, :2] - st.rigid_body_pos[0, 0, :2]
    ok = (sim_pos - st.rigid_body_pos).norm(dim=-1).max(-1).values < 0.5
    rel = R.quat_mul(R.quat_conjugate(local_from_world(st.rigid_body_rot).reshape(-1, 4), True),
                     local_from_world(grs[s:s + n]).reshape(-1, 4), True).reshape(n, 23, 4)
    diffs.append(rotvec(rel)[ok]); ok_all.append(ok)
d = torch.cat(diffs)                  # [frames, 23, 3]: reference -> policy local rotation, in the parent-relative child frame
print(f"\n(c) policy vs reference local rotation, {d.shape[0]} tracked frames: mean rotation vector (deg, x/y/z) and |mean|/sd")
mean, sd = d.mean(0), d.std(0)
order = torch.argsort(mean.norm(dim=-1), descending=True)
for k in order[:10].tolist():
    print(f"  {BODIES[CH[k]]:<11} mean [{mean[k,0]:6.2f} {mean[k,1]:6.2f} {mean[k,2]:6.2f}]  |mean| {mean[k].norm():5.2f}  "
          f"sd [{sd[k,0]:5.2f} {sd[k,1]:5.2f} {sd[k,2]:5.2f}]  median angle {d[:, k].norm(dim=-1).median():5.2f}")
