"""Are the velocity features defined the same way on both sides? (card E2 item 5 follow-up)

Root linear / angular velocity: stored value vs a central finite difference of the stored root pose, on the reference
library (60 fps) and on the e15500 rollouts (30 Hz). Then the high-frequency content of each side's stored root velocity
(second difference at 30 Hz), and per-joint single-frame separability of the local rotations.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.utils import rotations as R
sys.path.insert(0, str(Path(__file__).parent))
from amp_separability import auroc, train_mlp, predict, SMPL_BODIES, SMPL_PARENTS
from protomotions.envs.obs.amp_features import amp_frame_features_v1, amp_features_v1_params

rec = json.loads((REPO / "data/reference_curation/releases/holds_repaired_ftC_posefix.release_v2.a2dda5d2ac.json").read_text())
ref = MotionLib(MotionLibConfig(motion_file=str(REPO / rec["dir"] / "motions.pt")), device="cpu")
stems = [Path(str(f)).name.rsplit(".motion", 1)[0] for f in ref.motion_files]
lib = torch.load(REPO / "results/smpl_yogi_v2_expert56_a2dda5d2ac/results/predicted_motion_lib_epoch_15500.pt",
                 map_location="cpu", weights_only=False)

def fd_check(pos, rot, vel, ang, dt):
    """central differences of root position / rotation vs stored root velocities; returns rms(stored), rms(stored - fd)"""
    p, q = pos[:, 0].double(), rot[:, 0].double()
    v_fd = (p[2:] - p[:-2]) / (2 * dt)
    dq = R.quat_mul(q[2:], R.quat_conjugate(q[:-2], True), True)          # world-frame rotation over 2 dt
    dq = torch.where(dq[:, 3:4] < 0, -dq, dq)
    w_fd = R.quat_to_exp_map(dq, True) / (2 * dt)
    v, w = vel[1:-1, 0].double(), ang[1:-1, 0].double()
    return (v.norm(dim=-1).pow(2).mean().sqrt().item(), (v - v_fd).norm(dim=-1).pow(2).mean().sqrt().item(),
            w.norm(dim=-1).pow(2).mean().sqrt().item(), (w - w_fd).norm(dim=-1).pow(2).mean().sqrt().item())

x0 = [i for i, s in enumerate(stems) if not s.endswith(("_x3s", "_x7s"))]
acc = []
for m in x0:
    a, n = int(ref.length_starts[m]), int(ref.motion_num_frames[m])
    acc.append(fd_check(ref.gts[a:a+n], ref.grs[a:a+n], ref.gvs[a:a+n], ref.gavs[a:a+n], float(ref.motion_dt[m])))
r = np.mean(acc, axis=0)
print(f"reference (60 fps): root lin vel rms {r[0]:.4f}, |stored - central FD| rms {r[1]:.4f} m/s; "
      f"ang vel rms {r[2]:.4f}, |stored - FD| rms {r[3]:.4f} rad/s")
acc = []
for i in range(len(lib["motion_files"])):
    a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
    acc.append(fd_check(lib["gts"][a:a+n], lib["grs"][a:a+n], lib["gvs"][a:a+n], lib["gavs"][a:a+n], float(lib["motion_dt"][i])))
r = np.mean(acc, axis=0)
print(f"agent (30 Hz):      root lin vel rms {r[0]:.4f}, |stored - central FD| rms {r[1]:.4f} m/s; "
      f"ang vel rms {r[2]:.4f}, |stored - FD| rms {r[3]:.4f} rad/s")

# high-frequency content at the AMP sampling rate: second difference of the stored root velocity at 30 Hz
def hf(v):  # v [T, 3] at 30 Hz
    return (v[2:] - 2 * v[1:-1] + v[:-2]).norm(dim=-1).median().item()
hf_r, hf_a = [], []
for i, f in enumerate(lib["motion_files"]):
    stem = Path(str(f)).stem
    if stem.endswith(("_x3s", "_x7s")): continue
    a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
    t = (torch.arange(n, dtype=torch.float32) + 1.0) / 30.0
    st = ref.get_motion_state(torch.full((n,), stems.index(stem), dtype=torch.long), t)
    hf_r.append(hf(st.rigid_body_vel[:, 0])); hf_a.append(hf(lib["gvs"][a:a+n, 0]))
print(f"root lin vel 2nd difference at 30 Hz, median over clips of per-clip median: reference {np.median(hf_r):.4f}, "
      f"agent {np.median(hf_a):.4f} m/s")

# per-joint single-frame separability of the local rotation (6D), in-sample split
params = amp_features_v1_params(SMPL_BODIES, SMPL_PARENTS)
ag, rf = [], []
for i, f in enumerate(lib["motion_files"]):
    stem = Path(str(f)).stem
    if stem.endswith(("_x3s", "_x7s")) or "Scorpion_pose_or_vrischikasana-b" in stem: continue
    a, n = int(lib["length_starts"][i]), int(lib["motion_num_frames"][i])
    ag.append(amp_frame_features_v1(lib["gts"][a:a+n], lib["grs"][a:a+n], lib["gvs"][a:a+n], lib["gavs"][a:a+n], torch.zeros(n), **params))
    t = (torch.arange(n, dtype=torch.float32) + 1.0) / 30.0
    st = ref.get_motion_state(torch.full((n,), stems.index(stem), dtype=torch.long), t)
    rf.append(amp_frame_features_v1(st.rigid_body_pos, st.rigid_body_rot, st.rigid_body_vel, st.rigid_body_ang_vel, torch.zeros(n), **params))
A, B = torch.cat(ag), torch.cat(rf)
X = torch.cat([A, B]); Y = torch.cat([torch.zeros(len(A), dtype=torch.long), torch.ones(len(B), dtype=torch.long)])
rng = np.random.default_rng(0); test = rng.random(len(Y)) < 0.2
res = {}
for k, j in enumerate(params["child_ids"]):
    cols = list(range(10 + 6 * k, 16 + 6 * k))
    net = train_mlp(X[~test][:, cols], Y[~test], 3, 0, hidden=(64,))
    res[SMPL_BODIES[j]] = auroc(predict(net, X[:, cols])[test], Y.numpy()[test])
print("single-frame AUROC from ONE joint's local rotation:")
print("  " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(res.items(), key=lambda kv: -kv[1])))
