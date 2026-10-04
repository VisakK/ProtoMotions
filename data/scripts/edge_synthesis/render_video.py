"""Videos of executed edges, CPU only: MuJoCo's OSMesa software renderer and libx264 (never EGL or NVENC, the GPU).

``video(run_dir, out)`` renders an ``edge_mppi`` run (``trajectory.npz``, ``run.json``) at the control rate:

* two fixed cameras side by side -- a side view (azimuth 0: the hand line points at the camera, the body moves across
  the image) and a three-quarter view -- framed on the bounding box of the whole motion;
* for a run that tracked a T2 reference (``--reference``), the reference as a translucent ghost
  (``mujoco.mjv_addGeoms`` of a second ``MjData`` into the same scene), so the tracking error is visible;
* a caption: the edge, the time since the departure, the phase of the edge's schedule, and the measured ground
  loads per zone in body weights (the plant's contact sensors);
* ``speed`` < 1 gives slow motion: every frame is the nearest recorded 240 Hz physics state, nothing is interpolated or resimulated.

``montage(clips, out)`` joins clips with title cards (``ffmpeg`` concat; every clip shares size and rate).

CLI::

    PYTHONPATH=.:data/scripts python -m edge_synthesis.render_video output/edge_synthesis/runs/<run> [--speed 0.5]
    PYTHONPATH=.:data/scripts python -m edge_synthesis.render_video --showcase      # the results set + a montage
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from extract_contact_configs import ZONE_ORDER  # noqa: E402
from edge_synthesis import gpu_guard  # noqa: E402
from edge_synthesis import plant_mj as pm  # noqa: E402
from edge_synthesis import render_mj  # noqa: E402
from edge_synthesis import sketch as SK  # noqa: E402
from reference_curation import ids  # noqa: E402

OUT_ROOT = ids.REPO / "output/edge_synthesis/videos"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
VIEW = (640, 480)
CAPTION_H = 84
GHOST_RGBA = (0.35, 0.85, 0.45, 0.28)
ZONE_SHORT = {"L_FOOT": "L foot", "R_FOOT": "R foot", "L_HAND": "L hand", "R_HAND": "R hand", "HEAD": "head",
              "L_SHANK": "L shin", "R_SHANK": "R shin", "L_THIGH": "L thigh", "R_THIGH": "R thigh",
              "PELVIS": "pelvis", "TRUNK": "trunk", "L_FOREARM": "L forearm", "R_FOREARM": "R forearm",
              "L_UPPER_ARM": "L upper arm", "R_UPPER_ARM": "R upper arm"}


def _font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def _framing(plant: pm.Plant, qpos: np.ndarray, fovy_deg: float = 45.0, aspect: float = VIEW[0] / VIEW[1]):
    """``(lookat, distance)``: a fixed camera that keeps the whole motion in view from any azimuth."""
    p, _ = plant.fk(qpos[:: max(1, len(qpos) // 200)])
    lo, hi = p.reshape(-1, 3).min(0), p.reshape(-1, 3).max(0)
    lo[2] = min(lo[2], 0.0)
    centre = 0.5 * (lo + hi)
    half_w = 0.5 * float(np.linalg.norm((hi - lo)[:2])) + 0.25
    half_h = 0.5 * float(hi[2] - lo[2]) + 0.25
    t = np.tan(np.radians(fovy_deg) / 2)
    distance = max(half_h / t, half_w / (t * aspect)) * 1.15 + 0.4
    return centre, distance


def video(run_dir: Path, out: Path, speed: float = 1.0, fps: int = 30, views=((0.0, -10.0), (55.0, -18.0)),
          ghost: bool | None = None, t0: float | None = None, t1: float | None = None, title: str | None = None,
          note: str = "") -> Path:
    run_dir, out = Path(run_dir), Path(out)
    z = np.load(run_dir / "trajectory.npz")
    rec = json.load(open(run_dir / "run.json"))
    plant = pm.Plant(nthread=1)
    physx = rec.get("backend") == "physx"
    if physx:
        # a PhysX run (edge_mppi_physx): its states drawn through MuJoCo's kinematics (rendering only), its loads
        # its own terrain forces
        qpos = physx_qpos(plant, z)
        loads_all = np.einsum("zb,tb->tz", z["zone_matrix"], z["ground"][..., 2]) / float(z["weight_n"])
    else:
        qpos, sens = z["qpos"], z["sens"]
    dt, start = float(z["dt"]), float(z["t0"])
    t_phys = start + dt * (np.arange(len(qpos)) + 1)
    e = SK.edge(SK.load_edges(), rec["edge"])
    sched = SK.schedule(e, rec["durations_s"])
    t0 = t_phys[0] if t0 is None else t0
    t1 = t_phys[-1] if t1 is None else t1
    times = np.arange(t0, t1 + 1e-9, speed / fps)
    idx = np.clip(np.round((times - start) / dt).astype(int) - 1, 0, len(qpos) - 1)
    loads = loads_all if physx else plant.zone_forces(sens)[..., 2] / plant.weight_n   # [steps, 15] in BW
    ref = None
    ref_dir = rec.get("reference")
    if ref_dir is None and "_ref" in run_dir.name:       # runs made before run.json recorded its reference
        ref_dir = f"output/edge_synthesis/quasistatic/{rec['edge']}_{rec['timing'] if isinstance(rec['timing'], str) else 'mid'}"
    if (ghost is None or ghost) and ref_dir:
        ref = SK.RefSketch(plant, ids.REPO / ref_dir, sched.T)
    m = render_mj.render_model()
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, VIEW[0]), \
        max(m.vis.global_.offheight, VIEW[1])
    d, dg = mujoco.MjData(m), mujoco.MjData(m)
    r = mujoco.Renderer(m, height=VIEW[1], width=VIEW[0])
    vopt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    lookat, distance = _framing(plant, np.concatenate([qpos[idx]] + ([ref.qpos(times)] if ref is not None else [])))
    cams = []
    for az, el in views:
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.azimuth, c.elevation, c.distance = az, el, distance
        c.lookat[:] = lookat
        cams.append(c)
    W, H = VIEW[0] * len(views), VIEW[1] + CAPTION_H
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    f_title, f_text, f_small = _font(19, bold=True), _font(16), _font(14)
    m_ = rec["metrics"]
    head = title or f"{rec['edge']}  {rec['label']}"
    sub = (("PhysX (IsaacLab, the training plant), MPPI" if physx else "MuJoCo plant v2 (CPU), MPPI")
           + f"  |  final 6-body error {m_['final_d6_m']:.3f} m"
           + (f"  |  tracking {m_['sketch_body_err_mean_cm']:.1f} cm" if ref is not None and
              m_.get("sketch_body_err_mean_cm") is not None else "")
           + (f"  |  {note}" if note else "")
           + (f"  |  {speed:g}x speed" if speed != 1.0 else ""))
    for k, t in enumerate(times):
        d.qpos[:] = qpos[idx[k]]
        mujoco.mj_kinematics(m, d)
        if ref is not None:
            dg.qpos[:] = ref.qpos(np.array([t]))[0]
            mujoco.mj_kinematics(m, dg)
        tiles = []
        for c in cams:
            r.update_scene(d, camera=c)
            if ref is not None:
                n0 = r.scene.ngeom
                mujoco.mjv_addGeoms(m, dg, vopt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
                for i in range(n0, r.scene.ngeom):
                    r.scene.geoms[i].rgba[:] = GHOST_RGBA
            tiles.append(r.render())
        img = Image.new("RGB", (W, H), (250, 250, 250))
        for i, tile in enumerate(tiles):
            img.paste(Image.fromarray(tile), (i * VIEW[0], CAPTION_H))
        dr = ImageDraw.Draw(img)
        p = int(sched.phase_at(np.array([t]))[0])
        phase = (f"holding {e['source']['name'].split('_or_')[0].replace('_', ' ')}" if p < 0 else
                 f"holding {e['destination']['name'].split('_or_')[0].replace('_', ' ')}" if p >= len(sched.phase_names)
                 else f"phase: {sched.phase_names[p].replace('_', ' ')}")
        ld = loads[idx[k]]
        load_txt = "  ".join(f"{ZONE_SHORT[zn]} {ld[i]:.2f}" for i, zn in enumerate(ZONE_ORDER) if ld[i] > 0.03)
        dr.text((10, 6), head, fill=(20, 20, 20), font=f_title)
        right = f"t = {t:+.2f} s   {phase}"
        dr.text((W - 12 - dr.textlength(right, font=f_text), 8), right, fill=(20, 20, 20), font=f_text)
        dr.text((10, 34), sub, fill=(70, 70, 70), font=f_small)
        dr.text((10, 58), f"ground load (body weights): {load_txt or 'none'}", fill=(40, 40, 120), font=f_small)
        if ref is not None:
            dr.text((8, CAPTION_H + 6), "ghost: statics-certified reference (T2)", fill=(30, 110, 50), font=f_small)
        enc.stdin.write(np.asarray(img, np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    r.close()
    return out


def physx_qpos(plant: pm.Plant, z) -> np.ndarray:
    """A PhysX run's states (root, exp-map dof) as MuJoCo hinge ``qpos``, branch-continuous (for drawing)."""
    out = np.empty((len(z["dof"]), plant.nq))
    prev = None
    for i in range(len(out)):
        out[i] = plant.qpos_from_expmap(z["pos"][i, 0], z["rot"][i, 0], z["dof"][i], prev=prev)
        prev = out[i]
    return out


def title_card(text: list[str], out: Path, seconds: float = 2.0, fps: int = 30, size=(VIEW[0] * 2, VIEW[1] + CAPTION_H)):
    img = Image.new("RGB", size, (245, 245, 245))
    dr = ImageDraw.Draw(img)
    y = size[1] // 2 - 22 * len(text)
    for i, line in enumerate(text):
        f = _font(30 if i == 0 else 20, bold=i == 0)
        w = dr.textlength(line, font=f)
        dr.text(((size[0] - w) / 2, y), line, fill=(20, 20, 20) if i == 0 else (70, 70, 70), font=f)
        y += 50 if i == 0 else 32
    frame = np.asarray(img, np.uint8).tobytes()
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                            f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                            "-crf", "20", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for _ in range(int(seconds * fps)):
        enc.stdin.write(frame)
    enc.stdin.close()
    enc.wait()
    return out


def montage(parts: list[Path], out: Path) -> Path:
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(out)], check=True)
    lst.unlink()
    return out


# The results set: the best run per edge, two slow-motion dynamic edges, and the failures worth seeing.
RUNS = "output/edge_synthesis/runs"
SHOWCASE = [
    ("E1_press_slow_seed2", f"{RUNS}/E1_E1_high_s2_refff", 1.0, "admitted (provisional)"),
    ("E1_press_mid", f"{RUNS}/E1_E1_mid_s0_refff", 1.0, "fails naturalness only"),
    ("E3_handstand_to_crow_slow", f"{RUNS}/E3_E3_high_s0_refff", 1.0, ""),
    ("E4_tripod_to_crow_slow", f"{RUNS}/E4_E4_high_s0_refff", 1.0, ""),
    ("E2_crow_to_chaturanga_jumpback", f"{RUNS}/E2_mid_s0_land2", 1.0, ""),
    ("E2_crow_to_chaturanga_jumpback_slowmo", f"{RUNS}/E2_mid_s0_land2", 0.4, ""),
    ("E5_handstand_to_chaturanga_via_plank", f"{RUNS}/E5_mid_s1_v3", 1.0, ""),
    ("E5_handstand_to_chaturanga_via_plank_slowmo", f"{RUNS}/E5_mid_s1_v3", 0.4, ""),
    ("B1_crow_to_plank_jumpback", f"{RUNS}/B1_mid_s1_v4", 1.0, ""),
]
FAILURES = [
    ("fail_E1_press_slow_seed0_falls_in_hold", f"{RUNS}/E1_E1_high_s0_refff", 1.0, "falls in the final hold"),
    ("fail_E5_first_version_face_plant", f"{RUNS}/E5_mid_s0_land2", 1.0, "first version: the head hits the floor"),
    ("fail_B2_firefly_to_crow_not_held", f"{RUNS}/B2_B2_mid_s0_refff", 1.0, "crow not held (chest saturates)"),
    ("fail_E1_press_without_feedforward", f"{RUNS}/E1_mid_s0_ref", 1.0, "no gravity feed-forward: topples"),
]


# The PhysX regeneration (edge_mppi_physx): the best seed per edge and recipe, and the failures, same layout.
PX_RUNS = "output/edge_synthesis/physx/runs"
PX_SHOWCASE = [
    ("px_E1_press_slow", f"{PX_RUNS}/E1_E1_high_s3_px", 1.0, "D1 default: admitted"),
    ("px_E1_press_mid", f"{PX_RUNS}/E1_E1_mid_s0_px", 1.0, "D1 default: admitted"),
    ("px_E3_handstand_to_crow_slow", f"{PX_RUNS}/E3_E3_high_s0_px12", 1.0, "D1 default: admitted"),
    ("px_B1_crow_to_plank_high", f"{PX_RUNS}/B1_high_s2_px", 1.0, "D1 default: admitted"),
    ("px_B1_crow_to_plank_mid_slowmo", f"{PX_RUNS}/B1_mid_s0_px", 0.4, "D1 default: admitted"),
]
PX_FAILURES = [
    ("px_fail_E2_jumpback_low_chaturanga", f"{PX_RUNS}/E2_mid_s3_px12", 1.0, "lands, but in a low chaturanga (endpoints)"),
    ("px_fail_E2_jumpback_slowmo", f"{PX_RUNS}/E2_mid_s3_px12", 0.4, "the right elbow at its limit on ~100 % of steps"),
    ("px_fail_E5_float_down", f"{PX_RUNS}/E5_mid_s0_px12", 1.0, "lands via the plank, ends low (endpoints)"),
    ("px_fail_E4_tripod_to_crow", f"{PX_RUNS}/E4_E4_mid_s0_px12", 1.0, "statics, as on MuJoCo"),
    ("px_fail_B2_firefly_to_crow", f"{PX_RUNS}/B2_B2_mid_s0_px", 1.0, "crow not reached, as on MuJoCo"),
]


def showcase_physx(out_root: Path = ids.REPO / "output/edge_synthesis/physx/videos") -> dict:
    made = {}
    out_root.mkdir(parents=True, exist_ok=True)
    for name, run, speed, note in PX_SHOWCASE + PX_FAILURES:
        out = out_root / f"{name}.mp4"
        video(ids.REPO / run, out, speed=speed, note=note)
        made[name] = out
        print("wrote", ids.display_path(out), flush=True)
    cards = out_root / "_cards"
    cards.mkdir(parents=True, exist_ok=True)
    parts = [title_card(["Synthesised yoga transitions, regenerated in PhysX", "IsaacLab (the training plant), "
                         "sampling MPC (MPPI) on the GPU", "graph growth, lane T in PhysX (2026-10-04)"],
                        cards / "intro.mp4", 3.0)]
    for name, run, speed, note in PX_SHOWCASE:
        parts.append(title_card([name.replace("px_", "").replace("_", " "), note], cards / f"{name}.mp4", 1.5))
        parts.append(made[name])
    parts.append(title_card(["Failures kept on record", ""], cards / "failures.mp4", 1.5))
    for name, run, speed, note in PX_FAILURES:
        parts.append(title_card([name.replace("px_fail_", "").replace("_", " "), note], cards / f"{name}.mp4", 1.5))
        parts.append(made[name])
    made["montage"] = montage(parts, out_root / "physx_results_montage.mp4")
    print("wrote", ids.display_path(made["montage"]))
    return made


def showcase(out_root: Path = OUT_ROOT) -> dict:
    made = {}
    for name, run, speed, note in SHOWCASE + FAILURES:
        out = out_root / f"{name}.mp4"
        video(ids.REPO / run, out, speed=speed, note=note)
        made[name] = out
        print("wrote", ids.display_path(out), flush=True)
    cards = out_root / "_cards"
    cards.mkdir(parents=True, exist_ok=True)
    parts = [title_card(["Synthesised yoga transitions on plant v2", "MuJoCo, sampling MPC (MPPI), CPU only",
                         "graph growth, lane T (2026-10-03)"], cards / "intro.mp4", 3.0)]
    for name, run, speed, note in SHOWCASE:
        label = name.replace("_", " ")
        parts.append(title_card([label, note or ""], cards / f"{name}.mp4", 1.5))
        parts.append(made[name])
    parts.append(title_card(["Failures kept on record", ""], cards / "failures.mp4", 1.5))
    for name, run, speed, note in FAILURES:
        parts.append(title_card([name.replace("fail_", "").replace("_", " "), note], cards / f"{name}.mp4", 1.5))
        parts.append(made[name])
    made["montage"] = montage(parts, out_root / "lane_t_results_montage.mp4")
    print("wrote", ids.display_path(made["montage"]))
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--showcase", action="store_true")
    ap.add_argument("--showcase-physx", action="store_true", help="the PhysX regeneration's results set + a montage")
    ap.add_argument("--threads", type=int, default=4, help="llvmpipe render threads")
    args = ap.parse_args(argv)
    os.environ.setdefault("LP_NUM_THREADS", str(args.threads))
    print("cpu policy", gpu_guard.be_polite(cpus=tuple(range(16, 24))), flush=True)
    if args.showcase:
        showcase()
        return 0
    if args.showcase_physx:
        showcase_physx()
        return 0
    out = args.out or OUT_ROOT / f"{args.run.name}{'' if args.speed == 1 else f'_x{args.speed:g}'}.mp4"
    print("wrote", video(args.run, out, speed=args.speed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
