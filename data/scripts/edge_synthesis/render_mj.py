"""Contact sheets of an executed edge (CPU only: MuJoCo's OSMesa software renderer, never EGL -- EGL is the GPU).

``sheet(qpos, times)`` renders the plant at the given frames from a fixed side camera (perpendicular to the hand
line) and a three-quarter camera, with the floor, and tiles them with their times; ``ghost`` adds the destination
pose as a second, translucent body. For reading a run, not for review packets.

CLI: ``PYTHONPATH=.:data/scripts python -m edge_synthesis.render_mj output/edge_synthesis/runs/<run> [--fps 6]``
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from edge_synthesis import plant_mj as pm  # noqa: E402


def render_model() -> mujoco.MjModel:
    """The plant's model (same bodies, joints and hinge order) dressed for reading: a headlight, a checker floor, the
    left side blue and the right side red."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(pm.model_xml(sensors=False))
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "headlight", ambient="0.45 0.45 0.45", diffuse="0.7 0.7 0.7", specular="0.1 0.1 0.1")
    ET.SubElement(vis, "quality", shadowsize="0")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1="0.92 0.92 0.92",
                  rgb2="0.78 0.78 0.78", width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="10 10", reflectance="0")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.62 0.72 0.86",
                  rgb2="0.97 0.97 1.0", width="512", height="512")
    for body in root.iter("body"):
        name = body.get("name", "")
        rgba = "0.25 0.45 0.85 1" if name.startswith("L_") else "0.85 0.3 0.25 1" if name.startswith("R_") else \
            "0.75 0.72 0.6 1"
        for g in body.findall("geom"):
            g.set("rgba", rgba)
    for g in root.find("worldbody").findall("geom"):
        if g.get("name") == "floor":
            g.set("material", "grid")
            g.set("size", "4 4 0.05")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def render_frames(qpos: np.ndarray, azimuths=(0.0, 60.0), size=(320, 320), distance=2.4, lookat=None,
                  elevation=-15.0) -> list[list[np.ndarray]]:
    """``[frame][camera] -> HxWx3`` images of the plant at each ``qpos`` (hinge layout). Azimuth 0 looks along +x
    (the hand line of every edge's endpoints), i.e. from the body's side."""
    m = render_model()
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, size[0])
    m.vis.global_.offheight = max(m.vis.global_.offheight, size[1])
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = distance, elevation
    if lookat is not None:
        centre = np.asarray(lookat)
    else:                                   # frame the whole motion: the mean root in xy, mid-height of the bodies
        plant = pm.Plant(nthread=1, sensors=False)
        p, _ = plant.fk(qpos)
        centre = np.r_[qpos[:, :2].mean(0), 0.5 * (p[..., 2].max() + 0.0) * 0.9]
        distance = max(distance, 1.6 * float(p[..., 2].max()) + 0.6)
    out = []
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        row = []
        for az in azimuths:
            cam.azimuth = az
            cam.lookat[:] = centre
            r.update_scene(d, camera=cam)
            row.append(r.render().copy())
        out.append(row)
    r.close()
    return out


def sheet(qpos: np.ndarray, times: np.ndarray, path: Path, title: str = "", cols: int = 6, **kw) -> Path:
    frames = render_frames(qpos, **kw)
    ncam = len(frames[0])
    h, w = frames[0][0].shape[:2]
    rows = int(np.ceil(len(frames) / cols))
    img = Image.new("RGB", (cols * w, rows * ncam * h + 24), "white")
    dr = ImageDraw.Draw(img)
    dr.text((6, 4), title, fill="black")
    for i, (row, t) in enumerate(zip(frames, times)):
        c, rr = i % cols, i // cols
        for k, im in enumerate(row):
            y = 24 + (rr * ncam + k) * h
            img.paste(Image.fromarray(im), (c * w, y))
            if k == 0:
                ImageDraw.Draw(img).text((c * w + 4, y + 4), f"t={t:+.2f}s", fill="black")
    path = Path(path)
    img.save(path)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args(argv)
    z = np.load(args.run / "trajectory.npz")
    rec = json.load(open(args.run / "run.json"))
    qpos, dt, t0 = z["qpos"], float(z["dt"]), float(z["t0"])
    t = t0 + dt * (np.arange(len(qpos)) + 1)
    lo = t[0] if args.t0 is None else args.t0
    hi = t[-1] if args.t1 is None else args.t1
    pick = np.arange(len(t))[(t >= lo) & (t <= hi)][:: max(1, int(round(1.0 / (args.fps * dt))))]
    m = rec["metrics"]
    title = (f"{rec['edge']} {rec['label']}  T={rec['T']}s  final d6 {m['final_d6_m']} m  hand drift {m['hand_drift_cm']} cm"
             f"  landing {m['landing_peak_bw']}")
    out = sheet(qpos[pick], t[pick], args.run / "sheet.png", title, cols=args.cols)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def reference_sheet(ref_dir: Path, fps: float = 5.0, cols: int = 8) -> Path:
    """A contact sheet of a ``quasistatic`` reference (``reference.npz``: exp-map coordinates at 60 fps)."""
    z = np.load(Path(ref_dir) / "reference.npz")
    rec = json.load(open(Path(ref_dir) / "record.json"))
    plant = pm.Plant(nthread=1, sensors=False)
    from scipy.spatial.transform import Rotation

    quat = Rotation.from_matrix(z["root_rot"]).as_quat()
    qpos = np.empty((len(z["times"]), plant.nq))
    prev = None
    for i in range(len(qpos)):
        qpos[i] = plant.qpos_from_expmap(z["root_pos"][i], quat[i], z["dof"][i], prev=prev)
        prev = qpos[i]
    pick = np.arange(0, len(qpos), max(1, int(round(60 / fps))))
    c = rec["certification"]
    title = (f"{rec['edge']} {rec['label']} reference  T={rec['T']}s  statics s* max {c['statics']['s_star_max']}  "
             f"COM margin {c['com_margin_min_cm']} cm  new overlaps {c['new_overlaps_1mm']}")
    return sheet(qpos[pick], z["times"][pick], Path(ref_dir) / "sheet.png", title, cols=cols)
