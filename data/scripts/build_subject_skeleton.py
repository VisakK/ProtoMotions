# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rescale the ``smpl_boxhands_lowtorque`` MJCF into a *subject-specific* skeleton.

The stock boxhands MJCF is the SMPL neutral **mean shape** (betas=0). The MOYO
yoga clips were performed by subject ``yogi_03596`` (female), whose body shape is
recorded as 300/400 SMPL-X betas + a personalized ``v_template`` mesh. Measuring
her mesh (surface markers -> joint centers) against the boxhands skeleton shows
the mismatch is concentrated in the **arms**:

    upper arm  0.312 m  vs  0.261 m   (subject +19 %)
    forearm    0.253 m  vs  0.249 m   (subject +1.3 %)
    shin       0.391 m  vs  0.401 m   (subject -2.3 %)
    torso      ~match   |  legs ~match within surface-marker noise

and the humanoid mass (54.5 kg) is far below the subject's ~74 kg ground-reaction
weight. That short-arm error is exactly what breaks the arm-supported inversions
(handstand / scorpion / headstand / crow): with the same joint angles on shorter
arms at the recorded pelvis height, the hands land in the wrong place.

This script applies **per-segment length scales** (default = the measured ratios)
to the relevant body ``pos`` offsets *and* the collision-geom ``fromto`` of the
segment that spans them, then uniformly scales all geom densities so the total
mass matches ``--target-mass``. Everything else (joint names, ranges, actuator
force ranges, stiffness/damping/armature, gears) is left **untouched**, so the
parsed ``kinematic_info`` (dof names + limits) stays identical to the boxhands
asset and the ProtoMotions robot config remains compatible.

Only ``rigid_body_pos`` / bone geometry / mass change; joint DOFs are unchanged,
so re-running ``convert_yoga_frames_to_proto.py`` on the new MJCF is a
contact-preserving retarget (scale-invariant angles -> FK on the correct-size
skeleton reproduces the subject's true contact positions).

Usage::

    python data/scripts/build_subject_skeleton.py \
        --src data/assets/smpl/smpl_boxhands_lowtorque.xml \
        --out data/assets/smpl/smpl_yogi03596_lowtorque.xml \
        --target-mass 74.0
"""

import argparse
import re

import numpy as np


# (body_name -> factor) for the body's own ``pos`` (= segment length from parent)
POS_SCALE = {
    "L_Knee": "thigh", "R_Knee": "thigh",   # pelvis-side: hip->knee = thigh length
    "L_Ankle": "shin", "R_Ankle": "shin",   # knee->ankle = shin length
    "L_Elbow": "ua",   "R_Elbow": "ua",     # shoulder->elbow = upper-arm length
    "L_Wrist": "fa",   "R_Wrist": "fa",     # elbow->wrist = forearm length
}
# (body_name -> factor) for the capsule ``fromto`` (the segment geom lives in the
# PARENT body and spans toward the child)
FROMTO_SCALE = {
    "L_Hip": "thigh",      "R_Hip": "thigh",       # hip body holds the thigh capsule
    "L_Knee": "shin",      "R_Knee": "shin",       # knee body holds the shin capsule
    "L_Shoulder": "ua",    "R_Shoulder": "ua",     # shoulder body holds the upper-arm capsule
    "L_Elbow": "fa",       "R_Elbow": "fa",        # elbow body holds the forearm capsule
}


def _scale_numbers(s, f, ndigits=6):
    vals = [float(x) for x in s.split()]
    return " ".join(f"{v * f:.{ndigits}f}".rstrip("0").rstrip(".") if v != 0 else "0"
                    for v in vals)


def _body_block_span(text, body_name):
    """(start, end) covering ``<body name="X" ...>`` up to (but not including) its
    first *child* ``<body ...>`` -- i.e. the body's own tag + joints + own geoms."""
    m = re.search(r'<body\s+name="%s"' % re.escape(body_name), text)
    if not m:
        raise KeyError(f"body {body_name!r} not found")
    start = m.start()
    nxt = text.find("<body ", m.end())
    end = nxt if nxt != -1 else len(text)
    return start, end


def scale_pos(text, body_name, f):
    """Scale the ``pos`` on the opening tag of ``body_name`` by ``f``."""
    pat = re.compile(r'(<body\s+name="%s"[^>]*?\spos=")([^"]+)(")' % re.escape(body_name))
    def repl(m):
        return m.group(1) + _scale_numbers(m.group(2), f) + m.group(3)
    new, n = pat.subn(repl, text)
    assert n == 1, f"pos edit for {body_name}: matched {n} (expected 1)"
    return new


def scale_fromto(text, body_name, f):
    """Scale the first geom ``fromto`` inside ``body_name``'s own block by ``f``."""
    s, e = _body_block_span(text, body_name)
    block = text[s:e]
    pat = re.compile(r'(fromto=")([^"]+)(")')
    m = pat.search(block)
    assert m is not None, f"no fromto in body {body_name}"
    block2 = block[:m.start()] + m.group(1) + _scale_numbers(m.group(2), f) + m.group(3) + block[m.end():]
    return text[:s] + block2 + text[e:]


def scale_all_densities(text, f):
    return re.sub(r'density="([\d.]+)"',
                  lambda m: f'density="{float(m.group(1)) * f:.3f}"', text)


def total_mass(xml_path):
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml_path)
    return float(m.body_mass.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/assets/smpl/smpl_boxhands_lowtorque.xml")
    ap.add_argument("--out", default="data/assets/smpl/smpl_yogi03596_lowtorque.xml")
    ap.add_argument("--ua", type=float, default=1.194, help="upper-arm length scale")
    ap.add_argument("--fa", type=float, default=1.013, help="forearm length scale")
    ap.add_argument("--thigh", type=float, default=0.977, help="thigh length scale")
    ap.add_argument("--shin", type=float, default=0.977, help="shin length scale")
    ap.add_argument("--target-mass", type=float, default=74.0, help="total mass (kg)")
    ap.add_argument("--model-name", default="smpl_yogi03596_lowtorque")
    args = ap.parse_args()

    factors = {"ua": args.ua, "fa": args.fa, "thigh": args.thigh, "shin": args.shin}
    text = open(args.src).read()
    text = text.replace('<mujoco model="smpl_boxhands_lowtorque">',
                        f'<mujoco model="{args.model_name}">')

    # 1) geometry: scale segment bone offsets and their capsule geoms
    for bn, key in POS_SCALE.items():
        text = scale_pos(text, bn, factors[key])
    for bn, key in FROMTO_SCALE.items():
        text = scale_fromto(text, bn, factors[key])

    # write geometry-only version, measure mass, then scale densities to target
    tmp = args.out + ".geom.tmp"
    open(tmp, "w").write(text)
    m0 = total_mass(tmp)
    dfac = args.target_mass / m0
    text = scale_all_densities(text, dfac)
    open(args.out, "w").write(text)
    import os
    os.remove(tmp)

    # verify
    import mujoco
    m = mujoco.MjModel.from_xml_path(args.out)
    print(f"geometry-scaled mass {m0:.2f} kg -> density x{dfac:.4f} -> {m.body_mass.sum():.2f} kg "
          f"(target {args.target_mass})")
    print("\nsegment lengths after rescale (m):")
    for bn in ["L_Knee", "L_Ankle", "L_Elbow", "L_Wrist"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
        print(f"  {bn:8s} |pos|={np.linalg.norm(m.body_pos[bid]):.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
