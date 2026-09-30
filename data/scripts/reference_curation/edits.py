# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pass C (BUILD_PLAN Step 8): the reviewer's before/after check of the retarget's edits, and its calibration.

Every other claim the reviewer makes restates deterministic evidence (floats, floor contact, left and
right); Pass C asks what only a reader of the images can judge: is the edited reference the same pose,
does it look like a person, does it carry an artefact. The question is relative (A against B), over the
performer's own body, and blind to which side was edited.

Packets
-------
``REVIEW_ROOT/render_c1/<packet_id>/{img_1..5.png, packet.json}``: one moment of one hold, the two versions
side by side from the same camera (the left column is A, the right column B, each panel labelled), both
over the performer's translucent grey mesh at the same moment:

1. and 2. four eye-level views; 3. the high and the grazing view; 4. floor-level close-ups of the parts the
grey body rests on; 5. three moments (the main one and 0.5 s either side) as a time strip.

Which version is A is a hash of the item, recorded only in the private index
(``output/reference_curation/packets/render_c1/_index_edits.jsonl``). The drop lines are render_v3's;
the capture markers are left out (the mesh is the human). The camera is fitted to both avatars and the
mesh together, so neither is cropped.

Items
-----
``edit``      the shipped reference against the retarget at a hold's corrected exemplar: what the step
              is judged on; there is no truth, the verdict is the reviewer's.
``identity``  the shipped reference twice: the same pose, equally close to the human, no difference.
``swap``      another clip's exemplar pose (another pose family), moved onto the reference's pelvis and
              heading and grounded: not the same pose.
``lift``      the reference raised 7 cm: the unraised side lies closer to the human, and the raised one
              floats wherever the human rests on the floor.
``sink``      the reference lowered 6 cm: its supports go through the floor.
``kink``      one elbow or knee turned 80 deg the wrong way: an impossible joint.

The controls are built from shipped references only, so they stay valid while the retarget changes; their
truth is by construction, so no human labels are needed (BUILD_PLAN §1).

Contract
--------
``prompts/pass_c.md`` and ``schemas/pass_c.json`` (draft-7 keywords). ``validate`` checks the schema and
the panel tags; the runner is Step 4's CLI call (``review.call_claude``) and blindness check, with this
module's validator and the same ledger (``<packet_id>.C.<n>.json``).

Calibration and acceptance
--------------------------
Classes, each admitted as evidence under Step 4's rule (precision >= 0.9 over >= 30 claims, answered rate
>= 0.5 over >= 30 items):

``pose_change``   "not the same pose" detects a swap (an identity, a lift and a sink are the same pose; a
                  kink is not scored here: the prompt calls a limb in a clearly different place a different
                  pose, and the contract pilot's reviewer did exactly that);
``artefact``      a *new* major artefact (through-floor, through-body, impossible-joint or pulled-apart, named
                  on the modified side and not also on the other) on a sink or a kink; none on a lift (it
                  floats, which is not a major artefact) and no one-sided one on an identity. One-sided,
                  because that is what acceptance asks of an edit: the shipped references carry artefacts
                  of their own, and naming them on both sides is right;
``closer``        the unmodified side of a lift is closer to the human; an identity is ``equal``.

An edit is **accepted** when the reviewer calls it the same pose, names no major artefact on the retarget
that it does not also name on the shipped reference, and does not call the shipped reference closer to the
human. Acceptance is reported with the admission of the three classes beside it: a class that is not
admitted makes its part of the rule advisory.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.edits --build-controls 30
    ... --build-edits --retarget-dir <dir> [--stem ...]
    ... --review --max-cost 30 [--kind edit]
    ... --calibrate
"""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import math
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from reference_curation import render  # first: fixes the GL backend before anything imports mujoco
from extract_contact_configs import ZONE_ORDER
from reference_curation import human_mesh as hm, ids, packets, review, verdicts

MODULE = "reference_curation.edits"
SCHEMA_VERSION = 1
PASS = "C"
RENDER_V = "render_c1"
CONTRACT = "pass_c.v1"
INDEX_NAME = "_index_edits.jsonl"
KINDS = ("edit", "identity", "swap", "lift", "sink", "kink")
CONTROLS = KINDS[1:]
LIFT_M, SINK_M, KINK_DEG = 0.07, 0.06, 80.0
STRIP_S = 0.5
MAX_INSET_ROWS = 3
PARTS = tuple(render.ZONE_WORDS[z] for z in ZONE_ORDER)
ARTEFACT_KINDS = ("through_floor", "through_body", "impossible_joint", "pulled_apart", "jitter", "other")
MAJOR_KINDS = ("through_floor", "through_body", "impossible_joint", "pulled_apart")
CALIBRATION_DIR = verdicts.CALIBRATION_DIR / "pass_c"

LEGEND = {
    "what": "Renders of one moment of a motion, in two versions, A and B. A simulated avatar made of rigid "
            "capsules, spheres and boxes shows each version. The translucent grey body is the human performer "
            "at the same moment, the same in both columns.",
    "layout": "In every image the left column shows avatar A and the right column avatar B, each pair of panels "
              "from the same camera. Every panel carries its tag and the letter of its avatar in its corners.",
    "colours": "Every body segment of the avatar has its own colour (see palette). Left parts are the dark shade "
               "and right parts the light shade of their colour.",
    "drop_lines": "A thin vertical line in the colour of an avatar part runs from the part's lowest point straight "
                  "down to the floor and ends in a small cross. A part that touches the floor has no line.",
    "floor": "The floor has 10 cm checks.",
}
IMAGE_TEXT = {"eye": "Eye-level views, one camera per row.", "high_grazing": "Row 1: seen from high above. Row 2: "
              "seen from a camera just above the floor.", "insets": "Close-ups at floor level, one place per row.",
              "strip": "Three moments, 0.5 s apart, one per row; the middle row is the main moment."}


# --------------------------------------------------------------------------- #
# Items and the motions they compare
# --------------------------------------------------------------------------- #
def _h(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


def a_side(item_id: str) -> str:
    """Which version is A: ``"first"`` (the shipped reference, or the unmodified control) or ``"second"``."""
    return "first" if _h(item_id + "#side") % 2 == 0 else "second"


def modified_side(item: dict) -> str:
    """The avatar letter of the edited or modified version."""
    return "B" if item["a_is"] == "first" else "A"


def _clip(stem: str, pos: np.ndarray, rot: np.ndarray, fps: int = 60) -> render.Clip:
    return render.Clip(stem, fps, np.asarray(pos, float), np.asarray(rot, float), None, None, (), ())


def load_motion(stem: str, motion_dir) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = torch.load(ids.motion_path(stem, motion_dir), map_location="cpu", weights_only=False)
    return m["rigid_body_pos"].double().numpy(), m["rigid_body_rot"].double().numpy(), m["dof_pos"].double().numpy()


def _fk_rot(root_pos, root_quat, dof):
    """``(pos, quat xyzw)`` of every frame from the root and exp-map joint coordinates (the plant's FK)."""
    from protomotions.utils.rotations import matrix_to_quaternion, quaternion_to_matrix
    from reference_curation import retarget as rt

    rp = torch.as_tensor(root_pos)
    rr = quaternion_to_matrix(torch.as_tensor(root_quat), w_last=True)
    pos, rot = rt.fk(rt.skeleton(), rp, rr, torch.as_tensor(dof))
    return pos.numpy(), matrix_to_quaternion(rot, w_last=True).numpy()


def _yaw(q_xyzw: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    fwd = Rotation.from_quat(q_xyzw).apply([1.0, 0.0, 0.0])
    return math.atan2(fwd[1], fwd[0])


def control_motion(kind: str, stem: str, frame: int, other: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(pos, rot)`` of the modified version of a control (every frame of the shipped clip)."""
    from scipy.spatial.transform import Rotation

    pos, rot, dof = load_motion(stem, ids.SHIPPED_DIR)
    if kind == "identity":
        return pos, rot
    if kind in ("lift", "sink"):
        out = pos.copy()
        out[..., 2] += LIFT_M if kind == "lift" else -SINK_M
        return out, rot
    if kind == "kink":
        from reference_curation import retarget as rt

        sk = rt.skeleton()
        joint, axis, sign = [("L_Elbow", 2, 1.0), ("R_Elbow", 2, -1.0), ("L_Knee", 1, -1.0),
                             ("R_Knee", 1, -1.0)][_h(f"{stem}@{frame}#kink") % 4]
        j = 3 * (sk.names.index(joint) - 1) + axis
        bound = (sk.upper if sign > 0 else sk.lower)[j].item()
        d = dof.copy()
        d[:, j] = bound + sign * math.radians(KINK_DEG)
        return _fk_rot(pos[:, 0], rot[:, 0], d)
    if kind == "swap":
        o_stem, o_frame = other
        opos, orot, _ = load_motion(o_stem, ids.SHIPPED_DIR)
        p, r = opos[o_frame].copy(), orot[o_frame].copy()
        dyaw = _yaw(rot[frame, 0]) - _yaw(r[0])
        turn = Rotation.from_euler("z", dyaw)
        p = turn.apply(p - p[0]) + np.array([pos[frame, 0, 0], pos[frame, 0, 1], 0.0])
        r = (turn * Rotation.from_quat(r)).as_quat()
        from reference_curation import capture

        low = capture.body_min_z(p[None], r[None]).min()
        p[:, 2] += 0.005 - low
        T = pos.shape[0]
        return np.repeat(p[None], T, 0), np.repeat(r[None], T, 0)
    raise ValueError(kind)


def edit_items(labels: dict, retarget_dir: Path, stems=None) -> list[dict]:
    """One ``edit`` item per hold of ``stems`` (default all) at its corrected exemplar."""
    from reference_curation import retarget as rt

    out = []
    for stem, holds in labels["clips"]:
        if (stems and stem not in stems) or not rt.has_human(stem):
            continue
        if not ids.motion_path(stem, retarget_dir).exists():
            continue
        for h in holds:
            item_id = f"{h['hold_id']}#edit"
            out.append({"item_id": item_id, "kind": "edit", "hold_id": h["hold_id"], "stem": stem,
                        "frame": int(h["frame_hold"]), "retarget_dir": ids.display_path(retarget_dir),
                        "a_is": a_side(item_id), "truth": None})
    return out


def control_items(labels: dict, per_kind: dict) -> list[dict]:
    """``per_kind[kind]`` controls of each kind, on family holds first then the rest, spread over clips by a
    hash order; a swap takes a hold of another pose family (another clip) as its other pose."""
    from reference_curation import retarget as rt

    holds = [(s, h) for s, hs in labels["clips"] for h in hs if rt.has_human(s)]
    holds.sort(key=lambda sh: (not sh[1].get("extend"), _h(sh[1]["hold_id"])))
    fam = {s: s.split("_", 1)[1].rsplit("_-", 1)[0].rsplit("-", 1)[0] for s, _ in holds}
    out = []
    for kind in CONTROLS:
        n = per_kind.get(kind, 0)
        pool = sorted(holds, key=lambda sh: _h(sh[1]["hold_id"] + kind))
        pool.sort(key=lambda sh: not sh[1].get("extend"))
        for stem, h in pool[:n]:
            item = {"item_id": f"{h['hold_id']}#{kind}", "kind": kind, "hold_id": h["hold_id"], "stem": stem,
                    "frame": int(h["frame_hold"]), "a_is": a_side(f"{h['hold_id']}#{kind}")}
            if kind == "swap":
                others = [(s, o) for s, o in holds if fam[s] != fam[stem] and o.get("extend")]
                s2, o2 = others[_h(h["hold_id"] + "#other") % len(others)]
                item["other"] = [s2, int(o2["frame_hold"])]
            m = modified_side(item)
            item["truth"] = {"same_pose": "no" if kind == "swap" else "yes",
                             "closer": {"identity": "equal", "lift": "B" if m == "A" else "A"}.get(kind),
                             "artefact_side": m if kind in ("sink", "kink") else None,
                             "modified": None if kind == "identity" else m}
            out.append(item)
    return out


def item_clips(item: dict) -> tuple[render.Clip, render.Clip]:
    """``(A, B)`` clips of an item."""
    stem, frame = item["stem"], item["frame"]
    pos, rot, _ = load_motion(stem, ids.SHIPPED_DIR)
    first = _clip(stem, pos, rot)
    if item["kind"] == "edit":
        p2, r2, _ = load_motion(stem, ids.REPO / item["retarget_dir"])
    else:
        p2, r2 = control_motion(item["kind"], stem, frame, tuple(item["other"]) if item.get("other") else None)
    second = _clip(stem, p2, r2)
    return (first, second) if item["a_is"] == "first" else (second, first)


# --------------------------------------------------------------------------- #
# Paired sheets
# --------------------------------------------------------------------------- #
def strip_frames(item: dict, num_frames: int, fps: int = 60) -> list[int]:
    f, d = item["frame"], int(round(STRIP_S * fps))
    return [max(0, f - d), f, min(num_frames - 1, f + d)]


def plan_pair(a: render.Clip, b: render.Clip, item: dict, zones) -> list[tuple[str, list]]:
    """``[(kind, [panels])]``: each image's panels in row-major order, A then B in every row, each pair from
    one camera fitted to both avatars and the human mesh."""
    f = item["frame"]
    side = render.side_azimuth(a.pos[f], a.rot[f])
    pts = np.concatenate([render.framing_points(a, f), render.framing_points(b, f), item["_mesh_pts"]])
    size = render.panel_px(2)
    eye = [render.Panel("eye", f, render.fit_free(pts, side + 90 * k, render.EYE_ELEVATION_DEG, size, size))
           for k in range(4)]
    high = render.Panel("high", f, render.fit_free(pts, side + render.HIGH_AZIMUTH_OFFSET_DEG,
                                                   render.HIGH_ELEVATION_DEG, size, size))
    graz = render.Panel("grazing", f, render.fit_floor(pts, side, size, size))
    sheets = [("eye", [eye[0], eye[0], eye[1], eye[1]]), ("eye", [eye[2], eye[2], eye[3], eye[3]]),
              ("high_grazing", [high, high, graz, graz])]
    targets = [t for t in render.inset_targets(a, f, zones) if t[0]][:MAX_INSET_ROWS]
    if targets:
        ins = []
        centre = a.pos[f][:, :2].mean(0)
        for zs, xy in targets:
            inward = centre - xy
            az = side if np.linalg.norm(inward) < render.OUTSIDE_M else math.degrees(math.atan2(inward[1], inward[0]))
            p = render.Panel("inset", f, render.fit_inset(xy, az, size, size), zones=zs,
                             target=(float(xy[0]), float(xy[1]), 0.0))
            ins += [p, p]
        sheets.append(("insets", ins))
    frames = strip_frames(item, a.num_frames, a.fps)
    union = np.concatenate([render.framing_points(c, g) for c in (a, b) for g in frames] + [item["_mesh_pts"]])
    cam = render.fit_free(union, side, render.EYE_ELEVATION_DEG, size, size)
    sheets.append(("strip", [p for g in frames for p in (render.Panel("strip", g, cam),) * 2]))
    return sheets


def compose_pair(images: list[np.ndarray], number: int) -> tuple[Image.Image, list[dict]]:
    """Two columns (A left, B right); each panel tagged ``<number><letter>`` top-left and with its avatar
    letter top-right."""
    size = images[0].shape[0]
    rows = len(images) // 2
    W, H = 2 * size + render.GUTTER_PX, rows * size + (rows - 1) * render.GUTTER_PX
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    boxes = []
    for k, img in enumerate(images):
        col, row = k % 2, k // 2
        x, y = col * (size + render.GUTTER_PX), row * (size + render.GUTTER_PX)
        canvas.paste(Image.fromarray(img), (x, y))
        tag = f"{number}{chr(ord('a') + k)}"
        for text, anchor in ((tag, (x + 8, y + 6)), ("AB"[col], (x + size - 28, y + 6))):
            l, t, r, bb = draw.textbbox(anchor, text, font=render._font())
            draw.rectangle((l - 4, t - 3, r + 4, bb + 3), fill=(255, 255, 255), outline=(0, 0, 0))
            draw.text(anchor, text, fill=(0, 0, 0), font=render._font())
        boxes.append({"tag": tag, "avatar": "AB"[col], "box": [x, y, x + size, y + size]})
    return canvas, boxes


def render_item(item: dict, layer) -> list[tuple[Image.Image, list, str]]:
    a, b = item_clips(item)
    zones = [z for z in ZONE_ORDER if item.get("_human_touch", {}).get(z)]
    sheets = plan_pair(a, b, item, zones)
    out = []
    done = {}
    with render.Scene(a, layer, markers=False) as sa, render.Scene(b, layer, markers=False) as sb:
        for n, (kind, panels) in enumerate(sheets, 1):
            imgs = []
            for k, p in enumerate(panels):
                # A panel identical to one already drawn is copied, not drawn again: a clip's last frame
                # repeats in the time strip (f + 0.5 s is clipped to f), and drawing it a second time
                # left 30 bytes of stale memory in the image's first row (11 of 300 packets differed
                # between two builds of the same motions).
                key = (k % 2, repr(p))
                if key not in done:
                    done[key] = (sa if k % 2 == 0 else sb).render(p)
                imgs.append(done[key])
            image, boxes = compose_pair(imgs, n)
            out.append((image, boxes, kind))
    return out


def _mesh_points(layer, frames) -> np.ndarray:
    v = np.concatenate([layer.vertices(f) for f in frames])
    lo, hi = v.min(0), v.max(0)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _human_touch(stem: str, frame: int) -> dict:
    from reference_curation import sources

    st = sources.load(stem)["human_floor_state"][frame]
    return {z: bool(st[i] == 1) for i, z in enumerate(ZONE_ORDER)}


# --------------------------------------------------------------------------- #
# Packets and the private index
# --------------------------------------------------------------------------- #
def index_path(index_root: Path = packets.INDEX_ROOT) -> Path:
    return Path(index_root) / RENDER_V / INDEX_NAME


def read_index(index_root: Path = packets.INDEX_ROOT) -> dict:
    p = index_path(index_root)
    return {json.loads(l)["item_id"]: json.loads(l) for l in p.read_text().splitlines()} if p.exists() else {}


def build_packet(item: dict, *, review_root: Path = ids.REVIEW_ROOT, index: dict | None = None) -> dict:
    """Render one item into ``REVIEW_ROOT/render_c1/<packet_id>/`` and return its index entry."""
    stem = item["stem"]
    layer = hm.render_layer(stem)
    if layer is None:
        raise ValueError(f"{stem}: no human mesh")
    a, _ = item_clips(item)
    item = {**item, "_mesh_pts": _mesh_points(layer, strip_frames(item, a.num_frames, a.fps)),
            "_human_touch": _human_touch(stem, item["frame"])}
    sheets = render_item(item, layer)
    images, pngs = [], []
    for n, (image, boxes, kind) in enumerate(sheets, 1):
        buf = _png_bytes(image)
        pngs.append((f"img_{n}.png", buf))
        images.append({"file": f"img_{n}.png", "sha256": hashlib.sha256(buf).hexdigest(), "kind": kind,
                       "text": IMAGE_TEXT[kind], "panels": [b["tag"] for b in boxes],
                       "avatars": {b["tag"]: b["avatar"] for b in boxes}})
    content = {"schema_version": SCHEMA_VERSION, "pass": PASS, "contract": CONTRACT, "render_v": RENDER_V,
               "images": images, "legend": LEGEND, "palette": render.palette_legend()}
    pid = hashlib.sha1(json.dumps(content, sort_keys=True).encode()).hexdigest()
    d = Path(review_root) / RENDER_V / pid
    d.mkdir(parents=True, exist_ok=True)
    for name, buf in pngs:
        (d / name).write_bytes(buf)
    (d / "packet.json").write_text(json.dumps({**content, "packet_id": pid}, indent=1))
    return {"item_id": item["item_id"], "packet_id": pid, "pass": PASS, "render_v": RENDER_V, "dir": str(d),
            "kind": item["kind"], "hold_id": item["hold_id"], "stem": stem, "frame_hold": item["frame"],
            "a_is": item["a_is"], "truth": item["truth"], "other": item.get("other"),
            "retarget_dir": item.get("retarget_dir"), "images": {n: hashlib.sha256(b).hexdigest() for n, b in pngs},
            "item_key": ids.sha256_json({k: v for k, v in item.items() if not k.startswith("_")}),
            "gl": render.gl_info(), "audit_id": None, "stratum": item["kind"], "purpose": "pass_c"}


def _png_bytes(image: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _build_one(job: tuple) -> tuple[str, dict | None, str | None]:
    item, review_root = job
    try:
        return item["item_id"], build_packet(item, review_root=Path(review_root)), None
    except Exception as exc:  # noqa: BLE001 -- report every broken item, then fail
        return item["item_id"], None, f"{item['item_id']}: {type(exc).__name__}: {exc}"


def build(items: list[dict], index_root: Path = packets.INDEX_ROOT, review_root: Path = ids.REVIEW_ROOT,
          force: bool = False, workers: int = 1) -> tuple[dict, list[str]]:
    """Render the items that have no current packet (``workers`` processes; OSMesa is bit-exact across
    processes, so a packet id does not depend on who rendered it) and rewrite the index once, at the end."""
    import concurrent.futures
    import multiprocessing

    index = read_index(index_root)
    failures = []
    todo = []
    for item in items:
        key = ids.sha256_json({k: v for k, v in item.items() if not k.startswith("_")})
        old = index.get(item["item_id"])
        if old and not force and old["item_key"] == key and Path(old["dir"]).exists():
            continue
        todo.append((item, str(review_root)))
    if workers > 1 and len(todo) > 1:
        with hm._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            results = list(ex.map(_build_one, todo))
    else:
        results = [_build_one(j) for j in todo]
    for item_id, entry, error in results:
        if error:
            failures.append(error)
        else:
            index[item_id] = entry
    p = index_path(index_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(e) + "\n" for e in sorted(index.values(), key=lambda e: e["item_id"])))
    return index, failures


def fixed_texts() -> str:
    return json.dumps([LEGEND, IMAGE_TEXT, verdicts.prompt_path(PASS).read_text()]).lower()


def name_tokens() -> set:
    """Every name token of the corpus (the Step 3 leak rule): the fixed texts must avoid them all."""
    tokens = set()
    for clip in ids.load_manifest()["clips"]:
        for name in [clip["stem"], clip.get("family") or "", ids.recording_id(clip["stem"]) or ""] + \
                [h["name"] for h in clip["holds"]] + [h.get("orientation") or "" for h in clip["holds"]]:
            tokens |= {t.lower() for t in re.split(r"[_\-()\s]+", name) if len(t) >= 4 and re.search("[A-Za-z]", t)}
    return tokens - packets.GENERIC_TOKENS


# --------------------------------------------------------------------------- #
# Contract, runner and ledger
# --------------------------------------------------------------------------- #
def validate(answer, packet: dict) -> tuple[list[str], list[str]]:
    import jsonschema

    errors = [f"schema: /{'/'.join(map(str, e.absolute_path))}: {e.message}"
              for e in jsonschema.Draft202012Validator(verdicts.load_schema(PASS)).iter_errors(answer)]
    if errors or not isinstance(answer, dict):
        return errors or ["schema: not an object"], []
    tags = {t for im in packet["images"] for t in im["panels"]}
    cited = [t for a in answer["artefacts"] for t in a["panels"]]
    errors += [f"panel {t} is not in the packet" for t in sorted(set(cited) - tags)]
    return errors, []


def make_record(entry: dict, packet: dict, packet_dir: Path, reviewer: review.Reviewer, result: dict) -> dict:
    """Step 4's ledger record (``review.make_record``) with this pass's validator."""
    orig = verdicts.validate
    try:
        verdicts.validate = lambda answer, pkt, pass_="A": validate(answer, pkt)
        return review.make_record(entry, packet, packet_dir, reviewer, result)
    finally:
        verdicts.validate = orig


def run_review(entries: list[dict], reviewer: review.Reviewer, *, max_cost: float, call=review.call_claude,
               ledger_dir: Path = verdicts.LEDGER_DIR, review_root: Path = ids.REVIEW_ROOT, parallel: int = 3,
               log=functools.partial(print, flush=True)) -> dict:
    """Review the entries that have no valid verdict under ``reviewer``'s key, at most ``parallel`` at once,
    stopping when the spend so far plus the calls in flight could pass ``max_cost``."""
    ledger = verdicts.read_ledger(ledger_dir)
    done = {r["packet_id"] for r in ledger if r["pass"] == PASS and r["status"] == "valid"
            and r["reviewer"]["key"] == reviewer.key}
    failed = collections.Counter(r["packet_id"] for r in ledger if r["pass"] == PASS and r["status"] != "valid"
                                 and r["reviewer"]["key"] == reviewer.key)
    todo = [e for e in entries if e["packet_id"] not in done and failed[e["packet_id"]] < 2]
    out = {"pending": len(todo), "spent": 0.0, "status": collections.Counter(), "failures": []}
    costs = []

    def one(entry):
        d, packet = review.check_blind(entry, review_root)
        record = make_record(entry, packet, d, reviewer, call(reviewer, d))
        return record, verdicts.write_verdict(record, ledger_dir)

    with ThreadPoolExecutor(parallel) as pool:
        running = {}
        while todo or running:
            est = max(0.3, max(costs, default=0.0))
            while todo and len(running) < parallel and out["spent"] + (len(running) + 1) * est <= max_cost:
                e = todo.pop(0)
                running[pool.submit(one, e)] = e
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                e = running.pop(fut)
                try:
                    record, _ = fut.result()
                except Exception as exc:  # noqa: BLE001
                    out["failures"].append(f"{e['item_id']}: {exc}")
                    continue
                out["spent"] += record["cost_usd"] or 0.0
                costs.append(record["cost_usd"] or 0.0)
                out["status"][record["status"]] += 1
                log(f"{record['status']:7s} {e['kind']:8s} {e['item_id'][:60]} ${record['cost_usd'] or 0:.3f} "
                    f"{record['duration_s']:.0f} s (total ${out['spent']:.2f})")
    out["unstarted"] = len(todo)
    out["status"] = dict(out["status"])
    return out


# --------------------------------------------------------------------------- #
# Scoring, calibration and acceptance
# --------------------------------------------------------------------------- #
def score(answer: dict, entry: dict) -> list[dict]:
    """Rows ``{class, truth, claim}`` of one control verdict (``claim`` None = abstained)."""
    t, kind = entry["truth"], entry["kind"]
    rows = []
    sp = answer["same_pose"]["answer"]
    if kind != "kink":   # a limb turned the wrong way is "a limb in a clearly different place" to the prompt
        rows.append({"class": "pose_change", "truth": t["same_pose"] == "no",
                     "claim": None if sp == "cannot_tell" else sp == "no"})
    maj = _majors(answer)
    if kind in ("sink", "kink", "lift"):      # a new major artefact on the modified side: true of a sink or kink
        mod = t["modified"]
        other = "A" if mod == "B" else "B"
        rows.append({"class": "artefact", "truth": kind != "lift", "claim": bool(maj[mod] - maj[other])})
    elif kind == "identity":                  # the two sides are one reference: any one-sided artefact is false
        rows.append({"class": "artefact", "truth": False, "claim": bool(maj["A"] ^ maj["B"])})
    c = answer["closer_to_human"]
    if kind in ("lift", "identity"):
        rows.append({"class": "closer", "truth": t["closer"], "claim": None if c == "cannot_tell" else c})
    return rows


def class_metrics(rows: list[dict], name: str) -> dict:
    mine = [r for r in rows if r["class"] == name]
    answered = [r for r in mine if r["claim"] is not None]
    if name == "closer":
        correct = sum(r["claim"] == r["truth"] for r in answered)
        claims = len(answered)
        prec = correct / claims if claims else None
        recall = None
    else:
        pos = [r for r in answered if r["claim"]]
        claims = len(pos)
        prec = sum(r["truth"] for r in pos) / claims if claims else None
        truth_pos = [r for r in mine if r["truth"]]
        recall = sum(bool(r["claim"]) for r in truth_pos) / len(truth_pos) if truth_pos else None
    answered_rate = len(answered) / len(mine) if mine else None
    admitted = bool(prec is not None and prec >= verdicts.PRECISION_MIN and claims >= verdicts.CLAIMS_MIN
                    and answered_rate is not None and answered_rate >= verdicts.ANSWERED_MIN
                    and len(mine) >= verdicts.ITEMS_MIN)
    return {"items": len(mine), "answered_rate": _r(answered_rate), "claims": claims, "precision": _r(prec),
            "precision_low": _r(verdicts.wilson_low(round(prec * claims), claims)) if claims else None,
            "recall": _r(recall), "evidence": admitted}


def _r(x):
    return None if x is None else round(float(x), 4)


def _majors(answer: dict) -> dict:
    """``{avatar: {(kind, part)}}`` of the major artefacts the reviewer names."""
    maj = collections.defaultdict(set)
    for a in answer["artefacts"]:
        if a["severity"] == "major" and a["kind"] in MAJOR_KINDS:
            maj[a["avatar"]].add((a["kind"], a["part"]))
    return maj


def accept(answer: dict, entry: dict) -> dict:
    """The acceptance of one edit (module docstring)."""
    edited = modified_side(entry)
    shipped = "A" if edited == "B" else "B"
    maj = _majors(answer)
    new = maj[edited] - maj[shipped]
    closer = answer["closer_to_human"]
    ok_pose = answer["same_pose"]["answer"] == "yes"
    ok = ok_pose and not new and closer != shipped
    return {"accepted": bool(ok), "same_pose": answer["same_pose"]["answer"], "new_major_artefacts": sorted(map(list, new)),
            "closer": "edited" if closer == edited else "shipped" if closer == shipped else closer,
            "more_natural": "edited" if answer["more_natural"] == edited else
            "shipped" if answer["more_natural"] == shipped else answer["more_natural"]}


def verdict_table(index: dict, ledger_dir: Path = verdicts.LEDGER_DIR, key: str | None = None) -> dict:
    """The calibration of the controls and the acceptance of the edits, from the ledger (lowest valid ``n``
    per packet under one reviewer key)."""
    by_pid = {e["packet_id"]: e for e in index.values()}
    chosen = {}
    for r in verdicts.read_ledger(ledger_dir):
        if r["pass"] != PASS or r["status"] != "valid" or r["packet_id"] not in by_pid:
            continue
        if key and r["reviewer"]["key"] != key:
            continue
        chosen.setdefault(r["packet_id"], r)
    rows, edits = [], []
    for pid, r in chosen.items():
        e = by_pid[pid]
        if e["kind"] == "edit":
            edits.append({"item_id": e["item_id"], "hold_id": e["hold_id"], **accept(r["answer"], e)})
        else:
            rows += [{**x, "kind": e["kind"], "item_id": e["item_id"]} for x in score(r["answer"], e)]
    classes = {c: class_metrics(rows, c) for c in ("pose_change", "artefact", "closer")}
    by_kind = collections.defaultdict(collections.Counter)
    for x in rows:
        by_kind[x["kind"]][f"{x['class']}:{x['claim']}"] += 1
    return {"classes": classes, "by_kind": {k: dict(v) for k, v in by_kind.items()},
            "edits": {"reviewed": len(edits), "accepted": sum(x["accepted"] for x in edits),
                      "same_pose_no": sum(x["same_pose"] == "no" for x in edits),
                      "closer": dict(collections.Counter(x["closer"] for x in edits)),
                      "more_natural": dict(collections.Counter(x["more_natural"] for x in edits)),
                      "rejected": [x for x in edits if not x["accepted"]]},
            "verdicts": len(chosen)}


def main(argv: list[str] | None = None) -> int:
    from reference_curation import statics

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-controls", type=int, default=0, help="controls per kind (identity gets half)")
    ap.add_argument("--build-edits", action="store_true")
    ap.add_argument("--retarget-dir", type=Path)
    ap.add_argument("--stem", nargs="*")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--kind", nargs="*", default=list(KINDS))
    ap.add_argument("--max-cost", type=float, default=10.0)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="render processes for the builds")
    args = ap.parse_args(argv)
    start = time.time()
    labels = statics.load_labels(statics.default_labels_dir())
    failures = []
    bad = sorted(t for t in name_tokens() if t in fixed_texts())
    if bad:
        print(f"FAILED the fixed texts use corpus name tokens: {bad}", file=sys.stderr)
        return 1
    if args.build_controls:
        n = args.build_controls
        items = control_items(labels, {"identity": max(1, (2 * n) // 5), "swap": n + 2, "lift": n,
                                       "sink": (2 * n) // 3, "kink": (2 * n) // 3})
        _, f = build(items, workers=args.workers)
        failures += f
    if args.build_edits:
        if args.retarget_dir is None:
            print("FAILED --build-edits needs --retarget-dir", file=sys.stderr)
            return 1
        _, f = build(edit_items(labels, args.retarget_dir, args.stem), workers=args.workers)
        failures += f
    index = read_index()
    if args.review:
        reviewer = review.Reviewer(effort=args.effort, pass_=PASS)
        entries = sorted((e for e in index.values() if e["kind"] in args.kind and (not args.stem or e["stem"] in args.stem)),
                         key=lambda e: (KINDS.index(e["kind"]), e["item_id"]))
        res = run_review(entries, reviewer, max_cost=args.max_cost)
        failures += res["failures"]
        print(f"pass C review: {res['status']}, ${res['spent']:.2f}, {res['unstarted']} unstarted")
    if args.calibrate or args.review:
        table = verdict_table(index)
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        rec = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [verdicts.prompt_path(PASS), verdicts.schema_path(PASS)]),
               "render_v": RENDER_V, "contract": CONTRACT, "rule": verdicts.RULE, **table}
        (CALIBRATION_DIR / f"{RENDER_V}.json").write_text(json.dumps(rec, indent=1) + "\n")
        c, e = table["classes"], table["edits"]
        print("pass C calibration: " + "; ".join(f"{k} p={v['precision']} ({v['claims']} claims) r={v['recall']} "
                                                  f"{'E' if v['evidence'] else 'adv'}" for k, v in c.items())
              + f"; edits accepted {e['accepted']}/{e['reviewed']}")
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    print(f"edits: {len(index)} packets indexed in {time.time() - start:.0f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
