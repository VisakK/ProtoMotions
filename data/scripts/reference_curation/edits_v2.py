# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pass C on plant v2 (BodyFix Step 4, item 7): the reviewer's blind before/after check of the Step 3 edits, on
render_c2.

Step 8's Pass C (``edits.py``) compared the shipped reference with Step 8's retarget. Plant v2's references edit her
own fit, so the comparison is **her unedited fit on plant v2** (``fit_writer``, the "before") against **the Step 3
retarget** (``retarget_v2``, the "after"), both over her translucent grey mesh. The contract (``prompts/pass_c.md``,
``schemas/pass_c.json``), the A/B hash, the controls and their truth, the scoring, the admission rule and the
acceptance rule are ``edits.py``'s, called unchanged. What changes is the render (render_c2):

* plant v2's avatar in render_v4's segment hues (``render_v4``), inside ``render_v4.plant_context()``;
* **drop lines only under the zones the grey body rests on** at that moment (TODO D2): render_c1 drew a line under
  any part 1-15 cm up, so an edit that lowered a limb the human holds up read as floating (Peacock -a@788's
  rejection). The legend says so.

Controls are built from the before (the fit) on plant v2: ``identity`` (the fit twice), ``swap`` (another pose
family's exemplar on the fit's pelvis and heading, grounded), ``lift`` (+7 cm), ``sink`` (-6 cm), ``kink`` (an elbow
or knee 80 deg past plant v2's box). Edits: every non-transition hold of labels v2 at its exemplar.

An edit is **accepted** by ``edits.accept``: the same pose, no new major artefact, and the before not closer to the
human. Where Step 3 had to move the reference away from her because the plant cannot do what she does (the soft
supports, poses past the box, palm loading) "the fit is closer" is expected and physically required: the gate
makes it a flag, not an exclusion, unless the reviewer calls a different pose (BodyFix Step 4's card).

Packets: ``REVIEW_ROOT/render_c2/<packet_id>/``; index ``output/reference_curation/packets/render_c2/_index_edits.jsonl``
(``edits.read_index``'s layout); calibration ``data/reference_curation/calibration/pass_c/render_c2.json``.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.edits_v2 --build-controls 30
    ... --build-edits
    ... --review --kind identity swap lift sink kink --max-cost 16
    ... --review --kind edit --max-cost 40
    ... --calibrate
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from reference_curation import render_v4  # first: render (OSMesa)
from reference_curation import edits as E
from reference_curation import capture_v4, fit_writer as fw, human_mesh as hm, ids, packets, render, review, verdicts
from extract_contact_configs import ZONE_ORDER

MODULE = "reference_curation.edits_v2"
SCHEMA_VERSION = 1
PASS = E.PASS
RENDER_V = "render_c2"
CONTRACT = E.CONTRACT
FIT_ID = "fit_v2.v1.b8f5e86fc9"
CALIBRATION_DIR = E.CALIBRATION_DIR
LEGEND = {**E.LEGEND,
          "colours": "Every body segment of the avatar has its own colour (see palette): the upper arm, forearm and hand "
                     "of an arm have three different colours, and so do the thigh, shin and foot of a leg. Left parts are "
                     "the dark shade and right parts the light shade of their colour.",
          "drop_lines": "Under the parts on which the grey body rests on the floor, a thin vertical line in the colour of "
                        "an avatar part runs from that avatar part's lowest point straight down to the floor and ends in a "
                        "small cross. An avatar part that touches the floor has no line. Parts the grey body keeps off the "
                        "floor never have a line."}


def fit_dir() -> Path:
    return fw.OUT_ROOT / FIT_ID


def ref_dir() -> Path:
    return capture_v4.motion_path("x").parent


def index_path(index_root: Path = packets.INDEX_ROOT) -> Path:
    return Path(index_root) / RENDER_V / E.INDEX_NAME


def read_index(index_root: Path = packets.INDEX_ROOT) -> dict:
    p = index_path(index_root)
    return {json.loads(l)["item_id"]: json.loads(l) for l in p.read_text().splitlines()} if p.exists() else {}


# --------------------------------------------------------------------------- #
# Motions: the fit (before), the retarget (after) and the controls on plant v2
# --------------------------------------------------------------------------- #
def control_motion(kind: str, stem: str, frame: int, other: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``edits.control_motion`` built from the fit on plant v2 (call inside ``render_v4.plant_context()``)."""
    from scipy.spatial.transform import Rotation

    from reference_curation import capture, retarget as rt

    pos, rot, dof = E.load_motion(stem, fit_dir())
    if kind == "identity":
        return pos, rot
    if kind in ("lift", "sink"):
        out = pos.copy()
        out[..., 2] += E.LIFT_M if kind == "lift" else -E.SINK_M
        return out, rot
    if kind == "kink":
        sk = rt.skeleton()
        joint, axis, sign = [("L_Elbow", 2, 1.0), ("R_Elbow", 2, -1.0), ("L_Knee", 1, -1.0),
                             ("R_Knee", 1, -1.0)][E._h(f"{stem}@{frame}#kink") % 4]
        j = 3 * (sk.names.index(joint) - 1) + axis
        bound = (sk.upper if sign > 0 else sk.lower)[j].item()
        d = dof.copy()
        d[:, j] = bound + sign * math.radians(E.KINK_DEG)
        return E._fk_rot(pos[:, 0], rot[:, 0], d)
    if kind == "swap":
        o_stem, o_frame = other
        opos, orot, _ = E.load_motion(o_stem, fit_dir())
        p, r = opos[o_frame].copy(), orot[o_frame].copy()
        turn = Rotation.from_euler("z", E._yaw(rot[frame, 0]) - E._yaw(r[0]))
        p = turn.apply(p - p[0]) + np.array([pos[frame, 0, 0], pos[frame, 0, 1], 0.0])
        r = (turn * Rotation.from_quat(r)).as_quat()
        p[:, 2] += 0.005 - capture.body_min_z(p[None], r[None]).min()
        T = pos.shape[0]
        return np.repeat(p[None], T, 0), np.repeat(r[None], T, 0)
    raise ValueError(kind)


def item_clips(item: dict) -> tuple[render.Clip, render.Clip]:
    """``(A, B)``: the fit is ``first``; the retarget (an edit) or the control is ``second``."""
    stem, frame = item["stem"], item["frame"]
    pos, rot, _ = E.load_motion(stem, fit_dir())
    first = E._clip(stem, pos, rot)
    if item["kind"] == "edit":
        p2, r2, _ = E.load_motion(stem, ref_dir())
    else:
        p2, r2 = control_motion(item["kind"], stem, frame, tuple(item["other"]) if item.get("other") else None)
    second = E._clip(stem, p2, r2)
    return (first, second) if item["a_is"] == "first" else (second, first)


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #
def labels_clips(labels_dir: Path) -> list[tuple[str, list[dict]]]:
    m = ids.load_manifest(Path(labels_dir) / "holds.yaml")
    return [(c["stem"], c["holds"]) for c in m["clips"]]


def edit_items(clips: list, stems=None) -> list[dict]:
    out = []
    for stem, holds in clips:
        if stems and stem not in stems:
            continue
        for h in holds:
            if h["labels"]["status"] == "transition":
                continue
            item_id = f"{h['hold_id']}#edit"
            out.append({"item_id": item_id, "kind": "edit", "hold_id": h["hold_id"], "stem": stem,
                        "frame": int(h["frame_hold"]), "retarget_dir": ids.display_path(ref_dir()),
                        "before_dir": ids.display_path(fit_dir()), "a_is": E.a_side(item_id), "truth": None})
    return out


def control_items(clips: list, per_kind: dict) -> list[dict]:
    """``edits.control_items``'s choice (family holds first, a hash order; a swap takes another family's exemplar)
    over the corpus's holds, the fit as the before."""
    labels = {"clips": [(s, [h for h in hs if h["labels"]["status"] != "transition"]) for s, hs in clips]}
    import reference_curation.retarget as rt

    saved = rt.has_human
    rt.has_human = lambda stem: True       # every clip of the corpus has her mesh (fit_writer.corpus)
    try:
        return E.control_items(labels, per_kind)
    finally:
        rt.has_human = saved


# --------------------------------------------------------------------------- #
# Rendering and packets
# --------------------------------------------------------------------------- #
def render_item(item: dict, layer, touch: np.ndarray) -> list:
    """``edits.render_item`` with render_v4's Scene and D2's drop lines (only under zones the grey body rests on:
    ``touch [T, Z]`` from capture store v3's ``human_floor_state``)."""
    a, b = item_clips(item)
    zones = [z for z in ZONE_ORDER if touch[item["frame"], ZONE_ORDER.index(z)]]
    sheets = E.plan_pair(a, b, item, zones)
    drop = lambda f: {z for zi, z in enumerate(ZONE_ORDER) if touch[f, zi]}  # noqa: E731
    out, done = [], {}
    with render_v4.Scene(a, layer, markers=False, drop_zones=drop) as sa, \
            render_v4.Scene(b, layer, markers=False, drop_zones=drop) as sb:
        for n, (kind, panels) in enumerate(sheets, 1):
            imgs = []
            for k, p in enumerate(panels):
                key = (k % 2, repr(p))
                if key not in done:
                    done[key] = (sa if k % 2 == 0 else sb).render(p)
                imgs.append(done[key])
            image, boxes = E.compose_pair(imgs, n)
            out.append((image, boxes, kind))
    return out


def build_packet(item: dict, touch: np.ndarray, review_root: Path = ids.REVIEW_ROOT) -> dict:
    """Render one item into ``REVIEW_ROOT/render_c2/<packet_id>/`` (inside ``render_v4.plant_context()``)."""
    stem = item["stem"]
    layer = hm.render_layer(stem)
    if layer is None:
        raise ValueError(f"{stem}: no human mesh")
    a, _ = item_clips(item)
    item = {**item, "_mesh_pts": E._mesh_points(layer, E.strip_frames(item, a.num_frames, a.fps))}
    sheets = render_item(item, layer, touch)
    images, pngs = [], []
    for n, (image, boxes, kind) in enumerate(sheets, 1):
        buf = E._png_bytes(image)
        pngs.append((f"img_{n}.png", buf))
        images.append({"file": f"img_{n}.png", "sha256": hashlib.sha256(buf).hexdigest(), "kind": kind,
                       "text": E.IMAGE_TEXT[kind], "panels": [b["tag"] for b in boxes],
                       "avatars": {b["tag"]: b["avatar"] for b in boxes}})
    content = {"schema_version": SCHEMA_VERSION, "pass": PASS, "contract": CONTRACT, "render_v": RENDER_V,
               "images": images, "legend": LEGEND, "palette": render_v4.palette_legend()}
    pid = hashlib.sha1(json.dumps(content, sort_keys=True).encode()).hexdigest()
    d = Path(review_root) / RENDER_V / pid
    d.mkdir(parents=True, exist_ok=True)
    for name, buf in pngs:
        (d / name).write_bytes(buf)
    (d / "packet.json").write_text(json.dumps({**content, "packet_id": pid}, indent=1))
    motions = {"before": ids.sha256_file(ids.motion_path(stem, fit_dir())),
               "after": ids.sha256_file(ids.motion_path(stem, ref_dir()))}
    return {"item_id": item["item_id"], "packet_id": pid, "pass": PASS, "render_v": RENDER_V, "dir": str(d),
            "kind": item["kind"], "hold_id": item["hold_id"], "stem": stem, "frame_hold": item["frame"],
            "a_is": item["a_is"], "truth": item["truth"], "other": item.get("other"),
            "retarget_dir": item.get("retarget_dir"), "before_dir": ids.display_path(fit_dir()), "motions": motions,
            "images": {n: hashlib.sha256(b).hexdigest() for n, b in pngs},
            "item_key": ids.sha256_json({**{k: v for k, v in item.items() if not k.startswith("_")}, "motions": motions,
                                         "code": [ids.sha256_file(f) for f in (__file__, render_v4.__file__)]}),
            "gl": render.gl_info(), "audit_id": None, "stratum": item["kind"], "purpose": "pass_c2"}


def _touch(stem: str) -> np.ndarray:
    from reference_curation import sources

    return np.asarray(sources.load(stem)["human_floor_state"]) == 1


def _build_job(job: tuple) -> list[tuple[str, dict | None, str | None]]:
    """One clip's items in a spawned single-threaded worker (render_v4 is OSMesa: bit-exact across processes)."""
    items, touch, review_root = job
    torch.set_num_threads(1)
    out = []
    with render_v4.plant_context():
        for item in items:
            try:
                out.append((item["item_id"], build_packet(item, touch, Path(review_root)), None))
            except Exception as exc:  # noqa: BLE001
                out.append((item["item_id"], None, f"{item['item_id']}: {type(exc).__name__}: {exc}"))
    return out


def build(items: list[dict], workers: int = 8, force: bool = False,
          review_root: Path = ids.REVIEW_ROOT) -> tuple[dict, list[str]]:
    """Render the items without a current packet (keyed on the item, both motions and the code)."""
    import concurrent.futures
    import multiprocessing

    index = read_index()
    by_stem = collections.defaultdict(list)
    for item in items:
        stem = item["stem"]
        motions = {"before": ids.sha256_file(ids.motion_path(stem, fit_dir())),
                   "after": ids.sha256_file(ids.motion_path(stem, ref_dir()))}
        key = ids.sha256_json({**{k: v for k, v in item.items() if not k.startswith("_")}, "motions": motions,
                               "code": [ids.sha256_file(f) for f in (__file__, render_v4.__file__)]})
        old = index.get(item["item_id"])
        if old and not force and old["item_key"] == key and Path(old["dir"]).exists():
            continue
        by_stem[stem].append(item)
    jobs = [(its, _touch(stem), str(review_root)) for stem, its in by_stem.items()]   # stores read here, plant v1
    if workers > 1 and len(jobs) > 1:
        with hm._single_threaded_children(), concurrent.futures.ProcessPoolExecutor(
                min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as ex:
            results = [r for part in ex.map(_build_job, jobs) for r in part]
    else:
        results = [r for j in jobs for r in _build_job(j)]
    failures = []
    for item_id, entry, error in results:
        if error:
            failures.append(error)
        else:
            index[item_id] = entry
    p = index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(e) + "\n" for e in sorted(index.values(), key=lambda e: e["item_id"])))
    return index, failures


def verdict_table(index: dict) -> dict:
    """``edits.verdict_table`` over this index, with the edits' ``closer`` named for plant v2 (``fit`` = the before)."""
    t = E.verdict_table(index)
    e = t["edits"]
    rename = lambda d: {("fit" if k == "shipped" else k): v for k, v in d.items()}  # noqa: E731
    e["closer"], e["more_natural"] = rename(e["closer"]), rename(e["more_natural"])
    for x in e["rejected"]:
        x["closer"] = "fit" if x["closer"] == "shipped" else x["closer"]
        x["more_natural"] = "fit" if x["more_natural"] == "shipped" else x["more_natural"]
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, required=True, help="the labels v2 folder (pinned, never the newest)")
    ap.add_argument("--build-controls", type=int, default=0, help="controls per kind (edits.py's mix)")
    ap.add_argument("--build-edits", action="store_true")
    ap.add_argument("--stem", nargs="*")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--kind", nargs="*", default=list(E.KINDS))
    ap.add_argument("--max-cost", type=float, default=10.0)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    start = time.time()
    clips = labels_clips(args.labels)
    failures = []
    bad = sorted(t for t in E.name_tokens() if t in json.dumps([LEGEND, E.IMAGE_TEXT]).lower())
    if bad:
        print(f"FAILED the fixed texts use corpus name tokens: {bad}", file=sys.stderr)
        return 1
    if args.build_controls:
        n = args.build_controls
        items = control_items(clips, {"identity": max(1, (2 * n) // 5), "swap": n + 2, "lift": n,
                                      "sink": (2 * n) // 3, "kink": (2 * n) // 3})
        _, f = build(items, args.workers)
        failures += f
    if args.build_edits:
        _, f = build(edit_items(clips, args.stem), args.workers)
        failures += f
    index = read_index()
    if args.review:
        reviewer = review.Reviewer(effort=args.effort, pass_=PASS)
        entries = sorted((e for e in index.values() if e["kind"] in args.kind and (not args.stem or e["stem"] in args.stem)),
                         key=lambda e: (E.KINDS.index(e["kind"]), e["item_id"]))
        res = E.run_review(entries, reviewer, max_cost=args.max_cost)
        failures += res["failures"]
        print(f"pass C (render_c2) review: {res['status']}, ${res['spent']:.2f}, {res['unstarted']} unstarted")
    if args.calibrate or args.review:
        table = verdict_table(index)
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        rec = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [verdicts.prompt_path(PASS), verdicts.schema_path(PASS)]),
               "render_v": RENDER_V, "contract": CONTRACT, "rule": verdicts.RULE, "before": f"fit_writer {FIT_ID}",
               "after": capture_v4.RETARGET_ID, **table}
        (CALIBRATION_DIR / f"{RENDER_V}.json").write_text(json.dumps(rec, indent=1) + "\n")
        c, e = table["classes"], table["edits"]
        print("pass C (render_c2) calibration: " + "; ".join(
            f"{k} p={v['precision']} ({v['claims']} claims) r={v['recall']} {'E' if v['evidence'] else 'adv'}"
            for k, v in c.items()) + f"; edits accepted {e['accepted']}/{e['reviewed']}")
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    print(f"edits_v2: {len(index)} packets indexed in {time.time() - start:.0f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
