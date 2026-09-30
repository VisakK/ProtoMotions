# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Labels v1.1 (BUILD_PLAN Step 8): labels v1 with the statics of the retargeted references merged in.

Step 7's statics judged the *shipped* references, most of which could not realise their supports, so a
contact's necessity came from the counterfactual (the human's contacts closed). On the retargeted references
the supports are realised, so the gated verdict speaks for the reference itself (``necessity_basis`` =
``gated``). This module carries that evidence into the labels, and changes no rule of labels v1:

* every configured annotation gets a ``statics`` block: the hold's verdict on the retargeted reference
  (held, feasible, beyond_plant, infeasible, support_not_realised), the contact's gated necessity
  (required, useful, redundant, ...), its representative load and its load interval within the plant's
  torque limits;
* a ground support whose role is ``required_touch`` (the capture confirms the touch, the mat cannot confirm
  the load) becomes ``required_support`` when statics finds it **required**: the plant cannot hold the pose
  without it, whatever the other contacts do. That is load confirmed by physics instead of by the mat
  (the plan's example: 82 such supports in the counterfactual). The change is recorded as
  ``role_from: statics``;
* body-body roles do not change: no labelled pair is statically required on the shipped references
  (Step 7), and the necessity attached here is load-path evidence for the Step 9 sidecar, not a critical
  contact (Step 6's caveat);
* ``holds.yaml`` points every clip's ``source`` at its retargeted motion, so the Step 9 consumers read the
  retargeted references. ``labels.source_v1`` keeps labels v1's source (the grounded conversion),
  ``labels.retarget_input`` the motion the retarget edited (the shipped ftC reference, from the retarget's
  record) and ``labels.retargeted`` whether it was edited (false: no human mesh, the shipped motion copied).

``data/reference_curation/labels/<manifest stem>.labels_v1_1.<hash>/`` holds ``holds.yaml``,
``annotations.jsonl`` (v1's evidence ids plus ``statics:<statics_id>``), ``labels.json`` and
``summary.md``.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.roles \\
        --statics <statics dir on the retargeted references> --retarget-dir <retarget motions>
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path

import yaml

from reference_curation import ids, retarget, statics

MODULE = "reference_curation.roles"
SCHEMA_VERSION = 1
LABELS_VERSION = "v1_1"
STATICS_KEYS = ("load_n", "load_min_n", "load_max_n", "s_without", "effort_relief", "height_cm", "gap_cm", "realised")


def merge(labels_dir: Path, statics_dir: Path, retarget_dir: Path) -> dict:
    """``{"holds", "annotations", "changes", "statics_id"}``: labels v1 with the statics merged in."""
    labels_dir, statics_dir = Path(labels_dir), Path(statics_dir)
    manifest = copy.deepcopy(ids.load_manifest(labels_dir / "holds.yaml"))
    anns = [json.loads(l) for l in open(labels_dir / "annotations.jsonl")]
    holds = {json.loads(l)["hold_id"]: json.loads(l) for l in open(statics_dir / "holds.jsonl")}
    rows = {(r["hold_id"], r["contact"]): r for r in map(json.loads, open(statics_dir / "contacts.jsonl"))}
    sid = next(iter(holds.values()))["statics_id"]
    changes = collections.Counter()
    out = []
    for a in anns:
        a = copy.deepcopy(a)
        r = rows.get((a["hold_id"], a["contact"]))
        h = holds.get(a["hold_id"])
        if r is not None:
            g = r["gated"]
            a["statics"] = {"statics_id": sid, "hold_verdict": h["verdict"], "necessity": g["necessity"],
                            "basis": "gated_retarget", **{k: g.get(k) for k in STATICS_KEYS}}
            a["evidence_ids"] = a["evidence_ids"] + [f"statics:{sid}"]
            if a["kind"] == "ground" and a["target_role"] == "required_touch" and g["necessity"] == "required":
                a["role_v1"] = a["target_role"]
                a["target_role"] = "required_support"
                a["role_from"] = "statics"
                changes["required_touch->required_support"] += 1
            changes[f"{a['kind']}:{g['necessity']}"] += 1
        out.append(a)
    by_hold = collections.defaultdict(dict)
    for a in out:
        if a.get("role_from") == "statics":
            by_hold[a["hold_id"]][a["contact"]] = a["target_role"]
    record = json.loads((retarget.RECORD_ROOT / Path(retarget_dir).name / "retarget.json").read_text())
    edited_from = ids.REPO / record["motion_dir"]
    for clip in manifest["clips"]:
        motion = ids.motion_path(clip["stem"], retarget_dir)
        clip["labels"] = {**clip.get("labels", {}), "source_v1": clip["source"],
                          "retarget_input": str(ids.motion_path(clip["stem"], edited_from).resolve()),
                          "retargeted": clip["stem"] in record["motions"]}
        clip["source"] = str(motion.resolve())
        for hold in clip["holds"]:
            for contact, role in by_hold.get(hold["hold_id"], {}).items():
                hold["labels"]["roles"][contact] = role
            v = holds.get(hold["hold_id"])
            if v is not None:
                hold["labels"]["statics"] = {"statics_id": sid, "verdict": v["verdict"],
                                             "s_star": v["gated"]["s_star"]}
    return {"holds": manifest, "annotations": out, "changes": dict(changes), "statics_id": sid}


def labels_id(labels_dir: Path, statics_dir: Path, retarget_dir: Path) -> str:
    key = {"schema": SCHEMA_VERSION, "labels": Path(labels_dir).name, "statics": Path(statics_dir).name,
           "retarget": Path(retarget_dir).name, "generator": ids.sha256_file(__file__)}
    stem = ids.load_manifest(Path(labels_dir) / "holds.yaml")["labels"]["source_manifest"]
    return f"{Path(stem).stem}.labels_{LABELS_VERSION}.{ids.sha256_json(key)[:10]}"


def write(result: dict, labels_dir: Path, statics_dir: Path, retarget_dir: Path,
          out_root: Path = statics.LABELS_DIR) -> Path:
    lid = labels_id(labels_dir, statics_dir, retarget_dir)
    out = Path(out_root) / lid
    out.mkdir(parents=True, exist_ok=True)
    manifest = result["holds"]
    manifest["labels"] = {**manifest["labels"], "labels_id": lid, "module": MODULE, "base_labels_id":
                          Path(labels_dir).name, "statics_id": result["statics_id"],
                          "retarget_id": Path(retarget_dir).name}
    (out / "holds.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=120))
    (out / "annotations.jsonl").write_text("".join(json.dumps({**a, "labels_id": lid}) + "\n"
                                                   for a in result["annotations"]))
    inputs = [Path(labels_dir) / "holds.yaml", Path(labels_dir) / "annotations.jsonl",
              Path(statics_dir) / "holds.jsonl", Path(statics_dir) / "contacts.jsonl",
              retarget.RECORD_ROOT / Path(retarget_dir).name / "retarget.json"]
    record = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), "labels_id": lid,
              "base_labels_id": Path(labels_dir).name, "statics_id": result["statics_id"],
              "retarget_id": Path(retarget_dir).name, "changes": result["changes"]}
    (out / "labels.json").write_text(json.dumps(record, indent=1) + "\n")
    lines = [f"# Labels `{lid}`", "", f"Labels v1 `{Path(labels_dir).name}` with the statics of the retargeted "
             f"references (`{result['statics_id']}`) merged in (`reference_curation.roles`).", "",
             "| Change or class | Count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(result["changes"].items())]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--statics", type=Path, required=True)
    ap.add_argument("--retarget-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=statics.LABELS_DIR)
    args = ap.parse_args(argv)
    labels_dir = args.labels or statics.default_labels_dir()
    try:
        result = merge(labels_dir, args.statics, args.retarget_dir)
        missing = [c["stem"] for c in result["holds"]["clips"] if not Path(c["source"]).exists()]
        if missing:
            raise FileNotFoundError(f"no retargeted motion for {missing[:3]} ...")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    out = write(result, labels_dir, args.statics, args.retarget_dir, args.out_root)
    print(f"labels v1.1: {result['changes']} -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
