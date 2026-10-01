# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Labels v2 with the statics restoration (BodyFix Step 4, item 6): the body-body pairs B6 demoted that a hold needs
on the plant come back to its configuration.

B6 (``b6.py``) keeps a configured pair only when every reviewer sample calls it ``required_touch``. The samples can
agree that a contact carries the pose and disagree on the zone it lands in. Peacock -a@788's forearms press into
her lower belly: one sample names TRUNK, the other PELVIS, so neither pair reproduces, all four are demoted, and
statics v2 finds the hold beyond the plant (s* 0.147 -> 1.192: the spine carries what the forearms carried). A
single-pair necessity test cannot see it: each brace is ``useful`` alone, and only the four together are required.

Rule (machine evidence only)
----------------------------
A hold's *demoted* pairs (configured in labels v1.1, not critical, not carried) are restored when

* statics v2 on the **critical-only configuration** (labels v2 built with B6's critical pairs, nothing restored)
  finds the hold ``beyond_plant`` or ``infeasible``, and
* statics v2 on **labels v1.1's configuration** finds it ``held`` or ``feasible``.

The two records must request exactly those configurations at the hold (checked: a record of another configuration
is refused). Every demoted pair of the hold comes back (``label_action: restored``, reasons ``not_critical`` +
``statics_required_jointly``) and the window is again where they hold. A restored pair is configured and
``required_touch`` but **not critical**: the gate's critical flags do not see it.

``build()`` is ``labels_v2.build`` with the restored pairs passed alongside B6's critical pairs (that is what puts
them in the demanded set and the window rule), then relabelled as restored. ``labels_v2.write`` writes it, with this
module's rule and both statics records in the evidence, so the labels id covers them; ``labels_v2.py`` itself is
untouched, so the folders it wrote before (B6's draft, pass 1) still verify by sha256. ``restored.json`` in the
folder lists the restorations.

Two builds, as for labels v2 itself (statics needs the configuration, the labels cite the statics)::

    # pass 2: the restored configuration, statics blocks from the critical-only statics
    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.labels_v2_restore \\
        --critical <b6.json> --votes <b6.json> --critical-statics <statics of pass 1> --statics <statics of pass 1>
    # statics v2 on pass 2, then the final labels citing it
    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.statics_v2 --labels <pass 2>
    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.labels_v2_restore \\
        --critical <b6.json> --votes <b6.json> --critical-statics <statics of pass 1> --statics <statics of pass 2>
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import yaml

from reference_curation import ids, labels_v2 as L2, statics_v2

MODULE = "reference_curation.labels_v2_restore"
RULE_ID = f"rule:{MODULE}"
RESTORE_FROM = ("beyond_plant", "infeasible")     # the critical-only configuration's hold verdict
RESTORE_TO = ("held", "feasible")                 # labels v1.1's configuration's hold verdict
REASONS = ["not_critical", "statics_required_jointly"]


def _row_id(statics_id: str) -> str:
    return f"statics_holds:{statics_id}"


def evidence_row(statics_dir: Path) -> dict:
    """The hold verdicts a restoration reads: the statics record's ``holds.jsonl`` by sha256."""
    st = statics_v2.load(statics_dir)
    sid = st["record"]["statics_id"]
    path = Path(statics_dir) / "holds.jsonl"
    return {"id": _row_id(sid), "kind": "statics_holds", "plant": L2.PLANT, "labels_id": st["record"]["labels_id"],
            "path": ids.display_path(path), "sha256": ids.sha256_file(path)}


def rule_row() -> dict:
    return {"id": RULE_ID, "kind": "rule", "path": ids.display_path(__file__), "sha256": ids.sha256_file(__file__)}


def demoted_pairs(v11: dict, critical: dict) -> dict:
    """``{hold_id: [pair]}``: what ``labels_v2.reconcile`` demotes under ``critical`` (a pair configured in labels v1.1,
    not critical at its hold, not a carried label)."""
    out = {}
    for c in v11["manifest"]["clips"]:
        for h in c["holds"]:
            hid = h["hold_id"]
            crit = critical.get(hid) or {}
            ps = [a["contact"] for a in v11["anns"][hid] if a["kind"] == "pair" and a["in_configuration"]
                  and a["contact"] not in crit and not (a["source_state"] == "unknown" and a["source_label"])]
            if ps:
                out[hid] = ps
    return out


def restorations(v11: dict, critical: dict, critical_statics_dir: Path, v11_statics_dir: Path) -> dict:
    """``{hold_id: {pair: block}}``: the module docstring's rule."""
    sc, sv = statics_v2.load(critical_statics_dir), statics_v2.load(v11_statics_dir)
    cid, vid = sc["record"]["statics_id"], sv["record"]["statics_id"]
    if sv["record"]["labels_id"] != v11["id"]:
        raise ValueError(f"{vid} is statics v2 of {sv['record']['labels_id']}, not of labels v1.1 {v11['id']}")
    configured = {h["hold_id"]: {a["contact"] for a in v11["anns"][h["hold_id"]] if a["kind"] == "pair" and a["in_configuration"]}
                  for c in v11["manifest"]["clips"] for h in c["holds"]}
    out = {}
    for hid, demoted in sorted(demoted_pairs(v11, critical).items()):
        rc, rv = sc["holds"].get(hid), sv["holds"].get(hid)
        if rc is None or rv is None:
            continue
        crit_only = configured[hid] - set(demoted)
        if set(rc["request"]["pairs"]) != crit_only | (set(critical.get(hid) or {}) - configured[hid]):
            raise ValueError(f"{hid}: {cid} requests {sorted(rc['request']['pairs'])}, not the critical-only configuration")
        if set(rv["request"]["pairs"]) != configured[hid]:
            raise ValueError(f"{hid}: {vid} requests {sorted(rv['request']['pairs'])}, not labels v1.1's configuration")
        if rc["verdict"] in RESTORE_FROM and rv["verdict"] in RESTORE_TO:
            block = {"basis": "statics_required_jointly",
                     "critical_only": {"statics_id": cid, "verdict": rc["verdict"], "s_star": rc["gated"]["s_star"],
                                       "top_joints": rc["gated"]["top_joints"][:3]},
                     "v1_1_configuration": {"statics_id": vid, "verdict": rv["verdict"], "s_star": rv["gated"]["s_star"]},
                     "restored_with": sorted(demoted),
                     "evidence_ids": [_row_id(cid), _row_id(vid)]}
            out[hid] = {p: copy.deepcopy(block) for p in demoted}
    return out


def build(v11: dict, audit_dir: Path, statics_dir: Path, critical: dict, restored: dict, votes: dict | None = None,
          vote_ids: dict | None = None, extra_evidence: list[dict] = (), critical_statics_dir: Path | None = None,
          v11_statics_dir: Path | None = None) -> dict:
    """``labels_v2.build`` with ``restored`` demanded, then relabelled (module docstring)."""
    merged = copy.deepcopy(critical)
    for hid, ps in restored.items():
        clash = set(ps) & set(merged.get(hid) or {})
        if clash:
            raise ValueError(f"{hid}: {sorted(clash)} both critical and restored")
        merged.setdefault(hid, {}).update(copy.deepcopy(ps))
    rows = [rule_row()] + [evidence_row(d) for d in (critical_statics_dir, v11_statics_dir) if d is not None]
    result = L2.build(v11, audit_dir, statics_dir, critical=merged, votes=votes, vote_ids=vote_ids,
                      extra_evidence=list(extra_evidence) + rows)
    for a in result["annotations"]:
        blk = (restored.get(a["hold_id"]) or {}).get(a["contact"])
        if blk is None:
            continue
        a.pop("critical", None)
        a["restored"] = blk
        a["label_action"] = "restored"
        a["reasons"] = [r for r in a["reasons"] if r not in REASONS] + REASONS
        a["evidence_ids"] = list(dict.fromkeys(a["evidence_ids"] + [RULE_ID]))
    for hid, ps in restored.items():
        h = result["holds"].get(hid)
        if h is None:
            continue
        blk = next(iter(ps.values()))
        h["labels"]["restored"] = {"pairs": sorted(ps), **{k: blk[k] for k in ("basis", "critical_only", "v1_1_configuration")}}
        h["labels"]["changes"] = h["labels"]["changes"] + [f"restored:{p}" for p in sorted(ps)]
        result["notes"][hid] = {**result["notes"].get(hid, {}), "restored_pairs": sorted(ps)}
    result["critical"] = critical
    result["restored"] = restored
    return result


def check(result: dict) -> list[str]:
    """``labels_v2.check``, plus: the restored annotations are exactly ``restored``, configured, ``required_touch``,
    never critical, and cite this rule and both statics records."""
    problems = L2.check(result)
    seen = set()
    for a in result["annotations"]:
        if a.get("label_action") != "restored":
            continue
        key = (a["hold_id"], a["contact"])
        seen.add(key)
        if a["contact"] not in (result["restored"].get(a["hold_id"]) or {}):
            problems.append(f"{key}: restored but not in the restoration record")
        if not a["in_configuration"] or a["target_role"] != "required_touch" or a.get("critical"):
            problems.append(f"{key}: a restored pair must be configured, required_touch and not critical")
        if not set(a["restored"]["evidence_ids"] + [RULE_ID]) <= set(a["evidence_ids"]):
            problems.append(f"{key}: the restoration's evidence is not cited")
    want = {(h, p) for h, ps in result["restored"].items() for p in ps}
    problems += [f"{k}: in the restoration record but not restored" for k in sorted(want - seen)]
    return problems


def write(result: dict, out_root: Path = L2.LABELS_DIR) -> Path:
    """``labels_v2.write``, plus ``restored.json``, the restorations in ``labels.json`` / ``holds.yaml`` and a section
    in ``summary.md``."""
    out = L2.write(result, out_root)
    hs = result["holds"]
    rec = {"module": MODULE, "rule": {"restore_from": list(RESTORE_FROM), "restore_to": list(RESTORE_TO),
                                      "reasons": REASONS},
           "holds": {hid: {"pairs": sorted(ps), "statics_now": hs[hid]["labels"]["statics"],
                           **{k: next(iter(ps.values()))[k] for k in ("critical_only", "v1_1_configuration")}}
                     for hid, ps in sorted(result["restored"].items())},
           "pairs": sum(len(ps) for ps in result["restored"].values())}
    (out / "restored.json").write_text(json.dumps(rec, indent=1) + "\n")
    lab = json.loads((out / "labels.json").read_text())
    lab["restored"] = {"module": MODULE, "pairs": rec["pairs"], "holds": sorted(rec["holds"])}
    (out / "labels.json").write_text(json.dumps(lab, indent=1) + "\n")
    m = yaml.safe_load((out / "holds.yaml").read_text())
    m["labels"]["restored_by"] = MODULE
    (out / "holds.yaml").write_text(yaml.safe_dump(m, sort_keys=False, width=120))
    lines = ["", "## Pairs restored by statics (`labels_v2_restore`)", "",
             "B6 demoted them (not critical); statics v2 finds the hold beyond the plant without them and holdable with "
             "labels v1.1's configuration.", "",
             "| Hold | Pairs | Critical-only s* | v1.1 configuration s* | Now |", "|---|---|---|---|---|"]
    for hid, r in rec["holds"].items():
        now = r["statics_now"] or {}
        lines.append(f"| `{hid}` | {', '.join(r['pairs'])} | {r['critical_only']['s_star']} ({r['critical_only']['verdict']}) | "
                     f"{r['v1_1_configuration']['s_star']} ({r['v1_1_configuration']['verdict']}) | "
                     f"{now.get('s_star')} ({now.get('verdict')}, `{now.get('statics_id')}`) |")
    with open(out / "summary.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, default=L2.LABELS_V11, help="labels v1.1 (the human-side decisions)")
    ap.add_argument("--audit", type=Path, help="audit v2 (calibrated) of those labels (default: the only one)")
    ap.add_argument("--critical", type=Path, required=True, help="B6's record (b6.json), admitted")
    ap.add_argument("--votes", type=Path, help="the reviewer's variant votes per hold (b6.json)")
    ap.add_argument("--critical-statics", type=Path, required=True,
                    help="statics v2 of labels v2 built with the same critical pairs and nothing restored")
    ap.add_argument("--v11-statics", type=Path, help="statics v2 of labels v1.1 (default: the only one)")
    ap.add_argument("--statics", type=Path, required=True, help="statics v2 of the configuration being labelled")
    ap.add_argument("--out-root", type=Path, default=L2.LABELS_DIR)
    args = ap.parse_args(argv)
    start = time.time()
    try:
        v11 = L2.read_v11(args.labels)
        audit_dir = args.audit or L2.default_audit_dir(v11["id"])
        v11_statics = args.v11_statics or L2.default_statics_dir(v11["id"])
        crit = json.loads(args.critical.read_text())
        if not crit.get("admitted"):
            raise ValueError(f"{args.critical}: the critical-contact class is not admitted; nothing to restore against")
        critical = crit["critical"]
        restored = restorations(v11, critical, args.critical_statics, v11_statics)
        votes, vote_ids, vote_rows = L2.load_votes(args.votes)
        result = build(v11, audit_dir, args.statics, critical, restored, votes=votes, vote_ids=vote_ids,
                       extra_evidence=crit.get("evidence", []) + vote_rows, critical_statics_dir=args.critical_statics,
                       v11_statics_dir=v11_statics)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    failures = result["failures"] + check(result)
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    if failures:
        print(f"labels_v2_restore: {len(failures)} failures; nothing written", file=sys.stderr)
        return 1
    out = write(result, args.out_root)
    a = L2.metrics(result)["all"]
    print(f"labels_v2_restore {out.name}: restored {sum(len(p) for p in restored.values())} pairs on {len(restored)} holds "
          f"{sorted(restored)}; {a['holds']} holds {a['status']}; pairs configured {a['pairs_configured']} (critical "
          f"{a['critical_pairs']}, not realised {a['pairs_configured_not_realised']}); windows trimmed "
          f"{a['windows_trimmed_for_pairs']}; statics {result['statics_id']} in {time.time() - start:.0f} s -> "
          f"{ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
