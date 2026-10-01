# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Critical body-body contacts by the reviewer's self-consistency (BodyFix Step 4, item 6; TODO B6), and the pose
role votes of the secondary holds.

Which body-body contacts *define* a pose (Crow's shins on the upper arms, Tree's foot on the thigh) is a semantic
question. Labels v1's ``required_touch`` is a physics test (labelled, touched, Tier-1 consequential); statics
cannot decide it (no pair is statically required); and no human truth set will be made (minimal human input). The
route left is the reviewer agreeing with itself (BUILD_PLAN Step 6's caveat, Step 7's note):

* **Packets.** One informed render_v4 packet per hold (``packets_v4.build_packet_b``: the Pass-B contract
  unchanged, the source labels, the plant-v2 avatar, the skin's and the avatar's gaps per candidate pair). Holds:
  every non-transition hold of the labels v2 draft with a configured pair, a pair the render_v3 reviewer called
  ``required_touch`` on a stable human contact, or a stable Tier-1 consequential human contact (78), plus the
  secondary (``_h``) holds whose pose role the draft leaves ``undecided``.
* **Samples.** ``SAMPLES`` independent calls per packet (fresh context each, ``review.run(samples=...)``), all in the
  shared ledger (``<packet_id>.B.<n>.json``).
* **Within-packet reproducibility** (the claim class ``critical``: one sample calls a candidate ``required_touch``):
  over every ordered pair of valid samples of one packet, how often the other sample repeats a ``required_touch``
  claim. Step 4's rule decides admission: precision >= 0.9 on >= 30 claims, answered rate (a role other than
  ``cannot_tell``) >= 0.5 over >= 30 items. The same is reported for the other value (``allowed``/``incidental``),
  because a reviewer that called everything critical would repeat itself perfectly and decide nothing; the class
  is admitted only if both values reproduce (the two-valued rule: the lower precision).
* **Across takes** (reported, and a check): a contact pattern (pose family, pair with left/right mirrored to one
  canonical side) seen in two or more clips; how often another take's consensus repeats a ``required_touch``
  consensus. The render_v3 reviewer's single sample, where it exists, is reported against the v4 consensus
  (``cross_render``).

A contact is **critical** when the class is admitted, every valid v4 sample calls it ``required_touch``, and the
human holds it stably at the exemplar (``source_state`` observed, ``stable``): Step 6's deferred rule. Without
admission nothing is critical and labels v2 keeps the deterministic configuration. A reproducible claim is
reliability, not validity: a reviewer can be consistently wrong. That caveat goes with the result.

Output: ``data/reference_curation/b6/<labels_id>.b6.<hash>/b6.json`` (``admitted``, ``critical`` per hold, the
metrics, the ``votes`` of every hold's variant answers for ``labels_v2.pose_role``, and the ``evidence`` rows of
every verdict used) and ``summary.md``.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.b6 --build
    ... --review --max-cost 55
    ... --decide
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import sys
import time
from pathlib import Path

from reference_curation import packets_v4  # first: render (OSMesa)
from reference_curation import capture_v4, ids, labels as L1, labels_v2, packets, review, verdicts

MODULE = "reference_curation.b6"
SCHEMA_VERSION = 1
SAMPLES = 2
B6_DIR = ids.DATA_ROOT / "b6"
NOT_CRITICAL = ("allowed", "incidental")


def labels_v2_draft() -> Path:
    found = sorted(labels_v2.LABELS_DIR.glob("*.labels_v2.*"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError("no labels v2 folder: run -m reference_curation.labels_v2 first")
    return found[0]


def read_labels(d: Path) -> tuple[dict, dict]:
    m = ids.load_manifest(Path(d) / "holds.yaml")
    anns = collections.defaultdict(list)
    for line in open(Path(d) / "annotations.jsonl"):
        a = json.loads(line)
        anns[a["hold_id"]].append(a)
    return m, dict(anns)


def select_holds(manifest: dict, anns: dict) -> dict:
    """``{hold_id: why}``: the holds whose candidate pairs could be critical, and the undecided secondary holds."""
    out = {}
    for c in manifest["clips"]:
        for h in c["holds"]:
            if h["labels"]["status"] == "transition":
                continue
            ps = [a for a in anns[h["hold_id"]] if a["kind"] == "pair"]
            stable = [a for a in ps if a["source_state"] == "observed_contact" and a["stable"]]
            why = []
            if any(a["in_configuration"] for a in ps):
                why.append("configured")
            if any((a.get("review") or {}).get("role") == "required_touch" for a in stable):
                why.append("v3_critical")
            if any(a["load_path_class"] == "consequential_internal" for a in stable):
                why.append("consequential")
            if h["labels"]["pose_role"] == "undecided":
                why.append("pose_role_undecided")
            if why:
                out[h["hold_id"]] = why
    return out


def jobs(manifest: dict, holds: dict, audit_dir: Path, labels_v11: Path) -> list[tuple]:
    """``(item, audit record, store v4 record, labels v2 hold, purpose)`` per hold, stores read here (plant v1)."""
    aud = packets.load_audit(audit_dir)
    queue = {q["hold_id"]: q for q in aud.queue if q["pass"] == "B"}
    by_id = {h["hold_id"]: h for c in manifest["clips"] for h in c["holds"]}
    out = []
    for hid in sorted(holds):
        record = aud.records[hid]
        item = queue.get(hid) or packets.item_for(record, "B")
        rec = capture_v4.load(record["stem"], rebuild=False)
        out.append((item, record, rec, by_id[hid], "b6" if holds[hid] != ["pose_role_undecided"] else "pose_role"))
    return out


# --------------------------------------------------------------------------- #
# Claims and agreement (pure)
# --------------------------------------------------------------------------- #
def v4_claims(index: dict, ledger: list[dict], reviewer_key: str) -> dict:
    """``{hold_id: [claims]}``: every valid v4 Pass-B verdict of the indexed packets (``labels.parse_review``)."""
    by_pid = {e["packet_id"]: e for e in index.values()}
    out = collections.defaultdict(list)
    for r in ledger:
        if r["pass"] != "B" or r["render_v"] != packets_v4.RENDER_V or r["status"] != "valid":
            continue
        if r["packet_id"] not in by_pid or r["reviewer"]["key"] != reviewer_key:
            continue
        packet = json.loads((Path(r["packet"]["dir"]) / "packet.json").read_text())
        c = L1.parse_review(r, packet)
        c["path"] = str(verdicts.verdict_path(verdicts.LEDGER_DIR, r["packet_id"], "B", r["n"]))
        out[r["hold_id"]].append(c)
    return dict(out)


def reproducibility(claims: dict) -> dict:
    """Within-packet agreement of the role claims over every ordered pair of samples of one packet (module doc)."""
    rows = {"critical": [0, 0], "not_critical": [0, 0]}
    items = answered = 0
    for hid, cs in claims.items():
        for pair in sorted({p for c in cs for p in c["candidates"]}):
            roles = [c["roles"].get(pair) for c in cs]
            items += 1
            answered += sum(r is not None for r in roles) / len(roles)
            for i, j in itertools.permutations(range(len(roles)), 2):
                ri, rj = roles[i], roles[j]
                if ri == "required_touch":
                    rows["critical"][0] += 1
                    rows["critical"][1] += rj == "required_touch"
                elif ri in NOT_CRITICAL:
                    rows["not_critical"][0] += 1
                    rows["not_critical"][1] += rj in NOT_CRITICAL
    out = {"items": items, "answered_rate": round(answered / items, 4) if items else None}
    for k, (n, agree) in rows.items():
        out[k] = {"claims": n, "agree": agree, "precision": round(agree / n, 4) if n else None,
                  "precision_lo95": verdicts.wilson_low(agree, n)}
    return out


def consensus(roles: list) -> str | None:
    """The role every sample gives (``None`` on any disagreement or abstention)."""
    return roles[0] if roles and all(r == roles[0] for r in roles) and roles[0] is not None else None


MIRROR = {"L_": "R_", "R_": "L_"}


def mirror(pair: str) -> str:
    zs = [(MIRROR[z[:2]] + z[2:]) if z[:2] in MIRROR else z for z in pair.split("+")]
    return L1.pair_name(sorted(zs, key=L1.ZI.get))


def family(stem: str) -> str:
    """The pose family of a clip (the stem without its session prefix and take suffix)."""
    return re.sub(r"[_-]+[a-d]$", "", stem.split("_", 1)[1])


def cross_take(claims: dict) -> dict:
    """Across takes: per (family, canonical pair), the consensus of each clip's holds; how often another take repeats
    a ``required_touch`` consensus."""
    by_pattern = collections.defaultdict(dict)
    for hid, cs in claims.items():
        stem = hid.rsplit("@", 1)[0]
        for pair in sorted({p for c in cs for p in c["candidates"]}):
            con = consensus([c["roles"].get(pair) for c in cs])
            if con is None:
                continue
            key = (family(stem), min(pair, mirror(pair)))
            by_pattern[key].setdefault(stem, []).append(con)
    n = agree = 0
    patterns = []
    for (fam, pair), takes in by_pattern.items():
        if len(takes) < 2:
            continue
        per_take = {s: collections.Counter(v).most_common(1)[0][0] for s, v in takes.items()}
        crit = [s for s, r in per_take.items() if r == "required_touch"]
        for s in crit:
            for t in per_take:
                if t != s:
                    n += 1
                    agree += per_take[t] == "required_touch"
        patterns.append({"family": fam, "pair": pair, "takes": per_take})
    return {"patterns": len(patterns), "claims": n, "agree": agree, "precision": round(agree / n, 4) if n else None,
            "precision_lo95": verdicts.wilson_low(agree, n), "detail": patterns}


def admitted(rep: dict) -> tuple[bool, str]:
    """Step 4's rule on both values of the role claims (the lower precision decides)."""
    crit, other = rep["critical"], rep["not_critical"]
    checks = [(crit["claims"] >= verdicts.CLAIMS_MIN, f"critical claims {crit['claims']} < {verdicts.CLAIMS_MIN}"),
              (other["claims"] >= verdicts.CLAIMS_MIN, f"non-critical claims {other['claims']} < {verdicts.CLAIMS_MIN}"),
              ((crit["precision"] or 0) >= verdicts.PRECISION_MIN, f"critical reproducibility {crit['precision']} < 0.9"),
              ((other["precision"] or 0) >= verdicts.PRECISION_MIN,
               f"non-critical reproducibility {other['precision']} < 0.9"),
              ((rep["answered_rate"] or 0) >= verdicts.ANSWERED_MIN, f"answered rate {rep['answered_rate']} < 0.5"),
              (rep["items"] >= verdicts.ITEMS_MIN, f"items {rep['items']} < {verdicts.ITEMS_MIN}")]
    failed = [msg for ok, msg in checks if not ok]
    return not failed, "; ".join(failed) or "admitted"


def decide(manifest: dict, anns: dict, claims: dict, v3_votes: dict, is_admitted: bool) -> tuple[dict, dict, dict, dict]:
    """``(critical, votes, vote_verdicts, cross_render)``: the critical contacts (admitted class, every v4 sample
    ``required_touch``, a stable human contact), every hold's variant votes (render_v3's sample and the v4 samples)
    and the verdict ids behind them, and how the render_v3 reviewer's roles compare with the v4 consensus."""
    critical, votes, vote_ids = {}, {}, {}
    cross = collections.Counter()
    for c in manifest["clips"]:
        for h in c["holds"]:
            hid = h["hold_id"]
            cs = claims.get(hid, [])
            v3 = v3_votes.get(hid)
            votes[hid] = ([v3[0]] if v3 else []) + [x["variant"]["matches_label"] for x in cs]
            vote_ids[hid] = ([v3[1]] if v3 else []) + [x["verdict_id"] for x in cs]
            if not cs:
                continue
            by_contact = {a["contact"]: a for a in anns[hid] if a["kind"] == "pair"}
            for pair in sorted({p for x in cs for p in x["candidates"]}):
                roles = [x["roles"].get(pair) for x in cs]
                con = consensus(roles)
                a = by_contact.get(pair)
                v3 = ((a or {}).get("review") or {}).get("role")
                if v3 is not None and con is not None:
                    cross["agree" if v3 == con else "differ"] += 1
                if not is_admitted or con != "required_touch" or a is None:
                    continue
                if a["source_state"] != "observed_contact" or not a["stable"]:
                    continue
                critical.setdefault(hid, {})[pair] = {
                    "consensus": con, "samples": len(roles), "v3_role": v3,
                    "evidence_ids": [x["verdict_id"] for x in cs]}
    return critical, votes, vote_ids, dict(cross)


def _verdict_row(verdict_id: str, render_v: str) -> dict:
    """``ledger:<packet>.B.<n>`` -> its evidence row (path and sha256 of the ledger file)."""
    pid, _, n = verdict_id.split(":", 1)[1].rsplit(".", 2)
    path = verdicts.verdict_path(verdicts.LEDGER_DIR, pid, "B", int(n))
    return {"id": verdict_id, "kind": "verdict", "render_v": render_v, "path": ids.display_path(path),
            "sha256": ids.sha256_file(path), "packet_id": pid}


def evidence_rows(claims: dict, vote_ids: dict) -> list[dict]:
    """Every verdict b6 used: the v4 samples and the render_v3 samples behind the votes."""
    rows = {c["verdict_id"]: _verdict_row(c["verdict_id"], packets_v4.RENDER_V) for cs in claims.values() for c in cs}
    for vids in vote_ids.values():
        for v in vids:
            rows.setdefault(v, _verdict_row(v, "render_v3"))
    return [rows[k] for k in sorted(rows)]


def v3_votes(manifest: dict) -> dict:
    """render_v3's single Pass-B variant answer per hold and its verdict id (labels v2's ``review_v3``)."""
    out = {}
    for c in manifest["clips"]:
        for h in c["holds"]:
            r = h["labels"].get("review_v3")
            if r:
                out[h["hold_id"]] = (r["variant"]["matches_label"], r["verdict"])
    return out


PASS_B_CALIBRATION = verdicts.CALIBRATION_DIR / "pass_b" / f"{packets_v4.RENDER_V}.json"


def calibrate_b(ledger: list[dict], index: dict, records: dict) -> tuple[dict, list[str]]:
    """The Pass-A classes of the v4 Pass-B verdicts (one per packet and reviewer key, the lowest valid ``n``) scored
    at each packet's main moment against ``packets_v4.truth`` on the reference: ``informed.calibrate``'s table on
    render_v4. The reviewer is shown the measurements, so admission means it reads them faithfully."""
    from reference_curation import capture

    cal = capture.load_calibration()
    by_pid = {e["packet_id"]: e for e in index.values()}
    mine = [r for r in ledger if r["pass"] == "B" and r["render_v"] == packets_v4.RENDER_V and r["packet_id"] in by_pid]
    truths, failures = {}, []
    for r in mine:
        e = by_pid[r["packet_id"]]
        if r["status"] != "valid" or e["packet_id"] in truths:
            continue
        try:
            rec = capture_v4.load(e["stem"], rebuild=False)
            truths[e["packet_id"]] = packets_v4.truth(records[e["hold_id"]], rec, cal, e["frame_hold"], "reference")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{e['item_id']}: {type(exc).__name__}: {exc}")
    reviewers = []
    for key in sorted({r["reviewer"]["key"] for r in mine}):
        first = {}
        for r in mine:
            if r["reviewer"]["key"] == key and r["status"] == "valid" and r["packet_id"] in truths:
                first.setdefault(r["packet_id"], r)
        chosen = list(first.values())
        if not chosen:
            continue
        rows = [x for v in chosen for x in verdicts.score(v["answer"], truths[v["packet_id"]], {"packet_id": v["packet_id"]})]
        classes = {cls: verdicts.class_metrics([x for x in rows if x["class"] == cls], kind)
                   for cls, kind in verdicts.CLASSES.items()}
        calls = [r for r in mine if r["reviewer"]["key"] == key]
        reviewers.append({"key": key, "model": chosen[0]["reviewer"]["model"], "effort": chosen[0]["reviewer"]["effort"],
                          "packets": len(chosen), "calls": dict(collections.Counter(c["status"] for c in calls)),
                          "cost_usd": verdicts._stats([c.get("cost_usd") for c in calls]), "classes": classes,
                          "evidence": [c for c, m in classes.items() if m["evidence"] and c != "pose_identity"],
                          "advisory": sorted({c for c, m in classes.items() if not m["evidence"]} | {"pose_identity", "roles"})})
    return ({"render_v": packets_v4.RENDER_V, "pass": "B", "contract": packets_v4.CONTRACT_B, "rule": verdicts.RULE,
             "truth": {**verdicts.TRUTH, "frame": "the packet's main moment (labels v2's exemplar)",
                       "avatar": "the reference on plant v2"},
             "note": "pose_identity is informed here (the label names the pose); roles have no truth set (b6.py)",
             "packets": len(truths), "reviewers": reviewers}, failures)


def summary_markdown(rec: dict) -> str:
    rep, ct = rec["reproducibility"], rec["cross_take"]
    lines = [f"# B6 critical contacts `{rec['b6_id']}`", "",
             f"Generated by `{MODULE}` (BodyFix Step 4, item 6): the reviewer's self-consistency over {rec['samples']} "
             f"independent render_v4 Pass-B samples of {rec['holds']} holds. The rules are in the module docstring.", "",
             f"**Admitted: {'yes' if rec['admitted'] else 'no'}** ({rec['admission']}).", "",
             "| Claim value | Claims | Repeated by another sample | Precision (95 % low) |", "|---|---|---|---|"]
    for k in ("critical", "not_critical"):
        x = rep[k]
        lines.append(f"| {k} | {x['claims']} | {x['agree']} | {x['precision']} ({x['precision_lo95']}) |")
    lines += ["", f"Items (hold, candidate pair): {rep['items']}; answered rate {rep['answered_rate']}.", "",
              f"Across takes: {ct['patterns']} patterns seen in two or more clips; a `required_touch` consensus is "
              f"repeated by another take {ct['agree']} of {ct['claims']} times ({ct['precision']}).", "",
              f"render_v3's single sample against the v4 consensus: {rec['cross_render']}.", "",
              f"Critical contacts: {sum(len(v) for v in rec['critical'].values())} in {len(rec['critical'])} holds.", "",
              "Caveat: reproducibility is reliability, not validity. A reviewer can repeat a wrong judgement; no human truth "
              "set exists by design (minimal human input).", ""]
    if rec["critical"]:
        lines += ["| Hold | Critical contacts |", "|---|---|"]
        lines += [f"| `{h}` | {', '.join(sorted(v))} |" for h, v in sorted(rec["critical"].items())]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--build", action="store_true")
    what.add_argument("--review", action="store_true")
    what.add_argument("--decide", action="store_true")
    what.add_argument("--calibrate", action="store_true", help="the Pass-B (render_v4) calibration table")
    ap.add_argument("--labels", type=Path, help="the labels v2 draft (default: the oldest labels v2 folder)")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-cost", type=float, default=55.0)
    ap.add_argument("--effort", choices=review.EFFORTS, default="high")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    start = time.time()
    ldir = args.labels or labels_v2_draft()
    manifest, anns = read_labels(ldir)
    holds = select_holds(manifest, anns)
    audit_dir = labels_v2.default_audit_dir(manifest["labels"]["base_labels_id"])
    reviewer = review.Reviewer(model="stub" if args.dry_run else review.MODEL, effort=args.effort, pass_="B")
    if args.build:
        js = jobs(manifest, holds, audit_dir, labels_v2.LABELS_V11)[:args.limit]
        entries, failures, built = packets_v4.build_batch_b(js, labels_v2.LABELS_V11 / "holds.yaml")
        for f in failures:
            print(f"FAILED {f}", file=sys.stderr)
        print(f"b6 build: {len(entries)} of {len(js)} informed packets ready ({built} built; "
              f"{dict(collections.Counter(e['purpose'] for e in entries))}) in {time.time() - start:.0f} s")
        return 1 if failures else 0
    index = packets.read_index(packets_v4.index_path("B"))
    if args.review:
        entries = sorted((e for e in index.values() if e["hold_id"] in holds), key=lambda e: e["item_id"])
        entries = entries[:args.limit] if args.limit else entries
        if not args.dry_run and not review.cli_version(reviewer.cli):
            print("FAILED the reviewer CLI does not run", file=sys.stderr)
            return 1
        from reference_curation import informed

        out = review.run(entries, reviewer, informed.call_stub if args.dry_run else review.call_claude,
                         ledger_dir=review.DRY_LEDGER_DIR if args.dry_run else verdicts.LEDGER_DIR,
                         samples=args.samples, max_cost=args.max_cost, est_cost=0.35)
        for f in out["failures"]:
            print(f"FAILED {f}", file=sys.stderr)
        print(f"b6 review {packets_v4.RENDER_V} key {reviewer.key}: {len(entries)} packets x {args.samples} samples, "
              f"{out['current']} current, {sum(out['status'].values())} made {out['status']}, {out['unstarted']} left by "
              f"the budget, {len(out['blocked'])} blocked; ${out['spent']:.2f}")
        return 1 if out["failures"] or out["blocked"] else (2 if out["unstarted"] else 0)
    if args.calibrate:
        aud = packets.load_audit(audit_dir)
        table, failures = calibrate_b(verdicts.read_ledger(), index, aud.records)
        for f in failures:
            print(f"FAILED {f}", file=sys.stderr)
        if failures or not table["reviewers"]:
            return 1
        inputs = [Path(__file__), verdicts.prompt_path("B"), verdicts.schema_path("B"), packets_v4.index_path("B")]
        PASS_B_CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
        PASS_B_CALIBRATION.write_text(json.dumps({**ids.provenance(SCHEMA_VERSION, MODULE, __file__, inputs), **table},
                                                 indent=1) + "\n")
        for rv in table["reviewers"]:
            print(f"pass B {packets_v4.RENDER_V} key {rv['key']}: {rv['packets']} packets; evidence {rv['evidence']}")
            for cls, m in rv["classes"].items():
                print(f"  {cls:17s} {'EVIDENCE' if m['evidence'] else 'advisory':8s} precision {m['precision']} on "
                      f"{m['support']} claims, answered {m['answered_rate']} of {m['items']}")
        return 0
    claims = v4_claims(index, verdicts.read_ledger(), reviewer.key)
    claims = {h: cs for h, cs in claims.items() if h in holds}
    short = {h: len(cs) for h, cs in claims.items() if len(cs) < args.samples}
    rep = reproducibility({h: cs for h, cs in claims.items() if len(cs) >= 2})
    ok, why = admitted(rep)
    critical, votes, vote_ids, cross = decide(manifest, anns, claims, v3_votes(manifest), ok)
    ct = cross_take(claims)
    key = {"schema": SCHEMA_VERSION, "labels": manifest["labels"]["labels_id"], "generator": ids.sha256_file(__file__),
           "verdicts": sorted(c["verdict_id"] for cs in claims.values() for c in cs)}
    bid = f"{manifest['labels']['labels_id']}.b6.{ids.sha256_json(key)[:10]}"
    rec = {**ids.provenance(SCHEMA_VERSION, MODULE, __file__, [Path(ldir) / "annotations.jsonl"]), "b6_id": bid,
           "labels_id": manifest["labels"]["labels_id"], "reviewer_key": reviewer.key, "samples": args.samples,
           "holds": len(claims), "holds_selected": len(holds), "holds_short_of_samples": short,
           "admitted": ok, "admission": why, "rule": verdicts.RULE, "reproducibility": rep,
           "cross_take": {k: v for k, v in ct.items() if k != "detail"}, "cross_take_patterns": ct["detail"],
           "cross_render": cross, "critical": critical, "votes": votes, "vote_verdicts": vote_ids,
           "evidence": evidence_rows(claims, vote_ids),
           "role_counts": dict(collections.Counter(r or "cannot_tell" for cs in claims.values() for c in cs
                                                   for r in c["roles"].values()))}
    out = B6_DIR / bid
    out.mkdir(parents=True, exist_ok=True)
    (out / "b6.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "summary.md").write_text(summary_markdown(rec))
    print(f"b6 {bid}: {len(claims)} holds; critical reproducibility {rep['critical']['precision']} on "
          f"{rep['critical']['claims']} claims, non-critical {rep['not_critical']['precision']} on "
          f"{rep['not_critical']['claims']}; admitted {ok} ({why}); critical contacts "
          f"{sum(len(v) for v in critical.values())} -> {ids.display_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
