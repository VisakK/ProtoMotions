# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless reviewer (BUILD_PLAN Step 4): the blind review of built packets with ``claude -p``,
every call written to the verdict ledger (``verdicts.py``).

For each pending packet the runner:
1. checks the packet is blind: outside the repo, no ``CLAUDE.md`` in it or above it, and holding
   exactly the indexed files with their indexed sha256;
2. runs the reviewer with the packet directory as its working directory;
3. validates the answer (``verdicts.validate``);
4. writes the call to ``data/reference_curation/ledger/verdicts/<packet_id>.A.<n>.json``, valid or
   not.

A packet is *pending* while it has fewer valid verdicts under this reviewer's key than
``--samples``, and fewer than ``--max-attempts`` failed calls. So a run can be stopped and resumed,
and a rebuild (``-m reference_curation.verdicts``) replays the ledger and never re-queries. At most
3 calls run at once. ``--max-cost`` stops launching calls once the spend so far plus the expected
cost of the calls in flight would pass it. ``--per-call-max`` caps each call (``--max-budget-usd``).

The reviewer
------------
The command (CLI 2.1.280, verified 2026-09-28)::

    claude -p <prompts/pass_a.md> --model claude-opus-5-5 --effort high
        --tools Read --allowedTools Read --restricted
        --strict-mcp-config --disable-slash-commands --no-session-persistence
        --exclude-dynamic-system-prompt-sections
        --output-format json --json-schema <schemas/pass_a.json> --max-budget-usd <cap>

What the flags do (measured in a probe call):
- ``--restricted`` confines the file tools to the working directory: a read of the repo's
  ``CLAUDE.md`` was denied. It also ignores the user's settings, whose ``effortLevel`` is xhigh.
- ``--tools Read`` makes Read the only tool.
- Without MCP servers and skills, the input drops from the spike's ~38k tokens to ~6k.
- Effort: on render_v3's calibration, high costs $0.135 and 33 s per packet against medium's $0.121
  and 24 s. It lifts float precision from 0.986 to 1.000 and admits left/right, so it is the
  default.
- The prompt goes right after ``-p``: ``--tools`` and ``--allowedTools`` are variadic, and would
  swallow it.

The reviewer ``key`` hashes the model, the effort, the pass, the prompt, the schema and these
flags. The CLI version and the per-call cap are recorded, but not keyed.

CLI::

    PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m reference_curation.review --limit 4
    ... --max-cost 30                        # every built Pass-A packet, at high effort
    ... --effort medium                      # 12 % cheaper; one evidence class fewer (render_v3)
    ... --dry-run                            # the stub reviewer, into output/reference_curation/ledger_dry_run
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from reference_curation import ids, packets, render, verdicts

MODULE = "reference_curation.review"
SCHEMA_VERSION = 1
MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MAX_PARALLEL = 3
DRY_LEDGER_DIR = ids.OUTPUT_ROOT / "ledger_dry_run" / "verdicts"
FLAGS = ("--tools", "Read", "--allowedTools", "Read", "--restricted", "--strict-mcp-config",
         "--disable-slash-commands", "--no-session-persistence", "--exclude-dynamic-system-prompt-sections",
         "--output-format", "json")
MEMORY_FILES = ("CLAUDE.md", "CLAUDE.local.md", ".claude/CLAUDE.md")


@dataclass(frozen=True)
class Reviewer:
    """One reviewer configuration. The prompt and schema are read once, on first use, so an edit
    during a run cannot put two configurations under one key."""
    model: str = MODEL
    effort: str = "high"      # Step 4's calibration: +12 % cost over medium, and left/right admitted
    pass_: str = "A"
    cli: str = "claude"
    per_call_max_usd: float = 2.0
    timeout_s: float = 1200.0

    @functools.cached_property
    def contract(self) -> dict:
        prompt, schema = verdicts.prompt_path(self.pass_).read_bytes(), verdicts.schema_path(self.pass_).read_bytes()
        return {"prompt": prompt.decode(), "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
                "schema": json.dumps(json.loads(schema), separators=(",", ":")),
                "schema_sha256": hashlib.sha256(schema).hexdigest()}

    @functools.cached_property
    def key(self) -> str:
        return ids.sha256_json({"model": self.model, "effort": self.effort, "pass": self.pass_, "flags": FLAGS,
                                "prompt": self.contract["prompt_sha256"],
                                "schema": self.contract["schema_sha256"]})[:16]

    def argv(self) -> list[str]:
        return [self.cli, "-p", self.contract["prompt"], "--model", self.model, "--effort", self.effort, *FLAGS,
                "--json-schema", self.contract["schema"], "--max-budget-usd", f"{self.per_call_max_usd:g}"]


@functools.lru_cache(maxsize=4)
def cli_version(cli: str) -> str | None:
    try:
        return subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=60).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------- #
# Blindness
# --------------------------------------------------------------------------- #
def check_blind(entry: dict, review_root: Path = ids.REVIEW_ROOT) -> tuple[Path, dict]:
    """``(packet dir, packet.json)`` once the packet is safe to show a blind reviewer; raises
    ``ValueError`` naming what is wrong otherwise."""
    d = packets.packet_dir(entry, review_root).resolve()
    if d.is_relative_to(ids.REPO):
        raise ValueError(f"{d} is inside the repo; its CLAUDE.md would reach the reviewer")
    for folder in [d, *d.parents, Path.home() / ".claude"]:
        found = [folder / m for m in MEMORY_FILES if (folder / m).exists()]
        if found:
            raise ValueError(f"{found[0]} would reach the reviewer")
    packet = json.loads((d / "packet.json").read_text())
    if packet["pass"] != entry["pass"] or not packet["packet_id"] == entry["packet_id"] == d.name:
        raise ValueError(f"{d}: packet.json does not match the index entry")
    files = sorted(p.name for p in d.iterdir())
    if files != sorted([*entry["images"], "packet.json"]):
        raise ValueError(f"{d}: holds {files}, the index lists {sorted(entry['images'])} and packet.json")
    for name, sha in entry["images"].items():
        if hashlib.sha256((d / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"{d / name}: sha256 differs from the index")
    return d, packet


# --------------------------------------------------------------------------- #
# One call
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def call_claude(reviewer: Reviewer, packet_dir: Path) -> dict:
    """Run the reviewer CLI in ``packet_dir``: ``{envelope, returncode, stderr, seconds, started}``."""
    started, t0 = _now(), time.time()
    try:
        proc = subprocess.run(reviewer.argv(), cwd=packet_dir, capture_output=True, text=True,
                              timeout=reviewer.timeout_s)
        out, err, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        out, err, code = exc.stdout or "", f"timeout after {reviewer.timeout_s:g} s", None
        out = out.decode() if isinstance(out, bytes) else out
    try:
        envelope = json.loads(out) if out.strip() else None
    except json.JSONDecodeError:
        envelope = None
        err = f"stdout is not JSON: {out[:300]!r}; {err}"
    return {"envelope": envelope, "returncode": code, "stderr": (err or "")[-2000:],
            "seconds": round(time.time() - t0, 2), "started": started}


def stub_answer(pass_: str = "A") -> dict:
    """A schema-valid answer that abstains everywhere."""
    parts = verdicts.load_schema(pass_)["properties"]["floor"]["required"]
    return {"pose": {"description": "cannot_tell", "candidates": []},
            "floor": {p: {"avatar": "cannot_tell", "markers": "cannot_tell", "panels": []} for p in parts},
            "discrepancies": [], "body_body": {"answer": "cannot_tell", "contacts": []}, "implausible": []}


def call_stub(reviewer: Reviewer, packet_dir: Path) -> dict:
    """The dry run's reviewer: no process, no cost, the abstaining answer."""
    if not (packet_dir / "packet.json").exists():
        raise FileNotFoundError(packet_dir / "packet.json")
    envelope = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 0, "duration_ms": 0,
                "total_cost_usd": 0.0, "session_id": "stub", "permission_denials": [],
                "usage": {"input_tokens": 0, "output_tokens": 0}, "structured_output": stub_answer(reviewer.pass_)}
    return {"envelope": envelope, "returncode": 0, "stderr": "", "seconds": 0.0, "started": _now()}


def make_record(entry: dict, packet: dict, packet_dir: Path, reviewer: Reviewer, result: dict) -> dict:
    """The ledger record of one call (``n`` is set when it is written)."""
    env = result["envelope"] or {}
    errors, warnings, answer = [], [], None
    if result["returncode"] != 0 or not result["envelope"]:
        status = "error"
        errors.append(f"exit {result['returncode']}: {result['stderr'][-500:]}")
    elif env.get("is_error") or env.get("subtype") != "success":
        status = "error"
        errors.append(f"{env.get('subtype')}: {env.get('api_error_status') or env.get('result') or ''}"[:500])
    elif env.get("structured_output") is None:
        status = "invalid"
        errors.append("no structured_output")
    else:
        answer = env["structured_output"]
        errors, warnings = verdicts.validate(answer, packet, reviewer.pass_)
        status = "invalid" if errors else "valid"
    denied = [d.get("tool_input") for d in env.get("permission_denials") or []]
    warnings += [f"permission denied: {json.dumps(d)[:200]}" for d in denied]
    usage = env.get("usage") or {}
    return {
        **ids.provenance(SCHEMA_VERSION, MODULE, __file__, []),
        "packet_id": entry["packet_id"], "pass": entry["pass"], "render_v": entry["render_v"],
        "item_id": entry["item_id"], "hold_id": entry["hold_id"], "stem": entry["stem"],
        "frame_hold": entry["frame_hold"], "stratum": entry.get("stratum"), "purpose": entry.get("purpose"),
        "audit_id": entry["audit_id"],
        "reviewer": {"key": reviewer.key, "model": reviewer.model, "effort": reviewer.effort, "cli": reviewer.cli,
                     "cli_version": cli_version(reviewer.cli) if reviewer.model != "stub" else "stub",
                     "flags": list(FLAGS), "per_call_max_usd": reviewer.per_call_max_usd},
        "prompt": {"path": ids.display_path(verdicts.prompt_path(reviewer.pass_)),
                   "sha256": reviewer.contract["prompt_sha256"]},
        "schema": {"path": ids.display_path(verdicts.schema_path(reviewer.pass_)),
                   "sha256": reviewer.contract["schema_sha256"]},
        "packet": {"dir": str(packet_dir), "packet_json_sha256": ids.sha256_file(packet_dir / "packet.json"),
                   "images": dict(entry["images"])},
        "status": status, "errors": errors, "warnings": warnings, "answer": answer,
        "cost_usd": env.get("total_cost_usd"), "duration_s": result["seconds"], "num_turns": env.get("num_turns"),
        "usage": {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                  "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                  "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                  "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens")},
        "session_id": env.get("session_id"), "started_utc": result["started"],
        "returncode": result["returncode"], "stderr_tail": result["stderr"][-500:] if status != "valid" else "",
    }


# --------------------------------------------------------------------------- #
# Selection and the run
# --------------------------------------------------------------------------- #
def select_entries(index: dict, queue: list[dict], pass_: str = "A", holds=(), packet_ids=(),
                   limit: int | None = None) -> list[dict]:
    """The pass's built packets in queue priority order (items the queue lacks, such as extra
    pilot holds, go last by hold id), narrowed to ``holds`` / ``packet_ids`` and cut to ``limit``."""
    priority = {q["item_id"]: q["priority"] for q in queue if q["pass"] == pass_}
    entries = sorted((e for e in index.values() if e["pass"] == pass_),
                     key=lambda e: (priority.get(e["item_id"], 1 << 30), e["hold_id"]))
    if holds or packet_ids:
        entries = [e for e in entries if e["hold_id"] in set(holds) or e["packet_id"] in set(packet_ids)]
    return entries[:limit] if limit is not None else entries


def pending(entries: list[dict], ledger: list[dict], key: str, samples: int = 1,
            max_attempts: int = 2) -> tuple[list[dict], list[dict], int]:
    """``(calls to make, blocked entries, current entries)``: one call per missing valid sample."""
    todo, blocked, current = [], [], 0
    for e in entries:
        mine = [r for r in ledger if r["packet_id"] == e["packet_id"] and r["pass"] == e["pass"]
                and r["reviewer"]["key"] == key]
        valid = sum(r["status"] == "valid" for r in mine)
        if valid >= samples:
            current += 1
        elif len(mine) - valid >= max_attempts:
            blocked.append(e)
        else:
            todo += [e] * (samples - valid)
    return todo, blocked, current


def run(entries: list[dict], reviewer: Reviewer, call=call_claude, *, ledger_dir: Path = verdicts.LEDGER_DIR,
        review_root: Path = ids.REVIEW_ROOT, samples: int = 1, max_attempts: int = 2, max_cost: float = 30.0,
        est_cost: float = 0.5, parallel: int = MAX_PARALLEL, log=functools.partial(print, flush=True)) -> dict:
    """Review the pending ``entries``. Returns the counts, the spend and the failures."""
    todo, blocked, current = pending(entries, verdicts.read_ledger(ledger_dir), reviewer.key, samples, max_attempts)
    parallel = max(1, min(parallel, MAX_PARALLEL))
    out = {"pending": len(todo), "current": current, "blocked": [e["item_id"] for e in blocked],
           "status": {}, "spent": 0.0, "failures": [], "unstarted": 0}
    costs = []

    def one(entry):
        d, packet = check_blind(entry, review_root)
        record = make_record(entry, packet, d, reviewer, call(reviewer, d))
        return record, verdicts.write_verdict(record, ledger_dir)

    start, done_n = time.time(), 0
    with ThreadPoolExecutor(parallel) as pool:
        running = {}
        while todo or running:
            est = max(est_cost, max(costs, default=0.0))
            while todo and len(running) < parallel and out["spent"] + (len(running) + 1) * est <= max_cost:
                entry = todo.pop(0)
                running[pool.submit(one, entry)] = entry
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                entry = running.pop(fut)
                done_n += 1
                try:
                    record, path = fut.result()
                except Exception as exc:  # noqa: BLE001 -- a packet that is not blind is never sent
                    out["failures"].append(f"{entry['item_id']}: {type(exc).__name__}: {exc}")
                    log(f"[{done_n}] FAILED {entry['packet_id'][:12]}: {exc}")
                    continue
                cost = record["cost_usd"] or 0.0
                out["spent"] += cost
                costs.append(cost)
                out["status"][record["status"]] = out["status"].get(record["status"], 0) + 1
                if record["status"] != "valid":
                    out["failures"].append(f"{entry['item_id']}: {record['status']}: {'; '.join(record['errors'])[:300]}")
                log(f"[{done_n}] {record['status']:7s} {entry['packet_id'][:12]} {entry.get('stratum') or '-':11s} "
                    f"${cost:.3f} {record['duration_s']:.0f} s -> {path.name} (total ${out['spent']:.2f}, "
                    f"{time.time() - start:.0f} s)")
    out["unstarted"] = len(todo)
    out["spent"] = round(out["spent"], 4)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pass", dest="pass_", choices=verdicts.PASSES, default="A")
    ap.add_argument("--effort", choices=EFFORTS, default="high")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--hold", nargs="*", default=[], help="only these holds")
    ap.add_argument("--packet", nargs="*", default=[], help="only these packet ids")
    ap.add_argument("--limit", type=int, help="the first N packets in queue priority order")
    ap.add_argument("--samples", type=int, default=1, help="valid verdicts wanted per packet")
    ap.add_argument("--max-attempts", type=int, default=2, help="failed calls before a packet is left alone")
    ap.add_argument("--max-cost", type=float, default=30.0, help="USD for the whole run")
    ap.add_argument("--per-call-max", type=float, default=2.0, help="USD cap of one call")
    ap.add_argument("--est-cost", type=float, default=0.5, help="USD a call is expected to cost before any has run")
    ap.add_argument("--parallel", type=int, default=MAX_PARALLEL)
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--cli", default="claude")
    ap.add_argument("--dry-run", action="store_true", help="the abstaining stub instead of the CLI; its own ledger")
    ap.add_argument("--ledger-dir", type=Path)
    ap.add_argument("--audit", type=Path, help="the audit whose queue orders the packets (default: the newest)")
    ap.add_argument("--index-root", type=Path, default=packets.INDEX_ROOT)
    ap.add_argument("--review-root", type=Path, default=ids.REVIEW_ROOT)
    args = ap.parse_args(argv)

    model = "stub" if args.dry_run else args.model
    reviewer = Reviewer(model=model, effort=args.effort, pass_=args.pass_, cli=args.cli,
                        per_call_max_usd=args.per_call_max, timeout_s=args.timeout)
    ledger_dir = args.ledger_dir or (DRY_LEDGER_DIR if args.dry_run else verdicts.LEDGER_DIR)
    try:
        aud = packets.load_audit(args.audit or packets.default_audit_dir())
        index = packets.read_index(packets.index_path(render.RENDER_V, args.index_root))
        entries = select_entries(index, aud.queue, args.pass_, args.hold, args.packet, args.limit)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not entries:
        print(f"FAILED no built pass-{args.pass_} packets selected in {args.index_root}", file=sys.stderr)
        return 1
    if not args.dry_run and not cli_version(args.cli):
        print(f"FAILED the reviewer CLI {args.cli!r} does not run", file=sys.stderr)
        return 1
    out = run(entries, reviewer, call_stub if args.dry_run else call_claude, ledger_dir=ledger_dir,
              review_root=args.review_root, samples=args.samples, max_attempts=args.max_attempts,
              max_cost=args.max_cost, est_cost=args.est_cost, parallel=args.parallel)
    for f in out["failures"]:
        print(f"FAILED {f}", file=sys.stderr)
    for b in out["blocked"]:
        print(f"BLOCKED {b}: {args.max_attempts} failed calls; see the ledger", file=sys.stderr)
    print(f"review pass {args.pass_} {render.RENDER_V} {model} effort {args.effort} key {reviewer.key}: "
          f"{len(entries)} packets, {out['current']} current, {out['pending']} calls pending, "
          f"{sum(out['status'].values())} made {out['status']}, {out['unstarted']} left by the ${args.max_cost:g} budget, "
          f"{len(out['blocked'])} blocked; ${out['spent']:.2f} -> {ids.display_path(ledger_dir)}")
    if out["failures"] or out["blocked"]:
        return 1
    return 2 if out["unstarted"] else 0


if __name__ == "__main__":
    sys.exit(main())
