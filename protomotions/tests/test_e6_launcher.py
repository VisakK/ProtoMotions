# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Card E6 (graph-growth PLAN.MD): ``data/scripts/run_expert_graph_ft.sh``'s checks, under ``DRY_RUN=1``.

The launcher runs from a throw-away mirror of the repo (the script copied, ``protomotions/`` and ``data/smpl/``
linked, a scratch ``results/`` and releases directory), so no test writes under the real ``results/`` or
``data/reference_curation/releases/``. Release v2 is real; the "v3" releases reuse v2's graph, tables and G1's plans
over a stub package whose ``motion_files`` add two SYN_ variants (x0/x3s/x7s), one a prefix of the other's name.
Nothing here starts ``train_agent.py``. Skipped when release v2 is not on disk.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
V2 = "holds_repaired_ftC_posefix.release_v2.a2dda5d2ac"
V2_RECORD = REPO / "data/reference_curation/releases" / f"{V2}.json"
LAUNCHER = REPO / "data/scripts/run_expert_graph_ft.sh"
SYN = ["SYN_E1_press_high_s0_t6px", "SYN_E1_press_high_s0_t6px12"]
AMP_EXPERIMENT = "examples/experiments/mimic/mlp_goal_conditioned_amp.py"
LAUNCHER_ENV = ("SMOKE", "TIMING", "CONFIG_ONLY", "RESUME", "DRY_RUN", "WANDB", "RELEASE", "EXPERIMENT",
                "CHECKPOINT", "AMP_LINEAGE", "AMP_DEMO_EXCLUDE", "EVAL_MAX_STEPS", "SMOKE_EPOCHS", "TIMING_EPOCHS",
                "MAX_EPOCHS", "NUM_ENVS", "EXTRA", "PY")

pytestmark = pytest.mark.skipif(not V2_RECORD.is_file(), reason="release v2 is not on disk")


def _fake_release(root: Path, name: str, rec_v2: dict, edit=None, extra_plans=None) -> str:
    """A v3-shaped release: v2's artifacts with a stub package that adds the SYN_ variants, and its own plans/."""
    d = root / "data" / name
    (d / "plans").mkdir(parents=True)
    stems = list(rec_v2["motions"]) + [f"{s}{x}" for s in SYN for x in ("", "_x3s", "_x7s")]
    torch.save({"motion_files": [f"motions/{s}.motion" for s in stems]}, d / "motions.pt")
    v2_plans = REPO / rec_v2["dir"] / "plans"
    for p in v2_plans.glob("*.json"):
        (d / "plans" / p.name).symlink_to(p)
    base = json.loads((v2_plans / "fork_Crane_Crow_Pose_or_Bakasana.json").read_text())
    edge = dict(base, start={"clip": SYN[0], "time": 0.0},
                goals=[dict(g, pose_clip=SYN[0]) for g in base["goals"]])
    (d / "plans" / "edge_E1.json").write_text(json.dumps(edge))
    (d / "plans" / "nohijack_E1.json").write_text(json.dumps(dict(edge, goals=edge["goals"][:1])))
    (d / "plans" / "fork_edge_E1.json").write_text(json.dumps(edge))          # the funnel's, not the panel's
    for fname, plan in (extra_plans or {}).items():
        (d / "plans" / fname).write_text(json.dumps(plan))
    rec = json.loads(json.dumps(rec_v2))
    rec["artifacts"]["package"]["path"] = f"data/{name}/motions.pt"
    rec["dir"] = f"data/{name}"
    rec["purpose"] = "candidate"
    rec["motions"] = {s: "0" * 64 for s in stems}
    if edit:
        edit(rec)
    (root / "data/reference_curation/releases" / f"{name}.json").write_text(json.dumps(rec))
    return name


@pytest.fixture(scope="module")
def mirror(tmp_path_factory):
    root = tmp_path_factory.mktemp("e6_launcher")
    (root / "data/scripts").mkdir(parents=True)
    shutil.copy(LAUNCHER, root / "data/scripts/run_expert_graph_ft.sh")
    (root / "protomotions").symlink_to(REPO / "protomotions")
    (root / "data/smpl").symlink_to(REPO / "data/smpl")
    (root / "data/reference_curation/releases").mkdir(parents=True)
    (root / "data/reference_curation/releases" / f"{V2}.json").symlink_to(V2_RECORD)
    (root / "results").mkdir()
    (root / "ckpt.ckpt").write_bytes(b"")
    rec_v2 = json.loads(V2_RECORD.read_text())
    bad_clip = json.loads((REPO / rec_v2["dir"] / "plans/fork_Tree_Pose_or_Vrksasana.json").read_text())
    capped = json.loads(json.dumps(bad_clip))
    bad_clip["goals"][0]["pose_clip"] = "SYN_E9_no_such_clip"
    capped["goals"][-1]["reach_s"] = 30.0
    names = {
        "v3": _fake_release(root, "zz_v3", rec_v2),
        "bad_plan": _fake_release(root, "zz_v3_bad_plan", rec_v2, extra_plans={"edge_E9.json": bad_clip}),
        "capped": _fake_release(root, "zz_v3_capped", rec_v2, extra_plans={"nohijack_E9.json": capped}),
        "odd_record": _fake_release(root, "zz_v3_odd", rec_v2, edit=lambda r: (
            r.update(synthetic="holds_repaired_ftC_posefix.synthetic_v3.abcdef0123"), r.pop("motions"),
            r["training"].update(eval_max_steps=2400.0))),
        "wrong_motions": _fake_release(root, "zz_v3_wrong", rec_v2, edit=lambda r: r["motions"].pop(SYN[0])),
    }
    return root, names


def _run(root, **env_vars):
    env = {k: v for k, v in os.environ.items() if k not in LAUNCHER_ENV}
    env.update(PY=sys.executable, OMP_NUM_THREADS="1", DRY_RUN="1", CHECKPOINT="ckpt.ckpt")
    env.update({k: str(v) for k, v in env_vars.items()})
    p = subprocess.run(["bash", str(root / "data/scripts/run_expert_graph_ft.sh")], env=env, cwd=root,
                       capture_output=True, text=True, timeout=180)
    return p.returncode, p.stdout + p.stderr


def _command(out):
    return next(line for line in out.splitlines() if "command        :" in line)


def _resume_dir(root, name, **config):
    d = root / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "last.ckpt").write_bytes(b"")
    (d / "config.yaml").write_text(json.dumps(config))
    return d


def test_required_and_malformed_inputs_refuse_before_anything_is_read(mirror):
    root, _ = mirror
    assert _run(root, RELEASE=V2)[0] == 1
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_a", CHECKPOINT="")
    assert rc == 1 and "CHECKPOINT is required" in out
    for bad in ("..", ".", "-x", "a/b"):
        assert _run(root, RELEASE=V2, EXPERIMENT=bad)[0] == 1, bad
    rc, out = _run(root, RELEASE=f"../../releases/{V2}", EXPERIMENT="zz_a")
    assert rc == 1 and "must be a release id" in out
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_a", SMOKE="true")
    assert rc == 1 and "SMOKE must be 0 or 1" in out
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_a", SMOKE=1, TIMING=1)
    assert rc == 1 and "exclusive" in out
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_a", MAX_EPOCHS="6e3")
    assert rc == 1 and "MAX_EPOCHS must be a positive integer" in out


def test_an_existing_last_ckpt_refuses_on_the_name_actually_used(mirror):
    root, _ = mirror
    (root / "results/zz_used").mkdir()
    (root / "results/zz_used/last.ckpt").write_bytes(b"")
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_used")
    assert rc == 1 and "results/zz_used/last.ckpt exists" in out
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_used", CONFIG_ONLY=1)
    assert rc == 1
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_used", SMOKE=1)          # the smoke's own name is free
    assert rc == 0 and "-> results/zz_used_smoke" in out
    (root / "results/zz_used_smoke").mkdir()
    (root / "results/zz_used_smoke/last.ckpt").symlink_to(root / "results/nowhere.ckpt")   # dangling counts
    assert _run(root, RELEASE=V2, EXPERIMENT="zz_used", SMOKE=1)[0] == 1


def test_a_live_run_s_name_is_refused_in_both_argument_forms(mirror):
    root, _ = mirror
    for form in (["--experiment-name", "zz_live"], ["--experiment-name=zz_live"]):
        dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "train_agent.py", *form])
        try:
            time.sleep(0.3)
            rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_live")
            assert rc == 1 and "a live train_agent.py is writing to results/zz_live" in out, form
        finally:
            dummy.kill()
            dummy.wait()


def test_resume_accepts_only_this_launcher_s_run_on_this_library(mirror):
    root, _ = mirror
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_nothing", RESUME=1)
    assert rc == 1 and "train_agent would start a fresh run from scratch" in out
    v2_motions = json.loads(V2_RECORD.read_text())["artifacts"]["package"]["path"]
    _resume_dir(root, "zz_e15500_like", experiment_path="examples/experiments/mimic/mlp_goal_conditioned.py",
                motion_file=v2_motions)
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_e15500_like", RESUME=1)
    assert rc == 1 and "was launched with examples/experiments/mimic/mlp_goal_conditioned.py" in out
    _resume_dir(root, "zz_g1_like", experiment_path=AMP_EXPERIMENT, motion_file=v2_motions)
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_g1_like", RESUME=1)
    assert rc == 1 and "has no amp_lineage_weights" in out
    _resume_dir(root, "zz_g3_other_lib", experiment_path=AMP_EXPERIMENT, motion_file="data/x/motions.pt",
                amp_lineage_weights=["SYN_=0.5"])
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_g3_other_lib", RESUME=1)
    assert rc == 1 and "a resume keeps its own library" in out
    _resume_dir(root, "zz_g3", experiment_path=AMP_EXPERIMENT, motion_file=v2_motions, amp_lineage_weights=[])
    assert _run(root, RELEASE=V2, EXPERIMENT="zz_g3", RESUME=1, CONFIG_ONLY=1)[0] == 1
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_g3", RESUME=1)
    assert rc == 0, out
    assert "--checkpoint" not in _command(out) and "--warm-start-optimization-state" not in _command(out)


def test_release_v2_default_command(mirror):
    root, _ = mirror
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_g3_v2")
    assert rc == 0, out
    cmd = _command(out)
    for flag in ("--checkpoint ckpt.ckpt --warm-start-optimization-state",
                 "--segment-start-prob 0.4 --segment-end-prob 0.2 --segment-pre-roll-s 0.5 --motion-prior package "
                 "--freeze-obs-normalizers True",
                 "--amp-reward-w 0.5 --amp-w-start-epoch 0 --amp-w-full-epoch 0 --amp-calibrate-style-ratio 0 "
                 "--amp-disc-batch-size 4096 --amp-grad-penalty 10.0 --amp-lineage-weights SYN_=0.5 "
                 "--amp-demo-exclude-motions Scorpion_pose_or_vrischikasana-b SYN_",
                 "--support-rule v2", "--eval-every 500 --eval-max-steps 2250 --save-every 500",
                 "--viz-sequences-every 500 --viz-num-sequences 12 --viz-max-seconds 24.0",
                 "--num-envs 4096 --batch-size 16384 --training-max-steps 786432000",
                 "--use-wandb --overrides env.ref_respawn_offset=0.005"):
        assert flag in cmd, flag
    assert "purpose n/a; 0 synthetic variants: n/a" in out
    assert "SYN_=0.5 -> 0 of 168 motions (WARNING: none)" in out           # no SYN_ on v2: a note, not a refusal
    assert "10 plans (10 G1, 0 edge, 0 no-hijack; all resolve)" in out


def test_smoke_and_timing_modes(mirror):
    root, _ = mirror
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_e6_smoke_v2", SMOKE=1, SMOKE_EPOCHS=10,
                   AMP_LINEAGE="Crane_Crow=0.5")
    assert rc == 0, out
    cmd = _command(out)
    assert "--experiment-name zz_e6_smoke_v2_smoke" in cmd and "--use-wandb" not in cmd
    assert "--eval-every 2 --eval-max-steps 150 --save-every 2 --viz-sequences-every 0" in cmd
    assert "--amp-disc-batch-size 1024" in cmd and "--num-envs 256 --batch-size 1024 --training-max-steps 81920" in cmd
    assert "Crane_Crow=0.5 -> 12 of 168 motions" in out
    rc, out = _run(root, RELEASE=V2, EXPERIMENT="zz_t", TIMING=1)
    assert rc == 0, out
    # TIMING keeps the real evaluation length: its one evaluation is the memory peak checklist item 6 reads
    assert "--eval-every 100000 --eval-max-steps 2250 --save-every 100000" in _command(out)
    assert _run(root, RELEASE=V2, EXPERIMENT="zz_t", TIMING=1, EVAL_MAX_STEPS=1000)[0] == 1


def test_v3_shaped_release(mirror):
    root, names = mirror
    rc, out = _run(root, RELEASE=names["v3"], EXPERIMENT="zz_g3_v3")
    assert rc == 0, out
    assert f"purpose candidate; 2 synthetic variants: {SYN[0]}, {SYN[1]}" in out
    assert "(174 motions, 6 SYN_)" in out and "SYN_=0.5 -> 6 of 174 motions" in out
    # edge_ and nohijack_ join the panel; fork_edge_ (the funnel battery's) does not
    assert "12 plans (10 G1, 1 edge, 1 no-hijack; all resolve) -> --viz-num-sequences 14" in out
    assert "edge_E1.json" in _command(out) and "fork_edge_E1.json" not in _command(out)


@pytest.mark.parametrize("demo_exclude, expected", [
    ("Scorpion_pose_or_vrischikasana-b", f"would make synthetic clips demonstrations: {SYN[0]} {SYN[1]}"),
    ("", "would make synthetic clips demonstrations"),
    ("SYN_", "drops G1's exclusion [Scorpion_pose_or_vrischikasana-b]"),
])
def test_v3_demonstration_guards(mirror, demo_exclude, expected):
    root, names = mirror
    rc, out = _run(root, RELEASE=names["v3"], EXPERIMENT="zz_g3_v3", AMP_DEMO_EXCLUDE=demo_exclude)
    assert rc == 1 and expected in out, out


@pytest.mark.parametrize("lineage, expected", [
    ("SYN_", "expected PATTERN=W"),                                     # what train_agent's argparse would reject
    ("SYN_*=0.5", "['SYN_*'] match no motion"),                         # a glob is not a substring
    ("SYN_=0.5 SYN_E1=0.25", "['SYN_E1'] match no motion"),            # shadowed: first match wins
    (f"{SYN[1]}=0.25", f"leaves 3 SYN_ motions at style weight 1.0"),  # t6px12 does not cover t6px
    ("", "leaves 6 SYN_ motions at style weight 1.0"),
])
def test_v3_lineage_guards(mirror, lineage, expected):
    root, names = mirror
    rc, out = _run(root, RELEASE=names["v3"], EXPERIMENT="zz_g3_v3", AMP_LINEAGE=lineage)
    assert rc == 1 and expected in out, out


def test_v3_lineage_echo_is_first_match_wins(mirror):
    root, names = mirror
    rc, out = _run(root, RELEASE=names["v3"], EXPERIMENT="zz_g3_v3",
                   AMP_LINEAGE=f"{SYN[1]}=0.25 SYN_=0.5")
    assert rc == 0, out
    # counted independently, SYN_ would read 6
    assert f"{SYN[1]}=0.25 -> 3 of 174 motions; SYN_=0.5 -> 3 of 174 motions" in out


def test_panel_plans_that_do_not_resolve_fail_the_launch(mirror):
    root, names = mirror
    rc, out = _run(root, RELEASE=names["bad_plan"], EXPERIMENT="zz_g3_v3")
    assert rc == 1 and "edge_E9.json does not resolve" in out and "does not resolve against this graph" in out, out
    rc, out = _run(root, RELEASE=names["capped"], EXPERIMENT="zz_g3_v3")
    assert rc == 1 and "nohijack_E9.json: --viz-max-seconds 24.0 keeps" in out, out


def test_record_shapes(mirror):
    root, names = mirror
    # 'synthetic' as an id string and no 'motions' key: the stems come from the package, nothing crashes
    rc, out = _run(root, RELEASE=names["odd_record"], EXPERIMENT="zz_g3_v3")
    assert rc == 0, out
    assert "2 synthetic variants" in out and "SYN_=0.5 -> 6 of 174 motions" in out
    assert "over 2400 steps (record: >= 2400)" in out                     # a float eval_max_steps is honoured
    assert _run(root, RELEASE=names["odd_record"], EXPERIMENT="zz_g3_v3", EVAL_MAX_STEPS=2250)[0] == 1
    rc, out = _run(root, RELEASE=names["wrong_motions"], EXPERIMENT="zz_g3_v3")
    assert rc == 1 and "are not the package's 174 motion_files" in out
