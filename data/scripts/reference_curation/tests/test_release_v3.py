# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Card R3 of ``expert_revist/graph_growth_2026_10_03/PLAN.MD``: release v3 (``release_v3``, ``contact_targets_v3``,
``physics_tables_v3``, ``make_edge_probe_plans``).

Pure rules first (the drop list by exact stem, the purposes, D5's weights, the padding rule, the tables wrapper, the
generator's phase rule, the plan-clip rule), then the committed candidate release: the human slice equals release v2
(with a negative control per artifact), the node keys equal v2's (with a builder-level negative control), a perturbed
real frame fails the lineage check, the inherited rows, the record as training reads it, and ``--check``.

    OMP_NUM_THREADS=1 PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest \\
        data/scripts/reference_curation/tests/test_release_v3.py -q
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import make_edge_probe_plans as mep
from reference_curation import contact_targets_v3 as ct3
from reference_curation import ids
from reference_curation import physics_tables_v3 as tv3
from reference_curation import release_v3 as R
from reference_curation import splice_v3 as S3

# The committed candidate (expert_revist/graph_growth_2026_10_03/r3_release/README.MD); never "the newest folder".
RELEASE_ID = "holds_repaired_ftC_posefix.release_v3.2f132f4299"
HAVE_INPUTS = R.synthetic_record_path().exists() and R.v2_record_path().exists()


def _order():
    return json.loads(R.synthetic_record_path().read_text())["order"]


# --------------------------------------------------------------------------- #
# Pure rules
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_INPUTS, reason="the synthetic record is not on disk")
def test_the_drop_list_matches_stems_exactly():
    order = _order()
    short, long_ = "SYN_E2_jumpback_mid_s1_t6px", "SYN_E2_jumpback_mid_s1_t6px12"
    assert short in long_ and short in order and long_ in order          # one stem is a substring of another
    kept = R.apply_drop(order, [short])
    assert short not in kept and long_ in kept and len(kept) == len(order) - 1
    kept = R.apply_drop(order, list(R.USER_DROP))
    assert len(kept) == 28 and "SYN_E2_jumpback_mid_s3_t6px12" in kept           # its t6px12 twin stays
    with pytest.raises(ValueError, match="exact match"):
        R.apply_drop(order, ["SYN_E2_jumpback_mid_s1"])                          # a prefix is not a stem
    with pytest.raises(ValueError, match="repeats"):
        R.apply_drop(order, [short, short])


def test_the_purposes_keep_the_users_drop():
    R.check_purpose("candidate", list(R.USER_DROP))
    with pytest.raises(ValueError, match="pre-G3 drop"):
        R.check_purpose("candidate", [])
    with pytest.raises(ValueError, match="G2's failures"):
        R.check_purpose("final", list(R.USER_DROP))                              # nothing beyond the user's drop
    R.check_purpose("final", [*R.USER_DROP, "SYN_E5_floatdown_mid_s0_t6rpx12"])
    with pytest.raises(ValueError):
        R.check_purpose("release", list(R.USER_DROP))


def test_d5_gives_each_edge_three_human_clips_of_mass():
    edges = {"E1": ["a"] * 8, "E3": ["b"] * 3, "B1": ["c"] * 5, "E2": ["d"] * 11, "E5": ["e"]}
    w = R.edge_weights(edges)
    assert w == {"E1": 0.375, "E3": 1.0, "B1": 0.6, "E2": 3 / 11, "E5": 3.0}
    clips = [{"stem": "h"}] * 168 + [{"synthetic": {"edge": e}} for e, v in edges.items() for _ in v for _ in range(3)]
    m = np.asarray(R.motion_weights(clips, w))
    assert m[:168].tolist() == [1.0] * 168
    per_edge = {e: sum(w[e] for c in clips[168:] if c["synthetic"]["edge"] == e) for e in edges}
    assert all(abs(v - 9.0) < 1e-12 for v in per_edge.values())                 # three clips' x0/x3s/x7s each
    assert abs(m[168:].sum() / m.sum() - 45 / 213) < 1e-12


def test_rows_equal_is_exact_after_padding():
    v2 = torch.tensor([[1.0, 2.0], [3.0, float("nan")]])
    v3 = torch.full((3, 4), float("inf"))
    v3[:2, :2] = v2
    assert R.rows_equal(v3, v2, 2, float("inf"))
    bad = v3.clone()
    bad[1, 0] = 3.0 + 1e-6
    assert not R.rows_equal(bad, v2, 2, float("inf"))                           # a value inside v2's extent
    bad = v3.clone()
    bad[0, 3] = 0.0
    assert not R.rows_equal(bad, v2, 2, float("inf"))                           # padding that is not padding
    assert not R.rows_equal(v3.double(), v2.double().float(), 2, float("inf"))  # dtype
    b2 = torch.tensor([[True, False]])
    b3 = torch.zeros(1, 3, dtype=torch.bool)
    b3[0, 0] = True
    assert R.rows_equal(b3, b2, 1, False) and not R.rows_equal(~b3, b2, 1, False)


def test_the_tables_wrapper_strips_only_the_synthetic_motions(tmp_path):
    full = {"rigid_body_pos": torch.zeros(2, 1, 3), "ground_reaction": torch.zeros(2, 3),
            "rigid_body_ground_forces": torch.zeros(2, 1, 3), "ground_reaction_valid": torch.zeros(2, 3)}
    torch.save(full, tmp_path / "SYN_a.motion")
    torch.save(full, tmp_path / "220923_b.motion")
    proxy = tv3.NoPressureTorch(torch, [tmp_path / "SYN_a.motion"])
    a, b = proxy.load(tmp_path / "SYN_a.motion"), proxy.load(str(tmp_path / "220923_b.motion"))
    assert "ground_reaction" not in a and "rigid_body_pos" in a and "ground_reaction" in b
    assert proxy.stripped == ["SYN_a"] and proxy.zeros is torch.zeros               # everything else is torch
    assert tv3.synthetic_stems({"clips": [{"stem": "SYN_a", "synthetic": {}}, {"stem": "b"}]}) == ["SYN_a"]
    with pytest.raises(ValueError, match="disagree"):
        tv3.synthetic_stems({"clips": [{"stem": "SYN_a"}]})                         # a SYN_ stem without its block
    with pytest.raises(ValueError, match="disagree"):
        tv3.synthetic_stems({"clips": [{"stem": "b", "synthetic": {}}]})


@pytest.mark.skipif(not HAVE_INPUTS, reason="the synthetic record is not on disk")
def test_the_phase_rule_is_the_generators():
    sketch = pytest.importorskip("edge_synthesis.sketch")
    edges = json.loads((ids.REPO / "expert_revist/graph_growth_2026_10_03/edges.json").read_text())
    syn = json.loads(R.synthetic_record_path().read_text())
    seen = set()
    for stem in syn["order"]:
        entry = yaml.safe_load(open(ids.REPO / syn["clips"][stem]["holds"]["path"]))["synthetic"]
        with np.load(ids.REPO / syn["clips"][stem]["lineage"]["path"]) as z:
            t = z["variant_t"][~np.isnan(z["variant_t"])]
        e = ct3.edge_spec(edges, entry["edge"])
        sch = sketch.schedule(e, entry["durations_s"])
        mine = ct3.phase_at(t, entry["durations_s"])
        assert np.array_equal(mine, sch.phase_at(t))
        for p in sorted(set(mine.tolist())):
            ground, braces = ct3.planned_config(e, p)
            ti = float(t[mine == p][0])
            g2, b2, _ = sch.config_at(ti)
            assert (set(ground), set(braces)) == (g2, b2)
            seen.add((entry["edge"], p))
    assert {p for e, p in seen if e == "E1"} == {-1, 0, 1, 2}                  # settle, both phases, the hold at D


def test_the_plan_clip_is_the_slowest_then_the_most_accurate():
    def clip(stem, edge, T, d6, variant_s=0.0):
        return {"stem": stem, "variant_s": variant_s,
                "synthetic": {"edge": edge, "T": T, "admission": {"row": {"endpoints": {"final_d6_m": d6}}}}}
    clips = [clip("a", "E1", 2.95, 0.010), clip("b", "E1", 4.5, 0.050), clip("c", "E1", 4.5, 0.031),
             clip("c_x3s", "E1", 4.5, 0.001, 3.0), clip("d", "E2", 1.075, 0.08), clip("e", "E2", 1.075, 0.08),
             {"stem": "220923_x", "variant_s": 0.0}]
    picks = {e: c["stem"] for e, c in mep.plan_clips(clips).items()}
    assert picks == {"E1": "c", "E2": "d"}                                       # ties keep the release's order


# --------------------------------------------------------------------------- #
# The committed candidate
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def release():
    path = R.RECORD_ROOT / f"{RELEASE_ID}.json"
    if not path.exists() or not (ids.REPO / json.loads(path.read_text())["dir"]).exists():
        pytest.skip("the release is not on disk")
    rec = json.loads(path.read_text())
    out = ids.REPO / rec["dir"]
    inp = R.pinned_inputs(rec["drop"], rec["purpose"])
    load = R.load
    v2 = inp["v2_out"]
    return SimpleNamespace(rec=rec, out=out, inp=inp, H=len(inp["v2"]["motions"]),
                           g=load(out / "contact_graph.pt"), g2=load(v2 / "contact_graph.pt"),
                           t=load(out / "physics_tables.pt"), t2=load(v2 / "physics_tables.pt"),
                           s=load(out / "contact_targets.pt"), s2=load(v2 / "contact_targets.pt"),
                           ext=yaml.safe_load(open(out / "holds_extended.yaml")))


def test_the_human_slice_is_release_v2s(release):
    r = release
    pk = R.load(r.out / "motions.pt")
    pk2 = torch.load(r.inp["v2_out"] / "motions.pt", map_location="cpu", weights_only=False, mmap=True)
    assert R.human_slice_package(pk, pk2, r.H) == []
    assert R.human_slice_graph(r.g, r.g2, r.H) == []
    assert R.human_slice_tables(r.t, r.t2, r.H) == []
    assert R.human_slice_sidecar(r.s, r.s2, r.H) == []
    # negative controls: one value in one human row of each artifact
    pk["gts"][1000, 3, 2] += 1e-6
    assert R.human_slice_package(pk, pk2, r.H) == ["gts"]
    g = dict(r.g, seg_start=r.g["seg_start"].clone())
    g["seg_start"][5, 1] += 1e-3
    assert R.human_slice_graph(g, r.g2, r.H) == ["seg_start"]
    t = dict(r.t, swing=r.t["swing"].clone())
    t["swing"][7, 100, 3] = ~t["swing"][7, 100, 3]
    assert R.human_slice_tables(t, r.t2, r.H) == ["swing"]
    s = dict(r.s, seg_role=r.s["seg_role"].clone())
    s["seg_role"][9, 0, 0] = 3
    assert R.human_slice_sidecar(s, r.s2, r.H) == ["seg_role"]
    # and the synthetic rows are not part of the slice: a change there is not the human slice's
    s = dict(r.s, frame_pair_ok=r.s["frame_pair_ok"].clone())
    s["frame_pair_ok"][r.H, 0, 0] = ~s["frame_pair_ok"][r.H, 0, 0]
    assert R.human_slice_sidecar(s, r.s2, r.H) == []


def test_the_node_keys_are_release_v2s(release):
    from build_hold_graph_v2 import build_graph_v2
    from extract_contact_configs import ORIENT_BINS, zone_pairs

    assert list(release.g["node_keys"]) == list(release.g2["node_keys"]) and len(release.g["node_keys"]) == 123
    assert list(release.g["pair_names"]) == list(release.g2["pair_names"])
    assert list(release.g["orientation_names"]) == list(release.g2["orientation_names"])
    # negative control: one synthetic hold renamed is a new configuration, and graph v2 then renumbers the nodes
    clips = json.loads(json.dumps(release.ext["clips"]))
    names = [c["stem"] for c in clips]
    frames = [c["num_frames"] for c in clips]
    syn = next(c for c in clips if c["stem"] == release.rec["variants"][0])
    hold = next(h for h in syn["holds"] if h.get("extend"))
    for c in clips:                                   # every duration variant of that hold (one hold id)
        for h in c["holds"]:
            if h["hold_id"] == hold["hold_id"]:
                h["name"] = "Crane_Crow_Pose_or_Bakasana_renamed"
    payload, _ = build_graph_v2(clips, names, list(zone_pairs()), list(ORIENT_BINS), motion_num_frames=frames)
    assert list(payload["node_keys"]) != list(release.g2["node_keys"]) and len(payload["node_keys"]) == 124


def test_a_perturbed_real_frame_fails_the_lineage_check(release):
    syn = release.inp["syn"]
    stem = release.rec["variants"][0]
    mot = R.load(release.out / "motions" / f"{stem}.motion")
    with np.load(ids.REPO / syn["clips"][stem]["lineage"]["path"]) as z:
        lin = {k: z[k] for k in z.files}
    info = json.loads((ids.REPO / syn["clips"][stem]["json"]["path"]).read_text())
    rel = S3.release_v2()
    sources = {s: S3.release_motion(rel, s) for s in (info["S"]["stem"], info["D"]["stem"])}
    assert S3.check_real_frames(mot, lin, sources)["pass"]
    real = int(np.nonzero(lin["kind"] == S3.KIND["real"])[0][40])
    bad = dict(mot, rigid_body_pos=mot["rigid_body_pos"].clone())
    bad["rigid_body_pos"][real, 5, 0] += 1e-4                                 # 0.1 mm on one body of one real frame
    assert not S3.check_real_frames(bad, lin, sources)["pass"]
    # and a duration variant that is not its x0 spliced at S
    entry = release.inp["entries"][stem]
    name = f"{stem}_x3s"
    e = next(c for c in release.ext["clips"] if c["stem"] == name)
    lineage = np.load(release.out / "lineage.npz")
    lin_v = {k: lineage[f"{name}.{k}"] for k in ("index", "inserted", "stems", *R.SYN_LINEAGE)}
    m = R.load(release.out / "motions" / f"{name}.motion")
    sha = release.inp["plant"]["plant_sha256"]
    assert R.synthetic_variant_problems(m, mot, entry, 3.0, e, lin_v, lin, sha) == []
    m["dof_pos"][real + 2, 0] += 1e-3
    assert R.synthetic_variant_problems(m, mot, entry, 3.0, e, lin_v, lin, sha) == ["splice"]


def test_every_synthetic_segment_carries_its_inherited_row(release):
    r = release
    assert R.inherited_row_problems(r.s, r.g, r.ext["clips"], r.H) == []
    s = dict(r.s, seg_ground_free=r.s["seg_ground_free"].clone())
    m = r.H + 1
    s["seg_ground_free"][m, 0, 0] = ~s["seg_ground_free"][m, 0, 0]
    hid = r.g["hold_ids"][int(r.g["seg_hold_index"][m, 0])]
    assert R.inherited_row_problems(s, r.g, r.ext["clips"], r.H) == [hid]


def test_the_synthetic_sidecar_frames(release):
    summary = json.loads((release.out / "contact_targets.json").read_text())
    assert summary["known_free_conflicts"] == 0
    frames = summary["synthetic_frames"]
    assert len(frames) == len(release.rec["variants"])
    assert max(f["real_frame_gap_vs_capture_m"] for f in frames.values()) < 1e-6   # the kernel, on placed frames
    # the crow source's braces are planned on its 17 settle frames, the press plans none after the departure
    e1 = frames["SYN_E1_press_high_s0_t6px"]
    assert e1["phases"]["-1"] == 17 and e1["nonreal_planned"] == {"L_SHANK+L_UPPER_ARM": 17, "R_SHANK+R_UPPER_ARM": 17}
    assert summary["human_x0"] == json.loads((release.inp["v2_out"] / "contact_targets.json").read_text())


def test_no_planned_support_is_known_free_rederived(release):
    """The build's known-free check, re-derived on another path: the commanded segment from the graph's own segment
    times (the window holding the frame, else the next), the planned ground from the generator's own schedule
    (``sketch.Schedule``), the known-free zones from the sidecar payload. A planted zone made known-free must fail."""
    sketch = pytest.importorskip("edge_synthesis.sketch")
    from extract_contact_configs import ZONE_ORDER

    r = release
    edges = r.inp["edges"]
    lineage = np.load(r.out / "lineage.npz")
    names = list(r.g["motion_names"])

    def conflicts(free):
        out, checked = [], 0
        for name in r.rec["variants"]:
            m = names.index(name)
            syn = r.inp["entries"][name]["synthetic"]
            sch = sketch.schedule(ct3.edge_spec(edges, syn["edge"]), syn["durations_s"])
            kind, vt = lineage[f"{name}.kind"], lineage[f"{name}.variant_t"]
            n_seg = int(r.g["seg_count"][m])
            for f in np.nonzero(kind != 0)[0]:
                t = np.float32(f / 60.0)
                k = next((k for k in range(n_seg) if r.g["seg_end"][m, k] >= t), None)
                if k is None:
                    continue
                ground, _, _ = sch.config_at(float(vt[f]))
                zones = {ZONE_ORDER[z] for z in np.nonzero(free[m, k].numpy())[0]}
                checked += 1
                if ground & zones:
                    out.append((name, int(f), sorted(ground & zones)))
        return out, checked

    found, checked = conflicts(r.s["seg_ground_free"])
    assert found == [] and checked > 5000
    bad = r.s["seg_ground_free"].clone()
    m = names.index("SYN_E2_jumpback_mid_s0_t6px12")
    for k in range(int(r.g["seg_count"][m])):
        bad[m, k, ZONE_ORDER.index("L_FOOT")] = True               # the landing's planted foot made "known free"
    assert any(n == "SYN_E2_jumpback_mid_s0_t6px12" for n, _, _ in conflicts(bad)[0])


def test_the_edge_plans(release):
    desc = json.loads((release.out / "contact_graph.json").read_text())
    plans, picks = mep.edge_plans(desc, release.ext["clips"])
    assert picks == {e: s for e, s in R.CARD_PLAN_CLIPS.items()}
    for name, plan in plans.items():
        assert json.loads((release.out / "plans" / f"{name}.json").read_text()) == plan
    edge, nohijack = plans["edge_E1"], plans["nohijack_E1"]
    assert edge["start"] == nohijack["start"] and edge["goals"][0] == nohijack["goals"][0]
    assert edge["goals"][1]["config"].startswith("Handstand_pose_or_Adho_Mukha_Vrksasana|")
    assert nohijack["goals"][1]["config"] == "standing|L_FOOT:G|R_FOOT:G@upright"        # Crow -a's own next hold
    assert nohijack["goals"][1]["pose_time"] == 16.1833 and edge["goals"][1]["reach_s"] == 5.0
    v2_plans = sorted(p.name for p in (release.inp["v2_out"] / "plans").glob("*.json"))
    assert sorted(p.name for p in (release.out / "plans").glob("*.json")) == sorted(v2_plans + [f"{n}.json"
                                                                                              for n in plans])


def test_the_record_is_what_training_reads(release):
    rec = release.rec
    assert rec["kind"] == "reference_release" and rec["release_id"] == RELEASE_ID and rec["purpose"] == "candidate"
    assert rec["drop"] == list(R.USER_DROP) and len(rec["variants"]) == 28
    assert {e: len(v) for e, v in rec["synthetic"]["edges"].items()} == {"E1": 8, "E3": 3, "B1": 5, "E2": 11, "E5": 1}
    assert rec["counts"]["motions"] == 252 and rec["counts"]["clips"] == 84
    assert rec["training"]["eval_max_steps"] == 1890 and rec["robot"] == "smpl_yogi_v2"
    assert R.verify_record(R.RECORD_ROOT / f"{RELEASE_ID}.json") == []
    assert rec["checks"]["failed"] == [] and not R.calibration_differs(rec["calibration_c3"],
                                                                       release.inp["v2"]["calibration_c3"])


def test_the_runtime_loaders_accept_the_release(release):
    from extract_contact_configs import mjcf_body_names
    from protomotions.components.contact_graph import ContactGraph
    from protomotions.envs.control.contact_targets import ContactTargets
    from protomotions.envs.control.physics_terms import PhysicsTables
    from protomotions.utils.release_identity import ReleaseMismatchError, load_release, require_artifact

    out = release.out
    rec = load_release(R.RECORD_ROOT / f"{RELEASE_ID}.json")
    for role in ("package", "graph", "physics_tables", "contact_targets", "holds_extended"):
        require_artifact(rec, role, ids.REPO / rec["artifacts"][role]["path"])
    with pytest.raises(ReleaseMismatchError):                                    # v2's graph is not v3's
        require_artifact(rec, "graph", release.inp["v2_out"] / "contact_graph.pt")
    graph = ContactGraph.from_file(out / "contact_graph.pt")
    names = list(graph.motion_names)
    graph.validate_against_motion_lib([f"{n}.motion" for n in names], motion_num_frames=graph.motion_num_frames, fps=60)
    PhysicsTables(str(out / "physics_tables.pt"), names, mjcf_body_names(str(R.plant_mjcf())), "cpu",
                  plant_mjcf=str(R.plant_mjcf()), motion_num_frames=graph.motion_num_frames, fps=60)
    ContactTargets(str(out / "contact_targets.pt"), graph, names, "cpu", motion_num_frames=graph.motion_num_frames,
                   fps=60, graph_sha256=ids.sha256_file(out / "contact_graph.pt"), plant_mjcf=str(R.plant_mjcf()))


def test_check_passes_on_the_committed_release(release):
    passed, failed = R.check_recorded(RELEASE_ID, log_fn=lambda *_: None)
    assert failed == []
    assert any(p.startswith("release v2 passes its own identity checks") for p in passed)
