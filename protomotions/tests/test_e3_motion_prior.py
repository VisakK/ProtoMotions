# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The package prior in the mixture curriculum's uniform share, graph_growth PLAN.MD card E3.

``mixture_sampling_probs(..., prior=w)`` spreads the uniform share as ``u * w_m / sum(w)``;
``HoldCurriculumConfig.motion_prior = "package"`` captures ``w`` from the library when the
evaluator is built. The regression tests run the code from before E3 (loaded from git)
beside the new code on G1's frozen curriculum config and its epoch-5000 scores.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from protomotions.agents.evaluators.config import HoldCurriculumConfig, HoldCurriculumEvaluatorConfig
from protomotions.agents.evaluators.hold_curriculum import mixture_sampling_probs, uniform_share
from protomotions.agents.evaluators.hold_curriculum_evaluator import HoldCurriculumEvaluator
from protomotions.tests.test_e3_departure_anchoring import G1_DIR, RELEASE_V2, module_at_pre_e3_commit


# --------------------------------------------------------------------------- #
# mixture_sampling_probs
# --------------------------------------------------------------------------- #
def _cases():
    gen = torch.Generator().manual_seed(0)
    for n in (1, 2, 3, 7, 56, 168, 170, 183, 213, 1000):
        for u in (0.0, 0.1, 0.2, 1.0 / 3.0, 0.5, 0.7, 0.8, 0.9, 0.99, 1.0):
            for power in (1.0, 2.0):
                yield n, u, power, torch.rand(n, generator=gen)


def test_without_a_prior_the_probabilities_are_the_code_before_e3(tmp_path):
    old = module_at_pre_e3_commit("protomotions/agents/evaluators/hold_curriculum.py", "hc_pre_e3", tmp_path)
    for n, u, power, score in _cases():
        assert torch.equal(mixture_sampling_probs(score, u, power=power, eps=1e-3),
                           old.mixture_sampling_probs(score, u, power=power, eps=1e-3))


def test_a_prior_of_ones_is_bit_identical(tmp_path):
    old = module_at_pre_e3_commit("protomotions/agents/evaluators/hold_curriculum.py", "hc_pre_e3b", tmp_path)
    count = 0
    for n, u, power, score in _cases():
        with_prior = mixture_sampling_probs(score, u, power=power, eps=1e-3, prior=torch.ones(n))
        assert torch.equal(with_prior, old.mixture_sampling_probs(score, u, power=power, eps=1e-3))
        count += 1
    assert count == 200
    # The naive order u * w / sum(w) is NOT bit-identical -- the reason uniform_share
    # orders the product as (u / n) * (w * n / sum(w)).
    naive_differs = 0
    for n, u, power, score in _cases():
        w = torch.ones(n)
        need = (1.0 - score).clamp(min=0.0).pow(power) + 1e-3
        naive = u * w / w.sum() + (1.0 - u) * (need / need.sum())
        naive_differs += int(not torch.equal(naive, old.mixture_sampling_probs(score, u, power=power, eps=1e-3)))
    assert naive_differs > 0


def test_the_prior_spreads_the_uniform_share_by_weight():
    weights = torch.tensor([1.0, 1.0, 3.0, 0.0])
    score = torch.tensor([0.9, 0.2, 0.5, 0.7])
    probs = mixture_sampling_probs(score, 0.8, prior=weights)
    no_prior = mixture_sampling_probs(score, 0.8)
    expected_uniform = 0.8 * weights / weights.sum()
    assert torch.allclose(probs - (no_prior - 0.8 / 4), expected_uniform, atol=1e-7)
    assert torch.allclose(uniform_share(4, 0.8, weights), expected_uniform, atol=1e-7)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
    # All uniform: the probabilities are the normalised prior.
    assert torch.allclose(mixture_sampling_probs(score, 1.0, prior=weights), weights / weights.sum())


@pytest.mark.parametrize(
    "weights, match",
    [(torch.ones(3), "3 weights for 4"), (torch.tensor([1.0, -1.0, 1.0, 1.0]), "non-negative"),
     (torch.zeros(4), "not all zero"), (torch.tensor([1.0, math.nan, 1.0, 1.0]), "finite")],
)
def test_invalid_priors_are_refused(weights, match):
    with pytest.raises(ValueError, match=match):
        mixture_sampling_probs(torch.rand(4), 0.8, prior=weights)


# --------------------------------------------------------------------------- #
# The evaluator: capture once, keep across evaluations, log per group
# --------------------------------------------------------------------------- #
class _MotionManager:
    def __init__(self, weights):
        self.motion_weights = weights.clone()

    def update_sampling_weights(self, w):
        self.motion_weights[:] = w


def _library(tmp_path, weights, stems, suffix=".pt", yaml_weights=None):
    """A packed library ``motions.pt`` with its ``motions.yaml`` beside it (or a yaml library)."""
    root = tmp_path / "package"
    root.mkdir(exist_ok=True)
    entries = [{"file": f"motions/{s}.motion", "weight": float(w)}
               for s, w in zip(stems, yaml_weights if yaml_weights is not None else weights)]
    (root / "motions.yaml").write_text(yaml.safe_dump({"motions": entries}))
    return SimpleNamespace(
        motion_weights=torch.tensor(weights, dtype=torch.float32),
        motion_files=tuple(str(root / "motions" / f"{s}.motion") for s in stems),
        motion_file=str(root / f"motions{suffix}"),
    )


def _evaluator(motion_lib, prior="package", groups=None, uniform_fraction=0.8):
    n = len(motion_lib.motion_files)
    env = SimpleNamespace(
        dt=1.0 / 30.0, motion_manager=_MotionManager(motion_lib.motion_weights),
        robot_config=SimpleNamespace(kinematic_info=SimpleNamespace(num_bodies=24, num_dofs=69, body_names=[])),
    )
    agent = SimpleNamespace(motion_lib=motion_lib, env=env)
    config = HoldCurriculumEvaluatorConfig(
        curriculum=HoldCurriculumConfig(motion_prior=prior, uniform_fraction=uniform_fraction))
    ev = HoldCurriculumEvaluator(agent, SimpleNamespace(device="cpu"), config)
    ev._groups = groups or ["human"] * n
    ev._report_excluded = [False] * n
    ev._stems = [Path(f).stem for f in motion_lib.motion_files]
    ev._x0 = [True] * n
    return ev


def _scores(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return {k: torch.rand(n, generator=gen) for k in ("score", "p_track", "p_hold", "p_family",
                                                        "p_family_event", "support_violation", "track_fail_frac")}


def test_the_prior_is_captured_at_build_and_kept_across_evaluations(tmp_path):
    stems = ["human_a", "human_b", "human_c", "SYN_edge"]
    lib = _library(tmp_path, [1.0, 1.0, 1.0, 3.0], stems)
    ev = _evaluator(lib, groups=["human", "human", "human", "edge"])
    assert torch.equal(ev._motion_prior, torch.tensor([1.0, 1.0, 1.0, 3.0]))
    # The manager's weights are overwritten by every evaluation; the prior is not.
    for seed in range(3):
        scores = _scores(4, seed)
        probs = ev._update_curriculum(scores)
        assert torch.equal(ev.env.motion_manager.motion_weights, probs)
        expected = mixture_sampling_probs(ev._score_ema, 0.8, prior=torch.tensor([1.0, 1.0, 1.0, 3.0]))
        assert torch.equal(probs, expected)
        logs = ev._curriculum_logs(scores, probs)
        assert logs["eval/curriculum/prior_share/edge"] == pytest.approx(0.5)
        assert logs["eval/curriculum/prior_share/human"] == pytest.approx(0.5)
        assert logs["eval/curriculum/group_prob/edge"] + logs["eval/curriculum/group_prob/human"] == \
            pytest.approx(1.0, abs=1e-6)
        assert logs["eval/curriculum/ess"] == pytest.approx(float(1.0 / (probs ** 2).sum()))
        # The prioritized part is what is left once the PRIOR's uniform share is removed.
        prioritized = probs - 0.8 * torch.tensor([1.0, 1.0, 1.0, 3.0]) / 6.0
        assert float(prioritized.sum()) == pytest.approx(0.2, abs=1e-6)
        assert float(prioritized.min()) > 0.0
    assert torch.equal(ev._motion_prior, torch.tensor([1.0, 1.0, 1.0, 3.0]))


def test_no_prior_logs_no_prior_keys(tmp_path):
    lib = _library(tmp_path, [1.0, 2.0], ["a", "b"])
    ev = _evaluator(lib, prior="none")
    assert ev._motion_prior is None
    scores = _scores(2)
    logs = ev._curriculum_logs(scores, ev._update_curriculum(scores))
    assert not any(k.startswith("eval/curriculum/prior_share") or k.startswith("eval/curriculum/group_prob")
                   for k in logs)
    assert "eval/curriculum/ess" in logs


def test_fractional_package_weights_match_their_yaml(tmp_path):
    # D5's weights (3/n for an edge with n variants): 0.6 is not a float32, and must still match.
    weights = [1.0, 0.375, 1.0, 0.6, 0.25, 3.0]
    lib = _library(tmp_path, weights, ["h", "SYN_E1_a", "SYN_E3_a", "SYN_B1_a", "SYN_E2_a", "SYN_E5_a"])
    assert torch.equal(_evaluator(lib)._motion_prior, torch.tensor(weights, dtype=torch.float32))


def test_yaml_and_packed_library_must_agree(tmp_path):
    lib = _library(tmp_path, [1.0, 3.0], ["a", "SYN_b"], yaml_weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="disagree on 1 weights"):
        _evaluator(lib)
    # No yaml beside the .pt: the packed weights are the prior.
    lone = _library(tmp_path, [1.0, 3.0], ["a", "SYN_b"])
    (Path(lone.motion_file).with_suffix(".yaml")).unlink()
    assert torch.equal(_evaluator(lone)._motion_prior, torch.tensor([1.0, 3.0]))
    # A yaml library is its own source.
    as_yaml = _library(tmp_path, [1.0, 3.0], ["a", "SYN_b"], suffix=".yaml", yaml_weights=[5.0, 5.0])
    assert torch.equal(_evaluator(as_yaml)._motion_prior, torch.tensor([1.0, 3.0]))


def test_yaml_and_library_names_are_matched_by_the_same_rule(tmp_path):
    # A stem may contain dots, and a file need not be a .motion: both sides strip
    # '.motion' when present and use Path.stem otherwise, so they always agree.
    lib = _library(tmp_path, [1.0, 3.0], ["a.v2", "SYN_b"])
    assert torch.equal(_evaluator(lib)._motion_prior, torch.tensor([1.0, 3.0]))
    root = Path(lib.motion_file).parent
    (root / "motions.yaml").write_text(yaml.safe_dump(
        {"motions": [{"file": "motions/a.v2.npz", "weight": 1.0}, {"file": "motions/SYN_b.npz", "weight": 3.0}]}))
    npz = SimpleNamespace(motion_weights=lib.motion_weights, motion_file=lib.motion_file,
                          motion_files=(str(root / "motions" / "a.v2.npz"), str(root / "motions" / "SYN_b.npz")))
    assert torch.equal(_evaluator(npz)._motion_prior, torch.tensor([1.0, 3.0]))


def test_unknown_prior_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="motion_prior"):
        _evaluator(_library(tmp_path, [1.0], ["a"]), prior="yaml")


def test_old_pickled_configs_read_no_prior():
    cfg = HoldCurriculumConfig()
    del cfg.__dict__["motion_prior"]                       # a config frozen before the field existed
    assert cfg.motion_prior == "none"


# --------------------------------------------------------------------------- #
# Regression on G1's frozen curriculum config and its epoch-5000 scores
# --------------------------------------------------------------------------- #
def _g1_inputs():
    resolved, checkpoint = G1_DIR / "resolved_configs.pt", G1_DIR / "epoch_5000.ckpt"
    library = RELEASE_V2 / "motions.pt"
    if not (resolved.is_file() and checkpoint.is_file() and library.is_file()):
        pytest.skip("G1's run or release v2's library is not on this machine")
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    curriculum = torch.load(resolved, map_location="cpu", weights_only=False)["agent"].evaluator.curriculum
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    score_ema = state["evaluator"]["score_ema"].float()
    motion_lib = MotionLib(MotionLibConfig(motion_file=str(library)), device="cpu")
    manifest = yaml.safe_load(open(RELEASE_V2 / "holds_extended.yaml"))
    group = {c["stem"]: c.get("group", "all") for c in manifest["clips"]}
    groups = [group[Path(f).name[: -len(".motion")]] for f in motion_lib.motion_files]
    return curriculum, score_ema, motion_lib, groups


def _g1_scores(score_ema, seed):
    n = score_ema.numel()
    scores = _scores(n, seed)
    gen = torch.Generator().manual_seed(100 + seed)
    noise = (torch.rand(n, generator=gen) - 0.5) * 0.2
    scores["score"] = (score_ema + noise).clamp(0, 1)
    scores["score_v2"] = (score_ema - noise).clamp(0, 1)
    for k in ("p_hold_v2", "p_family_v2", "p_family_event_v2", "support_violation_v2"):
        scores[k] = torch.rand(n, generator=gen)
    for k in ("holds_tracked_v2", "supports_scored_v2", "supports_realised_v2", "substitution_holds_v2"):
        scores[k] = torch.randint(0, 4, (n,), generator=gen).float()
    return scores


@pytest.mark.parametrize("prior", ["none", "package"])
def test_g1_curriculum_probabilities_match_the_code_before_e3(tmp_path, prior):
    curriculum, score_ema, motion_lib, groups = _g1_inputs()
    assert "motion_prior" not in curriculum.__dict__ and curriculum.motion_prior == "none"
    assert curriculum.support_rule == "v2" and curriculum.uniform_fraction == 0.8
    old_hc = module_at_pre_e3_commit("protomotions/agents/evaluators/hold_curriculum.py", f"hc_g1_{prior}", tmp_path)
    old_ev = module_at_pre_e3_commit(
        "protomotions/agents/evaluators/hold_curriculum_evaluator.py", f"hce_g1_{prior}", tmp_path)
    old_ev.mixture_sampling_probs = old_hc.mixture_sampling_probs      # every piece pre-E3

    def build(cls, motion_prior):
        cfg = HoldCurriculumConfig(**{k: v for k, v in curriculum.__dict__.items()})
        cfg.motion_prior = motion_prior
        ev = object.__new__(cls)
        ev.config = SimpleNamespace(curriculum=cfg)
        ev.agent = SimpleNamespace(motion_lib=motion_lib,
                                   env=SimpleNamespace(motion_manager=_MotionManager(motion_lib.motion_weights)))
        ev._support_rule, ev._score_ema, ev._groups = "v2", None, groups
        ev._report_excluded, ev._x0, ev._drag_tables = [False] * len(groups), [True] * len(groups), None
        ev._stems = [Path(f).stem for f in motion_lib.motion_files]
        if cls is HoldCurriculumEvaluator:
            ev._motion_prior = ev._capture_motion_prior()
        return ev

    new = build(HoldCurriculumEvaluator, prior)
    ref = build(old_ev.HoldCurriculumEvaluator, "none")
    if prior == "package":                                   # release v2: every weight 1.0, checked
        assert torch.equal(new._motion_prior, torch.ones(168))
    for seed in range(3):                                    # three evaluations, the EMA carried
        scores = _g1_scores(score_ema, seed)
        probs_new = new._update_curriculum(scores)
        probs_ref = ref._update_curriculum(scores)
        assert torch.equal(probs_new, probs_ref)
        assert torch.equal(new.env.motion_manager.motion_weights, ref.env.motion_manager.motion_weights)
        logs_new = new._curriculum_logs(scores, probs_new)
        logs_ref = ref._curriculum_logs(scores, probs_ref)
        extra = {k for k in logs_new if k.startswith(("eval/curriculum/prior_share/", "eval/curriculum/group_prob/"))}
        assert bool(extra) == (prior == "package")
        assert {k: v for k, v in logs_new.items() if k not in extra} == logs_ref
