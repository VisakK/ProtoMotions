"""Card E1 acceptance 5: with the default ``--support-rule v1`` the expert experiment builds the configs it built
before the change.

Rebuilds the env and agent configs of a finished run from its own saved CLI arguments (``config.yaml``) with the
current code, and compares them field by field with the run's frozen ``resolved_configs.pt`` -- which this also
proves still unpickles under the new ``HoldCurriculumConfig``. New fields must sit at their defaults.

    PYTHONPATH=. python expert_revist/graph_growth_2026_10_03/e1_support_v2/check_config_identity.py \\
        --run results/smpl_yogi_v2_expert56_a2dda5d2ac
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NEW_FIELDS = {"support_rule": "v1", "support_v2_load_frac_bw": 0.03, "support_v2_min_share": 0.2,
              "support_v2_down_m": 0.02, "support_v2_realised_share": 0.9, "support_v2_tracked_share": 0.9}


def diff(a, b, path="", out=None, seen=None):
    """Recursive structural comparison; MdpComponents compare by compute function and parameters."""
    out = [] if out is None else out
    seen = set() if seen is None else seen
    if id(a) in seen:
        return out
    seen.add(id(a))
    from protomotions.envs.mdp_component import MdpComponent

    if isinstance(a, MdpComponent) or isinstance(b, MdpComponent):
        fa = getattr(a, "compute_func", None)
        fb = getattr(b, "compute_func", None)
        if getattr(fa, "__qualname__", fa) != getattr(fb, "__qualname__", fb):
            out.append((path + ".compute_func", fa, fb))
        diff(getattr(a, "static_params", None), getattr(b, "static_params", None), path + ".static_params", out, seen)
        diff(str(getattr(a, "dynamic_vars", None)), str(getattr(b, "dynamic_vars", None)), path + ".dynamic_vars",
             out, seen)
        return out
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        if type(a).__name__ != type(b).__name__:
            out.append((path, type(a).__name__, type(b).__name__))
            return out
        names = {f.name for f in dataclasses.fields(type(b))}
        for name in sorted(names):
            va, vb = getattr(a, name, "<missing>"), getattr(b, name, "<missing>")
            diff(va, vb, f"{path}.{name}", out, seen)
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}[{k!r}]", out, seen)
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append((path + ".len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, seen)
        return out
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape
                and torch.equal(a.cpu(), b.cpu())):
            out.append((path, "tensor", "tensor"))
        return out
    if a != b and not (a != a and b != b):
        out.append((path, a, b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/smpl_yogi_v2_expert56_a2dda5d2ac")
    args_cli = ap.parse_args()
    run = REPO / args_cli.run
    frozen = torch.load(run / "resolved_configs.pt", map_location="cpu", weights_only=False)
    saved = json.loads((run / "config.yaml").read_text())

    from examples.experiments.mimic import mlp_goal_conditioned as exp
    from protomotions.robot_configs.factory import robot_config

    parser = argparse.ArgumentParser()
    exp.additional_experiment_arguments(parser)
    args = parser.parse_args([])
    for k, v in saved.items():                      # the run's own arguments, base parser's included
        setattr(args, k, v)
    assert getattr(args, "support_rule", "v1") == "v1"
    robot = robot_config(saved.get("robot_name", "smpl_yogi_v2"))
    exp.configure_robot_and_simulator(robot, SimpleNamespace(), args)
    env = exp.env_config(robot, args)
    agent = exp.agent_config(robot, env, args)

    old_eval, new_eval = frozen["agent"].evaluator, agent.evaluator
    cur = old_eval.curriculum
    for name, default in NEW_FIELDS.items():        # the frozen pickle predates them: class defaults show through
        assert name not in vars(cur) and getattr(cur, name) == default, name
        assert getattr(new_eval.curriculum, name) == default, name
    ev = diff(old_eval, new_eval, "agent.evaluator")
    ag = diff(frozen["agent"], agent, "agent")
    en = diff(frozen["env"], env, "env")
    print(f"evaluator config: {len(ev)} differences")
    for d in ev[:20]:
        print("   ", d)
    print(f"agent config (all): {len(ag)} differences")
    for d in ag[:20]:
        print("   ", d)
    print(f"env config: {len(en)} differences")
    for d in en[:20]:
        print("   ", d)
    return 0 if not ev else 1


if __name__ == "__main__":
    sys.exit(main())
