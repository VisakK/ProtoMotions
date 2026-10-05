# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Card S1's two parity tests (``expert_revist/graph_growth_2026_10_03/PLAN.MD``): is the release-v3 student env,
with G3's expert ported into it, the expert's own env?

``--test obs`` (parity 2: expert observations bit-identical on the same state). In the student env the frozen expert
drives (``rollout_actor=EXPERT``: the action is its mean action on its ``expert_*`` observations). At every step every
expert actor input the student env computed is compared, bit for bit, with the same input computed the expert's own
way on the same state: the expert's ``ContactGraphControl``, built from its resolved control config (2 slots, every
body and half visible), attached to the same env as a *twin*, and the expert's own observation components
(un-prefixed, bound to ``ctx.masked_mimic`` / ``ctx.contact_goal``) executed through their own ``ComponentManager`` as
the expert's env executes them. Three phases: every motion from t = 0; episodes reset by the student's own motion
manager and terminations; a panel-style phase on S1's route plans (manual goals, ``timing='training'``). Beside it the
rebuild self-checks: ``env.rebuild_observations`` after a step and after a reset equals the env's own observation on
every key, while the pre-S1 naive rebuild does not.

``--test eval`` (parity 1: the expert's own evaluation scores). The expert's ``HoldCurriculumEvaluator`` (its training
config, 2,250 steps, terminations included, as ``expert_revist/ft_c/eval_checkpoint.py``) with the expert acting:
``--env expert`` in the expert's own env, ``--env student`` in the student env (actions from ``expert_*``). Both
seeded identically before the evaluation; each writes the evaluator's CSV and aggregate, its predicted motion library
and every active env's action at every step. ``--test compare --runs A B`` says where two runs first differ.

    PY=../env_isaaclab/bin/python; S=results/smpl_yogi_v2_student_release_v3_s1_parity; O=output/s1_port
    PYTHONPATH=. $PY data/scripts/s1_port_parity.py --test obs --student-dir $S --out-dir $O/obs
    PYTHONPATH=. $PY data/scripts/s1_port_parity.py --test eval --env expert --out-dir $O/eval_expert
    PYTHONPATH=. $PY data/scripts/s1_port_parity.py --test eval --env student --student-dir $S --out-dir $O/eval_student
    PYTHONPATH=. $PY data/scripts/s1_port_parity.py --test compare --runs $O/eval_expert $O/eval_student
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--test", required=True, choices=("obs", "eval", "compare"))
parser.add_argument("--env", choices=("student", "expert"), default="student", help="eval: whose env")
parser.add_argument("--student-dir", default="results/smpl_yogi_v2_student_release_v3_s1_parity",
                    help="a CONFIG_ONLY run of run_student_distill_release_v3.sh (its resolved_configs.pt)")
parser.add_argument("--expert-checkpoint", default="results/smpl_yogi_v2_expert56_g3_2f132f4299/epoch_3420.ckpt")
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=300, help="obs: steps per phase")
parser.add_argument("--plans", nargs="*", default=None, help="obs: the panel phase's plans (default: S1's route plans)")
parser.add_argument("--eval-max-steps", type=int, default=2250)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--warmup-steps", type=int, default=0,
                    help="eval: before seeding the evaluation, reset every env to its clip's t = 0 (motion e %% M, "
                         "seeded) and step this many zero actions, so both envs enter the evaluation with the same "
                         "PhysX history; otherwise each carries its own construction-time reset (its own motion "
                         "manager's draws)")
parser.add_argument("--runs", nargs=2, default=None, help="compare: two eval output dirs")
parser.add_argument("--out-dir", default=None)
parser.add_argument("--simulator", default="isaaclab")
args = parser.parse_args()


def say(msg: str) -> None:
    print(f"s1-parity: {msg}", flush=True)


# --------------------------------------------------------------------------------------------------------------- #
# compare (CPU)
# --------------------------------------------------------------------------------------------------------------- #
def compare(a_dir: Path, b_dir: Path) -> int:
    import csv

    import numpy as np

    report = {"runs": [str(a_dir), str(b_dir)]}
    a, b = np.load(a_dir / "actions.npz"), np.load(b_dir / "actions.npz")
    A, B = a["actions"], b["actions"]
    report["action_steps"] = [int(A.shape[0]), int(B.shape[0])]
    n = min(A.shape[0], B.shape[0])
    diff = np.abs(A[:n].astype(np.float64) - B[:n].astype(np.float64)).reshape(n, -1).max(axis=1)
    first = int(np.argmax(diff > 0)) if bool((diff > 0).any()) else None
    report["actions_bit_identical"] = bool(first is None and A.shape == B.shape)
    report["first_differing_step"] = first
    report["max_abs_action_diff"] = float(diff.max()) if n else None
    rows = {}
    for d in (a_dir, b_dir):
        table = sorted((d / "curriculum").glob("eval_epoch_*.csv"))[-1]
        rows[d] = {r["motion"]: r for r in csv.DictReader(open(table))}
    ra, rb = rows[a_dir], rows[b_dir]
    report["motions"] = [len(ra), len(rb)]
    differing = {}
    for m in sorted(set(ra) & set(rb)):
        cols = [c for c in ra[m] if ra[m][c] != rb[m].get(c)]
        if cols:
            differing[m] = {c: [ra[m][c], rb[m].get(c)] for c in cols}
    report["csv_rows_differing"] = len(differing)
    report["csv_differences"] = differing
    ga, gb = (json.loads((d / "aggregate.json").read_text())["log"] for d in (a_dir, b_dir))
    report["aggregate_differing"] = {k: [ga[k], gb.get(k)] for k in sorted(ga) if ga[k] != gb.get(k)}
    for key in ("eval/perf/score", "eval/success_rate"):
        report[key] = [ga.get(key), gb.get(key)]
    groups = sorted(k for k in ga if k.startswith("eval/perf_group/") and k.endswith("_score"))
    report["groups"] = {k: [ga[k], gb.get(k)] for k in groups}
    out = Path(args.out_dir or b_dir) / f"compare_{a_dir.name}_vs_{b_dir.name}.json"
    out.write_text(json.dumps(report, indent=1))
    say(f"actions bit-identical: {report['actions_bit_identical']} (first differing step {first}, max |da| "
        f"{report['max_abs_action_diff']}); CSV rows differing {len(differing)} of {len(ra)}; aggregate keys "
        f"differing {len(report['aggregate_differing'])}; perf score {report['eval/perf/score']} -> {out}")
    return 0


if args.test == "compare":
    raise SystemExit(compare(Path(args.runs[0]), Path(args.runs[1])))

# --------------------------------------------------------------------------------------------------------------- #
# GPU tests
# --------------------------------------------------------------------------------------------------------------- #
args.headless = True
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from policy_setup import build, motion_names  # noqa: E402

EXPERT_DIR = Path(args.expert_checkpoint).parent


def expert_configs() -> dict:
    return torch.load(EXPERT_DIR / "resolved_configs.pt", map_location="cpu", weights_only=False)


def build_env(which: str) -> dict:
    """``student``: the CONFIG_ONLY run's training config, no student weights (the expert loads in create_model).
    ``expert``: the expert's own training config and checkpoint."""
    ns = argparse.Namespace(simulator=args.simulator, num_envs=args.num_envs, motion_file=None, headless=True,
                            overrides=[], resolved_configs="resolved_configs.pt")
    if which == "student":
        ns.checkpoint = str(Path(args.student_dir) / "untrained.ckpt")
        return build(ns, AppLauncher, load_checkpoint=False)
    ns.checkpoint = args.expert_checkpoint
    return build(ns, AppLauncher)


class ExpertActor(torch.nn.Module):
    """``agent.model`` for an evaluator in the student env: the frozen expert's mean action on ``expert_*``."""

    def __init__(self, agent):
        super().__init__()
        self.__dict__["_agent"] = agent        # not a submodule: the agent owns the expert

    def forward(self, obs_td):
        action = self._agent._collect_external_expert_action(obs_td)
        return TensorDict({"action": action, "mean_action": action}, batch_size=obs_td.batch_size)


class Recorder(torch.nn.Module):
    """Wraps ``agent.model``: records the first ``keep`` envs' mean action at every call."""

    def __init__(self, inner, keep: int):
        super().__init__()
        self.inner = inner
        self.keep = keep
        self.log = []

    def forward(self, obs_td):
        out = self.inner(obs_td)
        self.log.append(out["mean_action"][: self.keep].float().cpu().clone())
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------------------------------------------- #
# parity 1: the evaluation
# --------------------------------------------------------------------------------------------------------------- #
def run_eval(out: Path) -> dict:
    from protomotions.agents.evaluators.hold_curriculum_evaluator import HoldCurriculumEvaluator

    g3 = expert_configs()
    built = build_env(args.env)
    env, agent, fabric = built["env"], built["agent"], built["fabric"]
    ev_cfg = copy.deepcopy(g3["agent"].evaluator)
    ev_cfg.max_eval_steps = int(args.eval_max_steps)
    if args.env == "student":
        # The expert's evaluation protocol: its terminations and episode length (the student trains with 0.25 m).
        env.config.termination_components = copy.deepcopy(g3["env"].termination_components)
        env.config.max_episode_length = g3["env"].max_episode_length
        env.max_episode_length = g3["env"].max_episode_length
        model = ExpertActor(agent)
    else:
        model = agent.model
    agent.model = Recorder(model, keep=env.motion_lib.num_motions())
    agent.root_dir = out
    evaluator = HoldCurriculumEvaluator(agent=agent, fabric=fabric, config=ev_cfg)
    evaluator.eval_count = 0
    if args.warmup_steps:
        ids = torch.arange(env.num_envs, device=env.device)
        env.motion_manager.motion_ids[:] = ids % env.motion_lib.num_motions()
        env.motion_manager.motion_times[:] = 0.0
        seed_all(args.seed)
        with torch.no_grad():
            env.reset(ids, sample_flat=True, disable_motion_resample=True)
            zero = torch.zeros(env.num_envs, env.robot_config.number_of_actions, device=env.device)
            for _ in range(args.warmup_steps):
                env.step(zero)
    seed_all(args.seed)
    t0 = time.time()
    try:
        with torch.no_grad():
            log, score, n = evaluator.evaluate()
    finally:
        actions = torch.stack(agent.model.log).numpy() if agent.model.log else np.zeros(0)
        np.savez_compressed(out / "actions.npz", actions=actions, motions=np.array(motion_names(env.motion_lib)))
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()
    log = {k: float(v) for k, v in log.items()}
    agg = dict(env=args.env, checkpoint=args.expert_checkpoint, student_dir=args.student_dir, num_envs=env.num_envs,
               seed=args.seed, warmup_steps=args.warmup_steps, eval_max_steps=args.eval_max_steps, score=score, evaluated=n, log=log,
               seconds=round(time.time() - t0, 1))
    (out / "aggregate.json").write_text(json.dumps(agg, indent=1))
    say(f"eval in the {args.env} env: score {score:.6f}, success {log.get('eval/success_rate')}, "
        f"{actions.shape[0] if actions.ndim else 0} action steps recorded ({agg['seconds']} s)")
    return agg


# --------------------------------------------------------------------------------------------------------------- #
# parity 2: the observations
# --------------------------------------------------------------------------------------------------------------- #
class Twin:
    """The expert's own control and observation components, evaluated on the student env's state."""

    def __init__(self, env, g3: dict):
        from protomotions.agents.supervised.expert_utils import get_expert_actor_in_keys
        from protomotions.envs.component_manager import ComponentManager
        from protomotions.envs.control.contact_graph_control import ContactGraphControl

        self.env = env
        self.keys = list(get_expert_actor_in_keys(g3["agent"]))
        self.components = {k: copy.deepcopy(g3["env"].observation_components[k]) for k in self.keys}
        # The control reads its conditionable bodies off the robot config: the expert's are every body.
        robot = env.robot_config
        mine = list(robot.trackable_bodies_subset)
        robot.trackable_bodies_subset = list(g3["robot"].trackable_bodies_subset)
        try:
            self.control = ContactGraphControl(copy.deepcopy(g3["env"].control_components["contact_graph"]), env)
        finally:
            robot.trackable_bodies_subset = mine
        self.manager = ComponentManager(env.device)

    # Driven as the env drives its own control: reset() after env.reset, step() after env.step (the env steps its
    # controls in post_physics_step, before the observation build, on the same motion state).
    def reset(self, env_ids):
        self.control.reset(env_ids)

    def step(self):
        self.control.step()

    def observe(self) -> dict:
        side = copy.copy(self.env._current_context)
        self.control.populate_context(side)
        return self.manager.execute_all(self.components, side)


class Checks:
    def __init__(self, keys):
        self.max_abs = {k: 0.0 for k in keys}
        self.compared = 0
        self.rebuild = {"step_max_abs": 0.0, "reset_max_abs": 0.0, "naive_contact_obs_v1_max_abs": 0.0,
                        "naive_after_reset_contact_obs_v1_max_abs": 0.0, "steps": 0, "resets": 0}

    def expert(self, obs: dict, twin_obs: dict, phase: str, by_phase: dict) -> None:
        for k, value in twin_obs.items():
            d = float((obs[f"expert_{k}"].double() - value.double()).abs().max())
            self.max_abs[k] = max(self.max_abs[k], d)
            by_phase.setdefault(phase, {}).setdefault(k, 0.0)
            by_phase[phase][k] = max(by_phase[phase][k], d)
        self.compared += 1


def max_diff(a: dict, b: dict) -> float:
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)


def naive_rebuild(env) -> dict:
    env._current_context = env._build_global_context(env.simulator.get_robot_state())
    env.compute_observations(context=env._current_context)
    return env.get_obs()


def run_obs(out: Path) -> dict:
    from protomotions.agents.evaluators.sequence_viz import SequenceVizConfig, SequenceVizRunner, fill_goal_slots
    from protomotions.components.contact_graph import ContactGraph

    g3 = expert_configs()
    built = build_env("student")
    env, agent = built["env"], built["agent"]
    agent.eval()
    E, dev = env.num_envs, env.device
    student = env.control_manager.components["contact_graph"]
    twin = Twin(env, g3)
    checks = Checks(twin.keys)
    by_phase: dict = {}
    names = motion_names(env.motion_lib)
    say(f"student env: {E} envs, {len(names)} motions; the expert's {len(twin.keys)} actor inputs compared; "
        f"expert view slots {student._expert_view_steps}, bodies {student._expert_view_num_bodies}")

    def td(obs):
        return agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

    def act(obs):
        return agent._collect_external_expert_action(td(obs))

    def compare(obs, phase):
        # The twin reads env._current_context, whose previous_contact_forces _finalize_contact_state has overwritten
        # in place since env.step built it (S0's trap 1). Rebuilding first (exact on every key, checked below) makes
        # the context the one the observation was built from.
        env.rebuild_observations()
        checks.expert(obs, twin.observe(), phase, by_phase)

    def rebuild_check(obs, after_reset_rows=None):
        rebuilt = env.rebuild_observations()
        key = "reset_max_abs" if after_reset_rows is not None else "step_max_abs"
        checks.rebuild[key] = max(checks.rebuild[key], max_diff(obs, rebuilt))
        naive = naive_rebuild(env)
        d = (naive["contact_obs_v1"] - obs["contact_obs_v1"]).abs()
        if after_reset_rows is not None:
            d = d[after_reset_rows]
            name = "naive_after_reset_contact_obs_v1_max_abs"
        else:
            name = "naive_contact_obs_v1_max_abs"
        checks.rebuild[name] = max(checks.rebuild[name], float(d.max()) if d.numel() else 0.0)
        env.rebuild_observations()          # put the exact observation back
        checks.rebuild["resets" if after_reset_rows is not None else "steps"] += 1

    seed_all(args.seed)
    ids = torch.arange(E, device=dev)
    # ---- phase 1: every motion from t = 0 ------------------------------------------------------------------- #
    env.motion_manager.motion_ids[:] = ids % len(names)
    env.motion_manager.motion_times[:] = 0.0
    obs, _ = env.reset(ids, sample_flat=True, disable_motion_resample=True)
    twin.reset(ids)
    compare(obs, "clip_starts")
    rebuild_check(obs, after_reset_rows=ids)
    t0 = time.time()
    for step in range(args.steps):
        obs, *_ = env.step(act(obs))
        twin.step()
        compare(obs, "clip_starts")
        rebuild_check(obs)
    say(f"phase 1 (clip starts): {args.steps} steps in {time.time() - t0:.0f} s; max |d| "
        f"{max(by_phase['clip_starts'].values()):.3g}")
    # ---- phase 2: the student's own resets (its motion manager, its terminations) ---------------------------- #
    obs, _ = env.reset(ids)
    twin.reset(ids)
    compare(obs, "training_resets")
    resets = 0
    for step in range(args.steps):
        obs, _rew, dones, _term, _extras = env.step(act(obs))
        twin.step()
        compare(obs, "training_resets")
        rebuild_check(obs)
        done = dones.nonzero(as_tuple=True)[0]
        if done.numel():
            resets += int(done.numel())
            obs, _ = env.reset(done)
            twin.reset(done)
            compare(obs, "training_resets")
            rebuild_check(obs, after_reset_rows=done)
    say(f"phase 2 (training resets): {resets} env resets; max |d| {max(by_phase['training_resets'].values()):.3g}")
    # ---- phase 3: manual goals on S1's route plans, training timing ------------------------------------------ #
    plans = args.plans or sorted(str(p) for p in (REPO / "data/scripts/plans_release_v3_route").glob("*.json"))
    runner = SequenceVizRunner.__new__(SequenceVizRunner)
    runner.config = SequenceVizConfig(viz_every=1, num_sequences=len(plans), plan_files=list(plans), max_seconds=60.0)
    runner.graph = ContactGraph.from_file(student.config.graph_file)
    runner.motion_names = names
    seqs = [s for s in (runner._load_plan(p) for p in plans) if s is not None]
    S = len(seqs)
    env_seq = ids % S
    env.motion_manager.motion_ids[:] = torch.tensor([s.start_motion for s in seqs], device=dev)[env_seq]
    env.motion_manager.motion_times[:] = 0.0
    obs, _ = env.reset(ids, sample_flat=True, disable_motion_resample=True)
    twin.reset(ids)
    dt = float(env.dt)
    active = None
    for step in range(args.steps):
        t = step * dt
        now = [s.active_index(t) for s in seqs]
        if active is None or now != active or step % 15 == 0:
            for control, slots in ((student, student.config.num_goal_steps), (twin.control, 2)):
                control.set_manual_goal(**fill_goal_slots(seqs, env_seq, t, slots, 1.2, dev, timing="training"))
            obs = env.rebuild_observations()
            active = now
        compare(obs, "manual_goals")
        obs, *_ = env.step(act(obs))
        twin.step()
        compare(obs, "manual_goals")
        rebuild_check(obs)
    student.clear_manual_goal()
    say(f"phase 3 (manual goals, {S} route plans): max |d| {max(by_phase['manual_goals'].values()):.3g}")
    if hasattr(env.simulator, "shutdown"):
        env.simulator.shutdown()
    result = dict(
        student_dir=args.student_dir, checkpoint=args.expert_checkpoint, num_envs=E, steps_per_phase=args.steps,
        expert_inputs=twin.keys, comparisons=checks.compared, max_abs_by_key=checks.max_abs, by_phase=by_phase,
        bit_identical=all(v == 0.0 for v in checks.max_abs.values()), rebuild=checks.rebuild,
        rebuild_exact=checks.rebuild["step_max_abs"] == 0.0 and checks.rebuild["reset_max_abs"] == 0.0)
    say(f"expert inputs bit-identical over {checks.compared} comparisons: {result['bit_identical']} "
        f"{ {k: v for k, v in checks.max_abs.items() if v} }; rebuild exact: {result['rebuild_exact']} {checks.rebuild}")
    return result


def main() -> int:
    out = Path(args.out_dir or f"output/s1_port_parity/{args.test}_{args.env}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = run_eval(out) if args.test == "eval" else run_obs(out)
    (out / f"{args.test}.json").write_text(json.dumps(result, indent=1))
    say(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
