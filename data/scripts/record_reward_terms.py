# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record every reward term, per policy step, over N parallel rollouts of one clip.

Answers "are the reward terms doing their job?" by dumping the *decomposed*
reward landscape of a trained policy rather than the scalar the trainer logs:

* every ``reward_components`` entry, raw and weight-scaled, per env per step
  (exactly the tensors ``combine_rewards`` produced -- read out of ``extras``);
* ``contact_match_rew`` split into its **false-positive** (simulated contact the
  reference does not have) and **false-negative** (reference contact the policy
  does not make) halves, per body;
* ``pair_contact_rew`` split per pair into its four factors -- the reference gate
  tau_p(t), the surface gap d_p, the shaping phi(d+), the load factor -- and the
  resulting per-pair contribution, which sums back to the scalar term;
* ``pair_forbid_rew`` likewise (gate, gap, psi ramp, load).

Two correctness guarantees, both asserted every step (not spot-checked):

1. the per-pair contributions sum to the env's own ``raw_r/pair_contact_rew``;
2. the per-body FP+FN mean equals the env's own ``raw_r/contact_match_rew``.

So the decomposition is provably the same arithmetic the policy was trained on,
not a re-derivation that might have drifted.

**Why this does not take a bare ``.motion`` file.** The pair-reward kernels gate
every target on ``motion_ids`` indexed into ``yoga_yogi_crow_pair_reftargets.pt``
(row 0 = crow, row 1 = side crow). A single-clip motion file is always motion id
0, so scoring side crow that way would silently apply *crow's* targets -- there
is no assert on this path, since the reftargets/motion-file contract check only
runs in the training-time ``env_config``. This script therefore loads the full
two-clip package and pins every env to the requested clip with the motion
manager's ``fixed_motion_ids_per_env``, so ids stay 0/1 as packaged.

Variation across the N rollouts comes from PhysX non-determinism (documented in
``notes/Skill_graph_handstand_lessons_2.MD``); actions are the deterministic mean
unless ``--stochastic``. All envs start at t=0 (``init_start_prob=1.0``), and the
run stops one step before the clip ends so no env ever resets mid-rollout.

Terminations are absent from the frozen inference config, so a diverging env
keeps producing reward instead of being cut. That is deliberate -- it is the only
way to see the whole clip -- and a weight-0 ``diag_worst_body_err`` diagnostic
(the same kernel arithmetic as the training termination) is recorded so the
plotter can mark where an env *would* have terminated.

Example::

    python data/scripts/record_reward_terms.py \
      --checkpoint results/smpl_yogi_crow_pair_pair_rew_scratch/last.ckpt \
      --motion-file data/smpl/yoga_yogi_crow_pair_v3_contacts.pt \
      --motion 220923_Crane_Crow_Pose_or_Bakasana_-a \
      --num-rollouts 50 --out-dir results/Reward_landscape
"""

from __future__ import annotations

import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--motion-file",
        type=str,
        default="data/smpl/yoga_yogi_crow_pair_v3_contacts.pt",
        help="The packaged multi-clip .pt the checkpoint was trained on. Must "
        "keep the motion_id order the pair-reward reftargets table was built "
        "for -- see the module docstring.",
    )
    parser.add_argument(
        "--motion",
        type=str,
        required=True,
        help="Clip stem (or integer motion id) to pin every env to.",
    )
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="results/Reward_landscape")
    parser.add_argument("--simulator", type=str, default="isaaclab")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of the deterministic mean action.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cap on policy steps (0 = run the clip to one step before its end).",
    )
    return parser


parser = create_parser()
args, _unknown = parser.parse_known_args()

# isaacgym/isaaclab must be imported before torch.
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")


class _Log:
    """Isaac's app launcher reconfigures python logging; print survives it."""

    @staticmethod
    def info(fmt, *fmt_args) -> None:
        print("record_reward_terms: " + (fmt % fmt_args if fmt_args else fmt), flush=True)


log = _Log()


# --------------------------------------------------------------------------- #
# Probes. Each wraps the *live* compute_func: it calls through and returns the
# original tensor unchanged, then recomputes the decomposition and asserts the
# parts sum to the whole. Installed before the first env.step(), so the
# ComponentManager's torch.compile cache is still empty.
# --------------------------------------------------------------------------- #
class EncourageProbe:
    """Per-pair breakdown of PairEncourageReward."""

    def __init__(self, inner):
        self.inner = inner
        self.gate, self.gap, self.phi, self.load, self.sat, self.contrib = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.ground_fz, self.net_fmag, self.body_pos = [], [], []
        self.max_abs_err = 0.0

    @torch._dynamo.disable
    def __call__(
        self, body_pos, body_rot, net_forces, ground_forces, motion_ids, motion_times
    ):
        from examples.experiments.mimic.pair_contact_terms import (
            _body_body_load,
            _pair_surface_gap,
        )

        out = self.inner(
            body_pos, body_rot, net_forces, ground_forces, motion_ids, motion_times
        )
        k = self.inner
        rt = k.rt
        E, P = body_pos.shape[0], len(rt.pair_names)
        nan = float("nan")

        gate = rt.encourage_mask[rt.frame_lookup(motion_ids, motion_times)]  # [E,P]
        gap = torch.full((E, P), nan, device=body_pos.device)
        phi = torch.full((E, P), nan, device=body_pos.device)
        load = torch.full((E, P), nan, device=body_pos.device)
        sat = torch.zeros((E, P), device=body_pos.device)

        if k.bb_pair_idx.numel() > 0:
            p0, p1, radii = rt.world_segments(body_pos, body_rot, k.uniq_bodies)
            g = _pair_surface_gap(
                p0[:, k.a_in_uniq],
                p1[:, k.a_in_uniq],
                radii[k.a_in_uniq],
                p0[:, k.b_in_uniq],
                p1[:, k.b_in_uniq],
                radii[k.b_in_uniq],
            )
            ph = torch.exp(
                -((g.clamp_min(0.0) - k.gap_target).clamp_min(0.0) / k.gap_sigma) ** 2
            )
            bl = _body_body_load(net_forces, ground_forces, k.load_ref_n)
            pl = torch.minimum(bl[:, k.bb_a], bl[:, k.bb_b])
            gap[:, k.bb_pair_idx] = g
            phi[:, k.bb_pair_idx] = ph
            load[:, k.bb_pair_idx] = pl
            sat[:, k.bb_pair_idx] = ph * (0.5 + 0.5 * pl)

        for p in k.g_pair_idx.tolist():
            zone = rt.ground_zone_bodies[p]
            fz = ground_forces[:, zone, 2].clamp_min(0.0).sum(dim=-1)
            sat[:, p] = (fz / k.ground_ref_n).clamp(0.0, 1.0)

        w = rt.pair_kappa.unsqueeze(0) * gate
        contrib = w * sat / w.sum(dim=-1, keepdim=True).clamp_min(0.1)

        err = (contrib.sum(dim=-1) - out).abs().max().item()
        self.max_abs_err = max(self.max_abs_err, err)
        assert err < 2e-5, (
            f"pair-encourage decomposition does not reproduce the env's reward "
            f"(max |sum(parts) - whole| = {err:.3e}) -- the probe has drifted "
            "from the kernel"
        )

        self.gate.append(gate.cpu().numpy())
        self.gap.append(gap.cpu().numpy())
        self.phi.append(phi.cpu().numpy())
        self.load.append(load.cpu().numpy())
        self.sat.append(sat.cpu().numpy())
        self.contrib.append(contrib.cpu().numpy())
        self.ground_fz.append(ground_forces[..., 2].cpu().numpy())
        self.net_fmag.append(net_forces.norm(dim=-1).cpu().numpy())
        self.body_pos.append(body_pos.cpu().numpy())
        return out


class ForbidProbe:
    """Per-pair breakdown of PairForbidPenalty."""

    def __init__(self, inner):
        self.inner = inner
        self.gate, self.gap, self.psi, self.load, self.contrib = [], [], [], [], []
        self.max_abs_err = 0.0

    @torch._dynamo.disable
    def __call__(
        self, body_pos, body_rot, net_forces, ground_forces, motion_ids, motion_times
    ):
        from examples.experiments.mimic.pair_contact_terms import (
            _body_body_load,
            _pair_surface_gap,
        )

        out = self.inner(
            body_pos, body_rot, net_forces, ground_forces, motion_ids, motion_times
        )
        k = self.inner
        rt = k.rt
        gate = rt.forbid_mask[rt.frame_lookup(motion_ids, motion_times)]  # [E,F]
        p0, p1, radii = rt.world_segments(body_pos, body_rot, k.uniq_bodies)
        gap = _pair_surface_gap(
            p0[:, k.a_in_uniq],
            p1[:, k.a_in_uniq],
            radii[k.a_in_uniq],
            p0[:, k.b_in_uniq],
            p1[:, k.b_in_uniq],
            radii[k.b_in_uniq],
        )
        psi = (1.0 - gap.clamp_min(0.0) / k.margin).clamp(0.0, 1.0)
        bl = _body_body_load(net_forces, ground_forces, k.load_ref_n)
        pl = torch.minimum(bl[:, rt.forbid_body_a], bl[:, rt.forbid_body_b])
        contrib = gate * psi * pl / gate.sum(dim=-1, keepdim=True).clamp_min(1.0)

        err = (contrib.sum(dim=-1) - out).abs().max().item()
        self.max_abs_err = max(self.max_abs_err, err)
        assert err < 2e-5, (
            f"pair-forbid decomposition does not reproduce the env's reward "
            f"(max |sum(parts) - whole| = {err:.3e})"
        )

        self.gate.append(gate.cpu().numpy())
        self.gap.append(gap.cpu().numpy())
        self.psi.append(psi.cpu().numpy())
        self.load.append(pl.cpu().numpy())
        self.contrib.append(contrib.cpu().numpy())
        return out


class ContactMatchProbe:
    """False-positive / false-negative split of compute_contact_match_rew."""

    def __init__(self, inner):
        self.inner = inner
        self.sim, self.ref, self.fp, self.fn = [], [], [], []
        self.body_ids = None
        self.max_abs_err = 0.0

    @torch._dynamo.disable
    def __call__(self, sim_contacts, ref_contacts, contact_body_ids, normalize=False):
        out = self.inner(
            sim_contacts, ref_contacts, contact_body_ids, normalize=normalize
        )
        sim = sim_contacts[:, contact_body_ids].float()
        ref = ref_contacts[:, contact_body_ids].float()
        fp = (sim - ref).clamp_min(0.0)  # policy touches where the reference does not
        fn = (ref - sim).clamp_min(0.0)  # reference touches where the policy does not
        n = max(int(sim.shape[1]), 1)
        recon = (fp + fn).sum(dim=1) / (n if normalize else 1)

        err = (recon - out).abs().max().item()
        self.max_abs_err = max(self.max_abs_err, err)
        assert err < 2e-5, (
            f"contact-match FP/FN split does not reproduce the env's reward "
            f"(max |sum(parts) - whole| = {err:.3e})"
        )

        if self.body_ids is None:
            self.body_ids = contact_body_ids.cpu().numpy().copy()
        self.sim.append(sim.cpu().numpy())
        self.ref.append(ref.cpu().numpy())
        self.fp.append(fp.cpu().numpy())
        self.fn.append(fn.cpu().numpy())
        return out


def worst_body_err(current_rigid_body_pos, ref_rigid_body_pos):
    """Same arithmetic as ``terminations.compute_tracking_error`` (threshold 0.5),
    exposed as a weight-0 reward component so it rides the existing context
    plumbing instead of being re-derived from a second motion-lib query."""
    return (
        (ref_rigid_body_pos - current_rigid_body_pos).pow(2).sum(-1).sqrt().max(-1)[0]
    )


def resolve_motion_id(motion_lib, motion_arg: str) -> int:
    """Clip stem or integer -> motion id, with the available stems on failure."""
    files = [Path(str(f)).stem for f in motion_lib.motion_files]
    if motion_arg.isdigit():
        mid = int(motion_arg)
        assert 0 <= mid < len(files), f"motion id {mid} not in {list(enumerate(files))}"
        return mid
    matches = [i for i, f in enumerate(files) if f == motion_arg]
    if not matches:
        matches = [i for i, f in enumerate(files) if motion_arg in f]
    assert len(matches) == 1, (
        f"--motion {motion_arg!r} matched {len(matches)} clips; available: "
        f"{list(enumerate(files))}"
    )
    return matches[0]


def main():
    out_root = Path(args.out_dir)
    checkpoint = Path(args.checkpoint)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert resolved_configs_path.exists(), f"missing configs at {resolved_configs_path}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    resolved = torch.load(resolved_configs_path, map_location="cpu", weights_only=False)
    robot_config = resolved["robot"]
    simulator_config = resolved["simulator"]
    terrain_config = resolved.get("terrain")
    scene_lib_config = resolved["scene_lib"]
    motion_lib_config = resolved["motion_lib"]
    env_config = resolved["env"]
    agent_config = resolved["agent"]

    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    simulator_config.num_envs = args.num_rollouts
    simulator_config.headless = True
    motion_lib_config.motion_file = args.motion_file

    # Every rollout starts at t=0 so the time axis is comparable across envs.
    env_config.motion_manager.init_start_prob = 1.0
    env_config.motion_manager.resample_on_reset = True
    assert not env_config.termination_components, (
        "expected the frozen inference config to have terminations cleared; got "
        f"{list(env_config.termination_components)}"
    )

    from lightning.fabric import Fabric
    from protomotions.utils.fabric_config import FabricConfig

    fabric: Fabric = Fabric(
        **FabricConfig(
            accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[]
        ).as_kwargs()
    )
    fabric.launch()

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher = AppLauncher({"headless": True, "device": str(fabric.device)})
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # Weight-0 diagnostic: added before the env is built so it is part of the
    # component dict the ComponentManager iterates.
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    env_config.reward_components["diag_worst_body_err"] = MdpComponent(
        compute_func=worst_body_err,
        dynamic_vars={
            "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
            "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
        },
        static_params={"weight": 0.0},
    )

    from protomotions.utils.component_builder import build_all_components

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
        **simulator_extra_params,
    )

    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.utils.hydra_replacement import get_class

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    from protomotions.agents.base_agent.agent import BaseAgent

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent
    )
    agent.setup()
    agent.load(str(checkpoint), load_env=False, load_training_state=False)
    agent.eval()

    motion_lib = components["motion_lib"]
    motion_id = resolve_motion_id(motion_lib, args.motion)
    stem = Path(str(motion_lib.motion_files[motion_id])).stem
    N = args.num_rollouts

    mm = env.motion_manager
    assert mm.available_motion_ids is None, (
        "motion_manager is in subset mode; fixed motion ids would be ignored"
    )
    mm._setup_fixed_motion_ids(
        torch.full((N,), motion_id, dtype=torch.long, device=env.device)
    )

    # ------------------------------------------------------------------ #
    # Install probes (compile cache is still empty: no step has run).
    # ------------------------------------------------------------------ #
    rc = env.config.reward_components
    enc_probe = EncourageProbe(rc["pair_contact_rew"].compute_func)
    rc["pair_contact_rew"].compute_func = enc_probe
    fbd_probe = ForbidProbe(rc["pair_forbid_rew"].compute_func)
    rc["pair_forbid_rew"].compute_func = fbd_probe
    cm_probe = ContactMatchProbe(rc["contact_match_rew"].compute_func)
    rc["contact_match_rew"].compute_func = cm_probe
    env._component_manager._compiled.clear()

    obs, _ = env.reset()
    assert bool((mm.motion_ids == motion_id).all()), "clip pinning failed"
    assert float(mm.motion_times.abs().max()) == 0.0, "not every env started at t=0"

    dt = float(env.dt)
    clip_len = float(motion_lib.motion_lengths[motion_id])
    total_steps = int(clip_len / dt) - 1
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    log.info(
        "clip %s (motion_id %d): %.2f s, dt %.4f s -> %d steps x %d rollouts",
        stem,
        motion_id,
        clip_len,
        dt,
        total_steps,
        N,
    )

    reward_names, raw_hist, scaled_hist, total_hist, time_hist = None, [], [], [], []
    weights = None

    for step in range(total_steps):
        agent.pre_collect_step(step)
        obs = agent.add_agent_info_to_obs(obs)
        obs_td = agent.obs_dict_to_tensordict(obs)
        with torch.no_grad():
            model_outs = agent.model(obs_td)
        if args.stochastic or "mean_action" not in model_outs:
            action = model_outs["action"]
        else:
            action = model_outs["mean_action"]

        time_hist.append(mm.motion_times.cpu().numpy().copy())
        obs, _rew, dones, _term, extras = env.step(action)

        assert not bool(dones.any()), (
            f"env reset at step {step} -- the rollout window overruns the clip"
        )

        if reward_names is None:
            reward_names = sorted(
                k[len("raw_r/") :] for k in extras if k.startswith("raw_r/")
            )
            weights = np.array(
                [
                    float(rc[n].get_params().get("weight", 0.0))
                    for n in reward_names
                ],
                dtype=np.float32,
            )
        raw_hist.append(
            np.stack(
                [extras[f"raw_r/{n}"].cpu().numpy() for n in reward_names], axis=-1
            )
        )
        scaled_hist.append(
            np.stack(
                [
                    (
                        extras[f"scaled_r/{n}"].cpu().numpy()
                        if f"scaled_r/{n}" in extras
                        else np.zeros(N, dtype=np.float32)
                    )
                    for n in reward_names
                ],
                axis=-1,
            )
        )
        total_hist.append(extras["total_env_reward"].cpu().numpy().copy())

        if step % 100 == 0:
            log.info("step %d/%d", step, total_steps)

    # ------------------------------------------------------------------ #
    # Assemble. All arrays are [T, N, ...].
    # ------------------------------------------------------------------ #
    from examples.experiments.mimic.pair_contact_terms import _load_reftargets
    from examples.experiments.mimic.mlp_pair_contact_rew import REFTARGETS_PT

    rt = _load_reftargets(REFTARGETS_PT, env.device)
    body_names = list(robot_config.kinematic_info.body_names)
    scored = cm_probe.body_ids

    st = lambda xs: np.stack(xs, axis=0).astype(np.float32)  # noqa: E731
    out = {
        "motion_name": stem,
        "motion_id": motion_id,
        "num_rollouts": N,
        "dt": dt,
        "clip_length_s": clip_len,
        "checkpoint": str(checkpoint),
        "motion_file": args.motion_file,
        "stochastic": bool(args.stochastic),
        "t": (np.arange(total_steps, dtype=np.float32) + 1) * dt,
        "motion_time": st(time_hist),
        "reward_names": np.array(reward_names),
        "reward_weights": weights,
        "raw": st(raw_hist),
        "scaled": st(scaled_hist),
        "total": st(total_hist),
        # contact-match FP/FN
        "contact_body_names": np.array([body_names[i] for i in scored]),
        "sim_contact": st(cm_probe.sim),
        "ref_contact": st(cm_probe.ref),
        "contact_fp": st(cm_probe.fp),
        "contact_fn": st(cm_probe.fn),
        # pair encourage
        "pair_names": np.array(rt.pair_names),
        "pair_kappa": rt.pair_kappa.cpu().numpy(),
        "pair_is_ground": rt.ground_pair_ids.cpu().numpy(),
        "pair_gate": st(enc_probe.gate),
        "pair_gap": st(enc_probe.gap),
        "pair_phi": st(enc_probe.phi),
        "pair_load": st(enc_probe.load),
        "pair_sat": st(enc_probe.sat),
        "pair_contrib": st(enc_probe.contrib),
        # pair forbid
        "forbid_names": np.array(rt.forbid_names),
        "forbid_gate": st(fbd_probe.gate),
        "forbid_gap": st(fbd_probe.gap),
        "forbid_psi": st(fbd_probe.psi),
        "forbid_load": st(fbd_probe.load),
        "forbid_contrib": st(fbd_probe.contrib),
        # raw physics
        "ground_fz": st(enc_probe.ground_fz),
        "net_force_mag": st(enc_probe.net_fmag),
        "body_names": np.array(body_names),
        # kernel constants, so the plotter never hardcodes them
        "gap_target": enc_probe.inner.gap_target,
        "gap_sigma": enc_probe.inner.gap_sigma,
        "load_ref_n": enc_probe.inner.load_ref_n,
        "ground_ref_n": enc_probe.inner.ground_ref_n,
        "forbid_margin": fbd_probe.inner.margin,
        "probe_max_abs_err": max(
            enc_probe.max_abs_err, fbd_probe.max_abs_err, cm_probe.max_abs_err
        ),
    }

    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "rollout_rewards.npz", **out)
    log.info("wrote %s", out_dir / "rollout_rewards.npz")
    log.info(
        "decomposition check: max |sum(parts) - env reward| = %.3e "
        "(encourage %.3e, forbid %.3e, contact-match %.3e)",
        out["probe_max_abs_err"],
        enc_probe.max_abs_err,
        fbd_probe.max_abs_err,
        cm_probe.max_abs_err,
    )

    if hasattr(env.simulator, "shutdown"):
        env.simulator.shutdown()


if __name__ == "__main__":
    main()
