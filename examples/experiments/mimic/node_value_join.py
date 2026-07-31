# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join an **edge** to its target **node** using the node's own value function.

The problem this solves
-----------------------
``notes/Skill_graph_nodes.MD`` §6.3 measured, for both edges, a long interval in
which *no* node policy can stabilise the character -- 8.3 s of the 10.7 s
kick-up, 3.0 s of the 5.3 s exit. An edge controller therefore has to do more
than track its reference: it has to **finish somewhere the next node can catch
it**. Tracking error alone does not express that, and §6.3 showed why -- at
t = 2.0 s of the kick-up the character passes through a near-perfect down-dog
*pose* (0.141 m error) and the down-dog node still drops it in 4 steps, because
the limb is mid-swing. Pose proximity is not a hand-off condition; the target
node's own judgement of the state is.

So the edge is rewarded by **the target node's critic**, evaluated on the edge's
state:

    r_join(s) = exp( -((V_node(s) - V_nominal) / delta)^2 )

using the *band* form rather than a one-sided threshold, because §6.2 found
V-thresholding is anti-predictive on two of the three nodes (AUC 0.298 / 0.235)
while ``-|V - V_nominal|`` is consistently informative (AUC 0.887 / 0.923 / 0.930).
``V_nominal`` and ``delta`` come straight from ``value_band.json``.

How V_node is evaluated inside the edge env
-------------------------------------------
The node's critic takes ``[max_coords_obs, mimic_target_poses, previous_actions]``.
Two of those depend only on the robot state and are already identical in the edge
env (same factories, same 1027-wide layout). Only ``mimic_target_poses`` differs,
because it is built against the *reference*, and the edge env's reference is the
edge clip rather than the node's frozen pose.

That observation is a **pure function** of (current state, reference), so it is
recomputed here with the node's frozen pose substituted in, via the very same
kernel the env uses (``build_max_coords_target_poses``). No reimplementation, so
no drift between what the node critic was trained on and what it is fed.

The node's frozen pose lives in source-clip coordinates and must be placed in
this env's frame. ``ctx.respawn_root_offset`` is exactly that translation, and it
is valid to apply it to the node clip because the node and edge clips were cut
from one source with no re-centering (``build_skill_graph_clips.py``) -- so they
already share a coordinate frame.

Caveat, stated plainly: the certificate being optimised is imperfect (best band
accuracy 0.82 on the handstand node). Optimising an imperfect certificate can be
gamed. The trained edge must therefore be judged by the *empirical* hand-off test
(``handoff_check.py``), never by this reward.
"""

from typing import Optional

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent


class NodeValueJoin:
    """Callable that scores a state by the target node's critic.

    Holds the node's critic weights, its observation normaliser and its frozen
    reference pose. Instantiated once per experiment and used as an
    ``MdpComponent.compute_func``.
    """

    def __init__(
        self,
        node_checkpoint: str,
        node_motion_pt: str,
        v_nominal: float,
        delta: float,
        device: str = "cuda:0",
        clip_steps: Optional[int] = None,
        value_scale: Optional[float] = None,
        tail: str = "gaussian",
    ):
        self.node_checkpoint = node_checkpoint
        self.node_motion_pt = node_motion_pt
        self.v_nominal = float(v_nominal)
        self.delta = float(delta)
        self.clip_steps = clip_steps
        self.tail = tail
        self._device = device
        self._ready = False
        self._value_scale = value_scale

    # -- lazy build: the simulator/device does not exist at config time -------
    def _build(self, device):
        from protomotions.utils.hydra_replacement import get_class
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig

        ck = torch.load(self.node_checkpoint, map_location="cpu", weights_only=False)
        cfg_path = self.node_checkpoint.rsplit("/", 1)[0] + "/resolved_configs.pt"
        cfgs = torch.load(cfg_path, map_location="cpu", weights_only=False)
        critic_cfg = cfgs["agent"].model.critic

        CriticClass = get_class(critic_cfg._target_)
        critic = CriticClass(config=critic_cfg)

        # Pull just the critic's parameters out of the full model state dict.
        sub = {k[len("_critic.") :]: v for k, v in ck["model"].items()
               if k.startswith("_critic.")}
        from protomotions.agents.utils.normalization import (
            materialize_lazy_running_stats_from_state_dict,
        )
        materialize_lazy_running_stats_from_state_dict(critic, sub)
        if hasattr(critic, "materialize_from_state_dict"):
            critic.materialize_from_state_dict(sub)
        critic.load_state_dict(sub)
        critic.to(device).eval()
        for p in critic.parameters():
            p.requires_grad_(False)
        self.critic = critic
        self.in_keys = list(critic_cfg.in_keys)
        self.out_key = list(critic_cfg.out_keys)[0]

        if self._value_scale is None:
            rrn = ck.get("running_reward_norm")
            self._value_scale = (
                float(torch.sqrt(rrn["var"].double() + 1e-5).item())
                if rrn is not None and "var" in rrn else 1.0
            )

        # The node's frozen pose: one frame, constant over the whole clip.
        ml = MotionLib(MotionLibConfig(motion_file=self.node_motion_pt), device=str(device))
        st = ml.get_motion_state(
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, device=device),
        )
        self.ref_pos = st.rigid_body_pos[0].clone().to(device)        # [B,3]
        self.ref_rot = st.rigid_body_rot[0].clone().to(device)        # [B,4]
        self.ref_vel = torch.zeros_like(self.ref_pos)
        self.ref_ang_vel = torch.zeros_like(self.ref_pos)
        del ml
        self._ready = True

    # ComponentManager torch.compiles compute_funcs. This one is a stateful
    # object that lazily loads a checkpoint and runs a second nn.Module, so it is
    # forced to eager -- correctness over the ~15 % step cost of one extra critic
    # forward. (The manager does have a compile-failure fallback, but relying on
    # a silent fallback for correctness is not a good trade.)
    @torch._dynamo.disable
    def build_node_obs(
        self,
        body_pos: Tensor,
        body_rot: Tensor,
        body_vel: Tensor,
        body_ang_vel: Tensor,
        ground_height: Tensor,
        body_contacts: Tensor,
        historical_actions: Tensor,
        respawn_root_offset: Tensor,
        observe_contacts: bool = True,
        local_obs: bool = True,
        root_height_obs: bool = True,
        w_last: bool = True,
    ):
        """The node's observation for an arbitrary state, as a TensorDict.

        Shared by the reward path (critic) and by closed-loop composition, where
        the node's *actor* has to drive a character whose env reference is the
        edge clip. Validated to reproduce the node agent's own observation to
        7e-7 relative (see notes/Skill_graph_nodes.MD §7.1).
        """
        from tensordict import TensorDict
        from protomotions.envs.obs import (
            compute_humanoid_max_coords_observations,
            build_max_coords_target_poses,
            compute_historical_actions_from_state,
        )

        if not self._ready:
            self._build(body_pos.device)
        n = body_pos.shape[0]

        max_coords = compute_humanoid_max_coords_observations(
            body_pos=body_pos, body_rot=body_rot, body_vel=body_vel,
            body_ang_vel=body_ang_vel, ground_height=ground_height,
            body_contacts=body_contacts, local_obs=local_obs,
            root_height_obs=root_height_obs, observe_contacts=observe_contacts,
            w_last=w_last,
        )
        xy_offset = torch.zeros_like(respawn_root_offset)
        xy_offset[:, :2] = respawn_root_offset[:, :2]
        ref_pos = self.ref_pos.unsqueeze(0).expand(n, -1, -1) + xy_offset.unsqueeze(1)
        target_poses = build_max_coords_target_poses(
            current_state_body_pos=body_pos,
            current_state_body_rot=body_rot,
            current_state_body_vel=body_vel,
            current_state_body_ang_vel=body_ang_vel,
            mimic_ref_pos=ref_pos.unsqueeze(1),
            mimic_ref_rot=self.ref_rot.unsqueeze(0).expand(n, -1, -1).unsqueeze(1),
            mimic_ref_vel=self.ref_vel.unsqueeze(0).expand(n, -1, -1).unsqueeze(1),
            mimic_ref_ang_vel=self.ref_ang_vel.unsqueeze(0).expand(n, -1, -1).unsqueeze(1),
            with_velocities=True, w_last=w_last,
        )
        prev_actions = compute_historical_actions_from_state(
            historical_actions=historical_actions, history_steps=1
        )
        return TensorDict(
            {
                "max_coords_obs": max_coords,
                "mimic_target_poses": target_poses,
                "previous_actions": prev_actions,
            },
            batch_size=n,
            device=body_pos.device,
        )

    @torch._dynamo.disable
    def __call__(
        self,
        body_pos: Tensor,
        body_rot: Tensor,
        body_vel: Tensor,
        body_ang_vel: Tensor,
        ground_height: Tensor,
        body_contacts: Tensor,
        historical_actions: Tensor,
        respawn_root_offset: Tensor,
        progress_buf: Tensor,
        observe_contacts: bool = True,
        local_obs: bool = True,
        root_height_obs: bool = True,
        w_last: bool = True,
    ) -> Tensor:
        _unused = (observe_contacts, local_obs, root_height_obs, w_last)
        td = self.build_node_obs(
            body_pos=body_pos, body_rot=body_rot, body_vel=body_vel,
            body_ang_vel=body_ang_vel, ground_height=ground_height,
            body_contacts=body_contacts, historical_actions=historical_actions,
            respawn_root_offset=respawn_root_offset,
            observe_contacts=observe_contacts, local_obs=local_obs,
            root_height_obs=root_height_obs, w_last=w_last,
        )
        with torch.no_grad():
            v = self.critic(td)[self.out_key].squeeze(-1) * self._value_scale
        self.last_v = v
        x = (v - self.v_nominal) / self.delta
        if self.tail == "cauchy":
            # Heavy tails. The Gaussian form was measured to die on edge_exit:
            # tadasana's V falls steeply off-distribution, and at V = 96.9
            # exp(-((96.9-105.03)/2.45)^2) = 1.7e-5 -- numerically zero, so the
            # policy got no gradient telling it which way to move and never came
            # back. Cauchy decays polynomially: the same point scores 0.083.
            r = 1.0 / (1.0 + x ** 2)
        else:
            r = torch.exp(-(x ** 2))
        if self.clip_steps is not None:
            # Quadratic ramp over the clip rather than a hard terminal gate.
            #
            # A hard gate ("pay only in the last 30 steps") is unlearnable here:
            # an untrained edge dies around step 65 of 320, so the term is
            # identically zero and never shapes anything -- measured, not
            # assumed. The ramp still concentrates almost all the weight at the
            # end (0.25 at the midpoint, 1.0 at the last step) while giving a
            # gradient as soon as the policy survives far enough to matter.
            #
            # Distortion risk is low even early, because the band reward is
            # already self-localising: far from the node pose the target node's
            # V is nowhere near V_nominal, so exp(-(.)^2) is ~0 regardless.
            ramp = (progress_buf.float() / float(self.clip_steps)).clamp(0.0, 1.0) ** 2
            r = r * ramp
        return r


def node_value_join_rew_factory(
    node_checkpoint: str,
    node_motion_pt: str,
    v_nominal: float,
    delta: float,
    weight: float = 1.0,
    clip_steps: Optional[int] = None,
    tail: str = "gaussian",
) -> MdpComponent:
    """Reward the edge for ending inside the target node's certified V band."""
    fn = NodeValueJoin(
        node_checkpoint=node_checkpoint,
        node_motion_pt=node_motion_pt,
        v_nominal=v_nominal,
        delta=delta,
        clip_steps=clip_steps,
        tail=tail,
    )
    comp = MdpComponent(
        compute_func=fn,
        dynamic_vars={
            "body_pos": EnvContext.current.rigid_body_pos,
            "body_rot": EnvContext.current.rigid_body_rot,
            "body_vel": EnvContext.current.rigid_body_vel,
            "body_ang_vel": EnvContext.current.rigid_body_ang_vel,
            "ground_height": EnvContext.ground_heights,
            "body_contacts": EnvContext.body_contacts,
            "historical_actions": EnvContext.historical.actions,
            "respawn_root_offset": EnvContext.respawn_root_offset,
            "progress_buf": EnvContext.progress_buf,
        },
        static_params={
            "observe_contacts": True,
            "local_obs": True,
            "root_height_obs": True,
            "w_last": True,
            # consumed by combine_rewards, not forwarded to compute_func
            "weight": weight,
        },
    )
    return comp
