"""``render_policy_videos.py`` with the ghost showing the *reference*, for the G3 review videos.

The stock renderer poses the green ghost (``simulator.ghost_robot=True``) as the commanded slot-0 hold, and its
sphere markers show that hold too (``ContactGraphControl._update_ghost_char``). Every earlier review had to warn that
"inverted while the ghost stands" is not a failure, because the ghost runs seconds ahead of the clip. For judging
tracking of a clip (and of a synthesised edge) the reference itself is the comparison, so this wrapper:

* poses the ghost as the reference at the env's current clip time, with the same spawn offset the tracking error
  uses (``env.get_spawn_to_ref_pose_offset_with_terrain_height_correction``), so ghost and character share one world
  frame up to the ghost's display offset (``simulator.ghost_offset``, default +1.8 m in x);
* hides the goal sphere markers (moved 100 m below the floor; the keys stay, so the simulator's marker table is
  untouched);
* replaces the follow camera with a closer, level tracking shot: 4.5 m back, looking at the midpoint between the
  character and the ghost at 1.0 m height (frame -0.5 to 2.4 m, so a handstand's feet stay in view), following in XY
  only, so the horizon does not bob.

Everything else is ``render_policy_videos.py`` unchanged: one env, every clip from t = 0, deterministic actions,
the same flags. Run it exactly like that script, e.g.::

    DISPLAY=:0 PYTHONPATH=. ../env_isaaclab/bin/python expert_revist/graph_growth_2026_10_03/g3/render_review.py \\
        --checkpoint results/smpl_yogi_v2_expert56_g3_2f132f4299/epoch_3420.ckpt --simulator isaaclab \\
        --motion-ids 0 3 --fail-threshold 0 --min-seconds 0 --skip-existing \\
        --overrides env.ref_respawn_offset=0.005 simulator.ghost_robot=True --out-dir output/renderings/x
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCRIPTS = REPO / "data" / "scripts"
for p in (str(REPO), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import policy_setup  # noqa: E402  (no torch import: the render script launches the simulator first)

_original_build = policy_setup.build


def _patched_build(args, app_launcher_cls=None):
    built = _original_build(args, app_launcher_cls)
    import torch

    from protomotions.envs.control.contact_graph_control import ContactGraphControl
    from protomotions.simulator.base_simulator.simulator_state import ResetState

    def _update_ghost_char(self) -> None:
        simulator = getattr(self.env, "simulator", None)
        if simulator is None or not getattr(simulator, "ghost_enabled", False) or not self._initialized:
            return
        mm = self.env.motion_manager
        ref_state = self.env.motion_lib.get_motion_state(mm.motion_ids, mm.motion_times)
        offset = self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(ref_state.rigid_body_pos)
        reset_state = ResetState.from_robot_state(ref_state)
        reset_state.root_pos = reset_state.root_pos + offset[:, 0]
        active = torch.ones(self.env.num_envs, dtype=torch.bool, device=reset_state.root_pos.device)
        simulator.set_ghost_state(reset_state, active=active)

    original_markers = ContactGraphControl.get_markers_state

    def get_markers_state(self):
        markers = original_markers(self)          # also poses the ghost (patched above)
        for state in markers.values():
            if state.translation.numel():
                state.translation = state.translation.clone()
                state.translation[..., 2] = -100.0
        return markers

    ContactGraphControl._update_ghost_char = _update_ghost_char
    ContactGraphControl.get_markers_state = get_markers_state

    # A closer, level tracking shot framing both figures: the look-at point is the midpoint between the character
    # and the ghost (half the ghost's display offset), at a fixed height, and the camera follows it in XY only, so
    # the horizon does not bob when the root drops to the floor or rises into an inversion.
    import numpy as np

    from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator

    sim = built["simulator"]
    gx, gy = (float(v) for v in getattr(sim.config, "ghost_offset", (1.8, 0.0)))
    target_off = np.array([gx / 2.0, gy / 2.0, 0.0])

    def _look(self):
        root = self._get_simulator_root_state(self._camera_target["env"]).root_pos.cpu().numpy().reshape(-1)[:3]
        target = np.array([root[0], root[1], CAM_TARGET_Z]) + target_off
        return target + CAM_OFFSET, target

    def _init_camera(self) -> None:
        self._cam_prev_char_pos = self._get_simulator_root_state(0).root_pos.cpu().numpy()
        pos, target = _look(self)
        self._perspective_view.set_camera_view(pos, target)

    def _update_camera(self) -> None:
        pos, target = _look(self)
        self._perspective_view.set_camera_view(pos, target)

    IsaacLabSimulator._init_camera = _init_camera
    IsaacLabSimulator._update_camera = _update_camera
    print("render_review: ghost = reference at the current clip time; goal markers hidden; "
          f"camera {CAM_OFFSET.tolist()} from a look-at point {target_off.tolist()} + z {CAM_TARGET_Z}", flush=True)
    return built


CAM_OFFSET = __import__("numpy").array([0.0, -4.5, 0.4])    # camera position relative to the look-at point (m)
CAM_TARGET_Z = 1.0                                           # look-at height above the floor (m): frame -0.5 .. 2.4 m


policy_setup.build = _patched_build

if __name__ == "__main__":
    sys.argv[0] = str(SCRIPTS / "render_policy_videos.py")
    runpy.run_path(str(SCRIPTS / "render_policy_videos.py"), run_name="__main__")
