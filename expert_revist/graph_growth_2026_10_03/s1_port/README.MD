# S1, the student port to the release-v3 expert, and S0's three fixes (2026-10-05)

Card S1 of [`../PLAN.MD`](../PLAN.MD) (Lane S). The teacher is G3's `epoch_3420.ckpt`
(`results/smpl_yogi_v2_expert56_g3_2f132f4299`, [`../g3/README.MD`](../g3/README.MD)); the release is
`holds_repaired_ftC_posefix.release_v3.2f132f4299`. S0's record is [`../s0_x0/README.MD`](../s0_x0/README.MD).

## 0. Bottom line

1. **Both of the card's parity tests pass.**
   - **Expert observations are bit-identical between the expert's env and the student env on the same state.** In
     the student env, with the frozen expert driving, each of G3's 12 actor inputs is compared with what the expert's
     own control and own components compute on the same state. They agree to the bit (max |Δ| 0) over 1,497
     comparisons of 4,096 envs, in three phases (§2):
     - every clip from t = 0;
     - the student's own resets (1,917 env resets);
     - manual goals on 42 route plans.
   - **The student env, with the expert acting, reproduces the expert's own evaluation scores** (§3). G3's own
     evaluator in the student env scores perf 0.9884 and success 0.9603. In the expert's own env, with the same seed,
     it scores 0.9851 / 0.9563 and 0.9847 / 0.9563. Per motion, 244 of 252 scores are identical (241 between the
     expert's own two runs). The 4 that differ by more than 0.01 are clips that flip between evaluations.
     With both envs given the same seeded warm-up first (identical PhysX history), the student env
     is indistinguishable from the expert's own same-seed rerun on every statistic: 245 identical motion scores
     against 241, the same 3 / 1 / 0 motions beyond 0.01 / 0.05 / 0.2, and the median first divergence at step 308
     against 282. Neither pair is bit-identical: GPU PhysX does not reproduce across processes even in the expert's
     own env.
2. **The decision the card asked for: G3 e3420's lost Warrior III, Tree and Side Plank forks were the plans', not
   the teacher's** (§4).
   - On plans with training's deadline (route plans), all three succeed on every replica (1.00; they were 0.00 /
     0.05 / 0.00).
   - G3 e3420 holds the commanded pose in **35 of 36** scorable family forks, and **no family falls** (14 fell under
     the shipped plans). Family success goes 13 → 28 of 36.
   - The 2 × 2 attributes it: the plan carries it (13 → 27 with the legacy timing protocol); the training timing
     protocol adds one more (28); the timing alone on the shipped plans gives 14; the rebuild fix alone reproduces
     S0's 13.
3. **What the student will inherit** (§4.2). Seven families hold the commanded pose (hold rate 1.00, 0.03–0.05 m)
   but do not load a labelled support:
   - Bridge (right forearm), Cobra (thighs), Dolphin Plank (forearms), Legs-up (trunk, head), Pincha (hands: it
     stands on its forearms only), Koundinya (head), Supported Shoulderstand (trunk);
   - plus Half Moon 0.63, Plow 0.58 and Upward Plank 0.42.

   These are the teacher's support habits, the same kind as G3's open substitution gate (7.7 against ≤ 6).
4. **S0's three fixes are in, and each is measured** (§5).
   - The shared rebuild is exact after a step and after a reset (0.0 on every observation key). The naive rebuild is
     off by up to 0.78–1.0 in `contact_obs_v1` after a step, and **by up to 1.0 after a reset**, a second trap that
     every panel and probe tool hit at its first step.
   - `goal_causality_x0.py`'s manual-path check reports no differing actor input with the new dwell duration. The
     legacy call still changes `contact_goal_obs` by 1.0, as S0 found.
   - X0 re-runs to S0's numbers: excess 0.214 / 0.310 / 0.241 σ_a against 0.212 / 0.308 / 0.236.
5. **The port runs end to end.** A 4-epoch training smoke (256 envs, DAgger engaged from epoch 0) has finite losses
   on all 97 logged scalars, and the imitation and DAgger losses fall (0.0131 → 0.0081, 0.0127 → 0.0078). Release
   v3's `--check` still passes 68 of 68: no builder was touched. 42 new CPU tests; the full suite is 1,816 passed,
   with the 2 failures that were there before.
6. **For S2.** The launcher and the decisions it carries are in §6, with what is still open (the student
   evaluator's 20 s window, the panel's cost, pushes).

## 1. What was built

### 1.1 S0's three fixes

**Rebuilding observations (S0 trap 1).** `BaseEnv.rebuild_observations()` recomputes every observation exactly as
the latest build did. Two inputs have moved since that build, and a naive rebuild
(`_build_global_context(get_robot_state())` + `compute_observations`) reads both wrong:
- after `env.step`, `_finalize_contact_state` has already promoted this step's force to `previous_contact_forces`,
  so the force rate reads 0 (S0's measured trap);
- after `env.reset`, the reset's build used a cleared contact sample, but IsaacLab does not refresh contact sensors
  on a state write, so the live state still carries the pre-teleport forces. **This second trap is new**: every
  panel and probe tool rebuilt right after its reset.

`post_physics_step` keeps a clone of the contact sample its build read (three small tensors per step), and `reset`
marks the rows it built. The panel (`SequenceVizRunner._refresh_obs`), the renderer
(`render_contact_goal_sequence.py`), the swap tools (`far_swap_test.py`, `hub_swap_test.py`,
`expert_handoff_screen.py`) and S0's harness (`goal_causality_x0.Harness.rebuild`) call it. Older analysis scripts
that rebuild for their own purposes (`decode_all_codes.py`, `check_student_xy_invariance.py`,
`check_realign_teacher_impact.py`, `query_contact_goal.py`, ...) are left as they were, so their recorded results
reproduce.

**The dwell channels of a manual goal (S0 trap 2).** `ContactGraphControl.set_manual_goal` takes two new arguments:
- `hold_duration` feeds the `duration` channel and stays constant; `hold_seconds` keeps feeding `remaining`, which
  counts down. Omitted, `hold_duration` reuses `hold_seconds` (the pre-S1 behaviour, kept so old probe results
  reproduce);
- `deadline_floor`: the deadline counts down to this and stops. Omitted, it is `min_lead_s` (the legacy schedule's
  floor). Under `include_current_segment` a reached goal's deadline is 0 in training.

The panel gets `SequenceVizConfig.timing` (default `"legacy"`, the shipped protocol). `"training"` serves each plan
goal as the segment a scheduled slot carries, with `t_hold` = the goal's reach end and `t_end` = its end:
deadline `max(t_hold - t, 0)`, duration `hold_s`, remaining `max(t_end - t, 0)`, floor 0, and the slots advance on
the step a goal ends instead of at the next 0.5 s re-arm. It refuses a control trained without
`include_current_segment`. The same option is on `run_sequence_panel.py --timing`,
`render_contact_goal_sequence.py --timing` and `goal_causality_x0.py --fork-timing`.

**Fork plans on training's deadline (S0's third finding).** `data/scripts/make_route_probe_plans.py` writes
`data/scripts/plans_release_v3_route/`: 37 `fork_<family>` plans and 5 `route_edge_<edge>` plans. A route plan
starts in its clip at t = 0 and commands the clip's own segments in order, through the target hold and the next
hold:
- goal k: `reach_s = t_hold[k] - t_end[k-1]` and `hold_s = t_end[k] - t_hold[k]`, so each goal's reach ends at its
  hold frame and the goal ends at its segment's end, on the clip's own clock;
- under `timing="training"` that is exactly the window the interval schedule served at that clip time;
- the family plans take the release's fork exemplar (the same clip and hold frame as the release's plan, checked
  for all 37); the edge plans take each edge's plan clip (`make_edge_probe_plans.plan_clips`);
- each plan names its scored goal (`target`: index, node key, hold window).

`make_hold_graph_probe_plans.py` is one of release v3's hash-pinned builders (`release_v3 --check` recomputes the
release id from the builders' sources), so it is untouched and the new generator is a separate file. The release's
own plans stay as they are. Warrior III -b, for example, becomes standing (0.88 s) -> h1 (reach 0.68 s, hold 0.68 s)
-> h2 (reach 1.87 s, hold 0) -> Warrior III (reach 9.62 s to the frame at 13.73 s, hold 1.22 s) -> standing,
against the release plan's single Warrior III goal at a 3.25 s reach and a 4 s hold.

### 1.2 The port

**The expert's view of the goal.** G3's actor reads 12 inputs; six of them are the goal (`masked_mimic_target_*`,
`contact_goal_*`), and its copied components were bound to `ctx.masked_mimic` / `ctx.contact_goal`. In the
student env those carry the student's view: 5 slots, a 6-body subset, masked. G3 was trained on 2 slots x 24
bodies, always visible.
- `ContactGraphControlConfig.expert_view_steps / expert_view_num_bodies / expert_view_contract`: the control
  publishes slots `[:2]` of its own window as `ctx.expert_masked_mimic` / `ctx.expert_contact_goal`, with all-ones
  body masks over 24 bodies, `visible = goal_valid`, and the same gathered contacts, orientations, dwell channels
  and deadlines. `EnvContext` has the two new fields.
- `ContactGraph.next_goal_indices` serves a prefix of the same window for any slot count (`[head, head+1, ...]`),
  so the slice is the expert's own window.
- At construction the control checks the contract recorded from the expert's resolved config: the slot count; full
  visibility in training; the five schedule keys (`include_current_segment`, `dwell_channels`, `min_lead_s`,
  `interval_schedule`, `history_time_clip_s`); no far-goal promotion; the env's `ref_respawn_offset`, contact
  hysteresis, reference smoothing, realignment rule and history depth; the graph's sha256 and pair vocabulary.
  Inference builds re-check it.
- `protomotions/agents/supervised/expert_port.py`: `expert_goal_contract` (from the expert's resolved config),
  `rewire_expert_goal_view` (the copied components' bindings move from `masked_mimic.*` / `contact_goal.*` to the
  expert view), `goal_view_bindings` (what still reads the student's view: must be empty) and
  `port_identity_problems`.

**The student's schedule has the expert's semantics.** The experiment copies the five schedule keys from the
expert's control config (`include_current_segment` True, `dwell_channels` True, `min_lead_s` 0.2,
`interval_schedule` True, `history_time_clip_s` 10.0) and the control asserts them.

**The merge-loop crash** (`contact_graph_transformer.env_config`): the expert's `future_steps` feeds its
`mimic_target_poses`, which a Design-B expert reads in its critic only. The loop now raises the student's window
only when the expert's actor reads that input. For G3 the window is `[1, 5, 10, 15]`, a list, which the old int
comparison crashed on.

**Identity asserts** (`port_identity_problems`, at config time):
- robot `SmplYogiV2RobotConfig`, its assets and plant sha256;
- the simulator's rates and PhysX parameters;
- the motion package by sha256;
- the graph by sha256 and pair vocabulary;
- one expert, and an all-zeros routing table over the graph's motions in order
  (`data/scripts/make_single_expert_routing.py` ->
  `data/smpl/student_release_v3/holds_repaired_ftC_posefix.release_v3.2f132f4299.g3_e3420.experts.json`).

At runtime the control's release record re-verifies the package, graph, physics tables and sidecar by sha256, and
`MultiExpertSupervisedAgent` checks the routing table's names against the library.

**The experiment and its launcher.**
`examples/experiments/masked_mimic/contact_graph_fsq_release_v3.py` is v9's recipe with the port, and
`data/scripts/run_student_distill_release_v3.sh` launches it (`CONFIG_ONLY`, `SMOKE`, `DRY_RUN`, `RESUME`; it
refuses an existing `last.ckpt`). The student env inherits from the teacher: its graph, release record, physics
tables and sidecar, `ref_respawn_offset`, contact hysteresis, reference smoothing (7) and realignment rule (off). Its
panel runs the five `route_edge_*`, the five `edge_*` and six route forks on `timing="training"`. At inference the
experts, their components and the expert view are dropped.

## 2. Parity test 2: the expert's observations, bit for bit

`data/scripts/s1_port_parity.py --test obs` builds the student env from the `CONFIG_ONLY` run
(`results/smpl_yogi_v2_student_release_v3_s1_parity`; no student weights, the expert loads in `create_model`) at
4,096 envs. The frozen expert drives: the action is its mean action on its `expert_*` observations (what
`rollout_actor=EXPERT` steps with).

**The reference computation.** At every step the expert's 12 actor inputs are recomputed the expert's own way on the
same state:
- a twin of the expert's `ContactGraphControl`, built from G3's resolved control config (2 slots, probabilities 1.0)
  with the robot's goal bodies set to G3's 24 for its construction, attached to the same env and reset and stepped
  as the env steps its own;
- G3's own observation components (un-prefixed, bound to `ctx.masked_mimic` / `ctx.contact_goal`), executed through
  their own `ComponentManager`, so they are compiled as in the expert's env.

| Phase | Steps | What it covers | max \|Δ\| over the 12 inputs |
|---|---|---|---|
| clip starts | 300 | every motion from t = 0 (`motion = env % 252`) | 0 |
| training resets | 300 | the student's own motion manager and terminations; 1,917 env resets, each rebuilt and compared | 0 |
| manual goals | 300 | the 42 route plans through `set_manual_goal`, `timing="training"`: the student's 5 slots, the twin's 2 | 0 |

1,497 comparisons, each over 4,096 envs and 12 inputs, all exactly equal (`data/parity_obs.json`).

**The rebuild self-checks** ran beside it at every step and reset:
- `env.rebuild_observations()` equals the env's own observation on every key, student and expert (0.0 after 900
  steps and 296 resets);
- the pre-S1 naive rebuild differs in `contact_obs_v1` by up to 1.0 after a step, and by up to 1.0 on the rows just
  reset.

**The harness hit S0's trap 1 itself first.** The first run compared the twin against `env._current_context` as
`env.step` left it. Its `previous_contact_forces` tensor had already been overwritten in place by
`_finalize_contact_state`, so the twin's `contact_obs_v1` read a zero force rate. It differed by up to 1.0, and only
on that key, in every phase (`output/s1_port/obs/obs_first_run_harness_trap.json`). The twin now evaluates on the
rebuilt context, the one the observation was built from. Any tool that recomputes from the cached context after a
step has this trap, not only the rebuilders.

## 3. Parity test 1: the expert's own evaluation scores

`s1_port_parity.py --test eval` runs G3's `HoldCurriculumEvaluator`:
- its training config's evaluator: release v3's hold manifest, support rule v2, 2,250 steps, 4,096 envs, every
  motion from t = 0;
- the expert acting in both envs: the PPO model's mean action in the expert's env, the expert actor on `expert_*` in
  the student env;
- in the student env, G3's terminations (0.5 m) and episode length for the evaluation;
- Python, NumPy and torch seeded identically just before the evaluation, and in the last pair both envs first
  given the same seeded warm-up (`--warmup-steps 5`);
- every active env's action recorded at every step.

| Pair | Identical motion scores | > 0.01 / > 0.05 / > 0.2 | Mean \|Δ score\| | Motions never diverging (\|Δa\| > 1e-3) | First divergence step p10 / p50 | Perf score | Success |
|---|---|---|---|---|---|---|---|
| expert vs expert, same seed | 241 / 252 | 3 / 1 / 0 | 0.00047 | 150 | 137 / 282 | 0.98511 / 0.98473 | 0.9563 / 0.9563 |
| expert vs student | 244 / 252 | 4 / 2 / 1 | 0.00395 | 98 | 2 / 86 | 0.98511 / 0.98837 | 0.9563 / 0.9603 |
| expert rerun vs student | 241 / 252 | 4 / 3 / 1 | 0.00430 | 97 | 2 / 79 | 0.98473 / 0.98837 | 0.9563 / 0.9603 |
| **expert vs student, both warmed up** | **245 / 252** | **3 / 1 / 0** | **0.00066** | **143** | **142 / 308** | **0.98487 / 0.98530** | **0.9563 / 0.9563** |

(`data/parity_eval_pairs.json`; the runs' raw output is in `output/s1_port/eval_*`.)

- **Without a warm-up the student env diverges earlier** (p10 at step 2 against 137), though its scores are just as
  close. Actions are identical for the first two steps, and §2 shows the inputs are identical on a shared state, so
  the state itself drifts. Each env enters the evaluation with its own PhysX history: the construction-time reset
  is drawn by its own motion manager (arrival anchoring 0.6 in the student, 0.4 / 0.2 in the expert), and PhysX
  warm-starts its contacts. `--warmup-steps 5` resets every env to its clip's t = 0 under one seed and steps five
  zero actions in both. The early divergence disappears: the port adds nothing beyond the expert's own floor.
- **The four motions beyond 0.01 are clips that flip between evaluations:**
  - Half Moon -b x3s (0.11 against 1.00), as at G3's 3,420 standalone;
  - Upward Plank -a's exit;
  - Scorpion -a x3s, in the expert's own rerun.
- Group scores agree within 0.002 in the warmed pair: single-leg 0.9545 / 0.9545, inversion 0.9732 / 0.9731, arm
  balance 0.9979 / 0.9980, connective 0.9852 / 0.9869, edges 1.00 / 1.00. Jerk is 54.7 / 54.3 and drag 40.2 /
  39.7 J.

## 4. The fork battery on training's deadline

`goal_causality_x0.py --parts forks` on G3 e3420, 4,096 envs, in four arms against S0's run: the release's plans
or the route plans, each under the legacy or the training timing. Collated by `forks_compare.py` into
`data/forks_compare.json`. Success is the battery's own rule over the goal's hold window: pose held
(< 0.15 m, best yaw, 24 bodies), every commanded zone down, no other zone loaded.

| Arm | Plans | Timing | Code | Family forks succeeding (≥ 0.5 of replicas) | Pose held | Families that fell |
|---|---|---|---|---|---|---|
| S0 | release | legacy | S0's | 13 of 37 | 20 | 14 |
| release_legacy | release | legacy | S1 (rebuild fix) | 13 of 37 | 20 | 14 |
| release_training | release | training | S1 | 14 of 37 | 20 | 15 |
| route_legacy | route | legacy | S1 | 27 of 36 | 35 | 0 |
| **route_training** | route | training | S1 | **28 of 36** | **35** | **0** |

- "Fell" is a pelvis below 0.2 m at the hold with the pose not held.
- Peacock's family hold has its exemplar frame at the segment's end (`t_hold = t_end`), so its route hold window is
  empty and the battery does not score it.
- Over its whole segment its supports hold on 0.85 of replicas (`segment_supports`, the support rule over
  `[t_start, t_end]` from the saved traces). Counting it, 28 of 37 families hold their supports over the whole
  target segment.

### 4.1 Per family

| Family | S0 | release, legacy | release, training | route, legacy | **route, training** | route, training: pose held / error (m) / pelvis z (m) |
|---|---|---|---|---|---|---|
| Bridge | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 / 0.046 / 0.29 |
| Cobra | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 / 0.027 / 0.10 |
| Crow | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.031 / 0.73 |
| Dolphin Plank | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 / 0.029 / 0.28 |
| Dolphin | 1.00 | 1.00 | 0.00 | 1.00 | **1.00** | 1.00 / 0.029 / 0.70 |
| Downward Facing Dog | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.028 / 0.79 |
| Eagle | 0.00 | 0.00 | 0.00 | 1.00 | **1.00** | 1.00 / 0.041 / 0.86 |
| Extended Revolved Side Angle | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.039 / 0.72 |
| Extended Revolved Triangle | 0.05 | 0.05 | 0.50 | 1.00 | **1.00** | 1.00 / 0.051 / 0.78 |
| Feathered Peacock (Pincha) | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 / 0.037 / 0.79 |
| Firefly | 0.00 | 0.00 | 0.00 | 1.00 | **1.00** | 1.00 / 0.073 / 0.40 |
| Four-Limbed Staff (chaturanga) | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.020 / 0.28 |
| Garland | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.036 / 0.26 |
| Half Moon | 0.95 | 0.88 | 0.29 | 1.00 | **0.63** | 0.63 / 0.076 / 0.82 |
| Handstand | 0.00 | 0.00 | 0.00 | 1.00 | **1.00** | 1.00 / 0.042 / 1.05 |
| Intense Side Stretch | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.033 / 0.81 |
| Legs Up the Wall | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 / 0.029 / 0.09 |
| Lord of the Dance | 0.00 | 0.00 | 0.00 | 1.00 | **1.00** | 1.00 / 0.030 / 0.95 |
| Low Lunge | 0.00 | 0.00 | 0.00 | 1.00 | **1.00** | 1.00 / 0.048 / 0.34 |
| Peacock | 0.00 | 0.00 | 0.00 | — | **—** | hold window empty; supports over the segment 0.85 |
| Plank | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.029 / 0.49 |
| Plow | 0.00 | 0.00 | 0.00 | 0.45 | **0.58** | 1.00 / 0.065 / 0.36 |
| Koundinya | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 / 0.030 / 0.48 |
| Scorpion | 0.00 | 0.00 | 0.00 | 0.00 | **0.94** | 0.94 / 0.058 / 0.75 |
| Shoulder Pressing | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.030 / 0.34 |
| Side Crane Crow | 0.00 | 0.00 | 0.00 | 1.00 | **1.00** | 1.00 / 0.034 / 0.38 |
| Side Plank | 0.00 | 0.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.043 / 0.75 |
| Standing Forward Bend | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.021 / 0.85 |
| Standing Split | 0.10 | 0.23 | 1.00 | 1.00 | **1.00** | 1.00 / 0.064 / 0.89 |
| Big Toe Hold | 0.00 | 0.00 | 0.00 | 1.00 | **1.00** | 1.00 / 0.030 / 0.99 |
| Supported Headstand | 0.00 | 0.00 | 0.00 | 1.00 | **1.00** | 1.00 / 0.037 / 0.76 |
| Supported Shoulderstand | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 / 0.040 / 0.41 |
| Tree | 0.05 | 0.03 | 0.00 | 1.00 | **1.00** | 1.00 / 0.043 / 0.98 |
| Upward Plank | 0.00 | 0.00 | 0.00 | 1.00 | **0.42** | 0.48 / 0.167 / 0.42 |
| Warrior III | 0.00 | 0.00 | 0.00 | 1.00 | **1.00** | 1.00 / 0.033 / 0.90 |
| Warrior II | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.021 / 0.74 |
| Reverse Warrior | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 / 0.032 / 0.76 |

- **Warrior III, Tree and Side Plank are 1.00 on route plans in both timings**, as are Handstand, Supported
  Headstand, Lord of the Dance, Firefly, Side Crow, Standing Split, Big Toe Hold, Low Lunge, Eagle (no kickstand:
  the support rule passes) and Extended Revolved Triangle (its hand reaches the floor). On the shipped plans most of
  these fell.
- The five `route_edge_*` plans (each edge on its own training clip from t = 0) succeed on every replica in both
  timings, and the release's `edge_*` / `nohijack_*` plans stay at 1.00 in every arm.
- Timing does matter for some plans; on the shipped plans it moves outcomes both ways.
  - `release_training` gains Side Plank and Standing Split; it loses Dolphin and Half Moon. `fork_edge_E3` goes
    0.00 → 1.00 and `fork_edge_E5` 0.54 → 0.00.
  - On route plans the two timings agree except Scorpion (0.00 → 0.94), Half Moon (1.00 → 0.63) and Upward Plank
    (1.00 → 0.42), the last two the clips that flip between evaluations.

### 4.2 What does not pass, and why

Under `route_training` every failing family but two holds the commanded pose (hold rate 1.00, error 0.03–0.05 m):

| Family | Commanded zones | Not loaded (replicas of 71) | Loaded instead |
|---|---|---|---|
| Bridge | feet, trunk, head, upper arms, forearms | R_FOREARM 71 | the rest |
| Cobra | feet, shanks, thighs, hands | thighs 71, feet 58/20 | shanks + hands |
| Dolphin Plank | feet, forearms, hands | forearms 71/57 | feet + hands (the experts' known way) |
| Legs-up | pelvis, trunk, head, arms | trunk, head, L_FOREARM 71 | pelvis + arms |
| Pincha (Feathered Peacock) | forearms, hands | hands 71 | forearms only |
| Koundinya | feet, head, hands | head 71 | feet + hands |
| Supported Shoulderstand | trunk, head, upper arms | trunk 71 | head + upper arms |
| Plow (0.58) | feet, trunk, head, upper arms | trunk 30 | |
| Half Moon (0.63) | L_FOOT, R_HAND | R_HAND 21 | 11 replicas on the left side of the body |
| Upward Plank (0.42) | feet, hands | 22–35 per zone | 18 on the trunk and upper arms |

These are support habits of the teacher, and a student distilled from it will copy them. S0 had already recorded
the first four, and G3's substitution gate is open (`../g3/README.MD` §6).

## 5. S0's fixes, re-measured with `goal_causality_x0.py`

`--parts corpus x0` on G3 e3420 with the S1 code (`data/x0_g3_e3420_summary.json`):

| Check | S0 | S1 |
|---|---|---|
| Rebuild of the own window against the stepped observation (172 checks) | 0.0 (the harness's own restore) | **0.0** (`env.rebuild_observations`) |
| Naive rebuild, `contact_obs_v1` | 0.78 | 0.78 |
| Manual path, legacy call: differing actor inputs | `contact_goal_obs` (1.0) | `contact_goal_obs` (1.0); action RMS p50 1.3e-3 raw |
| **Manual path, S1 call** (`hold_duration = t_end - t_hold`, `deadline_floor` 0) | — | **none** (action difference 0) |
| Schedule restored after either call | exactly | exactly |
| X0 excess near the end of the dwell: standing / crow / handstand | 0.212 / 0.308 / 0.236 | 0.214 / 0.310 / 0.241 |

The X0 numbers move by at most 0.005 σ_a. That is the cross-process nondeterminism of §3, not a change.

## 6. For S2

- **Launch:** `bash data/scripts/run_student_distill_release_v3.sh` (wandb; 1,024 envs, batch 8,192, 8,000 epochs,
  a checkpoint every 500). `CONFIG_ONLY=1`, `SMOKE=1` (it writes `results/smpl_yogi_v2_student_release_v3_g3e3420_smoke`,
  which a second smoke refuses until it is moved), `DRY_RUN=1` and `RESUME=1` work. S2's recipe is set: one-token
  code, DAgger action loss 0.1, prior rollouts 0.25 from epoch 1,000 (ramp 500), fixed sampling, root-relative
  student targets, no pushes.
- **The student's `eval/success_rate` window is 600 steps (20 s).** The release's longest clip is 63 s. S2 should
  decide whether to raise `max_eval_steps` (the release record asks for at least 1,890). `eval/success_rate` is not
  a quality signal for the student anyway.
- **Probes and panels on training's timing.** For a policy trained with `include_current_segment`, pass
  `timing="training"`: `SequenceVizConfig.timing`, `run_sequence_panel.py --timing training`,
  `render_contact_goal_sequence.py --timing training` and `goal_causality_x0.py --fork-timing training`. Use plans
  that mirror a clip (`plans_release_v3_route/`). The release's own plans stay as they are.
- **Score the student's forks against the teacher's** `route_training` row (§4), with the support habits of §4.2
  as the teacher's baseline. S2's acceptance ("hold rates within 0.1 of the teacher") needs exactly that table.
- **What S1 did not do:** no student was trained beyond the smoke; no partial-goal probes were written; the
  inference config drops the expert view, and nothing else changes for deployment.

## 7. Reproduce

```bash
PY=../env_isaaclab/bin/python; G3=results/smpl_yogi_v2_expert56_g3_2f132f4299; O=output/s1_port
R=data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v3.2f132f4299
# CPU: plans, routing table, resolved configs, tests, the release's own check
PYTHONPATH=.:data/scripts $PY data/scripts/make_route_probe_plans.py --graph $R/contact_graph.json \
    --manifest $R/holds_extended.yaml --out-dir data/scripts/plans_release_v3_route
PYTHONPATH=. $PY data/scripts/make_single_expert_routing.py --graph $R/contact_graph.pt --expert $G3/epoch_3420.ckpt \
    --out data/smpl/student_release_v3/holds_repaired_ftC_posefix.release_v3.2f132f4299.g3_e3420.experts.json
PY=$PY CONFIG_ONLY=1 WANDB=0 EXPERIMENT=smpl_yogi_v2_student_release_v3_s1_parity bash data/scripts/run_student_distill_release_v3.sh
$PY -m pytest -q protomotions/tests/test_s1_student_port.py
PYTHONPATH=.:data/scripts OMP_NUM_THREADS=1 $PY -m reference_curation.release_v3 --check holds_repaired_ftC_posefix.release_v3.2f132f4299
# GPU, one IsaacLab process at a time
PYTHONPATH=. $PY data/scripts/s1_port_parity.py --test obs --out-dir $O/obs
for E in expert student; do PYTHONPATH=. $PY data/scripts/s1_port_parity.py --test eval --env $E --out-dir $O/eval_$E; done
PYTHONPATH=. $PY data/scripts/s1_port_parity.py --test eval --env expert --out-dir $O/eval_expert_rerun   # the floor
for E in expert student; do PYTHONPATH=. $PY data/scripts/s1_port_parity.py --test eval --env $E --warmup-steps 5 \
    --out-dir $O/eval_${E}_warm; done
$PY data/scripts/s1_port_parity.py --test compare --runs $O/eval_expert_warm $O/eval_student_warm
PYTHONPATH=. $PY data/scripts/goal_causality_x0.py --checkpoint $G3/epoch_3420.ckpt --parts corpus x0 --out-dir $O/x0_g3_e3420
PYTHONPATH=. $PY data/scripts/goal_causality_x0.py --checkpoint $G3/epoch_3420.ckpt --parts forks --fork-timing training \
    --fork-max-seconds 55 --fork-plans data/scripts/plans_release_v3_route/*.json $R/plans/{edge,nohijack,fork_edge}_*.json \
    --out-dir $O/forks_route_training          # and the other three arms of §4
$PY expert_revist/graph_growth_2026_10_03/s1_port/forks_compare.py S0_shipped=output/goal_causality_x0/g3_e3420_forks@$R/plans \
    route_training=$O/forks_route_training@data/scripts/plans_release_v3_route,$R/plans
PY=$PY SMOKE=1 bash data/scripts/run_student_distill_release_v3.sh
```

## 8. Files

| Path | What |
|---|---|
| `protomotions/envs/base_env/env.py` | `rebuild_observations()`, the captured contact sample and reset rows |
| `protomotions/envs/control/contact_graph_control.py` | the expert view (`expert_view_*`, `_init_expert_view`, `_populate_expert_view`); `set_manual_goal(hold_duration, deadline_floor)` |
| `protomotions/envs/context_views.py` | `EnvContext.expert_masked_mimic`, `expert_contact_goal` |
| `protomotions/agents/supervised/expert_port.py` | contract, rewiring, identity checks, the single-expert routing table |
| `protomotions/agents/evaluators/sequence_viz.py` | `SequenceVizConfig.timing`, `fill_goal_slots(timing=)`, re-issue on goal advance, the exact rebuild |
| `examples/experiments/masked_mimic/contact_graph_transformer.py` | the merge-loop fix |
| `examples/experiments/masked_mimic/contact_graph_fsq_release_v3.py` | the ported student experiment |
| `data/scripts/run_student_distill_release_v3.sh` | its launcher |
| `data/scripts/make_route_probe_plans.py`, `data/scripts/plans_release_v3_route/` | the route plans (42) |
| `data/scripts/make_single_expert_routing.py`, `data/smpl/student_release_v3/` | the routing table |
| `data/scripts/s1_port_parity.py` | both parity tests and the run comparison |
| `data/scripts/goal_causality_x0.py` | the manual-path check's fixed call, `--fork-timing`, the env's rebuild |
| `data/scripts/render_contact_goal_sequence.py`, `run_sequence_panel.py` | `--timing`; the renderer's exact rebuild |
| `data/scripts/expert_handoff_screen.py`, `expert_revist/hub_swap/{far,hub}_swap_test.py` | the exact rebuild |
| `data/scripts/policy_setup.py` | `build(..., load_checkpoint=False)` |
| `protomotions/tests/test_s1_student_port.py` | 42 CPU tests |
| `forks_compare.py`, `data/` | the battery collation; `parity_obs.json`, `parity_eval*.json`, `forks_compare.json`, `x0_g3_e3420_summary.json` |
| `output/s1_port/` (not in git) | every run's raw output and log |
