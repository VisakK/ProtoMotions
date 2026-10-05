# Card E6: AMP plumbing and the Step 3 launcher (2026-10-04)

**What it is.** The four things G3 needs from the AMP expert that G1 did not (`PLAN.MD` card E6, items 1–4): a CLI
flag for the per-lineage style weights, per-lineage style-reward diagnostics, the AMP training state on a warm start,
and G3's own launcher. Items 1–4 are built and tested on the CPU, and review round 1's findings are fixed (below).
Item 5, the GPU smoke on release v2, belongs to the integration stage. Nothing here launched IsaacLab or
`train_agent.py`.

| File | What |
|---|---|
| `protomotions/agents/amp/goal_conditioned.py` | `parse_amp_lineage_rule` / `parse_amp_lineage_weights` (PATTERN=W, strict), `lineage_match` (first rule wins, in order), `LineageRewardMeter` (per-epoch syn/human sums on the device), `GoalConditionedAMPComponent.setup_lineage` (eager; prints what each rule matched, and why a rule matched nothing), the accumulation in `record_rollout_step`, its logs in `add_epoch_logging`, `load_warm_start_state`; `GoalConditionedAMP.setup` (builds the lineage tables) and `GoalConditionedAMP._load_optimization_state` (PPO's state, then the AMP state); `warm_start_calibration` (the calibration-off rule); `[amp demos]` printed lines |
| `examples/experiments/mimic/mlp_goal_conditioned_amp.py` | `--amp-lineage-weights PATTERN=W [...]` in the AMP group (validated per item at parse time, stored as strings so `config.yaml` round-trips), parsed into `GoalConditionedAMPAgentConfig.amp_lineage_weights` |
| `data/scripts/run_expert_graph_ft.sh` | G3's launcher, copied from `run_expert_amp_ft.sh` (unchanged, still G1's) |
| `protomotions/tests/test_e6_amp_plumbing.py` | 28 tests: the parser (good, 12 malformed, duplicates, the argparse flag), matching order, the meter, a rollout step through fakes, the WARNING wording, the demonstration lines, the warm start (AMP state present / absent / partial / non-AMP model), the calibration rule |
| `protomotions/tests/test_e6_launcher.py` | 18 tests: the launcher under `DRY_RUN=1` in a throw-away mirror of the repo (release v2 real; "v3" releases are v2 plus a stub package with two SYN_ variants), every refusal and both modes' commands. Skipped when release v2 is not on disk. ~70 s |

## Acceptance (this phase: CPU only)

| Check | Result |
|---|---|
| Unit tests | `test_e6_amp_plumbing.py` **28 passed**; `test_e6_launcher.py` **18 passed** |
| Existing tests | `test_goal_conditioned_amp.py`, `test_amp_agent_helpers.py`, `test_amp_features_v1.py`, `test_example_experiment_configs.py`: **69 passed** (69 before the change). Whole `protomotions/tests`: 1,769 passed, 1 skipped, 2 failed, both failing identically at HEAD (see "Full suite") |
| G1's config unchanged with no new flags | G1's saved `config.yaml` rebuilt with the final tree through `build_standard_configs` + its `--overrides` on the CPU, compared field by field with its pickled `resolved_configs.pt`: **0 differences** in robot, simulator, terrain, scene lib, motion lib, env and agent. Positive control `--amp-lineage-weights SYN_=0.5 Crane_Crow=0.25`: exactly 2 differences (the two dict entries). The tree holds E3's in-progress edits, so this covers E3's new fields at their defaults as of the final run (2026-10-04 evening) |
| The launcher's command, built on the CPU | the default `DRY_RUN` command parsed with `train_agent`'s own parser (extracted by AST) plus the experiment's, built through `build_standard_configs`, diffed against G1's `resolved_configs.pt`: **exactly 12 differences, all intended** — experiment name; `segment_start_prob` 0.6 → 0.4, `segment_end_prob` 0 → 0.2; `training_max_steps` 786,432,000 (6,000 epochs); `motion_prior` none → package; `freeze_obs_normalizers` True; `amp_reward_w_target` 0.1 → 0.5, start 200 → 0, full 500 → 0, calibration 0.25 → 0; `demo_exclude_motions` + `SYN_`; `amp_lineage_weights {'SYN_': 0.5}`. The default and SMOKE commands are token-for-token those of the pre-review launcher |
| AMP state in G1's `epoch_5000.ckpt` | `discriminator_optimizer`, `disc_critic_optimizer`, `running_amp_reward_norm` and `amp_weight_calibration` all present. The AMP model built from G1's resolved config loads it strictly. Discriminator optimiser 6 / 6 parameters, disc critic 8 / 8, each Adam moment the shape of its parameter; step **39,920** on both, lr 1e-4 (= configured). AMP reward normaliser count 655,360,000, var 155.76. Calibration: target 0.5, 1,000 ratio-history entries (epochs 4,000–4,999) |
| The warm-start path, end to end on the CPU | real model, real Fabric (CPU) `FabricAdam` optimisers from G1's resolved config, G3's AMP settings, through `BaseAgent.load`: discriminator step 0 → **39,920**, disc critic 39,920, actor 163,680; AMP normaliser count 655,360,000, var 155.758; task normaliser 2,686,976,000; `target_w()` 0.5; ratio history 1,000 → 0. On e15500 (`last.ckpt`, a PPO checkpoint, G1's calibration settings) it prints the skip note and the AMP state stays fresh (disc step 0, normaliser count 1, `target_w` 0.1) |
| `bash -n` | passes |

**For checklist item 4: `amp_parameters.discriminator_reward_w` reads 0.0 in every resolved config of this experiment,
by construction** (`mlp_goal_conditioned_amp.py` sets it to 0.0 and the schedule sets it each epoch inside
`add_advantages`; G1's own `resolved_configs.pt` reads 0.0 too). The configured weight is `amp_reward_w_target` (0.5),
with `amp_reward_w_start_epoch` = `amp_reward_w_full_epoch` = 0 and `amp_calibrate_style_ratio` 0. The live weight is
logged as `amp/reward_w`. `PLAN.MD`'s checklist item 4 still says `discriminator_reward_w` 0.5 and needs rewording
(orchestrator-owned).

## Item 1: `--amp-lineage-weights PATTERN=W [...]`

- A motion whose stem contains PATTERN has its style reward multiplied by W before the AMP reward normaliser; the first
  rule that matches wins, in the order given; unmatched motions weigh 1.0. Per motion, not per frame (D4).
- **Loud on bad input.** One `=`, a non-empty PATTERN without surrounding whitespace, a finite W ≥ 0; anything else
  is an argparse error at launch (exit 2, before the simulator). A pattern given twice raises when the config is built.
- Default empty: G1's config is unchanged (0 differences, above).
- **Substring match.** A pattern covers every stem that extends it: on v2, `Crane_Crow` matches 12 motions (Crane Crow
  -a and Side Crane Crow -a/-b/-c, each × {x0, x3s, x7s}); on v3, `SYN_E1_press_high_s0_t6px` would also match
  `…_t6px12` (R2's names collide this way). The help text says so: list the longer pattern first. G3 passes only
  `SYN_=0.5`.

## Item 2: per-lineage diagnostics

Only when at least one rule is set (G1 logs nothing new). Per env-step, from `motion_manager.motion_ids` at reward
time. The agent resets done envs at the *next* step's start, so these are the motions the transitions tracked, the same
ids the weight is gathered with. Summed on the device in float64 over the epoch and emitted in `add_epoch_logging`:

| Key | What |
|---|---|
| `amp/syn_sample_share` | share of the epoch's agent env-steps on a matched ("syn") motion |
| `amp/reward_mean_syn` | mean **unweighted** discriminator reward −log(1−D), before the normaliser, on syn env-steps |
| `amp/reward_mean_human` | the same on every other env-step |
| `amp/reward_mean_syn_weighted` | the syn mean after the lineage weight. There is no human twin: the human weight is 1.0, so it would equal `reward_mean_human` |

**A group with no samples in an epoch is left out of that epoch's log, never NaN.** NaN would pass through
`aggregate_scalar_metrics`, and wandb shows a gap either way. `amp/syn_sample_share` is logged whenever the epoch had
samples, 0.0 included.

## Item 3: the AMP state on a warm start

`GoalConditionedAMP._load_optimization_state` runs only under `--warm-start-optimization-state` on a warm start.
It calls PPO's version first (task reward normaliser, actor/critic optimisers, advantage EMA), then
`GoalConditionedAMPComponent.load_warm_start_state`:

| Restored | From | Note |
|---|---|---|
| discriminator optimiser | `discriminator_optimizer` | Adam moments and step count; the checkpoint's lr (1e-4, = configured), as PPO's warm start keeps its own |
| disc-critic optimiser | `disc_critic_optimizer` | the same |
| AMP reward normaliser | `running_amp_reward_norm` | Welford mean/var/count, **not frozen**: G1's 655,360,000 samples (5,000 epochs × 4,096 envs × 32 steps) carry 77 % of the weight at G3's epoch 1,500 and 45 % at 6,000, so `amp_reward_norm/var` drifts towards G3's lineage-weighted rewards. Keeping it keeps the restored disc critic's value scale; a fresh one would rescale it by √155.8 ≈ 12.5×. (PPO's task normaliser behaves the same way: G1 took it from 2,031,616,000 at e15500 to 2,686,976,000, so 24 % of it is G1's data) |
| weight calibration | `amp_weight_calibration` | see the rule below |
| *already loaded with the model* | `model._discriminator.*`, `model._disc_critic.*` | the networks and their observation normalisers (free; E3 freezes only the actor's and critic's) |

**Starts fresh** (nothing in the checkpoint): the discriminator replay buffer (its first epoch uses the current
samples, as on any start), the per-env bad-transition counters (irrelevant at threshold 0), the demonstration sampler
and lineage tables (built from this run's library and config), `current_w` (recomputed every epoch), and PPO's epoch
counter, so G3 counts from 0.

**Two things the resume path keeps are wrong on a warm start, and are fixed here:**
- **The calibration's ratio history is dropped.** It is keyed by the old run's epochs (G1: 4,000–4,999). A new run
  restarts at epoch 0, so a calibration window `[start − window, start)` would read G1's ratios.
- **Calibration off: the configured target wins.** With `--amp-calibrate-style-ratio 0`, a restored calibrated target
  is discarded, so `target_w()` returns `amp_reward_w_target`. The printed note says whether the two differed. G1's is
  0.5, equal to G3's `--amp-reward-w 0.5`.
- With calibration on, a restored target is kept and frozen (no re-calibration); a checkpoint without one calibrates
  at this run's start epoch.

**A checkpoint without an AMP model or AMP state** (a PPO checkpoint, e.g. e15500, G1's own warm start) is skipped with
a printed note, so G1's path is unchanged. A checkpoint with only part of the state raises `KeyError`.

**The resume path is unchanged.** `AMPAgentMixin._load_training_state` restores everything, the calibration and its
history included, on the same clock.

## Item 4: `data/scripts/run_expert_graph_ft.sh`

```bash
RELEASE=holds_repaired_ftC_posefix.release_v3.<id> EXPERIMENT=smpl_yogi_v2_expert56_g3_<id> \
CHECKPOINT=results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/epoch_5000.ckpt \
  setsid nohup bash data/scripts/run_expert_graph_ft.sh > output/graph_growth_g3/g3.log 2>&1 &
```

| Flag | G3 value | Env var (default) |
|---|---|---|
| `--checkpoint`, `--warm-start-optimization-state` | `CHECKPOINT`; both omitted on `RESUME=1` | `CHECKPOINT` (required), `RESUME` (0) |
| `--experiment-name` | `EXPERIMENT` (+ `_smoke` / `_timing`) | `EXPERIMENT` (required) |
| `--motion-file`, `--hold-graph-file`, `--release-record`, `--contact-targets`, `--physics-tables`, `--hold-manifest`, `--robot-name` | from the record's artifacts | `RELEASE` (required, a release id) |
| `--amp-reward-w` / `--amp-w-start-epoch` / `--amp-w-full-epoch` / `--amp-calibrate-style-ratio` | 0.5 / 0 / 0 / 0 | `AMP_W`, `AMP_START`, `AMP_FULL`, `AMP_CAL_RATIO` |
| `--amp-disc-batch-size` / `--amp-grad-penalty` | 4,096 (smoke 1,024) / 10.0 | `AMP_DISC_BATCH`, `AMP_GRAD_PENALTY` |
| `--amp-lineage-weights` | `SYN_=0.5` | `AMP_LINEAGE` (unset → default, `''` → no rules) |
| `--amp-demo-exclude-motions` | `Scorpion_pose_or_vrischikasana-b SYN_` | `AMP_DEMO_EXCLUDE` |
| `--segment-start-prob` / `--segment-end-prob` / `--segment-pre-roll-s` | 0.4 / 0.2 / 0.5 | `SEGMENT_START_PROB`, `SEGMENT_END_PROB`, `SEGMENT_PRE_ROLL_S` |
| `--motion-prior` / `--freeze-obs-normalizers` | package / True (every launch, resumes included) | `MOTION_PRIOR`, `FREEZE_OBS_NORMALIZERS` |
| `--curriculum mixture --uniform-fraction` | 0.8 | `UNIFORM_FRACTION` |
| `--support-rule`; support / swing / lean weights | v2; −0.3 / −0.3 / −0.3, the rest as G1 | `SUPPORT_RULE`, `SUPPORT_WEIGHT`, `SWING_WEIGHT`, `LEAN_WEIGHT` |
| `--goal-bodies all --sense-body-pair-contacts True --critic-future-steps 1 5 10 15 --interval-schedule True` | as G1 | — |
| `--drag-report-motions` | G1's 8 human clips | `DRAG_REPORT` |
| `--viz-plan-files` / `--viz-num-sequences` | G1's 10 + `edge_*.json` + `nohijack_*.json` / count + 2 (20 / 22 with five edges) | — |
| `--viz-sequences-every` / `--viz-max-seconds 24.0 --viz-log-scalars False` | 500 | `VIZ_EVERY` |
| `--eval-every` / `--eval-max-steps` / `--save-every` | 500 / max(2,250, record) / 500 | `EVAL_EVERY`, `EVAL_MAX_STEPS`, `SAVE_EVERY` |
| `--num-envs` / `--batch-size` / `--training-max-steps` | 4,096 / 16,384 / `MAX_EPOCHS` × envs × 32 | `NUM_ENVS`, `BATCH_SIZE`, `MAX_EPOCHS` (6,000) |
| `--use-wandb`; `--overrides env.ref_respawn_offset=0.005` | on; as G1 | `WANDB` (1), `EXTRA` (appended after the override) |

**Modes.**
- `SMOKE=1`: 256 envs, batch 1,024, `SMOKE_EPOCHS` (4) epochs, evaluation and save every 2 over 150 steps, no panel,
  no wandb, discriminator batch 1,024. The AMP schedule is not shortened, since AMP is on from epoch 0 anyway.
- `TIMING=1`: 4,096 envs × `TIMING_EPOCHS` (50), periodic evaluations, saves and the panel off, no wandb. **The
  evaluation length stays real** (2,250 steps, or the record's value): the loaded checkpoint's evaluation after
  epoch 1 still runs, and it allocates one `[motions × eval_max_steps × features]` buffer per robot-state field and
  metric (`base_evaluator.py:416-455`, `metrics.py:45`). That buffer is the memory question for checklist item 6, and it
  grows 1.52× on v3 (255 motions against 168). At 150 steps, as G1's `TIMING`, it is 15× smaller and the reading
  misses it. The panel's capture is copied to the CPU per frame, so leaving the panel off costs that reading little.
- `CONFIG_ONLY=1`: `--create-config-only`.
- `RESUME=1`: see below.
- `DRY_RUN=1`: every check, the echo and the exact command (`printf %q`), then exit 0.
- `PY` (default `python`) is the interpreter. The release reader uses it too (it imports torch and `protomotions`),
  so `PY=echo` is not a dry run.

**Refusals before anything is read,** on the name actually used (`_smoke` / `_timing` included):
- a missing `RELEASE`, `EXPERIMENT` or `CHECKPOINT`; an `EXPERIMENT` or `RELEASE` that is not a plain name (leading
  letter or digit, then `[A-Za-z0-9_.-]`, so `.`, `..`, `-x` and paths are refused);
- `SMOKE`, `TIMING`, `CONFIG_ONLY`, `RESUME`, `DRY_RUN` or `WANDB` other than 0 or 1 (`SMOKE=true` used to launch
  the full run); `NUM_ENVS`, `BATCH_SIZE`, `EVAL_EVERY`, `EVAL_MAX_STEPS`, `SAVE_EVERY` or `MAX_EPOCHS` not a positive
  integer;
- `results/${EXPERIMENT}/last.ckpt` exists (a dangling link counts) without `RESUME=1`;
- `RESUME=1` without `last.ckpt` (it passes no `--checkpoint`, so `train_agent` would start a fresh run from scratch);
- `RESUME=1` with `CONFIG_ONLY=1`, which would overwrite the run's frozen configs;
- a live `train_agent.py` with the same `--experiment-name NAME` or `--experiment-name=NAME`. This covers a run's
  first 500 epochs, before its first `last.ckpt`.

**Then, read-only, from the release** (one Python block, ~3 s):
- **The stems are the package's own `motion_files`** (`motions.pt` memory-mapped), the list the AMP component and the
  panel use at runtime. It refuses a package with none, and a record whose `motions` key lists a different set.
  Nothing depends on a record key R3 has not fixed: `variants` / `synthetic.variants` are used for the echo only when
  present and of a usable shape (a string `synthetic` id is fine), else the SYN_ x0 stems.
- an `AMP_DEMO_EXCLUDE` that lets a SYN_ x0 clip, or G1's excluded Scorpion -b, into the demonstrations (the flag
  replaces its default list);
- `AMP_LINEAGE` through the agent's own parser and first-match matcher: a malformed rule is refused (`train_agent`'s
  argparse would refuse it later anyway). **On a release with SYN_ motions** it also refuses a rule that matches
  nothing (a typo, or a rule shadowed by an earlier one) and any SYN_ motion no rule weights (pass `SYN_=1.0` for an
  unweighted arm). On a release without SYN_ motions (v2) a rule that matches nothing is only flagged in the echo
  (`(WARNING: none)`);
- `RESUME=1` on a run this launcher did not start: its `config.yaml` must name this experiment file, carry
  `amp_lineage_weights` (only this experiment's parser writes it; G1 and e15500 both train on v2's library and would
  otherwise pass), and name the release's library;
- an explicit `EVAL_MAX_STEPS` below the record's `training.eval_max_steps` (rounded up to an integer, so a float in the
  record is honoured). The default is raised to it instead. SMOKE alone skips this;
- **a panel plan that does not resolve.** Each of the panel's plans (G1's 10 by name, plus the globbed `edge_*.json`
  and `nohijack_*.json`; `fork_edge_*` stays out) goes through `SequenceVizRunner._load_plan`, the panel's own loader,
  against the package's stems and `ContactGraph.from_file` of the release's graph, at `--viz-max-seconds 24.0`. A plan
  whose clip or configuration does not resolve, or that loses a goal to the 24 s cap, fails the launch with the
  loader's warning. This runs in every mode, SMOKE and TIMING included, although neither runs the panel. At runtime the
  panel would swap such a plan for a graph-derived sequence with only a log warning. On v2: 10 plans, all resolve,
  none capped (longest 22.0 s).

**Resume.** `train_agent.py` resumes whenever `results/<name>/last.ckpt` exists (`detect_checkpoint_mode`). It then
overwrites `args` from that run's `config.yaml`, rebuilds nothing (`resolved_configs.pt`) and loads `last.ckpt` with
the full training state (`load_training_state=True`): PPO, AMP, calibration and evaluator. `--warm-start-optimization-state`
only acts in `warm_start` mode, so on a resume it would be inert. `RESUME=1` drops it and `--checkpoint` anyway, so the
command says what happens. The freeze is re-applied on a resume from the run's own frozen agent config
(`freeze_obs_normalizers` is a `PPOAgentConfig` field, E3). The launcher `exec`s `train_agent.py`, so no wrapper bash
process stays alive. Record and stop the run by `train_agent.py`'s own PID (`pgrep -af "[t]rain_agent.py"`); `$!` of an
interactive `setsid … &` can be a short-lived `setsid` parent.

**Echo.** The release's `purpose` and its variant list; the motion count and its SYN_ count; the lineage rules with
their first-match-wins counts; the expected demonstration count; the anchoring, prior and freeze settings; the panel
composition. Release v2 prints `purpose n/a; 0 synthetic variants: n/a`.

## Lines to grep (the smoke and G3's log)

`log.info` from the package does not reach a run's log (G1's log has `[amp parity]` but no "AMP demonstrations"), so
every line below is printed:

```
Warm start: restored AMP training state from checkpoint: discriminator optimizer (step 39920, lr 0.0001), disc-critic optimizer (step 39920, lr 0.0001), AMP reward normaliser (count 655360000, var 155.8)
Warm start: AMP weight target 0.5 (configured; calibration off; equals the checkpoint's calibrated target); 1000 calibration ratio-history entries from the old run's epoch clock dropped
Warm start: no AMP training state loaded -- ...                       (a PPO checkpoint instead)
[amp lineage] rule '<PATTERN>' (w <W>) matched <n> of <N> motions: <stems>
[amp lineage] WARNING: rule '<PATTERN>' (w <W>) matched no motion (of <N>; no stem contains it)
[amp lineage] WARNING: rule '<PATTERN>' (w <W>) matched no motion (of <N>; <k> stem(s) contain it, all taken by an earlier rule)
[amp lineage] <n> of <N> motions are 'syn' (matched); <N-n> are 'human' (w 1.0)
[amp demos] <k> of <N> motions serve as demonstrations, <s> s usable (t >= 0.667 s); excluded <a> by regex '_x\\d+s$', <b> by name [...], <c> too short
[amp demos] excluded by name: <stems>
[amp demos] demonstrations: <stems>                                   (no SYN_ stem may appear)
[amp demos] lineage-matched motions among the demonstrations: <m>
```

The first two come from G1's checkpoint as measured above, on the CPU. On v2 with `AMP_LINEAGE=Crane_Crow=0.5`, expect:
- `matched 12 of 168`;
- `12 of 168 motions are 'syn'`;
- about 55 demonstrations, of which 4 lineage-matched (the four Crane Crow x0 clips; they are human, so this is right
  on v2);
- `excluded … 1 by name` (Scorpion -b).

The demonstration lines appear at the first optimisation step (epoch 0); the lineage lines at agent setup.
E3's own lines (`freeze_obs_normalizers: froze …`, `… verified …`) come from `ppo/agent.py`.

## Deviations from the card

- **Extra log key** `amp/reward_mean_syn_weighted`; no weighted human key (identical by definition). Groups without
  samples are omitted (the card left NaN-or-absent open). Nothing is logged without rules.
- **`[amp demos]` lines are printed in every AMP run, G1's code path included.** This is output only, not behaviour.
  Without it the pre-launch checklist's "log lines for the demonstration set" would not exist: the old line was
  `log.info`.
- **Warm start drops the calibration's ratio history** and, with calibration on, keeps a restored target frozen. The
  card only says "restores … the weight calibration"; the history's epochs belong to the old run's clock.
- **`TIMING=1` keeps the real evaluation length** (G1's `TIMING` used 150 steps). The card says "as in G1's launcher";
  checklist item 6 reads peak memory from this mode, and at 150 steps it could not see the evaluation buffers.
- **Launcher extras beyond the card:**
  - `DRY_RUN=1`, `SMOKE_EPOCHS`, `TIMING_EPOCHS`;
  - `RESUME=1` drops `--checkpoint` and `--warm-start-optimization-state`, and accepts only this launcher's own run;
  - the input checks and refusals listed under item 4, including the offline panel-plan resolution, which is what
    makes `PLAN.MD`'s "a plan that does not resolve in v3 fails the launch" true;
  - an `EVAL_MAX_STEPS` default raised to the record's value instead of refused (an explicit lower value is still
    refused);
  - E3's values overridable by env vars; `exec`.
- **`CHECKPOINT` is required on `RESUME=1` too,** literally as the card says, although a resume ignores it.
- **The launcher's plan check calls `SequenceVizRunner._load_plan`, a private method, on a runner built without an
  env.** If `sequence_viz.py` changes that method's needs, the check fails closed (the reader exits 1), never open.

## Review round 1

Three lenses (spec, correctness, traps), no blocking or major findings. Each was reproduced before it was fixed; every
fix has a test in `test_e6_launcher.py` or `test_e6_amp_plumbing.py`.

| Finding | Disposition |
|---|---|
| RESUME=1 accepted e15500's or G1's run directory on v2 (only the library was checked) | **Fixed**: experiment file and `amp_lineage_weights` in `config.yaml` required |
| Panel plans only existence-checked; a non-resolving plan silently replaced; SMOKE/TIMING never build the panel | **Fixed**: offline resolution by the panel's own loader, every mode |
| Record reader crashed on a string `synthetic`; the SYN_ guard passed on a record without `motions` | **Fixed**: stems from the package; shape-tolerant variant read; refusals for an empty package or a mismatched `motions` |
| TIMING could not see the evaluation's memory | **Fixed**: TIMING keeps the real `EVAL_MAX_STEPS`. The size in GB is not measured; TIMING now measures it |
| Checklist item 4's `discriminator_reward_w 0.5` cannot hold | **Not a code defect** (0.0 by construction, as in G1). Rewording `PLAN.MD` is the orchestrator's |
| No guard for a dropped Scorpion -b exclusion or a lineage rule matching nothing | **Fixed**: both refused (lineage only on a release with SYN_ motions) |
| Wrong reason in the RESUME-without-`last.ckpt` refusal | **Fixed** (fresh start, not warm start), here and in the script |
| README called the AMP normaliser frozen | **Fixed**: Welford; G1's share 77 % at epoch 1,500, 45 % at 6,000, measured counts above |
| WARNING blamed an earlier rule that may not exist | **Fixed**: says which of the two it is |
| Echo counted rules independently, not first-match-wins | **Fixed**: the agent's `lineage_match` |
| Substring rules cover extended stems (`t6px` / `t6px12`) | **Documented** in the flag's help text and above; G3 uses only `SYN_` |
| Float `eval_max_steps` silently skipped the floor | **Fixed**: rounded up to an integer in the reader |
| DRY_RUN did not validate lineage syntax | **Fixed**: the agent's parser |
| `EXPERIMENT=..`, `SMOKE=true`, `--experiment-name=NAME`, unvalidated `RELEASE` | **Fixed**: all four |
| SMOKE_EPOCHS=10 writes ~5 GB of checkpoints | **Documented** in the smoke section below |
| Item 5 / GPU acceptance not verified | **Deferred** to the integration stage, by design |

## Full suite

Whole `protomotions/tests` on the final tree (E3's and R2's in-progress edits included): **1,769 passed, 1 skipped,
2 failed** in 87 s. The two failures are `test_release_docs_and_legal.py::test_quickstart_pretrained_table…` (asserts
`'SOMA GPC prior'` in the quickstart docs) and `test_scene_lib_loading_and_assets.py::test_scene_lib_main_example…`
(`ValueError: zero-size array to reduction operation minimum`). **Both fail identically at HEAD**: a `git archive HEAD`
export in the scratchpad gives the same two assertion messages. So they predate E6, and neither touches a file E6
changed.

## Smoke (integration stage)

Run 2026-10-04 on the A5000, one IsaacLab process at a time, under scratch names `zz_step3_*`; every scratch
`results/` directory was deleted afterwards (5.2 GB). Logs in this folder:
- `smoke_config_only.log`: the three `CONFIG_ONLY` runs, their recursive resolved-config diffs, and the refusals;
- `smoke_g3_v2.log`: the smoke's run log, trimmed;
- `smoke_g3_v2_tb.log`: its TensorBoard scalars;
- `smoke_g3_v2_ckpt.log`: optimiser step counts and normaliser state, G1's `epoch_5000.ckpt` against the smoke's
  checkpoints.

The diffs load both `resolved_configs.pt` on the CPU and walk dataclasses, dicts and lists. A field missing from the
old pickle's `__dict__` is listed as a **new field**, with its value and the class default. `getattr` would return the
default for it and hide it.

| Check | Result |
|---|---|
| **G1 unchanged.** `CONFIG_ONLY=1 EXPERIMENT=zz_step3_cfg_g1 bash data/scripts/run_expert_amp_ft.sh` (G1's launcher, untouched), against G1's `resolved_configs.pt` | **1 value difference**, `simulator.experiment_name`. **3 new fields, all at their defaults**: `env.motion_manager.segment_end_prob` 0.0, `agent.evaluator.curriculum.motion_prior` `'none'` and `agent.freeze_obs_normalizers` False. `amp_lineage_weights` `{}` is already in G1's pickle (E2's field; E6 adds only the flag). |
| **e15500 unchanged.** `EXPERIMENT=zz_step3_cfg_e15500 EXTRA=--create-config-only bash data/scripts/run_expert_release_v2.sh` (the launcher has no `CONFIG_ONLY`; argparse ends the `--overrides` list at the next option), against e15500's `resolved_configs.pt` | **1 value difference**, the name. **9 new fields, all at their defaults**: E3's three, plus E1's six (`support_rule` `'v1'`, `support_v2_*`). |
| **G3's config (checklist item 4).** `CONFIG_ONLY=1` on v2 with `EXPERIMENT=zz_step3_cfg_g3` and G1's `epoch_5000.ckpt`, against G1's `resolved_configs.pt` | **Exactly the 12 intended differences**, the same set as the CPU build above: the name; `segment_start_prob` 0.6 → 0.4; `segment_end_prob` 0.2; `training_max_steps` 786,432,000; `motion_prior` `'package'`; `freeze_obs_normalizers` True; AMP target 0.1 → 0.5, start 200 → 0, full 500 → 0, calibration 0.25 → 0.0; `demo_exclude_motions` `['Scorpion_pose_or_vrischikasana-b', 'SYN_']`; `amp_lineage_weights` `{'SYN_': 0.5}`. Values read for item 4: `pre_roll_s` 0.5; GP 10.0; discriminator batch 4,096; `support_rule` `'v2'`; `unwanted_support_rew`, `swing_penalty_rew` and `lean_penalty_rew` −0.3; `ref_respawn_offset` 0.005; 4,096 envs; batch 16,384; eval, save and panel every 500; `max_eval_steps` 2,250 (record 1,890); `num_sequences` 12 (10 plans + 2). `amp_parameters.discriminator_reward_w` reads **0.0**, as in G1, by construction (above). |
| **Refusals** (`PY=false`, so a refusal that failed to fire could not launch anything) | rc 1 with the launcher's own message for each of: `EXPERIMENT` unset; `RELEASE` unset; `CHECKPOINT` unset; an empty dummy `results/zz_step3_refuse/last.ckpt` without `RESUME=1` (in both real and `DRY_RUN` modes). With `RESUME=1`, the dummy directory is refused for its missing `config.yaml`. The dummy directory was deleted. |

**The smoke.** `SMOKE=1 SMOKE_EPOCHS=10 RELEASE=holds_repaired_ftC_posefix.release_v2.a2dda5d2ac
EXPERIMENT=zz_step3_smoke CHECKPOINT=results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/epoch_5000.ckpt
AMP_LINEAGE=Crane_Crow=0.5 bash data/scripts/run_expert_graph_ft.sh`. The launcher appends `_smoke` to the name. It
completed 10 epochs in about 200 s (256 envs, batch 1,024), with no Traceback, and wrote 7 checkpoints of 0.77 GB.

| Check (card E6 item 5 and acceptance) | Result |
|---|---|
| AMP state loaded from G1's checkpoint | `Warm start: restored AMP training state from checkpoint: discriminator optimizer (step 39920, lr 0.0001), disc-critic optimizer (step 39920, lr 0.0001), AMP reward normaliser (count 655360000, var 155.8)`, then `Warm start: AMP weight target 0.5 (configured; calibration off; equals the checkpoint's calibrated target); 1000 calibration ratio-history entries from the old run's epoch clock dropped`. The saved checkpoints continue G1's counters: discriminator and disc-critic step **39,922** at `epoch_2.ckpt` and **39,927** at `epoch_10.ckpt`, where a fresh optimiser would read 2 and 7. The AMP reward normaliser count goes 655,360,000 → 655,441,920, exactly +10 × 8,192 env-steps. |
| Lineage lines | `[amp lineage] rule 'Crane_Crow' (w 0.5) matched 12 of 168 motions: …`, listing Crane Crow -a and Side Crane Crow -a/-b/-c, each × {x0, x3s, x7s}; then `12 of 168 motions are 'syn' (matched); 156 are 'human' (w 1.0)` |
| Demonstration lines | `55 of 168 motions serve as demonstrations, 1290.8 s usable (t >= 0.667 s); excluded 112 by regex '_x\\d+s$', 1 by name […], 0 too short`; excluded by name: `220923_Scorpion_pose_or_vrischikasana-b`. Of the 55 listed stems, 0 match `SYN_`, Scorpion -b or `_x\d+s$`. 4 lineage-matched (the four Crane Crow x0 clips) |
| Finite losses | 0 non-finite values in 302 scalar tags. Over the 5 epochs that update the policy: actor PPO loss 0.055–0.44, critic 0.0075–0.10, discriminator 0.58–0.63, disc critic 0.012–0.085 |
| Style weight | `amp/reward_w` and `amp/reward_w_target` read **0.5 at every epoch from epoch 0** |
| Per-lineage means | All three keys are logged at every epoch: `amp/syn_sample_share` 0.066–0.085 (12/168 = 0.071); `amp/reward_mean_syn` 0.51–0.62 against `amp/reward_mean_human` 0.48–0.60. The identity `share × reward_mean_syn_weighted + (1 − share) × reward_mean_human = rewards/unnormalized_amp_rewards` holds at every epoch to a relative 5.6e-8 |
| Epoch-0 evaluation | The loaded checkpoint evaluates after epoch 0 (TB step 1). The evaluator builds its tables lazily at that first evaluation, checking the hold manifest against the record (`require_artifact`), and `eval/perf_group/*` is logged from it: arm balance 1.000, connective 0.9994, inversion 1.000, single-leg 1.000; at epoch 10, 0.9975–1.000. `ContactGraphControl: release … every loaded artifact matches its record` |
| E3 in the same run | Read in `../e3_sampling/README.MD`, "GPU freeze check": the actor/critic normalisers are bit-identical after 10 epochs, and the anchoring shares and the prior's shares are logged |

Two things in the smoke look odd and are not E6's or E3's:
- **Every update epoch prints `Skipping actor updates for remaining batches (clip_frac 0.61–0.71 > 0.6)`.** G1's
  own smokes print the same (`output/graph_growth_g1/smoke.log` 0.643 / 0.706, `smoke2.log` 0.643 / 0.751, and the
  no-AMP `smoke_control.log` 0.665 / 0.768). G1's full-size `g1.log` prints none. It is a 256-env smoke artefact.
- **`discriminator/agent_acc` goes from 0.78 to 0.96, and `pos_acc` from 0.95 to 0.86.** G1's own last training bin
  (epoch 5,000) read 0.943 / 0.958, so the restored discriminator returns to where G1 left it.

Not run here: **`TIMING=1`**, which belongs to G3's pre-launch on release v3 (checklist item 6). The panel was also
not exercised (`VIZ_EVERY=0` in `SMOKE`); its plans are resolved offline at every launch, and all 10 resolved here.
