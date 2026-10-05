# Card E3: sampling and fine-tune plumbing (2026-10-04)

**What it is.** Three fine-tune switches, all off by default.
- **Departure anchoring.** A reset can start just before a hold *ends*: `t_end − U(0, pre_roll)` of a uniformly
  chosen segment. Synthetic edges and get-ups begin there; arrival anchoring never starts there.
- **A package prior in the mixture curriculum.** The uniform share follows the package's yaml weights instead of
  `u/N`.
- **An actor/critic normaliser freeze.**

With every flag at its default, sampling is the same as before E3, random numbers included. So are the curriculum's
probabilities. Both are checked bit for bit against the pre-E3 commit `5fdbd50`.

`start_windows` was **not built**. The card builds it only if G0c runs, and G0c is not running.

| Flag (`mlp_goal_conditioned.py`, so also the AMP experiment) | Config field (dataclass home) | Default |
|---|---|---|
| `--segment-end-prob P` | `ContactGraphMotionManagerConfig.segment_end_prob` (`envs/motion_manager/config.py`) | 0.0 |
| `--motion-prior none\|package` | `HoldCurriculumConfig.motion_prior` (`agents/evaluators/config.py`) | `none` |
| `--freeze-obs-normalizers True\|False` | `PPOAgentConfig.freeze_obs_normalizers` (`agents/ppo/config.py`) | False |

All three are plain-default fields, so old pickled configs read the defaults. G1's frozen configs show this: the
fields are absent from their `__dict__` and read 0.0, `none` and False. The AMP experiment carries every
`PPOAgentConfig` field and builds its env and evaluator through the base file, so E6's launcher flags reach
`GoalConditionedAMPAgentConfig` (tested through `train_agent`'s own parser extended by each experiment file).
Inference zeroes `segment_end_prob` together with `segment_start_prob`. `--motion-prior package` without
`--curriculum mixture` is refused (`ValueError: --motion-prior package needs --curriculum mixture`): only the mixture
has a uniform share for the prior to shape.

## Departure anchoring (`contact_graph_motion_manager.py`)
- **One draw splits three ways.** The uniform that drew `fire` still draws it: arrival if `u < p_start`, departure if
  `p_start ≤ u < p_start + p_end`, otherwise the time `MimicMotionManager` drew. The segment draw and the pre-roll
  are shared, and a departure starts at `(seg_end − pre_roll)` clamped to `[0, length − env_dt]`.
- **Unchanged when off.** The same three `torch.rand` calls run in the same order, and the departure branch runs only
  when `p_end > 0`. `sample_motions` returns early only when both probabilities are 0. The config refuses
  `p_start + p_end > 1`. Both probabilities are read live from the config, so editing it zeroes both together.
- **Eligibility.** A segment is a departure candidate if it ends more than `env_dt` before its clip does. The departure
  CDF has the arrival CDF's weights, zeroed on ineligible slots. So `rare_node` still counts every live segment.
- **On release v2,** 642 of 810 segments are eligible. The 168 excluded are exactly each clip's last segment, which
  ends at the clip's end (±3e-5 s); every other segment ends at least 2.82 s before. All 168 clips keep at least one
  departure. A clip with none would keep its drawn time.
- **New attributes:** `seg_end`, `start_kind` (the kind of each env's last reset) and `pop_step_logs()`.

**Logged every epoch.** A 3-line hook in `BaseEnv.post_physics_step` puts `pop_step_logs()` into the step's extras.
The agent's existing extras path logs each key's mean over the epoch's steps:
- `env/anchor/arrival_resets`, `env/anchor/departure_resets`, `env/anchor/t0_resets`, `env/anchor/uniform_resets`:
  resets per step that started that way;
- `env/anchor/resets`: resets per step.

**The drawn share of a kind is `env/anchor/<kind>_resets ÷ env/anchor/resets`.** Both are means over the same steps,
so the ratio is the epoch's pooled count ratio exactly (replay below: 1.6e-8 from the pooled share, float16 meter
included). The meter stores float16, which holds every count up to 2,048 exactly; only a step that resets more
than 2,048 envs of one kind (a launch's first, full reset) rounds that count, by at most 1 in 2,048. All five keys
are emitted on every step, zeros included. Round 1 logged per-step shares only on steps
with a reset; that gave two ranks different key orders, and `aggregate_scalar_metrics` all-gathers key by key (see
round 2 below).

A G1-style run, with only arrivals on, logs these keys too. That run's random numbers do not change.

## Package prior (`hold_curriculum.py`, `hold_curriculum_evaluator.py`)
- **The probabilities.** `mixture_sampling_probs(..., prior=w)` gives the uniform share `u·w_m/Σw`. It is computed as
  `(u/N)·(w_m·N/Σw)`, so equal weights give the factor 1.0 exactly and a bit-identical result. The naive order
  `u·w/Σw` is not bit-identical; the test shows it differing.
- **Where the prior comes from.** `motion_prior="package"` captures `motion_lib.motion_weights` (the yaml `weight` as
  packed) once, in the evaluator's `__init__`, before any evaluation rewrites the manager's weights. It is never
  saved, so every launch re-reads it from the package.
- **The .pt is checked against its yaml.** When the library is a packed `X.pt` with `X.yaml` beside it, the two must
  agree motion for motion in float32, or the evaluator refuses to start. D5's 0.6 is not a float32 and still passes.
- **Logs.** Two keys per hold-manifest group, prior on only:
  - `eval/curriculum/prior_share/<group>`, the group's share of the uniform mass;
  - `eval/curriculum/group_prob/<group>`, the group's whole sampling probability.

  `eval/curriculum/ess` is logged as before. `prioritized_top10_share` now subtracts the prior's uniform share.
- **What G3 should read.** On R3's v3 package (168 human clips at 1.0, 87 synthetic at 3/n), the edges' prior share
  is 45/213 = 0.211, provided the manifest groups them as `edge`.
- **Printed at start-up:** `HoldCurriculumEvaluator: package motion prior over N motions (weight: motions) {...}`.

## Normaliser freeze (`common.py`, `ppo/agent.py`, `base_agent/agent.py`)
- **The hook.** A new no-op hook, `BaseAgent._before_first_rollout()`, runs in `fit()` after any checkpoint load and
  before the first rollout, on every launch. PPO's version calls `freeze_obs_normalizers` on `model._actor` and
  `model._critic`, which sets `_freeze_running` on every `NormObsBase` under them.
- **Paths, confirmed from the checkpoints on the CPU.** G1's `epoch_5000.ckpt` has exactly four normalisers:
  - `_actor.mu.norm` (3,572 statistics) and `_critic.norm` (5,876), which are frozen;
  - `_discriminator.models.0.norm` (1,304) and `_disc_critic.models.0.norm` (7,180), which stay free.

  e15500's `epoch_15500.ckpt` has only the first two.
- **The check.** `PPO.post_epoch_logging` compares the frozen **mean, var and count** buffers bit for bit after every
  epoch up to and including the launch's first epoch that updates the policy, and raises if any moved or lost the
  flag. Normalisers record only in training-mode forwards, which only a policy update runs. A resume skips its first
  epoch's update and the post-evaluation epoch's, so its first update is in launch epoch 3. A warm start (G3)
  updates in epoch 1, so it is verified after epoch 1. Skipped epochs are read from fit()'s own
  `skipped_policy_update` marker.
- **Messages to grep:**
  - `freeze_obs_normalizers: froze ['_actor.mu.norm', '_critic.norm'] (9448 statistics); they are checked bit-for-bit after the first epoch that updates the policy`
  - `freeze_obs_normalizers: verified -- 6 statistics buffers of ['_actor.mu.norm', '_critic.norm'] bit-identical through this launch's first policy update (K epoch(s); epoch counter now N)`
  - `freeze_obs_normalizers: epoch N skipped its policy update (resume, or the epoch after an evaluation); statistics unchanged, the check continues until an epoch updates the policy`
  - `RuntimeError: freeze_obs_normalizers: FAILED over this launch's first K epoch(s) (epoch counter now N) -- statistics changed: [...]; freeze flag cleared on: [...]`
  - `freeze_obs_normalizers: WARNING no checkpoint was loaded ...` (a fresh launch)

## Acceptance (CPU)
| Check | Result |
|---|---|
| Seeded replay of the real manager on release v2 (MotionLib CPU, 4,096 envs × 1,000 resets = 4.10 M) at 0.4 / 0.2, `init_start_prob` 0.2 | arrival 0.3997 / departure 0.2001 / t0 0.0801 / uniform 0.3201 against 0.40 / 0.20 / 0.08 / 0.32; worst relative deviation **0.16 %** (t0); limit 1 % |
| Departure landings (819,601) | all in `[t_end − 0.5, t_end]` of an eligible segment (lead 0.000–0.500 s, median 0.250); **0** in a clip's final-segment window; **0** at or after `length − env_dt`; closest to a clip end 2.82 s; all 642 eligible segments drawn |
| Logged path: 20,000 steps through `pop_step_logs` into the agent's `TensorAverageMeterDict` (float16), at 14 and 0.5 resets per step | one key order on every step (12,141 of the 0.5-per-step run's steps had no reset); `<kind>_resets ÷ resets` vs pooled share, max abs diff **1.6e-8** and **1.1e-8** |
| Flags off = pre-E3, G1's frozen manager config | pre-E3 manager (from git) vs new on the same seeds: **0 of 200** calls differ (12,378 resets): ids, times and RNG state. Unit test: 60 calls on 4,096 envs, 0 differ |
| Curriculum, flags off and prior on with v2's 168 weights of 1.0 | G1's pickled curriculum config, three evaluations seeded from G1's epoch-5000 `score_ema`: pre-E3 vs new probabilities `torch.equal` in both modes; every pre-E3 log key equal; prior keys only with the prior on. 200 random (N, u, power) cases: bit-identical |
| Freeze, CPU | frozen `NormObsBase` in train mode: mean/var/count bit-identical over 200 forwards, EMA and Welford; unfrozen: mean moves. G1's real AMP model, built from its resolved config with e5000's weights: the hook freezes exactly `_actor.mu.norm` and `_critic.norm`; after 5 train-mode forwards of the whole model the discriminator's and disc critic's statistics moved and the check printed `verified -- 6 statistics buffers` |
| CUDA (a tensor check, no IsaacLab) | `_anchor_to_segments` + `pop_step_logs` make no host syncs (`torch.cuda.set_sync_debug_mode("error")`) at p_end 0 and 0.2; arrival 0.3998, departure 0.2000 over 200 rounds |
| Unit tests | 50 new: `test_e3_departure_anchoring.py` 17, `test_e3_motion_prior.py` 16, `test_e3_freeze_obs_normalizers.py` 11, `test_e3_experiment_flags.py` 6. 350 in the related suites pass (E6's included). Full `protomotions/tests`: 1,774 pass, 1 skipped, 2 fail. The 2 are not in E3's code paths (a docs-table string, `SOMA GPC prior`; the scene-lib example's assets, a zero-size `min`), and failed identically before round 2 |

**Why the check compares mean and var, not only counts.** The card asks for unchanged counts. But every normaliser
here is an EMA (decay 0.999), and in EMA mode `count` never moves. G1 itself, unfrozen, shows this:
- from `epoch_4500.ckpt` to `epoch_5000.ckpt`, the actor's statistics moved (mean max |Δ| 0.092, var 1.27);
- its count stayed identical.

A counts-only check would pass an unfrozen run.

## For the GPU stage
- **The 10-epoch freeze check.**
  - Read the `froze` and `verified` lines in the log.
  - Then run `compare_normalizer_stats.py <source ckpt> <run>/last.ckpt`. It exits 1 unless `_actor.mu.norm` and
    `_critic.norm` are bit-identical, and it prints the free normalisers' drift.
- **Anchoring shares.**
  - With E6's flags, `env/anchor/<kind>_resets ÷ env/anchor/resets` should read about 0.40 / 0.20 / 0.08 / 0.32
    (arrival / departure / t0 / uniform). On v3 the departure share is lower for any synthetic clip with no eligible
    departure segment; `expected_shares` in the replay script computes it.
  - The manager prints `ContactGraphMotionManager: departure-anchoring 20% of resets to N segments over M/K motions`.
- **Files:**
  - `replay_departure_anchoring.py` → `replay_departure_anchoring.json`, the replay (9 s);
  - `compare_normalizer_stats.py`.

## Deviations from the card
- **Three files outside the listed ownership,** each because the field or hook has nowhere else to live:
  - `agents/ppo/config.py`, for the freeze field (the AMP experiment carries `PPOAgentConfig`'s fields);
  - `agents/evaluators/config.py`, for `HoldCurriculumConfig.motion_prior`, which the card names;
  - `envs/base_env/env.py`, the 3-line extras hook for the anchoring shares.

  `base_agent/agent.py` gets only the hook and its call.
- **The freeze check compares mean, var and count** (above).
- **The shares are logged as counts, and each share is a ratio of two logged keys.** The env-side extras path can
  only average per-step scalars. A share key would either be a per-step average emitted only on steps with a reset
  (round 1: not pooled, and a variable key set across ranks), or need an agent-side change outside this card's
  files.
- **Additions beyond the card:**
  - the yaml-vs-packed-weights check;
  - `eval/curriculum/group_prob/<group>`;
  - the `start_kind` buffer.
- **Re-resolving G1's config now shows three new keys,** all at their defaults: `segment_end_prob: 0.0`,
  `motion_prior: none`, `freeze_obs_normalizers: False`. E6's "diff the resolved configs" will see them as additions.
  Nothing existing changes. The E3 test rebuilds G1's manager, curriculum and agent flag from G1's own CLI arguments
  and finds them equal to the frozen ones.

## Round 2: review fixes (2026-10-04)
Three reviews found nothing blocking or major. Every finding was reproduced or refuted before it was changed.
- **Multi-GPU log keys (minor, fixed).** Round 1 emitted the four share keys only on steps with a reset, and
  `anchor/resets` on every step. The agent's meter keeps keys in first-seen order, and `aggregate_scalar_metrics`
  all-gathers one key at a time. Reproduced with the real `TensorAverageMeterDict`: an epoch whose first step reset
  something and one whose first step did not give different key orders. On two ranks that pairs the wrong metrics;
  a rank with no reset all epoch changes the number of collectives. Now all five keys go out every step, as counts
  (above). This also makes the logged share pooled rather than an average of per-step shares.
- **Resume made the freeze check vacuous (minor, fixed).** It ran once, after launch epoch 1. On a resume that epoch
  and the next skip the policy update, so nothing could have moved and "verified" meant nothing. It now compares
  after every epoch until the first one that updated the policy, and raises on a change in any of them (tests:
  `test_a_resume_defers_the_check_to_the_first_policy_update`, `test_a_change_during_a_skipped_epoch_still_raises`).
  G3's warm start still verifies after epoch 1.
- **`segment_end_prob` was cached while `segment_start_prob` was read live (nit, fixed).** Both are now read live
  (a property over the config), so editing the config zeroes both. No current code edits either at runtime.
- **`--motion-prior` was silently ignored without `--curriculum mixture` (nit, fixed).** It is now refused.
- **The yaml side of the prior's name match cut 7 characters from every entry (nit, fixed).** It agreed with the
  library side only for `.motion` files. Reproduced: `a.v2.npz` → library `a.v2`, yaml `a`. Both sides now use one
  rule, and the yaml file handle is closed. Every v2/v3 package lists `.motion` files, so no package changed.
- **The E6-flags test parsed with the base experiment's parser only (nit, fixed).** It now parses through
  `train_agent.create_parser()` (taken from the file by `ast`: importing it parses argv) extended by each experiment
  file. Both files give 0.4 / 0.2 / 0.5 / `package` / True, and the defaults when the flags are absent.
- **Edits outside the listed files (nit, no change).** Already recorded under deviations.

## GPU freeze check (integration stage, 2026-10-04)
Read from card E6's 10-epoch smoke on release v2: 256 envs, a warm start from G1's `epoch_5000.ckpt`, and E6's flags
(`--segment-start-prob 0.4 --segment-end-prob 0.2 --segment-pre-roll-s 0.5 --motion-prior package
--freeze-obs-normalizers True`). The run itself is recorded in `../e6_amp_plumbing/README.MD`, "Smoke (integration
stage)". This folder keeps the extract, `smoke_freeze_and_shares.log`. The run directory was deleted afterwards.

| Check | Result |
|---|---|
| Printed lines | `freeze_obs_normalizers: froze ['_actor.mu.norm', '_critic.norm'] (9448 statistics)`, then after epoch 0: `verified -- 6 statistics buffers … bit-identical through this launch's first policy update (1 epoch(s); epoch counter now 1)`. The assertion did not fire |
| **Freeze over 10 epochs** (`compare_normalizer_stats.py`, G1's `epoch_5000.ckpt` against the smoke's `epoch_2`, `epoch_10` and `last.ckpt`) | **exit 0 on all three.** `_actor.mu.norm` and `_critic.norm` have count, mean and var **bit-identical** |
| Free normalisers moved | after 10 epochs, `_discriminator.models.0.norm`: mean max \|Δ\| 0.0023, var 0.0086; `_disc_critic.models.0.norm`: mean 0.0080, var 0.030. Counts are identical in all four normalisers (EMA mode), which is why the check compares mean and var |
| The check is not vacuous | The policy updated in 5 of the 10 epochs: 0, 3, 5, 7 and 9. Each of the other five follows an evaluation and skips its update. Actor and critic optimiser steps 163,680 → 163,687. Weights from `epoch_5000` to `last.ckpt`: actor `mu` max \|Δ\| 5.7e-4 (14 of 14 tensors changed), critic 2.8e-3 (10 of 10) |
| Manager start-up | `anchoring 40% of resets to 810 segments over 168/168 motions`; `departure-anchoring 20% of resets to 642 segments over 168/168 motions` (the replay's 642 of 810) |
| **Drawn shares** (`env/anchor/<kind>_resets ÷ env/anchor/resets`; all five keys at every epoch; the kinds sum to the resets exactly) | Pooled over the run's **761 resets**: arrival **0.430**, departure **0.198**, t = 0 **0.067**, uniform **0.305**, against 0.40 / 0.20 / 0.08 / 0.32; binomial z +1.67 / −0.11 / −1.32 / −0.90. 517 of those resets are in epoch 0, which holds the launch's two full resets (2 × 256). Epochs 1–9 alone have 244 resets: 0.455 / 0.242 / 0.061 / 0.242, z +1.75 / +1.63 / −1.07 / −2.62. At 256 envs this is small-sample noise. The precision measurement is the 4.10 M-reset offline replay above (0.16 %) |
| **Motion prior** | `HoldCurriculumEvaluator: package motion prior over 168 motions (weight: motions) {1.0: 168}`. Logged from the epoch-0 evaluation (TB step 1) on, and constant over all six evaluations (it is captured once): `eval/curriculum/prior_share/` arm_balance 0.1607 (27/168), connective 0.375 (63/168), inversion 0.2321 (39/168), single_leg 0.2321 (39/168). With every weight at 1.0, the prior share is the group's size share. At epoch 0, `group_prob/` reads 0.155 / 0.399 / 0.223 / 0.223, and `eval/curriculum/ess` goes from 153.9 at epoch 0 to 114.8 at epoch 10 |

Not measured here: the shares on v3, where a synthetic clip with no eligible departure segment lowers the departure
share (`expected_shares` in the replay script computes it). That belongs to G3's own `SMOKE=1`.
