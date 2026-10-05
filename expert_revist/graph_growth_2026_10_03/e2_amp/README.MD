# Card E2: AMP for the goal-conditioned expert (2026-10-03)

**What it is.** The release-v2 expert plus an AMP style reward, warm-startable from epoch 15,500
(`expert_revist/graph_growth_2026_10_03/PLAN.MD` §1.2, card E2). `mlp_goal_conditioned.py` is not touched by this card
(card E1 added its `--support-rule` flag, default v1).

| File | What |
|---|---|
| `protomotions/envs/obs/amp_features.py` | `amp_features_v1`: heading-free, 163 values per frame (root height; gravity, root linear and angular velocity in the root frame; 23 parent-relative tan-norm rotations; head, hands, toes in the root frame). One per-frame function serves both sides: `compute_amp_features_v1_from_state` (the env's history buffer, step k = buffer index k = t − k·dt) and `compute_amp_features_v1_from_motion_lib` (the library at t − k·dt). Window steps `[1, 2, 3, 4, 6, 9, 14, 20]` (0.67 s) |
| `protomotions/agents/amp/goal_conditioned.py` | `GoalConditionedAMP(AMP)` + `GoalConditionedAMPComponent`: x0-only, per-frame-uniform demonstrations (t ≥ 20 dt, never Scorpion -b); the weight schedule (0 for epochs 0–199, linear to the target at 500, set inside `add_advantages`); unweighted style-advantage diagnostics; the per-lineage AMP weight hook (`amp_lineage_weights`, empty = 1.0); a one-time on-GPU parity self-check |
| `examples/experiments/mimic/mlp_goal_conditioned_amp.py` | Loads the base expert by path and adds `amp_obs` (history 15 → 20), the discriminator (MLP 1024-512 on `amp_obs`), the discriminator critic (3 × 1024 on `CRITIC_KEYS` + `amp_obs`), threshold 0, GP 5 by default (G1's launcher passes 10, below), replay 200k / keep 0.01, weight decay 1e-4 / logit 0.01; `--amp-calibrate-style-ratio` for the one-shot weight calibration |
| `protomotions/tests/test_goal_conditioned_amp.py` | 5 tests: the schedule, the demonstration set, the calibration (median window, freeze, clamps, off, no-history fallback) |
| `data/scripts/run_expert_amp_ft.sh` | G1's launcher: `run_expert_release_v2.sh`'s arguments + `--checkpoint …/epoch_15500.ckpt --warm-start-optimization-state` + `--support-rule v2`. `CONTROL=1` = the matched no-AMP continuation (base experiment file), `SMOKE=1`, `TIMING=1`, `CONFIG_ONLY=1` |
| `protomotions/tests/test_amp_features_v1.py` | 5 tests: 163 dims, heading invariance (yaw + XY shift ≤ 1e-5), step k reads buffer index k, **agent/demonstration parity on the release library**, the demonstration component binds through `AMPTrainingComponent._call_ref_obs_fn` |
| `amp_separability.py`, `frame_audit.py`, `velocity_audit.py` (this folder) | The offline separability check and its two follow-ups |

## Acceptance

| Check | Result |
|---|---|
| Parity test (≤ 1e-5) | **0.0** on 4,096 release windows (CPU, eager); **7.75e-07** max on the GPU through `torch.compile`, all 256 freshly reset envs of the smoke (`amp/parity_max_abs_amp_obs`, logged once per run) |
| Resolved configs vs the e15500 run's | only the intended differences: agent class/model (AMP), the new AMP fields, E1's support-rule fields, `training_max_steps` (5,000 epochs), env history 15 → 20 and `amp_obs`. Robot, simulator, terrain, motion lib, actor, critic, optimisers, evaluator and viz identical |
| Smoke (256 envs × 4 epochs, style weight on from epoch 1) | finite losses: discriminator 0.77 → 0.76, class 0.68, disc critic 1.41 → 0.13; `pos_acc` 0.63 / 0.61, `agent_acc` 0.55 / 0.48 (chance, as expected after 4 epochs); unnormalised AMP reward 0.70; no discriminator terminations; warm start restored the reward normaliser, PPO optimisers and advantage EMA |
| Timing (≤ +10 % s/epoch) | **+11.4 %, marginal miss**: 10.01 s/epoch (p50, epochs 5–50, 4,096 envs) against 8.99 s for the matched no-AMP warm start (`CONTROL=1 TIMING=1`, same episode-length transient: 427 vs 433 steps, `clip_frac` 0.27 vs 0.28). That is +1.4 h over 5,000 epochs. Recorded, not blocking. GPU memory 22.8 of 24.5 GB |
| First evaluation reproduces e15500's group scores within 0.01 | **yes** (G1's epoch-1 evaluation; a loaded checkpoint always evaluates after its first epoch): single-leg 0.9997 / inversion 0.9230 / arm balance 0.9792 / connective 0.9789 against 0.9999 / 0.9228 / 0.9766 / 0.9827; jerk 71.71 vs 71.90; drag 38.79 vs 38.41 J |
| Offline AUROC reported | below |
| Resume | a 4-epoch smoke re-run on its own `last.ckpt` resumes cleanly with the discriminator, both AMP optimisers, the AMP reward normaliser and the frozen calibration (`amp_weight_calibration`) |

**A warm-start effect the smoke surfaced, not caused by AMP.** At 256 envs (batch 1,024) both the AMP smoke and the
matched control smoke skipped the remaining actor minibatches at epochs 0 and 3 (`clip_frac` 0.64–0.77 > 0.6,
`approx_kl` 0.11–0.32), against the e15500 run's steady `clip_frac` p50 0.28 (max 0.34, never skipped) at batch 16,384.
Same in both arms, so it is the small-batch warm start against an Adam state restored from 16× larger batches, not AMP.

## Offline separability (item 5): perfect, and it is real posture, not a feature mismatch

`amp_separability.py`: 137,940 agent windows from the e15500 evaluator rollouts (130,899 tracked: every body within
0.5 m across the window) against 38,697 x0 reference windows at the same clip times; MLP 512-256 on standardised,
clamped inputs.

| Classifier | AUROC (tracked) | by family: single-leg / inversion / arm balance / connective |
|---|---|---|
| in-sample (random 20 % held out) | **1.000** | 1.000 / 1.000 / 1.000 / 1.000 |
| leave-clips-out (5 folds over the 56 clips) | **0.991** | 0.996 / 0.999 / 0.994 / 0.974 |
| newest frame only | 1.000 | |
| block: local rotations (all frames) | 1.000 | |
| block: key bodies | 0.949 | |
| block: root velocities | 0.840 | |
| block: posture (height, gravity) | 0.676 | |

The plan says near-perfect separability is "a feature mismatch or a real artefact; find out which before spending GPU".
The top single features are one tan-norm component of the left hip and of the torso, offset by |d| ≈ 0.5 *at every lag*
— the signature a frame-convention bug would leave. Two audits rule that out:

- **No convention mismatch** (`frame_audit.py`). Simulator body rotations equal pose_lib FK of the simulator's own
  joint coordinates to 0.000° on every joint (p99 0.000°); the reference library's stored rotations equal its stored
  dofs to 0.000°. No body of plant v2 has a rest-rotation offset.
- **The difference is the policy's posture.** At matched clip times the policy's parent-relative rotations sit a median
  10–15° from the human's on every joint, with systematic biases of 3–6° (|mean| / sd 0.3–0.6) in the loaded joints —
  left hip 6.3°, right hand 6.3°, torso 5.1°, right thorax 4.8°, right hip 4.5° — the PD-droop pattern. No single joint
  separates on its own (one joint's rotation, newest frame: AUROC 0.61–0.76; head highest): the separation is the sum
  of small, real, distributed deviations.
- **Velocities mean the same thing on both sides** (`velocity_audit.py`). Stored root velocities agree with central
  differences of the stored root pose on both sides (linear: 0.018 m/s residual on 0.169 rms for the reference, 0.028 on
  0.156 for the agent). The agent's *angular* velocity residual is larger (0.20 vs 0.085 rad/s): PhysX's instantaneous
  velocity carries oscillation between control steps — the control-rate jitter the features are meant to expose. At the
  AMP sampling rate the agent's root velocity has slightly *more* second-difference energy than the reference's (median
  0.0099 vs 0.0089 m/s), so the style reward cannot pay for added jitter.

So on human clips the discriminator will be sharp, as the plan predicted ("its measurable effect there will be jitter");
expect `agent_acc` above the 0.55–0.85 band and apply the plan's remedy (GP 5 → 10) if it stays above 0.9.

## Two settings changed from the card's starting point, both by the card's own rules (measured 2026-10-03)

The 50-epoch timing run (style weight still 0) measured the two quantities the card says to tune by:

| | epochs 1–10 | 11–30 | 31–50 | card's rule |
|---|---|---|---|---|
| `discriminator/agent_acc` (GP 5) | 0.76 | 0.95 | 0.98 | raise GP 5 → 10 if agent accuracy stays above 0.9 |
| `discriminator/pos_acc` | 0.70 | 0.97 | 1.00 | health band 0.55–0.85 |
| unnormalised AMP reward | 0.67 | 0.56 | 0.46 | band 0.3–1.0 |
| style/task advantage-std ratio *at w = 0.1* | 0.078 | 0.081 | 0.069 | tune w so that it is 0.2–0.35 |
| discriminator-critic explained variance | 0.39 | 0.81 | 0.93 | |

- **Gradient penalty 10.** Agent accuracy passed 0.9 by about epoch 11, as the offline AUROC of 1.0 predicted.
- **The weight is calibrated, not set to 0.1.** At the card's starting guess the style advantage would carry about
  0.6 % of the advantage variance, a quarter of the band's lower edge, and G1 could not move jerk. The card says to
  calibrate "by advantage scale, not by guess". So `--amp-calibrate-style-ratio 0.25` freezes the target once, at
  epoch 200: 0.25 divided by the median raw style/task std ratio of epochs 100–199 (all at w = 0, so the style reward
  has not yet changed the policy), clamped to [0.05, 0.5]. It is logged as `amp/reward_w_target`, kept in checkpoints,
  and the ramp to it runs over epochs 200–500 as planned. `AMP_CAL_RATIO=0` restores the fixed `AMP_W`.

## G1 launch

`bash data/scripts/run_expert_amp_ft.sh` (2026-10-03 13:51 PDT): `results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac`,
wandb `sfgpfbn5`, log `output/graph_growth_g1/g1.log`, 5,000 epochs at about 10.1 s each, plus an evaluation every 500.

**First 213 epochs** (all at w = 0 until 200; GP 10):

| epochs | 2–49 | 50–99 | 100–149 | 150–199 | 200–213 |
|---|---|---|---|---|---|
| s/epoch | 10.09 | 9.76 | 9.73 | 9.74 | 9.73 |
| actor `clip_frac` / KL | 0.270 / 0.0216 | 0.276 / 0.0220 | 0.279 / 0.0222 | 0.275 / 0.0217 | 0.276 / 0.0218 |
| `discriminator/pos_acc` / `agent_acc` | 0.981 / 0.951 | 0.985 / 0.984 | 0.988 / 0.986 | 0.998 / 0.985 | 1.000 / 0.984 |
| unnormalised AMP reward | 0.59 | 0.49 | 0.41 | 0.36 | 0.32 |
| raw style / task advantage std | 0.85 | 0.66 | 0.54 | 0.47 | 0.43 |
| disc-critic explained variance | 0.83 | 0.96 | 0.98 | 0.98 | 0.99 |

- **Calibration at epoch 200:** the median raw ratio over epochs 100–199 was 0.499, so the target is 0.25 / 0.499 =
  0.501, clamped to **w = 0.50** (5× the card's starting guess). The ratio fell while the discriminator critic learned,
  so at w = 0.5 the style/task std ratio will sit near 0.21, at the band's lower edge.
- **The discriminator stays above its health band at GP 10** (agent accuracy 0.98). The card has no further remedy;
  read the band again after the ramp, when the policy is paid to move toward the reference.
