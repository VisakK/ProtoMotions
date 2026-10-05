# G3 pre-launch checklist, items 6 and 7 (2026-10-04)

Both pass on the candidate release `holds_repaired_ftC_posefix.release_v3.2f132f4299` (card R3; items 2-5 are in
[`../../r3_release/README.MD`](../../r3_release/README.MD)). Item 1 (GPU free, no `train_agent.py`) is re-checked at
the launch.

## Item 6: `TIMING=1` (4,096 envs × 50 epochs)

```bash
PY=../env_isaaclab/bin/python RELEASE=holds_repaired_ftC_posefix.release_v3.2f132f4299 EXPERIMENT=zz_g3_2f132f4299 \
CHECKPOINT=results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/epoch_5000.ckpt TIMING=1 bash data/scripts/run_expert_graph_ft.sh
```

Run 19:51-20:07 PDT, alone on the GPU (nothing else on it; only the desktop on the CPU, load ≈ 2.7), with
`nvidia-smi` logged every second. The launcher keeps the real `EVAL_MAX_STEPS` (2,250) in this mode, so the loaded
checkpoint's evaluation after epoch 0 ran over all 252 motions and its buffers are in the peak.

| | Measured | Gate / reference |
|---|---|---|
| **Peak GPU memory** | **23,133 MiB** of 24,564 (the plan's "GB" = MiB/1,000: **23.13 of 24.56**), during training at 20:05:57 | < 24,000 MiB: **pass**, by 867 MiB |
| By phase | startup 22,313; epoch 0 + the full evaluation 22,373; training epochs 1-50 23,133 MiB | G1's timing run 22.8, G1's full run 23.6 |
| **Epoch time** | **9.711 s p50** over epochs 5-50 (E2's protocol); mean 9.708, range 8.80-10.47 s | about 10 s: **pass** (G1 9.87 s median; E2's G1 timing run 10.01 s) |
| The evaluation epoch | 344 s (epoch 0, the evaluation over 2,250 steps, its rollout library and `score_based.ckpt`) | |
| Health | 325 scalar tags, 0 non-finite; `amp/reward_w` 0.5 at every epoch; `advantages/style_to_task_std` 0.23-0.31; the normaliser freeze verified bit-identical | |
| **A warm-start transient** | **7 of the first 10 epochs (3-8 and 10) skipped the rest of their actor update** (`clip_frac` 0.60-0.66 > 0.6 in a minibatch). `actor/approx_kl` rose 0.08 → **1.65 at epoch 5**, was back to 0.03 by epoch 14 and stayed there; the epoch-mean `clip_frac` peaked at 0.53 (epoch 5) and fell to 0.21 by epoch 50 | G1's timing and full-run logs have no skipped update; E2 measured `clip_frac` 0.27 at 4,096 envs |

**What it implies for G3.** 6,000 epochs at 9.71 s are 16.2 h; 12 evaluations at about 5.7 min and 12 panel runs add
about 1.5-2 h, so about 18 h (the plan said about 19 h). **The memory margin is thin:** G1 went from 22.8 GB in its
timing run to 23.6 GB over its full run. If G3 grows by the same 0.8 GB, it reaches about 23.9 of 24.56 GB, under the
card's 24.0 GB line but with nothing to spare. Nothing else may use the GPU while G3 runs (§2, Operations).

**Update (G3's own launch, 20:20 PDT): the transient did not repeat.** Through G3's epoch 9, `approx_kl` stayed at
or below 0.14 and `clip_frac` at or below 0.46, with no skipped update. It is run-to-run variation, not a fixed
property of v3. What follows was written before that.

**The transient's likely source** is the shift G3 introduces at once: 84 new motions, three edges of which G1 tracks
poorly (below), and 20 % of starts moved to departures. The PPO guard did what it is for and the run settled within
about 14 epochs. Whether it costs the human clips anything shows at G3's first evaluations (its no-forgetting gates).
Attributing it to the clips or to the sampling would need a v2 timing run with G3's settings, which was not run.

**A free "before" for G2.** The loaded checkpoint's evaluation is G1's `epoch_5000` on the 28 synthetic clips before any
fine-tuning (`eval_epoch_1_g1e5000_on_v3.csv`). `p_track` is the evaluator's share of a clip's frames whose max-body
error is under the training termination's threshold, over the whole clip; it is not R4b's transition-window measure
and does not say where tracking was lost. Human groups: single-leg 0.999, inversion 0.944, arm balance 0.999,
connective 0.974; `edge` 0.701.

| Edge (x0 clips) | `p_track` | `p_family` (S's hold) |
|---|---|---|
| E2 jump-back ×11 | 0.962-0.968 | 1.0 |
| B1 crow → plank ×5 | 0.834-0.964 | 1.0 |
| E5 float-down ×1 | 0.588 | 0.0 (the handstand) |
| E1 press ×8 | 0.373-0.468 | 1.0 (crow) |
| E3 lower ×3 | 0.102, 0.103, 0.544 | 0.0 (the handstand; the clip starts in Handstand -a's mid-clip 3-point hold) |

G3's own epoch-1 evaluation will reproduce this (same checkpoint and library, up to PhysX nondeterminism), with its
rollout library saved.

## Item 7: disk

**128 GB free** (86 % of 916 GB used) after the scratch runs were deleted. Gate ≥ 20 GB: **pass**.

What G3 will write, from measured sizes:
- 14 checkpoints (`epoch_500` … `epoch_6000`, `last`, `score_based`) at 0.773 GB: 10.8 GB (G1's are 772.96 MB; this
  run's `last.ckpt` 772.93 MB);
- rollout libraries at **0.409 GB** on v3 (v2: 0.298 GB). G1 wrote them at epochs 1, 1,500, 3,000 and 4,500, so G3
  writes 4-5: about 2.0 GB;
- logs, panel videos and wandb: about 0.3-0.5 GB (G1: 0.27 GB).

In all about 13.3 GB, against the card's 13 GB estimate.

## Files

| File | What |
|---|---|
| `timing.log` | the run log's key lines (the launcher's echo, the release check, the AMP state, the freeze) |
| `timing_tb.txt` | every TensorBoard scalar of the run |
| `timing_gpu_mem.csv` | `nvidia-smi` memory and utilisation, every second |
| `eval_epoch_1_g1e5000_on_v3.csv` | the evaluator's per-motion scores at the loaded checkpoint (G2's "before") |

The scratch run `results/zz_g3_2f132f4299_timing` (1.9 GB) was deleted. The full log is
`output/graph_growth_g3/timing.log`.
