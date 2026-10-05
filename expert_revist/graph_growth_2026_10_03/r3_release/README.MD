# Card R3: release v3 (2026-10-04)

**What it is.** Release v2 plus the 28 spliced synthetic clips of R2's `synthetic_v3.d034199f23` (its 29 minus the
user's drop of `SYN_E2_jumpback_mid_s3_t6rpx12`), built by `reference_curation/release_v3.py`, a wrapper around v2's
builders. The human part is release v2, slice for slice: every human file, row and plan is v2's, bit for bit. Nothing
in release v2, its builders, `splice_v3` or any runtime file was edited.

**The candidate: `holds_repaired_ftC_posefix.release_v3.2f132f4299`** (`purpose: candidate`).
- Heavy files (untracked, 2.0 GB): `data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v3.2f132f4299/`.
- Record: `data/reference_curation/releases/holds_repaired_ftC_posefix.release_v3.2f132f4299.json` (+ `.md`).
- Built in 88 s on one core; 53 checks pass in the build, **68** under `--check` (the build's plus the artifacts,
  plans and motions against the record, the record as training reads it, and the swing calibration re-run).
- The id is reproducible: `--print-id` and `--check` with today's builders give the same id.

| File | What |
|---|---|
| `data/scripts/reference_curation/release_v3.py` | the wrapper, every check, the record; CLI `--build [--drop ...] [--purpose candidate\|final]`, `--check <id>`, `--print-id`. Refuses to overwrite an id; `--force` deletes only a failed build's folder |
| `data/scripts/reference_curation/contact_targets_v3.py` | the sidecar: v2's `compile_targets` on a human-only view of the graph, then the synthetic rows |
| `data/scripts/reference_curation/physics_tables_v3.py` | the tables wrapper: `build_physics_tables_v2.main`, unchanged, with synthetic clips on its no-pressure path |
| `data/scripts/make_edge_probe_plans.py` | R4a: three plans per edge (`edge_`, `nohijack_`, `fork_edge_`) |
| `data/scripts/reference_curation/tests/test_release_v3.py` | 17 tests, 43 s (below) |
| `r3_release/diff_resolved_configs.py` | checklist item 4: a recursive diff of two `resolved_configs.pt` (dataclasses, plain objects, dicts, lists) |
| `r3_release/physx_reset_check.{json,log}` | checklist item 3's record |
| `r3_release/config_only_diff_vs_g1.txt` | checklist item 4's diff |
| `r3_release/smoke_v3.log`, `smoke_v3_tb.txt` | checklist item 5: the run log's key lines and every TensorBoard scalar |

## What the release holds

| | Release v2 | Release v3 (candidate) |
|---|---|---|
| Clips (groups) | 56 (single-leg 13, inversion 13, arm balance 9, connective 21) | 84: v2's 56 + 28 in group `edge` (E1 ×8, E3 ×3, B1 ×5, E2 ×11, E5 ×1) |
| Holds / motions / frames / inserted | 270 / 168 / 282,936 / 38,400 | 382 / 252 / 387,588 / 55,200 (synthetic: 112 / 84 / 104,652 / 16,800) |
| Package weights | 1.0 each | human 1.0; per synthetic motion 3/n: E1 0.375, E3 1.0, B1 0.6, E2 3/11, E5 3.0. Each edge weighs 9 (three human clips' x0/x3s/x7s); the edges hold **45 / 213 = 0.211** of the uniform mass |
| Graph | 123 nodes, 196 edges, 810 segments, 270 hold ids, ≤ 9 segments per motion | **123 nodes, the same keys in the same order**; 201 edges (the 5 new ones are exactly the S → D of the kept edges, none in v2 before); 1,146 segments; 382 hold ids (v2's 270 first); ≤ 9 segments |
| Tables | 345,854 swing zone-frames; pressure on 168 motions; COP-valid 768, share-valid 735 segments | 427,680 (synthetic **81,826, all from the velocity rule**); pressure on the same 168; COP-valid 768, share-valid 735 (none synthetic); lean-gated 90 → 207 |
| Sidecar (x0 holds) | critical 59, restored 4, masks `R_HAND:G` 1, known-free zone-holds 3,233 | critical 161, restored 4, masks the same 1, known-free 4,619 (synthetic: 112 holds, critical 102 = the crow shin braces, restored 0) |
| Plans | 148 | 163: v2's 148 (byte for byte) + 15 edge plans |
| Longest motion | 62.967 s → `eval_max_steps` ≥ 1,890 | the same (the longest synthetic, an x7s, is 27.367 s) |

The edge plans' clips are the card's R4a picks, reproduced by the rule (slowest timing, then the lowest final 6-body
error, then the release's order): E1 `SYN_E1_press_high_s0_t6px`, E3 `SYN_E3_lower_high_s3_t6px12`, B1
`SYN_B1_jumpplank_high_s1_t6rpx`, E2 `SYN_E2_jumpback_mid_s0_t6px12`, E5 `SYN_E5_floatdown_mid_s0_t6rpx12`. Panel
lengths: edge 8.1–11.5 s, no-hijack 9.95–10.22 s, fork 14.8–20.5 s, all under the launcher's 24 s cap.

## Acceptance

| Check | Result |
|---|---|
| `--build` writes the candidate record; `--check` passes | **68 of 68** (36 s) |
| Release v2's `--check` still passes | **26 of 26** (and run inside v3's checks on every build/check) |
| The human part is v2, slice for slice | manifest entries; the 168 `.motion` files (sha256 = v2's record); every v2 `lineage.npz` array; package: every per-frame field on v2's 282,936 frames and every per-motion value; graph: node tables, vocabularies and every segment row; tables: every row, the body constants, rules and non-path parameters; sidecar: every row, the vocabularies and rules, and v2's `contact_targets.json` verbatim; v2's 148 plans byte for byte; the graph description's human clips and every v2 edge's human occurrences. Bit for bit after padding (the padding is v2's: 9 segments, 3,778 frames) |
| v2's 26 checks restated on the whole of v3 | all pass: extension, package (incl. the synthetic motions frame by frame), graph (segments, A4 over the inherited hold ids, manual goals resolve on all 1,146 segments), tables, sidecar (commanded = configured − masks = `seg_contact`; counts over the inherited ids), the runtime loaders, the plans |
| Synthetic lineage (R2's checks on the clips as packaged) | real frames = release v2 after the transform to **6.3e-07 m**, `dof_pos` exact; real-frame velocities ≤ 0.0065 m/s (20 angular entries are genuine horizon ties); transition frames = the admitted variant's re-export (sha256) to 6.0e-07 m, `dof_pos` exact; FK round trip ≤ 4.6e-07 m; pressure zero with validity 0 on every frame; windows hold their labels; x3s/x7s = hold extension of x0 at S (kinematics exact, lineage read through the index) |
| Inherited labels | every synthetic hold's copied fields equal its `inherits` hold's; every synthetic sidecar row equals the inherited hold's row |
| Graph keys | the node-key list equals v2's (no id moves); `pair_names` and `orientation_names` equal v2's in order; each kept S → D present once per synthetic motion (E1 24/24, E3 9/9, B1 15/15, E2 33/33, E5 3/3) |
| Sidecar frames | real-frame pair gaps measured on the spliced geometry equal capture v4's to **2.2e-07 m** (the same kernel on rigidly placed frames); **0** known-free conflicts |
| `calibrate_swing` on the human x0 stems | equals v2's `calibration_c3` on every key but `all_motions` (which now counts the synthetic motions) |
| Runtime loaders | `ContactGraph` (+ `validate_against_motion_lib`), `PhysicsTables`, `ContactTargets` accept the release; `release_identity.require_artifact` on the 5 roles and the plant check accept the record; v2's graph is refused as v3's |
| Tests | `test_release_v3.py` **17 passed** (43 s): the drop by exact stem, the purposes, D5, padding, the tables wrapper, the generator's phase rule, the plan-clip rule; on the candidate, the human slice with a negative control per artifact, the node keys with a builder-level negative control (one renamed synthetic hold gives 124 nodes), a perturbed real frame and a perturbed x3s frame failing the lineage checks, the inherited rows with a negative control, the known-free check re-derived on another path (graph times, the generator's own `Schedule`) with a negative control, the edge plans, the record and the runtime loaders, `--check`. `test_release_v2.py` + `test_splice_v3.py`: 30 passed (no regression) |
| Launcher `DRY_RUN=1` | passes every check: 28 variants, `SYN_=0.5` on 84 of 252 motions, 55 demonstrations with no `SYN_` among them, 20 panel plans resolve (10 G1, 5 edge, 5 no-hijack), eval 2,250 ≥ 1,890 steps |
| **Checklist item 3** (PhysX reset check, GPU) | `retarget_v2_physx.py` on the release's 84 synthetic motions (symlinked), `--per-clip 20`: the library loads on `smpl_yogi_v2` (a v1 library is refused); stored bodies = PhysX FK to **8.3e-07 m**, 1.2e-04° over 1,680 frames; **0 of 1,680 resets launch** (positive control 3/3; max upward root speed 0.0022 m/s; lowest collider at reset −0.06 cm). `frames_of` takes stems without labels-v1.1 exemplars (it falls back to the evenly spaced frames); the record is written here, not into the retarget's |
| **Checklist item 4** (`CONFIG_ONLY=1`) | against G1's `resolved_configs.pt`: **21 differences, all intended**: E6's 12 (the name; `segment_start_prob` 0.6 → 0.4; `segment_end_prob` 0.2; `training_max_steps`; `motion_prior` `package`; `freeze_obs_normalizers` True; `amp_reward_w_target` 0.5, start 0, full 0, calibration 0; `demo_exclude_motions` + `SYN_`; `amp_lineage_weights` `{'SYN_': 0.5}`) and 9 that name release v3: the package, graph (×2), tables, sidecar, record, hold manifest, the panel's plan files and `num_sequences` 22 (20 plans + 2). Log: `output/graph_growth_r3/config_only_diff_vs_g1.txt` |
| **Checklist item 5** (`SMOKE=1`) | `SMOKE=1 EXPERIMENT=zz_r3_2f132f4299` (256 envs × 4 epochs, from G1's `epoch_5000.ckpt`): no Traceback; **311 scalar tags, 0 non-finite values**. Log lines present: `[amp lineage] rule 'SYN_' (w 0.5) matched 84 of 252 motions` (`84 … 'syn'; 168 … 'human'`); `[amp demos] 55 of 252 motions serve as demonstrations` (excluded: 168 by `_x\d+s$`, 29 by name = Scorpion -b + the 28 SYN x0; lineage-matched among them 0); the AMP state restored (disc step 39,920) with target 0.5; `ContactGraphControl: release … every loaded artifact matches its record`; the package prior `{0.2727: 33, 0.375: 24, 0.6: 15, 1.0: 177, 3.0: 3}`; anchoring 40 % over 1,146 segments and departures 20 % over 894. The epoch-0 evaluation (TB step 1) loaded v3's manifest and logged every group, **`edge` included** (`edge_score` 0.969; the smoke's 150 steps see only the lead-ins); `eval/curriculum/prior_share/edge` **0.2113** (= 45/213); `amp/reward_w` 0.5 at every epoch; `advantages/style_to_task_std` 0.26–0.31; `amp/syn_sample_share` 0.18–0.21; drawn starts at step 1: arrival 0.42, departure 0.18, t = 0 0.07, uniform 0.34. The normaliser freeze verified bit-identical. `Skipping actor updates (clip_frac 0.65–0.71 > 0.6)` is the known 256-env smoke artefact (§2). Evidence: `smoke_v3.log`, `smoke_v3_tb.txt`; the scratch `results/zz_r3_*` (3.0 GB) were deleted |

## Deviations from the card, and why

1. **v2's 26 checks are restated on v3, not called on it.** `release_v2.check` reads per-clip human evidence
   (gated ports, the gate manifest, capture v4) and would crash on a synthetic clip, so v3 runs v2's own `check` on
   release v2 (26/26) and restates each of the 26 over the whole of v3, reading the synthetic clips' lineage where
   v2's read human evidence. 53 checks in all, 68 under `--check`.
2. **`lineage.npz` adds three arrays per synthetic variant beyond the card's four** (`index`, `inserted`, `kind`,
   `source`): `source_frame`, `variant_t` and `stems`. With them the sidecar resolves a frame's source evidence and its
   phase from the release alone; `source` indexes `stems` ([S stem, D stem, variant]). All are R2's per-clip lineage
   read through `index`, which a check verifies.
3. **The tables wrapper is a view, not a fork.** `physics_tables_v3` runs `build_physics_tables_v2.main` in-process,
   unchanged, with the builder's `torch` replaced by a proxy whose `load` returns a synthetic clip's motion without
   its three measured channels. `pressure_fields` then returns None and the clip takes the builder's own
   pressure-free path. Human motions load untouched; every synthetic stem must be stripped exactly once.
4. **The sidecar's human rows are v2's compiler on a human-only view**: a temporary graph payload holding v3's
   first 168 motion rows (v3's padding and hold-id list), the release's manifest and lineage linked in. The view's
   graph sha256 is replaced by v3's. The rows equal v2's sidecar bit for bit (a check).
5. **The phase rule is the generator's** (`sketch.Schedule.phase_at` / `config_at`, mirrored without importing
   MuJoCo; a test checks they agree on every variant's clock). Settle frames (t < 0) plan S's configuration, the
   hold at D (t ≥ T) plans D's.
6. **The card's "ref will see the E2 knee and thigh contacts" does not reach the sidecar.** Its per-frame arrays
   cover only the 24 configured body-body pairs; floor contacts are not among them. The known-free mask applies only
   inside the commanded hold's window (`in_hold`), so the E2 knee touches on transition frames are not charged by the
   support term. They stay recorded (`synthetic.transition_contacts` in the record).
7. **The known-free check is read for every planned frame**: a synthetic or blend frame's planned ground zones may
   not be known-free in the hold the schedule commands there (the window's hold, or outside every window the next
   hold, `include_current_segment`'s slot 0). 0 conflicts.
8. **`calibrate_swing` must equal v2's on every key but `all_motions`**, which counts every v3 motion. It runs v2's
   own function on a view whose manifest lists the human motions only.
9. **R4a's plan details the card left open**: the first goal of `edge_` and `nohijack_` is the dwell plans' (reach
   0.5 s, hold 3 s); the no-hijack's second goal is the S clip's own next hold with reach = that clip's own
   transition time + 0.5 s (Crow -a 3.72 s, Handstand -a 3.45 s), so the causality pair differs only in the second
   goal; the fork's S reach is `make_hold_graph_probe_plans`' fork rule on S's own clip (crow 7.25 s, handstand 10 s,
   the cap).
10. **`--purpose final` requires drops beyond the user's**; with none, the candidate is the final (D3). `--drop`
    defaults to the user's drop, and every build must hold it.

## What G3 must know

- **`eval/perf_group/edge_*` reads S's hold, not the edge.** A synthetic clip's only `extend` hold is S (crow or
  handstand), and the evaluator's family metrics are its `extend` holds, so `edge_family` scores the source hold.
  The transition and the arrival at D are R4b's `edge_tracking.py`.
- **Crow now has the most dwell in the graph**: 505.8 s over 84 segments, against standing's 501.8 s. In v2 the
  crow node had 21.5 s over 3 segments, the handstand 63.7 s (now 310.1 s) and standing 423.2 s. Each synthetic
  clip carries S's window (and D's), and x3s / x7s grow S's by 3 / 7 s. Sampling is set by the package weights,
  not by dwell; the only runtime reader of dwell is SequenceViz's `derive_node_dwell`, which picks the panel's
  graph-derived hold sequences by dwell. The panel's 2 extra slots (22 = 20 plans + 2) therefore hold crow and
  standing (G1's held standing and Pincha), the crow one placed at the node's longest segment, a synthetic clip's
  S window.
- **Synthetic stems still contain each other** (9 collisions among the kept clips, all `t6px` ⊂ `t6px12` /
  `t6rpx` ⊂ `t6rpx12`); runtime filters match substrings. `SYN_=0.5` matches all 84, as intended.
- **The release record has `purpose`, `drop` and `variants`**; the launcher echoes them.

## Open, for the orchestrator

- Checklist items 6 (`TIMING=1`) and 7 (disk) ran the same evening and pass: peak GPU memory 23,133 MiB of 24,564,
  9.71 s per epoch, 128 GB free. See [`../g3/prelaunch/README.MD`](../g3/prelaunch/README.MD), which also records a
  warm-start transient in the first 10 epochs and G1's tracked share on the synthetic clips before fine-tuning.
- R4b (the funnel test, `edge_tracking.py`, `g3_gates.py`) is not built.
- A final release, if G2 drops variants: `--build --purpose final --drop SYN_E2_jumpback_mid_s3_t6rpx12 <G2's
  failures>` (exact stems).

## How to run

```bash
export PYTHONPATH=.:data/scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # no REFERENCE_PLANT
../env_isaaclab/bin/python -m reference_curation.release_v3 --print-id          # holds_repaired_ftC_posefix.release_v3.2f132f4299
../env_isaaclab/bin/python -m reference_curation.release_v3 --check holds_repaired_ftC_posefix.release_v3.2f132f4299   # 68 checks, 36 s
../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests/test_release_v3.py -q          # 17 tests, 43 s
```
