# Card E1: evaluator support rule v2 (2026-10-03)

**What it is.** `HoldCurriculumEvaluator` now also scores every hold by *load* on the zones the release's sidecar says
the human keeps free (`hold_curriculum.support_v2_holds`): a known-free zone violates when its terrain-filtered
vertical load is ≥ 3 % BW (21.8 N on plant v2's 74 kg) on ≥ 20 % of the hold window `[t_start, t_end]`, the load
mask dilated ±7 frames (±0.23 s) first, inside the window. Geometry alone never counts. Commanded supports are
"realised" when every commanded zone's lowest collider point is ≤ 2 cm on ≥ 90 % of the window.

**What it changes.** Nothing logged before changes: `eval/perf/*`, `eval/perf_group/*`, the CSV's existing columns and
the score behind `score_based.ckpt` stay v1. v2 goes to new keys (`eval/perf_v2/*`, `eval/perf_group_v2/*`,
`eval/perf/support_violation_v2`, `eval/perf/support_realised_v2[_x0]`, `eval/perf/substitution_holds_v2[_x0]`,
`eval/perf/holds_tracked_v2[_x0]`) and to CSV columns appended at the end. `--support-rule v2`
(`HoldCurriculumConfig.support_rule`) makes only the curriculum's sampling run on the v2 score. `_x0` keys count the
unextended clips, the population of the e15500 review. Evaluators now also save the simulator's ground forces in the
predicted library as `sim_rigid_body_ground_forces` (not MotionLib's `gnf`, which means *measured* force).

## Acceptance on e15500 (`predicted_motion_lib_epoch_15500.pt`, CPU)

| Check | Result |
|---|---|
| v1 re-score vs `curriculum/eval_epoch_015500.csv` | 0 of 1,008 cells differ (6 columns × 168 motions, 4 decimals) |
| Commanded supports realised, x0 tracked holds | **232 / 260**, exactly README §4's |
| README §4's 10 substitution holds | all 10 flagged (proxy, below) |
| Koundinya -a @20.35 head hover | not flagged (proxy); geometry-only would flag it |
| Default `--support-rule v1` builds the e15500 run's configs | evaluator and agent configs: 0 differences from its frozen `resolved_configs.pt`, which still unpickles |
| Unit tests | `protomotions/tests/test_hold_curriculum_support_v2.py`, 13 pass (65 with the evaluator/release suites) |
| Evaluator path = offline path | identical flags on identical forces (`check_evaluator_path.py` vs `rescore_library.py --synth-forces`) |
| Cost | +0.8 s per evaluation on the CPU (scoring 0.56 → 1.35 s for 168 motions) |

**The e15500 library carries no ground forces**, so "loaded" is a proxy there: the per-body contact flag on any of the
zone's bodies **and** the zone's lowest collider point ≤ 2 cm. The flag alone (body–body contact included) flags 115
holds; geometry alone flags 20 including the hover; the proxy flags 14 tracked x0 holds: the README's 10, plus four the
README excluded by its "reference keeps the zone ≥ 4 cm up" criterion — Bridge -a's right hand @14.13 and @17.60 and
Upward Plank -a's left thigh @11.32 (reference 2.3–2.4 cm up: "a hand beside the hip, a seated thigh") and
Koundinya -a's right foot @17.22 (3.0 cm). The sidecar's human evidence says all four zones are free, so v2 counts them
if they carry ≥ 3 % BW; whether they do is only measurable with forces — **read G1's first saved library with
`rescore_library.py` (its `force` mode runs automatically)**. Per-body synthesised forces (flag and lowest point on the
same body) drop Eagle -a @7.23 (the README's 50 %-flag hold) and give 13.

Files: `rescore_library.py` (acceptance re-score, all modes) → `e15500.json`; `check_evaluator_path.py` (the evaluator's
own path on a library) → `e15500_evaluator_path.json`; `check_config_identity.py` (acceptance 5).

## With real forces: G1's epoch-1 library (2026-10-03)

`rescore_library.py --library results/smpl_yogi_v2_expert56_amp_ft_a2dda5d2ac/results/predicted_motion_lib_epoch_1.pt`
→ `g1_epoch1_forces.json`. G1's epoch-1 policy is e15500 plus one PPO update; its evaluation reproduces e15500's group
scores within 0.004. On this one rollout:

| Mode | Tracked x0 holds flagged | README's 10 |
|---|---|---|
| **force** (the runtime rule) | **6** | hits 6; misses Eagle -a @13.33, @11.67, @7.23 and Bhujapidasana -a @9.05 |
| proxy (flag AND ≤ 2 cm) | 12 | hits all 10 |
| geometry alone | 18 | hits all 10, and flags Koundinya -a's hover |
| contact flag alone | 111 | hits all 10 |

v1 against the epoch-1 CSV: 0 of 1,008 cells differ. Supports realised: 232 / 259.

**The four misses are hovers, not supports.** On the same frames where the proxy fires, the measured vertical ground
load is under 1 N on every frame of the window:
- Eagle -a's wrapped right foot (`R_Ankle` + `R_Toe`) @7.2 and @13.3 (and > 1 N on only 9 % of @11.7);
- Bhujapidasana -a's left foot @9.1.

The contact flag comes from body-body contact (the wrapped foot on the calf), and the toe tip sits within 2 cm of the
floor. This is the same failure mode as Koundinya -a's head. So the e15500 review's count of 10 included four
near-floor hovers. **Under the force rule the substitution baseline is 6**, so G1's gate "no more than e15500's 10" is
read against 6.

The six are load-bearing:
- Peacock -a's head @13.1: 208 N median, 29 % BW, 100 % of the window;
- Side Crow -c's right foot @22.5: 66 N on 82 %;
- Koundinya -b @17.7 and @19.6 (right foot);
- Side Crow -a @14.5;
- Upward Plank -a @23.5 (head, right forearm).
