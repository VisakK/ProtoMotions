# Lane T: synthesised edges, cards T0–T5 (2026-10-03, CPU only beside G1)

**What it is.** The generator of PLAN.MD Step 2: about five edges between released holds that no clip demonstrates,
produced by trajectory optimisation that uses the offline reviewer's contact labels of the end holds, and the gate
that admits them (`../PLAN.MD` §1.3, cards T0–T5). Everything ran on the CPU while G1 trained (§"Running beside a GPU
job" below); nothing here touched the GPU or G1's code. All code is in `data/scripts/edge_synthesis/`.

| File | Card | What |
|---|---|---|
| `edge_spec.py` → `../edges.json` | T0 | One machine-readable spec per edge (E1–E5, B1, B2, the free edge): endpoints, contact edits, known-free zones, the phase schedule with duration ranges and bases, statics rows, e15500 tracking, hand anchors. `--check` re-derives and compares |
| `plant_mj.py` → `../t1_plant_mj.json` | T1 | Plant v2 in MuJoCo for batched rollouts: PD servos at the training gains (gear 1, `actuatorfrcrange` limits), dt 1/240 `implicitfast`, 8 substeps per 30 Hz step, floor 0.75 / robot 0.5 with **elliptic** cones, condim 3, hinge limits off, **per-body hinge orders** (below), converters exp-map ↔ hinge, numpy FK, the exp-map box, ground-force sensors per zone, `mujoco.rollout` with a persistent pool. `--acceptance` |
| `gpu_guard.py` | ops | `be_polite()`: nice 19 + affinity to one 4-core E-core cluster (CPUs 16–19); `epoch_seconds()` reads the training job's per-epoch time from its TensorBoard file |
| `mppi.py` | T3, T1 | MPPI over PD-target splines (4 knots / 0.8 s, N 192–256, 2 iterations, λ 0.05, replan every 0.1 s), `Rollouts` (boundary states, per-step force peaks, FK on demand), the benchmark's hold objective. `--hold` = T1's regression |
| `sketch.py` | T3 | Endpoints on one floor (D moved by yaw + XY so its hands land on S's), the **hand-anchored** minimum-jerk slerp sketch, `RefSketch` (a dense T2 reference as a sketch), the contact schedule (`Schedule.masks`) |
| `costs.py` | T3 | The transition cost library, term by term (module doc): decaying tracking, terminal attainment, planted supports, hand anchor, palm flatness, slip, landing placement, known-free zones (head/trunk 6 cm), landing ≤ 3 BW, balance + **capture point** on quasi-static phases, torque, speed, smoothness, the exp-map box |
| `edge_mppi.py` | T3/T4 | One edge end to end: endpoints → sketch (`--keyframes land`, `--via <hold>`, or `--reference <T2 dir>`) → MPPI with optional **statue feed-forward** (`--feedforward`) → `output/edge_synthesis/runs/<run>/{trajectory.npz, run.json}` with the T3/T5 metrics |
| `quasistatic.py` | T2 | Keyframe drafts (hybrids of the endpoints), the anchor-only `retarget_v2.Problem` built from the schedule, the LM solve, statics every 3rd frame and the card's contract → `output/edge_synthesis/quasistatic/<edge>_<timing>/{reference.npz, record.json}` |
| `export_motion.py` | T4 | An executed run → plant-v2 `.motion`: 60 fps, **projected** into PhysX's box by T2's solver under the run's own executed contacts, then `fit_writer.write_motion` (FK, contacts, `plant_sha256`), pressure zero / validity 0 → `output/edge_synthesis/motions/SYN_*.{motion,json}` |
| `admit.py` → `../admitted.json` | T5 | Every pre-registered machine check per variant; critic (E5) and PhysX reset launches recorded as **pending**, never assumed |
| `render_mj.py` | — | CPU (OSMesa) contact sheets of runs and references, for reading only |
| `render_video.py` | — | CPU (OSMesa + libx264) MP4s of runs. Two views, the T2 reference as a translucent ghost, the phase and the measured ground loads in the caption, and an optional slow motion. `--showcase` renders the results set and a montage into `output/edge_synthesis/videos/` |
| `tests/test_edge_synthesis.py` | — | 8 tests, 4 s: spec reproduction, FK, hinge-order round trip, branch continuity, box representative, friction, sketch anchoring, schedule masks |


## Where it stands (2026-10-04, 06:00 PDT)

**Two variants are provisionally admitted: the crow → handstand press at the slow timing, seeds 1 and 2.** They pass
every computed check of T5. Two checks are pending: E5's critic, and PhysX's launches at reset (the GPU is free since
G1 ended at 05:16 PDT on 4 Oct). They are in
`output/edge_synthesis/motions/SYN_E1_E1_E1_high_s{1,2}_refff.motion`, and all 21 gated variants are in
[`../admitted.json`](../admitted.json).

| Edge | T2 reference (statics-certified) | Best MuJoCo execution | Export projection | T5 (computed checks) |
|---|---|---|---|---|
| **E1** crow → handstand (press) | mid (2.95 s) and slow (4.5 s): s* ≤ 0.63, COM margin ≥ 1.5 cm | slow + feed-forward: **2.53 / 2.83 cm** mean body error (gate 3 cm), ends 0.041 / 0.037 m from the handstand, held 2 s, no saturation; **2 of 5 seeds** (seeds 0, 3, 4 fall in the final hold). Mid: 3.26 cm, held (1 of 1) | edits ≤ 1.2 cm | **slow seeds 1–2: provisional** (all computed checks pass). Mid: naturalness only (right knee 1.33× p99) |
| **E2** crow → chaturanga (jump-back) | — (dynamic) | lands at 0.87 / 0.84 BW, ends 0.019 m from chaturanga, held; **the hands shuffle 6 cm** at the push-off | edits ≤ 14.5 cm (airborne feet) | fails **naturalness only** (pelvis/knees 3× p99) |
| **E3** handstand → crow (lower) | mid and slow: s* ≤ 0.62, braces close to −0.1 cm | slow + feed-forward: 4.24 cm, ends 0.021 m, held; mid: 5.64 cm, 0.025 m | edits ≤ 1.6 cm | dynamics (right wrist saturated 9 %), naturalness (hips 1.5–1.6×) |
| **E4** tripod headstand → crow | mid (3.35 s): s* ≤ 0.86 after the pre-load rule; slow (5.5 s): s* ≤ 0.91, COM margin 3.3 cm | slow + feed-forward: 4.1 / 4.4 cm, lifts the head, ends **0.036 / 0.045 m** from crow, held (2 of 2 seeds); mid: 0.059 m | the execution unloads the head early and moves the weight onto the hands with momentum: on its own contacts the wrists reach s* 1.07–1.16 and some frames are infeasible | statics, naturalness |
| **E5** handstand → chaturanga (float-down) | — (dynamic) | lands **via the plank**: feet 2.9 BW, no head contact, ends 0.041 m from chaturanga, held | edits ≤ 11 cm | dynamics (right wrist 15 %), naturalness |
| B1 crow → plank (backup) | — (dynamic) | lands at 1.0 BW, ends **0.018 m** from the plank, held | edits ≤ 18 cm (airborne toes) | dynamics (toes 8–10 %, wrists 6 %), naturalness |
| B2 firefly → crow (backup) | s* ≤ 0.30 | **does not hold crow** (chest saturated on 52 % of steps) | — | — |

Every T5 row also carries two pending checks: the frozen critic (card E5 does not exist yet) and PhysX's launches at
reset (needs the GPU). The CPU part of the PhysX block passes everywhere: the stored bodies equal FK to ≤ 5e-7 m, and
the clip loads into a `MotionLib` whose plant is smpl_yogi_v2's.

## The decision this needs: T5's naturalness rule

The card pre-registers "per-body peak speed and jerk ≤ the corpus p99 for that body class". **Only the slow press
passes it, and for the dynamic edges none can.**

- The corpus has no jumps. Its per-body speed p99 is 0.8 m/s at the pelvis and 1.2–1.7 m/s at the knees (60 fps,
  x0 clips; `output/edge_synthesis/corpus_body_kinematics.json`).
- A jump-back moves the legs about 0.8 m while airborne. At a 0.43 s flight the knees reach 2.6–3× p99, and at 0.6 s
  still 2×. Both exceed even the corpus **maximum** at the knees (1.1–1.4×).
- Landing jerk is an impact. The rule exempts accelerations inside declared impact windows, but it has no such
  exemption for jerk.
- The quasi-static edges exceed p99 by 1.1–1.6× at the mid timing. **At the slow timing the press passes it**: its
  two admitted variants are under p99 on every body. The slow lower still reads 1.06× at the hips.

The options, which are the user's to choose:

1. **Keep the rule and drop the dynamic edges.** E2, E5 and B1 cannot pass it.
2. **Read it per body class (pooled), as the card's wording allows.** This is more lenient: pooling the limbs puts
   the feet's speeds into the knees' bound.
3. **Bound dynamic phases by the corpus maximum and exempt the impact windows from jerk.**
4. **Make speed advisory and let E5's frozen critic decide naturalness.** That is the critic's stated claim class.

`admit.py` reports the pre-registered reading as the decision. Beside it, it reports the speed against the corpus
maximum and the jerk outside impacts, so options 3 and 4 need no rerun.

## T0: the spec

- `edges.json`: 7 edges and the free edge. For each endpoint it holds the hold id, the exemplar frame, the window,
  the node and its key; the ground set and braces (with role, critical flag, statics necessity and LP load); the
  sidecar's known-free zones; the statics row; e15500's tracking; the hand anchor and COM.
- For each phase it holds the planted zones, closed braces, free zones, duration range and basis, and whether the
  phase is quasi-static. Events are derived from consecutive configurations, so no edit is ever implied without being
  listed.
- **Duration bases, measured** (`corpus_timing`):
  - x0 hands-anchored transitions with an inverted end take 0.38 / 1.92 / 2.77 s between holds (p10 / p50 / p90,
    19 transitions);
  - the human tripod → legs-on-arms → arm-balance passages (Koundinya -a/-b, Side Crow -b) take 0.35–2.05 s per step.
- Acceptance: the contact edits equal §1.3's table, and every hold id resolves in release v2. The test suite
  re-derives the file.
- Crow -a@439's exemplar is **frame 651**: labels v2 moved the exemplar and the id keeps 439.
- The free edge (Scorpion -b's hop) is recorded with its source hold only. Its destination is card R1's to label.

## T1: plant v2 in MuJoCo

| Check (the card's) | Result |
|---|---|
| FK ≤ 1e-6 m against pose_lib, 100 release frames | **1.9e-8 m** (46 clips); exp-map round trip 1e-15 |
| Throughput ≥ 20k control steps/s at 256 × 30, 22 threads | **missed**: 4.9k on the 8 E-cores (the CPU policy beside G1); 11.5k at 16 threads on all cores. The gate came from the benchmark's dt 1/120 (39k there, but 35 % of noisy rollouts blew up) |
| MPPI hold: crow and handstand 5 s, 3 of 3 seeds, training gains | **3/3 and 3/3**, at 1.2–1.8 s of planning per 0.1 s of motion on the E-cores |
| Slide test μ 0.75 | **pass** with elliptic cones: a plant box sticks at tan 0.74 (≤ 0.2 cm/s creep) and slides at 0.76, at 0°, 45° and 90° |
| Calibration replay of PhysX PD targets (informative) | waits for G0(b) |

Measured departures from the card:

1. **Elliptic cones.** MuJoCo's default pyramid is a diamond. The same box slides at tan 0.70 on the diagonal
   (μ 0.53), while PhysX's patch friction is isotropic. Elliptic costs about 37 % of throughput.
2. **Per-body hinge orders.** An XYZ-type chain is singular where its middle angle reaches ±90°. The MJCF's x-y-z
   order puts y in that slot, and y is the largest range of the hips (−140..60°), knees (−5..160°), ankles, toes and
   spine. Every crow ↔ handstand edge folds a knee through 90°.
   - `HINGE_ORDER` puts each body's smallest-range axis in the middle (hips `yzx`, knees `yxz`, elbows `zxy`, …). That
     axis stays below 70° in the box, except at the shoulders and hands, whose three ranges all span ≥ 180°.
   - With the MJCF order the handstand hold failed 3 of 5 seeds; with `HINGE_ORDER` it passed 3 of 3.
   - A knee sweep through 90° with 3° of abduction now moves each hinge by at most the sweep step.
3. **Hinge limits stay off, and the box cannot become hinge limits.** Per-body Euler angles differ from the exp-map
   components by tens of degrees at the elbows, wrists and hips. 14,505 of 81,512 corpus frames would hit a false
   limit. The box is a cost (`box_excess`, which takes each joint's better representative) and an export projection.
4. **PhysX's hard stops carry load that free hinges cannot.** The first jump-back drooped a wrist 16° past the box
   under load. Saturation shares measured in MuJoCo therefore overstate PhysX's wherever a joint would rest on its stop.

## T2: the quasi-static generator

**How a reference is made.**
- **Drafts.** Keyframes are hybrids of the endpoints. The press's tuck is the handstand's arms, shoulders and trunk
  with the crow's folded legs; the tripod's tuck is the tripod's upper body with the crow's legs. Each draft is
  translated so the hands stay where S has them.
- **Anchor.** The drafts are joined by minimum-jerk slerp on the edge's phase clock at 60 fps.
- **Solve.** `retarget_v2`'s objective is used unchanged, with every human-evidence input replaced by the schedule:
  - resting faces on the floor, palms flat;
  - known-free zones off the floor;
  - no slide;
  - braces closed, or kept 2 cm open after a break;
  - balance on every frame;
  - the 253-pair guard and the hard box.
- **Certification.** Statics runs on every 3rd frame with that frame's contact set, plus the card's contract.
- Each solve takes 50–90 s single-threaded.

| Reference | T | Statics: s* max (frames checked) | COM margin min | Floor min | New overlaps > 1 mm | Box frames > 1° | Certified |
|---|---|---|---|---|---|---|---|
| E1 mid | 2.95 s | 0.63 (76) | 1.59 cm | +0.31 cm | 0 | 0 | yes |
| E1 slow | 4.50 s | 0.61 (107) | 1.54 cm | +0.31 cm | 0 | 0 | yes |
| E3 mid | 2.65 s | 0.61 (70) | 1.58 cm | +0.31 cm | 0 | 0 | yes; braces −0.1 cm |
| E3 slow | 4.00 s | 0.62 (97) | 1.52 cm | +0.31 cm | 0 | 0 | yes; braces −0.1 cm |
| E4 | 3.35 s | 0.86 (84) | 3.19 cm | +0.17 cm | 0 | 0 | yes; braces −0.08 cm |
| B2 | 1.90 s | 0.30 (55) | 1.58 cm | +0.31 cm | 0 | 0 | yes |

**The press is within the plant.** Its statics peaks at s* 0.60 at the chest mid-pike and 0.62 at the wrists in the
handstand. §1.3's open risk, wrist and shoulder torque mid-pike beyond the plant, did not materialise.

**What each rule cost to find:**
- **New overlaps must exclude the braces the schedule closes.** The first check counted the crow's closing shin
  brace, resting at the guard's −1 mm floor, as a new overlap.
- **A breaking support must be pre-loaded.** E4's head lifted with the COM inside the hands' hull (1.2 cm margin)
  and the left wrist still bound: s* 1.10 at the break frame. Being inside the hull is necessary, not sufficient.
  - The rule: for 0.5 s before a break, the breaking zone leaves the balance hull and the COM sits at least 4 cm
    inside the rest. That gives s* 0.86.
  - It reaches `retarget_v2._balance` through an in-process wrapper. `retarget_v2.py` is not edited, because its
    sha is part of the retarget record.
- **The penetration guard is 10× stiffer and aims at 0.8 mm.** At retarget_v2's weight E4's folded heel sat 2.3 mm
  in the thigh; at the 1 mm floor the solver stopped at 1.1 mm.

**The MuJoCo execution check** (`edge_mppi --reference … --feedforward`; the card's gate is ≤ 3 cm mean body error
and the end held 2 s):

| Reference | Mean / p95 body error | Final 6-body error to D | Held 2 s | Notes |
|---|---|---|---|---|
| E1 mid | **3.26 / 6.8 cm** | 0.037 m | yes | no saturation; hands drift 1.2 cm; reproduced bit for bit |
| E1 slow, seeds 1 / 2 | **2.83 / 2.53 cm** | 0.037 / 0.041 m | yes | **within the card's gate**; no saturation; box ≤ 0.4° |
| E1 slow, seeds 0 / 3 / 4 | 3.45 / 2.94 / 4.60 cm | 0.86–0.89 m | **no**: each falls during the final handstand hold | the hold is the weak point, not the press |
| E3 mid | 5.64 cm | 0.025 m | yes | right wrist saturated 9 % |
| E3 slow | 4.24 cm | 0.021 m | yes | right wrist saturated 9 % |
| E4 mid | 4.16 cm | 0.059 m | yes | the head lifts; neck/head saturate 2–4 % |
| E4 slow, seeds 0 / 1 | 4.11 / 4.42 cm | 0.036 / 0.045 m | yes | head/neck saturate 2–5 % |
| B2 | 4.15 cm | 0.19 m | **no** | chest saturated 52 %: the crow end is not held |

Without the feed-forward, the press lagged from the start: the pelvis was 13 cm low at t = 0.8 s. Then its COM
drifted 8 → 97 cm from the hands and it toppled (29 cm mean error). The lower (54 cm) and the tripod → crow (the head
never lifted) failed the same way.

## T3: sampling MPC and the cost library

The core is the setting the benchmark measured: 4 knots over 0.8 s, N 192–256, 2 iterations per replan, λ 0.05,
replan every 0.1 s, a shifted warm start. Execution goes through the same rollout function, so an executed motion is
exactly a planned one. Runs are deterministic: the press reproduced bit for bit.

What each term or rule fixed, in the order the failures appeared:

| Failure | Fix | Effect |
|---|---|---|
| A plain slerp sketch drags the planted hands across the floor | the sketch solves the root each frame so the hand midpoint stays on S's | drift 0.00 mm along the sketch |
| B1's feet landed 41 cm short: the shoulders ended 40 cm ahead of the hands and the chest saturated holding the lean | landing placement (making zones land where D has them) and 2.5× stronger tracking | final 6-body error 0.172 → **0.018 m** |
| Slip read from a zone's lowest point flickered as the box rocked between corners | slip from the zone's body origins | the spurious slip cost disappeared |
| E2's palms rocked onto two corners and shuffled 5 cm, inside the friction cone | palm flatness and a 4× hand anchor | drift 6.1 → 5.2 cm (still open) |
| E5's float-down hit the floor face first (head 6.1 BW) | touch down via the plank (`--via`), then bend into chaturanga; head/trunk clearance 6 cm | no head contact; feet 2.9 BW |
| Quasi-static references toppled | statue feed-forward and a capture point | press 29 → **3.3 cm** |

**T3's acceptance:**

| Check | Result |
|---|---|
| Crow and handstand held 5 s, 3 of 3 seeds | **3/3 and 3/3** |
| T2's press reference executed within 3 cm mean body error | **3.26 cm**: a marginal miss, with the handstand held |
| B1: peak vertical force ≤ 3 BW, final 6-body error ≤ 0.05 m | **1.02 BW, 0.018 m** |

**Feed-forward helps the quasi-static edges and hurts the jump-backs.** With it B1/E2 ended 0.059/0.053 m from D;
without it, 0.018/0.019 m. A statue along a ballistic sketch is not a static pose. Use `--feedforward` for
`--reference` runs and E5, and not for B1/E2.

## T4: variants and the export

The export (`export_motion.py`) works in three steps:
1. It reads the run's 60 fps frames. Each joint's exp-map representative is chosen against the box.
2. T2's solver **projects** the motion into PhysX's box. The anchor is the executed motion itself, under its own
   executed contacts: a zone is planted where the floor carries more than 5 N and its lowest point is within 2 cm.
   Before the departure, S's scheduled configuration is used instead. Landing zones get a ±0.2 s exemption from the
   keep-off rows.
3. `fit_writer.write_motion` writes the clip.

| Run | Box excess before | Projection edits: p50 / max | After: box, floor, overlaps, statics |
|---|---|---|---|
| E1 mid (press) | 1.4° | 0.4 / 1.2 cm | clean; s* ≤ 0.55 |
| E3 mid (lower) | 2.7° | 0.4 / 1.6 cm | clean; s* ≤ 0.67 |
| E2 (jump-back) | 7.1° | 0.9 / 14.5 cm | clean; s* ≤ 0.28 |
| B1 (jump-back) | 16.2° | 0.9 / 18.4 cm | clean; s* ≤ 0.18 |
| E5 (float-down) | 11.7° | 1.2 / 11.1 cm | clean; s* ≤ 0.33 |
| E4 (tripod → crow) | 0.4° | 0.7 / 21.0 cm | **statics fails**: the executed head-lift is dynamic (s* 1.17 on 10 frames; slow: 1.07–1.16 and infeasible frames) |
| E1 slow, seeds 1 / 2 (press) | ≤ 0.4° | ≤ 1 cm | clean |

- Every export round-trips to ≤ 5e-7 m and carries plant v2's `plant_sha256`. Pressure is zero with validity 0.
- The large edits are all at airborne feet, where a jump-back's hips and knees sat 8–16° past the box.
- **Variants:** timings (`low`/`mid`/`high`) and MPPI seeds. Starts from G0(a)'s arrival banks wait for the GPU;
  every run so far starts from S's exemplar.

## T5: the admission gate

`admit.py` computes each pre-registered group per exported variant:
- **contract**: box, floor, new overlaps, and accelerations outside ±0.1 s impact windows;
- **statics**: on quasi-static frames;
- **dynamics**: on the MuJoCo execution, saturation share per joint and the friction cone per zone;
- **naturalness**: per-body speed and jerk against the corpus p99;
- **endpoints**;
- **the CPU half of PhysX**: the round trip, and `MotionLib` on plant v2.

The critic and the reset launches are listed as pending.

A variant is `provisional` when every computed group passes. Of 21 exported variants, the two slow-press seeds are;
the table at the top gives the others' reasons.

**Variants (T4) and seeds.** MPPI's outcomes vary by seed, so a variant is a (timing, seed) pair:
- slow press: 2 of 5 seeds held;
- jump-backs: 3 of 3 seeds held for B1 and for E2;
- float-down: 3 of 3 held, one ending at 0.055 m;
- slow tripod → crow: 2 of 2 held.

Two configurations failed outright and are recorded as negatives:
- a 10× torque cost (`torque` 2.0) made the slow lower fall (44–51 cm);
- B2 (firefly → crow) never holds crow, because its chest saturates.

Two facts for whoever runs G2 next:
- **The dynamics gate counts saturation that PhysX would not have.** E3's and E5's wrists, and B1's toes, saturate
  in MuJoCo, where nothing else carries the load at a box edge. In PhysX the hard stop takes part of it.
- **E2 passes every computed group except naturalness.** E1 does too.

## Running beside a GPU job

- **CPU only.** Every process ran with `CUDA_VISIBLE_DEVICES=""`. Never import `reference_curation.plant_v2_physx`
  from this package: it launches IsaacLab at import.
- **G1 was never stopped, restarted or touched.** Its GPU memory stayed at 23.6 GB throughout.
- **Measured cost to G1's epoch time.** G1's own time drifted from 9.72 s (epochs 1,700–2,200) to ~10.1 s from
  20:46 PDT, for reasons outside this lane. With every lane-T job paused (23:12–23:28, 85 epochs) it ran at **10.12 s**.
  Against that concurrent baseline:
  - 16–20 unpinned rollout threads: 13.0–13.8 s;
  - the 8 E-cores at nice 19: 10.45–10.51 s (+3.5 %);
  - **one 4-core E-core cluster (CPUs 16–19) at nice 19: 10.14 s (+0.2 %)**. Every CLI here has pinned itself so since
    23:00 PDT (`gpu_guard.be_polite()`); MPPI runs about twice as slowly.
  - Compare against a window with your own jobs paused (`kill -STOP` by PID), not against an older stretch: the drift
    alone is +4 %.
- Check the effect with `python -m edge_synthesis.gpu_guard`, which reads the run's own `times/last_epoch_seconds`.
- **Whole-run accounting** (after G1 ended, 47 min after its launch estimate):
  - The evaluations ran long: +33 min.
  - About 10 min of regular-epoch time fell between 20:40 and 23:12, when this lane ran unpinned and then on all 8
    E-cores: median 10.39 s, against 9.74 s before. The two evaluations in that window ran 1–2 min long.
  - Under the 4-core policy afterwards, the median was 9.87 s.

## How to run

```bash
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:data/scripts
python -m edge_synthesis.edge_spec [--check]                       # T0
python -m edge_synthesis.plant_mj --acceptance                     # T1 (writes t1_plant_mj.json; ~10 min)
python -m edge_synthesis.quasistatic --edge E1 [--timing high]     # T2 reference (~1 min)
python -m edge_synthesis.edge_mppi --edge E1 --reference output/edge_synthesis/quasistatic/E1_mid \
    --feedforward --samples 256 --noise 0.05 --weights '{"track": 20, "track_end": 20}'   # T2's dynamic check
python -m edge_synthesis.edge_mppi --edge B1 --keyframes land --samples 256 --noise 0.08 # a jump-back (T3)
python -m edge_synthesis.edge_mppi --edge E5 --keyframes land --via 220923_Plank_Pose_or_Kumbhakasana_-a@541 \
    --feedforward --samples 256 --noise 0.08                                              # the float-down
python -m edge_synthesis.export_motion output/edge_synthesis/runs/<run>                   # T4 (~1 min)
python -m edge_synthesis.admit output/edge_synthesis/motions/SYN_*.json                   # T5
python -m edge_synthesis.render_mj output/edge_synthesis/runs/<run>                       # a contact sheet
python -m pytest data/scripts/edge_synthesis/tests -q                                     # 8 tests, 4 s
```

## Open, in order

1. **The naturalness rule** (above): a user decision. The press needs no change; it passes at the slow timing.
2. **G0(a)'s arrival banks**, the GPU's next job (G1 has ended). T4's variants should start from the expert's own states
   at S, converted to MuJoCo; that closes the hand-off gap.
3. **T5's pending checks.** PhysX launches at reset need the GPU, which is free now. The frozen critic is card E5, CPU-only and not
   started; its feature function is E2's.
4. **The handstand hold at the end of the press** fails 3 of 5 seeds. **E2's hand shuffle** is 5–7 cm at the
   push-off. **E4's head-lift** is executed with momentum, which statics cannot certify.
   - A better balance objective for hands-only holds (the benchmark's COM-over-hands term worked 3 of 3) is the first
     thing to try.
5. **Card R2** (splicing an admitted edge into a clip with real lead-in and lead-out) feeds G2.
6. **The MuJoCo ↔ PhysX sim-to-sim gap is deferred to a later step** (the user, 2026-10-04).
   - Until then, these edges are MuJoCo results, and G2's PhysX tracking decides what enters a release.
   - The measured differences are listed in PLAN.MD §1.3, "Sim-to-sim".
