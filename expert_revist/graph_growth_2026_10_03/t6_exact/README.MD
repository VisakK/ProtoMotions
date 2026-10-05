# Card T6: endpoint-exact references, re-run on the PhysX generator (2026-10-04)

**Why.** PLAN.MD §1.4 ("What the Step 3 build must handle") measured that the D1 variants did not meet the real clips
at their ends: on the press (E1) and the lower (E3) the hands sat 3.7–5.8 cm off both exemplars, and the landings
(B1, E2, E5) put the feet 5–14 cm from D's. R2 cannot splice such a variant without moving a loaded hand or foot. T6
rebuilds the references with exact endpoints, re-runs the D1 recipes in PhysX, and admits the result with a new
seam group.

Nothing here touched release v2, the pre-T6 runs or motions, or any training code. The pre-T6 admission and
selection are kept byte for byte as [`../admitted_physx_pre_t6.json`](../admitted_physx_pre_t6.json) and
[`../selected_physx_pre_t6.json`](../selected_physx_pre_t6.json).


## Where it stands

- **The press and the lower are fixed, with two residuals.** 20 re-runs (E1 slow and mid, E3 noise 0.05 and 0.12;
  5 seeds each):
  - every variant starts at S's exemplar: every body within 0.48 cm, the hands within 0.29 cm (before T6:
    4.4–5.8 cm);
  - the knuckle bodies (`L_Hand`, `R_Hand`) end within 0.33–0.96 cm of D's on every held variant (before:
    3.7–5.6 cm; the one E1 mid seed that falls ends 2.8 cm off);
  - the press got better, not worse: E1 slow ends 0.030–0.034 m (6-body) from the handstand on 5/5 seeds (before:
    0.038–0.049 m, 4/4); E1 mid 0.016–0.032 m on 4/5.
  - **Residual 1, the palms (the exemplars' own).** The human placed her palms about 12° differently in crow and in
    the handstand, a pure yaw on the floor. A variant that keeps S's palms (the card's rule) therefore ends with the
    wrist bodies 2.8–3.2 cm from D's, the exemplars' own 2.7–2.9 cm. No rigid placement of D removes it (below).
  - **Residual 2, palm rocking in PhysX.** On the held variants the planted palms rock 0.7–1.3 cm (the wrist body's
    origin) during the lift and come back to 0.03–0.4 cm by the end (traced on E1 mid s0 and s2). The card's drift gate is 0.5 cm. Its "today ≤ 0.4 cm" was the T2
    *reference's* drift; the executed PhysX hands drifted 0.4–3.4 cm before T6 too (`seams.legacy` on the pre-T6
    selection).
- **The landings are not fixed, and the cause is the plant.** The landing sketch drove the feet through the floor;
  that is fixed in the reference, which now touches down within 0.02 cm of D's feet. Executed in PhysX, every one
  of 70 landing runs (two cost settings) still touches down 5–42 cm from D's feet (B1 17–42, E2 5–17, E5 14–25) and
  slides the loaded feet 2–29 cm toward D. The cause is the arms. During B1's flight the right elbow's PD demand is
  2.0–2.1× its 100 N·m limit, so the shoulders move about 6 cm *forward* instead of 17 cm back over the hands as the
  reference does. With the shoulders forward, the extending legs come down short: 10–14 cm in the run traced below.
  This is T7's weak right elbow (`t_edges_physx/README.MD`), and it is the training plant's.
- **Admission.** All 90 re-run variants pass the PhysX reset check: 0 of 1,800 resets launch, FK within
  1.06e-6 m, plant v1 refused. Under the card's own thresholds no variant passes the seam group.
- **The user decided the three questions this raised (2026-10-04; "Decided by the user" below):**
  - drift is read at 1.0 cm;
  - the palms' end offset is accepted ("the policy should learn around it");
  - the landings' slide is accepted ("the key goal is can the policy even pick up these dynamic edges").
- **[`../selected_physx.json`](../selected_physx.json) holds 29 variants of all five edges:** E1 ×8, E3 ×3, B1 ×5,
  E2 ×12, E5 ×1. These are R2's input.


## Files

| File (`data/scripts/edge_synthesis/`) | What |
|---|---|
| `exact.py` | The endpoint-exact references (below), written in `quasistatic`'s layout to `output/edge_synthesis/exact/<edge>_<timing>/`. `--d1` builds every D1 (edge, timing) |
| `seams.py` | T6's seam group (`check`), and `evidence/seam_offsets.py`'s measure moved here (`legacy`) |
| `costs.py`, `costs_torch.py` | `EdgeWeights.cone`: the landing cone as an MPPI cost, the card's 1.5 cm tolerance (weight 0 unless a recipe sets it, so every earlier recipe is unchanged) |
| `quasistatic.py` | One hook: a problem may carry `plan["balance_w"]` (the balance hull's points when they differ from the support rows'). Problems without it are unchanged: E4 slow, the one T2 reference whose record carries provenance, regenerates bit for bit. The E1/E3 references the pre-T6 variants tracked predate that block and lane T's later rules, so they do not regenerate with lane T's committed code (E1 mid differs by up to 2.1 cm). That gap predates T6, and T6 replaces those references |
| `admit.py` | The `seams` group on every variant (literal and `pass_attainable`), `--physx-launch-check` (fills the reset launches from `retarget_v2_physx.py`'s record of the same files, sha256-checked), `--policy d1` (the D1 verdict per variant and the policy itself in the record) |
| `select_variants.py` | `policy()` / `verdict()`: the D1 rule as data, used by `admit`; the seam group and the reset check are hard where a record carries them; `--seam-end strict\|attainable`, `--drift-cm` (re-reads the recorded drift; the default is the card's 0.5 cm) |
| `t6_report.py` | Each T6 variant against the pre-T6 variant of the same (edge, timing, seed, pass) → [`t6_report.json`](t6_report.json) |
| `run_t6.sh` | The whole card: `refs` → `mppi` (11 recipes × 5 seeds) → `export` → `check` → `admit` → `select` |
| `export_physx_batch.sh` | `RUNS`, `OUT` and `SKIP_EXISTING` (the defaults are unchanged) |
| `tests/test_t6.py` | 3 tests, 1 s: the pin and cone rows against central differences, the frozen frames, the seam group on a clip that holds S's exemplar; `test_physx_port.py` checks the cone term torch against numpy |

Outputs: the references in `output/edge_synthesis/exact/`; runs in `output/edge_synthesis/physx/runs_t6/` (the T6
recipe) and `runs_t6_nocone/` (the landing recipes as recorded, on the new reference); all 90 motions in
`output/edge_synthesis/physx/motions_t6/`; the reset check `output/edge_synthesis/physx/physx_launch_check_t6.json`;
the probes behind the recipe choices in `output/edge_synthesis/physx/runs_t6test/`; logs in `.../logs_t6*/`.


## The references (`exact.py`)

Built with T2's solver (`retarget_v2`'s LM; the box a hard bound; the 253-pair guard; the schedule's support, off,
no-slide, brace and balance rows), plus four rules:

1. **Frozen ends.** Every frame up to the departure *is* S's exemplar, and every frame from the arrival on *is* D'.
   Their variables are bounds-pinned (lo = hi), through an in-process wrapper of `retarget_v2.bounds`.
   `retarget_v2.py` is not edited: its sha is part of the retarget record.
2. **Pinned hands.** `L_Wrist`, `L_Hand`, `R_Wrist` and `R_Hand` keep S's exemplar pose, position and orientation,
   on every frame. These are new LM rows: 0.5 mm and 0.5° scales, weight 10.
   - The pinned zones drop out of the `support` and `flat` rows, as the card asks, but stay in the balance hull.
   - A landing's feet are pinned from touchdown on, ramped in over 0.1 s: to D's own pose on B1 and E2; on E5 the
     toes only, so the feet can roll from the plank into chaturanga.
3. **D'.** D's exemplar, re-anchored by `sketch.hand_anchor_transform` as before, is IK-edited by a static solve of
   the same objective, so that its hands are S's (and on a landing its feet are D's own). Statics runs on D'.
   - E5's touchdown keyframe (the plank) is edited onto S's hands and D's toes, so it lands where chaturanga keeps
     its feet.
4. **The landing cone.** Before touchdown, every foot collider point that touches the floor at the touchdown
   keyframe stays at least 0.25 × its horizontal distance from its own spot above that spot's height.

| Reference | Frames | Pins (worst) | Start vs S | End, planted bodies vs D | Statics s* max | COM margin | Touchdown: points vs spot; feet height 0.2 s before |
|---|---|---|---|---|---|---|---|
| E1 slow (4.5 s) | 319 | 0.023 mm, 0.01° | 0.0 cm | 2.91 cm (the palms) | 0.594 | 1.46 cm | — |
| E1 mid (2.95 s) | 226 | 0.022 mm, 0.01° | 0.0 cm | 2.91 cm | 0.602 | 1.54 cm | — |
| E3 slow (4.0 s) | 289 | 0.008 mm, 0.006° | 0.0 cm | 2.91 cm; braces −0.08 cm | 0.602 | 1.50 cm | — |
| B1 high (1.6 s) | 145 | 0.024 mm, 0.022° | 0.0 cm | 2.23 cm (the palms) | 0.165 | 1.81 cm | 0.01 cm at 0.617 s (0.6); ≥ 2.7 cm |
| B1 mid (1.075 s) | 113 | 0.021 mm, 0.018° | 0.0 cm | 2.23 cm | 0.165 | 1.81 cm | 0.01 cm at 0.433 s (0.425); ≥ 6.3 cm |
| E2 mid (1.075 s) | 113 | 0.023 mm, 0.033° | 0.0 cm | 3.01 cm (the palms) | 0.260 | 1.81 cm | 0.02 cm at 0.433 s (0.425); ≥ 6.5 cm |
| E5 mid, via the plank (1.65 s) | 148 | 0.016 mm, 0.015° | 0.0 cm | 0.48 cm | 0.322 | 2.07 cm | 0.5 cm at 1.017 s (1.0); ≥ 17 cm |

- Every reference certifies (`quasistatic.certify`): box 0 frames, floor ≥ +0.31 cm, no new overlaps, no jerk frames.
- **The box never binds a pin**: 0 frames beyond 1 mm / 1°, so there are no infeasible frames to report. The 2.2–3.0 cm
  end offsets are the exemplars' palms (next section), not the solver.
- D' moves D by 2.9 cm (E1, E3), 2.2 cm (B1), 3.0 cm (E2) and 0.6 cm (E5): the palms again. D' certifies, at s*
  0.32 / 0.23 / 0.11 / 0.26 / 0.26. E5's plank touchdown keyframe moves 5.0 cm (s* 0.11): her plank's feet sit 5.7 cm
  ahead of her chaturanga's.
- Each reference solves in 17–67 s on one E-core.


## Three findings

**1. The exemplars place the palms differently.** After the hand-anchor transform, the knuckle bodies of S's and D's
exemplars meet within 0.4–1.0 cm. But each palm is turned about the vertical: crow vs the handstand
−12.9° (left) and +12.0° (right); crow vs the plank −11.4° / +14.3°; crow vs chaturanga −8.0° / +11.0°; the handstand
vs chaturanga +4.8° / −0.9°. The rotation axis is vertical to within 1 % in every case.
- So the wrist origins sit 1.8–3.0 cm apart (E1/E3 2.9 / 2.7, B1 1.8 / 2.2, E2 2.5 / 3.0, E5 0.5 / 0.3).
- No rigid placement of D removes a mirror-symmetric yaw of the two palms. A four-body planar Procrustes leaves
  2.8 / 2.7 cm on E1.
- A loaded palm cannot make that turn without its heel sliding about 3 cm.
- So T6's rule (the hands keep S's pose) and its end seam (planted bodies within 1.5 cm of D's) cannot both hold on
  E1, E3, B1 or E2. `seams.check` reports the end literally, and also `pass_attainable`, which passes a body whose
  offset is exactly that (the variant within 0.5 cm of S's pose there, and S's exemplar beyond 1.5 cm from D's).

**2. The landing sketch went through the floor.** The `--keyframes land` sketch is a minimum-jerk slerp from the
crow's tuck to D. It drives the feet 4–7 cm below the floor while they are still 41–47 cm short of D's feet (B1 at
0.27–0.33 s, E2 at 0.18–0.27 s). Every pre-T6 landing touched down short: 20–68 cm along the hands-to-feet axis on
B1, 2–11 cm on E2, 13–24 cm on E5. It then dragged the loaded feet back for about a second. E5's sketch also lifted
the feet 5 cm off the plank on its way into chaturanga. The exact references fix the kinematics: the feet stay
≥ 2.7–6.3 cm up until 0.2 s before landing and touch down within 0.02 cm of their spots.

**3. In PhysX the arms cannot push the body back, so the feet still land short.** B1's executions sit 11–14 cm
(mean body error) from their reference. One B1 run traced against its reference
(`runs_t6test/B1_B1_high_s1_t6slip`):

| t (s) | Reference: shoulders / pelvis, along the hands→feet axis | Executed | Right elbow \|PD torque\| / limit |
|---|---|---|---|
| 0.0 | −11.8 / 19.4 cm | −11.2 / 20.6 cm | 1.98 |
| 0.2 | −9.1 / 27.4 cm | −9.3 / 19.7 cm | 2.09 |
| 0.4 | +0.5 / 43.0 cm | −16.4 / 18.1 cm | 1.43 |
| 0.6 | +5.1 / 46.4 cm | −13.7 / 26.2 cm | 0.75 |
| 0.75 (touchdown) | +5.2 / 46.5 cm | −1.7 / 34.4 cm | 0.23 |

- The elbow saturates for the first 0.3 s, and the shoulders move about 6 cm forward instead of 17 cm back.
- The legs extend about a fixed shoulder, so the feet land short. MPPI then slides them onto D's spot, because
  every term that prices the end pose prefers D's spot.
- A position cost (the cone) cannot stop that: once a foot is down short, sliding toward the spot *lowers* the cone
  cost. Pricing the slide (`slip` ×10) cuts it to 3–15 cm but leaves the feet 1.4–9 cm from D at the end.

**The probes behind the recipe choices** (`runs_t6test/`, 5 seeds each, PhysX):

| Probe | Touchdown short of D | Slide after touchdown | End: feet vs D | Final 6-body | Hands drift (wrist, run metric) |
|---|---|---|---|---|---|
| B1 mid, exact reference, recorded weights | 15–26 cm | 10–20 cm | 0.2–4.1 cm | 0.044–0.070 m | 0.75–1.15 cm |
| + cone 10 | 12–22 cm | 7–14 cm | 0.6–5.0 cm | 0.064–0.097 m | 0.65–0.93 cm |
| + cone 10, flat 100, slip 20 (mid / high) | 9.5–26 cm | 3–15 cm | 1.4–9.1 cm | 0.043–0.083 m | 0.44–1.06 cm |
| E1 mid, recorded weights | — | — | — | 0.016–0.032 m (4/5 held) | 0.39–0.53 cm |
| E1 mid, anchor 300, flat 100 | — | — | — | 0.032–0.22 m (4/5 held) | 0.28–0.43 cm |
| E2 mid, cone 10, flat 100 (noise 0.12) | 3–10 cm | 0.4–3.7 cm | 1.4–6.3 cm | 0.078–0.092 m | right hand 2.0–2.6 cm |

So the T6 recipe:
- keeps the recorded weights for E1 and E3. A 10× hand anchor halves the drift but costs the press its endpoint;
- adds the card's landing aim (cone 10) to B1 and E5;
- runs E2 with cone 10 and flat 100 at the default anchor. The card keeps the hands planted, and under E2's lane-T
  override (anchor 8, flat 0) the pre-T6 E2 hands slid 1.8–3.4 cm.

The landing recipes were also run exactly as recorded, on the new reference only (`t6rpx`, `runs_t6_nocone/`), to
separate the reference's effect from the cone's.


## The re-execution (`edge_mppi_physx --reference output/edge_synthesis/exact/... --start exemplar`)

11 recipes × 5 seeds: the D1 selection's (edge, timing, noise), 512 samples, 16 start iterations, the T7 settings.
Plus the 7 landing recipes as recorded, on the new reference (`reference only`). 90 variants in all; one IsaacLab
process per recipe, two at a time, 3–6 min each. From [`t6_report.json`](t6_report.json); "before" is the pre-T6
variant of the same (edge, timing, seed, pass) (seed 4 is new).

| Recipe | Held | Final 6-body (m) | before T6 | Hands start / end (cm, `L_Hand`/`R_Hand`) | before | Feet end (cm) | before | Seam drift (cm) |
|---|---|---|---|---|---|---|---|---|
| E1 slow, noise 0.05 | 5/5 | 0.030–0.034 | 0.038–0.049 | 0.13–0.18 / 0.44–0.75 | 5.7–5.8 / 5.0–5.2 | — | — | 0.81–1.09 |
| E1 mid, 0.05 | 4/5 | 0.016–0.032 (one fall, 0.52) | 0.033–0.042 | 0.13–0.18 / 0.64–2.84 | 4.4 / 3.7–5.4 | — | — | 0.81–1.48 |
| E3 slow, 0.05 | 5/5 | 0.045–0.051 | 0.049 (s2) | 0.17–0.24 / 0.39–0.56 | 5.1–5.2 / 5.4–5.6 | — | — | 0.80–1.22 |
| E3 slow, 0.12 | 5/5 | 0.038–0.054 | 0.046–0.049 | 0.17–0.28 / 0.33–0.96 | 4.9–5.2 / 5.2–5.6 | — | — | 0.72–1.28 |
| B1 high, 0.08 + cone | 1/5 | 0.074–0.124 | 0.029–0.049 | 0.14–0.20 / 0.27–0.54 | 0.1–0.3 / 0.2–0.7 | 2.5–3.7 | 7.3–14.2 | 10.1–18.4 |
| B1 high, 0.08, reference only | 5/5 | 0.027–0.074 | 0.029–0.049 | 0.09–0.22 / 0.42–0.98 | | 1.4–23.3 | 7.3–14.2 | 22.5–28.5 |
| B1 high, 0.12 + cone | 5/5 | 0.069–0.085 | 0.041–0.074 | 0.14–0.19 / 0.38–0.76 | | 4.9–9.1 | 11.0–22.7 | 6.5–9.4 |
| B1 high, 0.12, reference only | 5/5 | 0.044–0.060 | 0.041–0.074 | | | 9.4–21.6 | 11.0–22.7 | 15.4–18.5 |
| B1 mid, 0.08 + cone | 2/5 | 0.064–0.097 | 0.038–0.068 | 0.21–0.32 / 0.18–0.45 | | 1.7–5.1 | 1.8–5.7 | 10.9–14.5 |
| B1 mid, 0.08, reference only | 5/5 | 0.044–0.070 | 0.038–0.068 | | | 1.7–4.1 | 1.8–5.7 | 13.2–20.8 |
| B1 mid, 0.12 + cone | 5/5 | 0.057–0.081 | 0.042–0.065 | | | 5.3–7.7 | 6.2–12.1 | 7.7–10.3 |
| B1 mid, 0.12, reference only | 5/5 | 0.034–0.065 | 0.042–0.065 | | | 5.8–10.1 | 6.2–12.1 | 8.4–14.1 |
| E2 mid, 0.08 + cone, flat 100 | 3/5 | 0.077–0.106 | 0.087–0.152 | 0.20–0.39 / 1.7–2.4 | 0.2–0.4 / 2.7–4.2 | 2.8–5.7 | 5.9–10.0 | 2.9–4.2 |
| E2 mid, 0.08, reference only | 3/5 | 0.077–0.147 | 0.087–0.152 | | | 3.4–7.5 | 5.9–10.0 | 3.4–8.2 |
| E2 mid, 0.12 + cone, flat 100 | 5/5 | 0.078–0.092 | 0.075–0.112 | | | 3.2–6.3 | 3.1–5.2 | 2.3–3.7 |
| E2 mid, 0.12, reference only | 5/5 | 0.079–0.111 | 0.075–0.112 | | | 3.1–7.0 | 3.1–5.2 | 3.0–5.3 |
| E5 mid, 0.12 + cone | 0/5 | 0.090–0.122 | 0.081–0.092 | 0.22–0.30 / 0.76–1.26 | 0.2–0.3 / 0.5–1.7 | 3.1–5.9 | 5.3–8.3 | 8.8–13.6 |
| E5 mid, 0.12, reference only | 1/5 | 0.077–0.098 | 0.081–0.092 | | | 4.1–9.6 | 5.3–8.3 | 12.0–19.7 |

- "Seam drift" is the seam group's measure: every body of a planted zone, horizontally, over the whole variant (for
  a landing foot, from 0.1 s after its touchdown). On the landings it is the feet's slide.
- Over all 70 landing variants (verified on the records): with the cone the feet touch down 17–25 cm (B1), 7–11 cm
  (E2) and 14–19 cm (E5) from D's and slide 6–18, 2–4 and 9–13 cm. Reference only: 19–42, 5–17 and 19–25 cm, sliding
  8–29, 3–8 and 12–16 cm.
- The cone does what the card asks of the *landing* where the plant allows it: the feet end nearer D's (B1 high,
  noise 0.08: 2.5–3.7 cm against 7.3–14.2 before; E2, noise 0.08: 2.8–5.7 against 5.9–10.0). But because the feet
  still land short, the slide remains. On B1 it costs the endpoint: the final 6-body error rises to 0.057–0.124 m.
- Reference only, B1 keeps its endpoints (0.027–0.074 m; 6 of 20 within 0.05 m) and slides further.
- **Starts.** On 10 of the 70 landing variants the first frame, 1/60 s after the restored exemplar, has a planted
  hand body 0.51–0.68 cm off S's (`R_Wrist` on 8 of them), which fails `start`'s 0.5 cm hands rule. On every variant
  every body is within 0.98 cm.


## Admission (T5 + the seam group) and the selection

[`../admitted_physx.json`](../admitted_physx.json) (`admit.py --physx-launch-check ... --policy d1`; 90 variants).
It carries every T5 group, the `seams` group per variant (literal, attainable, and the legacy measure), the reset
check, the D1 policy, and each variant's D1 verdict.

| Reading (`select_variants`) | Variants | Edges |
|---|---|---|
| **The user's (default): drift ≤ 1.0 cm, end attainable, the landings' slide accepted** → [`../selected_physx.json`](../selected_physx.json) | **29** | E1 ×8, E3 ×3, B1 ×5, E2 ×12, E5 ×1 |
| The same without the landings (`--landing-slide reject`) | 11 | E1 ×8, E3 ×3 |
| The card's literal: drift ≤ 0.5 cm, landings rejected (`--drift-cm 0.5 --landing-slide reject`) | 0 | — |
| Any drift threshold, end strict (`--seam-end strict --landing-slide reject`) | 0 | — (the palms) |
| Before T6: no seam group ([`../selected_physx_pre_t6.json`](../selected_physx_pre_t6.json)) | 24 | E1 ×7, E3 ×3, B1 ×9, E2 ×4, E5 ×1 |

- **E1 and E3 (11).** Every body starts within 0.48 cm of S (the hands within 0.29 cm). The knuckles end within
  0.75 cm of D, and the wrists at the exemplars' own 2.8–3.2 cm. Drift 0.72–0.94 cm; final 6-body error
  0.016–0.049 m.
  - E1: slow s0, s2, s3, s4 and mid s0, s2, s3, s4.
  - E3: noise 0.05 s0, s2; noise 0.12 s3.
- **The landings (18).** On a landing edge only the start seam is hard; drift and end are recorded as advisory
  (`seams_drift`, `seams_end` in `advisory_failed`).
  - **B1 ×5:** high s1, s2 and mid s0 (noise 0.08); high s4 and mid s1 (noise 0.12). All are reference-only runs,
    because every cone run fails B1's hard endpoints. Final 6-body error 0.027–0.049 m. The feet slide 7–26 cm, the
    hands drift 0.8–1.6 cm, and the planted bodies end 3–10 cm from D's.
  - **E2 ×12,** from both sets. Final 6-body error 0.078–0.096 m (the low chaturanga, advisory). The feet slide
    2–5 cm, the hands 1.3–2.6 cm (the right hand), and the planted bodies end 2.8–7.0 cm from D's.
  - **E5 ×1:** reference-only s0, the only E5 run that holds chaturanga. Final 0.089 m; the feet slide 11–12 cm and
    end 4.5 cm from D's.
- **What R2 now has to absorb at the D seam.** The planted bodies' end offsets: the E1/E3 palms (2.8–3.2 cm, a 12°
  yaw) and the landings' 2.8–10 cm. R2's card blends inside the variant's hold toward D's exemplar and accepted
  "planted bodies move ≤ 0.5 cm across each seam". Under the user's decisions that blend carries these offsets as a
  further slide; R2 has to say how (PLAN.MD R2).
- **What else D1 rejects** (the hard groups besides seams; most variants fail several):
  - endpoints, hard on E1, E3 and B1: 38 variants. That is all 20 B1 runs with the cone, 14 of 20 reference-only B1
    runs, three E3 (0.0505–0.054 m), and E1 mid s1, the seed that falls;
  - held: 21, which is 20 landings and the E1 fall;
  - statics: 4, which is E1 mid s1 and three E2 landings.
- The reset check: 90 clips load on `smpl_yogi_v2`, plant v1 refused, FK ≤ 1.06e-6 m on 1,800 frames, **0 of 1,800
  resets launch**, positive control 3/3.


## Decided by the user (2026-10-04)

`select_variants.T6_DECISION` records these, and `admit`/`select_variants` apply them by default.

1. **The drift gate is read at 1.0 cm.**
   - The card's 0.5 cm was set against "today ≤ 0.4 cm", which was the T2 reference's drift. Executed PhysX hands
     drifted 0.4–3.4 cm before T6.
   - The press's planted palms rock 0.7–1.3 cm transiently and settle back.
2. **The palms' end seam is read `attainable`:** "the policy should learn around it". The wrists end where S's
   exemplar has them, 2.7–2.9 cm from D's, because the human turned her palms 12° between the two poses.
3. **The landings' slide is accepted:** "the key goal is can the policy even pick up these dynamic edges". On B1, E2
   and E5 only the start seam stays hard.
   - The feet land 5–42 cm from D's and slide 2–29 cm, because the plant's right elbow cannot push the shoulders back
     over the hands.
   - The expert may therefore learn a loaded-foot slide after a jump-back. G3's edge metrics and renders will show it.

What the decisions cost, and their alternatives, as recorded before the decision:
- at 0.5 cm drift nothing entered Step 3;
- without the landings, two edges remained, one short of the three the "new edges" claim needs;
- generating the landings with the expert as MPPI's sampling prior (RGB), or a plant fix of the right elbow, remain
  the routes to landings without a slide.


## How to run

```bash
export PYTHONPATH=.:data/scripts
bash data/scripts/edge_synthesis/run_t6.sh                    # refs, mppi, export, check, admit, select
CUDA_VISIBLE_DEVICES= python -m edge_synthesis.exact --edge E1 --timing high      # one reference (CPU, ~1 min)
LAND='{}' LAND_E2='{"anchor": 8, "box": 2, "flat": 0}' R=output/edge_synthesis/physx/runs_t6_nocone \
  MOT=output/edge_synthesis/physx/motions_t6_nocone L=output/edge_synthesis/physx/logs_t6_nocone \
  STAGES="mppi" RECIPES="B1h B1h12 B1m B1m12 E2m E2m12 E5m12" bash data/scripts/edge_synthesis/run_t6.sh
  # the landing recipes as recorded; their runs were renamed *_t6rpx / *_t6rpx12 and exported into motions_t6
CUDA_VISIBLE_DEVICES= python -m edge_synthesis.select_variants      # the user's reading -> selected_physx.json
CUDA_VISIBLE_DEVICES= python -m edge_synthesis.select_variants --drift-cm 0.5 --landing-slide reject --out <json>  # the card's
CUDA_VISIBLE_DEVICES= python -m edge_synthesis.t6_report
CUDA_VISIBLE_DEVICES= python -m pytest data/scripts/edge_synthesis/tests -q        # 18 tests, 7 s
```
