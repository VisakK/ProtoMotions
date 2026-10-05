# Card R2: the synthetic clip writer (2026-10-04)

**What it is.** Each of the 29 selected PhysX variants ([`../selected_physx.json`](../selected_physx.json), the user's
T6 reading) is now a full clip in S's clip frame: a real lead-in from S's clip, S's exemplar, the variant (its settle
blended from S's exemplar, its transition as executed, its hold at D blended toward D'), and a real lead-out from D's
clip placed by the generator's hand-anchor transform. Each clip has holds that inherit S's and D's labels (`extend`
on S only) and per-frame lineage. Nothing in release v2, `edge_synthesis/` or the v2 builders was touched.

**The build: `synthetic_v3.d034199f23`**, the only one. `--check` passes 469 checks with 0 failed, in 19 s. The
round-1 build `synthetic_v3.a04f2adb04` was deleted after review: the fixes below changed the builder, so the id changed.
- Clips: `data/smpl/reference_curation/synthetic_v3/synthetic_v3.d034199f23/`: 29 × (`.motion`, `.holds.yaml`,
  `.lineage.npz`, `.json`), 116 files, 75 MB.
- Record: `data/reference_curation/synthetic_v3/synthetic_v3.d034199f23.json`. It holds every file's sha256 and the
  id key: the selection, the admission, release v2's record and `edges.json` by sha256; each variant's motion,
  export record and `run.json`; the builders; plant v2's MJCF sha256; the parameters. It also holds R3's clip order,
  the name collisions, the trimmed windows, the unplanned transition contacts and the library checks.
- Built in 86 s on one core (`OMP_NUM_THREADS=1`, no `REFERENCE_PLANT`).
- The build is reproducible. A second build into a scratch folder gave byte-identical `.motion`, `.lineage.npz` and
  `.json` files for all 29 clips. The `.holds.yaml` files differed only in `source`, which is the output path.

| File | What |
|---|---|
| `data/scripts/reference_curation/splice_v3.py` | the writer and every check; CLI `--build`, `--check <id>`, `--print-id`. It refuses to overwrite an id; `--force` deletes only a failed build's folder. |
| `data/scripts/reference_curation/tests/test_splice_v3.py` | 18 tests, 22 s. They cover the clock, the blend weights and the no-repeat rule, the layout, the names and their collisions, the transform (FK on a real frame), the hands-midpoint placement, the holds on a synthetic case (with trimmed windows and the extension plan), the content-window rule, the near-repeat rule, and the lineage partition. Three are negative controls: a perturbed real frame, an unlabelled contact in a hold, and a non-tie horizon. The rest run `--check` on the build, re-splice a clip in memory and compare it to the build bit for bit, and check the knee-down clip's trimmed window. |

| Edge (clips) | Frames | Length | L / k_a / N / lead-out | x7s length (R3) |
|---|---|---|---|---|
| E1 press ×8 | 1,114–1,204 | 18.6–20.1 s | 335 / 194, 287 / 318, 408 / 460 | 25.6–27.1 s |
| E3 lower ×3 | 1,222 | 20.4 s | 522 / 257 / 378 / 321 | 27.4 s |
| B1 jumpplank ×5 | 929–959 | 15.5–16.0 s | 335 / 82, 113 / 204, 234 / 389 | 22.5–23.0 s |
| E2 jumpback ×12 | 949 | 15.8 s | 335 / 82 / 204 / 409 | 22.8 s |
| E5 floatdown ×1 | 1,172 | 19.5 s | 522 / 116 / 240 / 409 | 26.5 s |

Every lead-in, lead-out and hold follows the card's table, with four holds per clip:
- lead-ins: Crow -a 316–650 and Handstand -a 671–1192;
- lead-outs: Handstand -a 1193–1652, Crow -a 651–971, Four-Limbed Staff -a 485–893 and Plank -a 541–929.

The one exception is the start of D's window on `SYN_E2_jumpback_mid_s3_t6rpx12` (deviation 3).


## Acceptance (final build)

| Check | Result |
|---|---|
| FK round trip (`fit_writer.round_trip`, ≤ 1e-5 m) | **4.6e-07 m** worst over 29 clips; plant v2's `plant_sha256` on all; pressure channels 0, validity 0 |
| Real frames vs sources after the recorded transform | positions **6.3e-07 m**, rotations 4.5e-05°, `dof_pos` **bit-identical**, contacts identical, on all 29 |
| Real-frame velocities (≤ 1e-2 away from seams) | linear **0.0065 m/s** worst on 22,286 compared frames (111 excluded next to a seam); `dof_vel` 1.8e-05 rad/s. Angular: 21 entries above 1e-2 (worst 0.032 rad/s per component, 0.040 as a vector), **every one a genuine horizon tie**: the stored value equals another horizon's candidate whose magnitude is within 1.6e-05 rad/s of the smallest. 0 unexplained |
| Transition frames = the variant's | positions 6.0e-07 m, `dof_pos` bit-identical |
| Four junctions, planted bodies ≤ 0.5 cm frame to frame | lead-in → S 0.005, S → settle 0.053, settle → transition **0.110** (B1; the variant's own step), hold → lead-out 0.052 cm |
| D blend planted-body travel (recorded, not gated) | Offset carried: E1 2.8–3.0 and E3 2.9–3.2 cm (the wrists: the palms' 12°); B1 3.1–10.1, E2 2.7–6.9 and E5 4.5 cm (the feet). Peak planted speed in the blend, including the variant's own slide: E1 2.7–3.4, E3 2.8–3.5, B1 4.4–10.4, E2 5.8–13.9, E5 13.7 cm/s |
| Seam acceleration rule, **card reading** (gated) | Sources: S's and D's hold windows in the human clips, on all four junctions. All 116 junctions pass. Tightest margin **1.22 m/s²**: E1 mid s0, `R_Toe`, settle → transition, 16.40 against 17.62 |
| Seam acceleration rule, **parts reading** (gated) | Sources: each junction's own two parts (S; the variant's settle and hold at D; D). All 116 junctions pass; tightest margin 5.92 m/s² |
| No two consecutive frames identical | **0** identical pairs. 1–4 near-repeats per clip (a step < 1e-5 m on every body, smallest 3.3e-07 m), **all inside a hold window** (gated): the quintic's flat ends. S's exemplar → first settle frame 3.7e-04–0.053 cm; last D blend frame → D' 3.3e-05–0.052 cm |
| Blend frames: certify's contract, per part | box excess **0.00°**, floor ≥ **−0.099 cm**, **0** new overlaps > 1 mm after de-penetration (23 parts in 22 clips; deviation 6) |
| Hold windows hold their labels (gated) | 0 non-real frames in any hold window touch the floor with an unlabelled zone. Labelled ground zones touch on 100 % of those frames. One window was trimmed (deviation 3) |
| `hold_extension_v2.velocity_drift` mean ≤ 0.02 per field | **5e-06** worst mean (`rigid_body_ang_vel`; worst single entry 0.117 rad/s) |
| MotionLib on `smpl_yogi_v2`, CPU | **197 motions** (168 v2 + 29) load, plant v2 accepted, the pressure channel packed, frame counts match |
| Holds well formed | Ordered and disjoint on all 29. Every `inherits` resolves to a v2 hold with equal name, pairs, orientation and every copied field. Every `frame_hold` is that hold's own real exemplar frame |
| Hold extension dry run (R3's step) | x3s and x7s of every clip: S is the only insertion, inserted frames repeat S's exemplar, pressure stays zero, holds stay well formed, and S's window grows by exactly 180 / 420 frames |
| `build_hold_graph_v2` dry run (4 endpoint x0 + 29) | 8 nodes, 18 edges, 135 hold ids. **Every key is one of v2's**, and `pair_names` equal v2's. S → D is present for all five edges, with exactly one synthetic occurrence per clip (8 / 3 / 5 / 12 / 1) |
| Same on x0 + x3s + x7s (87 synthetic motions, R3's library) | same keys; S → D occurrences 24 / 9 / 15 / 36 / 3 |
| Variant in S's clip frame | variant frame 0 is **0.19–0.97 cm** from S's exemplar on every body (E1 0.27–0.48, E3 0.19–0.38, B1 0.46–0.77, E2 0.31–0.97, E5 0.35); the build stops above 1 cm |
| Transform = the generator's | yaw −1.218° (E1), +1.218° (E3), +1.131° (B1), −0.782° (E2), +0.437° (E5) against `run.json`'s −1.22 / 1.22 / 1.13 / −0.78 / 0.44: **≤ 0.0033°** apart (run.json rounds to 0.01°). FK of the transformed root on a real frame equals the rigidly transformed bodies to **2.8e-16 m**; the lead-out's first frame is D' to 2.4e-07 m |
| Names | the card's three asserts pass; no real stem contains `SYN_`; 9 synthetic-vs-synthetic collisions recorded |


## Deviations from the card, and why

1. **The D-seam weight is `smoothstep((k − k_a + 1)/(N − k_a + 1))`, not `/(N − k_a)`.**
   - The card's formula reaches 1 on the last variant frame. That frame then *is* D', and the lead-out's first frame
     (D' itself) repeats it.
   - With the `+ 1` no two consecutive frames are identical.
   - The quintic's flat ends still leave 1–4 near-repeats per clip, steps below 1e-5 m: the last D-blend frames (1 − w
     ≈ 10/(n+1)³ ≈ 5e-6) and the first settle frame.
   - These are gated to lie inside a hold window, and all do. Nothing outside a hold stalls.
2. **Synthetic stems collide with each other by substring.**
   - `t6px` is a prefix of `t6px12`, and `t6rpx` of `t6rpx12`. So 3 stems sit inside 3 others and inside their
     x3s / x7s names: 9 collisions, recorded as `name_collisions`.
     - `SYN_E2_jumpback_mid_s1_t6px` in `…_s1_t6px12` (and `_x3s`, `_x7s`);
     - `SYN_E2_jumpback_mid_s1_t6rpx` in `…_s1_t6rpx12` (and `_x3s`, `_x7s`);
     - `SYN_E2_jumpback_mid_s4_t6px` in `…_s4_t6px12` (and `_x3s`, `_x7s`).
   - The card's names are kept, because R3 and R4a use them. **Every per-stem filter downstream (R3's `--drop`, any
     drop list) must match stems exactly.**
   - **The runtime's own motion filters match by substring:** `support_exclude_motions` and
     `physics_exclude_motions` (`contact_graph_control.py`), and `report_exclude_motions` and `drag_report_motions`
     (`hold_curriculum_evaluator.py`). A G3 config pattern naming one of the three shorter stems also selects the
     longer one and its variants.
   - Patterns on the prefix `SYN_` or `SYN_<edge>` are unaffected.
3. **A hold's window must hold its label, so one D window starts late.**
   - The card puts D's window at the arrival. `SYN_E2_jumpback_mid_s3_t6rpx12`'s variant lands **knee down**: its
     own hold at D has `L_Knee` on the floor on 100 of 122 frames (admission `free_violations` L_SHANK 77 /
     L_THIGH 66).
   - The blend toward D' lifts the knee only after 23 frames. In the round-1 build, D's window (Four-Limbed Staff
     -a@485, labelled feet + hands, shank and thigh known-free) therefore opened with 23 knee-down and 15 thigh-down
     frames.
   - Rule (`content_windows`): no synthetic or blend frame in S's or D's window may touch the floor (stored
     `rigid_body_contacts`, the 1 cm rule) with a zone the hold does not label. D's window starts after the last
     such frame, and S's ends before the first.
   - Effect: the trim applies to this one window, **418 → 441** (D's `frame_hold` 540 and its end 557 are
     unchanged). The other 28 clips keep the card's windows.
   - At frame 441 the knee is 1.06 cm above the floor (the lowest collider), just past the 1 cm touch rule. It rises
     toward D's 11.7 cm by the lead-out.
   - The rule is gated and recomputed by `--check`. The record lists the trim under `windows_trimmed`.
4. **The variant's clip frame is checked, not assumed.** Variant frame 0 sits 0.19–0.97 cm from S's exemplar, so no
   transform is applied to the variant.
5. **The transform, checked three ways.**
   - Recomputed from release v2's exemplars, it equals `run.json`'s (≤ 0.0033°).
   - It moves the root only (position and rotation; the velocities follow). `dof_pos` is untouched. Quaternions are
     xyzw: `fit_writer.write_motion` takes `[T, 4]` xyzw.
   - A planar Procrustes fit of D's raw exemplar onto each variant's last frame (all 24 bodies) finds a yaw −2.6° to
     +4.3° off the hand anchor's. That is the variants' whole-body end pose, not the hands.
6. **The blend frames are checked directly, per part, and 22 of the 29 clips needed de-penetration.**
   - `quasistatic.certify` needs a solved problem and a schedule. Its three rules are therefore computed on the
     blend frames with its geometry (`contract`):
     - no joint coordinate more than 1° past the box;
     - collider points ≥ −0.5 cm;
     - no pair deeper than 1 mm (+0.1 mm tolerance) unless it is already that deep in the part's own endpoints.
   - The endpoints are part by part. The settle uses S's exemplar and the variant's first frame. The hold at D uses
     D's exemplar and the variant's last frame. Only the part's own hold's braces are exempt, and only when the
     exemplar holds them within `PAIR_REACH_M` (2 cm). This is certify's endpoints-and-closed-braces rule.
   - **23 blend parts failed and were de-penetrated** (`export_physx.DEPENETRATE`, 40 iterations):
     - 21 settles. The crow's feet rest together (`L_Ankle`/`R_Ankle` 1.0 mm deep in S's exemplar). Slerping two
       leg chains between them drove the ankles 1.1–10.1 mm into each other.
     - E3 high s3's hold at D: the same ankles, toward crow.
     - E2 mid s3 t6rpx12's hold at D: the blend put the left ankle 1.38 cm through the floor.
   - Edits: settles 0.08–0.51 cm max (median 0.004–0.017 cm); the two D holds 0.34 and 1.63 cm. After the solve 0
     frames fail. The box rule failed nowhere (0.00°).
   - In three clips an exempt brace overlaps by more than 1.1 mm. Each is recorded as `brace_depth_cm`:
     - E3 high s0's D blend: crow's `R_SHANK+R_UPPER_ARM`, 1.87 mm. D' and the variant's last frame are both at
       about 0.8 mm (review), so slerping between the two braced poses deepens it.
     - E3 high s2's D blend: `L_SHANK+L_UPPER_ARM`, 1.24 mm.
     - B1 mid s1 t6rpx12's settle: `L_SHANK+L_UPPER_ARM`, 1.24 mm.
   - **Deviation:** `export_physx`'s solve pins nothing. Here the 2 frames on either side of the blend part (real or
     transition frames) are bounds-pinned, through an in-process `retarget_v2.bounds` wrapper as `exact.py` does.
     The edit then tapers to nothing (the first and last edited frames move ≤ 0.053 cm), and real frames stay
     exact. `retarget_v2.py` is untouched.
   - Every edit is in `<stem>.json` (`depenetration`) and flagged per frame in the lineage (`depenetrated`).
7. **The seam acceleration rule, defined and gated twice.**
   - Acceleration is admit's metric, `|p[f+1] − 2p[f] + p[f−1]|·60²`, on the written `rigid_body_pos`. A junction's
     window is its two frames ± 3 (`VELOCITY_MAX_HORIZON`). The limit per body is the larger of the two sources'
     hold-window maxima + 10 m/s².
   - The **card reading** takes the human sources, S's and D's hold windows in their clips, on every junction. The
     **parts reading** takes each junction's own parts: S; the variant's settle plus its hold at D; D.
   - Both readings are computed, stored per junction in `<stem>.json` (`acc_rule.card`, `acc_rule.parts`) and gated.
   - Maxima use interior stencils, the stricter choice: S 6.0 (crow) / 7.6 (handstand); D 7.6 / 6.0 / 1.6
     (chaturanga) / 6.6 (plank); the variant 6.2–22.6 m/s². Counting the windows' end-frame stencils as well gives
     the card's 7.9 / 8.4; that is recorded as `hold_window_max_full`.
   - Worst spliced value per junction: lead-in → S 8.6, S → settle 11.5, settle → transition 16.4, hold → lead-out
     6.3 m/s².
   - The card reading's tightest margin, 16.40 against 17.62, is the S blend chasing the variant's settle sag (see
     "What R3 must know").
8. **Angular velocity is compared tie-aware.**
   - pose_lib stores, per body, the smallest of three forward-difference horizons. Where two nearly tie, float32
     round-off in the placed lead-out picks the other one. Example: |h1| = 0.395299 vs |h2| = 0.395300 rad/s on Four-Limbed Staff -a's `R_Toe`
     at its frame 597.
   - Linear velocity is gated at the card's 1e-2 m/s.
   - An angular entry above 1e-2 must equal a source horizon candidate within 1e-3 rad/s. That candidate's
     magnitude must be within 1e-4 rad/s of the smallest candidate. A wrong horizon choice fails this, and a
     negative-control test proves it.
9. **"Within 3 frames of a seam" is read from the stencil.** pose_lib's velocity at f reads frames f…f+3. A real frame
   is compared when those are its source's own consecutive frames, or when both clips end there. That excludes 3
   frames per clip (the last lead-in frames), and 6 on E1, whose lead-out ends mid-clip at Handstand -a 1652.
10. **The id key is wider than the card's.**
    - Besides the four inputs and `splice_v3.py`, it hashes each variant's motion, export record and `run.json`.
    - It also hashes every module whose code shapes the written bytes: `fit_writer`, `retarget`, `mosh_replay`,
      pose_lib, `sketch`, `retarget_v2`, `quasistatic`, `export_physx` and `extract_contact_configs` (the zones).
    - It hashes plant v2's MJCF sha256.
    - `quasistatic.py` is uncommitted in the working tree. Any change to it gives a new id, which `--check` notes
      without failing.
11. **Pinned copies, not module constants.**
    - `T6_DECISION` is read from the selection's `rule.t6_decision`, which is pinned by its sha256, not from
      `select_variants.py`. The two are equal today.
    - Release v2's `release_holds.yaml` and `contact_graph.json` are checked against its record's artifact sha256s
      before use.
    - `--check` resolves each variant from its clip's own `synthetic` block and the record's key hashes, never from
      today's selection. A changed or missing variant is a named failure.
12. **`<stem>.holds.yaml`'s `source` is absolute**, as in v2's `release_holds.yaml`; `release_v2.verify_sources` opens
    `Path(c["source"])`. **`<stem>.json` carries no wall-clock field**, so all three per-clip data files and the json
    reproduce byte for byte.
13. **The card's "about 5 cm/s" in the D blend is a mean.** Smoothstep peaks at 1.875× the mean, and the variants' own
    hold slides add to it (E5 slides its feet 6.4 cm inside its own hold). The peaks are in the table above.


## Review round 1: what changed

Three reviewers (spec, data, code) found no blocker. Their findings and the fixes:
- **Major (data):** E2 mid s3 t6rpx12's D window opened knee down. Fixed by deviation 3, with a gated content check.
- **Minor:** the seam acceleration rule gated only the loose reading. The card reading is now computed, stored and
  gated (deviation 7).
- **Minor:** runtime filters match by substring. They are now named in the record's `name_note` and in deviation 2.
- **Minor:** the id key was too narrow, and the release manifests were read unverified. Fixed by deviations 10 and 11.
- **Nit:** near-repeats. Now measured and gated inside holds (deviation 1).
- **Nit:** brace exemptions crossed parts, and the box rule was missing. Both fixed (deviation 6). The output did not
  change: the same 23 parts were de-penetrated.
- **Nit:** the tie check did not require a genuine tie (deviation 8).
- **Nit:** `--check` resolved variants by name (deviation 11).
- **Nit:** the per-clip json held wall-clock seconds, and `source` was relative (deviation 12).
- **Nit (disclosed, kept):** S's copied `speed_at_hold` and `labels.frame_*`. See "What R3 must know".


## What R3 must know

- **Read the record, not the folder.**
  - `order` is R3's fixed order: edge (E1, E3, B1, E2, E5), then timing, seed and tag.
  - `clips.<stem>` gives each file's path and sha256. All four per-clip files are reproducible, so all four can be
    pinned.
  - `edges` lists the stems per edge (8 / 3 / 5 / 12 / 1).
- **Drop lists and G3's motion-filter patterns:** see deviation 2.
- **`<stem>.holds.yaml` is the clip's manifest entry.**
  - Fields: `stem`, `group: edge`, `family: SYN_<edge>`, `source` (absolute), `fps`, `num_frames`, `length_s`,
    `capture_available: false`, `synthetic`, `holds`.
  - `synthetic` holds the variant and its sha256s, `recipe`, `T`, `durations_s`, `clock`, S, D, the transform, the
    full admission row, the D1 policy and `T6_DECISION`.
  - Each hold copies its source hold verbatim, except `hold_id` (`<stem>@<frame_hold>`), `inherits`, the frames and
    times, and `extend`. Hold ids of `SYN_` stems sort after every `220923…` id.
  - **`labels` is copied verbatim and carries the source clip's frame numbers** (e.g. Crow -a@439's `frame_start
    425`, `frame_hold 651`). Never read `labels.frame_*` on a `SYN_` hold as clip frames; resolve provenance through
    `inherits`.
- **The holds.**
  - S's window runs from its own start in the lead-in to the departure at L + 18; `frame_hold` = L, S's real exemplar.
    Inside it, the variant's 0.3 s settle moves the free bodies 3–14 cm from S's exemplar (the toes: E1 8.9–14.2,
    B1 6.1–12.0, E2 5.1–11.8, E3 2.9–5.5, E5 7.8 cm) and the hands ≤ 1.3 cm. This is the PhysX crow sag that T7
    found. The crow's feet stay ≥ 15 cm above the floor, so the labels hold.
  - D's window runs from the arrival to D's own end, except on `SYN_E2_jumpback_mid_s3_t6rpx12`, where it starts at
    441, not 418 (deviation 3). `frame_hold` = the lead-out's first frame, D's real exemplar under the transform.
  - The copied `speed_at_hold` and `pelvis_z` are the source's. The clip's own mean body speed at S's `frame_hold`
    differs by ≤ 0.018 m/s, because its velocity reads settle frames (pose_lib's forward stencil). Every other
    hold's speed matches its source to 4 decimals.
- **Unplanned floor contacts on transition frames** (record: `transition_contacts`).
  - In 11 of 12 E2 clips, the left knee and thigh touch the floor for 5–31 transition frames during the jump-back
    landing; s2 t6px12 also touches the right thigh for 4 frames. Only s4 t6rpx12 is clean.
  - No phase of E2 plans those zones on the floor; `float_back` lists them as free.
  - These frames are the variants' own executed content, and the admission recorded them (`free_violations`). R2
    keeps transition frames exact, so they are recorded, not edited.
  - R3's sidecar takes "ref" for synthetic frames from the clip's geometry, so it will see these contacts.
- **The lineage** (`<stem>.lineage.npz`; the record's `lineage_format`):
  - `stems` = [S stem, D stem, variant]. `source`, `source_frame` and `kind` (0 real, 1 synthetic, 2 blend) are per
    frame; blend and synthetic frames point at the variant's frame.
  - `blend_w`, `blend_source` and `blend_source_frame`: the real exemplar a blend frame mixes in, and its weight.
  - `yaw` and `xy`: the transform placing the frame's real source (or blend partner). Heights do not change under
    it, so real frames can take the source stem's capture-v4 evidence through `source_frame` directly.
  - `variant_t`: the variant's clock, for the phase of edges.json's schedule with `synthetic.durations_s`.
  - `depenetrated`, and the segment table `seg_*`.
- **Reusable checks** for R3's "synthetic lineage" group:
  - `splice_v3.check_real_frames(motion, lineage, {stem: v2 motion})` (R2's tolerances; tested to fail on a 0.1 mm
    perturbation and on a non-tie horizon);
  - `check_synthetic_frames(motion, lineage, variant motion)`;
  - `check_hold_content`, `name_checks`, `holds_well_formed`.
- **Pressure is zero with validity 0 on every frame** of every clip. The real frames' measured pressure stays in the
  source clips.
- The graph dry run covers only the four endpoint clips. R3 owns the remaining checks: the full node-key equality
  with v2, the sidecar rows, the tables wrapper and the plans.


## Open, for the orchestrator

- **Decided by the user (2026-10-04): drop `SYN_E2_jumpback_mid_s3_t6rpx12`, and keep the rest.**
  - Its knee spends 31 transition frames on the floor plus 23 blend frames before the window. The admission peak there
    is 2.5 BW on the shank and 4.2 BW on the thigh.
  - R3 applies the drop with `--drop SYN_E2_jumpback_mid_s3_t6rpx12`, matching the stem exactly, so release v3
    carries 28 variants (E2 ×11).
  - This build keeps all 29 clips unchanged.
  - The 10 other E2 clips whose left knee brushes the floor in transition stay in.
- T5's PhysX reset check on the spliced clips (pre-launch checklist item 3, `retarget_v2_physx.py`) needs IsaacLab
  and the GPU. It was not run; it belongs to the integration stage.
- PLAN.MD's status was not updated: the orchestrator owns it.


## How to run

```bash
export PYTHONPATH=.:data/scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
../env_isaaclab/bin/python -m reference_curation.splice_v3 --print-id           # synthetic_v3.d034199f23
../env_isaaclab/bin/python -m reference_curation.splice_v3 --check synthetic_v3.d034199f23   # 469 checks, 19 s
../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests/test_splice_v3.py -q  # 18 tests, 22 s
```
