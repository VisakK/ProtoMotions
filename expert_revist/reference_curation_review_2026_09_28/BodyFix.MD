# Body fix: rebuild the plant on the subject's skeleton, then re-run the reviewer (2026-09-30)

**Why this file exists.** The simulated body (the "plant", `data/assets/smpl/smpl_yogi03596_lowtorque.xml`)
is not the MOYO performer's body. Most of the floats, the missing headstand crowns, the infeasible holds, and the
large Step 8 retarget edits that caused slide and jerk come from that mismatch. This file hands over the
fix. It **comes before** `TODO_before_step9.MD` and BUILD_PLAN Step 9.

**Kickoff prompt for a new session:** `Implement the next todo step of expert_revist/reference_curation_review_2026_09_28/BodyFix.MD`.
Read §1–§3, then only the card of the first `todo` step in §4, and update §5 when done.

**Decisions already made by the user (2026-09-30). Do not re-open them.**
- **Fix the skeleton and the masses, and keep the body segments commensurate** (each collider spans its
  own segment at the new lengths). **Do not fit collider shapes to the skin.** Skin-fitted colliders were
  prototyped. They made human-sized legs overlap each other (knee–knee overlap ×2.9, thighs touching at rest)
  and added only 12 of the 286 floats the prototype removed.
- **The existing `.motion` files and every trained policy are disposable.** Regenerate and retrain; no backward
  compatibility is needed.
- **Order: fix the asset, then re-run the offline reviewer to strengthen the labels, then train.** No policy is
  trained until the reviewer has signed off the new references and labels.
- **Firefly -b is dropped** (decided after Step 3). Its trunk folds 24.9° past the plant's Torso_y stop, with the
  thighs wedged into the upper arms, and every retarget of it either jerked or kept 14–22 new overlaps. It joins
  the two lotus clips and Standing big toe hold -c in `fit_writer.DROPPED`; the corpus is 56 clips.

---

## 1. What was found (verified, with numbers)

Investigation outputs: `output/reference_curation/subject_body_check/` (git-ignored; see §3). Every
number below was reproduced by an independent checker (`verify_conventions/`, `verify_numbers/`,
`adoption_risks/`).

**The subject.**
- The personal SMPL-X female template is byte-identical in all 336 readable MoSh fits, across the 220923,
  220926 and 221004 sessions. It equals `moyo_toolkit/data/v_templates/220923_yogi_03596_minimal_simple_female/mesh.ply`
  (sha1 `9f7d1711…`).
- All betas are 0 and the shape was never re-fitted per clip.
- Stature 1.748 m. Closed-mesh volume 72.4 L, i.e. 71–77 kg at 985–1060 kg/m³; the mat reads 70.7–76.0 kg on
  single-leg clips. The plant's 74 kg total is right.
- One file is corrupt: Standing big toe hold -c (`UnpicklingError`). That clip has no fit and should be dropped.

**The plant.**
- `build_subject_skeleton.py` took the SMPL **neutral mean shape** (boxhands) and rescaled only arm and leg
  lengths from a surface-marker measurement. That measurement is biased by −3.7 cm on the shin. As a result it
  **shortened** legs that are actually longer (x0.977 applied; x1.04–1.10 needed).
- The torso, head, hands, feet and every collider radius were never fitted to her.

| Quantity | Subject | Plant | Note |
|---|---|---|---|
| Stature | 1.748 m | 1.577 m | −17 cm |
| Hip→ankle | +4.7 / +5.3 cm longer (L/R) | — | after correcting for SMPL-vs-SMPL-X joint definitions |
| Thigh / shin (SMPL-X female J_regressor) | 0.404 / 0.415 m (L) | 0.368 / 0.391 m | the thigh/shin split moves ±2 cm with the regressor used; hip→ankle is the robust number |
| Wrists | — | ~3 cm too lateral, 1–3 cm arm excess | definition-corrected |
| Crown above pelvis joint | 0.773 m | 0.671 m (head-sphere top) | see "head" below |
| Collider volume | 72.4 L | 37.8 L (52 %) | pelvis 0.20x, trunk 0.35x, thighs 0.47x |
| Mean density | ~1020 kg/m³ | 1959 kg/m³ (1355–2710) | shanks, feet and forearms 1.4x too heavy; trunk light (45.9 % vs 51.8 % of mass) |

**Head.** The crown shortfall is the head collider's **placement**, not the neck length:
- The shipped head is a sphere of r 9.5 cm centred on SMPL's head joint, and that joint sits 20.2 cm below its own
  crown even on the SMPL mean body.
- On her template, a sphere of r ≈ 9.45 cm centred at (−0.9, −0.4, +4.7) cm from the SMPL-X head joint
  (head frame: x forward, y left, z up) covers the cranium and crown. The crown sits 0.14 cm inside it, and the
  chin lies outside.
- The apparent neck→head gap (0.083 vs 0.146 m) is mostly the two models defining the head joint in different
  places (`verify_conventions/defs_align.json`).

**A second mismatch: the reference angles come from a different fit.**
- The shipped `.motion` body rotations equal MOYO's **AMASS SMPL-X neutral** fit (`YOGI_2_latest_smplx_neutral/*.npz`,
  300 fixed betas) × `retarget.AXES`, to 0.001°. The exceptions are the 2,379 frames that `repair_armbalance_holds.py`
  edited in 8 clips.
- The human mesh, capture stores and labels use the **female MoSh** fit. The two differ by 2.9° p50, with toes and
  ankles p95 10–19°.
- The two fits give the same surface (p50 0.41–0.49 cm). Their angles differ because their skeletons differ.
- **Rule: use the female MoSh fit everywhere.**
- MimicKit dropped the betas and the pelvis joint offset J0, and the per-frame grounding then re-derived the
  root height from the wrong plant. As a result the pelvis sits 6.0–6.8 cm too low in standing clips and
  3.5–4.7 cm too high in hand- or head-supported clips.

**What the subject-sized skeleton buys.** The comparison replays the female fit's own motion. Both sides are
grounded to +0.5 cm (the fair comparison) over 59 clips:

| | Shipped refs | Shipped plant + female fit | Subject skeleton, shipped colliders (grounded) | Skin-fitted prototype (grounded) | Step 8 retarget |
|---|---|---|---|---|---|
| Labelled supports floating > 2 cm (of 919) | 461 | 399 | 261 | 175 | 27 |
| Floating > 5 cm | 197 | 190 | — | 2 | 27 |
| Headstand crowns floating (of 27) | 27 | 27 | 27 (head sphere still on the joint) | 3 | 27 |
| Gated-LP holdable holds (of 300) | 38 | 41 | not run | 141 (190 at 0 cm clearance) | 270 |
| COM vs mat COP, p50 | 1.69 cm | 1.91 cm | — | 1.10 cm | 1.67 cm |
| Frames with a body > 100 m/s² | 130 | 1 | 2 | 2 | 321 |

- **Where the float gain comes from (grounded):**
  - skeleton: −138;
  - finger-based hand rotation (`hand="fingers"`, a kinematic choice): −74;
  - neutral → female fit: −62;
  - skin-fitted colliders: −12.

  The skeleton and the hand rotation carry the result. Collider shape does not, which is why shape matching
  is dropped.
- **Slide.** The female fit's own motion slides less than the shipped references: clip p90 2.8 vs 4.5 cm/s,
  against 16.4 after the Step 8 retarget. The Step 8 edits were mostly compensating for the wrong body.

**What the body fix does NOT fix (design for these in §4):**
1. **Joint box.** 282/300 hold exemplars exceed PhysX's hard exp-map box by more than 2°, whatever the body.
   - Clipping the female-fit motion into the box moves the farthest body 8 cm (p50, p95 27 cm), and on the
     prototype floats rise 12 → 41 (351 once re-grounded).
   - So a retarget is still needed, only with smaller edits.
   - Hotspots (share of frames > 2° past): Thorax_z 37 % / 32 % (p95 19°), Elbow_x 26 % / 24 % (p95 27–40°),
     L_Ankle_y 21 %, Elbow_z/y 17–18 %, R_Toe_x 11 % (27 % with the female fit).
   - Several of these are likely SMPL-convention artefacts (e.g. forearm pronation stored at the elbow) rather
     than anatomy (`adoption_risks/results/joint_box_by_joint.json`).
2. **Foot supination is in her fit, not in the box.** The right-foot box is about 11° off her skin sole on every
   plant. Her own ankle frame tilts about 16° in stance while her sole is flat (skinning blend). This needs a
   foot-orientation correction in the retarget, not a new collider.
3. **Wrist and hand torque limits (20 / 10 N·m) bind** subject-sized arm balances: beyond_plant 7–14; Scorpion -b
   needs s* 2.8.
4. **Floor penetration under load.** The skin sinks (pelvis −5 cm, palm −1 cm) and rigid colliders cannot. Do not
   ground the whole body per frame: that re-introduces 2–5 cm floats on about 19 % of supports. Keep the human's
   own root and correct penetration locally.

**Corrections to earlier records** (do not rely on these):
- The `yogi-subject-skeleton` memory, `build_subject_skeleton.py`'s docstring and `notes/Bring_up.md` §"Subject-specific
  skeleton" claim that "legs match". They do not.
- README §3.2's "limb length is not the cause; the error is angular" is refuted for the legs.
- TODO B3's "a new head collider … not this release" is superseded by this file.
- TODO B3 also says the 9 masked crown holds are "held by the forearms". Only Supported Headstand -a@1051 is. Five
  are kneeling or entry transitions on feet and shins, 1 is on feet and hands, and 2 are on the hands alone.

---

## 2. Facts and traps for whoever builds this

- **Joint definition.** Build the skeleton from the **SMPL-X female J_regressor on her template**
  (`human_mesh.SMPLX().J_regressor @ v_template`), not from SMPL joints. The motion comes from SMPL-X fits, so
  this makes FK reproduce her joints exactly (verified to 2e-6 m). The regressor carries a small L/R asymmetry
  (~1.4 cm of the hip→ankle L–R difference). Keep it: it is the skeleton the fit was solved on, and symmetrising
  would break the exact replay.
- **Offsets.** Child `pos` = `AXES.T @ (J_child − J_parent)`, with rest orientations kept at identity.
  - The hand body's origin is the mean of the 4 finger-base joints (`retarget.FINGER_BASES`).
  - The toes are SMPL-X joints 10 and 11.
  - `retarget.AXES` is verified: the SMPL-X zero pose is the avatar rest pose, and `dof_pos` = rotvec(R_parentᵀR_child)
    = PhysX's own coordinate.
  - The prototype's `fit.json` → `offsets_m` already holds all 23 offsets.
- **Hand rotation mode.** Take the hand body's rotation from the finger bases (`replay.mosh_kinematics(..., hand="fingers")`).
  This puts non-zero values in the L_Hand and R_Hand dofs, which no shipped reference had; the plant has these
  joints, and the mode is worth −74 floats. The shipped `"wrist"` mode leaves them at 0.
- **Models on disk.**
  - Female: `../moyo_toolkit/data/body_models/smplx/SMPLX_FEMALE{,_v1_1}.npz`.
  - Neutral and male: `~/Downloads/models/smplx/SMPLX_{NEUTRAL,MALE}.npz`.
  - SMPL template sample: `moyo_toolkit/data/essentials/hd_model/smpl/smpl_neutral_hd_sample_from_mesh_out.pkl`.
  - No SMPL J_regressor is on disk.
- **The pipeline assumes one geom per body, of type sphere, capsule or box.** `build_physics_tables`,
  `static_hold_lp` (`TYPED[b][0]`), `PhysicsTables`, `apply_mjcf_masses_to_usd` and `parse_typed_geoms` all depend
  on it. Keep it.
- **Masses: set them through geom density, not `<inertial>`.** `usd_convert/apply_mjcf_masses_to_usd.py` writes a
  single body-level density, and PhysX and MuJoCo then derive the same mass, COM and inertia from the primitive.
  An `<inertial>` element would be dropped at USD conversion, and `build_physics_tables.body_constants` reads the
  geom centre as the body COM. So per-body mass is right, but COM sits at the collider centroid and inertia follows
  the collider shape. Measure how far that is from her voxel inertia (§4 Step 1); only switch to explicit MassAPI
  authoring if it matters.
- **The MJCF→USD converter strips `<contact>`.** MJCF `<exclude>` pairs never reach PhysX
  (`convert_robot_mjcf_to_usda.py` `ELEMENTS_TO_STRIP`). A design that needs collision filtering needs a USD
  FilteredPairsAPI patch step. Commensurate colliders were chosen partly to avoid this.
- **USD conversion gotchas** (memories `usd-conversion-drops-mjcf-density`, `maskedmimic-yoga-boxhands-isaaclab`).
  Density is dropped unless `apply_mjcf_masses_to_usd.py` runs. The post-patch step does not run under `nohup`,
  so add `over "worldBody" ( active = false ) { }` by hand. Always check with `usd_convert/check_simulated_mass.py`.
  - *Resolved in Step 2.* The importer hangs on exit even outside `nohup`, so the wrapper never post-processes.
    Rebuild plant v2's USD with `data/scripts/build_plant_v2_usd.py`, which runs the steps itself.
- **MotionLib serves the stored `rigid_body_pos`, not FK.** A `.motion` built for another plant loads silently
  wrong. On the prototype, every shipped clip starts 5–6.6 cm inside the floor.
  - *Resolved in Step 2.* A library built for another plant now raises `PlantMismatchError`, and so does one
    whose `.motion` files carry no `plant_sha256` when it runs on v2. Writers must write the key.
- **Plant identity was not checked anywhere.** *Resolved in Step 2*, except for the finished capture and statics
  loaders (see "What Step 2 found"). The original findings:
  - `capture.load()` checks the schema, generator and calibration id, but not the MJCF.
  - `physics_tables.pt` carries no plant identity, so ftC tables would load on a new plant.
  - The MJCF path is hard-coded in about 12 scripts: `reference_curation/ids.py:51`, `static_hold_lp.py:35-36`,
    `build_physics_tables.py:71` (no flag), `add_onmat_gate_to_motions.py` (defaults `--mjcf`), `package_pair_targets`,
    `audit_pair_label_quality`, `validate_pair_reward`, `kinematic_share_baseline`, `repair_foot_contact_labels`,
    `plot_contact_physics`, `propose_hold_manifest`.
  - The full list with line numbers is in the adoption verifier's report (`adoption_risks/`).
- **§2 convention of BUILD_PLAN still holds:** extend a finished module with a new module or store version;
  don't edit it. `capture.py` and `audit.py` are hashed into the audit id, which the Pass A calibration replay pins.
  Since Step 4 the same holds for `labels_v2.py`, whose sha256 is evidence in every labels v2 folder, and for
  `b6.py`, whose sha256 is part of the B6 id.
- **Records of plant v2 go in new folders.** Step 1–8 tests read the newest folder under `gate/` (`test_gate`),
  the first three folders under `statics/` (`test_plant_v2_step2`), and every `calibration/render_v*.json`
  (`test_verdicts`). So Step 4 writes `gate_v2/`, `statics_v2/` and `calibration/pass_*/`.
- **Run the reference_curation tests single-threaded:** `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1`. With BLAS threads, `test_retarget` hangs (Step 3's 1000× `solveh_banded` slowdown).
- **Disk:** 120 GB free now (BUILD_PLAN §1's 3.8 GB is stale).
- **The PhysX rollouts `plant.py` used are gone** (`results/Contact_Physics_analysis*`). Force validation needs new
  rollouts from a policy on the new plant.

---

## 3. What already exists and can be reused

All in `output/reference_curation/subject_body_check/`, which is git-ignored. Copy what you adopt into
`data/scripts/` and commit it.

| Path | What it gives you |
|---|---|
| `prototype/build_plant.py` | Offsets from the template; voxel mass partition (3 mm); support-plane fits; XML text surgery applied identically to the main and `_flat` files; rest-overlap resolution. **The design to adopt reuses its offset, mass and XML-surgery code, not its collider-fitting rules.** |
| `replay/build_hybrid_plant.py` + `smpl_yogi03596_hybrid_protoskel_shippedgeoms.xml` | **Closest to the chosen design.** The subject skeleton with the shipped colliders: capsules re-spanned along the new bones by a similarity, radius unchanged; spheres and boxes riding the new joints. Step 1 starts from this and adds the head, foot/hand-box and mass changes. |
| `prototype/replay.py` | `mosh_kinematics(stem, hand=…)` (female fit → root pos/rot + 69 exp-map dofs in the clip frame, exact); `skeleton_for(xml)`; `fk_bodies`; `zone_lowest`; `pair_gaps`; `mass_model_for`; `use_plant(xml, flat)`, a context manager that repoints `ids.MJCF`, `static_hold_lp` and the lru caches in-process. |
| `prototype/fit.json` | `offsets_m`, `mass.body_mass_kg` (the voxel partition), `support_planes`, template joints. |
| `replay/replay_metrics.py` | The corpus metrics: floats with Step 8's definitions, touch-frame floats and penetration, crowns, self-overlap, pair gaps, slide (`build_physics_tables.zone_patch_speed` rule), box tilt, joint box, acceleration spikes. |
| `statics/run_statics_proto.py`, `compare_statics.py`, `com_vs_cop.py` | The Step 7 LP and statue on any plant, unchanged code (it reproduces be7cdb20a7 on 303/303 holds), plus COM vs mat COP. |
| `verify_numbers/`, `verify_conventions/` | Independent kernels (MuJoCo-compiled geoms, SMPL-X FK); SMPL vs SMPL-X definitional offsets (`defs_align.json`). |
| `adoption_risks/` | The hard-coded path list, joint-box hotspots per coordinate, USD collider check (the importer reproduces MJCF geoms to 1e-5 mm), physics sanity (inertia, PD, root height). |
| `provenance/` | The fit survey, shipped-vs-fit rotation comparison, pipeline trace, body size and mass. |

---

## 4. The five steps

### Step 1 — Build plant v2: subject skeleton, subject masses, commensurate colliders

**Goal.** A new MJCF pair `data/assets/smpl/smpl_yogi03596_v2.xml` and `_v2_flat.xml`, made by a committed,
reproducible builder `data/scripts/build_subject_plant_v2.py`. It must not edit `build_subject_skeleton.py`.

**Rules.**
- **Skeleton.** Every body `pos` from §2, i.e. her SMPL-X female joints. Joint names, axes and actuators stay
  identical to the shipped plant.
- **Colliders stay the shipped primitives and sizes** (radii and box half-extents unchanged, one geom per body)
  and follow the new segments:
  - **Capsules** (thigh, shin, upper arm, forearm): end points re-spanned along the new bone
    (`build_hybrid_plant.py`'s similarity rule), radius unchanged.
  - **Trunk and thorax spheres, neck:** ride the new joints at their shipped body-local offsets. Nudge one only if
    it creates a new rest overlap or leaves a gap between neighbouring trunk spheres.
  - **Head sphere:** keep r 0.095; **re-centre it on her skull** so it reaches the crown, near (−0.9, −0.4, +4.7) cm
    in the head frame (§1). Verify on the template: crown within 0.5 cm of the surface, and no rest overlap with
    Neck or Chest.
  - **Foot and toe boxes:** keep the sizes; **place them so the bottom face lies on her sole plane** relative to
    the new ankle joint. Her ankle joint is 6.8 / 7.6 cm above the sole (L/R) against the box's 5.3 cm, so move
    each box down by 1.5 / 2.3 cm. Keep the toe box flush with the foot box (the hybrid's parent-anchored rule).
    Optional, only if Step 1's statics show standing holds failing on it: the box is 3–4 cm short at the heel
    (her foot 26.9 cm vs 21.5 cm).
  - **Wrist and hand boxes:** keep the sizes; **rotate the hand box to the palm plane**. It is rotated 18.8° in its
    body today, 14.4° off her palm. Place the palm face at her palm relative to the wrist.
  - These are placements, not shape fits. **No collider may grow to the skin.**
- **Masses.** Per-body masses from her template volume at a uniform density that gives 74 kg (~1021 kg/m³), put on
  each geom as density = body mass / geom volume. Two checks:
  - **Segment boundaries.** The prototype's voxel partition (dominant skinning joint) gives Pelvis 15.3 kg and
    thighs 8.2 / 7.6 kg: it puts the buttocks on the pelvis. Use anatomical cut planes at the hip, knee, shoulder
    and elbow joints instead, and cross-check each segment against de Leva (1996) female mass fractions. At 74 kg
    that is approximately thigh 10.9, shank 3.6, foot 0.95, upper arm 1.9, forearm 1.0, hand 0.4 kg, head+neck
    4.9 kg, and trunk 31.5 kg in all. Record any segment more than 20 % off, with the reason.
  - **Inertia.** Report each body's inertia (collider-shape inertia at that mass) against her voxel inertia, and
    the whole-body rest COM and inertia against her uniform-density template. The prototype reached a whole-body
    inertia of 1.08 / 1.09 / 0.94x; the shipped plant is at 1.15–1.18x.
- **Joint limits and torque limits: decide them in this step**, so the plant changes only once.
  - Measure her female-fit motion per exp-map coordinate (the shares and p95 in §1, item 1).
  - For each hotspot coordinate, decide whether the violation is a representation artefact (widen) or a real
    anatomical limit the reference exceeds (keep; the retarget clips it). Justify each with numbers.
  - Decide wrist and hand torque limits with the Step 1 statics below (the s* of the arm balances).
  - Knee flexion: leave as is. The lotus clips (Cockerel -b, Scale -a) are dropped (TODO B1).
  - Record every changed range and limit in a JSON next to the XML.
- **PD gains.** Keep them unless a body's joint-space inertia changes by more than ~1.5x (the prototype's neck
  was 2.2x). Record any change.

**Acceptance.** All of these are MuJoCo or kinematic; there is no PhysX yet.
- Both XMLs load; total 74.00 kg; `pose_lib.extract_kinematic_info` gives the same dof names, and the same limits
  unless you changed them.
- Rest FK of the new XML reproduces the subject's SMPL-X joints to under 1e-5 m.
- **Zero rest-pose overlaps** among the 253 colliding pairs.
- On the 59-clip female-fit replay (`replay.mosh_kinematics(hand="fingers")` + `replay_metrics.py`),
  **self-overlap pair-frames > 1 cm are no worse than the shipped plant driven by the same motion**. That
  baseline is 80,058 (config C1 in `replay/results/summary.json`): the shipped plant on the shipped motion reads
  72,659, but her own motion overlaps more even on the old body, because soft tissue compresses where rigid
  colliders can't. No new overlap on a pair the human keeps apart (C1: 7,746 pair-frames).
- **Crowns:** head within 2 cm of the floor on all 27 crown holds, without grounding.
- **Floats:** labelled supports floating > 2 cm, grounded (+0.5 cm), ≤ 261/919 (the skeleton-only hybrid);
  target ≤ 200.
- **Gated statics** (`statics/run_statics_proto.py` pattern) on the grounded and 0-clearance poses: report
  LP-holdable counts against 41 (shipped plant, same motion) and 141–190 (the skin-fitted prototype).
- **COM vs mat COP** p50 ≤ 1.3 cm.

**Output.** The XML pair, the builder script, a `plant_v2.json` (offsets, colliders, masses, inertia comparison,
limit decisions, acceptance numbers) and a test file `tests/test_plant_v2.py` pinning those numbers.

**What Step 1 found (2026-09-30).** Built and accepted. Every number is in `data/reference_curation/plant_v2/plant_v2.json`
(`acceptance` carries the sha256 of the XML pair it measured). Rebuild with
`build_subject_plant_v2.py` (15 s, byte-identical), re-measure with `-m reference_curation.plant_v2` (50 s).
Read these before Steps 2 and 3:
- **Masses.** Her template at 1020.9 kg/m³ (72.48 L voxelised), cut by planes through the hip, knee, shoulder
  and elbow centres perpendicular to the distal bone. The crotch is split at the midsagittal plane: 0.29 kg of
  left-thigh-skinned tissue crosses it.
  - Pelvis 15.28 → 13.79 kg; thighs 9.14 / 8.88 kg (de Leva ×0.84 / 0.81). Tilting the hip plane anywhere from
    −20° to +45° moves ≤ 0.3 kg: her gluteal-pelvic volume sits above the hip joint.
  - Upper arms 2.53 / 2.42 kg (×1.34 / 1.28) are the only segments > 20 % off. The reason is recorded:
    collar-skinned axillary tissue lies lateral of the glenohumeral plane.
  - Whole-body inertia 1.02 / 1.03 / 0.91 of her template (shipped 1.17 / 1.16 / 1.15).
  - COM sits at the collider centroids, 1.6–5.8 cm from her voxel segments, which puts the whole-body COM
    2.5 cm low. **It does not matter**: COM vs mat COP p50 is 1.0 cm. No MassAPI authoring.
- **Neck sphere nudged.**
  - Re-centring the head on her cranium opens a 5.0 cm neck–head gap: her neck→head joint is 14.6 cm, against
    8.3 cm shipped. No position of the fixed-size neck sphere closes both that gap and the chest gap.
  - It sits on the chest–head centre line with equal gaps, 2.96 cm each.
  - At the shipped offset it overlaps her more medial collar spheres: 4,763 Neck–Thorax pair-frames on the
    corpus, 528 once nudged. Without the nudge the corpus overlaps are 84,178, above C1.
- **Joint box.** 16 decisions (`LIMIT_DECISIONS`), each with swing–twist evidence per side; the box stays mirror-symmetric.
  - **Widened:**
    - collar: protraction 60°, elevation 50°, axial −35°;
    - elbow x to 70°: SMPL-X stores part of the humeral rotation at the elbow, so the flexion plane is turned ~58°;
    - elbow twist to ±70°, elbow hyperextension to 20°;
    - knee x to 50° (the same effect for the femur), tibial rotation ±40°, knee extension stop −5° (the flexion stop stays 160°);
    - ankle plantarflexion 75°, neck extension −55°, MCP flexion 110°.
  - **Kept:** R_Toe_x (a 40–71° MTP roll, right foot only) and R_Ankle_x (right-foot inversion). Both are the
    fit's right-foot supination, which Step 3 corrects.
  - Hold exemplars past the box by > 2°: 282 → 100 of 300, 82 of them the right-toe roll.
- **Torque limits: wrist and hand raised 50 %, everything else kept.** At Step 2 the user decided on a safety
  margin (2026-09-30): wrist 20 → 30, hand 10 → 15 N·m (`TORQUE_DECISIONS` in the builder; the actuator `gear`
  follows `actuatorfrcrange`). With the shipped 20 / 10 they bound on 1 hold of her own motion (Scorpion -b@945,
  1.73×) and on 2 / 5 in the grounded arms. With 30 / 15, re-measured:
  - Scorpion -b@945 still binds, at 1.15× (R_Hand_y).
  - In the grounded arms 2 / 3 holds still bind, at ≤ 1.27×: Cockerel -b@2355, Firefly -b@639 and
    Bhujapidasana -a@543.
  - Peacock -a@788 (Chest_y) and Tree -a@804 (R_Toe_x) are bound by other joints.
  - LP-holdable 95 / 164 at +0.5 / 0 cm (was 95 / 162); her root 251.
  - The prototype's "7–14 beyond-plant holds" does not carry over.
- **PD gains.** Only the neck's joint-space inertia moves past 1.5×: it rises ×2.32 because the head now sits on
  the cranium. Step 2's robot config must scale the neck group's gains by that ratio (500 / 50 → 1158 / 116) to
  keep its 91 rad/s natural frequency. Every other group moves 0.75–1.11×.
- **What stays for Step 3** (measured on her own motion):
  - Grounded floats are 254, missing the 200 target. The top causes:
    - the right foot box sits below the left in stance (~40 supports): the right-foot supination dips its edge;
    - a hand is the lowest point: the wrist box sits ~16° off her flat palm.
  - Hand-touch frames sit at p50 −1.1 / −1.3 cm (her skin: −0.2 / 0.0), and 49 % of right-foot touch frames
    are below −0.5 cm. This is local penetration, not a grounding problem.
  - The 7 PELVIS supports float: the 9 cm pelvis sphere does not reach her buttocks, a consequence of the
    no-skin-fit decision.
  - Her motion's own acceleration spikes: 2 frames.
- **Traps.**
  - `statics.JOINT_RANGE` and `N_HINGE` are copied from the shipped plant at import. A plant swap must redirect
    them, or the stop columns use the old box. `mosh_replay.use_plant` does, even when statics is first imported
    inside the context.
  - Read the shipped ranges from MuJoCo's float64 model, not pose_lib's float32 parse. Otherwise every untouched
    range gets rewritten as `-30.000001`.
  - A capsule's MuJoCo `geom_size[1]` is its half-length and follows the bone. Only its radius is "unchanged".
  - The C1 baseline uses hand = "wrist". The same fingers motion on the shipped plant (S1) reads 79,557 overlaps,
    so v2 (79,943) is level with the same-motion baseline, not below it. Of v2's human-separated pair-frames,
    916 are (frame, pair) combinations C1 did not have (mostly feet together).

### Step 2 — Make plant v2 the training plant (USD, robot config, plant-identity plumbing)

**Goal.** The new plant loads in IsaacLab with the right mass, and no script can silently use the old one.

**Do.**
- **USD.**
  1. Flatten with `usd_convert/flatten_mjcf.py` (it must compile to the same model as `_v2_flat.xml`).
  2. Convert with `convert_robot_mjcf_to_usda.py`, not under `nohup`, and add the `worldBody` override.
  3. Run `apply_mjcf_masses_to_usd.py`.
  4. `check_simulated_mass.py` must report 74.00 kg, with every body within 0.1 % of the MJCF.
  5. Re-run `adoption_risks/usd_collider_check.py` on the new USD: every collision prim within 1e-3 mm / 1e-3°.
- **Robot config.** A new `protomotions/robot_configs/smpl_yogi_v2.py` registered in `factory.py`, with the v2 asset
  paths and `default_root_height` set to the new standing pelvis height above the lowest collider (measure it; the
  prototype was 0.976 m). Keep `smpl_yogi` for the old checkpoints.
- **Plant identity.**
  - Give every MJCF consumer a plant parameter: an env or config override for `reference_curation/ids.MJCF`,
    `static_hold_lp` FLAT and MJCF, a `--mjcf` flag in `build_physics_tables.py`, `--mjcf` passed to
    `add_onmat_gate_to_motions.py` from the corpus build scripts, and the other hard-coded scripts in §2.
  - Write the plant's sha256 into `physics_tables.pt` and every capture/statics record, and **fail loudly on a
    mismatch** in `PhysicsTables` and the loaders.
- **PhysX sanity**, with no policy:
  - a zero-action reset of a standing pose produces no depenetration launch;
  - the stored `rigid_body_pos` of a Step 3 motion equals the robot's FK to 1e-5 m.
  - The IsaacLab statue (`isaac_statue.py`) was never made to work (TODO D3). Do not block on it.

**Acceptance.** The mass check passes, the collider check passes, the reset is stable, and a deliberate plant
mismatch raises an error.

**What Step 2 found (2026-09-30).** Done and accepted.
- **Files and reruns.**
  - The package `data/assets/smpl/smpl_yogi03596_v2_usd/` is built by `data/scripts/build_plant_v2_usd.py`
    (77 s). Record: `data/reference_curation/plant_v2/usd_v2.json`, with every layer's sha256.
  - The robot is `smpl_yogi_v2` (`protomotions/robot_configs/smpl_yogi_v2.py`).
  - The PhysX side is `reference_curation.plant_v2_physx` (~2 min). Records: `physx_smpl_yogi_v2.json`, and
    `physx_smpl_yogi.json`, the same harness on v1.
  - Plant identity lives in `protomotions/utils/plant_identity.py`.
  - Tests: `protomotions/tests/test_plant_identity.py` (12) and
    `data/scripts/reference_curation/tests/test_plant_v2_step2.py` (10).
- **The converter wrapper cannot finish.** Isaac Sim's importer writes every layer in 16 s. It then hangs in
  `simulation_app.close()` indefinitely: > 10 min idle on a futex, and SIGTERM is ignored.
  `convert_robot_mjcf_to_usda.py` waits on it forever, so its post-processing never runs: the `_cleaned` rename,
  the `worldBody` override and the densities. The shipped conversion lost its post-processing at the same exit;
  there, under `nohup`, the parent died with the importer.
  - The builder runs the wrapper's own functions in order, and kills the importer's process group 60 s after
    its `Generated USD file:` line.
  - The visual-mesh patch is another Isaac Sim app. It is skipped because the MJCF has no mesh geom, which is
    checked.
- **One articulation root.** The importer authors `ArticulationRootAPI` + `PhysxArticulationAPI` on the empty
  `worldBody` Xform of the physics layer. The builder removes them there (MimicKit's rule), and the `.usda` also
  deactivates `worldBody`. The composed stage has one root, `/…/Pelvis/Pelvis`.
- **The USD is the MJCF** (every body, geom and joint):
  - colliders ≤ 8e-6 mm / 5e-6°, and the adoption verifier's own `usd_collider_check.py` re-run on v2 agrees;
  - body rest ≤ 5e-5 mm, joint frames ≤ 3e-5 mm, limits ≤ 7e-15°;
  - densities ≤ 5e-8 relative, no authored mass.
- **PhysX simulates it** (read from `root_physx_view`):
  - 74.000003 kg, every body within 2.6e-7 of the MJCF (the card asks 1e-3);
  - COM ≤ 8e-9 m, inertia ≤ 3.9e-7 relative;
  - dof limits ≤ 8e-8 rad, and max force exactly the MJCF's (wrist 30, hand 15 N·m);
  - stiffness, damping, armature and velocity limit exactly the robot config's (neck 1158 / 116).
  - `check_simulated_mass.py` passes (74.000 kg). It prints only grams, which alone puts the 0.11 kg hands
    0.2 % off, so the per-body check reads PhysX at full precision.
- **PhysX FK = the Step 3 writer's FK.** 418 frames of her female fit (300 holds + 2 transition frames per clip),
  clipped into the box (299 coordinates, ≤ 42.6°), went through `Simulator.reset_envs`.
  - PhysX's link positions equal pose_lib's float32 FK to 1.4e-6 m (p99 6.4e-7); the card asks 1e-5.
  - Rotations agree to 1.3e-4°, and the dof read-back is exact.
  - Trap: `2 acos|q1·q2|` reads float32 rounding as 0.05–0.08°. Use `2 atan2(|v|, |w|)` of the relative
    quaternion.
- **Reset: no launch.**
  - Default pose, zero action (the action offset of a 3-dof joint is 0, so this holds the rest pose): the lowest
    collider starts +0.5 mm up and settles 0.6 mm, rise 0.0.
    - v1's default reset (0.95 m) starts 4.36 cm in the air and lands at 1.37 m/s.
  - 136 two-feet standing holds of her fit, grounded + the training scripts' 5 mm respawn offset: none launches
    (overshoot 0.0, upward ≤ 0.07 m/s).
    - Without a policy they sag at training gains: median root −15.7 cm after 1 s.
    - 125 of the 136 start with self-overlapping colliders (her skin is thinner than the colliders: feet
      together, palms together). The overlap does not predict the first-step joint speed (r = −0.19).
- **In training's configuration PhysX does not launch a penetrating pose; it sinks it**
  (`max_depenetration_velocity` 1 m/s):
  - 1 or 2 cm into the floor: the pose is moved to the surface. The root rises 0.93 / 1.94 cm at ≤ 0.05 m/s,
    and nothing is thrown.
  - 5 cm in, as a v1 reference starts on v2: the 2.5 cm half-height foot boxes start entirely under the
    one-sided terrain mesh, and the body **falls through the floor** (root −9 cm after 1 s).
  - So the risk of a mismatched or sunk reference is sinking, not a launch.
  - The detector's positive control is therefore a root thrown up at 1 m/s (seen on 3 of 3 replicas), not a
    sink.
  - **For Step 3:** her raw fit's standing holds sit in the floor on v2 (lowest collider median −1.0 cm, worst
    −4.6 cm). The penetration-only floor correction is required, not cosmetic.
- **Plant identity.** A plant is its MJCF's sha256 (`plant_identity.PLANTS`: v1, v2).
  - The offline scripts take the plant from `REFERENCE_PLANT` (default v1, so the finished records keep their
    meaning): `ids.MJCF` / `MJCF_FLAT`, `static_hold_lp`'s `FLAT` / `MJCF`, the new `--mjcf` of
    `build_physics_tables.py`, and the `--mjcf` defaults of `add_onmat_gate_to_motions.py` and the §2 scripts.
    `build_expert60_ftR.sh` passes `--mjcf`.
  - Training takes the plant from the robot config (`plant_identity.robot_mjcf`).
  - `physics_tables.pt` carries `plant` and `plant_sha256`. `PhysicsTables`, `build_physics_tables` and
    `add_onmat_gate` raise `PlantMismatchError` on another plant.
  - A `.motion` carries `plant_sha256`, which Step 3's writer must write. MotionLib refuses a library that
    mixes plants, and `build_all_components` refuses a library built for another plant. Data without an
    identity counts as v1: accepted on v1, refused on v2.
  - Packed libraries keep the identity, and old packed libraries load as legacy.
  - Capture and statics records already carry the plant: the MJCF is a provenance input, and
    `ids.require_plant(record, what)` refuses them on v2 (tested on the stores).
  - **Not done here, and deliberately:** `capture.load` and the statics loaders are finished modules. Editing
    `capture.py` changes its source hash, which is the records' generator id. Every record would then be rebuilt
    and the audit id would move.
    - **Step 4's new store versions must load through `ids.require_plant`.**
    - Until then, under plant v2, use the finished Step 1–8 modules only for their functions on poses and
      geometry. Never run their CLIs and pipelines, and never read the avatar-side fields of their stores: those
      fields are v1 records.
      - Usable functions: `retarget` FK, gaps, the box-box penetration depth (`_box_box_sat`) and
        `nearest_representative`; the statics LP.
      - Avatar-side fields: capture `avatar_min_z` and attribution, statics, and the labels' avatar evidence.
      - Under `REFERENCE_PLANT=v2`, `capture.load` hands the v1 records back silently.
      - Human-side fields (markers, mat, the human mesh) are plant-free.
- **Robot config.**
  - `default_root_height` 0.975 m: the rest pelvis sits 0.9745 m above the lowest collider.
  - Neck gains ×2.3154 → 1158 / 116.
  - Everything else is `smpl_yogi`'s.
- **Traps.**
  - IsaacLab's `modify_collision_properties` cannot edit the instanced collider prims (it warns), so training's
    `contact_offset` / `rest_offset` never reach the robot's colliders. This happens identically on v1, and was
    not changed here.
  - PhysX ignores the MJCF joint `stiffness` imported onto the joints (610 warnings, the same on v1). The drives
    get the robot config's gains, as measured above.

### Step 3 — Regenerate the references on plant v2 from the female MoSh fit

**Goal.** New `.motion` files: her own motion on her own skeleton, inside the joint box, with supports realised,
not sliding, and no jerk.

**Do.**
1. **Writer.** Female fit → `.motion`, via `replay.mosh_kinematics(hand="fingers")` → qpos → pose_lib FK.
   - 60 Hz; root = her pelvis (trans + J0 + (0, 0, 10.17 mm) + `VICON_TO_CLIP_XY`); no `fix_height` and no
     per-frame grounding.
   - Geometric `rigid_body_contacts` (lowest surface ≤ 1 cm).
   - Drop Standing big toe hold -c (no fit) and the lotus clips (TODO B1).
2. **A small retarget on plant v2.** This is a new version of `retarget.py`'s solver (a new module or version, not
   edits to the Step 8 module), anchored to the writer's motion. It must satisfy **the joint box as a hard bound**
   and fold in the open TODO fixes:
   - **A1, no slide:** a velocity-matching term that keeps each planted point's horizontal motion equal to the
     female fit's own. Measure slide with `build_physics_tables.zone_patch_speed`, which reproduces TODO A1's
     numbers exactly.
   - **A2, balance margin:** tied to the support polygon, and active on every quasi-static frame, not only in
     hold windows.
   - **A3, no jerk:** continuation on all contact terms.
   - **Penetration-only floor correction:** lift a sunk support to the floor locally; never ground the whole body.
   - **Foot/palm orientation:** a flatness term only where her sole or palm is flat (supination, §1 item 2).
3. Run the retarget exit metrics (`retarget.clip_metrics` / `replay_metrics.py`).

**Carried in from Steps 1–2** (details in their "What Step N found" notes):
- **Select plant v2 explicitly.** The default is still v1.
  - Use `REFERENCE_PLANT=v2` (it sets `ids.MJCF` and `static_hold_lp` at import) or `mosh_replay.use_plant()`
    in-process.
  - Pass `xml_path=mosh_replay.V2_XML` to `mosh_kinematics`. Its default skeleton and its `nearest` joint box
    come from `mosh_replay.SHIPPED_XML`, which is bound to `ids.MJCF` at import, so `use_plant()` does not move it.
  - Never read avatar-side fields of the finished stores (Step 2 notes, "Plant identity").
- **The writer.**
  - Write `plant_sha256 = plant_identity.sha256("v2")` into every `.motion`. Without it, training on
    `smpl_yogi_v2` refuses the library.
  - Step 2 validated the FK path: `plant_v2_physx.poselib_fk` (qpos = root, root quat wxyz, 69 exp-map dofs,
    then pose_lib in float32) equals PhysX's FK to 1.4e-6 m. Copy its few lines; don't import it from a CPU
    module, because at import it parses argv and imports IsaacLab.
  - Step 2 tested the card's "stored `rigid_body_pos` = robot FK to 1e-5 m" on Step-3-style frames, because no
    Step 3 file existed. Re-run that check on the written `.motion` files: load, `reset_envs`, compare.
- **The floor correction is required.**
  - Her raw fit's standing holds sit in the floor on v2: lowest collider median −1.0 cm, worst −4.6 cm.
  - In training's PhysX, a pose 1 or 2 cm in is lifted to the surface, and one 5 cm in falls through. Nothing
    in between was measured; the foot box's centre crosses the surface at about 2.5 cm.
- **The joint box.**
  - 100 of 300 hold exemplars still exceed v2's box by more than 2°. 82 of them are the right-toe roll, kept on
    purpose: it is the right-foot supination this step's flatness term corrects.
  - On 418 frames, `nearest_representative` plus clipping clipped 299 coordinates, by up to 42.6°.
- **Step 1's "What stays for Step 3":**
  - grounded floats 254: right-foot supination, and the wrist box 16° off the palm;
  - hand-touch frames at p50 −1.1 / −1.3 cm;
  - the 7 PELVIS supports float, because the pelvis sphere does not reach her buttocks;
  - her motion's own acceleration spikes: 2 frames.
- **The robot.** `smpl_yogi_v2`: `default_root_height` 0.975 m; the training scripts' `ref_respawn_offset` is
  0.005.

**Acceptance** (on the new motions):
- 0 frames more than 1° past the joint box.
- Labelled floats > 2 cm ≤ 5 % (Step 8's exit test); crowns 0/27.
- Touch-frame penetration below −0.5 cm no worse than the skin's own 15.6 %.
- Slide clip p90 ≤ 4.5 cm/s and ≤ 10 % of touch frames above 5 cm/s.
- No acceleration spikes beyond the female fit's own (2 frames corpus-wide).
- No new overlaps > 1 cm.
- Body edit p50 well under Step 8's (3.6 cm median, 15.5 cm worst body per clip).

**What Step 3 found (2026-09-30).** Built; 9 of the 10 acceptance tests met once the user dropped Firefly -b, and the
one miss below is open. Every number is in `data/reference_curation/retarget_v2/<rid>/` (`retarget.json`,
`clips.jsonl`, `summary.md`, `physx.json`), rid `holds_repaired_ftC_posefix.labels_v1_1.3b3fc75828.retarget_v2.e4278ada02`.
- **Files and reruns.**
  - The writer is `reference_curation/fit_writer.py`. It writes her female fit on plant v2, unchanged, to
    `output/reference_curation/fit_writer/fit_v2.v1.b8f5e86fc9/` (56 clips in 3 s). Record:
    `data/reference_curation/fit_writer/<fit_id>/fit.json`. It is the converter's path in float32, with no
    `fix_height`, geometric contacts and `plant_sha256`; the stored bodies equal her SMPL-X joints to 5e-7 m.
  - The retarget is `reference_curation/retarget_v2.py`, motions and lineage in
    `output/reference_curation/retarget_v2/<rid>/` (14 min on 16 workers: `--all --workers 16`). It reuses Step 8's
    kernels inside `on_plant()` (`mosh_replay.use_plant`); `retarget.py` is unchanged.
  - The PhysX check is `reference_curation/retarget_v2_physx.py` (an IsaacLab process, about 25 s).
  - Tests: `tests/test_retarget_v2.py` (11, about 1.5 min).
  - Corpus: 56 clips, 81,512 frames. Dropped: Cockerel -b and Scale -a (B1), Standing big toe hold -c (B2), and
    Firefly -b (the user, after the first run; below).
- **Acceptance** (her fit on plant v2 → the retarget):

  | Test | Fit | Retarget | Limit | |
  |---|---|---|---|---|
  | Frames > 1° past the joint box | 26,532 | 0 | 0 | met |
  | Labelled supports > 2 cm (labels v1.1, 864) | 48 (5.6 %) | 1 (0.1 %) | ≤ 5 % | met |
  | Crowns floating > 2 cm (21 on these clips; the card's other 6 are Cockerel -b's) | 0 | 0 | 0 | met |
  | Touch zone-frames < −0.5 cm | 43.0 % | 0.01 % | her skin 15.3 % | met |
  | Slide: clip p90 / frames > 5 cm/s | 2.96 / 6.7 % | 2.99 cm/s / 6.9 % | 4.5 / 10 % | met |
  | Frames > 100 m/s² that her fit has not | — | 0 | 0 | met |
  | New body-pair overlaps > 1 cm | — | 0 | 0 | met |
  | Mean body displacement, clip p50 | — | 1.37 cm | Step 8 3.61 | met |
  | Worst body per clip, p50 | — | 10.6 cm (p25 8.8, p75 12.6) | Step 8 15.5 | **missed** (bar: half) |

  Also: body-pair overlaps > 1 cm 63,201 → 604; balance frames with the COM outside the targeted supports
  3,026 → 40; COM vs mat COP p50 1.02 → 1.10 cm (208 holds); resting-face tilt (clip p50) 14.0° → 0.28°.
- **PhysX** (`physx.json`):
  - all 56 motions load on `smpl_yogi_v2`, and a plant v1 library raises `PlantMismatchError`;
  - stored `rigid_body_pos` = PhysX FK to 1.2e-6 m on 701 frames (every exemplar + 8 per clip);
  - 0 of 701 resets launch, and a root thrown at 1 m/s is caught on 3 of 3 replicas.
- **Firefly -b, dropped.** The first run (57 clips) missed a second test on it alone. It asks for 24.9° of lumbar
  flexion past the plant's box (Torso_y to 115°, box 90°). The fold must come from the spine joints above, which
  deepens the pelvis–spine sphere overlap. With her thighs wedged 5.5–8.5 cm into her upper arms, every variant
  either jerked (42–56 frames > 100 m/s²) or, after the local pass, kept 14–22 new overlaps (≤ 3 cm); the kept
  solution's legs sat > 10 cm from hers for the whole balance. The user dropped it like the lotus clips. The
  56-clip rerun reproduces the other 56 motions and lineages byte for byte; the 57-clip outputs are deleted.
- **The miss: worst body p50 10.6 cm** vs Step 8's 15.5. The mean displacement is well under (1.37 vs 3.61 cm). The
  per-clip maximum is a single body on a single frame. The worst body per clip is a knee/hip in 25 clips,
  an elbow/wrist/hand in 17, a toe/ankle in 11, the head/trunk in 3. Its sources:
  - the soft supports, physically required (below), 12–22 cm in the 7 supine/prone/seated clips;
  - poses past the box (`summary.md` lists 5): Peacock -a's elbows (15.1 cm), Upward Plank -a's elbow and
    Plow -b's hand (16–17 cm, both soft-support clips too); Lord of the Dance -c and Intense Side Stretch -b
    stay under 7 cm;
  - palm loading and unloading, 8–11 cm at the elbow: a palm going flat turns the forearm. A fixed rotation of
    the wrist box would not remove it (measured after the run on every palm-flat frame):
    - the correction the retarget applies is 15–16° at the wrist box and 14–17° at the finger box (p50);
    - only ~10° of it is a consistent rotation (12° at R_Hand), about different axes on the two sides;
    - the frames scatter 11–12° (p50) around it.

    So a plant change (Steps 1–3 again) would buy only the consistent third of the edit. Not recommended.
- **What was decided, with the measurement behind each** (the rules are in the module doc):
  - **Supports are realised, soft ones included.** Seated, her buttocks reach 11 cm below the 9 cm pelvis
    sphere centred on her hip joint: Upward Plank's sphere bottom is at 12.9 cm while her skin is at 1.8 cm.
    The trunk spheres sit 4.5–6.6 cm up in Plow, Bridge and Shoulderstand. Step 1's statics on her raw fit calls
    every such hold `support_not_realised`, so the plant sits and lies lower than she does there.
  - **Body–body requests are closed only within 2 cm of her fit** (104 closed, 60 reported as
    `geometry_incompatible`). Her requested gaps on plant v2 are bimodal:
    - pressed contacts overlap 4–6 cm (thighs into upper arms; Tree's sole into the thigh);
    - thigh–trunk contacts sit 4–9 cm apart (the trunk spheres do not fill her belly).

    Closing a 3 cm one moved Tree -a's knee 15 cm. Which contacts are critical is B6 (Step 4).
  - **The guard against self-penetration:**
    - no pair the plant collides overlaps by more than 1 mm (Step 8 allowed 5 mm);
    - where her mesh shows the two zones in contact, the pair may not get deeper than in her fit, and
      `pen_soft` (weight 10, 1 cm scale) asks it apart.

    Tried and rejected:
    - a full guard slid Firefly -b's legs to the other side of the arms (50 jerk frames, a toe 44 cm);
    - a fixed 2 cm push summed to a 19 cm toe over Eagle's wrap;
    - `pen_soft` at weight 1 left Tree's sole 1.5 cm in the thigh.
  - **A face rests flat only while all four of its corners touch.** Step 8's spread rule held a 7 cm palm flat
    through a 30° peel (Crow's elbow 10 cm). The new rule qualifies planted palms on 94–98 % of their down
    frames.
  - **The toe's resting face is chosen with her toe roll removed.** Otherwise the supinated right toe put its
    side face down (Warrior II's ankle swung 14 cm). A sole-only rule broke Cobra (dorsum) and Side Plank (side).
  - **Balance (A2)** applies on quasi-static frames whose targeted support zones are unchanged for ±0.25 s, with a
    margin of max(1.5 cm, 25 % of the inradius). Without the stability rule a slow weight shift dragged Garland's
    pelvis 8 cm.
  - **Jerk (A3)** runs direct, then gentle (every contact term continued), then local (smoothness ×10, ×30,
    ×100 within 0.25 s of a remaining spike). Only Firefly -b needed it; every one of the 56 clips is solved
    by the direct pass.
  - **The floor contract is −1 cm.** Cobra's straight prone leg cannot rest thigh, shin and dorsum each at
    +5 mm; the solve leaves 0.53 cm, and a 10× stiffer floor floated both thighs 2.5 cm on 530 frames instead.
- **Traps.**
  - **BLAS threads make `solveh_banded` 1000× slower**: 123 s against 0.11 s on a 1,212-frame clip. Run the solve
    in spawned single-threaded workers (`run`). Also, `pkill -f <name>` matches the shell running it.
  - The per-row Gram (`retarget._gram`) cost 3.4 s a step. A per-frame batched matmul (`_gram_rows`) costs 0.35 s.
  - **Step 2's reset rule flags falls.** A single-leg pose without a policy falls: Tree's root dropped 84 cm and
    bounced at 0.47 s, up to 0.87 m/s, on the unedited fit too (1.63 m/s). The velocity test now covers only the
    first 0.1 s, with a positive control.
  - `human_evidence` refuses to load the capture stores under any plant but v1. A stale store would be rebuilt
    with v2 avatar fields into the v1 records. Never run this module with `REFERENCE_PLANT=v2`.
- **For Step 4.**
  - Pass C compares `fit_writer/<fit_id>/` (unedited) with `retarget_v2/<rid>/`.
  - `clips.jsonl` carries, per clip:
    - `fit_box_excess_deg` (every coordinate past the box);
    - the pair requests' status and gaps (the 60 `geometry_incompatible` feed B6);
    - the edit, the guard's soft pair-frames and the solver ladder.
  - The lineage has the targets, flats and balance frames.

### Step 4 — Re-run the offline reviewer on plant v2 and strengthen the labels

**Goal.** Labels v2 and a release gate v2 built on the new plant and references. The known label weaknesses are
closed, or explicitly carried as masks.

**Human-side evidence does not change and is reused:** capture markers and mat measurements (store v1 human and mat
arrays), the human mesh (store v2), seam arbitration (store v3), and the human-side decisions of labels v1 (ground
sets, windows, exemplars, Tier-1 load paths).
**Everything that measures the avatar must be redone.**

**Do, in order.** Each item is a new version of the module or record; don't edit the finished Step 1–8 modules.
1. **Capture store (avatar side).** A new store version whose `avatar_min_z`, `attr_visible` and attribution are
   measured on plant v2 and the Step 3 motions.
   - **Re-run the pressure attribution on the new geometry** (`attribute_pressure_to_bodies.py` →
     `add_pressure_to_motions.py` → `add_onmat_gate_to_motions.py --mjcf <v2>`, TODO C3).
   - Re-derive `required_support` vs `required_touch` from the new attribution. The 445 mat-based roles in labels
     v1 were measured on the floating shipped avatar.
2. **Audit v2** on labels v1's manifest and the new store: the float, missed-touch and phantom metrics on the new
   references.
3. **Render and Pass A recalibration.** The avatar changed, so the render_v3 admissions (float, floor contact,
   left/right) must be re-measured on a new `render_v`. About $7 for the 52-packet calibration at high effort. Fold
   in the known render fixes:
   - distinct shades per segment (upper arm vs forearm, thigh vs shin);
   - Pass C drop lines only under zones the human touches (TODO D2, render_c2, ~$15 of controls).
4. **Statics and statue on plant v2** (the gated LP and witness with the plant parameter from Step 2). Relax the
   statue's contact rule to loaded zones (TODO D4).
5. **Labels v2 = labels v1's human-side decisions + plant v2 evidence.**
   - Recompute each annotation's avatar evidence and statics block.
   - Re-derive the statics promotions (labels v1.1 promoted 63 `required_touch` supports).
   - Keep hold ids stable.
6. **Strengthen the labels.** These are the open label problems from Steps 6–8; the rule is minimal human input,
   so use machine evidence only.
   - **Critical body–body contacts (TODO B6).** Run the reviewer's self-consistency route: several independent
     Pass B samples per hold plus agreement across takes. Admit a critical-contact class only if it meets the
     Step 4 rule. Also review the ~30 holds with body–body contacts that never had a Pass B verdict (21 unflagged
     + 9 hover-only).
   - **Unresolved and short `_h` holds.** Re-decide the 9 unresolved holds, and the `_h` preparation holds that
     Pass B called "not the labelled pose" (34 of 35).
   - **any_of groups.** Make them non-transitive, so a union-find chain cannot merge a whole leg wrap into
     "any one".
   - **Demanded pairs.** Measure them over the window, not only at the exemplar: 11 of 196 `required_touch` pairs
     touch on fewer than 90 % of their window.
7. **Pass C on the Step 3 edits.** Compare them with the unedited female-fit motion on plant v2, under the grey
   mesh. Re-run the controls first (~$15), then the edits (~$0.13 each).
8. **Gate v2.**
   - Pin the gate id and the Pass C retarget id; don't read the newest folder.
   - Check that each Pass C verdict was given on the current retarget.
   - Add the statue's motion verdict as a flag for single-leg holds (it caught 5 of the 6 single-leg holds ft_c
     failed).
   - Gate on the joint box, floor minimum and collision scan as well as Pass C.

**Carried in from Step 3** (details in "What Step 3 found"):
- **Firefly -b is dropped** (the user's decision). Every store and label set below covers the 56 clips of
  `fit_writer.corpus()`.
- **Inputs, by id; never read the newest folder.**
  - References: `output/reference_curation/retarget_v2/…retarget_v2.e4278ada02/` (`plant_sha256` = v2).
  - The unedited fit, for Pass C and any "before": `output/reference_curation/fit_writer/fit_v2.v1.b8f5e86fc9/`.
  - Per-clip records: `data/reference_curation/retarget_v2/<rid>/clips.jsonl`.
  - Labels: v1.1 for the human-side decisions.
- **The finished modules stay finished.** Capture, audit, statics, labels and gate each get a new version that:
  - calls their functions inside `retarget_v2.on_plant()` (or `mosh_replay.use_plant()`);
  - reads human-side arrays the way `retarget_v2.human_evidence` does: outside the plant context, and refusing
    under v2;
  - writes records that verify through `ids.require_plant`.

  Never run the Step 1–8 CLIs with `REFERENCE_PLANT=v2`.
- **The pressure attribution chain:**
  1. `attribute_pressure_to_bodies.py --motion-dir output/reference_curation/retarget_v2/<rid> --mjcf
     data/assets/smpl/smpl_yogi03596_v2.xml`;
  2. `add_pressure_to_motions.py --bodies-dir …`;
  3. `add_onmat_gate_to_motions.py --mjcf <v2>`.
- **What changed under the labels.** Re-derive each of these; do not carry them over:
  - **Crowns are realised.** All 21 heads are within 2 cm, and the head refusal is gone. Gate v1's 9 crown masks
    and 17 crown exclusions came from the old head collider.
  - **The soft supports are realised.** The plant sits up to 11 cm lower than she does where her buttocks or
    back carry her. Plow, Bridge, Shoulderstand, Legs-up and Upward Plank holds should leave
    `support_not_realised`; statics v2 must confirm it.
  - **Body–body.** 104 requests are closed within 1 cm. 60 are `geometry_incompatible` (`pair_requests` in
    `clips.jsonl`): pressed 4–6 cm into a limb, or 4–9 cm from the trunk spheres. These are B6's starting list.
    Unless the critical-contact route admits one, labels v2 must carry it as not realised (a mask).
  - **Overlaps.** 604 pair-frames over 1 cm remain, all on pairs her mesh shows touching (no deeper than her
    fit). The gate's collision scan must count *new* overlaps against the fit, as Step 3's acceptance does;
    counting every overlap would drop most clips with a bind or a leg on an arm.
- **Pass C.** Step 3 had to edit where the plant cannot do what she does: the soft supports and the five poses
  past the box (`summary.md`). There "the fit is closer" is expected and physically required, so make it a flag,
  not an exclusion, unless the reviewer calls a different pose.
- **PhysX reset checks.** Use `retarget_v2_physx.py`'s rule: velocity only in the first 0.1 s, plus the
  thrown-root control. A passive single-leg pose falls (Tree), and Step 2's full-window rule calls that a launch.
  The MuJoCo statue stays the witness; the IsaacLab statue is still unusable (D3).

**Acceptance.**
- 0 body–ground disagreement with the capture, as in labels v1.
- Every annotation cites evidence ids that verify by sha256.
- The calibration replay tests pass for every new table.
- Every hold's commanded ground set equals labels v2's configured supports minus the gate's masks (TODO A4).
- Gate counts reported; every exclusion has a reason.

**What Step 4 found (2026-10-01).** Built; all five acceptance tests met.
- Labels v2 is `holds_repaired_ftC_posefix.labels_v2.e03994ad3c` (280 holds, 56 clips).
- Gate v2 is `holds_repaired_ftC_posefix.gate_v2.b2f76f2606`: 270 of 280 holds released (204 pass, 65 flagged,
  1 masked) and 10 excluded.
- Reviewer spend $106.94 against the plan's ~$90, on 673 calls, all valid:
  - Pass A $13.83 (101 packets), against ~$7 planned: the float and left/right material added packets;
  - B6 $45.53 (93 holds × 2 samples), against $25–40 planned;
  - Pass C $47.58 (114 controls $14.46, 272 edits $33.12), within plan.

- **Records, in build order.** Each record names its inputs by id. Nothing reads "the newest folder".

  | Item | Module (`reference_curation/`) | Record (under `data/reference_curation/` unless noted) |
  |---|---|---|
  | 1 mat attribution on v2 | `pressure_v2` | `pressure_v2/<rid>/pressure.json`; ports in `output/reference_curation/pressure_v2/<rid>/` |
  | 1 capture store v4 | `capture_v4` | `output/reference_curation/capture/v4/`; `calibration/capture_v4_crosscheck.json` |
  | 2 audit v2 | `audit_v2` | `audits/…labels_v1_1.3b3fc75828.audit_v2.calibrated.9278736bb1` (marker rule), `….sources.dafada25ad` (the labels' own rule) |
  | 3 render v4, Pass A | `render_v4`, `packets_v4`, `calibrate_v4` | `calibration/pass_a_v4/render_v4.json` |
  | 4 statics, statue | `statics_v2`, `witness_v2` | `statics_v2/`, one folder per configuration: labels v1.1's `f57d85d3fa`, B6's critical-only `fb432fa15d`, the final `aab60189f4` |
  | 5–6 labels v2 | `labels_v2`, `b6`, `labels_v2_restore` | `labels/…labels_v2.*`: draft `74a61979cf` (B6's base), pass 1 `b61fad7e9d` (B6's critical pairs), pass 2 `46024365fa` (plus the restoration), **final `e03994ad3c`** |
  | 6 B6 | `b6` | `b6/…labels_v2.74a61979cf.b6.590f05ccca/`; Pass B table `calibration/pass_b/render_v4.json` |
  | 7 Pass C | `edits_v2` | `calibration/pass_c/render_c2.json`; packets in `output/reference_curation/packets/render_c2/` |
  | 8 gate v2 | `gate_v2` | `gate_v2/holds_repaired_ftC_posefix.gate_v2.b2f76f2606/` (`release_holds.yaml`, `holds.jsonl`, `summary.md`) |

  Tests: 59 in `test_step4_{evidence,rules,review,final}.py` (~30 s). Run them with `OMP_NUM_THREADS=1`.
- **Rebuild, in order.** Each command is `PYTHONPATH=.:data/scripts OMP_NUM_THREADS=1 ../env_isaaclab/bin/python -m
  reference_curation.<module> …`. Never set `REFERENCE_PLANT=v2`.
  - The machine steps are deterministic and reproduce the ids above.
  - The review steps replay the ledger: a packet with a verdict on file is never sent again.

  1. `pressure_v2`, then `capture_v4 --all` (it also writes the crosscheck), then `audit_v2` and
     `audit_v2 --rule sources`.
  2. `statics_v2` (labels v1.1's configuration), then `labels_v2` (the draft).
  3. `calibrate_v4 --build`, then `--review --max-cost 15`, then `--calibrate` (Pass A).
  4. `b6 --labels <draft> --build`, then `--review`, then `--decide`, then `--calibrate` (the Pass B table). Pass
     `--labels` every time, because the default is the oldest labels v2 folder by mtime.
  5. Pass 1: `labels_v2 --critical <b6.json> --votes <b6.json>`, then `statics_v2 --labels <pass 1>`.
  6. Pass 2: `labels_v2_restore --critical <b6.json> --votes <b6.json> --critical-statics <pass 1's statics>
     --statics <pass 1's statics>`, then `statics_v2 --labels <pass 2>`.
  7. The final labels: the same `labels_v2_restore` command with `--statics <pass 2's statics>`.
  8. `edits_v2 --labels <final> --build-controls 30`, then `--build-edits`, then
     `--review --kind identity swap lift sink kink --max-cost 16`, then `--review --kind edit --max-cost 40`, then
     `--calibrate`.
  9. `gate_v2 --labels <final>`.
- **Acceptance.**

  | Test | Result | |
  |---|---|---|
  | 0 body–ground disagreement with the capture | 0 on 280 holds, at every exemplar and on every window frame | met |
  | Every annotation cites evidence ids that verify by sha256 | 1,678 annotations, 613 evidence rows, 0 failures | met |
  | Calibration replay for every new table | Pass A v4, Pass B v4, B6 and render_c2 rebuilt from the ledger equal the stored tables | met |
  | A4: commanded ground set = configured supports − masks | 0 problems on the 270 released holds; the 10 excluded holds are absent | met |
  | Gate counts reported; every exclusion has a reason | the table below; every reason names a rule | met |
- **The plant-v2 evidence (items 1–4).**
  - **Mat attribution.** The chain was re-run on the Step 3 motions.
    - The reference explains ≥ 90 % of the measured load on 99.9 % of 81,140 loaded frames. ftR's reference did
      so on 89.5 %, the shipped port's pose on 71.2 %.
    - COP residual p90 is 0.16 cm (ftR 3.14, shipped 29.5).
    - All 56 clips now carry the on-mat gate column, and usable frames rose from 76 % to 89 %.
  - **Capture store v4.**
    - The plant-free mat fields equal capture v1's exactly (difference 0.0). `attr_visible` changed on 134,945
      zone-frames.
    - Re-derived on the new attribution, every human-side touch threshold moves < 0.1 cm: feet 4.36 / 4.26 cm,
      hands 2.43 / 2.47 cm, head 3.64 cm, mesh 2.25 cm.
    - **The head's marker test no longer passes its admission rule.** With the crowns realised, 2,379 frames of
      10 clips confirm it (v1: 911 frames of 3 clips). Their spread (p99 − p50 2.1 cm) exceeds the rule's 1 cm.
      The threshold itself holds. The committed calibration is human-side, so no label moves.
  - **Audit v2 on labels v1.1.**
    - Labelled supports floating > 2 cm: 434 → 1 of 864 (Triangle -b@669's right hand, 2.9 cm).
    - Holds with unexplained mat load: 33 → 0.
    - Under the labels' own evidence rule: 0 missed touches and 0 phantom supports.
    - Under the marker rule: the same 1 + 1 as on v1 (Bridge -a@652's right hand, Peacock -a@1183's head). These
      are human-side disagreements, not plant ones.
    - The 8 `exemplar_unstable` holds are the 8 transitions (below).
  - **Statics v2 and the statue.** Statics is the gated LP on the reference with plant v2's torques. The statue
    (`witness_v2`) runs at plant v2's training gains: neck 1158 / 116, wrists 30 N·m, hands 15 N·m.
    - On labels v1.1's configuration, 276 of 280 holds are holdable. For comparison: Step 8 on plant v1, 270 of
      300; the shipped references, 38.
    - 3 are beyond the plant: Scorpion -b's wrists (two holds), and Peacock -a@1183's spine, whose configured
      PELVIS braces are not realised.
    - 1 is `support_not_realised`: Triangle -b@669's right hand cannot reach the floor. The hold is held without it
      (s* 0.27), so the gate masks the support.
    - **D4** requires only the zones the LP loads to stay down. Under it, the statue passes 194 of 276 holds
      against 180 under Step 7's rule. The 14 it adds are forearm poses whose unloaded hands rise. 231 of 276 stay
      still (settle < 5 cm, drift < 2 cm).
    - On the final configuration: 277 of 280 holdable, only Scorpion -b@945 and @1163 beyond the plant; the statue
      passes 198 of 277 (183 under the strict rule, 231 still).
    - **Step 3's predictions hold.** No Plow, Bridge, Shoulderstand, Legs-up or Upward Plank hold is
      `support_not_realised` any more. No crown mask or exclusion survives; gate v1 had 9 and 17.
- **Render v4 and Pass A (item 3).**
  - Every segment has its own shade: upper arm purple, forearm orange, hand red, thigh yellow, shin green, foot
    blue, head magenta. Neighbouring segments differ by > 20° of hue. Torso and pelvis share one grey.
  - 101 packets at high effort, $13.83.
  - Admitted as evidence:
    - floor contact, avatar (0.997);
    - floor clearance (1.0);
    - floor contact, human (1.0);
    - float (1.0 on 95 claims, recall 1.0);
    - left / right (1.0).
  - Advisory: body–body (1.0 on 12 claims, recall 0.11) and pose identity (0.525).
  - **Floats are rare on v2** (1 of 864 supports), so the float class needed new material:
    - 24 packets of the reference lifted 2.5–12 cm;
    - 17 of her unedited fit's natural 2–3 cm floats.

    Recall is 51/51 at 2–5 cm, 43/43 at 5–15 cm and 27/27 on the fit.
  - Left/right needed 12 side-asymmetric packets; before them it had only 25 claims.
- **Pass B v4 and B6 (item 6).**
  - **The run.**
    - 93 holds: 78 with a candidate critical pair and 15 with an undecided pose role, two independent samples each.
    - The Pass B table admits floor contact, floor clearance, human floor, body–body (1.0 on 162 claims, recall
      0.84) and left/right.
    - Float stays advisory because nothing floats to claim. Pose identity (0.81) and the roles stay advisory.
  - **Self-consistency, the Step 4 rule.**
    - Another sample repeats 126 of 135 `required_touch` claims (0.933; 95 % low 0.878) and 842 of 851 other
      claims (0.989), over 493 items, all answered. **Admitted.**
    - Across takes (mirrored sides, or the same pose in another clip), a `required_touch` consensus is repeated
      6 of 7 times.
    - render_v3's single sample agrees with the v4 consensus on 289 of 311 roles.
  - **The configured body–body pairs are now the critical ones: 59 in 44 holds** (labels v1.1 configured 138).
    They are the contacts the poses are named for:
    - Crow's and Side Crow's shins and thighs on the upper arms;
    - Firefly's and Koundinya's thigh braces;
    - Tree's sole on the thigh, and Eagle's wrap;
    - the headstand's interlaced hands, the shoulderstand's hands on the back;
    - Peacock's forearms in the belly.
  - **Step 3's 60 `geometry_incompatible` pairs:**
    - 48 unrealised and 6 realised pairs are demoted (not critical);
    - 3 were never configured;
    - 1 is critical and realised at the exemplar (Shoulderstand -a@914, TRUNK+L_HAND);
    - 2 come back through the restoration below.

    **No configured pair is unrealised (0 of 63), so the gate masks no pair.**
  - **Caveat:** reproducibility is reliability, not validity. A reviewer can repeat a wrong judgement, and by design
    no human truth set exists.
- **The statics restoration (`labels_v2_restore`, new).**
  - **The problem.** Statics on B6's critical-only configuration found one hold newly beyond the plant:
    Peacock -a@788 (s* 0.147 → 1.192, its spine at 119 %).
    - Both samples say the forearms carry her. One places them on TRUNK, the other on PELVIS.
    - **render_v4 draws torso and pelvis in one grey**, so the zone is a guess. Neither pair reproduces, and all
      four were demoted. This is the only such split among the 93 holds.
    - A single-pair necessity test cannot see it either: each brace alone is only `useful`.
  - **The rule.** A hold's demoted pairs come back when both of these hold:
    - statics on the critical-only configuration finds the hold beyond the plant;
    - statics on labels v1.1's configuration finds it holdable.

    Both records are checked to request exactly those configurations at the hold.
  - **The result.** It restores Peacock -a@788's four braces and nothing else.
    - Statics on the restored configuration reproduces the hold's v1.1 row exactly (feasible, 0.1467).
    - It leaves the other 279 rows unchanged.
    - The restored pairs are configured `required_touch`, but not critical.
  - **Why a new module.** `labels_v2.py` is not edited, because its sha256 is the `rule:` evidence of the draft and
    pass 1. The new module wraps `labels_v2.build` instead.
- **The other label decisions (items 5–6), on 280 holds.**
  - **Transitions.** Labels v1.1's 8 unresolved holds change their support set every 0.3–0.9 s and never keep one
    for 1 s. Their status is `transition`, and the gate excludes them.
  - **Windows are where the configuration holds.** 10 are trimmed; Tree's sole lands on the thigh up to 1.8 s after
    the foot leaves the floor. Configured pairs held on < 90 % of their window: 11 → 0.
  - **`any_of` groups are pairwise.** Two configured pairs form a group when they share a zone and their other
    zones are adjacent segments of one limb; there are no union-find chains. The draft had 49 groups; the critical
    configuration has none.
  - **Ground roles on the plant-v2 mat.**
    - `required_support` 473 → 522 and `required_touch` 391 → 342 (59 touch → support, 10 support → touch).
    - 455 come from the mat (≥ 50 N) and 64 from statics (labels v1.1: 63, from Step 8's statics).
  - **Pose roles of the 103 `_h` holds.**
    - variant 53, preparation 35, undecided 8, family 7.
    - The role is the majority of every valid Pass B sample (1–3 per hold). `preparation` also needs a ground set
      that no family hold of the clip has.
    - Of the 30 holds that render_v3's single sample called "not the labelled pose", 26 are preparations.
    - The names are kept, because they are node identity downstream.
  - **Orientation bins**, recomputed on the reference: 2 change (Plow -a@756, Plow -b@1644).
- **Pass C on the Step 3 edits (item 7).** render_c2 draws drop lines only under zones the human touches (D2).
  - **Controls first:** 114 (identity 12, swap 32, lift 30, sink 20, kink 20).
    - `pose_change` is admitted: 0.970 on 33 claims, recall 1.0. The one false call is a sink control
      (Shoulder-Pressing -a@543).
    - `closer` is admitted (1.0 on 42 claims). Artefacts stay advisory (recall 0.53).
  - **Then all 272 non-transition holds.** None was called a different pose, and 248 were accepted.
  - **The 24 rejected edits** are all "the same pose, her unedited fit closer, no new artefact". They fall where
    Step 3 had to edit:
    - soft supports: Bridge ×3, Plow ×2, Upward Plank;
    - pressed body–body contacts: Tree ×3, Eagle ×3, Side Crow ×2, Koundinya;
    - forearm and side supports: Feathered Peacock ×3, Dolphin Plank, Side Plank ×2;
    - also Warrior III, Lord of the Dance and Revolved Side Angle.

    As the card asked, they are a flag (`fit_closer`), not an exclusion.
- **Gate v2 (item 8).**
  - Its inputs are pinned by id: the labels name the statics, the retarget and the fit.
  - A Pass C verdict counts only if its packet's sha256 of the before and after equal the motions in use; 0 are
    stale and 0 missing.
  - A Pass C claim decides only if its class is admitted.

  | Decision | Holds | Reasons |
  |---|---|---|
  | exclude | 10 | the 8 transitions; Scorpion -b@945 and @1163, beyond the plant (wrists) |
  | mask | 1 | Triangle -b@669: the right hand cannot reach the floor; the hold is held without it (s* 0.27) |
  | flag | 65 | `pose_preparation` 35, `fit_closer` 24, `statue_moved_single_leg` 17, `critical_partly_realised` 1 |
  | pass | 204 | |

  - **The plant's contract holds in every window.** No hold breaks the joint box, the floor contract, the
    new-overlap scan or the jerk test inside its window:
    - 0° past the box;
    - floor ≥ −0.50 cm;
    - 0 new overlap pair-frames;
    - 0 jerk frames, and also 0 outside the windows.

    The largest window edit is 16.6 cm for one body (budget 25) and 5.8 cm mean p95 (budget 10).
  - **`statue_moved_single_leg`** flags 17 of the 24 single-leg holds: without a policy their statues fall, like
    Step 3's Tree.
  - **`critical_partly_realised`:** Shoulderstand -a@914's TRUNK+L_HAND is realised on only 47 % of the frames where
    she makes it. This needs Step 5's per-frame masks.
  - No clip loses all its holds.
- **Traps.**
  - **Statics runs once per configuration, and the labels cite the statics of their own configuration.** So every
    configuration change takes two builds (labels → statics → labels).
    - The chain here: pass 1 → statics `fb432fa15d`; pass 2 → statics `aab60189f4`; the final cites `aab60189f4`.
    - A test checks that the final's configuration equals pass 2's.
  - **Don't edit a module whose sha256 is evidence.** `labels_v2.py` is the `rule:` row of three labels folders,
    and `b6.py`'s sha256 is in the B6 id. Extend with a new module instead.
  - **render_v4 cannot separate torso from pelvis** (one grey), so a contact at that boundary does not reproduce by
    zone.
  - **A single-pair necessity test misses jointly required contacts**, such as Peacock's four braces, each only
    `useful`.
  - **Two Step 1–8 tests read "the newest folder".** `test_gate` reads under `gate/`, and `test_plant_v2_step2`
    reads the first three folders under `statics/`. So Step 4 writes `gate_v2/` and `statics_v2/`.
  - `test_verdicts` globs `calibration/render_v*.json`, so the new tables live in `pass_a_v4/`, `pass_b/` and
    `pass_c/`.
  - **`b6.py --labels` defaults to the oldest labels v2 folder by mtime** (today the draft); always pass it.
    `edits_v2 --labels` is now required.
  - **Three plant-v1 bindings survive `use_plant`.** `render_v4`, `witness_v2` and `statics_v2` each work around one:
    - `render._body_geoms` is lru-cached on `ids.MJCF`, and the cache survives `use_plant`;
    - `witness.control_info` hard-codes plant v1's robot config;
    - `statics.CONFIG` binds the plant at import.
  - **Run the tests with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`**; otherwise `test_retarget`
    hangs.
  - The Pass C legend's leak guard rejects corpus tokens: "holds" contains "hold".
- **For Step 5.**
  - **Read `gate_v2/holds_repaired_ftC_posefix.gate_v2.b2f76f2606/release_holds.yaml`.** It is labels v2's manifest
    without the 10 excluded holds. Per hold:
    - `pairs` and `pairs_ground` are the commanded contacts;
    - `pairs_configured` holds the configured ones;
    - `gate` carries the decision, the masks and the flags.
  - **The contact-target sidecar (C1).**
    - 63 configured pairs (59 critical + 4 restored), all realised at the exemplar.
    - 1 masked support.
    - The `critical_partly_realised` hold needs a per-frame mask.
  - **Node identity.** The 8 transitions are gone, and 2 Plow holds changed orientation bin. `_h` names are kept;
    `pose_role` says what each shows, and the 35 preparations are flagged.
  - **The 17 single-leg statue flags** mark holds a policy must balance actively. They are not a release decision.

### Step 5 — Build the release on plant v2 and prepare training (BUILD_PLAN Steps 9–10)

**Goal.** A training-ready, immutable release built from three things only: gate v2's release manifest (labels v2
minus the gate's exclusions and masks), the Step 3 motions and plant v2. Then train from scratch.

**Do.**
- **Corpus chain** (the `build_expert60_ftR.sh` pattern, pointed at v2 everywhere): hold-extended variants
  (hold extension v2: real-frame pressure, masks on inserted frames), package, hold graph (graph v2, keyed on
  hold_id), pressure port and on-mat gate, and physics tables with the plant sha.
- **Identity checks.** Port `check_expert60_ftR.py` to the new release. Package, graph, tables and manifest must
  agree on hashes, motion order, fps, body order, `pair_names` and plant sha, and fail loudly otherwise (TODO C5).
- **Contact-target sidecar** (TODO C1): the commanded contacts, masks and flags from the release manifest, and the
  roles from labels v2's annotations. Re-check the swing term's calibration on the new tables (TODO C3).
- **Smoke run:** 256 envs, 4 epochs, weight 0 on the new terms (BUILD_PLAN Step 9 exit).
- **Train fresh** under a new experiment identity with fresh normaliser statistics. Old checkpoints are not reused.
- **Evaluate** with the full-clip evaluator (2,250 steps) plus the realisability metrics: drag, slide, support
  violation, float, and COM vs COP.
- Then Pass D: review policy rollouts with the same packet machinery. Record new PhysX rollouts so `plant.py`'s
  force validation can be re-run.

**Carried in from Step 4** (details, numbers and the rebuild commands in "What Step 4 found"):
- **The input is gate v2's release manifest, by id. Never read the newest folder.** It is
  `data/reference_curation/gate_v2/holds_repaired_ftC_posefix.gate_v2.b2f76f2606/release_holds.yaml`.
  - **What it contains.** Labels v2's manifest (`…labels_v2.e03994ad3c`) without the 10 excluded holds: 270 holds
    on 56 clips, all at 60 fps. Its `mjcf` is plant v2's and its `plant_sha256` is v2's.
  - **The motions.** Each clip's `source` is its Step 3 reference (`…retarget_v2.e4278ada02/<stem>.motion`). Each
    reference carries plant v2's sha, and its frame count equals the manifest's (all 56 verified).
  - **The contacts of a hold.**
    - `pairs` and `pairs_ground` are the *commanded* contacts: configured minus the gate's masks.
    - `pairs_configured` holds the configured ones.
    - `gate` holds the decision, the masks and the flags.
    - `labels` holds the status, roles, `any_of`, `not_realised`, `pose_role` and the statics block.
  - **The frames of a hold.**
    - `hold_id` names the source exemplar; `frame_hold` is the corrected exemplar.
    - `frame_start` and `frame_end` are labels v2's window. 10 windows are trimmed to where the demanded pairs hold.
    - Time and frame fields agree on all 270 holds.
  - **Per-contact detail** is in `labels/…labels_v2.e03994ad3c/annotations.jsonl`: `target_role`, `role_from`, a
    `critical` (B6) or `restored` (statics) block, and the avatar, mat and statics evidence. The gate's per-hold
    evidence is in `gate_v2/…/holds.jsonl`.
- **What the contact-target sidecar (C1) gets.**
  - Ground roles: `required_support` 522 (455 from the mat at ≥ 50 N, 64 from statics), `required_touch` 342,
    `forbidden_support` 18.
  - Body–body: 63 configured pairs, all closed at the exemplar.
    - 59 are critical: the reviewer's self-consistency was admitted for them.
    - 4 are Peacock -a@788's forearm braces, restored by statics.
    - Pairs outside the configuration are `allowed` or `incidental`. No `any_of` group remains.
  - One mask: Triangle -b@669's `R_HAND:G`. The plant's hand cannot reach the floor, and the hold stands without it.
    A masked contact leaves every contact target.
  - **Flags are not release decisions:**
    - `pose_preparation` 35, `fit_closer` 24, `statue_moved_single_leg` 17;
    - `critical_partly_realised` 1: Shoulderstand -a@914's TRUNK+L_HAND is closed on only 47 % of the frames where
      she makes it, so it needs a per-frame mask.
- **What changes for the hold graph** (node identity = name, ground set, orientation).
  - The 10 excluded holds are gone: 8 transitions, and Scorpion -b@945 and @1163, beyond the plant's wrists.
  - 2 Plow holds changed orientation bin on the new references (Plow -a@756, Plow -b@1644).
  - `_h` names are kept. `pose_role` says what each hold shows: family 72, standing 112, variant 53,
    preparation 35, undecided 8.
  - The ground sets are labels v1.1's human-side sets, unchanged.
- **Pressure.** Step 4 ran the attribution chain on the Step 3 references (`pressure_v2`). The gated ports, with all
  three validity columns, are in `output/reference_curation/pressure_v2/<rid>/gated/`, and `pressure_v2.check`
  verifies them. The hold-extended variants are new motions and need the chain run on their own frames.
- **Statics of the final configuration** is
  `data/reference_curation/statics_v2/…labels_v2.46024365fa.statics_v2.aab60189f4/`. It holds per-hold LP loads and
  per-contact necessity: 277 of 280 holds are holdable. The statue at training gains, under D4, passes 198 of 277.
- **If a label must change.**
  - Every configuration change takes labels → statics → labels.
  - Don't edit `labels_v2.py` or `b6.py`: their sha256 is evidence in the labels folders and part of the B6 id.
    Wrap them instead, as `labels_v2_restore` does.
- **Traps.**
  - Run the reference_curation tests with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`: 263 pass in
    ~10 min. Otherwise `test_retarget` hangs.
  - Step 1–8 tests read the newest folder under `gate/`, the first three under `statics/`, and every
    `calibration/render_v*.json`. Put new records in new folders.
  - Never run a Step 1–8 CLI with `REFERENCE_PLANT=v2`. Load the capture stores outside `on_plant()`.
- **Still open.**
  - D3, the IsaacLab statue.
  - `plant.py`'s PhysX force validation, which needs new rollouts (Pass D).
  - The head's marker admission fails on the v4 evidence. This is recorded, not applied.

**Acceptance.** BUILD_PLAN Step 9's exit tests on the new release, and a first expert whose single-leg and
headstand clips do not regress against the ftC-era numbers (probe: single-leg 0.998, inversion 0.998 on ftC).

**What Step 5 found (2026-10-01).** In progress. BUILD_PLAN Step 9's exit is met and the release is recorded; the
fresh expert is training. The expert half of the acceptance, the evaluation and Pass D wait for checkpoints.
- **The release** is `holds_repaired_ftC_posefix.release_v2.a2dda5d2ac`.
  - Record: `data/reference_curation/releases/<id>.json` (+ `.md`), with every artifact's sha256.
  - Heavy files: `data/smpl/reference_curation/<id>/`, 1.4 GB. They are untracked, like every corpus in `data/smpl/`.
  - Inputs, pinned by id and content: gate v2's manifest (its sha256 is pinned in the builder), each of the 56
    references against the retarget record, and each gated port against the pressure record. A port must carry
    its reference's kinematics exactly.
  - Build with `-m reference_curation.release_v2 --build` (38 s); re-check with `--check <id>` (7 s). The id
    hashes the inputs, the builders' sources and the parameters. A release is never rebuilt in place, and a
    folder without a record is not a release.
- **Files.** The v1 builders are unchanged, so ftC and ftR stay reproducible.
  - Builders: `data/scripts/build_hold_graph_v2.py` and `build_physics_tables_v2.py`;
    `reference_curation/{hold_extension_v2, contact_targets_v2, release_v2}.py`.
  - Runtime, new: `protomotions/utils/release_identity.py`, `protomotions/envs/control/contact_targets.py`.
  - Runtime, edited: `contact_graph.py`, `contact_graph_control.py`, `physics_terms.py`, `support_penalty.py`,
    `context_views.py`, `hold_curriculum_evaluator.py`, and `mlp_goal_conditioned.py` (`--contact-targets`,
    `--release-record`). Without the new flags every run behaves exactly as before.
  - Launcher: `data/scripts/run_expert_release_v2.sh` (`SMOKE=1` gives the exit run). It reads every path from
    the record.
  - Evaluation: `reference_curation/realisability_v2.py` scores any rollout library the evaluator saves
    (`results/<run>/results/predicted_motion_lib_epoch_<N>.pt`, every 3rd evaluation) for slide, float and COM vs
    COP. `--reference` scores the reference itself as the baseline (10 s).
  - Tests: `reference_curation/tests/test_release_v2.py` (12) and `protomotions/tests/test_release_v2_runtime.py`
    (13). Full suites: reference_curation 275 pass and 3 skip (pre-existing conditional skips), 10.5 min at
    `OMP_NUM_THREADS=1`; the 16 runtime test files touching the changed modules, 290 pass.
- **The chain**, `build_expert60_ftR.sh`'s, pointed at v2.

  | Item | Result |
  |---|---|
  | Hold extension v2 | 56 clips x (0, 3, 7 s) = 168 motions, 282,936 frames, 38,400 inserted (64 extendable holds). Real frames keep the port's measurement; inserted frames carry zeros with every validity column 0. The d = 0 variants are the gated ports byte for byte, and the kinematics are v1's splice exactly. `lineage.npz` gives each frame's x0 source. |
  | Package | `motions.pt`, equal to the motions frame by frame, the three measured channels included; plant v2's sha256 |
  | Graph v2 | 123 nodes, 196 edges, 810 segments, 270 hold ids; node ids are the sorted keys |
  | Physics tables v2 | 345,854 swing zone-frames; lean-gated 90, COP-valid 768, share-valid 735 segments; the sha256 of the graph, the package and the manifest |
  | Sidecar (C1) | 63 configured body-body contacts (59 critical, 4 restored), 1 mask, 4 flags, roles, statics loads, 3,233 known-free ground zone-holds, a per-frame pair table |
  | Probe plans | 148 fork and dwell plans, built from the release's own graph (`plans/`) |
  | Identity checks (C5) | **26 of 26 pass** |

- **Identity checks (C5, the port of `check_expert60_ftR.py`).** Every derived artifact is re-derived or compared
  with what its inputs determine.
  - Manifest and sources pinned.
  - Every variant is its port spliced: kinematics, plant, fps and lineage. Pressure: real frames are the port's,
    inserted frames zero and invalid. The d = 0 variants are byte-identical to their ports. The holds are
    re-timed with their hold ids kept.
  - Package order, frame counts, fps and plant; the package equals the motions frame by frame.
  - Graph: built from this package and manifest. Motion order, frames, fps and vocabularies match. Segments match
    the manifest's holds (hold id, times, commanded contacts, node key). A4 holds on every segment. Every hold's
    pose resolves to its own contact set.
  - Tables: graph, package and manifest sha256; names, frames, fps, body order and plant; the graph's pair names,
    zone order and layout. No pressure-sourced swing label falls on an inserted frame.
  - Sidecar: hashes, names, pairs, hold ids and frames. Commanded = configured − masks = the graph's
    `seg_contact`. Its counts equal the labels' and the gate's.
  - The runtime loaders accept all three, and the plans resolve.
  - **Runtime identity checks fail loudly.** These are new:
    - graph v2 against the library's frame counts and fps;
    - physics tables against frame counts and fps on every version, and v2 tables against the graph's pair names
      and sha256;
    - `ContactTargets` against the graph (names, pairs, zone order, layout, hold ids, commanded set, sha256),
      frames, fps and plant.
    - With `--release-record`, the package, graph, tables, sidecar and the evaluator's hold manifest must match
      the record by sha256 (`ReleaseMismatchError`).
- **Graph v2: what the 0.5-vote union got wrong.** Five nodes hold side-specific body-body pairs.
  - Side Crow -c's two holds share one node: @606 has the left shelf, @1296 the right. Each shelf is in half
    the node's segments, so v1's `node_contact` commanded **both** shelves for a manual goal at either hold.
  - Koundinya, Side Plank, Standing big toe hold and the supported headstand gain a pair one hold lacks in the
    same way.
  - At its own hold's pose, v1's rule would have served a manual goal another contact set on 18 segments. v2
    serves each segment its own set (`ContactGraph.manual_contact`). `node_contact` is now the pairs every source
    hold of the node commands, and conflicts are recorded in the description.
- **Physics tables v2 and the swing re-check (C3).**
  - Rules: column 1 for the absolute 14 / 70 N thresholds; column 2 only for shares, and with column 0 for the
    COP. "Unloaded" counts only where the zone's lowest point is within the attribution's 6 cm band and over
    the mat.
  - On the same 168 motions, pressure-only labels fall 28,147 → 17,649. ftC had 18,254 and ftR 50,824. 6,807
    of v1's "unloaded" frames were invisible to the attribution.
  - Against the human evidence, on the x0 clips:
    - velocity-rule labels: the human is separated on 99.88 % of 107,457 zone-frames; 4 % of the 629 the mat
      sees carry over 14 N, none over 70 N;
    - pressure-rule labels are placed but unloaded limbs: mesh contact 82 %, the mat over 14 N on 0 %.
  - The human's own load on labelled feet and hands is p50 0 N, p99 12.9 N, max 30.8 N, against the term's
    72.6 N saturation. A perfect imitator is never charged to saturation, so ft_c's −0.3 calibration carries
    over.
- **The sidecar (C1).**
  - Roles on the 270 released holds:
    - ground: `required_support` 515, `required_touch` 299, `forbidden_support` 16;
    - pairs: `required_touch` 63, `allowed` 283, `incidental` 196, `unresolved` 66, `unspecified` 146.
  - Flags (not release decisions): `pose_preparation` 30 (35 before the 5 excluded transitions),
    `fit_closer` 24, `statue_moved_single_leg` 17, `critical_partly_realised` 1.
  - **Known free.** A zone is a known negative when it is not configured and the human evidence
    (`sources.ground_source` on capture store v4) calls it separated on ≥ 95 % of the window.
    - 3,233 of 3,236 non-configured zone-holds are separated on every frame.
    - The other 3 are never seen touching but are undecided on 9–79 % of their frames: Standing Split
      -a@540 R_FOOT, Garland -a@334 PELVIS, big toe hold -a@950 L_THIGH. They stay unknown and are never charged.
    - Configured zones are in contact on 100 % of their frames.
  - **Per-frame pair masks.** A pair target is eligible where the reference closes the pair (v4's plant-v2 gap
    ≤ 1 cm) and the human's mesh touches. It reproduces the gate's flag: Shoulderstand -a@914's TRUNK+L_HAND is
    0.471 of her contact frames.
  - The two human-mesh pairs outside the graph's vocabulary (`HEAD+*_UPPER_ARM`, adjacent zones) are counted,
    not carried; none is configured.
- **C4 is closed for this experiment.** The only reader of the reference's `rigid_body_contacts` is the
  contact-match reward, which the goal-conditioned expert never binds. Graph, tables and evaluator read the
  manifest.
- **Smoke run (the Step 9 exit).** `SMOKE=1`: 256 envs, 4 epochs, every non-tracking term at weight 0.
  - It completed with the release check passing at startup. The sidecar loaded exactly 3x the x0 counts.
  - Every term logged finite values:
    - required-support gate 0.69–0.73, with the support gate at 0.70–0.74;
    - pair-target gate ~0.02; known-free load 1–4 N;
    - swing gate 0.26–0.30, swing load 6–13 N (ft_c's smoke 0.30–0.45 and 5–16 N); lean gate 0.016–0.022.
  - **Index agreement.** On uniformly sampled (clip, time), the runtime lookup reproduces the sidecar's offline
    coverage: in-hold 0.557 / 0.557, required support 0.515 / 0.516, pair target 0.126 / 0.127. The smoke's lower
    pair gate is an untrained policy's state distribution (short episodes, 60 % segment-anchored resets).
- **Training (Step 10), running since 2026-10-01 16:32.**
  - Run `smpl_yogi_v2_expert56_a2dda5d2ac` (wandb `06k9vjpn`, PID 215359, log
    `results/smpl_yogi_v2_expert56_a2dda5d2ac_train.log`), launched by `run_expert_release_v2.sh` with
    `setsid nohup`, so it outlives the session. Stop it with `kill 215359`.
  - Robot `smpl_yogi_v2`, no checkpoint, fresh normalisers; 4,096 envs and 30,000 epochs.
  - Measured 9.57 s per epoch, evaluations included, so it ends around **2026-10-05 00:19**.
  - Disk: checkpoints every 500 epochs, 0.637 GB each (~38 GB in all); a rollout library
    (`results/<run>/results/predicted_motion_lib_epoch_<N>.pt`, 0.26 GB) every third evaluation. 110 GB was free
    at epoch 1,558.
  - Reward: ft_c's stack, unchanged in kind:
    - tracking;
    - support −0.3 (0.25 s EMA), now gated by the sidecar's known-free zones;
    - swing −0.3 and lean −0.3;
    - the sidecar diagnostics at weight 0.
  - Curriculum mixture (0.8 uniform); full-clip evaluator every 500 epochs over 2,250 steps (the longest motion
    needs 1,890).
  - **Read the acceptance off** `eval/perf_group/single_leg_score` and `inversion_score`, against ft_c's 0.998 /
    0.998 on ftC: the same `HoldCurriculumEvaluator`, deterministic, 2,250 steps. Confirm a candidate standalone
    with `expert_revist/ft_c/eval_checkpoint.py --checkpoint results/<run>/epoch_N.ckpt --config
    results/<run>/resolved_configs.pt --out-dir <dir>`. Allow ±0.035 evaluation noise (the ft_c memory), and pick
    checkpoints per family from the per-clip CSVs in `results/<run>/curriculum/`.
  - **Realisability.** The evaluator logs drag (`eval/drag/*`) and support violation. Score slide, float and COM vs
    COP with `realisability_v2 --rollouts <predicted_motion_lib>`; only tracked frames count. The baseline is
    the reference itself (`--reference`), which a perfect imitator matches:
    - tracked 1.0; float 0.0 of commanded supports;
    - slide p50 / p90 0.30 / 2.54 cm/s, 5.9 % of planted human-touch frames above 5 cm/s;
    - COM–COP p50 / p90 1.44 / 3.44 cm over every hold frame where the mat's COP is valid.

    Its per-motion tracked share equals the evaluator's `eval/perf/track`: 0.1817 on the smoke, and 0.2295 vs
    0.229 at epoch 500. That is an independent check of its frame alignment. Read `tracked_motion_mean`;
    `tracked_share` pools frames, and full-length rollouts differ in length.
  - **Progress at epoch 1,558 (2026-10-01 20:41).** Three full-length evaluations, no errors. The 2,250-step window
    covers every motion; the longest needs 1,890.

    | Evaluation (epoch) | 500 | 1,000 | 1,500 | ft_c on ftC |
    |---|---|---|---|---|
    | `eval/success_rate` | 0.107 | 0.179 | 0.202 | 0.961 |
    | `eval/perf/score` / track / hold | 0.277 / 0.229 / 0.324 | 0.394 / 0.393 / 0.396 | 0.462 / 0.471 / 0.454 | 0.962 / 0.984 / 0.940 |
    | family hold / support violation | 0.024 / 0.456 | 0.158 / 0.391 | 0.233 / 0.323 | 0.903 / — |
    | group score: **single-leg** / **inversion** | 0.249 / 0.195 | 0.295 / 0.257 | **0.458 / 0.304** | **0.998 / 0.998** |
    | group score: arm balance / connective | 0.252 / 0.355 | 0.293 / 0.584 | 0.385 / 0.596 | 0.899 / 0.951 |
    | drag, all / gate set (J per rollout) | 41.5 / 32.9 | 72.5 / 65.7 | 111.0 / 127.3 | 23.3 / 57.3 |
    | normalised jerk | 97.8 | 133.8 | 211.8 | 87.1 |
    | episode length in training, nearest epoch (steps) | 99 | 147 | 188 | — |

    - Every tracking and hold metric rises from one evaluation to the next. The acceptance bar is far off at
      1,500 of 30,000 epochs, as expected from scratch. The ft_c column is the probe's (`probe_ft_c/summary.md`):
      the same evaluator, after ~54k cumulative epochs.
    - **Drag and jerk are rising** with survival. A policy that falls within 0.3 s drags little.
      - At epoch 1,500, drag is 4.8x ft_c's corpus figure and jerk 2.4x.
      - ft_c's gates were "drag down, jerk ≤ baseline". Watch whether these come down once tracking saturates.
      - If they do not, the swing term's weight is the first lever.
    - **The pair-target gate has reached the sidecar's offline coverage**: about 0.02 in the smoke and at the run's
      start, 0.131 at epoch 1,560, against 0.127 offline. The policy now lives through the holds.
      - The configured body-body contacts are made on ~2 % of their eligible frames, and nothing asks for them
        at weight 0.
      - This is the measurement the weight-0 decision waits for.
    - The penalties stay small (raw means, weight −0.3): support 0.013–0.019, swing 0.055–0.071, lean
      0.005–0.014. Required-support met is 0.89–0.91.
    - Realisability, on the only rollout library so far (epoch 500; the next is at 2,000): float 12.6 % of
      commanded supports, slide p90 0.96 cm/s, COM–COP p50 4.3 cm. Only tracked frames count: 19.7 % of the
      rollout frames, 0.23 per motion. This is the from-scratch baseline the later checkpoints are read against.
- **Decisions, with the measurement behind each.**
  - **The attribution chain is not re-run on the variants**, a deviation from "Carried in from Step 4".
    `add_pressure_to_motions.py` pairs a motion's frame i with capture frame i, so on a variant every frame after
    the first insertion would be misaligned with the mat. That is why v1 dropped the channels and the v1 tables
    re-timed the port. A variant's real frames are x0 frames with identical poses, so their attribution is the
    port's. Hold extension v2 carries it through the splice index, and the identity checks prove it equals the
    port at each frame's source; inserted frames carry no measurement.
  - "Evaluate on the original corpus" (BUILD_PLAN Step 10) is ft_c's recorded numbers on ftC. The shipped
    references are plant v1's, and plant identity refuses them on `smpl_yogi_v2` by design.
  - Known free at ≥ 95 % separated (above). Every threshold from 0.92 to 1.0 gives the same partition.
  - Graph node ids are the sorted keys. Plans name nodes by key, so nothing depended on the order.
  - The fresh run keeps ft_c's reward stack. The design's ablation C ("repaired references, unchanged reward
    stack") is the clean first comparison. New pair and pressure targets stay at weight 0 until a run on this
    release shows which contacts the policy actually loads.
  - The support term with the sidecar differs from ft_c's complement rule only on the 3 undecided zone-holds.
- **Traps.**
  - **Graph times are float32.** Compare with the float32 of the manifest's value, not with a tolerance: past 32 s
    the half-ulp is 1.9e-6. The first build's check failed Lord of the Dance -c's x3s and x7s on a 1e-6 tolerance.
  - **The release id hashes the builders' sources**, so editing a builder, even a comment, moves the id. `--check`
    on a frozen release reports that drift as a note and certifies it by artifact hashes and the 26 checks.
  - **`data/smpl/` is not git-ignored.** Leave the release folder untracked.
  - Run `release_v2` without `REFERENCE_PLANT`. The human-side capture stores refuse plant v2.
  - MotionLib prints a warning per unknown `.motion` key, so the lineage lives in `lineage.npz`, not in the
    motions.
  - A foreground `sleep` is blocked in this harness. Wait on a run with a background until-loop that also
    exits on `Traceback|Error:|Killed` or a dead PID.
- **Still open.**
  - The expert's acceptance, and its realisability scored on its own rollouts (the tooling and the reference
    baseline exist).
  - Pass D, and new PhysX rollouts for `plant.py`'s force validation.
  - D3, the IsaacLab statue.
  - The head's marker admission (recorded, not applied).

---

## 5. Status

| Step | State | Date | Measured / notes |
|---|---|---|---|
| 0: investigation | done | 2026-09-30 | §1; outputs in `output/reference_curation/subject_body_check/` |
| 1: plant v2 MJCF | done | 2026-09-30 | `smpl_yogi03596_v2{,_flat}.xml` + `_v2_limits.json`; builder `data/scripts/build_subject_plant_v2.py`, acceptance `reference_curation.plant_v2`, replay `reference_curation.mosh_replay`, record `data/reference_curation/plant_v2/plant_v2.json`, 20 tests (`test_plant_v2.py`, ~1 min). 74.000 kg; rest FK 1.6e-8 m; 0 rest overlaps (min gap 2.9 cm); overlaps 79,943 vs C1 80,058 (human-separated 1,962 vs 7,746); crowns 27/27; grounded floats 254/919 (≤ 261, target 200 missed); LP-holdable 95 / 164 at +0.5 / 0 cm (her root: 251; shipped 41); COM–COP p50 1.0 cm. 23 ranges widened; wrist / hand torque +50 % (30 / 15 N·m, the user's Step 2 decision, re-measured); neck gains ×2.3 for Step 2. See "What Step 1 found". |
| 2: USD, robot config, plant identity | done | 2026-09-30 | USD `smpl_yogi03596_v2_usd/` (`build_plant_v2_usd.py`, record `usd_v2.json`); robot `smpl_yogi_v2`; PhysX sanity `reference_curation.plant_v2_physx` (`physx_smpl_yogi_v2.json`, v1 baseline `physx_smpl_yogi.json`); `protomotions/utils/plant_identity.py`; 22 tests. One articulation root (Pelvis); colliders ≤ 8e-6 mm / 5e-6°; PhysX 74.000 kg, every body ≤ 2.6e-7; limits ≤ 8e-8 rad; wrist / hand 30 / 15 N·m; neck 1158 / 116; FK 418 frames ≤ 1.4e-6 m; default and 136 standing resets: no launch; v1 / legacy / mixed data raises `PlantMismatchError`. Capture / statics loader checks go to Step 4's new store versions. See "What Step 2 found". |
| 3: references on plant v2 | done; 9 of 10 acceptance tests met (open: worst-body p50) | 2026-09-30 | Writer `reference_curation/fit_writer.py` (`fit_v2.v1.b8f5e86fc9`, 56 clips; Cockerel -b, Scale -a, big toe hold -c and, by the user's decision after the first run, Firefly -b dropped); retarget `reference_curation/retarget_v2.py` (`…labels_v1_1.3b3fc75828.retarget_v2.e4278ada02`, 14 min on 16 workers; the other 56 motions are byte-identical to the 57-clip run's); PhysX `retarget_v2_physx.py`; 11 tests (`test_retarget_v2.py`). Fit → retarget: frames past the box 26,532 → 0; labelled floats 48 → 1 of 864; crowns 21/21; touch penetration 43.0 % → 0.01 % (skin 15.3 %); slide p90 2.96 → 2.99 cm/s, > 5 cm/s 6.7 → 6.9 %; new jerk frames 0; overlaps > 1 cm 63,201 → 604, 0 new; mean displacement p50 1.37 cm (Step 8 3.61), **worst body p50 10.6 cm (Step 8 15.5; soft supports, poses past the box, palm loading)**; COM–COP p50 1.10 cm. PhysX: library loads on v2, v1 refused, FK 1.2e-6 m on 701 frames, 0/701 resets launch (control 3/3). See "What Step 3 found". |
| 4: reviewer re-run, labels v2, gate v2 | done; all 5 acceptance tests met | 2026-10-01 | Modules `reference_curation/{pressure_v2, capture_v4, audit_v2, witness_v2, statics_v2, render_v4, packets_v4, calibrate_v4, b6, labels_v2, labels_v2_restore, edits_v2, gate_v2}.py`; 59 tests (`test_step4_{evidence,rules,review,final}.py`). Labels v2 `…labels_v2.e03994ad3c` (statics `…labels_v2.46024365fa.statics_v2.aab60189f4`); gate v2 `…gate_v2.b2f76f2606`: 270 / 280 holds released (204 pass, 65 flagged, 1 masked), 10 excluded (8 transitions, 2 Scorpion -b beyond the plant). Mat explained ≥ 90 % on 99.9 % of loaded frames; labelled floats 434 → 1 / 864; unexplained-load holds 33 → 0; LP-holdable 277 / 280; statue (D4) 198 / 277. Pass A v4 admits float, floor and left/right. B6 admitted (0.933 on 135 claims): 59 critical pairs in 44 holds, plus 4 restored by statics (Peacock -a@788); 0 configured pairs unrealised. Pass C: 0 / 272 called a different pose, 24 `fit_closer`. Reviewer $106.94 (plan ~$90). See "What Step 4 found". |
| 5: release and fresh training | in progress: Step 9 exit met, release recorded, expert training | 2026-10-01 | Release `holds_repaired_ftC_posefix.release_v2.a2dda5d2ac` (`reference_curation.release_v2`; record `data/reference_curation/releases/`, 1.4 GB in `data/smpl/reference_curation/`): 56 clips, 270 holds, 168 motions (0/3/7 s, 38,400 inserted frames carrying no measurement), graph v2 (123 nodes, 270 hold ids; manual goals resolve the side-specific segment, which v1's 0.5 vote got wrong on 18 segments, e.g. Side Crow -c's two shelves), tables v2 (pressure-only swing labels 28,147 → 17,649), sidecar (59 critical + 4 restored pairs, 1 mask, 3,233 known-free zone-holds, per-frame pair masks). 26 / 26 identity checks (C5); C3: velocity-rule labels human-separated 99.88 %, the human's own load on labels p99 12.9 N vs the 72.6 N saturation. Runtime checks: frames, fps, pair names, graph sha256, release record. Weight-0 smoke passed; the runtime lookup reproduces the sidecar's coverage (0.126 / 0.127). Realisability tooling (`realisability_v2`): the reference baseline is float 0, slide p90 2.5 cm/s, COM–COP p50 1.4 cm. 25 new tests. Training `smpl_yogi_v2_expert56_a2dda5d2ac` (wandb `06k9vjpn`, PID 215359) since 16:32. At epoch 1,558 (20:41) the score was 0.277 → 0.394 → 0.462 over the evaluations at 500 / 1,000 / 1,500, with single-leg 0.458 and inversion 0.304 against the 0.998 bar. Drag and jerk are rising with survival (watch them). The run ends around 2026-10-05 00:19. Step 5's files were staged on 2026-10-01. See "What Step 5 found". |

**How this relates to the existing plan.**
- `TODO_before_step9.MD` A1–A3 are done in Step 3:
  - A1: slide clip p90 2.99 cm/s, against the fit's 2.96;
  - A2: balance on every quasi-static frame with a stable support set; COM outside the supports 3,026 → 40 frames;
  - A3: no new jerk frames.
- A4 is done in Step 4: gate v2's `a4_check` passes on all 270 released holds. A5 is replaced by Step 5 (its
  fine-tune of ft_c is moot, because ft_c is discarded).
- B1 and B2 are decided: the lotus clips and Standing big toe hold -c are dropped. So is Firefly -b, after Step 3.
- B3 is done: the head collider was re-centred in Step 1, and all 21 crowns are realised in Step 3.
- B4 and B5 are moot as written (they judged gate v1 on plant v1). Gate v2 re-decides them.
- B6 is done in Step 4.
  - The reviewer's self-consistency is admitted, and its 59 critical pairs are the configured body–body contacts.
  - Peacock -a@788's four braces are restored by statics.
  - None of Step 3's 60 `geometry_incompatible` pairs is left configured and unrealised.
- C1–C6 carry into Step 5. C4 ("`rigid_body_contacts` now means something else") holds for the Step 3 motions
  too: they are geometric (≤ 1 cm).
  - Step 5 did all of them:
    - C1: the contact-target sidecar, with the gate's masks and flags;
    - C2: hold extension v2 and `lineage.npz`;
    - C3: the attribution chain (Step 4) and the swing re-check;
    - C4: no consumer in the expert;
    - C5: 26 identity checks plus the runtime checks.
  - C6 (re-review after a retarget change) did not arise: the references are Step 3's, unchanged.
- D:
  - D1 is closed: with Firefly -b dropped, Step 3 adds no new overlap.
  - D2 is done in Step 4: render_c2 draws drop lines only under zones the human touches.
  - D3 is still open (the IsaacLab statue).
  - D4 is done in Step 4: the statue's contact rule covers only LP-loaded zones (194 vs 180 of 276 pass).
  - D5 is mostly closed: legs and head now fit her. The box hands still need a palm correction of ~16°, and a
    fixed rotation would absorb only ~10° of it.
  - D6 is done: the Step 8 retarget's motions (pilot and corpus, 223 MB) are deleted. Its lineage stays, because
    `plant_v2.py` reads `head_blocked` from it.
- E1 is done:
  - Steps 1–3 are committed (`387665f`).
  - The v2 MJCF/USD and labels v1, which `plant_v2.py` reads, are in `77d7517`.
  - Step 4 is committed together with this text.
- BUILD_PLAN Steps 9–10 run after Step 4 here, on labels v2, plant v2 and the Step 3 references. Their cards' inputs
  ("labels v1.1", "the retargeted references", ftR) are superseded.

**Next, in order** (updated 2026-10-01, Step 5 in progress):
1. **Watch the fresh run** (`smpl_yogi_v2_expert56_a2dda5d2ac`, wandb `06k9vjpn`, PID 215359). It ends around
   2026-10-05 00:19. Every 500 epochs, check:
   - `eval/perf_group/single_leg_score` and `inversion_score` against ft_c's 0.998 / 0.998 (0.458 / 0.304 at
     epoch 1,500), and the other groups;
   - `eval/drag/*` and `eval/normalized_jerk_mean`. They rose 41.5 → 111.0 J and 97.8 → 211.8 over the first three
     evaluations; they should fall once tracking saturates, or the swing term's weight is the first lever;
   - `env/raw_r/diag_pair_target_met`, about 0.02 of eligible frames at weight 0;
   - the viz panel.

   Score each saved rollout library (epochs 500, 2,000, 3,500, …) with `realisability_v2 --rollouts`. Confirm a
   candidate checkpoint standalone with `eval_checkpoint.py` (see "What Step 5 found"). Then run Pass D on it and
   record PhysX rollouts for `plant.py`'s force validation.
2. **Step 5's files are staged** (2026-10-01) for the user's commit.
   - Code: the builders, the runtime changes, the launcher and the tests.
   - Records: the release record (`data/reference_curation/releases/`) and this text.
   - Not staged:
     - the 1.4 GB release folder, which stays untracked;
     - `CLAUDE.md`, which has never been tracked; its Step 5 paragraph is in the working copy only.
   - The record's `git_rev` is Step 4's commit `f2dc103`, because Step 5 was uncommitted when it was built. The
     sha256s of the 12 builders it carries pin Step 5's code instead.

The items below were the plan before Step 5 and are kept for the record:
1. **Step 4 is done and committed together with this text** (staged 2026-10-01 for the user's commit), so Step 5's
   records stamp a git rev that contains it, as E1 did for Steps 1–3. The commit contains:
   - the 13 modules and 4 test files under `data/scripts/reference_curation/`;
   - its records under `data/reference_curation/`: `pressure_v2/`, `audits/`, `statics_v2/`, the labels v2 folders,
     `b6/`, `calibration/`, `gate_v2/` and the ledger's new verdicts.

   Labels v2 is `e03994ad3c` and gate v2 is `b2f76f2606`; see "What Step 4 found". ftR and the probe rollouts are
   still on disk and disposable; delete them only on the user's word.
2. **Step 5 = BUILD_PLAN Step 9**, from gate v2's `release_holds.yaml` (never from the labels folder directly).
   - The corpus chain on v2, using `build_expert60_ftR.sh` as the template: hold extension v2, package, graph v2
     keyed on `hold_id`, pressure port and on-mat gate, physics tables v2 carrying the plant sha.
   - The contact-target sidecar from labels v2 (C1); the identity checks (C5, porting `check_expert60_ftR.py`);
     the swing recalibration (C3).
   - Exit: a package round trip, a runtime load on `smpl_yogi_v2`, and a weight-0 smoke run (256 envs, 4 epochs).
3. **BUILD_PLAN Step 10.**
   - Compile the immutable release, validate it and train fresh on `smpl_yogi_v2`.
   - Evaluate with the full-clip evaluator plus the realisability metrics, against single-leg 0.998 and
     inversion 0.998 (ftC).
   - Then Pass D, and new PhysX rollouts for `plant.py`'s force validation.
