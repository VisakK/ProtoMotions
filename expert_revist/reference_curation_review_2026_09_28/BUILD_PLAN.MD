# Offline reference reviewer: build plan (v1, 2026-09-28)

**Purpose.** Produce an immutable, curated dataset release for the next expert: reference motions, contact labels and reward targets. The work splits three ways:
- **Deterministic evidence does the measuring:** the capture markers, the pressure mat and physics checks.
- **Claude Opus 5.5 is the VLM reviewer, for semantics only.** It reads rendered evidence packets.
- **Validators decide what gets published.**

**How the plan is built.**
- Ten steps.
- Each step is one session and a bounded code change of a few hundred lines with tests.
- Each step leaves the whole system more useful than before.

The analysis this plan rests on is [`README.MD`](README.MD) in this folder.

### How to use this file in a new session (keep context small)

1. Read §1 (decisions and facts) and §2 (conventions).
2. Find the first step marked `todo` in §4. Read **only its card** in §3, plus the files the card lists under *Read first*. Do not read the long notes in `notes/` unless a card names a section.
3. Build it, run its tests, write the measured numbers into §4, and mark it `done`. If you learned a trap, add it to the card.

**Kickoff prompt:** `Implement the next todo step of expert_revist/reference_curation_review_2026_09_28/BUILD_PLAN.MD`.

**All ten steps are done (2026-10-05).** Step 10's expert went on through the graph-growth round
([`../graph_growth_2026_10_03/PLAN.MD`](../graph_growth_2026_10_03/PLAN.MD)), and the user has chosen its next stage:
**Lane S, the student, skipping the perturbation fine-tune.** Read "After Step 10" (§3) before starting it.

**Dependencies:**

| Step | Depends on | Notes |
|---|---|---|
| 1 | — | |
| 2 | 1 | |
| 3 | 1 | |
| 4 | 2, 3 | |
| 5 | 1 (the model is verified) | can run in parallel with 2–4 |
| 6 | 4 (and 5 if available) | |
| 7 | 1, 2 | can run in parallel with 3–6 |
| 8 | 6, 7 | |
| 9 | 8 | |
| 10 | 9 | |

---

## 1. Decisions and verified facts (2026-09-28)

**The ft_c run is stopped.**
- Launcher (PID 2841159) and trainer (PID 2841160) were killed at epoch ≈ 6278.
- `results/smpl_yogi_expert60_ft_c/last.ckpt` (11:37) and `epoch_6000.ckpt` are kept.
- **The next expert trains only on a release produced by this plan.** Training code does not change before Step 9.

**Human input on contact labels is minimal by design (2026-09-29).** The plan evaluates the system's ability to build the dataset with agentic methods, so no human truth sets or label approvals; admission comes from the system's own evidence. See Step 6's caveat on roles.

**The VLM is Claude Opus 5.5 (`claude-opus-5-5`).** Two invocation modes were verified.

| Mode | How | Measured | Use it for |
|---|---|---|---|
| Headless, one packet per call (primary) | `claude -p "<prompt>" --model claude-opus-5-5 --allowedTools Read --output-format json --json-schema '<schema>'`, run with the **working directory outside the repo** | exit 0; 116 s and 4 turns for a 2-image packet; $0.40 at list price; schema-valid output | batch review: scripted, resumable, a fresh context per packet |
| In-session subagent | Agent tool with a reviewer prompt; the images are read with `Read` | 4 packets (8 images) in one agent: 20.6 min, 16 tool calls, 209k tokens | pilots and debugging only |

**Step 4 hardened the headless mode** (`reference_curation/review.py`):
- Flags: `--effort high --tools Read --allowedTools Read --restricted --strict-mcp-config --disable-slash-commands --no-session-persistence --exclude-dynamic-system-prompt-sections --max-budget-usd <cap>`.
- The prompt goes right after `-p`: the tool flags are variadic.
- `--restricted` confines `Read` to the packet directory.
- Measured per 4-image packet (p50): $0.135 and 33 s at high effort; $0.121 and 24 s at medium, which admits one evidence class fewer (§4).

**Blind-review hygiene.**
- **No `CLAUDE.md` above the review directory.** `claude -p` auto-loads the `CLAUDE.md` of its working directory's ancestors, and this repo's `CLAUDE.md` names clips and findings. None exists above the repo, so a packet directory outside the repo is clean.
- **Don't use `--bare`.** It would skip `CLAUDE.md` and auto-memory, but it authenticates only with `ANTHROPIC_API_KEY`, and none is set; the login is OAuth.
- **A custom subagent can set `omitClaudeMd: true`.** Frontmatter keys: `name`, `description`, `tools: Read`, `model: claude-opus-5-5`, `effort`. New agent files are picked up at session start. A **general-purpose** subagent inherits `CLAUDE.md`, so it is not blind.

**SMPL-X female body model: downloaded and fully verified (2026-09-28): `smplx_model_check.py` exits 0.**

The files are in `../moyo_toolkit/data/body_models/smplx/`:
- `SMPLX_FEMALE.npz` (121 MB, sha256 `2312f6dd2df506c9…`): a slim variant with no hand arrays;
- `SMPLX_FEMALE_v1_1.npz`: the standard v1.1 file.

Their posing arrays (J_regressor, weights, posedirs, faces, kinematic parents) are **identical**. The output of the check is in `output_smplx_model_check.txt`.

- **What MoSh used:** `smplx/female/model.pkl` with the subject's template. That template is stored in every MoSh pkl as `stagei_debug_details.v_template` (10475 × 3), all 400 betas are zero, and it ran with `use_hands_mean=True`.
- **Two conventions, both verified. Get them wrong and the fingers are about 5 cm off:**
  1. **The stored hand pose is absolute.** Pose the mesh with `fullpose` exactly as stored. This is MOYO's own reader (`moyo/utils/mesh_utils.py`: `use_pca=False, flat_hand_mean=True`). **Never add the hand mean to the pose:** that puts the fingers about 5 cm off and 5 cm into the floor.
  2. **MoSh's latent markers live in a canonical space whose hands are in the SMPL-X mean pose.** The finger markers sit 1–2 mm from their vertex on the mean-posed hand, against 12–76 mm on the flat template. So only *rebuilding markers* needs `hands_meanl` / `hands_meanr`, from the v1.1 file. The mesh never does.
- **Result:** the check rebuilds MoSh's markers to:
  - **fingers:** 0.20–0.22 mm median, ≤ 1.2 mm max;
  - **wrist/hand:** 0.30–0.57 mm median;
  - **non-finger overall:** 0.42–0.70 mm median, with 85–96 % of markers under 2 mm.

  The pose correctives are the right ones: turning them off raises the body median 5×. A few pose-dependent 3–15 mm outliers (`LFSH`, `RBSH`, `MFWT`) come from the script's marker-attachment approximation, not from the model.
- **Registration:** MoSh's marker frame sits **(0, 0, +10.17 mm)** from `LBS + trans`, identically in every clip (10.16–10.19 mm; |x|, |y| < 0.05 mm).
  - With that shift, the human mesh lands on the floor where the capture says: Standing Split standing foot 0.4 cm and hand 0.1 cm; headstand crown −0.0 cm; Crow exemplar (7.32 s) toe 0.6 cm; Crow mid-hold feet 32–37 cm up.
  - In the clip frame, also add XY (−0.002, +0.3452).
- **First human-side body–body measurement:** Crow's shins are **0.2–0.5 cm** from the upper arms, so the knee shelf is real in the capture.

**Data and machine.**
- No MOYO RGB images are on disk (1.38 TB). The calibrations of all 8 cameras are.
- `/` is the only volume, with **3.8 GB free** after Step 8 (5.2 GB after Step 7, 8.6 GB when this plan was written). Step 8 added 226 MB of retargeted motions (`output/`), 538 MB of Pass C packets (`REVIEW_ROOT/render_c1/`) and about 7 MB under `data/reference_curation/`. The Step 1 store is 16 MB, the Step 3 packets 50 MB per `render_v` in `REVIEW_ROOT`, the Step 5 store 9 MB, the Step 6 store 5 MB and its informed packets 120 MB (in `REVIEW_ROOT/render_v3/`), the Step 7 statics 2.2 MB. Budget: packets ≤ 1.5 MB, workspace ≤ 3 GB.
- The GPU is free now that training has stopped.

**Lessons from the VLM spike (§5).**
- **Marker shadows get read as markers.** The reviewer reported "floor-level dots with no body"; they were the markers' shadows. Evidence glyphs must cast no shadow.
- **Pose names from capsule renders are unreliable.** A Peacock with toes down was named "Chaturanga, high confidence". Ask for a description plus candidate names, never a single verdict.
- **Floor contacts at hand/foot scale were read correctly.**
- **The reviewer found the head-collider offset unprompted.**
- **Camera framing must fit the whole body.** The spike's fixed-distance camera cropped a raised leg.
- **Hands and feet must not look alike.** Crow (hands only, feet 35 cm up) was read as "a standing forward fold on both feet". The support was inverted because hands and feet are same-coloured boxes per side.
- **When the geometry is readable, the reviewer reasons well.** On the headstand it saw the crown hovering and concluded the avatar "could not hold still".
- **Full-body views cannot show a float, even an 11 cm one.** On p1 the reviewer saw the planted-foot markers under the floating foot and still concluded "contacts match". Floor-level per-support close-ups are a requirement of Step 3, not an option. Float detection is not an admissible claim class until Step 4's calibration says so. *(It does under render_v3: precision 0.99 at medium effort, 1.00 at high; §4.)*
- **Cost is dominated by output tokens:** 11.9k out against ~38k cached in, for about $0.40 per packet. Replacing the system prompt saved nothing (p1: 128 s, $0.41). Use terse schemas (word limits per field) and try `--effort`.

## 2. Conventions (every step)

- **Package.** `data/scripts/reference_curation/`, importable with `PYTHONPATH=.:data/scripts`. One module per step: pure functions plus a thin `main()`. Every CLI prints a one-line summary and **exits non-zero on failure** (no printed PASS/FAIL without an exit code).
- **Python.** `../env_isaaclab/bin/python` (torch, mujoco 3.10, scipy, matplotlib, PIL). Record any new dependency in §1 before installing it.
- **Rendering.** `reference_curation.render` renders on the **CPU with OSMesa** (Mesa llvmpipe, already installed) and sets that up itself, overriding an inherited `MUJOCO_GL=egl`. OSMesa is bit-exact across processes; EGL on the NVIDIA driver is not (Step 3's card), and packet ids are content hashes. It is as fast here (~1 s per packet) and leaves the GPU to training. Import `render` before anything else imports `mujoco`: MuJoCo fixes its GL backend at import. `render.gl_info()` records what ran.
- **Tests.** `data/scripts/reference_curation/tests/test_<module>.py`, run with `PYTHONPATH=.:data/scripts ../env_isaaclab/bin/python -m pytest data/scripts/reference_curation/tests -q`. Tests use real clips and **assert numbers the review already measured** (the card lists them), so a regression shows up as a wrong number.
- **Artifacts.**
  - Bulky or regenerable (caches, renders): `output/reference_curation/` (git-ignored).
  - Small or precious (audits, verdict ledger, calibration, labels, release manifests): `data/reference_curation/`.
  - Packets for blind review: under `REVIEW_ROOT` (default `/home/visakii/Documents/moves/reference_review/`, **outside the repo**, see §1). Their private index lives in `output/reference_curation/`.
- **Provenance.** Every JSON record carries `schema_version`, `generator` {module, git rev, file sha256} and `inputs` {path: sha256}.
- **IDs.**
  - `recording_id` is the MOYO name with its punctuation (`…_Crane_(Crow)_Pose_or_Bakasana_-a`).
  - `stem` is the project name.
  - `hold_id` is `<stem>@<frame_hold>` in x0 source frames.
  - `packet_id` is an opaque sha1 of the packet's content.
- **Blindness.** Pass-A packets carry no names, labels or numbers, and their file names are opaque.
- **Never modify** the source data (MoSh fits, pressure ports, source `.motion`) or shipped training artifacts. Always write new versions.
- **Extend a finished step with a new module or store version; don't edit its module.** The audit id hashes `capture.py` and `audit.py`. The committed calibration tables pin that id (`test_calibration_table_replays_the_ledger`), and Pass-B packets carry it. Step 5's store v2 lives in `human_mesh.py` for this reason.

## 3. Step cards

### Step 1 — Capture evidence store

**Value.** One cached API that answers, for every clip, frame and zone, three questions: what did the human do, what does the reference do, and what does the mat say. It reproduces the review's findings for the whole corpus.

**Build.** `reference_curation/{__init__,ids,capture}.py` and `tests/test_capture.py`.

- **`ids.py`:** maps stem ↔ `recording_id` ↔ MoSh / pressure / `.motion` paths, and provides `hold_id()`.
- **`capture.py`:** `build(stem)` writes `output/reference_curation/capture/v1/<stem>.npz` plus `.json`, holding these per-frame `[T]` / per-zone `[Z = 15]` arrays:

  | Array | Meaning |
  |---|---|
  | `marker_min_z` | height of the zone's lowest observed marker |
  | `marker_resid` | median \|obs − fit\| of the zone's markers |
  | `avatar_min_z` | lowest collider surface in the shipped `.motion` |
  | `mat_total`, `mat_cop` | total mat force and COP |
  | `mat_valid_cov`, `mat_valid_body`, `mat_valid_share` | validity columns 0, 1 and 2, **kept separate** |
  | `mat_zone_load` | per-zone attributed load |
  | `mat_unexplained` | total minus attributed |
  | `attr_visible` | `avatar_min_z ≤ 0.06` |
  | `ground_state` | int8: 1 contact, 0 separated, −1 unknown |

- **Marker → zone map (73 MOYO markers → 15 zones):**
  - feet: HEE, TOE, MT1, MT5, ANK;
  - hands: FIN, THMB, IDX/MID/PNK/RNG/THM 3/6, IWR, OWR;
  - head: LFHD, RFHD, LBHD, RBHD, ARIEL;
  - shank: SHN, KNE, KNI, ANK;
  - thigh: THI, KNE;
  - pelvis: F/B/M waist markers;
  - trunk: C7, T10, CLAV, STRN, FSH, BSH;
  - upper arm: UPA, ELB, ELBIN;
  - forearm: FRM, ELB, ELBIN, IWR, OWR.
- **Thresholds, calibrated per zone** on mat-confirmed frames (zone load > 50 N with `attr_visible`, per-body gate column 1): touch = p99 of the loaded lowest-marker height + 0.5 cm; separation = touch + 2.5 cm. Write them into the JSON.

  Preview on the 60 ftC clips (2026-09-28). The loaded distributions are tight:

  | Zone | Loaded p50 / p99 | Implied touch / separation threshold |
  |---|---|---|
  | Feet | 2.9–3.2 / 3.7–3.8 cm | ≈ 4.3 / ≈ 6.8 cm |
  | Hands | 1.7 / 2.0 cm | ≈ 2.5 / ≈ 5.0 cm |
  | Head | 2.8 / 3.2 cm | ≈ 3.7 / ≈ 6.2 cm |

- **Markers measure touch, not load.** An unloaded limb resting on the floor reads the same height (unloaded p5 ≈ loaded p50). So `ground_state` means *touch*, and load comes only from the mat: the attributed load where `attr_visible`, otherwise the unexplained load. Keep them as separate fields.
- **Marker hygiene:** a NaN or a residual > 5 cm makes that zone *unknown* for that frame. Report the counts.

**Read first.**
- `expert_revist/reference_curation_review_2026_09_28/capture_evidence.py` (port it);
- README §3.2;
- `data/scripts/extract_contact_configs.py` (only `ZONES`, `ZONE_ORDER`, `mjcf_body_names`);
- `data/scripts/contact_geometry.py` (only `geom_to_world`, `geom_ground_distance`).

**Tests and exit.**
- name mapping, including `Crane_(Crow)`;
- MoSh frame count equals `.motion` frame count;
- Standing Split -a at 9.0 s: `ground_state[L_FOOT] = 1` while `avatar_min_z ≈ 0.116`;
- Crow -a L_FOOT: contact through 8.0 s, separated by 9.0 s;
- Peacock -a: both feet in contact over 9.0–10.3 s;
- the unreadable pkl (Standing big toe hold -c) yields `capture_available = false`, not a crash;
- the calibrated thresholds land within ±0.3 cm of the preview table above;
- all 60 ftC clips in under 10 minutes.

**Traps.**
- Avatar XY = Vicon XY + (−0.002, +0.3452); Z is shared, with the floor at 0.
- Build on x0 clips only; the x3s/x7s variants map back through `make_hold_extended_clips.splice_index`.
- *(Learned in the build.)* **Marker touch works only for the feet, hands and head.** Elsewhere the markers do not sit on the surface that touches the floor:
  - loaded thigh markers read p50 14–18 cm;
  - trunk markers under a lying back read −1.2 cm, gap-filled below the floor.

  Admission rule: p99 − p50 ≤ 1 cm over ≥ 500 frames from ≥ 3 clips. The other ten zones fail it and stay −1 until the mesh (Step 5). Their stats are in the calibration JSON.
- **No marker is ever NaN**, because occlusions are gap-filled upstream. So hygiene works on the fit residual:
  - A marker with a residual > 5 cm is dropped from the zone's minimum. Example: Plank -a's RANK is off on all 930 frames.
  - The zone becomes unknown only if its *median* residual is > 5 cm. That happens only on Standing Split -a's raised right foot (313 frames), the only unknowns in admitted zones.
- The middle finger carries MID0/MID6, not 3/6.
- **The pressure ports carry the unrepaired kinematics.** On the 8 ft_c-repaired clips these differ from the shipped pose by 3.5–14 cm. `attr_visible` is measured on the port's pose; 6 zone-frames differ, on Koundinya -b. After the Step 8 retarget this difference will be large.
- Only 39 of 60 ports have column 2; `mat_valid_share` is NaN on the other 21.
- `ground_state` is a Schmitt trigger: contact at or below `touch`, separated above `separation`, unchanged in between. It is −1 before the first decisive frame.

**Size.** About 350 lines plus tests.

### Step 2 — Corpus audit and review queue

**Value.** The worklist and the first fix list, plus the progress metrics that later steps drive to zero.

**Build.** `reference_curation/audit.py` and `tests/test_audit.py`.

- **Per hold, from any manifest** (default `data/smpl/expert60/holds_repaired_ftC_posefix.yaml`):
  - `hold_id`, label ground set, and the capture ground set at the exemplar and over the window;
  - `hover_cm` per labelled zone;
  - disagreements {`missed_touch`, `phantom_support`, `hover`};
  - capture-derived boundaries: the first and last frame where the capture's support set equals the label's;
  - `exemplar_stable`: no support change within ±0.5 s;
  - unexplained load;
  - reason codes and severity.
- **Outputs:** `data/reference_curation/audits/<audit_id>/{holds.jsonl, summary.md, queue.jsonl}`. The queue items carry hold, window, pass, questions and priority.

**Read first.** The Step-1 module docstrings and README §3.1–3.3.

**Tests and exit.**
- with the review's pooled thresholds (touch < 3.5 cm, lifted > 6 cm, feet and hands), exactly 15 missed touches in 11 holds and 4 phantom supports in 3 holds over 69 exemplars; with Step 1's calibrated per-zone thresholds, report the new counts in §4;
- 367 of 833 labelled supports above 2 cm;
- Crow -a's capture start about 8.3 s against the label's 7.08 s;
- `summary.md` lists every flagged hold.

**Traps** (found in Step 1).
- **The 15/11 and 4/3 counts need the review's marker map:** `capture.REVIEW_ZONE_MARKERS` (feet without ANK, hands without IWR/OWR), passed through `capture.marker_arrays(..., zone_markers=...)`.
  - The store's own map puts the wrists in the hand zone.
  - Bridge -a's forearms lie on the floor, so under that map its hands read their wrists at 3.5–4.4 cm, and the phantom count falls to 1 in 1 hold (Side Plank -a's top foot).
  - Both counts are pinned in `test_capture.py`.
  - The store's calibrated `ground_state` still reads Bridge's hands as separated at both exemplars.
- 367 / 833 reproduces exactly from the store's `avatar_min_z` (pinned).
- Crow -a's L_FOOT separates at 8.25 s in the store.
- *(Learned in the build.)* **The review's counts need its rule, not only its thresholds.**
  - Its exemplar read is the lowest marker over ±3 frames, with 3.5–6 cm left undecided.
  - The same thresholds through the Schmitt trigger give 17 in 12 and 5 in 3.
  - `--rule pooled` keeps the review's read; `calibrated` is the default.
- **Store heights at 3 decimals of a cm.** Two labelled supports sit at 1.002 and 1.004 cm. At 2 decimals, the > 1 cm count reads 465 instead of 467, and the metrics are counted from the records.
- **Gate the unexplained load on column 0 only.** Column 1 is low exactly when the unexplained share is high.
  - Per-zone `load_n` needs column 0 *and* `attr_visible`; otherwise it is `None`.
  - Peacock -a (34 % coverage-valid), Bridge -a (2–5 %) and Supported Headstand -b (0 %) therefore get no mat numbers. The README's 69 N on Peacock's feet is ungated.
- `ids.load_manifest` cached by path, so a manifest rewritten within one process was served stale. It now keys on size and mtime.
- A hover the markers confirm as touch asks no Pass-B question, because the retarget (Step 8) realises it. So Pass B (151 items) is much smaller than the flagged set (240).
- Standing Split -a's family hold is not in the Pass-A pool: its raised foot is *unknown* at the exemplar (fit residual). Add the pilot holds to Step 4's set explicitly.

**Size.** About 250 lines. Built: 580 lines, because the queue and the summary carry more than planned.

### Step 3 — Evidence packet renderer

**Value.** Anyone can inspect any hold by eye, you or the reviewer, with the same reproducible views.

**Build.** `reference_curation/render.py`, `reference_curation/packets.py` and `tests/test_render.py`.

- **Scene** (MuJoCo; built on OSMesa, see the traps):
  - **a segment palette in which hands, feet, head, arms and legs are all visibly different**, with left and right as two shades of each (the spike's two-colour scheme made Crow read as standing);
  - floor with 10 cm checks and a 1 m scale bar;
  - optional human markers as bright glyphs coloured by body segment, with **no shadow casting**;
  - **vertical drop lines** from the lowest point of each hand, foot and the head to the floor;
  - a human-mesh layer slot, filled in Step 5.
- **Views:**
  - full body auto-framed to the bodies' and markers' bounding box: four azimuths, one high and one **grazing** view (camera height ≤ 3 cm);
  - a zoomed floor-level inset, 0.6 m wide, of each hand, foot and the head when it is near the floor, plus every labelled or flagged support;
  - an optional five-frame temporal strip across a window.
- **Packet:** `REVIEW_ROOT/<render_v>/<packet_id>/{img_*.png, packet.json}`, with the private index at `output/reference_curation/packets/<render_v>/_index.jsonl`.
  - Pass A: images plus the view legend only.
  - Pass B: adds the labels and numeric companions (capture states, heights, loads).
- **Budget:** at most 6 images per packet, each at most 1024 px on its long side.

**Read first.** `expert_revist/contact_balance_investigation/render_pose_views.py` (the scene builder) and the spike renderer `vlm_spike/render_spike.py` in this folder. Packets are built from Step 2's `queue.jsonl` items (frames in `window.key_frames`, inset zones in `zones`): see the *Queue* section of the `audit.py` docstring.

**Tests and exit.**
- image hashes are deterministic for a fixed input;
- every body projects inside every full-body view;
- a pass-A `packet.json` contains no substring of the stem, the pose name or the labels;
- renders exist for the pilot holds: Crow -a; Side Crow -c (both holds); Standing Split -a; Supported Headstand -a and -b; Plow -b; Peacock -a; Upward Plank -a; Warrior II -a; Bridge -a. The questions each one answers are in README §6.

**Traps.** Marker shadows; fixed-distance cameras cropping raised limbs; the render's ghost/centring offsets must never enter a measurement.
- *(Learned in the build.)* **Nothing may be recentred.** The render world *is* the clip frame: no pelvis centring as in the spike. A drop line's length equals the store's `avatar_min_z` bit for bit (pinned), and markers sit exactly at `capture.markers()`.
- **EGL is not bit-exact.**
  - On the NVIDIA driver, a rebuild late in a busy test process came out 1 level off in 2 pixels of a smoothly shaded sphere, so its content-addressed `packet_id` changed.
  - It was deterministic for a given process history, and it took three render-heavy tests together to trigger. `__GL_THREADED_OPTIMIZATIONS=0` and `__GL_SHADER_DISK_CACHE=0` did not help.
  - OSMesa (llvmpipe) is bit-exact across runs and processes and renders a whole packet in about 1 s. `render.py` forces it; `REFERENCE_CURATION_GL=egl` overrides it for debugging, and `gl_info()` enters every build key.
- **No shadows at all.**
  - In MuJoCo only `mjCAT_DECOR` scene geoms skip the shadow pass (categories 0, 1 and 2 all cast), so glyphs are decor.
  - Under llvmpipe, the avatar's own shadow drew a false dark bar on the floor beyond the directional shadow map (±`shadowclip` × extent). Widening the map to cover the floor gave jagged edges and acne streaks instead.
  - The drop lines, the grazing view and the insets carry the float cue, so render v1 has no shadow-casting light.
- **`mjv_initGeom` leaves six fields uninitialised:** `category`, `objtype`, `objid`, `segid`, `transparent` and `camdist` keep whatever the malloc'd scene buffer held. A garbage `segid` crashed segmentation rendering. `render._glyph` sets them all.
- **Clip names are plain English.** "scale", "four", "side", "angle", "hold", "facing", "standing", "forward" and "half" all occur in a stem or a hold name, so the legend says "1 m bar", and "outside" failed on Side Crow (it contains "side"). `packets.leaks()` refuses such a packet, and a test scans every fixed text against every corpus token.
- **Glyph tuning:**
  - Markers glow at emission 0.2; at 0.55 the light (right-side) shades washed out to white.
  - Parts more than 60 cm up get no drop line: a raised foot's 1.7 m line read as a pole in the insets.
- *(From Step 4's calibration.)* **render_v2 and render_v3 change only the drop lines.** Cameras, framing and every other glyph are v1's, so a difference between versions in the calibration table is the lines' doing. Every version's packets stay in `REVIEW_ROOT/<render_v>/`, because the ledger's verdicts refer to them.
  - **v2: no line and no cross within 1 cm of the floor** (`DROP_MIN_M`). The shipped reference is grounded with a 5 mm clearance, so v1 drew a line under every planted hand and foot, against its own legend ("a part that touches the floor has no visible line"). The reviewer read it as a gap: 28 false floats, all at 0.50–0.97 cm.
  - **v2: every other part gets a line up to 15 cm** (`DROP_MAX_BODY_M`, the float range), unless the line would pass through another zone's geom (`drop_lines`); a shin over its own foot gets none. In v1 only the hands, feet and head had lines, and 7 of the 14 forearms 2–5 cm up were read as resting on the floor. With lines, v2 and v3 read none of them so (13 right, 1 abstention).
  - **v3: a line and its cross wear the colour of their part**, glow like the markers, and are 3.5 mm thick. v2's black lines could not say whose they were:
    - a forearm line beside a planted hand was read as the hand's ("Hand box has drop lines", Dolphin -a);
    - a raised foot's line landing next to the standing foot was read as the standing foot's (Tree -b).

    They made 17 false floats, all at 0.50–0.96 cm.
  - The occlusion test costs about 0.2 s per frame, so a packet now builds in about 3 s.
  - **Still open: the capsule avatar has no front or back.** None of the 6 plow holds was named plow, family holds included: 4 were read as a dolphin and 2 as a headstand. Head on the floor, hips up, feet down and forearms on the floor fit these poses without a face or chest cue. A candidate for the next `render_v` if pose identity is ever needed as evidence.
  - **Still open: a limb's two segments share a colour** (upper arm and forearm are both orange, thigh and shin both green). So a forearm's line beside a planted upper arm still read as the upper arm's in v3 (Plow, 3 times). Distinct shades per segment are the next candidate if the near-floor avatar class needs it.
- **Strip frames:** key frames closer than 0.25 s merge (Crow -a's 719 and 722). The capture's boundaries can sit far outside the window. Upward Plank -a's run spans 3.7–27.5 s, because sitting has the same admitted support set, so its strips cover about 24 s.
- **Pilot items:** a hold missing from a pass's pool gets its item from `audit.build_queue([record])` (`packets.item_for`), so windows, key frames and zones match the queue's rule.

**Size.** About 400 lines. Built: `render.py` 842 lines, `packets.py` 518, tests 387 (the views, the leak guard and the Pass-B companions carry more than planned).

### Step 4 — Headless reviewer, verdict ledger and calibration (Pass A)

**Value.** A working VLM-in-the-loop, with **measured** trust for each kind of claim.

**Build.** `reference_curation/review.py` (runner), `reference_curation/verdicts.py` (schema, validator, ingest), `reference_curation/prompts/pass_a.md`, `reference_curation/schemas/pass_a.json` and `tests/test_verdicts.py`. Optionally add `.claude/agents/motion-reviewer.md` for in-session use (`tools: Read`, `model: claude-opus-5-5`, `omitClaudeMd: true`).

- **Runner:** for each pending packet:
  1. `cd` into the packet directory;
  2. run `claude -p` as in §1;
  3. validate the answer;
  4. write `data/reference_curation/ledger/verdicts/<packet_id>.A.<n>.json` with the answer, model, CLI version, prompt and image sha256, cost and duration.

  It is resumable, runs at most 3 in parallel, and takes a `--max-cost` budget. **A rebuild replays the ledger and never re-queries.**
- **Pass-A questions** (blind), each allowing abstention:
  1. a description of the pose, plus up to three candidate names;
  2. floor contacts per zone, for the avatar and for the human markers;
  3. avatar-versus-human discrepancies;
  4. body–body contacts;
  5. physically implausible features.
- **Validator:** checks the schema; zone and body names against the vocabulary; that referenced images exist in the packet; and rejects force or newton claims.
- **Calibration:**
  - Run on about 40 packets whose truth is unambiguous from Step 2.
  - Compute precision, recall and abstention per claim class: floor contact per zone, float detection, body–body contact, pose identity, left/right.
  - Write the admission table to `data/reference_curation/calibration/<render_v>.json`.
  - A claim class becomes **evidence** only if precision ≥ 0.9 at a non-abstained rate ≥ 0.5 over at least 30 items. Otherwise it stays advisory.
  - The set must include folded and inverted poses (the crow family, headstands, plow) and floats of 2–15 cm. Those are where the spike failed.
  - **Expect render v1 to admit little.** The spike scored pose identity 3/6 and avatar contacts 1/5. Iterate on `render_v` against this table before running any batch.

**Read first.** This card, the `packets.py` docstring and §5 of this file. The calibration candidates are Step 2's Pass-A queue items, in stratified priority order; take the top ~40 plus the pilot holds. Their truth is each hold's record in `holds.jsonl` (see the `audit.py` docstring). **Those packets are already built** (Step 3): the pilot holds in both passes, plus the top 40 of Pass A. Find them through `output/reference_curation/packets/<render_v>/_index.jsonl` (`item_id` → `packet_id`, `dir`, `hold_id`); rerunning `-m reference_curation.packets` is a no-op while they are current. *(Built: render_v1 to render_v3; v3 is current.)*

**Tests and exit.**
- validator unit tests: bad names, missing images, extra fields;
- a `--dry-run` with a stub reviewer;
- a calibration table for the current `render_v`.

**Traps.**
- `CLAUDE.md` leakage if the working directory is inside the repo.
- `--bare` needs an API key.
- About $0.4 per 2-image packet at list price, dominated by about 12k output tokens. `--system-prompt` did not help. Put word limits in the schema's fields and measure `--effort medium` against `high` on the calibration set before any batch.
- *(From Step 3.)* A render v1 packet has 4 images, not the spike's 2, so budget more input tokens.
- The prompt reaches the reviewer too, so keep it free of clip-name words. Check it with the token scan in `test_render.test_fixed_texts_use_no_word_of_any_clip_name`.
- Key verdicts on `packet_id`, which is a content hash. Never re-render packets with `REFERENCE_CURATION_GL=egl`: EGL is not bit-exact.
- *(Learned in the build.)* **The CLI checks `--json-schema` as draft 7.** A draft 2020-12 `$schema` key made every call exit 1 before it reached the model. Keep the schema to draft-7 keywords (no `$schema`, `$defs` or `$ref`). `verdicts.validate` checks the same file as draft 2020-12, a superset for these keywords.
- **`--tools` and `--allowedTools` are variadic.** The prompt goes right after `-p`, or it is swallowed as a tool name.
- **`--restricted` is what makes the review blind.**
  - It confines `Read` to the working directory: a probe's read of the repo's `CLAUDE.md` was denied.
  - It ignores the user's settings, whose `effortLevel` is xhigh. That is why the spike thought for ~10k tokens.
  - With `--strict-mcp-config --disable-slash-commands`, the input fell from ~38k tokens to ~6k plus the images.
  - `review.check_blind` also refuses a packet dir inside the repo, a `CLAUDE.md` in or above it, a stray file, or an image whose sha256 differs from the index.
- **Medium effort costs $0.12 per 4-image packet (p50) and takes 25 s**, against the spike's $0.40 and 116 s at xhigh. A calibration pass of 52 packets is about $6.5 and 8 minutes at 3 in parallel.
- **Freeze the prompt per run.** The first runner re-read the prompt file for every call, so an edit mid-batch would have filed two prompts under one key. `Reviewer.contract` now reads it once.
- **Truth (`verdicts.packet_truth`):**
  - Human: the store's `ground_state`, only where it is decisive and stable for ±0.5 s.
  - Avatar: touching at ≤ 1 cm, off the floor at ≥ 2 cm.
  - Body–body: in contact at ≤ 1 cm, apart at ≥ 3 cm.
  - Anything in between is not scored.
  - Exact pair gaps cost 0.55 s per frame, spent on box distances between parts a metre apart. A 10 cm bounding-sphere gate gives identical truth on all 52 frames in a seventh of the time.
- **Pose names need a longest-match resolver** (`resolve_pose`): "Parsva Bakasana" contains "Bakasana", "Setu Bandha Sarvangasana" contains "Sarvangasana", and "Makara Adho Mukha Svanasana" contains "Adho Mukha Svanasana". Both Cobra takes are performed as up-dog (`notes/Contact_config_def.MD`), so "upward facing dog" counts for Cobra.
- **A two-valued class takes the lower of its two precisions.** Render v1 shows why: its "off_floor" claims on the avatar were 0.88 precise, its "touching" claims 0.83.
- **The admission rule needs a claims floor.** Render v3 at high effort "admitted" body–body at precision 1.000, but that precision rested on 4 positive claims. A detect class's answered rate also counts its implicit "no" claims, so it can clear the 0.5 bar without the precision meaning anything. A class's precision must now rest on at least 30 of its own claims (`support`, `CLAIMS_MIN`). That also keeps left/right advisory at medium effort (27/27 claims) and admits it at high (30/30).
- **Iterate the render before the prompt.** Each render_v changed one thing about the drop lines, and each table named what the next one had to fix. The numbers are in §4; the render side is in the Step 3 card.
- **Re-scoring is free; re-querying is not.** The pose aliases and the admission rule changed after the runs, and `-m reference_curation.verdicts` rebuilt all three tables from the ledger. `test_calibration_table_replays_the_ledger` fails whenever a committed table drifts from the ledger.
- **Python block-buffers stdout into a log file**, so the runner prints progress with `flush=True`.
- The optional `.claude/agents/motion-reviewer.md` was not added. A headless call takes 25 s and enforces the schema, and a new agent file loads only at session start.

**Size.** About 350 lines plus the prompts. Built: `verdicts.py` 714 lines (the docstring and the pose-alias table are about 170 of them), `review.py` 367, tests 452, and the prompt 32 lines. Step 3 grew by the drop-line rules (`render.py` 873 lines).

### Step 5 — Human mesh layer (model verified: `smplx_model_check.py` exits 0)

**Value.** Human-body truth for body–body contacts, and the human render the VLM reads best.

**Build.** `reference_curation/human_mesh.py` and `tests/test_human_mesh.py`.

- **Minimal SMPL-X LBS, with no pip install.** It already exists and is verified in `smplx_model_check.py` (`SMPLX`, `rodrigues`, `load_mosh`, `rebuild_markers`); port it, don't rewrite it.
  - Inputs: the model npz (J_regressor, weights, posedirs, kintree), plus the MoSh pkl's `v_template`, `fullpose` and `trans`.
  - **Use `fullpose` as stored.** The hand pose is absolute; never add the hand mean to the mesh.
  - Register each clip with the median world offset of its non-finger markers (expect (0, 0, +10.17 mm)), and assert it.
  - The hand mean (v1.1 file) is needed only to rebuild markers for verification (`rebuild_markers`).
- **Verification:** use the check script's criteria. Non-finger: median < 1.5 mm, with at least 80 % of markers under 2 mm. Fingers: median < 1.5 mm.
- **Human-side contacts:**
  - floor distance per IPMAN segment (`moyo_toolkit/data/essentials/yogi_segments/smplx/`);
  - region–region self-contact: Euclidean distance under 2 cm between non-adjacent segments.

  These feed `human_ground_state` and `human_pair_state` in capture store v2.
- **Render layer:** the human mesh in translucent grey, co-registered with the avatar.

**Tests and exit.**
- marker error below the threshold;
- the Standing Split -a standing sole within 1 cm of the floor;
- the Supported Headstand -b crown within 1 cm;
- the Crow -a knee–triceps contact present mid-hold.

**Traps.**
- `use_hands_mean=True` does **not** mean "add the mean to the pose". The stored hand pose is absolute; only the latent markers need the mean-posed canonical hands. Adding the mean to the pose looks plausible and is 5 cm wrong.
- The +10.17 mm world z offset between `LBS + trans` and the marker frame.
- The Vicon → clip XY offset.
- Posing the full mesh inside an optimiser is slow; skin only the vertices you need (`transforms(..., vids)`).
- `pkill -f <script>` matches its own shell.
- *(Learned in the build.)* **The IPMAN segments cannot be the zones.** They are ten closed parts made for volume. 100 % of each hand's vertices lie inside its forearm segment, each foot inside its lower leg and the pelvis inside the torso. A vertex's zone is its dominant skinning joint (`JOINT_ZONE`), the rule the MJCF body → zone map already follows (the `L_Knee` body is the shin). The IPMAN files are kept as the test of the left/right convention.
- **Store v2 is `human_mesh.load(stem)`, not an edited `capture.py`.**
  - Editing `capture.py` or `audit.py` changes the audit id, which hashes their generator sha, and fails `test_calibration_table_replays_the_ledger`.
  - So `output/reference_curation/capture/v2/<stem>.npz` holds only the four `human_*` arrays and the v1 record's identity, and `load` merges them onto `capture.load`.
  - `render.py` is untouched too. Its `HumanMesh` slot worked as built: MuJoCo keeps the mesh's vertices to 5 µm (the 5-decimal XML).
- **The check script's marker rule fails its own 80 % criterion on 1 of 59 fits** (Lord of the Dance -c, 79.2 %).
  - That rule turns a marker's offset with its vertex's skinning rotation.
  - Attaching the offset to the vertex's surface frame (normal and one edge) passes all 59, so it is the default (`attach="surface"`).
  - The ported rule stays as `attach="skinning"` and reproduces the script's printed numbers exactly (pinned).
  - RBSH, LFSH and MFWT stay 3–15 mm off under both rules: MoSh's attachment is neither. The mesh is not in question.
- **The mesh has no soft tissue, so a loaded part sinks into the floor.** Mat-confirmed medians: pelvis −5.1 cm, back −1.2 cm, back of the head −0.65 cm. So "touch" means "at or below 2.27 cm", never "near 0".
- **One floor threshold for all 15 zones.** It is Step 1's rule (loaded p99 + 0.5 cm) on the feet, hands and head pooled. Skin has no per-zone marker offset, and the other zones' mat-confirmed frames are attributed on the avatar and read wide (thigh p99 4.8 cm, mostly from one clip).
- **The band is 1 cm, not 2.5 cm.** The mesh is smooth: 7 runs under 0.1 s in the corpus. A 2.5 cm band held Shoulder-Pressing -a's left foot "touching" for 3.8 s with its sole 3.6 cm up. That band agrees more with the markers (99.36 % against 99.00 %) only because both then hold stale states in the same band.
- **The two thighs share mesh edges at the crotch**, so their plain zone distance is always 0. A vertex within 10 cm of the other zone, measured along the template surface, is left out of that pair. This trims only the thighs: the next closest non-adjacent pair along the surface is head–upper arm, at 14.4 cm.
- **Head–upper arm is a human pair.** The avatar's `ADJACENT` excludes it as a capsule artifact, but on the skin, arms by the ears is a contact (Low Lunge -a).
- **The head markers cannot see the face.** Peacock -a's chin is 2.6 cm from the floor while its lowest head marker reads 14 cm: no marker sits below the forehead. The mesh sees it; the markers' HEAD state cannot.
- **Speed.**
  - The ported `vertices` would hold 1.3 GB of transforms for 1000 frames, so the full mesh is skinned through BLAS in 64-frame chunks (`posed`).
  - A box gate runs before each pair's KD-tree.
  - That makes about 5 s per 1000 frames. `--all` runs 8 spawned workers with single-threaded BLAS: 65 s for the corpus, about 5 min serial.

**Size.** About 300 lines. Built: `human_mesh.py` 838 lines (about 80 of them the docstring), tests 334 (19 tests).

### Step 6 — Informed review (Pass B) and labels v1

**Value.** The corrected labels: the dataset's contact semantics.

**Build.** `reference_curation/labels.py`, the pass-B prompt and schema, and `tests/test_labels.py`.

- **Pass-B packets** for every flagged hold, plus a random 15 % of clean holds as controls. They show the labels, capture states, human-mesh contacts and avatar geometry.
- *(From Step 4.)* Render them with the current `render_v` (render_v3) and review them at `--effort high`.
  - Evidence classes under render_v3 + high effort: float detection, human floor contact from the markers, avatar floor contact, left/right (`data/reference_curation/calibration/render_v3.json`).
  - Pose identity and body–body stay advisory.
  - The reviewer marks almost every body–body contact `possible`, so body–body roles need the mesh (Step 5) and statics (Step 7), not the VLM.
  - The Pass-B prompt and schema are new, so they need their own calibration run before their claims count.
- *(From Step 5.)* **Human-side truth is `human_mesh.load(stem)`**, capture store v2: the v1 record plus `human_*` arrays.
  - `human_ground_state` `[T, 15]` decides every zone, not only the extremities: touching at or below 2.27 cm on the skin, separated above 3.27 cm.
  - `human_pair_state` `[T, 91]` uses the labels' pair names (`PAIR_NAMES`): all 65 labelled pairs, plus head–upper arm.
  - Both are −1 only where the marker fit fails (median residual > 5 cm) or the fit is unreadable (Standing big toe hold -c).
  - A human pair is skin contact, not a load path. Feet together and thighs together are contacts too, so `target_role` still needs statics (Step 7).
  - The render layer is `human_mesh.render_layer(stem)`, passed to `render.render_sheets(..., human_mesh=...)`. Putting it in packets makes a new `render_v`, whose claims need their own calibration.
- **What the reviewer proposes:**
  - a variant description;
  - confirmation of the source state per zone;
  - the body–body `target_role` (`required_touch` / `allowed` / `incidental`);
  - boundary and exemplar suggestions;
  - whether a repair is needed.
- **Deterministic reconciliation:**
  - body–ground source state comes from the capture (mesh first, then markers, then mat);
  - body–body source state comes from the mesh;
  - `load_path_class` comes from Tier-1 topology (`notes/Contact_label_consequence.MD` §6);
  - `target_role` comes from admitted VLM claim classes, refined by statics in Step 7;
  - conflicts become `unresolved`.
- **Output:** `data/reference_curation/labels/<labels_id>/{annotations.jsonl, holds.yaml}`. `holds.yaml` is a manifest with corrected ground sets, boundaries, exemplars and stable hold IDs.

**Tests and exit.**
- no contradictory roles;
- every label cites evidence IDs;
- body–ground disagreement with the capture is 0;
- the three crow-family exemplars move to capture-stable frames;
- the two Side Crow -c holds get distinct IDs.

**Traps** *(learned in the build)*.
- **The mesh's floor contact has seam artifacts, so the labels read capture store v3** (`sources.py`: v2 plus `human_floor_z` / `human_floor_state`).
  - A zone's lowest vertex can sit on its seam with a grounded neighbour. Standing Split -a at 9.0 s: the palm is pressed 1 cm into the floor, and three wrist-crease vertices (skinning weights 0.53 elbow / 0.47 wrist, 1.9 cm up) keep the left forearm "touching" for 5.6 s, while the wrist joint is 3.5 cm up and the wrist markers 6 cm. The same happens at Plow's, Shoulderstand's and Bridge's elbows (forearms rising from upper arms on the floor) and at Scale's gluteal fold.
  - v3 re-reads the eight limb zones. A vertex within 5 cm of an adjacent zone (along the template surface) is a seam vertex. A seam contact counts for the limb only while the neighbour's core is off the floor and the limb's side is the deeper one, so a kneecap stays a knee contact. Limb contact frames fall 54,362 → 44,782; Dolphin keeps ≥ 94 % of its forearm frames.
  - The trunk, pelvis and extremities keep v2's state. Arbitrating the trunk dropped Bridge's upper back, a real contact (the avatar's thorax).
- **Tier 1 needs two conditions.**
  - "Shortens a path, counted in joints" alone calls side crow's shin-on-the-other-thigh consequential: it skips the knee and hip and re-enters through the other hip, one joint shorter, and the note measured 0 N.
  - "Avoids the common root" alone calls the chair's knees consequential.
  - The rule is a strictly shorter route which, for two chains, avoids their lowest common ancestor.
- **Every consumer reads the legacy `pairs` vector as demanded.** So it carries only `required_touch` pairs and the labels the capture cannot decide (carried). A labelled contact that is touched but not required is *demoted*: annotated, not configured, never deleted. The cost: pose-defining braces (Cockerel's and Scale's lotus crossings) stay demoted until a semantic role class is admitted.
- **A hold id never follows a moved exemplar.** `hold_id` stays the source `<stem>@<frame_hold>`, and the corrected exemplar is `frame_hold`.
- **Exemplar rule:** the proposer's own criterion (minimum smoothed mean body speed), restricted to capture-good frames (support set equal to the ground set, no change within ±0.5 s). The *nearest* capture-good frame would put Crow -a's exemplar at 8.65 s, with the feet still rising.
- **Pass-B packets show the corrected exemplar as the main moment**, so the reviewer judges what ships. So `verdicts.calibrate`, which scores at the audit's source exemplar, cannot be used; `informed.calibrate` scores at the packet's frame.
- **The Pass-B contract is Pass A's five blocks unchanged plus four** (`variant`, `roles`, `timing`, `repair`), so Step 4's validator, scorer and runner apply without edits. The Pass-B-only checks (one role per candidate, a strip panel for the better moment) run at parse time and drop only the failing part.
- **Prompt substring traps.** "considered" contains *side*, "guidance" *dance*, "message" and "usage" *sage*, "understanding" *standing*, and "threshold" *hold*. The packet's `hold` block is renamed `labels` for the same reason.
- **Roles have no truth set, so Step 4's rule cannot admit them**, and none will be built by hand (see the caveat below). The target roles are deterministic (mesh, Tier 1, source label), and the reviewer's roles are attached as advisory.
- **Cost:** an informed packet is about 17 KB of JSON plus 4 images, and a high-effort review costs about twice Pass A's.
- The Step 6 tests share one session fixture (`tests/conftest.py`), which builds v1, v2 and v3 of all 60 clips in about 2 min.

**Caveat: labels v1's roles are not yet the critical contacts** *(2026-09-29; recorded, not acted on).*
- Labels v1's `required_touch`, and so the demanded `pairs`, is the *deterministic* role: labelled, touched by the human, and Tier-1 consequential. That is a physics test, not "the pose needs this contact". For rewards only the **critical** contacts matter, the ones that define the pose; secondary contacts are to be ignored.
- The informed reviewer's roles are the closer notion, and the two disagree on the critical boundary. Over the 465 reviewed pairs the human touches:
  - **both call it critical: 31**, the canonical shelves (Crow's shin on upper arm, the Firefly, Shoulder-Pressing, Koundinya and Side Crow shelves, Tree's foot on thigh, Eagle's thigh cross);
  - **deterministic only: 96**, all secondary: deep-fold by-products 44, the other half of a knee straddle 19, brushes beside a shelf 13, stacked legs 7, optional grips and placements 9, entry-phase shelves 4;
  - **reviewer only: 45**: pose-defining contacts that carry no weight 32 (the lotus crossing, Sirsasana I's hands cupping the head), contacts never labelled 4 (Peacock's elbows in the belly), unstable or unresolved 9.
- The reviewer's critical contacts form **24 pose-by-contact patterns**, consistent across takes: 67 pairs the human makes stably, 29 in family holds. The avatar realises only 35 of them today, and Peacock's elbows and the headstand's hand cup nowhere. These are what a contact reward is for, and what Step 8 has to close first.
- **Human input on contact labelling is kept minimal on purpose**: this plan evaluates the system's ability to build the dataset with agentic methods. So no human truth set or pattern approval will admit the reviewer's roles; admission has to come from the system's own evidence (agreement across independent samples and takes, the capture, statics).
- *(Step 7.)* **Statics cannot be that evidence for body–body contacts.** No `required_touch` pair is statically required (114 useful, 82 redundant), and the reviewer's critical pairs split 22 useful / 9 redundant: a pose-defining contact need not carry load.
- **Deferred:** make critical = the reviewer's `required_touch` and a stable human contact, keeping Tier 1 only as `load_path_class`; group each critical contact with its straddle partner (`any_of`); review the 21 unflagged holds that have body–body contacts but no Pass-B verdict (about $6). Until then read `review.role` beside `target_role` in `annotations.jsonl`; the labels' `summary.md` worklist lists every disagreement.

**Size.** About 400 lines. Built: `labels.py` 1037 lines (the docstring about 115 of them), `informed.py` 480, `sources.py` 316, the prompt 41 lines; tests 584.

### Step 7 — Physics adapters: gated statics and statue witness

**Value.** Physics verdicts that mean what they say, and "required versus useful" for each body–body contact.

**Build.** `reference_curation/statics.py`, `reference_curation/witness.py` and tests.

- **Gated LP** built on `data/scripts/static_hold_lp.py`:
  - ground points only inside the contact band; pairs only inside the gap band;
  - typed statuses: `optimal`, `infeasible`, `numerical`, `iteration_limit`, `support_not_realised`;
  - finite checks;
  - min/max pair load, as a necessity test;
  - optional unilateral joint stops.
- **MuJoCo statue witness:** PD targets = pose + Kp⁻¹·τ_LP, zero velocity, 3 s. It passes if drift < 2 cm and the contacts are unchanged.
- *(From Step 6.)* **Read labels v1, not the source manifest.** `holds.yaml` in `data/reference_curation/labels/<labels_id>/` has the corrected ground sets, exemplars and windows, and `annotations.jsonl` has the roles and evidence.
  - 479 of the 924 configured ground supports are `required_touch`, not `required_support`: the capture confirms the touch but the mat cannot confirm the load, because a floating avatar zone is invisible to the attribution. The gated LP decides these.
  - The 196 body–body `required_touch` pairs are labelled, touched by the human (mesh) and Tier-1 consequential: the candidates for the necessity test. Demoted braces and recorded contacts (`allowed` / `incidental`) are not supports.
  - The informed reviewer's roles are attached as `review.role`, advisory. Its disagreements with Tier 1 are listed in the labels' `summary.md` worklist; look there first.

**Tests and exit.**
- the 0.5 m lift control returns `support_not_realised`;
- an open-pair control carries zero load;
- the Side Crow -c shelf classifies as useful, not required;
- the witness passes on three easy holds.

**Traps** *(learned in the build)*.
- **The old LP answered a different question: it passed 295 of 303 holds within the plant's limits, the gated LP 38.** It loaded the four lowest corners of every labelled box whatever their height. Of its 295 passes, 221 rest on a labelled support the reference does not realise and 29 are infeasible on the supports they touch.
- **Gating is per point, and tilted boxes pay for it.** A realised zone (lowest point ≤ 2 cm) loads only its corners within 2 cm.
  - The shipped references tilt their boxes. The standing feet are supinated about 12° (inner edges 2–3.6 cm up), and Side Crow -c h2's right hand and Crow -a's hands are tilted.
  - So 31 holds with every support realised are infeasible: the COM lies outside the in-band corners (Side Crow -c h2 by 0.8 cm, Plow -b's first frame by 0.4 mm). A welded statue tips on these poses too.
- **Before Step 8, the counterfactual classifies the contacts.** It closes every contact the human makes; a closed ground zone adds its resting face, which is the old solver's patch.
  - It reproduces the investigation's what-if-closed numbers exactly: Side Crow -c h1 0.578 → 0.5135 with the shelf, h2 0.5241 → 0.3606.
  - Closed, 221 of the 226 `support_not_realised` holds and 29 of the 31 infeasible ones become optimal (all 250 within the limits).
  - It is never evidence about the reference: every closed contact and every such result is marked `counterfactual`.
- **"Useful" needs the summed utilisation, not the peak.** On Side Crow -c h1 the peak joint is a wrist (20 N m), which no shelf relieves. So `useful` means the least summed utilisation Σ|τ|/τ_max falls by ≥ 0.02 with the contact. The peak relief is reported beside it.
- **Necessity is strict, and its two tests are LP duals.** `required` means s* without the contact > 1, equivalently a minimum load > 0; `check` asserts they agree to 1e-3 N.
  - A 1 N epsilon broke that duality on two standing holds. Plank -a's left foot is required with a 0.1 N minimum (without it s* is 1.018); Dolphin -a's right foot is required (without it no solution balances) with a 0.5 N minimum.
  - `load_min_n` gives the magnitude.
- **A partial contact set decides nothing.** On a `support_not_realised` hold, the LP over the realised contacts biases the rest toward `required`. Their gated necessity is `undecided`, and the counterfactual decides.
- **HiGHS's simplex returns "model status unknown" when the COM sits on the polygon's edge** (Plow -b's first frame). One retry with the interior point decides it (infeasible), so the corpus has 0 `numerical`.
- **No `required_touch` pair is statically required.** Body–body forces are internal, and the hands and feet carry every pose within the limits: 114 are useful, 82 redundant.
  - The reviewer's critical pairs split 22 useful / 9 redundant. So statics is not the critical-contact oracle either (Step 6's caveat).
  - Ground: of the 479 `required_touch` supports, 82 are required (candidates for `required_support`), 276 useful, 101 redundant and 20 undecided. Of the 445 `required_support` ones, 9 are redundant.
- **"Beyond the plant" means the box hand.** 7 of the 8 `beyond_plant` holds peak at a wrist or hand hinge (10–20 N m): Koundinya ×4, Scorpion -b, Side Crow -a, Shoulder-Pressing. The eighth is Peacock -a's chest.
- **At the training gains the statue buckles, whatever the pose.** Ankle, knee and hip in series give each leg about 270 N m/rad, against m·g·h ≈ 600 N m/rad.
  - Warrior II -a falls at 1–3× the gains and holds from 5×; a welded statue holds.
  - So the witness runs 10× the training stiffness (damping ×√10) with the plant's own torque limits. A pass therefore means the plant can hold the pose statically, not that a policy can: a statue has stiffness where a policy has feedback.
  - MuJoCo's soft-contact creep is not the cause: elliptic cones, `impratio` 10 and `noslip` changed nothing.
- **Joint limits are off in the witness and optional in the LP.**
  - 290 of 303 exemplars violate a hinge range of the XYZ decomposition by > 2° (elbow twist in 145 and 139, lotus knees at 175°).
  - The training plant's D6 limits are soft: the USD's `physxLimit:rot*:stiffness` is 300–500, the MJCF joint stiffness.
  - With the LP's stops, s* falls by ≥ 0.02 on 4 of 54 decided holds.
- **The witness settles, then holds.**
  - Drift is measured after a 0.5 s settle, which must stay under 5 cm. Held poses settle 2.4–3.1 cm (the 5 mm grounding clearance and the tilted boxes), then move 0.2–2 cm; 31 failures settle ≥ 5 cm and 3 drift ≥ 2 cm.
  - Contacts use the three-way truth: a zone touching (≤ 1 cm) must stay down, nothing beyond 2 cm may land, and the LP's realised pairs must stay closed.
  - Incidental pairs are reported, not judged: feet overlapping by 0.9 cm in the pose get pushed apart.
- **4 of 38 holds pass the witness:** Warrior II -a@572 and Uttanasana -a@25, @719, @896. The shipped standing references rest on the outer edges of supinated foot boxes at uneven heights, and a statue tips off them. The witness is positive-only evidence here; after Step 8 it judges the retarget.
- Spawned workers need single-threaded BLAS (`human_mesh._single_threaded_children`): without it, 2 min of system time. The corpus runs in 8 s on 8 workers.

**Still open** *(from the post-build review, 2026-09-29; close before Step 8 uses statics as its acceptance test)*.
- **The physics layer has not been checked against the training plant.** Statics and the witness run in MuJoCo with XYZ hinges; training runs in PhysX with D6 joints and soft limits.
  - The rollouts that `notes/Contact_label_consequence.MD` §7 proposed as the estimator's validation are on disk: `results/Contact_Physics_analysis*/<clip>/rollout.npz`, 6 clips (Crow -a, Side Crow -a, Handstand -a, Scorpion -b, Chair -b, Warrior III -a), 85 MB, 74 kg, with PhysX-measured per-pair forces.
  - Run the gated LP on their recorded frames. Check that the measured ground and pair forces fall inside the LP's load intervals, and that the `useful` pairs are the ones PhysX loads. No GPU is needed.
  - Trap: `pair_force_w` is in simulator body order, not the COMMON/MJCF order of the `.motion` fields. Remap by name.
- **The joint limits are measured only in MuJoCo's XYZ decomposition.** 290 of 303 exemplars violate it by > 2°, but whether PhysX's own joint coordinates show the same violations is unmeasured. Set a few reference poses in IsaacLab and read the joint positions against the limits. Step 8's joint-range handling depends on the answer.
- **The witness is a MuJoCo statue, not the training plant** (README §4.3: MuJoCo first, then IsaacLab). An IsaacLab statue on the 4 held holds and a few failures would show whether its verdicts transfer.

**Closed in Step 8 (2026-09-29; `plant.py`, `isaac_statue.py`, Step 8's traps).**
- PhysX forces: the gated LP's load intervals contain the measured loads on 95.3 % of ground and 98.9 % of pair contact-frames; the pair necessity does not predict which pairs the policy loads (0.339 against 0.325).
- Joint limits: PhysX's coordinates are the exp-map and its limits a hard box; **the soft-limit trap above is wrong** (the USD's limit stiffness is ignored). 285 of 303 exemplars are > 2° outside the box.
- IsaacLab statue: built and run, not usable. It holds nothing, not even poses the MuJoCo statue holds within 1.5 cm, and the cause is not found.

**Size.** About 350 lines. Built: `statics.py` 754 lines (the docstring about 100 of them), `witness.py` 239, tests 313 (19 tests).

### Step 8 — Contact-constrained retarget (fix the float) and Pass C

**Value.** Reference motions that realise their supports: the dataset's motions.

**Build.** `reference_curation/retarget.py` and tests.

- **Per-frame IK**, generalising `repair_armbalance_holds.HoldIK`:
  - supports in capture contact are brought into the floor band;
  - mesh body–body contacts are brought into the gap band;
  - joint angles stay close to the original;
  - motion is temporally smooth;
  - there is no floor or self penetration;
  - every field is regenerated (FK, local rotations, exp-map `dof_pos`, velocities), with per-frame lineage.
- **Validation:** hover distribution, capture agreement, a collision scan, statics (Step 7), and edit budgets on both the mean and the per-body maximum.
- **Pass C:** before/after packets go to the reviewer, which checks the edit is the same asana, looks natural and has no artefacts.
- *(From Step 6.)* **Retarget to labels v1's supports.** Its configured supports float > 2 cm on 461 of 924 (50 %; > 5 cm on 197). That is more than the source's 44 % because the added missed touches float too (Peacock -a's and Plow's feet). The worklist is every ground annotation with `evidence.avatar.realised` false. Chaturanga's head and Bridge -a's hands are no longer supports, so they must *not* be brought down. The reviewer's `repair.kinds` per hold (`float_support`, `head_collider`, `pose_mismatch`) is advisory triage. Use the mesh from capture store v3 (`sources.py`), whose limb zones are free of seam artifacts.
- *(From Step 7.)* **Validate with `-m reference_curation.statics --motion-dir <retargeted dir>`**: the gated LP and the statue witness on every hold, in about 8 s.
  - On the shipped references, 38 of 303 holds are statically holdable on the supports they realise, and 4 hold as a statue.
  - **Flatten the supports, don't only lower them.** The standing references stand on the outer edges of foot boxes supinated about 12° (inner edges 2–3.6 cm up). Tilted hand boxes leave Side Crow -c h2 and Crow -a infeasible, although every support is within 2 cm. The human's soles and palms are flat (the mesh), so a support in capture contact should rest on its face.
  - The counterfactual in `holds.jsonl` (the human's contacts closed) is what each hold should reach once realised: 250 of the 257 unrealised or infeasible holds are feasible within the limits then.
  - **Measure the training plant's joint limits before choosing the retarget's joint-range handling.** 290 of 303 exemplars violate the MuJoCo XYZ hinge ranges (elbow twist most often), but whether PhysX's own joint coordinates show the same violations is unmeasured (Step 7's still-open list). If they do, the references ask for poses the plant cannot reach, and the retarget must bring them into range.
  - **Pilot before the corpus.** Retarget the pilot holds first and run statics on them, after closing Step 7's still-open checks. Once supports are flat, the gated LP and the witness should agree. If the witness still fails most of them, suspect the statue, not the retarget.
  - **Pass C is the reviewer's first test on semantics that only it can judge.** Its admitted classes (float, floor contact, left/right) restate the deterministic evidence, and pose identity is still advisory (precision 0.54 on capsule renders). Ask a relative before/after question over the human-mesh layer, and calibrate that prompt and `render_v` before Pass C's verdicts gate the release.
  - Statics is evidence the labels do not yet carry. Merge it (e.g. the 82 statically required `required_touch` ground supports → `required_support`) on the retargeted references, where the gated verdict replaces the counterfactual.

**Tests and exit.**
- labelled supports above 2 cm fall from 44 % to under 5 %;
- no new penetrations;
- an exact round trip;
- Pass C accepts at least 90 % of edits.

**Traps** *(learned in the build)*.
- **Why the references float, and why a retarget, not a grounding, fixes it.** The shipped clips copy the performer's joint rotations onto a different skeleton: every avatar body's world rotation is its SMPL-X joint's times one fixed axis permutation (`retarget.AXES`, 0.2–2.7° p50). Relative to the pelvis the avatar's ankles sit 5.9 cm higher and its wrists 4.5 cm further out, so a pose with a hand and a foot down cannot put both down with the performer's angles (Standing Split -a's standing foot floats 11.6 cm), and the foot boxes are tilted against the sole.
- **The training plant (Step 7's still-open checks; `plant.py`, `data/reference_curation/plant/`).**
  - PhysX's joint coordinates are the exp-map of the local rotation (8e-7 rad on 13 recorded rollouts; XYZ Euler differs by up to 3.5 rad): the `.motion` `dof_pos` convention is the plant's own.
  - **The limits are a hard box on those coordinates**: at most 1.04° past a range over 34,080 frames, with 6,689 joint-frames pinned at a stop by a saturated actuator. Step 7's "soft limits" trap was wrong: PhysX ignores the USD's limit stiffness and says so ("Stiffness attribute is unsupported for articulation joints").
  - **285 of 303 exemplars ask for a coordinate > 2° outside the box** (290 in the XYZ decomposition; 75,381 of 89,176 frames). The worst, Cockerel -b's knee "160° past", is a knee folded past 180° and stored turned the other way (next trap).
  - **The gated LP explains PhysX's forces**: its load intervals contain the measured ground loads on 95.3 % of 20,015 contact-frames and the pair loads on 98.9 % of 16,001. **Step 7's pair necessity does not predict what the policy loads**: `useful` pairs carry load in 0.339 of hold windows, `redundant` ones in 0.325.
  - **Self-collision is every pair but a parent and its child** (253 of 276; `contype = conaffinity = 1` everywhere, no excludes). The labels' `BODY_PAIRS` (218, non-adjacent zones) leave out chest–upper arm, pelvis–spine, head–chest and head–upper arm.
- **A knee folded past 180° is stored turned the other way.** Clipping that exp-map coordinate into the knee's range straightens the leg; the other representative `v − 2π v/|v|` keeps it folded. `nearest_representative` picks, per joint and frame, whichever clips closer: Cockerel -b's right knee, 914 frames, 143° closer, the only joint in the corpus where the choice matters.
- **Guard every pair the plant collides, on every frame, at every step.** The first corpus run guarded only the labels' zone pairs, on frames near the reference's own contacts: closing Peacock -a's forearms on its belly drove both upper arms 11.6 cm into the chest box, overlaps moved to the frames the monitors did not cover, and the corpus gained 10,127 deep-overlap pair-frames. The guard now re-finds overlaps in the current pose at every evaluation (0.6 s on a 1,790-frame clip) with a 1 mm scale.
- **Box against box needs the exact depth.** 26 surface points against the other box's SDF give a gradient toward the nearest *face*, which for two feet side by side is the top one: the solver pushed the feet into the floor instead of apart and stalled (Crow -a's ankles 2.2 cm deep, the energy rising along its own gradient). The separating-axis depth (`_box_box_sat`) is exact for intersecting boxes and its witnesses make the derivative exact (4e-11 against finite differences). **Residual:** at flat-face contacts (hands clasped, feet together) the depth has a concave kink (the |n·b| of a face parallel to the axis) where Levenberg-Marquardt's model mispredicts and stalls: a few hand–hand or foot–foot pair-frames stay 1.0–1.3 cm deep (3 in the pilot).
- **The solver.** L-BFGS-B on the same objective did not converge (1–5 mm scales are stiff). Levenberg-Marquardt with analytic Jacobians and one banded solve per step converges in 23–60 iterations, 70–400 s per pilot clip on one thread. Workers must stay single-threaded: a test process with 14 BLAS threads beside 8 workers ran past 30 min on a clip a worker solves in 5.
- **The support plan reads the mesh per box face.** Nearest-vertex correspondence mixed a corner's top and bottom surfaces and flipped toe boxes. Each face now reads the human vertices whose normals face its way (within 50°); the resting face is the one pointing down in the shipped pose; it rests flat when the human spread under it is ≤ 3.5 cm (the MoSh heel reads 2–3 cm under a flat foot), else on its lowest edge (the toes). Supports, off-floor zones and pairs fade in and out over 10 frames and ignore runs under 6: a term switched on in one frame added 115 m/s² of acceleration.
- **Without an anchor the edit slides.** The floor terms act on heights: the first solves slid Warrior II 35 cm sideways and rotated Plow 49°. A no-slip term dragged the body along pinned corners (real supports pivot) and was removed; every body is anchored to its shipped position (5 cm scale) and the edit is bounded (45° per joint, 25 cm and 30° for the root). With a 2 mm support scale the solve flipped a hip 136° to buy a few millimetres.
- **The head collider refuses a crown.** Its 9.5 cm sphere sits about 20 cm short of the crown, so while the head points down (up axis within 60° of down) its support is not targeted and is reported (`head_blocked`: 1,133 and 1,115 frames of the two supported headstands, 88 of Plow -b). Statics calls those 6 holds `support_not_realised`; the exit metric counts them apart.
- **Flat supports make the LP and the statue agree about stability, not about contacts** (the pilot's 59 holds; the same counts in the corpus statics on the final retarget). The gated LP finds 52 of 59 holds holdable within the plant's limits (34 held + 18 feasible; 6 crowns not realised; Peacock -a@619 beyond them at s* 1.04), against 7 of 59 on the shipped references. The statue holds 34 of 52 by Step 7's rule and 48 of 52 by motion alone (settle < 5 cm, drift < 2 cm). The 4 that tip are arm balances (Crow -a@439, Side Crow -c@606 and @1296, Peacock -a@1183), where a statue has no balance feedback. Ten of the other 14 are static supine holds (Plow, Bridge, a headstand; drift ≤ 0.04 cm) failed by the contact rule because an unloaded resting zone rose past 1 cm.
- **The IsaacLab statue is not usable yet** (`isaac_statue.py`, results in `output/reference_curation/statue/`). At 10× gains it holds 0 of 38 shipped poses and **0 of the 52 final pilot poses MuJoCo's statue holds within 1–1.5 cm**: they settle ≥ 8.9 cm (median 33 cm) and fall. At the first step a shoulder spins to 7.4 rad/s with the drive applying 0 N m, far from its limits; a T-pose falls; Warrior II holds only at 300×. The gains read back from PhysX are the configured ones and the joint order is mapped by name, so the cause is not found. The MuJoCo statue remains the only witness.
- **Pass C** (`edits.py`, `prompts/pass_c.md`, `schemas/pass_c.json`, `data/reference_curation/calibration/pass_c/render_c1.json`).
  - The controls are built from the shipped references with machine truth: identity 12, swap 32 (another pose family's exemplar on the same pelvis), lift 30 (+7 cm), sink 20 (−6 cm), kink 20 (an elbow or knee 80° past its range); 114 packets, $14.76 at high effort.
  - **"Not the same pose" and "closer to the human" are admitted** (precision 1.00 on 32 and 42 claims: every swap called different, no identity, lift or sink). **Artefacts are advisory**: precision 1.00 but 15 claims, recall 0.375 (9 of 20 sinks, 6 of 20 kinks). Pass C cannot vouch for "no artefacts"; the collision scan, the floor minimum and the joint box do.
  - A kink is not scored as a pose change (the prompt calls a limb in a clearly different place a different pose), and artefacts count one-sided (on the edit and not on the shipped side): the shipped references carry their own.
  - **The edits: 279 of 300 accepted (93.0 %), 57 of the pilot's 59.** 297 are called the same pose. Of the 21 rejections, 17 find the shipped version closer to the human with the same pose, 3 call it a different pose, and 1 names a new major artefact.
    - The lotus clips account for 7 (Cockerel -b 5 of 11, Scale -a 2 of 6); the plant's knee range cannot fold a lotus. Without them, 269 of 283 (95.1 %).
    - The rest are binds and arm balances where the joint box or the collision guard moves a limb away from the grey body: Eagle -a 3 of 7, Tree -a 2 of 6, Koundinya 2 of 9, and single holds.
  - **Drop lines are drawn under any part 1–15 cm up, so an edit that lowers a limb the human holds up reads as floating.** Peacock -a@788's left leg went from 17 to 13 cm (the human's is at 17) and gained lines, and was rejected for it. Plow -b@1384's rejection is a real trade-off: its shins went from 19 to 13 cm and its pelvis 5 cm lower to put the shorter legs' feet down. A render_c2 that draws lines only under the zones the human touches would need a new calibration.
  - The first corpus review (on motions that still jerked) accepted 272 of 300. The 71 packets of the 13 re-solved clips were reviewed again; unchanged clips keep their verdicts, because their motions are bit-identical and so are their packets.
  - **Rendering the same panel twice left 30 bytes of stale memory in an image.** A clip's last frame repeats in the time strip; the second draw of that panel differed between builds (11 of 300 packets). `render_item` now draws each distinct panel once, and a forced rebuild reproduces every packet id. In all, Pass C cost $62.66 for 498 verdicts, including the superseded ones.
  - A swap's other pose is picked by index into the pool, so changing the pool (dropping Standing big toe hold -c, whose fit is unreadable: `retarget.has_human`) re-rendered 29 swaps and orphaned their packets.
  - Packet ids hash the pixels. Rebuilt in single-threaded render workers (`--workers 8`), 2 of the 59 pilot edits differed from the single-process build in 10 pixels at one image's corner (the camera fit's float round-off follows the BLAS thread count), so their verdicts did not carry over and they were reviewed again.
- **Where the reference buries one limb in another, the edit jerks, and it is only partly fixed.**
  - Solved with every term at full weight, a limb the shipped reference has inside another (a forearm in a shin, a lotus foot in a thigh, legs on the arms) leaves each frame by its own nearest side. Neighbouring frames that leave by different sides jump between them.
  - The first final corpus had 28 clips with frames on which a body accelerates > 100 m/s² and the reference does not: Standing Forward Bend -a's elbow at 536 m/s² (its RMS 1.3 → 23.5 m/s²), Scale -a's hand at 1,950 m/s².
  - A 30× smoothness alone traded the jerks for overlaps up to 8 cm. Bringing the pair and penetration terms in by continuation (1 %, 10 %, 100 %) under a 5× smoothness (`solve_gently`) removed them on most: Forward Bend 23.5 → 1.7 m/s² and Side Crow -b 24.9 → 2.6, with no new overlap and every pair still closed.
  - `retarget_clip` re-solves every clip whose direct solution jerks (`spike_frames`) and keeps the smoother result: gentle on 13 of the 28.
  - **24 clips still add jerk frames**, 1–33 each and 191 in all (321 frames over 100 m/s² against the references' 130). The worst are Firefly -a (+33), Firefly -b (+26), Cockerel -b (+22), Scale -a (+21) and Side Crow -a (+20): arm-balance transitions and lotus binds, where supports, pairs and collisions still conflict.
  - Acceleration RMS rises by ×1.26 at the median, ×1.9 at p90 and ×3.7 at worst. Pass C does not see jerk (its strip is ±0.5 s apart), so `frames_acc_gt_spike` in the records is the gate.

**After Step 8, before Step 9** *(2026-09-30)*.
- **The per-hold release gate** (`reference_curation/gate.py`, `tests/test_gate.py`; `data/reference_curation/gate/…gate_v1.eb652a8568/`) reads labels v1.1, the statics on the retarget, Pass C and the retarget's lineage, and decides every hold: **235 pass, 9 mask, 26 flag, 33 exclude** (family holds 53 / 2 / 11 / 4).
  - Excluded: the carried -c clip, 9 unresolved labels, 3 not-the-same-pose, 5 statics beyond the plant or infeasible, and 18 holds that cannot be held within the torque limits without a support the plant cannot realise (17 crowns, 1 lotus pelvis).
  - Masked (the crown taken out of every target, the hold kept): 9 crown holds that the realised supports alone hold (`realised_s_star` ≤ 1).
  - Flagged: shipped closer (Pass C) 18, a jerk in the window 17, a new overlap 6, over the edit budget inside the window 8.
  - The transitions carry 135 jerk frames and 138 new overlap pair-frames, per clip in `clips.jsonl`.
- **Corpus ftR** (`data/scripts/build_expert60_ftR.sh`, `retarget_corpus_manifest.py`, `check_expert60_ftR.py`) is the ftC pipeline rebuilt on the retargeted motions with everything else held. It covers the manifest, 180 hold-extended motions, the package, the hold graph, the MOYO mat attribution and on-mat gate on the new poses, and the physics tables. `build_physics_tables.py` gained `--pressure-dir` and `--pressure-gated-dir`.
  - **All 20 identity checks pass.** The graph is identical to ftC's, and the measured force, COP and coverage are bit-identical to ftC's port.
  - **The mat vouches for the new geometry:** loaded frames the attribution explains ≥ 90 % go from 70 % to 87 %, and the attributed-vs-measured COP residual p90 goes from 28.2 to 6.0 cm (Standing Split -a/-b and Half Moon -a from 0.13–0.27 explained to 1.00). Share-valid load signatures go from 651 to 765.
- **The ft_c probe** (`probe_ft_c/run_probe.sh`, `analyse_probe.py`, `summary.md`) runs ft_c's `last.ckpt` with its own evaluator (2250 steps, not the 600-step default) on both corpora.
  - Success falls 0.961 → 0.856 and failed motions rise 7 → 26. Connective clips improve (score 0.951 → 0.994) and Upward Plank gains 0.46; single-leg clips collapse (0.998 → 0.804: Tree -a/-b, Lord of the Dance -c, Standing Split -a), and so do the supported headstands and Koundinya.
  - **This policy was tuned to the shipped references, so the probe cannot separate a worse reference from a shifted one.** Standing Split -a's retargeted COM is inside its supports at every frame where the shipped one was 20 cm outside, and the policy still falls.
  - The added jerk frames are not where tracking fails (median error 18 cm near one against 14 cm elsewhere; no first failure within 1/6 s of one).
  - **It did expose a retarget defect: planted supports slide.** The slowest material point of a support the human keeps on the floor moves at median 0.7 → 3.1 cm/s (clip p90 4.5 → 16.4 cm/s), and 9 % → 37 % of touching frames exceed 5 cm/s. Tree -a's standing foot drifts 7.5 cm in 10 s against 0.4 cm shipped. `retarget.py` constrains supports vertically only; its no-slip term was removed because pinning corners dragged the body.
  - The slide does not correlate with the probe's score drops (Spearman −0.00). A velocity-matching term (keep each planted point's horizontal motion to the shipped one's, which allows pivots) is the fix.
  - In single support the COM also sits exactly at `BALANCE_MARGIN_M` (Tree -a: 1.5 cm from the flat foot's edge from lift-off), the balance term's minimum.

**Size.** About 450 lines. Built: `retarget.py` 1,876 lines (its docstring about 70), `edits.py` 715, `plant.py` 394, `roles.py` 161, `isaac_statue.py` 164, tests 402 (22 tests).

### Step 9 — Training artifacts v2

**Value.** Training-ready artifacts, with their consumers fixed.

**Build, as several small PRs:**
- **Hold extension v2:** keep real-frame pressure; masks on inserted frames.
- **Graph v2:** stable IDs; manual goals resolve the side-specific segment, with no 0.5-vote union.
- **Physics tables v2:**
  - column 1 for the absolute thresholds, column 2 only for shares;
  - the attribution-visibility gate on "unloaded";
  - checks on `pair_names`, fps and hashes.
- **Contact-target sidecar:** roles plus masks.
- **Runtime identity checks** that fail loudly.
- *(From Step 6.)* Labels v1's `hold_id` is the stable id. It is the source `<stem>@<frame_hold>` and never follows a moved exemplar, so graph v2 keys on it. The manifest's `pairs` carry only demanded contacts: the ground set, the `required_touch` pairs, and the carried labels listed in `labels.unverified` (masks). Every other role, the `any_of` groups (`alternative_group`) and the evidence are in `annotations.jsonl`, the sidecar's source. Its `required_touch` is still the deterministic role, not the critical contacts: see Step 6's caveat before compiling contact rewards.
- *(From Step 7.)* The statics `contacts.jsonl` gives every configured contact a load interval, a representative load and a necessity (`necessity_basis`: gated or counterfactual). It is the source for load-share targets (κ ∝ load share, `notes/Contact_label_consequence.MD` §8). No `required_touch` pair is statically required, so "required" body–body targets cannot come from statics either.
- *(From Step 8.)* **Build on the retargeted references and labels v1.1, never on the shipped clips.**
  - Labels v1.1 (`data/reference_curation/labels/holds_repaired_ftC_posefix.labels_v1_1.<hash>/`, `roles.py`) points every clip's `source` at its retargeted motion in `output/reference_curation/retarget/<retarget_id>/`, next to a `<stem>.lineage.npz` (targets, refused crowns, solved coordinates, per-body displacement).
  - Every `.motion` field is regenerated. Velocities come from the converter's routine, and `rigid_body_contacts` is geometric (lowest surface ≤ 1 cm), not the shipped 0–16 %-precise heuristic.
  - **Hold-extension variants must be regenerated from the retargeted x0 clips, not copied.**
  - Standing big toe hold -c's fit is unreadable, so its shipped motion is carried (`labels.retargeted: false`). The supported headstands' 6 crown holds cannot realise the head support on this plant (the head collider); they are `support_not_realised`, and so is any target on them.
  - Pass C's artefact class is advisory. Gate the release on the collision scan, the floor minimum and the joint box, with Pass C's admitted same-pose verdict beside them.
  - The statics of the retargeted references (`statics_id` in labels v1.1) are gated, not counterfactual, wherever the supports are realised. That is the load-share source for κ.
  - **Per-hold release decisions Step 8 leaves open** (every number is in the retarget's `clips.jsonl`, the statics' `holds.jsonl` and the Pass C index and ledger):
    - Cockerel -b and Scale -a: the plant cannot fold their lotus, and they carry 145 new overlap pair-frames plus 7 Pass C rejections.
    - 24 clips with jerk frames (`frames_acc_gt_spike`).
    - 8 clips over the edit budget.
    - 21 Pass C rejections.
    - 28 holds with an unrealisable support (27 crowns).
    - 2 holds beyond the plant's torque limits.
- *(From Step 7's review.)* **Decide the critical contacts before compiling any contact target.** Statics cannot admit them. The machine route left is the reviewer agreeing with itself across independent samples and takes (Step 6's caveat). Until that is decided, the reward-target part of the release has no admitted body–body targets.

**Tests and exit.** A package round trip, a runtime load, and a weight-0 smoke run (256 envs, 4 epochs).

### Step 10 — Release and expert training

**Value.** The expert.

- Compile an immutable release: `data/reference_curation/releases/<id>.json`, with the heavy files under `data/smpl/reference_curation/<id>/`.
- Validate it.
- Train under a new experiment identity with fresh normaliser statistics.
- Evaluate on the original corpus plus the realisability metrics.
- Later, Pass D reviews policy rollouts with the same packet machinery.

**After Step 10** *(2026-10-05)*.
- **Step 10's expert was carried into the graph-growth round** ([`../graph_growth_2026_10_03/PLAN.MD`](../graph_growth_2026_10_03/PLAN.MD)):
  - e15500 ([`../expert56_v2_e15500/README.MD`](../expert56_v2_e15500/README.MD));
  - **G1**, an AMP warm start: posture 40 % closer to the human
    ([`g1_amp_eval/README.MD`](../graph_growth_2026_10_03/g1_amp_eval/README.MD));
  - **release v3** `holds_repaired_ftC_posefix.release_v3.2f132f4299`: release v2, unchanged, plus 28 synthesised
    transitions of five new edges, spliced between real clips (crow → handstand, handstand → crow, crow → plank,
    crow → chaturanga, handstand → chaturanga);
  - **G3**, G1 fine-tuned on release v3.
- **G3 was stopped by the user at epoch 3,420 and evaluated**
  ([`g3/README.MD`](../graph_growth_2026_10_03/g3/README.MD)). Use
  `results/smpl_yogi_v2_expert56_g3_2f132f4299/epoch_3420.ckpt`; `score_based.ckpt` (epoch 3,000) also holds every
  edge.
  - **Edges:** all 84 synthetic motions track their transition and hold the destination with its supports (0 of 84
    before G3). On command, from a held source pose, each edge succeeds on 163 of 163 replicas, and commanding the
    clip's own continuation instead also succeeds: the goal is causal at the hubs.
  - **From standing:** a real hands-only crow, and standing → crow → handstand on 94 % of replicas. The direct
    handstand, Pincha, headstand and side-plank forks fail; the side-plank fork was lost from epoch 2,500.
  - **Human clips:** every group floor, drag (36.7 J), jerk, realisability and posture pass. **Substitutions fail
    (7.7 against ≤ 6)**, the holds G1 already flipped in and out of. Supported Shoulderstand -a is fixed (0.44 →
    1.00); Half Moon -b and Scorpion -a are unreliable at this checkpoint.
  - **The training panel misreads prone goals:** its heading-chart pose error scored the learnt crow → chaturanga
    and handstand → chaturanga edges at 0. Use the best-yaw metric (`g3/run_panel_bestyaw.py`) for any prone or
    inverted goal.
- **Decision (user, 2026-10-05): skip the perturbation fine-tune and go directly to Lane S.** The expert's renders
  looked promising, so card E4 and run G4 are not run. The student is distilled from G3's epoch 3,420 (PLAN.MD §4,
  "Lane S": S0 → S1 → S2/G5 → S3/G6).
  - **The Lane S cards were written assuming G4.** Where they say "G3/G4" or "G4", read G3 epoch 3,420.
  - **S2's "pushes on, at G4's final magnitude" has no magnitude to take.** The teacher was never trained under
    pushes, so its DAgger labels on pushed states are untested. Distil without pushes, or first measure G3's own push
    recovery with E4's perturbed battery and push only where it recovers.
  - **The student cannot exceed its teacher here:**
    - the handstand is reachable from standing only through crow;
    - the side-plank fork is lost;
    - the substitution holds (Eagle's kickstand, Peacock's head, Koundinya -b) are what the expert does.

## 4. Status

| Step | State | Date | Measured / notes |
|---|---|---|---|
| 0: checks, spike, plan | done | 2026-09-28 | §1, §5 |
| 1: capture store | done | 2026-09-28 | `reference_curation/{ids,capture}.py`; 19 tests pass in 7.5 s. `--all`: 60 clips, 89,176 frames in 3.2 s, 16 MB. Calibration in `data/reference_curation/calibration/capture_v1.json`, column-1 gate ≥ 0.9. Touch / separation (cm): L_FOOT 4.32 / 6.82, R_FOOT 4.23 / 6.73, L_HAND 2.51 / 5.01, R_HAND 2.47 / 4.97, HEAD 3.65 / 6.15. **Admitted zones: feet, hands and head only.** The other 10 stay −1 (see the card's traps). Unknown: 0.07 % of admitted zone-frames. Reproduces the review: 367 of 833 floating supports; exemplar audit 15 missed in 11 holds and 4 phantom in 3 (review marker map; see Step 2's card) |
| 2: audit and queue | done | 2026-09-28 | `reference_curation/audit.py`; 21 tests pass (40 with Step 1, in 12 s). It audits 303 holds in 1.3 s into `data/reference_curation/audits/holds_repaired_ftC_posefix.calibrated.69fac8b873/` (1.0 MB). **The pooled rule reproduces the review hold by hold:** 15 missed in 11 and 4 phantom in 3, over 69 exemplars. **Calibrated rule** (per-zone thresholds, the store's marker map, head decided), family exemplars: **18 missed in 12 holds and 6 phantom in 4.** It adds Koundinya -b's right foot and head (0.8 cm), a second Dolphin -a hold, Chaturanga's `HEAD:G` (human head 26.2 cm) and Bridge -a's right hand at 14.13 s. Over all 303 holds: 99 missed in 57 holds and 13 phantom in 11. Hover > 1/2/3/5 cm: 467/367/237/120 of 833, exact and per zone as in README §3.1. Crow -a's capture starts at 8.25 s against the label's 7.08 s; Side Crow -c's at 7.98 s and 22.08 s against 6.78 s and 21.00 s. The §3.1 worst-holds table reproduces to ≤ 1 N and 0.05 cm. **Flagged 240 of 303** (high 16, medium 68, low 156). **60 holds (13 of them family holds) never show the label's support set in their window**, mostly feet the label omits. 15 labels start early, the worst by 4.3 s (Firefly -b; the mat has 200–330 N on each foot), 4.2 s (Shoulder-Pressing) and 2.7 s (Firefly -a). 18 exemplars are unstable (3 family); 35 holds carry > 100 N unexplained. Queue: Pass B has 124 flagged holds plus 27 controls; Pass A has 273 candidates in stratified order (arm balance 46, inversion 43, float 5–15 cm 19, float 2–5 cm 111, other 54) |
| 3: packet renderer | done | 2026-09-28 | `reference_curation/{render,packets}.py`; 17 tests pass (57 with Steps 1–2, in 45 s). **Built 67 packets** in `REVIEW_ROOT/render_v1/`: the 15 pilot holds in Pass A and in Pass B, plus the top 40 of Pass A (3 are also pilots). Each has 4 images (≤ 1024 px). Size p50 727 KB, max 937 KB (budget 1.5 MB), 50 MB in all. Build time p50 1.2 s per packet on the CPU. **Rendering is OSMesa (llvmpipe, Mesa 22.2.5), bit-exact:** a full rebuild in fresh processes reproduced all 67 packet ids, and EGL failed this (card traps). `Camera.project` matches MuJoCo's render to < 0.5 px (measured < 0.25). No avatar geom touches any full-body panel border in segmentation (all 15 pilots). The floats the spike missed are drawn as measured: Standing Split -a's standing foot has a drop line of exactly the store's 11.45 cm at 9.0 s (window median 11.6), with the human's foot markers on the floor under it, and Supported Headstand -b's head hovers 13.5 cm. Pass-A packets carry only images, panel texts and the legend, and `leaks()` blocks names, labels, frame numbers and times. Pass B adds the labels and per-frame store numbers over the same images. Deviations from the card: no shadows, drop lines only below 60 cm, OSMesa instead of EGL. **render_v2 and render_v3 (from Step 4's calibration) change only the drop lines** (card traps). The same 67 packets exist for each version under `REVIEW_ROOT/render_v{1,2,3}/`, 50 MB each, built at p50 3.2 s per packet; 18 tests |
| 4: reviewer and calibration | done | 2026-09-28 | `reference_curation/{verdicts,review}.py`, `prompts/pass_a.md`, `schemas/pass_a.json`; 18 tests pass (76 with Steps 1–3, in 125 s). **Reviewer:** headless `claude -p`, made blind by `--restricted` (see §1 and the card). **208 calls, 0 invalid or failed, $27.26 at list price.** **Calibration:** the same 52 Pass-A packets for each `render_v` (the queue's top 40 plus the 12 other pilot holds: arm balance 11, inversion 11, float 5–15 cm 13, float 2–5 cm 8, other 9), one verdict each, replayed from the ledger into `data/reference_curation/calibration/render_v{1,2,3}.json`. Precision (answered rate) at medium effort, E = evidence: float detection v1 0.723 (1.00) → v2 0.813 (0.99) → v3 **0.986 (0.97) E, recall 0.96**; avatar floor contact within 15 cm 0.833 → 0.889 → **0.967 E**; human floor contact from the markers 0.991 E → 0.982 E → **0.991 E**; left/right 1.000 on 15 / 23 / 27 answered claims, advisory (under 30 claims); pose identity 0.56 / 0.58 / 0.54, advisory; body–body advisory (3–7 claims). **The render iterations are the result.** v1 drew a line under every planted part (the reference's 5 mm clearance): 28 false floats. v2 removed lines within 1 cm and added lines under other parts up to 15 cm, and its black lines were then attributed to the wrong part: 17 false floats. v3 colours each line like its part: 1 false float in 72 claims. **Under render_v3, high effort:** float 1.000 on 72 claims (95 % lower bound 0.949, recall 0.97: 41/41 floats of 5–15 cm and 31/33 of 2–5 cm), avatar floor 0.976, human floor 0.991, and **left/right admitted** (1.000 on 30 claims). High costs $0.135 and 33 s per packet against medium's $0.121 and 24 s (+12 %); pose identity is the same 0.538, with 33/52 against 29/52 in the top three. **Use high effort for batches.** Advisory findings: none of the 6 plow holds was named plow (4 were read as a dolphin, 2 as a headstand: the capsule avatar has no front); the reviewer marks almost every body–body contact `possible` (3–4 clear claims in 52 packets), so body–body evidence has to come from Step 5's mesh; unprompted, it flagged the head markers floating above the head sphere in 24 of 52 packets (README §3.6). The ledger holds 208 files (1.7 MB) |
| 5: human mesh | done | 2026-09-28 | `reference_curation/human_mesh.py`; 19 tests pass (95 with Steps 1–4, in 155 s). **Capture store v2** is v1 plus `human_min_z` and `human_ground_state` (all 15 zones) and `human_pair_gap` and `human_pair_state` (the 91 non-adjacent pairs, which include all 65 labelled ones). `--all`: 60 clips, 89,176 frames in 65 s (8 workers), 8.9 MB; calibration in `data/reference_curation/calibration/capture_v2.json`. **All 59 readable fits register and verify:** offset z +10.13 to +10.19 mm, \|xy\| < 0.05 mm; non-finger median ≤ 0.67 mm, ≥ 83 % of markers under 2 mm, fingers ≤ 0.20 mm (surface attachment; the check script's rule fails Lord of the Dance -c at 79.2 %, see the card). **Floor:** touch 2.27 cm, separation 3.27 cm on every zone (loaded extremities p50 0.39, p99 1.77 cm over 77,404 mat-confirmed frames). The mesh agrees with the marker states on 99.0 % of 439,127 decided extremity frames (L/R foot 98.7/97.6 %, hands 99.6 %, head 99.5 %). **The card's truths:** Standing Split -a's standing sole is 0.4 cm up at 9.0 s while the avatar's foot floats 11.45 cm; Supported Headstand -b's crown is at −0.0 cm at 15.0 s (the avatar's head > 13 cm); Crow -a's shin–upper-arm gaps are 0.51 / 0.47 cm at 9.5 s, in contact on ≥ 95 % of the hands-only run (8.25–12.0 s), and its toe is still down at the label's exemplar (0.63 cm at 7.32 s). Most frequent human pairs: feet together 22.7k frames; thigh–torso 14.5k + 9.9k (the crow family, firefly, garland, forward bends); thighs together 13.8k. **Render layer:** a translucent grey mesh in the clip frame, every vertex inside every full-body view; `--render` draws any frame (Standing Split's grey foot on the floor under the floating blue box). Deviations from the card: zones from the skinning weights, not the IPMAN segments; a 1 cm band; head–upper arm included; store v2 built in `human_mesh.py` (card traps) |
| 6: Pass B and labels v1 | done | 2026-09-28 | `reference_curation/{sources,labels,informed}.py`, `prompts/pass_b.md`, `schemas/pass_b.json`; 27 tests pass (122 with Steps 1–5, in 314 s). **Capture store v3** (`sources`): seam arbitration of the limb zones' mesh floor contacts, 60 clips in 40 s. Limb contact frames fall 54,362 → 44,782 (forearms −6,597, upper arms −1,779, thighs −1,166): Standing Split -a's wrist-crease "forearm" 341 → 0, while Dolphin keeps ≥ 94 % of its forearm frames (card traps). **Labels v1** in `data/reference_curation/labels/holds_repaired_ftC_posefix.labels_v1.249a920cea/` (3.7 MB, 1,931 annotations; the contract check and **disagreement with the capture: 0**, on every frame of every window). 303 holds: 222 kept, 11 moved, 61 relabelled, 9 unresolved (all short `_h` transitions, 0.2–0.9 s); family holds 52 / 3 / 15 / 0. Exemplars moved on 38 holds (6 family), windows trimmed on 53 (20). **Crow family:** Crow -a 7.32 → 10.85 s (window from 8.15 s); Side Crow -c h1 kept at 10.10 s (from 7.87 s); h2 21.60 → 22.48 s (window 21.97–23.08 s). Side Crow -c's two holds keep distinct ids and carry opposite shelves. **Ground:** 814 kept, 110 added (missed touches: feet 84, head 12), 19 removed (Chaturanga's head, Bridge -a's hands). Roles: 445 `required_support` (mat ≥ 50 N), 479 `required_touch` (load unconfirmed), 19 `forbidden_support`. **Pairs:** of 562 labelled, 196 stay `required_touch`, 159 are demoted (touched, not required), 205 removed as proximity latches (145 of them leg–leg) and 2 carried; 426 unlabelled human contacts are recorded. Tier 1 over the human contacts: 386 consequential, 395 braces; 54 `any_of` groups. Configured supports floating > 2 cm: 461 / 924 (> 5 cm: 197), Step 8's worklist. **Pass B:** 151 informed packets (124 flagged, 27 controls; 120 MB, 3.6 s each; a full re-render under the final code reproduced every packet id), **151 calls, 0 invalid, $38.25** (median $0.251, 38 s). Calibration at each packet's main moment (`data/reference_curation/calibration/pass_b/render_v3.json`) admits floor avatar 1.000 (199 claims), floor human 0.998 (277), float 1.000 (248, recall 242 / 242 at 2–15 cm), body–body 1.000 (168, recall 0.70) and left/right 1.000 (140). The reviewer is *shown* the measurements, so admission means it reads them faithfully: a conflict detector. It contradicts the capture on 5 of 636 annotated zones, all outside the calibrated extremities (Scale's seated pelvis), so those stay advisory. **Roles stay advisory** (no truth set). The reviewer agrees with the deterministic roles on 48 % (186 / 387 touched pairs): it calls 94 of 127 deterministic `required_touch` pairs `allowed` (thigh–trunk and other secondary load paths), and `required_touch` on pose-defining braces Tier 1 demotes (Cockerel's lotus, Firefly's thighs) and on unlabelled contacts (a supported headstand's hands cupping the head, Peacock's elbows). Variants: 48 match, 68 variants, 35 not the labelled pose (mostly `_h` preparations). It keeps 139 of the 151 corrected exemplars, and asks for a repair on 146 holds (float 133, pose 64, penetration 52, head collider 33). All of this is in the labels' `summary.md` worklist. Deviations from the card: store v3, deterministic provisional roles, the demanded-only configuration, the corrected exemplar as Pass B's main moment (card traps) |
| 7: gated statics and witness | done | 2026-09-29 | `reference_curation/{statics,witness}.py`; 19 tests pass in 9 s. **Statics v1** in `data/reference_curation/statics/holds_repaired_ftC_posefix.labels_v1.249a920cea.statics_v1.<hash>/` (`holds.jsonl`, `contacts.jsonl`, `statics.json`, `summary.md`; 2.2 MB): every hold of labels v1 at its corrected exemplar on the shipped ftC references, 303 holds in 8 s on 8 workers. **Gated LP** (ground points ≤ 2 cm, pairs ≤ 1 cm, typed statuses, finite and residual checks): `support_not_realised` 226, `infeasible` 31, `beyond_plant` 8, within the limits 38, `numerical` 0. The old solver passed 295 of the 303 within the limits. **The card's controls:** the 0.5 m lift is `support_not_realised` (the old solver's s\* is unchanged by it); an open pair (Side Crow -c h2's other shelf, 12 cm) carries exactly 0 N and leaves s\* unchanged; **Side Crow -c's shelf is useful, not required, in both holds**: min load 0 N, s\* 0.578 → 0.5135 (h1) and 0.5241 → 0.3606 (h2) with it, in the counterfactual, because the reference cannot say (h1's right hand floats 3.85 cm; h2's tilted right hand leaves the COM 0.8 cm outside its in-band corners, gated `infeasible`). **Counterfactual** (the human's contacts closed): 221 of 226 unrealised holds and 29 of 31 infeasible ones become feasible within the limits. **Necessity** (strict; the min-load and s\*-without tests agree on every contact): the 196 `required_touch` pairs are 114 useful, 82 redundant, 0 required (the reviewer's critical pairs 22 / 9 / 0); the 479 `required_touch` ground supports are 82 required, 276 useful, 101 redundant, 20 undecided; 9 of the 445 `required_support` ones are redundant. `beyond_plant` is the box hand: 7 of 8 peak at a wrist or hand hinge (10–20 N m). Joint stops relieve s\* ≥ 0.02 on 4 of 54 decided holds; 290 of 303 exemplars violate an XYZ hinge range by > 2°. **Statue witness** (MuJoCo, 1 kHz, μ 0.75/0.5, the training gains ×10 with the plant's torque limits, joint limits off; settle < 5 cm in 0.5 s, then drift < 2 cm, contacts kept): **4 of 38 feasible holds held** — Warrior II -a@572 (settle 2.41, drift 0.22 cm) and Uttanasana -a@25, @719, @896. It fails on the standing references because their foot boxes are supinated about 12° at uneven heights (31 settle ≥ 5 cm). Deviations from the card: a stiffened statue (it buckles at the training gains), the settle window, the three-way contact rule, effort-based `useful`, the counterfactual closure (card traps). **Still open** (card): validation against the recorded PhysX forces, the training plant's joint limits, an IsaacLab statue |
| 8: retarget and Pass C | done; 3 of 4 exit tests met (not "no new penetrations") | 2026-09-29 | `reference_curation/{plant,retarget,edits,roles}.py`, `isaac_statue.py`, `prompts/pass_c.md`, `schemas/pass_c.json`. 22 new tests; the suite's 164 pass in 7.3 min. **Step 7's open checks are closed** (see its card): PhysX's joint coordinates are the exp-map and its limits a hard box, so Step 7's soft-limit trap was wrong. The gated LP holds 95.3 % of PhysX's measured ground loads and 98.9 % of its pair loads. The IsaacLab statue is built but not usable. **Retarget `…retarget_v1.22c19e569d`** (motions and lineage in `output/reference_curation/retarget/`, 226 MB; records in `data/reference_curation/retarget/`): 59 clips in 61 min on 8 workers. It is deterministic: two runs of one code are bit-identical. Standing big toe hold -c's fit is unreadable, so its shipped motion is carried. **Exit 1, met:** labelled supports floating > 2 cm fall from **367 / 833 (44.1 %) to 36 / 833 (4.3 %)**, and to **0 of the 797** that are neither head-collider crowns (17) nor supports labels v1 removed (19). Labels v1's 461 / 924 (49.9 %) fall to 27 / 924, 0 of 897 without the crowns. Human-touch zone-frames floating 47.5 % → 0.2 %; frames past a joint range 84.5 % → 0; plant body-pair overlaps > 1 cm 72,659 → 232 pair-frames; 184 of 220 requested pairs closed within 1 cm. **Exit 2, not met:** 214 of those overlaps are new, in 5 clips: the two lotus clips 145 (to 4.4 cm; the plant's knee range cannot fold a lotus), Supported Headstand -a 30, the two Fireflies 39. **Exit 3, met:** round trip ≤ 2.8e-7 m and rad. **Smoothness, not fully met:** 13 of 28 jerky clips are re-solved gently, and 24 clips still add 1–33 frames over 100 m/s² (card trap). **Statics** on the retarget (`statics_v1.7d3b9e15b4`): **270 of 303 holds holdable within the plant's limits** (182 held + 88 feasible), against 38 on the shipped references. 28 are `support_not_realised` (27 crowns, 1 pelvis), 3 infeasible (the carried -c clip), 2 beyond the plant (Scorpion -b s* 1.35, Peacock -a 1.04). The MuJoCo statue holds 182 / 270 by Step 7's rule and 228 / 270 by motion alone; its tips are balances. Pilot: 52 of 59 holdable, statue 34 / 52 and 48 / 52. **Labels v1.1** (`labels_v1_1.3b3fc75828`): 63 `required_touch` ground supports become `required_support`; no pair is statically required (78 useful, 63 redundant). **Pass C:** the controls admit "not the same pose" (32 / 32 swaps, 0 false) and "closer to the human" (42 / 42); artefacts stay advisory (recall 0.375). **Exit 4, met:** 279 of 300 edits accepted (93.0 %); 269 / 283 without the lotus clips. $62.66 for 498 verdicts. Budget flags: 8 clips over the edit budget |
| 9: training artifacts v2 | done (as BodyFix Step 5, on plant v2) | 2026-10-01 | Built from gate v2's release manifest, labels v2 and the `retarget_v2` references. Hold extension v2 (real-frame pressure, inserted frames invalid), graph v2 (stable hold ids; manual goals resolve the side-specific segment, no 0.5-vote union), physics tables v2 (column 1 for newtons, column 2 for shares, the attribution-visibility gate on "unloaded"), the contact-target sidecar and the runtime identity checks: `reference_curation/{hold_extension_v2,contact_targets_v2,release_v2}.py`, `build_hold_graph_v2.py`, `build_physics_tables_v2.py`, `protomotions/{utils/release_identity,envs/control/contact_targets}.py`. **Exit met:** the package round trip (frame by frame), the runtime load on `smpl_yogi_v2` (the release check passes at startup), and the weight-0 smoke (256 envs, 4 epochs). Numbers in `BodyFix.MD` "What Step 5 found". |
| 10: release and training | done; continued as the graph-growth round | 2026-10-05 | Release `holds_repaired_ftC_posefix.release_v2.a2dda5d2ac`: record `data/reference_curation/releases/<id>.json`, heavy files in `data/smpl/reference_curation/<id>/`; validated by 26 identity checks. Fresh expert `smpl_yogi_v2_expert56_a2dda5d2ac` (wandb `06k9vjpn`) on `smpl_yogi_v2` with new normalisers, stopped at epoch 15,500 and evaluated (`expert56_v2_e15500/README.MD`). The graph-growth round then warm-started it: G1 (AMP) and **G3** on release v3 `…release_v3.2f132f4299` (v2 + 28 synthesised edge clips). **G3 stopped by the user at epoch 3,420 and evaluated (`graph_growth_2026_10_03/g3/README.MD`):** all five edges learnt (84/84 motions; 163/163 on command), human gates pass except substitutions (7.7 > 6). **Next: Lane S, the student; the perturbation fine-tune (G4) is skipped (user, 2026-10-05).** See "After Step 10". Pass D not run. |

## 5. Spike: Claude as the VLM (2026-09-28)

**Setup.** Four frames with known truth, each rendered twice under opaque names:
- the avatar alone, with left/right colouring;
- the avatar plus the human's MoSh markers as black dots.

The renderer is `vlm_spike/render_spike.py`; the truth sits in a file the reviewer never saw.

- p1–p4 were reviewed by one in-session general-purpose subagent. It inherits `CLAUDE.md`, so it is not strictly blind.
- p1 and p2 were also reviewed headless (`claude -p` from a directory outside the repo: `vlm_spike/headless_p1_sysprompt.json`, `vlm_spike/headless_p2.json`).

| Packet | Truth (markers, mat, audit) | Reviewer |
|---|---|---|
| p1 Standing Split -a, 9.0 s | human: standing (left) foot and both hands planted; avatar standing foot 11.6 cm up | **headless** (short `--system-prompt`): pose **right** ("Urdhva Prasarita Eka Padasana, left leg standing"); human contacts **right**; avatar left foot "flat on the floor" (**wrong**) and "contacts match" (**float missed**), although it reported floor-level dots "with no avatar geometry… just below the avatar's left foot", which were the planted-foot markers. **In-session:** the same: pose right, human contacts right, float missed; weak head-offset remark |
| p2 Peacock -a, 10.0 s | human: hands plus both feet (toes, 69 N) down; avatar feet 3–5 cm up; the head markers sit 8–12 cm off the head sphere | **headless:** pose "Chaturanga, high" (**wrong**); hands and feet on the floor (**right**); small gap under the right foot noticed; head markers ahead of the head sphere (**right**, the collider defect); "floor dots with no body" (**hallucination**: marker shadows). **In-session:** again "Chaturanga, high" (**wrong**, consistently); contacts right; "toes may hover 2–5 cm, unsure" (**right**, correctly hedged); head offset (**right**) |
| p3 Supported Headstand -b, 15.0 s | human: crown on the floor (ARIEL 0.8 cm) and hands; avatar head 14 cm up | **in-session:** "Tripod Headstand, medium" (**right**); avatar crown "hovers 8–15 cm, unsure" (**right**); human head markers reach the floor (**right**); inferred that without head support "the avatar… could not hold still and would tip" (**right**: statics reasoning from a readable image); suspected a systematic head/neck offset (**right**) |
| p4 Crow -a, 9.5 s | human: hands only, feet 35–38 cm up (markers); in the ft_c-repaired reference the shins rest on the upper arms (0.3 cm). The human's knee contact is unverified until the Step 5 mesh | **in-session:** "unknown; standing forward fold on both feet, hands off the floor", with feet listed as the floor contacts for both avatar and human (**support inverted**: hands and feet are same-coloured boxes, and it reported it could not tell arm from leg segments); head offset (**right**) |

**Scorecard** (5 reviews of 4 packets; tiny, directional only):

| Claim | Result |
|---|---|
| Pose identity | right 3, wrong 3 (Peacock twice, with high confidence; Crow misread) |
| Human floor contacts from markers | right 4, wrong 1 (p4, inverted) |
| Avatar floor contacts | right 1 (p3), hedged 1 (p2), wrong 3 |
| Float detection | 14 cm head caught; 3–5 cm toes hedged; 11.6 cm standing foot missed twice |
| Plant head-collider offset | noticed in 4 of 5 reviews |

**Cost of the modes.**
- In-session review of the 4 packets took 20.6 min, 16 tool calls and 209k subagent tokens.
- Headless review took about 2 min per packet, which makes it the batch mode.

**The reviewer's own requests for the renderer**, all adopted in Step 3:
- no marker shadows;
- zoomed insets of each support;
- grazing floor-level views, and drop lines from the hands, feet and head to the floor;
- distinct colours per segment, with hands visibly different from feet and a distinct head colour;
- markers coloured by body segment;
- larger images.
