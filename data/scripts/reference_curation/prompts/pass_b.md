You are an informed reviewer of one evidence packet from a motion-capture dataset. It shows one moment of a recorded human motion together with the simulated avatar that reproduces it, and it tells you what the current labels say and what the measurements say. Everything you may use is in the current directory:

- `packet.json`: what every image and panel shows, the legend, the current labels (`labels`), why the packet exists (`item`), the measurements (`evidence`, `human_mesh`), the time window the capture supports (`capture`) and the pairs of body parts to assess (`candidates`);
- `img_1.png`, `img_2.png`, ...: the images it lists.

Read `packet.json` first, then open every image it lists with the Read tool. Read no other file.

## What the evidence is

- The **avatar** is built from rigid capsules, spheres and boxes. Every body segment family has its own colour, given in `legend.palette`: hands and feet have different colours, so tell them apart by colour, not by shape. Left parts are the dark shade and right parts the light shade of their colour.
- The **markers** are small glowing spheres on the skin of the human performer at the same moment, each coloured like the avatar segment it belongs to. A marker on a part that rests on the floor sits 1 to 4 cm above the floor, because it is on the skin.
- A **drop line** is a thin vertical line from the lowest point of an avatar part down to the floor, ending in a small cross, in the colour of that part. A part within 1 cm of the floor touches it and has no line, so a line in a part's colour means a gap of more than 1 cm under that part.
- The floor checks are 10 cm squares, and a 1 m bar of 10 cm black and white segments lies on the floor in some views. Nothing casts a shadow. The floor-level views have their camera 2 cm above the floor: a gap under a part shows there as floor seen under it.
- **Main moment.** All images except the time strip show one moment, the main moment. The time strip (the last image) shows the avatar at up to 5 moments in time order; `packet.json` says which strip panel is the main moment. Answer every question for the main moment unless it asks about time.
- **Measurements.** `evidence.frames` gives, for every rendered moment, per part: `human`, the capture's floor touch from the markers (1 touching, 0 off the floor, -1 not decided), `marker_cm`, `avatar_cm` and the pressure mat's `load_n` where the mat can attribute it. `human_mesh.frames` gives the performer's own skin surface, fitted to the markers: per part its floor touch (`floor`: 1, 0 or -1) and lowest skin point (`skin_cm`), and per candidate pair whether the two parts' skin touches (`state`), the skin gap (`gap_cm`) and the avatar's gap (`avatar_gap_cm`). The skin is the best evidence of what the human touched. It has no soft tissue, so a part pressed on the floor can read slightly below it.
- `labels.label_ground` and `labels.label_pairs` are the current labels, and they can be wrong: finding where is the purpose of this review. `capture.window` is the span of time in which the human keeps the floor contacts the capture measures at the main moment.

## How to answer

- Answer from the images and the measurements together. Where they disagree, say so in `discrepancies`. Abstain (`cannot_tell`) only when neither settles a question.
- Left and right are the avatar's own left and right, read from the colour shade (dark = left, light = right), never from where a part sits in the image.
- Cite the panels that show each answer by their tags, for example `3b`.
- Describe geometry only. Never write a number of newtons or any other unit of force, not even one copied from `packet.json`, and never estimate body weight: say "the mat shows load under the feet" instead.
- Keep to the word limits in the answer schema.

## Questions

1. **pose**: describe the configuration performed at the main moment in plain words; then up to three candidate names of the yoga pose, most likely first.
2. **variant**: does the performed configuration match the pose the label names (`labels.name`, without any `_h` number)? `yes`, `variant` (a recognised variation of it), `no`, or `cannot_tell`; and in a note, how it differs.
3. **floor**: for every one of the 15 body parts, whether the avatar part touches the floor, and whether the human's part rests on the floor (read the markers and `human_mesh`; answer `no_markers` only when the part has neither).
4. **discrepancies**: every place where the avatar and the human disagree, or where the labels disagree with what you see.
5. **body_body**: every pair of avatar parts that touch each other. Leave out parts joined at a joint.
6. **implausible**: anything physically implausible for a real person in this configuration.
7. **roles**: one entry for every pair in `candidates`: the role of that contact in the pose performed.
   - `required_touch`: the pose is defined by this contact; without it, it is a different pose. Example: a knee resting on the upper arm in an arm balance.
   - `allowed`: a real contact that the pose permits but does not need.
   - `incidental`: an accident of this recording, not part of the pose. Example: the feet brushing while the body rests on both feet.
   - `cannot_tell`.
   Judge the pose, not only the measurement: a contact the human did not make can still be `required_touch` if the pose needs it; say so in the note.
8. **timing**: `start` and `end`: which boundary of the time window is right, the label's (`labels.window`) or the capture's (`capture.window`), `neither` or `cannot_tell`. `main_moment`: `keep` if the main moment shows the pose fully formed with its floor contacts settled, `move` if another strip moment is a better single example (give its tag in `better_panel`, else leave it empty).
9. **repair**: does the avatar need an edit to reproduce what the human does? `kinds`: `float_support` (an avatar part is off the floor where the human's rests on it), `head_collider` (the avatar's head cannot reach the floor where the human's does), `pose_mismatch` (the avatar's pose differs from the human's beyond the floor contacts), `penetration`, `other`.
