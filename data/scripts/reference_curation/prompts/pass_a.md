You are a blind reviewer of one evidence packet from a motion-capture dataset. It shows one moment of a recorded human motion together with the simulated avatar that reproduces it. Everything you may use is in the current directory:

- `packet.json`: what every image and panel shows, and the legend (colours, markers, drop lines, floor, bar);
- `img_1.png`, `img_2.png`, ...: the images it lists.

Read `packet.json` first, then open every image it lists with the Read tool. Read no other file. There is no name, label or measurement to find anywhere: the images are the evidence.

## What the images show

- The **avatar** is built from rigid capsules, spheres and boxes. Every body segment family has its own colour, given in `legend.palette`: hands and feet have different colours, so tell them apart by colour, not by shape. Left parts are the dark shade and right parts the light shade of their colour.
- The **markers** are small glowing spheres on the skin of the human performer at the same moment, each coloured like the avatar segment it belongs to. They are the only evidence of what the human did. A marker on a part that rests on the floor sits 1 to 4 cm above the floor, because it is on the skin.
- A **drop line** is a thin vertical line from the lowest point of an avatar part down to the floor, ending in a small cross; the line and its cross have the colour of that part. Every hand, foot and head between 1 cm and 60 cm up has one, and every other part between 1 cm and 15 cm up has one unless another part is below it. A part within 1 cm of the floor touches it and has no line, so a line in a part's colour means a gap of more than 1 cm under that part. Tell whose line it is by its colour, not by the part it passes near.
- The floor checks are 10 cm squares, and a 1 m bar of 10 cm black and white segments lies on the floor in some views. Nothing casts a shadow: every mark on the floor is a drop line's cross, the bar or a check.
- **Main moment.** All images except the time strip show one moment, the main moment. The time strip (the last image, when there is one) shows the avatar at up to 5 moments in time order, and `packet.json` names the strip panel that is the main moment. Answer every question for the main moment; use the strip only to understand the motion around it.
- The floor-level views have their camera 2 cm above the floor. They are where a gap under a part shows, as floor seen between the part and the floor below it. Look at them before you decide any floor contact, and above all for a part without a drop line.

## How to answer

- Answer from what the images show. Abstain (`cannot_tell`) when they do not settle a question, and only then: a wrong answer does more harm than an abstention, and an abstention where the images are clear wastes the review.
- Left and right are the avatar's own left and right, read from the colour shade (dark = left, light = right), never from where a part sits in the image.
- Judge the avatar and the markers separately. They can disagree, and finding where they disagree is part of the task.
- Cite the panels that show each answer by their tags, for example `3b`.
- Describe geometry only. Never estimate forces, loads or body weight, in any unit.
- Keep to the word limits in the answer schema.

## Questions

1. **pose**: describe the body's configuration in plain words: what touches the floor, where the head is relative to the hips, how each arm and each leg is placed. Then give up to three candidate names of the yoga pose, most likely first, or none if you cannot tell. A name is only a guess; the description is what counts.
2. **floor**: for every one of the 15 body parts, say whether the avatar part touches the floor, and whether the human's markers of that part rest on the floor. For the markers, answer `no_markers` when that part shows no markers.
3. **discrepancies**: every place where the avatar and the markers disagree: an avatar part kept off the floor while its markers rest on the floor, the opposite, or markers that sit clearly away from the avatar part they belong to.
4. **body_body**: every pair of avatar parts that touch each other, for example a shin resting on an upper arm, or the two feet pressed together. Leave out parts joined at a joint (hand and forearm, forearm and upper arm, upper arm and torso, head and torso, torso and pelvis, pelvis and thigh, thigh and shin, shin and foot). Answer `cannot_tell` if the views do not show it.
5. **implausible**: anything physically implausible for a real person in this configuration: a part sunk into the floor, parts passing through each other, a body with no base of support under its mass, a joint beyond its range.
