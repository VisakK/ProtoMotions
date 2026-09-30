# Plant checks (BUILD_PLAN Step 7's still-open list)

Rollouts: 13 (`results/Contact_Physics_analysis*/*/rollout.npz`).

## Joint coordinates

PhysX's `dof_pos` equals the rotation vector of each local rotation to **8.0e-07 rad** on every frame of every rollout (XYZ Euler angles: up to 3.51 rad). The plant's joint coordinates are the exp-map, the `.motion` files' `dof_pos` convention.

## Joint limits

Largest excursion past a range over 34080 frames: **1.04 deg**, with 6689 joint-frames at a stop with the actuator saturated against it. The limits are a hard box on the exp-map coordinates.

| Rollout | Frames | Max past range (deg) | Worst joint | Applied / measured torque (N m) |
|---|---|---|---|---|
| `220923_Chair_Pose_or_Utkatasana_-b` (Contact_Physics_analysis) | 1516 | 1.038 | L_Hip_z | 117.9 / 157.6 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a` (Contact_Physics_analysis) | 1940 | 0.79 | L_Hip_z | -7.3 / 81.5 |
| `220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a` (Contact_Physics_analysis) | 5872 | 0.778 | R_Toe_y | 20.0 / 42.9 |
| `220923_Scorpion_pose_or_vrischikasana-b` (Contact_Physics_analysis) | 3612 | 1.005 | L_Hip_z | -133.4 / 75.1 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-` (Contact_Physics_analysis) | 2712 | 0.326 | L_Elbow_x | -24.3 / -40.3 |
| `220926_Warrior_III_Pose_or_Virabhadrasana_III_-a` (Contact_Physics_analysis) | 2532 | 0.176 | L_Elbow_z | -65.1 / -35.8 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a` (Contact_Physics_analysis_crow_pair) | 1940 | 0.49 | R_Elbow_x | 59.1 / 46.8 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-` (Contact_Physics_analysis_crow_pair) | 2712 | 0.706 | L_Wrist_x | -20.0 / -43.1 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a` (Contact_Physics_analysis_crow_pair_final) | 1940 | 0.758 | L_Knee_x | 200.0 / 67.3 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-` (Contact_Physics_analysis_crow_pair_final) | 2712 | 0.686 | L_Wrist_x | 20.0 / -40.7 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a` (Contact_Physics_analysis_crow_pair_resumed) | 1940 | 0.711 | L_Hip_z | 77.5 / 70.7 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-` (Contact_Physics_analysis_crow_pair_resumed) | 2712 | 0.672 | L_Wrist_x | 8.9 / -41.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a` (Contact_Physics_analysis_crow_pair_scorebased) | 1940 | 0.462 | R_Thorax_y | 101.5 / 45.7 |

## The shipped references against the box

Exemplars past a range by > 2 deg: **285 of 303** in exp-map (the plant's coordinates), 290 in MuJoCo's XYZ decomposition. Frames: 75381 of 89176.

By joint (exemplars): L_Elbow_z 143, R_Elbow_z 140, L_Thorax_z 74, R_Elbow_x 73, L_Elbow_x 68, R_Thorax_z 61, R_Knee_y 59, L_Ankle_y 52, L_Elbow_y 38, R_Elbow_y 29, R_Ankle_y 26, R_Toe_x 24

Worst: `220923_Cockerel_Pose-b@2614` 160.2 deg (R_Knee_y); `220923_Cockerel_Pose-b@2492` 158.5 deg (R_Knee_y); `220923_Cockerel_Pose-b@2355` 157.1 deg (R_Knee_y); `220923_Cockerel_Pose-b@1983` 156.8 deg (R_Knee_y); `220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-b@420` 48.4 deg (R_Elbow_x); `220923_Peacock_Pose_or_Mayurasana_-a@434` 47.1 deg (R_Elbow_y)

## Contact forces against the gated LP

LP on the recorded frames (full inverse dynamics, joint stops free): {'within_limits': 7974, 'infeasible': 325, 'beyond_limits': 75}.

- ground: measured load inside the LP interval (+-5 % body weight) on **19078 of 20015** (0.953); gated out 0, no interval 518
- pair: measured load inside the LP interval (+-5 % body weight) on **15822 of 16001** (0.989); gated out 0, no interval 276

## Step 7's pair necessity against PhysX's pair loads

| Hold | Pair | Necessity (basis) | PhysX loaded | Median N |
|---|---|---|---|---|
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | L_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_SHANK+R_UPPER_ARM | useful (counterfactual) | 1.0 | 492.5 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | L_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_THIGH+TRUNK | useful (counterfactual) | 1.0 | 1119.2 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_THIGH+R_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_SHANK+R_UPPER_ARM | redundant (counterfactual) | 1.0 | 515.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_THIGH+L_UPPER_ARM | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_THIGH+TRUNK | redundant (counterfactual) | 1.0 | 1101.6 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_THIGH+R_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@448` | L_SHANK+R_THIGH | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@448` | R_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.355 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@571` | L_SHANK+R_THIGH | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@571` | R_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.29 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | L_SHANK+R_THIGH | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_SHANK+L_THIGH | redundant (counterfactual) | 1.0 | 388.6 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_SHANK+L_UPPER_ARM | redundant (counterfactual) | 1.0 | 136.2 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_THIGH+L_FOREARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@871` | R_THIGH+L_UPPER_ARM | useful (gated) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@871` | R_THIGH+L_FOREARM | useful (gated) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | L_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_SHANK+R_UPPER_ARM | useful (counterfactual) | 1.0 | 608.1 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | L_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_THIGH+TRUNK | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_THIGH+R_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_SHANK+R_UPPER_ARM | redundant (counterfactual) | 1.0 | 582.4 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_THIGH+L_UPPER_ARM | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_THIGH+R_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@448` | L_SHANK+R_THIGH | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@448` | R_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.032 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@571` | L_SHANK+R_THIGH | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@571` | R_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | L_SHANK+R_THIGH | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_SHANK+L_THIGH | redundant (counterfactual) | 0.71 | 209.2 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_SHANK+L_UPPER_ARM | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_THIGH+L_FOREARM | useful (counterfactual) | 1.0 | 399.8 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@871` | R_THIGH+L_UPPER_ARM | useful (gated) | 1.0 | 318.3 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@871` | R_THIGH+L_FOREARM | useful (gated) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | L_SHANK+L_UPPER_ARM | useful (counterfactual) | 1.0 | 185.9 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_SHANK+R_UPPER_ARM | useful (counterfactual) | 1.0 | 427.8 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | L_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_THIGH+TRUNK | useful (counterfactual) | 1.0 | 856.6 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_THIGH+R_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_SHANK+L_UPPER_ARM | useful (counterfactual) | 1.0 | 194.7 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_SHANK+R_UPPER_ARM | redundant (counterfactual) | 1.0 | 439.1 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_THIGH+L_UPPER_ARM | redundant (counterfactual) | 1.0 | 112.1 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_THIGH+TRUNK | redundant (counterfactual) | 1.0 | 857.9 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_THIGH+R_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@448` | L_SHANK+R_THIGH | useful (counterfactual) | 0.484 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@448` | R_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.129 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@571` | L_SHANK+R_THIGH | useful (counterfactual) | 0.387 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@571` | R_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | L_SHANK+R_THIGH | useful (counterfactual) | 0.677 | 169.7 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_SHANK+L_THIGH | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_SHANK+L_UPPER_ARM | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_THIGH+L_FOREARM | useful (counterfactual) | 1.0 | 324.5 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@871` | R_THIGH+L_UPPER_ARM | useful (gated) | 1.0 | 516.6 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@871` | R_THIGH+L_FOREARM | useful (gated) | 0.742 | 169.6 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | L_SHANK+L_UPPER_ARM | useful (counterfactual) | 1.0 | 382.7 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_SHANK+R_UPPER_ARM | useful (counterfactual) | 1.0 | 536.8 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | L_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_THIGH+TRUNK | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_THIGH+R_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_SHANK+L_UPPER_ARM | useful (counterfactual) | 1.0 | 380.3 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_SHANK+R_UPPER_ARM | redundant (counterfactual) | 1.0 | 536.2 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_THIGH+L_UPPER_ARM | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_THIGH+R_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@448` | L_SHANK+R_THIGH | useful (counterfactual) | 0.677 | 54.1 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@448` | R_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.129 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@571` | L_SHANK+R_THIGH | useful (counterfactual) | 0.419 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@571` | R_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | L_SHANK+R_THIGH | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_SHANK+L_THIGH | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_SHANK+L_UPPER_ARM | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@680` | R_THIGH+L_FOREARM | useful (counterfactual) | 0.677 | 107.6 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@871` | R_THIGH+L_UPPER_ARM | useful (gated) | 1.0 | 276.5 |
| `220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-a@871` | R_THIGH+L_FOREARM | useful (gated) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | L_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_SHANK+R_UPPER_ARM | useful (counterfactual) | 1.0 | 586.5 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | L_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_THIGH+TRUNK | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@424` | R_THIGH+R_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_SHANK+L_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_SHANK+R_UPPER_ARM | redundant (counterfactual) | 1.0 | 564.2 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | L_THIGH+L_UPPER_ARM | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_THIGH+TRUNK | redundant (counterfactual) | 0.0 | 0.0 |
| `220923_Crane_Crow_Pose_or_Bakasana_-a@439` | R_THIGH+R_UPPER_ARM | useful (counterfactual) | 0.0 | 0.0 |
