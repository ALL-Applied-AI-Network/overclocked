# Team Overclocked: perception-driven pick-and-place on the FANUC ER-4iA (executive summary, 1 Nov 2026)

*Numbers come from our own runs on the real cell, 26 Oct - 1 Nov 2026. One cell, one operator, small n. Private run and video identifiers are intentionally omitted.*

## Goals
Put loose parts into a bin on a real FANUC ER-4iA cell with **no hand-taught positions**: cameras find the parts and the bin, the arm picks and places, a grasp check looks for a failed grasp, and the loop re-looks until the table is clear. Nothing is read from `placement.yaml`.

## 1. What problem in manufacturing are we solving?
A taught pick position is only right until something moves: a part lands 10 mm away, a fixture gets bumped, a camera or gripper is re-seated. Then the robot misses, usually silently, and an engineer re-teaches offsets. We lived this: the camera-to-gripper calibration moved by up to about 13 mm between 22 and 27 Oct, and a 3.4 mm-wide success window for a 10 mm cube had to be found again by hand (-3 to -6 mm hit the part and -7 to -10 mm picked it on 30 Oct; on 1 Nov -9 and -12 mm hit it and -15 and -18 mm picked every cube). The costs are scrap and damaged parts from collisions, rework of missed placements, downtime for re-teaching, and skilled labour spent on calibration instead of production.

## 2. How does the project use Physical AI?
Perception, decision and action closed on real hardware: table camera (colour/silhouette fit; OWLv2 open-vocabulary detector on a DGX Spark when reachable), wrist camera and depth to re-centre the fingers, a learned **grasp check** (side camera + frozen NVIDIA C-RADIOv2-B backbone + small trained head, TensorRT FP16, 1.39 ms) that estimates whether the gripper holds a part before the arm carries it (reliable on our held-out test, fooled in one staged case on 1 Nov), and a recovery loop (miss -> skip the carry -> re-look -> re-pick). Which models ran when: in the 1 Nov 4-of-4 run the Spark was unreachable, so cubes were found by the classical colour pipeline and the grasp check returned n/a (no AI model in that run); in the 1 Nov recovery run OWLv2 answered 2 looks, Cosmos3-Nano was queried and logged only (it does not move the arm), and the C-RADIO classifier scored 2 lifts (one wrongly); in the 1 Nov models-on run t03 (4 of 4 cubes in the bin) OWLv2 answered all 5 looks, Cosmos answered once (logged only), and the classifier scored the 4 held lifts 0.99, 0.71, 1.00, 0.26, so it would have rejected one good grasp; the wrist-camera check made the call. The learned grasp check is the only model that can change the arm's behaviour. We also built an offline self-calibration learner (table-to-wrist gap, fitted from logged views); it was not validated on the robot, so we make no claim for it.

## 3. How did we use the CSI equipment?
FANUC ER-4iA with the two-finger gripper (12 mm opening on straight faces), the table, wrist (RGB + depth) and side cameras through the organizers' client; DGX Spark (via YAAD) for OWLv2, Cosmos3-Nano and the verifier; all trials on the real cell (real-cell runs on 26-31 Oct and 1 Nov). The Isaac Sim starter gave us geometry only; our policy was developed and measured on the real cell.

## 4. What evidence shows it works?
| Measure | Result | Baseline |
|---|---|---|
| Cubes placed in the bin, full policy, 1 Nov, calibration re-found (-15 mm and -18 mm) | 12 of 12 across three clean runs (t01, the -15 mm run, t03), no faults | Stale calibration, as a taught cell would keep it (-9, -9, -12 mm): 0 of 3 attempts, 3 controller faults |
| Grasp check on held-out grasps (107 crops, 41 grasps, 30 Oct) | 3 of 51 empty grasps passed, 50 of 56 held kept, AUC 0.958 | Colour rule 28 of 51; Cosmos3-Nano zero-shot 2 of 28 held recognised |
| Recovery on hardware | dropped cube found, re-picked, placed (30 Oct); cube moved by hand, empty carry, look again, re-pick, place (1 Nov) | Recovery came from the look-again loop; the grasp check passed the empty grasp (P 0.99) |
| Where it fails | AA battery only sometimes (0.75 mm margin per side vs about 1 mm jitter); 4-5 mm screw, L-shaped hex key, 19 mm plastic screw do not fit the jaws | |

Limits we state plainly: today's side-camera crop had drifted about 14 px (4 of 9 fresh held lifts kept until re-aligned) and the check passed a staged empty grasp; cubes work, other parts mostly do not; one operator-watched cell; labels for the verifier are by eye with only 5 real empty lifts; the success window moves between days and was re-found by hand.

## Supporting material
Code: https://github.com/ALL-Applied-AI-Network/overclocked. Policy: `policy.py` and `cube_vision.py`. Grasp-check training and TensorRT scripts: `spark/verifier/`. Showcase video: submitted separately.
