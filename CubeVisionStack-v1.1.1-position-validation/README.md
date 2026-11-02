# CubeVisionStack-v1.1.1-position-validation

Conservative physical position-validation test for the FANUC ER-4iA.

> Detect one blue 10 mm cube from live table-camera images at varied positions and place it into a fixed bin.

This package isolates position-dependent pickup behavior. It does not add multiple cubes, multiple colours, arbitrary
yaw, stacking, or multiple bins.

## Fixed physical behavior

The physically validated v1.1 settings are unchanged: grasp TCP Z `-333 mm`, bin release TCP Z `-255 mm`, transit
speed `60 mm/s`, approach speed `15 mm/s`, maximum radial reach `500 mm`, the existing camera calibration, and the
existing safe target validation and source-verification flow. The bin stays at `(430, 80)` mm and cube yaw stays at
`0°`.

The source pick envelope is X `[340, 440]` mm and Y `[-160, -50]` mm: a 5 mm margin around the five layouts. The bin
is outside this region.

## Vision and safety contract

The placement coordinate is metadata for the physical setup and log only. It is never used to select a detection,
generate a motion target, or provide a fallback. The policy requires exactly one blue-cube detection from the live
table camera, at least `0.55` fitted-silhouette IoU, three observations with no more than `2.5 mm` spread, and a
detected X/Y inside the pick envelope. Missing, ambiguous, unstable, low-confidence, or out-of-envelope detections
end in a safe hold without descent.

After pickup, the policy moves clear and verifies source removal before transport. A cube still at the source is a
verified failed pickup and permits one retry only. After release, two additional observations must confirm that the
detected source is clear before the rollout completes.

## Ten rollout layouts

| Rollouts | Label | Declared cube X/Y/Z (mm) | Bin X/Y (mm) |
|---:|---|---:|---:|
| 1 and 6 | center | `(390, -105, 0)` | `(430, 80)` |
| 2 and 7 | front-left | `(345, -55, 0)` | `(430, 80)` |
| 3 and 8 | front-right | `(435, -55, 0)` | `(430, 80)` |
| 4 and 9 | back-left | `(345, -155, 0)` | `(430, 80)` |
| 5 and 10 | back-right | `(435, -155, 0)` | `(430, 80)` |

## Local verification

From WSL:

```bash
cd '/mnt/c/Users/glnva/OneDrive/Documents/NVIDIA+Rockwell/overclocked/CubeVisionStack-v1.1.1-position-validation'
uv sync
uv run python -m unittest -v test_position_validation.py
uv run run.py --mock --trials 10 --trial-s 3
```

## Copy into a fresh official client

Replace `/path/to/PhysicalAI-Hackathon-2026` with the clean official-client checkout. Do not copy an old
`.request.json`.

```bash
SRC='/mnt/c/Users/glnva/OneDrive/Documents/NVIDIA+Rockwell/overclocked/CubeVisionStack-v1.1.1-position-validation'
DST='/path/to/PhysicalAI-Hackathon-2026'
cp "$SRC/policy.py" "$SRC/cube_vision.py" "$SRC/test_position_validation.py" "$DST/"
cp "$SRC/submission/submission.yaml" "$SRC/submission/placement.yaml" "$DST/submission/"
cp -a "$SRC/calibration/." "$DST/calibration/"
cd "$DST"
uv sync
uv run python -m unittest -v test_position_validation.py
uv run run.py --mock --trials 10 --trial-s 3
```

## Submit, check, and run

Run these manually only when ready. Submission does not book a robot slot; booking remains a separate operator/user
action.

```bash
cd '/path/to/PhysicalAI-Hackathon-2026'
uv run run.py --submit
uv run run.py --status
# During the approved physical slot only:
uv run run.py
```

## Physical-run log

Record one row per rollout with: rollout number/layout label, declared X (mm), declared Y (mm), detected X (mm),
detected Y (mm), pickup success, whether the gripper appeared centered, miss direction, retry count, and cycle time
(seconds). Use `N/A` for miss direction after a centered successful pickup. These fields distinguish constant X/Y
offset, location-dependent camera mapping, grasp-height error, and gripper-alignment error.
