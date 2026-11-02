# CubeVisionStack-v1.2

Expanded-region, three-cube follow-up to the successful CubeVisionStack-v1 physical run and the narrow v1.1 trial.

## Scope

- Three blue 10 mm cubes per rollout.
- One fixed yellow bin at `(430, 80)` mm in UF0.
- Variable cube positions across a 170 x 150 mm pick region.
- The policy detects and places all three cubes, one at a time, during the same rollout.
- Every submitted cube is placed at yaw 0 degrees; grasp wrist R remains 0 degrees.

This version intentionally does not add new colours or 16 mm cubes. Those variables should be tested separately after
the expanded-region three-cube task succeeds.

## Expanded pick region

| Axis | v1.1 | v1.2 |
|---|---:|---:|
| X | 360 to 440 mm | 280 to 450 mm |
| Y | -140 to -60 mm | -180 to -30 mm |
| Size | 80 x 80 mm | 170 x 150 mm |
| Area | 6,400 mm2 | 25,500 mm2 |

The farthest v1.2 corner is about 485 mm from the robot base axis, inside the policy's conservative 500 mm radial
limit and the cell's approximate 550 mm reach limit.

This boundary was chosen from the failed-trial image: it includes the strong blue-cube detection at about
`(291, -55)` that v1.1 missed, while the `Y <= -30` cutoff excludes the blue bin-side clutter seen around `Y = -11`
and higher.

## Preserved physical calibration

The CSI operator confirmed the standard fingers put UTOOL1 at the fingertips. The work surface is near TCP Z
`-335 mm`; the server clamps below `-333 mm` and rejects below `-334.1 mm`. This version preserves the successful
physical values:

- Grasp TCP Z: `-333 mm`
- Bin release TCP Z: `-255 mm`
- Transit speed: `60 mm/s`
- Approach speed: `15 mm/s`
- Maximum radial reach: `500 mm`

Do not lower the grasp target or broaden the workspace limits without a new operator-reviewed calibration.

## Rollout behavior

Before each pick, the policy requires the expected number of remaining blue 10 mm cubes inside the expanded region:
three before the first pick, two before the second, and one before the third. Cubes must be at least 28 mm apart.

The first target is the isolated candidate with the highest detector IoU confidence. Coordinate tie-breakers make
equal-confidence scenes deterministic. The next two observations track that same cube by proximity, so the policy
does not switch targets if confidence scores fluctuate. Motion begins only after three stable detections with no more
than 2.5 mm spread. The arm moves to the detected X/Y; there is no per-position tolerance or blind placement fallback.

After each drop, two consecutive observations must show the expected number of loose cubes. Only after the third
verified placement does the policy log `TASK COMPLETE` and hold.

## Submitted layouts

| Rollout | Cube 1 | Cube 2 | Cube 3 | Bin |
|---:|---:|---:|---:|---:|
| 1 | (295, -55) | (380, -62) | (425, -63) | (430, 80) |
| 2 | (290, -165) | (365, -105) | (440, -45) | (430, 80) |
| 3 | (300, -40) | (380, -120) | (445, -175) | (430, 80) |
| 4 | (285, -130) | (360, -45) | (440, -130) | (430, 80) |
| 5 | (300, -170) | (375, -80) | (445, -35) | (430, 80) |

All dimensions are millimetres in UF0. Layout order does not determine pick order; the policy chooses from detected
positions. The operator should remove other blue cube-like objects and bins from the expanded region.

## Safe-hold conditions

The policy holds without beginning the next motion if the expected remaining-cube count is not visible, two cubes are
closer than 28 mm, detection is unstable, a target is outside the software envelope, or pickup/placement verification
remains ambiguous. Every pickup gets at most one retry at the same calibrated Z.

## Deploy into a fresh official client

Copy `policy.py`, `cube_vision.py`, `test_policy_v1_2.py`, and the two files under `submission/` into a clean clone of
the official client. Do not copy a `.request.json` from another submission. Then run:

```bash
uv sync
uv run python -m unittest -v test_policy_v1_2.py
uv run run.py --mock --trials 1 --trial-s 8
uv run run.py --submit
```

Book a new slot with the new request ID. During the approved slot, keep the CSI operator at the cell stop position and
run `uv run run.py`. Stop on unexpected motion, a coordinate disagreement, ambiguous detection, or a controller alarm.
