# CubeVisionStack-v1.1

Three-cube follow-up to the successful CubeVisionStack-v1 physical run.

## Scope

- Three blue 10 mm cubes per rollout.
- One fixed yellow bin at `(430, 80)` mm in UF0.
- Five approved three-cube layouts inside the conservative pick box.
- The policy detects and places all three cubes, one at a time, during the same rollout.
- Every cube is placed at yaw 0 degrees and grasp wrist R remains 0 degrees.

This version intentionally does not add new colours, 16 mm cubes, or unrestricted random placement. Those variables
should be tested separately after the three-cube task succeeds.

## Preserved physical calibration

The CSI operator confirmed the standard fingers put UTOOL1 at the fingertips. The work surface is near TCP Z
`-335 mm`; the server clamps below `-333 mm` and rejects below `-334.1 mm`. This version preserves the successful
v1 values:

- Grasp TCP Z: `-333 mm`
- Bin release TCP Z: `-255 mm`
- Transit speed: `60 mm/s`
- Approach speed: `15 mm/s`
- Maximum radial reach: `500 mm`

Do not lower the grasp target or broaden the workspace limits without a new operator-reviewed calibration.

## Rollout behavior

Before each pick, the policy requires the expected number of remaining cubes to be visible inside the approved pick
box. It selects the cube nearest the next submitted position, then requires three stable observations with no more
than 2.5 mm spread. The arm moves to the detected X/Y, not blindly to the placement-file coordinate.

After the source clears, the cube is carried to the bin and released. Two consecutive table observations must then
show the expected number of loose cubes before the next pick begins:

- After placement 1: two loose cubes must remain.
- After placement 2: one loose cube must remain.
- After placement 3: zero loose cubes may remain.

Only after the third verified placement does the policy log `TASK COMPLETE` and hold.

## Submitted layouts

| Rollout | Cube 1 | Cube 2 | Cube 3 | Bin |
|---:|---:|---:|---:|---:|
| 1 | (370, -130) | (400, -100) | (430, -70) | (430, 80) |
| 2 | (370, -70) | (400, -100) | (430, -130) | (430, 80) |
| 3 | (370, -120) | (400, -120) | (430, -120) | (430, 80) |
| 4 | (400, -130) | (400, -100) | (400, -70) | (430, 80) |
| 5 | (370, -120) | (430, -120) | (400, -70) | (430, 80) |

All dimensions are millimetres in UF0. The table lists the required pick order for each rollout.

## Safe-hold conditions

The policy holds without beginning the next motion if the expected remaining-cube count is not visible, the next cube
is more than 15 mm from its submitted position, detection is unstable, a target is outside the software envelope, or
pickup/placement verification remains ambiguous. Every pickup gets at most one retry at the same calibrated Z.

## Deploy into a fresh official client

Copy `policy.py`, `cube_vision.py`, `test_policy_v1_1.py`, and the two files under `submission/` into a clean clone of
the official client. Do not copy a `.request.json` from another submission. Then run:

```bash
uv sync
uv run python -m unittest -v test_policy_v1_1.py
uv run run.py --mock --trials 1 --trial-s 8
uv run run.py --submit
```

This three-cube revision needs a new request ID; the earlier one-cube v1.1 request does not describe this task. Book a
new slot with the new request ID. During the approved slot, keep the CSI operator at the cell stop position and run
`uv run run.py`. Stop on unexpected motion, a coordinate disagreement, ambiguous detection, or a controller alarm.
