# CubeVisionStack version history

This document separates physical-cell evidence from automated tests and the synthetic official mock. A mock pass
only proves that the client and cell accept the action format and limits; it does not prove a physical grasp.

Dashboard figures below were read on October 26, 2026. Rows still marked `IN PROGRESS` may change as more rollouts
are scored.

| Version | Purpose | Physical result visible on dashboard | What needed fixing |
|---|---|---|---|
| `v1` | First fixed-layout blue 10 mm cube pick into the yellow bin. Established the physical grasp and release motion. | 4/4 completed rollouts successful (100%), 65.5 s mean successful cycle, 0 faults; planned run showed 4/5 completed. | Fixed scene did not test camera mapping across positions or clutter. |
| `v1.1` | Archived source is a narrow-region, three-blue-cube follow-up with three-frame stability, source verification, and one retry. | The dashboard entry carrying the `CubeVisionStack-v1.1` label is a one-cube/five-position task: 3/5 successful (60%), 94.7 s mean successful cycle, 0 faults. See the provenance note below. | Narrow source box and planned-position association could reject valid live detections. The policy also required an exact remaining-cube count. |
| `v1.1.1` | Controlled position validation: exactly one blue cube, five declared positions repeated twice, live X/Y only, narrow 5 mm-margin envelope. | 1/2 completed rollouts successful (50%), 61.3 s mean successful cycle, 0 faults; planned run showed 2/10 completed. | It was intentionally over-constrained: extra blue detections caused a safe hold and the narrow envelope did not represent cluttered-table picking. |
| `v1.1.2` | Blue best-candidate selection. Tracks every safe blue detection across three frames and picks the candidate with the smallest positional spread, then confidence and grasp isolation. | Not physically run. Automated suite: 12/12 passed. Official synthetic mock: 10 rollouts, 99 actions, 0 rejected. | Still needs a supervised physical clutter run to validate broad-region camera mapping, candidate association, and grasp centering. |
| `v1.2` | Expanded-region three-cube policy: 170 x 150 mm source region, isolated/high-confidence candidate selection, stable tracking, and three verified placements. | No matching physical result was visible in the supplied team dashboard. | Needs physical validation. It still safe-holds unless the expected remaining count is visible, so it is stricter than the `v1.1.2` best-candidate policy. |

## Version details

### v1 — physical baseline

- Folder: [`CubeVisionStack-v1`](CubeVisionStack-v1)
- Task: pick one blue 10 mm cube from a fixed layout and place it in the yellow bin.
- Contribution: demonstrated that the physical grasp height, release height, gripper sequence, and conservative motion
  could succeed repeatedly.
- Limitation: a fixed cube position could not distinguish grasp calibration from camera-position error.

### v1.1 — narrow multi-cube follow-up

- Folder: [`CubeVisionStack-v1.1`](CubeVisionStack-v1.1)
- Task in the archived source: detect and place three blue 10 mm cubes from five narrow-region layouts.
- Selection: associates a detected cube with the next submitted position and rejects deviations over 15 mm.
- Verification: confirms source clearance and the expected loose-cube count after every drop.
- Limitation: valid cubes outside the 80 x 80 mm source box or too far from a submitted coordinate are rejected even
  when the live detection is otherwise good.

### v1.1.1 — single-cube position diagnostic

- Folder: [`CubeVisionStack-v1.1.1-position-validation`](CubeVisionStack-v1.1.1-position-validation)
- Task: isolate position-dependent error using exactly one blue cube at five declared positions.
- The placement coordinate is diagnostic metadata only; motion uses the stable live detection.
- Safety behavior intentionally rejects multiple blue detections, instability, low confidence, and targets outside a
  5 mm-margin envelope.
- Lesson: useful as a controlled calibration experiment, but the safe-hold-on-multiple behavior is wrong for a
  cluttered-table picking task.

### v1.1.2 — most-confirmed blue candidate

- Folder: [`CubeVisionStack-v1.1.2-blue-best-candidate`](CubeVisionStack-v1.1.2-blue-best-candidate)
- Task: when several blue cubes are visible, pick the cube whose X/Y is most repeatable across three observations.
- Ranking: lowest position spread, highest mean fit confidence, greatest separation from another candidate, then a
  deterministic coordinate tie-breaker.
- Non-blue objects are ignored by the color gate. Multiple blue cubes are no longer a reason to hold.
- Preserves the physically validated heights, speeds, calibration, workspace checks, source verification, and
  one-retry limit.

### v1.2 — expanded-region three-cube experiment

- Folder: [`CubeVisionStack-v1.2`](CubeVisionStack-v1.2)
- Task: pick and place all three blue cubes across a larger 170 x 150 mm source region.
- The expanded region was chosen to include a strong detection near `(291, -55)` that the narrow policy excluded,
  while keeping bin-side clutter out.
- Selection is detection-driven rather than layout-order-driven, but completion still requires all three cubes and
  exact remaining-count verification.

## Result provenance note

The archived `v1.1` folder describes a three-cube follow-up, while the dashboard entry labeled
`CubeVisionStack-v1.1` describes a one-cube/five-position task. That means the current archive and the deployed
dashboard artifact are not proven byte-for-byte identical. The history preserves both facts rather than claiming
that the 3/5 dashboard score reproduces the exact archived source.

Private result keys and `.request.json` files are intentionally excluded from the repository.
