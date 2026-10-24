# Cube yaw controls (degrees)

Yaw controls have passed host and container transport checks, with development
physics results recorded below. Fixed 45 degrees has mixed outcomes; it is not
an established reliability boundary. Actual run reports remain authoritative.

Modes:

- `original` (default): exactly `[0, 26, 32, -23, 14, -36]` degrees.
- `fixed`: one selected angle from 0 to 45 degrees, inclusive, for all six cubes.
- `random`: independent uniform samples from 0 to a selected maximum, at most
  45 degrees, using a separate integer yaw seed (default 20261009).

Yaw is about world +Z, in the existing scene convention. Controls set initial
scene orientation; settled image-measured yaw can differ after capture physics.
The two yellow cubes keep fixed XY, but their yaw changes in fixed/random mode.
No positions, camera calibration, friction, or validation thresholds change.
Randomness is separate from the layout generator and global random state.

## Launcher

Normal use: `bash ./run_stress_batch.sh` prompts for layout, yaw mode,
mode-specific settings, run count, and the existing confirmations.
CLI examples (existing environment exports and tmux required for real runs):

```bash
bash ./run_stress_batch.sh --seed 20261009 --runs 1 --preview
bash ./run_stress_batch.sh --seed 20261009 --runs 1 --yaw-mode fixed --yaw-deg 30 --preview
bash ./run_stress_batch.sh --seed 20261009 --runs 1 --yaw-mode random --yaw-max-deg 45 --yaw-seed 20261023 --preview
```

Remove `--preview` only to launch an authorized, resource-coordinated experiment.
Preview validates controls but never creates files or starts tests/models/physics.
CLI batches without yaw options always use original angles, regardless of
inherited yaw environment variables. `--yes` retains its existing meaning.
Repeat attempts in one batch share the same selected yaw seed and six angles;
they are fresh pipeline attempts, not different yaw samples. Change `--yaw-seed`
for a different set. Layout seed and yaw seed have different purposes.

## Direct pipeline environment

The launcher manages these automatically. Advanced direct pipeline use:

- `CUBE_YAW_MODE=original`; unset all other yaw variables.
- `CUBE_YAW_MODE=fixed`, `CUBE_YAW_DEG=30`; unset max/seed.
- `CUBE_YAW_MODE=random`, `CUBE_YAW_MAX_DEG=45`, `CUBE_YAW_SEED=20261023`;
  unset fixed degrees. Random max/seed have defaults when omitted.

Direct imports reject unused mode-specific variables, nonfinite/out-of-range
angles and invalid seeds. Every run_config.json and batch development outcome
records effective mode, active settings (inactive fields null), and all six yaws.
Run conditions identify yaw changes even when the layout is nominal.

Host/Cosmos inherit the selected environment. Capture and execution both use
isaac_run_entrypoint.py to load the recorded controls inside the container and
verify all six yaws and the layout before running target code. The shared Isaac
wrapper and old layout-only bridge remain unchanged. Configuration/ground truth
is for scene construction and evaluation only, never perception repair or grasps.

## Validation stages

1. Host tests: `test_yaw_config` plus the existing required suites. Includes
   original preservation, boundaries, invalid controls, independence, recorder,
   a subprocess bridge probe and rejected mismatches, and CLI preview checks.
2. Container-only bridge probe with no SimulationApp: compare generated settings
   with recorded config using the shared wrapper. Establishes transport only.
3. A fresh single-factor yaw development run on default layout/physics, after
   resource coordination. Inspect capture settings, actual perception and final
   execution report. Require complete, 6 pickups, 6 placements and no abort for
   physics success. Preserve all failed attempts; no automatic pass-seeking retry.

No model/physics validation is established by host checks or preview output.

## Checkpoint before yaw implementation

Uploaded branch checkpoint: c1d0eac0dc34e6b2885dbdd650e3d34c8aaa5785.
The inventory prompt change was followed by four fresh successful simulations:

- Default: lieland_seed_20261009_dev_20261023_190020_18718_1_wDvYLw
- Default: lieland_seed_20261009_dev_20261023_191322_19682_1_Htbr2C
- Alternate: lieland_seed_20261010_dev_20261023_192612_20554_1_Y1mPPz
- Alternate: lieland_seed_20261010_dev_20261023_192612_20554_2_GIlYpC

All four saved execution reports were inspected: task_complete=true,
verified_pickups=6, verified_placements=6, abort_reason=null. These establish
pre-yaw development validation, including the installed inventory prompt;
they do not validate the new yaw implementation. Earlier failures are retained.


## Observed yaw development results - 2026-10-23

All three attempts used default layout seed 20261009. Camera calibration,
friction and existing validation gates were unchanged. They are development
attempts, not formal protocol trials. Ground-truth/tracked poses remain
evaluation-only; no controller repair or recovery was implemented.

| Condition | Run ID | Pickups / placements | Outcome | Duration |
| --- | --- | --- | --- | --- |
| Random 0-45 deg, yaw seed 20261023 | lieland_seed_20261009_dev_20261023_205043_22633_1_VIxOT2 | 6 / 6 | physics_success | 9m53s |
| Fixed 45 deg, attempt 1 | lieland_seed_20261009_dev_20261023_212426_23805_1_jWAg7G | 4 / 4 | execution_incomplete | 7m56s |
| Fixed 45 deg, attempt 2 | lieland_seed_20261009_dev_20261023_215646_24552_1_wl5TBq | 6 / 6 | physics_success | 9m13s |

Random run effective initial angles, in cube-index order:

`[11.382658793761973, 35.04871729086073, 24.014872542583603, 10.832110016014418, 16.514895342515505, 18.324876825970293]` degrees.

The first fixed-45 attempt failed on the fifth pickup (stack_level 4), the
first red cube, tracked as SortCube3. The first four picks passed pickup,
placement and stack-support checks. The failed attempt had objects_lifted=[],
pickup_verified=false, and zero reported trajectory IK failures. The cube
remained on the table during lift. The immediate failure is a failed pickup;
the exact contact/grasp cause is unresolved. No stack-collapse cause is proven.

The repeat was a fresh full pipeline run, not recovery within the failed run.
It completed six pickups and six placements with no abort. Both attempts count:
one failure and one success, not a demonstrated fix or recovery and not a
formal success-rate estimate. No further pass-seeking repeats are planned.

Recorded top-level Python/shell source hashes match between the two fixed-45 attempts.

Validation scope: installation reported 65 host tests passed. Container-only
probes matched original, fixed-30 and random settings in
/workspace/lieland/yaw_bridge_check_ru1znwyb, without SimulationApp. The runs
above separately establish observed physical outcomes. Original-mode behavior
after the bridge change has host/container coverage; no new original-mode full
simulation was performed after installing yaw controls.

All source run folders and failures are preserved. Full reports, images and
frames remain in runs/<run ID>; selected JSON/text evidence is uploaded under
results/development/<run ID> on lieland-stress-tests. Main was not changed.
The failed-run side video was encoded afterward from saved side_frames at 30
fps: 1379 frames, about 46 seconds, yaw45_failure_side.mp4. It was downloaded
and shared by the user in Teams; media is excluded from the GitHub sync bundle.

Checkpoint uploads: random success c85f21e; first fixed-45 failure a27ce7f;
successful fixed-45 repeat 6cd1e35. These identify evidence snapshots, not
different controller fixes. Sai/Gouri received the grasp-failure report; the
team's physical-grasp investigation remains separate from this harness work.

Next harness feature: deliberate assumed camera-calibration mismatch, off by
default, with true calibration preserved for evaluation. Not implemented yet.
