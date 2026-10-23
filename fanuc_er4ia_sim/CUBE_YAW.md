# Cube yaw controls (degrees)

Implementation-stage feature. Host checks and a host/container bridge probe
must pass before the first development simulation. No yaw physics results are
claimed by this document. Actual run reports remain authoritative.

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
