# Deliberate camera-calibration mismatch

## Startup investigation update (October 26)

This is an unresolved startup problem, not a demonstrated calibration/grasp
failure. A nominal current-code run also stalled with camera errors zero.
`lieland_seed_20261009_dev_20261025_024124_27949_1_RUfxMl` was interrupted
after 594 seconds, exit 130, six planned picks, no execution report.
Historical post-yaw source at `674ad6c` then completed a nominal full run:
`lieland_seed_20261009_dev_20261025_025632_28578_1_GvojBg`, 539 seconds,
complete, six pickups, six placements, null abort. All 37 recorded top-level
Python/shell hashes matched that commit. These results were uploaded at
`db1a8bc2a745478e9df23001eb9701f11f93f710`.

A requested three-attempt current-code nominal batch stopped during its first
attempt, `lieland_seed_20261009_dev_20261026_170700_29891_1_pHZlWK`.
It completed capture/perception and planned six picks, then waited in native
Kit startup (`simulation_app.py:574`, `self.app.startup`). User interrupted
after 2376 seconds (39m36s), exit 130; no execution report. Attempts two and
three never started. GPU was idle; process status and loaded libraries were
saved. No log files were found under `/isaac-sim/kit/logs` in the live container.
Evidence uploaded at `5e62853b9f74e8526f3ba1d921d791d81fe57e97`.

Isolated startup diagnostics are NOT task/physics attempts. They execute a
transformed copy of the selected source that exits after SimulationApp startup,
using a common working directory, inputs and wrapper. A common 120-second
watchdog ends stalled probes. Under `/workspace/lieland/startup_compare_20261025`:

- `old_ttjdnw9p`, `old_cfpl5jp5`: old source timed out at 121.1s each.
- `current_rwrmypj4`: current source started in 26.0s; no extra logging.
- `old_native_3hrlwoz5`: old source with Python flushing plus native log settings
  started in 26.1s.
- `current_native_dp77rozh`: current source with the same diagnostic changes
  started in 26.0s.
- `current_40xhmf1_`: current source without extra logging timed out at 121.1s.
- `current_stdout_hcrtidk5`: current source with ONLY Python stdout/stderr
  line buffering and write-through enabled started in 26.1s.

The earlier successes without flushing mean output flushing is only a candidate
workaround; root cause and reliability are unproven. Instrumentation/time effects
remain possible. Plugin-path inspection showed the apparent telemetry copies of
core Carbonite plugins are the normal Kit symlink targets with identical hashes;
there was no evidence of a conflicting copy, and no image/library changes were made.

Candidate change: the project Isaac entrypoint configures stdout/stderr with
`line_buffering=True, write_through=True` before loading its target script.
No shared wrapper, simulator, calibration math, controller, or gate changes.
The synthetic-target host test asserts both stream settings before target work.
Full nominal pipeline validation and subsequent 1 mm physics validation are pending.
The 72-test installed and explicitly isolated diagnostic suites previously passed;
an earlier 71-vs-72 isolation issue was traced to compare_inventory's hardcoded
import-path insertion and corrected in the verification runner.

Windows upload helper accepts exact native-log/process-map/status filenames;
50 local helper checks passed. Detailed standalone probes remain separate from
the main project's run folders and are not automatically uploaded by that helper.

Installed October 24, 2026. Both isolated-candidate and installed host suites
passed 71 tests. Five configuration-only Isaac container probes passed: nominal,
1/2/3 mm translation, and 0.5 degree rotation. Probe evidence:
`/workspace/lieland/camera_bridge_check_eucutk4_`. These checks did not load Cosmos,
SimulationApp or physics. Physics validation remains pending.

First attempt: `lieland_seed_20261009_dev_20261025_015210_26073_1_eydoGZ`.
Default layout, original yaw, +1 mm world-X assumed camera translation, unchanged
physics. Capture and Cosmos perception completed; planning produced six picks
and skipped one outside-work-surface detection. Execution log stopped after
the configuration bridge. GPU was idle; the Python process was sleeping in a
futex wait. A separate bounded import probe passed numpy, fk, ik, cv2,
scipy.optimize and cube_pose, but did not locate the original wait.
User interrupted at 2026-10-25 02:14:25 UTC after 1335 seconds (22m15s), exit 130.
Saved outcome is interrupted, with no execution report and null physics counts.
Cause remains unresolved: no demonstrated calibration/grasp failure or success.
Container removal was verified. Code/evidence uploaded at
`6c79a27feabcfd8094b67a75673a868e69016816`.

Startup diagnostics added afterward: execution prints import, plan-validation,
and SimulationApp startup milestones. The Isaac bridge arms a diagnostic-only
120-second repeating Python traceback dump to `execution_startup_traceback.txt`.
It does not stop/retry execution. The timer is cancelled when SimulationApp
construction returns, or when the target exits. No controller or gate changes.
These diagnostics have not yet been exercised in a fresh physics attempt.
The Windows sync helper now includes true/assumed calibration metadata and
the exact startup traceback filename, retaining existing size/credential checks.

The rendered camera and its intrinsics remain nominal. Capture writes actual
USD calibration to `camera_true.json` (evaluation only), and a separate assumed
calibration to `camera.json`. Production perception and planning continue to
read only `camera.json`; no fallback reads true calibration. `camera_mismatch.json`
records the error definition. Scene layout, contact physics and gates are unchanged.

Translation is positive world X: assumed optical centre = true centre +
`[mm/1000, 0, 0]` metres. Choose 0 (off), 1, 2 or 3 mm. Rotation is positive
0.5 degrees about world Z through the camera optical centre. It rotates each
camera basis vector using the right-hand rule, retaining the optical centre.
Intrinsics and table-plane height remain unchanged. Rotation and translation
cannot be enabled together. These axes are development choices, not an adopted
formal team protocol; coordinate axis/result conventions with Andrew.

Environment controls: `CAMERA_TRANSLATION_MM` and `CAMERA_ROTATION_DEG`, both
default zero. Invalid/nonfinite magnitudes and simultaneous errors fail before
expensive pipeline stages. Batch CLI/interactive defaults override inherited
camera settings with zero unless explicitly selected. Direct pipeline use reads
the environment. `run_config.json` records canonical effective settings and
source hashes; batch outcomes copy the camera settings. The Isaac bridge clears
inherited camera variables and restores validated recorded values before capture
or execution. Older run configs without the new field retain nominal behaviour.

Consumer audit: `perceive.py` constructs CameraModel from camera.json for image
fitting; `plan_sort.py` and the alternate `plan_grasps.py` do likewise.
`cube_pose.py` uses its supplied camera model. `sort_scene.py` consumes the
saved plan, performs FK/IK, renders nominal review cameras and checks tracked
object poses for evaluation. It has no runtime visual pose estimator or
calibration repair path. True calibration is not imported by those consumers.

Preview only, from `/workspace/lieland/fanuc_er4ia_sim` in YAAD:

```bash
printf '\n\n\n========== PREVIEW ONE 1 MM CAMERA ERROR; NO SIMULATION ==========\n'
bash ./run_stress_batch.sh --seed 20261009 --runs 1 --yaw-mode original --camera-translation-mm 1 --preview
```

After host and container checks pass, the user can start the same command
without `--preview` in tmux, with the verified environment exports loaded.
Leave confirmations enabled. This is one development attempt with default
layout, original yaw and unchanged physics, not a sweep or reliability estimate.
Do not start it as part of installation. Save and upload failures as well as passes.

Required host checks include existing required suites plus
`test_calibration_mismatch`. New tests cover default equivalence, translation
units/sign, rotation sign/pivot/orthonormality, independent true/assumed files,
and invalid settings. Host tests do not establish Isaac/model/physics success.
Container bridge testing must precede the first experiment. Physics success
requires the saved execution report: complete, six pickups, six placements,
null abort reason.
