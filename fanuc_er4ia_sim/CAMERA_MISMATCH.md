# Deliberate camera-calibration mismatch

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
