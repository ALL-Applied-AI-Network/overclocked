# Deliberate camera-calibration mismatch

Local implementation candidate, October 24, 2026. YAAD host tests, container
transport, Cosmos and physics validation are pending. No experiments launched.

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
