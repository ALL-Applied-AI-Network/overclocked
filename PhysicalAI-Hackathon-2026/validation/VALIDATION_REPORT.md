# Battery validation — 28 October 2026

Status: **local setup and compatibility checks passed; new real trials still blocked. No physical repeatability claim.**

Source: Saai commit `fc55bfec203ff5147377ea6aa8690f7ce7fd00e7`.
Working branch: `lieland/battery-validation`. Policy, calibration, modes, offsets and submissions are unchanged.

## Results and evidence

- The handoff reports one end-to-end AA battery success on 27 October, Mode B. Treat this as a reported success, not an official score.
- `logs_summary/trials_27oct.txt` contains seven start/end sequences: one earlier cube sequence and six sequences in run `r20261027-4a6ed5`. IDs t02 and t03 each recur. Do not deduplicate them or interpret seven log sequences as seven officially scored rollouts.
- Three of the six sequences in that run stop with “no bin seen.” These are distinct logged attempts, not a measured population failure rate.
- The 15:30:35 t02 sequence detects one 53 × 20 mm blob near (380, -105) mm before the +10 mm table-y correction, selects (380.5, -95.0), yaw +4.5°, and reaches a grasp waypoint near (382, -100, -330). The implied across-jaw correction is approximately -5 mm at R≈5°. These are policy estimates/commands, not measured placement truth.
- That sequence subsequently reports “nothing left to do after 1 cube(s).” Neither this message nor the motion completion establishes successful release into the bin. Official scores and destination-confirming video are absent from the committed excerpt.
- Historical battery success rate: **not estimable from the supplied material**. Official scored denominator and object identity for every sequence are unknown.
- New controlled real trials: **0 started; success rate not available**, not 0% and not 100%.

See [historical_attempts.csv](historical_attempts.csv) for separate log occurrences. Missing values are intentionally blank.

## Source review

`Bbat` sets TRIAL_MODES=B, TARGET_OBJECT=battery, ADAPTIVE_OFFSET=1 and GRASP_FRAME=gripper.
Mode B applies WRIST_SHIFT_B=(2.3,13.4) mm. The committed `manual_offset.txt` is -5.0 mm; it overrides adaptive/probe offsets at each grasp. Preserve it and record its actual bytes before and after each trial. Do not tune it during the initial batch.

In `policy.py`:
- `look()` refuses a pick if no bin has been seen; after seeing one it can retain its previous location. Keep the bin fixed and visible throughout each trial.
- If the named detector finds nothing, class-agnostic table blobs can still become targets; with multiple candidates and an unavailable detector, ranking becomes largest-first. Battery-specific identification is therefore not guaranteed. Start with one battery and no distractors.
- `act()` queues pick and place together and increments `placed` without an observation confirming grasp or delivery. A later look can retry a still-detected object, but this is not verified recovery.
- Arrival timeout logs and proceeds after 15 seconds. Record such timeouts as a distinct failure signal.
- The fixed grasp height and narrow jaw margin are limitations; handoff calibration may change if hardware is swapped.

`mode_A.env` does not clear inherited TRIAL_MODES or TARGET_OBJECT. Use a fresh, explicitly configured environment for any future A/B comparison. Do not alternate modes casually in the baseline.

## Camera review

Inventory verified at the pinned source:
- `debug_samples/20261027-152847_r2_look0_raw.png` (table, before no-bin attempt).
- `debug_samples/20261027-153110_r2_look0_raw.png` (table, before reported battery success).
- `debug_samples/20261027-153133_r2_peek0_wrist.png` (wrist).
- `debug_samples/20261027-153205_r2_look1_raw.png` (table, after reported success).

The wrist frame was visually inspected: a battery is partly cut off by the bottom image edge; it cannot demonstrate a grasp or bin delivery. The three table images exceed the connector's inline file-content limit and returned empty content. Browser fallback reached a signed-out private-repository 404. Their pixels remain **unreviewed**; no before/after clearance or bin claim is inferred. Open them in an authenticated GitHub session or supply local copies.

## Local reproduction

41 original text files were copied to an isolated local snapshot and verified against their Git blob SHA-1 values again after the checks. Policy, calibration, mode files, offsets and uv.lock remain byte-for-byte unchanged.

Installed locally: uv 0.13.6; CPython 3.12.16; numpy 2.5.4; opencv-python-headless 5.0.0.93; PyYAML 6.0.3; requests 2.34.2. The user ran the official installer/setup helper because downloads from the agent shell were blocked. Dependencies are now installed and cached. Subsequent checks below ran with UV_OFFLINE=true.

| Command | Verified result |
|---|---|
| `uv sync` | PASS, exit 0; 8 installed packages checked |
| `uv run test_vision.py` | PASS, exit 0; 18/18 synthetic cube detections, maximum printed position error 0.43 mm and yaw error 2.8 degrees |
| `uv run run.py --mock --trials 1 --trial-s 40` | PASS, exit 0; 1 rollout, 357 actions, zero rejected |
| Same mock with Bbat environment | PASS, exit 0; 1 rollout, 356 actions, zero rejected |

Both mocks ended with “Ready for the real robot.” This is the client's format/API verdict, **not physical readiness certification or battery success**. Their synthetic scenes did not contain a detected bin or pickable target; the policy held after moving to the look pose. Neither grasp nor release was tested. Detector/Cosmos services were unreachable; fallback ran.

An initial user-run mock sent 354 actions with zero rejections, then failed in `run.py:98` while printing the check-mark under Windows cp1252. This was a genuine nonzero process exit, not a policy rejection. The Windows validation launcher now sets PYTHONUTF8=1 and PYTHONIOENCODING=utf-8 for its child processes and restores prior settings afterward. Separate stdout/stderr files prevent PowerShell 5.1 from presenting ordinary native stderr as a misleading RemoteException. No policy or client implementation was changed.

The launcher was exercised end to end after the fix. [local_check_results.json](local_check_results.json) contains sanitized outcomes. Private stdout/stderr, combined logs and summary remain in ignored `logs/validation-local-20261028-132904/`. No raw private logs are committed.

The default vision test uses synthetic cube frames, not batteries. Use [local_checks.ps1](local_checks.ps1) with `-BatteryMock` or [local_checks.sh](local_checks.sh) with `--battery-mock` to reproduce checks. These helpers do not start the real client.

## Provisional recommendation and next work

Initial baseline: **Bbat, manual across-jaw offset -5.0 mm in gripper frame**. This reproduces the handoff configuration; it is not evidence that B beats A. Require five comparable Bbat repetitions, then separately labelled position/angle variations. A/B selection requires matched conditions and enough trials per mode.

Follow [TRIAL_PROTOCOL.md](TRIAL_PROTOCOL.md). The next highest-priority validation fix is reliable bin acquisition: verify visibility with saved frames before spending repeated battery trials on a scene where no destination is detected. Do not weaken the no-bin gate. After measuring the unchanged baseline, prioritize explicit grasp/destination verification and distinguishing “no target,” “no bin” and “completed.”

## Exact blockers

1. Authenticated access/local copies for the three large table frames.
2. The active approved request/run and remaining rollout budget, confirmed by the operator; the handoff's “3 of 10 used” is stale relative to later t03 log entries.
3. The private request file available **locally on the one designated client laptop**. Do not paste keys, tokens or result links into chat or Git.
4. Confirmed CSI operator/slot, no competing client, unchanged gripper/tool/camera setup, and the operator's measured baseline battery/bin placement.
5. Official scores plus private recorded video/dataset to confirm delivery and reconcile repeated IDs.

No booking, new submission, client restart or real trial was performed. Public-safe report entries use IDs and relative filenames only; raw client/status logs and request files remain private.
