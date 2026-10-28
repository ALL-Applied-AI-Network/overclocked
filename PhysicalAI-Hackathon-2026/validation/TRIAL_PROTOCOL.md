# Controlled battery trials

Use the unchanged Saai baseline first. This protocol does not authorize changing the robot's workspace, tool, limits or calibration.

## Before any real trial

1. On the designated client laptop, check out `lieland/battery-validation` from the team repository. Keep the private request file outside version control. Do not use the organizers' repository as a push target.
2. Run `uv sync`, `uv run test_vision.py`, and `uv run run.py --mock --trials 1 --trial-s 40`. Record versions, source commit, exit codes and verdicts. A mock pass is not a physical success.
3. Confirm the active request, official remaining budget and operator-controlled slot privately. Do not start another client or restart an active one. `run.py --status` prints a keyed results URL: inspect it privately and do not paste its output.
4. Operator confirms gripper/tool/frame setup and the baseline battery, bin and approved workspace. Restore the successful scene using the table frame `20261027-153110_r2_look0_raw.png`; do not treat detection coordinates or the current placement.yaml as measured truth. The latter currently requests battery (330,-90,0), bin (375,205), which differs from the saved log.
5. Have the operator measure baseline battery centre P0=(x0,y0), long-axis angle theta0 in UF0, and bin centre. Photograph the scene; keep tape and yellow objects away from the bin's image.
6. Use Bbat in a fresh environment: TRIAL_MODES=B, TARGET_OBJECT=battery, ADAPTIVE_OFFSET=1, GRASP_FRAME=gripper; no inherited CAPTURE_ONLY, CAPTURE_DEPTH, GRASP_OFFSETS or SKIP_WRIST_XY. Detector setting remains the source default open_vocab. Record whether OWLv2/Cosmos tunnels are reachable and the actual detector route on every attempt.
7. Verify `manual_offset.txt` contains exactly -5.0 and preserve the original offset-state files. Freeze offset, source, services, lighting, camera, battery and bin for the comparable batch. An offset change creates a new condition; do not edit it while the robot is executing.
8. Agree the layout with the operator. Current submission says layout=same; variations require operator approval and any required request amendment. Do not silently change an approved setup.

## Trial order

Start with **five completed, operator-scored repetitions at the same P0/theta0**. These are B01–B05 in trials.csv, all initially marked planned. Keep a failed or interrupted started attempt in the ledger; create another row for any repeat.

Only after the baseline and with budget/operator approval, run separately labelled variations:
- V01: same position, theta0+30 degrees.
- V02: same position, theta0-30 degrees.
- V03: P0+(20,0) mm, original angle.
- V04: P0+(0,20) mm, original angle.

These are proposed relative changes, not approved robot targets. Operator must keep them within the cell's allowed setup and confirm clearances. If budget is limited, complete the comparable baseline first; reduce variation count, never hide failed starts. One variation attempt is exploratory; repeat it before claiming repeatability for that condition.

Do not recommend A over B or vice versa from unmatched placements. After the baseline, a future A/B comparison must use the same battery placements, service state and fixed manual offset, with TARGET_OBJECT=battery in both modes and TRIAL_MODES explicitly A or B. Use balanced order and report per-condition results.

## For each started trial

Record a unique local trial_id plus official run/rollout IDs and start time. Reused official IDs must retain separate local rows. Record measured placement and detected position separately.

Capture: source commit, mode, manual offset, detector route/service state, bin detection, battery detection, actual grasp, release into yellow bin, official score, reason for failure, interventions, elapsed time, and relative evidence filenames. Use unknown/blank when not observed.

Keep raw logs/frames in a private directory outside Git (or ignored logs/debug). Save before-table, hover-wrist, post-table and destination-confirming recording. Do not rely on disappearance from the original position. Verify delivery with the official recording/operator and score.

If the bin is not detected, the policy should refuse the pick. Record the attempt, preserve its frame and diagnose visibility before repeating. If a fault or unexpected behavior occurs, the operator handles it; do not switch/restart mid-trial.

## Reporting

Primary rate = officially successful / all started trials for the frozen condition. Also show scored successes/scored trials, pending scores, interruptions and interventions. A missing score is not an assumed failure or success; show bounds [successes/all starts, (successes+unknown)/all starts] until resolved. If no trials started, report “not measured.”

Report observed grasp and bin delivery rates separately; label missing evidence. A success after retry counts as final success but not first-attempt success. Do not pool old exploratory runs, mock checks, changed offsets or variation conditions into the baseline denominator.

Five successes out of five are encouraging but a small sample; they do not prove general reliability. Freeze the demo configuration only after reviewing actual scored trials and preserve failure footage/results.

Before commit, use explicit filenames for sanitized records. Never stage raw logs, request files or private result URLs. If a secret is accidentally found, stop publishing that artifact and sanitize locally; do not quote it in the report.
