# Lieland: development baseline, 2026-10-20

These are development runs, not formal protocol results.
No perturbation code had been added.

## Setup
Code: /workspace/lieland/fanuc_er4ia_sim
Host interpreter: /workspace/saai/fanuc_er4ia_sim/.venv/bin/python
Isaac wrapper: /workspace/saai/isaac/python.sh
Cosmos interpreter: /workspace/saai/cosmos-env/bin/python
Model snapshot: /workspace/hf/hub/models--nvidia--Cosmos3-Nano/snapshots/e59a53c25979a090fa8706c9acc0c254a6e89b92

Required four host-test modules returned OK.
Python and shell files matched Sai's current copy when compared.
This comparison does not identify the source version of Sai's older saved run.

## Attempt 1: failed before physical execution
Run: runs/lieland_baseline_dev_20261020_165035_yqvkBV
Log: runs/lieland_baseline_dev_20261020_165035_yqvkBV.log

The first blue cube's image-derived yaw was -35.2109 degrees.
Cosmos reported 14.2 degrees with confidence 0.9.
The disagreement exceeded the existing 3-degree gate modulo 90 degrees.
The pose was rejected; the planner produced 0 picks and 6 skips.
No physical pickups were attempted. This is not a missed-pickup result.
Evidence: detections.json, sort_plan.json and cube_pose_crops/cube_0.txt.

## Attempt 2: unchanged retry succeeded
Run: runs/lieland_baseline_retry_dev_20261020_175604_DAoCi5
Log: runs/lieland_baseline_retry_dev_20261020_175604_DAoCi5.log
Source hashes: runs/lieland_baseline_retry_dev_20261020_175604_DAoCi5.source.sha256

execution_report.json:
- task_complete: true
- verified_pickups: 6
- verified_placements: 6
- abort_reason: null

Approximate end-to-end duration: 9 minutes 30 seconds.
Both review videos were downloaded and watched by Lieland.

## Interpretation
One failure and one success were observed under unchanged development conditions.
Both runs are retained. No formal success rate is established.
Confidence, inventory, pose, pickup and support gates were not weakened.

## Current development status
After the successful unchanged baseline, selectable CUBE_LAYOUT_SEED and regression tests were added. All four required host-test modules returned OK on YAAD. The shared Isaac wrapper does not yet forward CUBE_LAYOUT_SEED into its container. Alternate-layout simulator validation is pending. This code snapshot includes that unfinished feature; the successful baseline predates it. Yaw, camera-error and friction controls are not implemented.

## Latest update: 2026-10-21
The earlier container-forwarding limitation is now resolved using isaac_seed_entrypoint.py. Host and container configurations matched for seed 20261010. All four required host-test modules passed after the changes. The updated default, with the seed unset, completed 6 verified pickups and placements in 535 seconds. One uninterrupted alternate-seed attempt stopped at the inventory gate; repeatability is not established. A separate attempt was user-interrupted. Small evidence bundles for all five development attempts are in results/development at the repository root. Yaw, camera-error and friction controls remain pending. Batch outcome recording is not yet automatic.
