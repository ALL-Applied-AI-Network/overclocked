# Tested pose-prompt update

Applied UTC: 2026-10-22T22:26:38.843692+00:00
Comparison: lieland_pose_compare_20261022_220638_27clbx5l
Comparison evidence uploaded at commit 87240b623c6d2cef716b5accae430e1ac97019e7.

The original prompt passed 35/40 saved crops; the candidate passed 40/40,
including 12/12 successful-default crops and the five previously rejected blue
crops. No passing case regressed in this one paired development comparison.
This is saved-crop model evidence, not physical stacking validation or a formal
statistical result. Two earlier source runs supplied only five eligible crops;
the comparison does not fix or prove complete scene inventory.

Only CUBE_POSE_PROMPT changes in production code. The candidate is read directly
from the preserved comparison run_config.json. Parsed source outside that
assignment was verified identical before applying. Category, colour, finite
number, confidence, yaw, image-fit, inventory, pickup and support gates stay
unchanged. Ground truth is not used to supply perception or grasps.

perceive.py SHA-256 before: 41f4eaaf46d8d3844ab3d89a4974bca25bf5e9b2617f8c490f5aa5af09156f74
perceive.py SHA-256 after: 7a9834a45e9a85b5347fe19d6bfe7134b60bd68aca8bf2cbd898b1156fbfbc29
Original source and package-list backup: /workspace/lieland/tested_pose_prompt_backup_thctidof

Fresh default and alternate-seed physics validation is pending at application
time. Read the later execution reports and development outcomes for results;
this note does not automatically change when validation runs finish. Preserve
all historical runs and all failed/interrupted follow-up attempts.
