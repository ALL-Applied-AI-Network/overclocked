# Inventory identification prompt change

Development comparison: lieland_pose_compare_inventory_20261023_185137_sdjdvdma

Two alternate-seed attempts stopped before execution on 2026-10-23:
lieland_seed_20261010_dev_20261023_182630_17307_1_IX81vR
lieland_seed_20261010_dev_20261023_182630_17307_2_cR5X6i
Each had one red cube reply missing confidence. Crop images were visually
reviewed: two red blocks and an orange-looking distractor per source run.

Paired comparison: 14 saved crops, 28 Cosmos calls, alternating prompt order,
300-token budget. Four red inventory crops, two orange controls, four yellow
and four blue pose-crop controls. Original: 8/14 correct gate decisions;
candidate: 14/14. All six original incorrect decisions omitted confidence.
Candidate accepted 12 intended cubes and rejected both orange controls.
Controls and duplicate layouts are correlated; this is development evidence,
not a formal success-rate or confidence-calibration study.

Exact candidate read from saved run_config.json and applied to detail_prompt.
AST comparison verified all other perception code unchanged. No gate changes,
no confidence insertion, no orange-to-red mapping, no ground-truth input.
Production inventory gate remains category=cube, matching colour, confidence
at least 0.8. Pose, planning, pickup and support gates remain unchanged.

Source before SHA-256: 7a9834a45e9a85b5347fe19d6bfe7134b60bd68aca8bf2cbd898b1156fbfbc29
Source after SHA-256: 8153646193f099270addaf71af7d560f8cfc13d280f663f808be199882bd5c5f
Comparison tool SHA-256: 4bdae131d697e6afb4840acf07d63067f758e3b4d44480252dd5c54517e73f56
Backups: /workspace/lieland/inventory_prompt_backup__mredbkq

Fresh full-pipeline default and alternate validation pending at application time.
Saved-crop inference is model evidence only; no physical execution was tested.
Comparison scripts use this verified YAAD workspace and preserved source runs.
