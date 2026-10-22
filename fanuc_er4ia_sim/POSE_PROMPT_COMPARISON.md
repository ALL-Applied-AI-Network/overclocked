# Saved-crop prompt comparison

Development diagnosis only. This tool never modifies perceive.py, ground truth,
captures, historical replies, planning, physics or any acceptance gate. It does
not launch Isaac. Ground-truth files are not read. The candidate removes the
concrete blue/14.2/confidence/example-reason response and describes output fields
in words. The actual image-derived measurement is appended to both prompts.
This tests the revised wording as a whole, not a proof about one token's effect.

## Prepare without GPU

Use the existing host interpreter to run `compare_pose_prompts.py --prepare`.
Default sources are the two successful default runs and five new alternate-seed
pose failures, listed in DEFAULT_SOURCE_RUNS near the top of the tool. For other
sources, repeat `--source-run FOLDER_NAME` to replace that default list. No source
images or runs are copied or overwritten. Every eligible saved cube crop from
those runs is included. Original image-fit IoU below 0.85 is excluded and recorded,
as in the production code; no model result is invented for excluded cases.

Preparation verifies that image, measurement and historical raw reply exist,
and creates a unique runs/lieland_pose_compare_* directory. run_config.json
contains source/image/measurement/reply hashes, exact prompt texts, case list,
and planned call count. Missing inputs stop preparation. The printed
PREPARED_FOLDER is the exact path to use for the execution step.

## Execute only after review and resource checks

Inside tmux, use the existing Cosmos interpreter with `compare_pose_prompts.py
--execute` followed by the actual prepared directory. Set the same existing
model/cache environment variables as the working pipeline. Use GPU 0, checked
for availability immediately before launch. Do not run concurrently with another
model or simulator job. This tool is separate from the batch launcher and does
not acquire its lock; team resource coordination is required.

The tool verifies its input and source hashes, loads Cosmos once, and makes two
calls per crop using existing ask(), its generation parameters, and the same
400-token budget as production. Order alternates original-first/candidate-first
between crops. It uses existing parse_json() and mirrors category/colour, finite
number, confidence [0.8,1], and square-yaw disagreement <=3-degree checks. These
are reporting checks only; it does not generate grasps or run the planner.

Each full prompt, raw reply, input measurement and validation result is saved
under cube_pose_crops with unique case/variant names. development_outcome.json
is updated after each completed call, and on caught errors/interruption. No
reply is silently retried. A started experiment cannot be executed again; make
a NEW preparation to repeat after an error. Hard process termination or machine
restart can leave status running; preserve it and inspect the saved evidence.

The existing upload helper includes run_config.json, development_outcome.json,
and the flat cube_pose_crops/*.txt and *_measurement.json files. Images remain
on YAAD and their original paths/hashes are recorded. Comparison folders count
as development evidence entries, not physical simulation attempts. Missing
pickups/placements/task_complete remain null.

## Interpretation

Compare paired outcomes by source run, cube and colour, including successful
baseline cases; total accepted counts alone can conceal regressions. One paired
pass on correlated saved crops is diagnostic evidence, not a formal statistical
protocol or physics validation. Even if the candidate improves replies, it
requires a separately reviewed production change, required host tests, then
fresh default and alternate-seed simulator validation. Preserve every attempt,
including unsuccessful comparisons. Do not weaken gates or fabricate confidence.

## Host checks

Run the four project unittest modules and test_pose_prompt_compare. New checks
cover the known copied-example rejection, symmetry, invalid/missing numeric
fields, confidence and angle boundaries, plus a mocked paired run that checks
evidence preservation and refusal to overwrite results. Mocked tests load no
model and establish host behavior only.
