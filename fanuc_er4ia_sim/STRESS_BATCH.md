# Development batch launcher

Run `bash run_stress_batch.sh` inside tmux, from this project, after exporting
the existing ISAAC, COSMOS, HOST_PYTHON and model-cache settings. No environments
are installed or copied. No system paths are embedded in the launcher.

Blank seed selects baseline 20261009. Seed 20261010 selects the alternate layout.
The script does not track completion across batches. Each attempt uses the same
selected seed and original yaw, calibration and physics. Only four cube positions
are seeded; the two yellow positions remain fixed.

Preview without launching: `bash run_stress_batch.sh --seed 20261010 --runs 4 --preview`.
Automation: supply `--seed`, `--runs`, and `--yes` after checking team resources.
`--yes` skips both confirmations. Preview needs no GPU, tmux or environment paths.

The launcher validates arguments and environment paths, takes a project-local
batch lock, checks for existing pipeline processes, displays GPU/container usage,
and runs the four required host-test modules before launching simulations.
The lock coordinates this launcher only; team resource coordination is still
required. A teammate can start work after the initial resource check.

Each sequential attempt creates a fresh `runs/lieland_seed_*` directory. Existing
pipeline outputs include configuration/source hashes, images, plans and reports;
execution videos exist only when the pipeline reaches encoding. Logs and timing
are sibling files. A development_outcome.json is saved inside each run, including
failures. Null means unavailable, not zero. Success requires the actual execution
report: task_complete true, six verified pickups and placements, no abort. Pipeline
exit status is recorded separately, including post-execution encoding failures.
If reports are missing, detailed stage attribution may need manual log review.

Ordinary pipeline failures are recorded and the next attempt starts. Interrupted
or timed-out attempts stop the batch; inspect processes/containers before resuming.
The 90-minute limit applies per attempt. Ctrl+C requests cancellation; do not use
it simply because the terminal is quiet. During a run, full output goes to its
log. The launcher prints a tail command for viewing from a second terminal.
Disconnecting SSH does not stop a batch running inside tmux. YAAD restarts can.

Run the Windows upload helper after completion; it uploads code and selected
small evidence in separate GitHub folders. It does not upload full logs or media.
The launcher never pushes Git, books trials or submits policies. These are
development attempts, not formal protocol experiments. Later perturbations need
their own validated controls before being added to this launcher.

## Live progress display

Each run now prints a heartbeat every five seconds, with the seed, run count,
elapsed time, current stage and time since log activity. Stage transitions are
explicit log markers; model-ready output advances the displayed perception
substage when available. Latest log messages and reported pickup/placement
events appear in the same terminal. Full logs remain saved unchanged apart
from the new stage markers. The shared Isaac wrapper is not modified; buffered
container output can delay visible pickup updates. No live counts are invented.

The batch bar counts PREVIOUS finished attempts, including failures, and does
not estimate percentage time remaining or successful runs. The active attempt
is counted when its final outcome is saved. A heartbeat establishes that the
monitor is alive and the pipeline PID exists, not that physics is advancing.
A quiet log does not establish failure. Saved execution reports and final
outcomes remain authoritative. Display failure does not change pipeline status.

Set PROGRESS_INTERVAL_SECONDS to a number from 0.2 to 60 before launching to
change refresh frequency. Default is 5. Demo: use the existing host interpreter
to run batch_progress.py --demo. The demo emits synthetic logs only and does
not use a model, GPU, Isaac, or real saved runs. test_batch_progress adds checks
for partial lines, repeated pickup messages, final log draining and missing logs.
