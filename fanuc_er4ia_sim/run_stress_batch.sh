#!/usr/bin/env bash
# Development runs only. Environment paths are supplied by the caller.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SEED=20261009
COUNT=
PREVIEW=0
YES=0
YAW_MODE=original
YAW_DEG=
YAW_MAX=
YAW_SEED=
while (($#)); do
    case "$1" in
        --seed|--runs)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in --seed) SEED=$2;; --runs) COUNT=$2;; esac
            shift 2;;
        --preview) PREVIEW=1; shift;;
        --yaw-mode|--yaw-deg|--yaw-max-deg|--yaw-seed)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            [[ -n "$2" ]] || { echo "Empty value for $1" >&2; exit 2; }
            case "$1" in
                --yaw-mode) YAW_MODE=$2;;
                --yaw-deg) YAW_DEG=$2;;
                --yaw-max-deg) YAW_MAX=$2;;
                --yaw-seed) YAW_SEED=$2;;
            esac
            shift 2;;
        --yes) YES=1; shift;;
        --help)
            echo 'Usage: bash run_stress_batch.sh [--seed INTEGER --runs COUNT] [--preview] [--yes]'
            echo 'Yaw: --yaw-mode original | --yaw-mode fixed --yaw-deg 30 | --yaw-mode random --yaw-max-deg 45 --yaw-seed 20261009'
            echo 'No arguments: interactive prompts. --preview never launches or creates run folders.'
            exit 0;;
        *) echo "Unknown option: $1" >&2; exit 2;;
    esac
done
if [[ -z "$COUNT" ]]; then
    [[ -t 0 ]] || { echo 'Noninteractive use requires --runs COUNT.' >&2; exit 2; }
    printf '\nChoose the layout seed:\n  20261009 = original baseline (press Enter)\n  20261010 = alternate layout requested by Gouri\n'
    read -r -p "Layout seed [$SEED]: " ANSWER
    SEED=${ANSWER:-$SEED}
    read -r -p "Yaw mode: original, fixed, random [$YAW_MODE]: " ANSWER
    YAW_MODE=${ANSWER:-$YAW_MODE}
    if [[ "$YAW_MODE" == fixed ]]; then
        read -r -p "Angle for ALL six cubes, 0-45 degrees [$YAW_DEG]: " ANSWER
        YAW_DEG=${ANSWER:-$YAW_DEG}
    elif [[ "$YAW_MODE" == random ]]; then
        read -r -p "Maximum random yaw, 0-45 degrees [${YAW_MAX:-45}]: " ANSWER
        YAW_MAX=${ANSWER:-${YAW_MAX:-45}}
        read -r -p "Independent yaw seed [${YAW_SEED:-20261009}]: " ANSWER
        YAW_SEED=${ANSWER:-${YAW_SEED:-20261009}}
    fi
    read -r -p 'Number of runs: ' COUNT
fi
[[ "$SEED" =~ ^-?[0-9]+$ ]] || { echo 'Seed must be an integer.' >&2; exit 2; }
[[ "$COUNT" =~ ^[1-9][0-9]*$ && ${#COUNT} -le 4 ]] || { echo 'Run count must be a whole number from 1 to 9999.' >&2; exit 2; }
valid_angle() {
    [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] && awk -v value="$1" 'BEGIN {exit !(value >= 0 && value <= 45)}'
}
case "$YAW_MODE" in
    original)
        [[ -z "$YAW_DEG$YAW_MAX$YAW_SEED" ]] || { echo 'Original yaw mode does not use angle/max/seed options.' >&2; exit 2; };;
    fixed)
        valid_angle "$YAW_DEG" || { echo 'Fixed yaw must be a number from 0 to 45 degrees.' >&2; exit 2; }
        [[ -z "$YAW_MAX$YAW_SEED" ]] || { echo 'Fixed yaw mode does not use max/seed options.' >&2; exit 2; };;
    random)
        [[ -z "$YAW_DEG" ]] || { echo 'Random yaw mode does not use --yaw-deg.' >&2; exit 2; }
        YAW_MAX=${YAW_MAX:-45}
        YAW_SEED=${YAW_SEED:-20261009}
        valid_angle "$YAW_MAX" || { echo 'Maximum yaw must be a number from 0 to 45 degrees.' >&2; exit 2; }
        [[ "$YAW_SEED" =~ ^-?[0-9]+$ ]] || { echo 'Yaw seed must be an integer.' >&2; exit 2; };;
    *) echo 'Yaw mode must be original, fixed, or random.' >&2; exit 2;;
esac
printf '\nDevelopment batch: %s sequential runs; layout seed %s\n' "$COUNT" "$SEED"
printf 'Yaw mode: %s; fixed degrees: %s; max degrees: %s; independent seed: %s\n' "$YAW_MODE" "${YAW_DEG:-unused}" "${YAW_MAX:-unused}" "${YAW_SEED:-unused}"
echo 'Original mode preserves [0,26,32,-23,14,-36]. Calibration, physics and all gates unchanged.'
echo 'Fresh capture and folder per attempt. Failures are saved, not retried.'
echo 'Photos/videos stay on YAAD. Run the Windows upload helper after the batch.'
if ((PREVIEW)); then echo 'PREVIEW ONLY: no tests, files or simulations started.'; exit 0; fi
if ((!YES)); then
    [[ -t 0 ]] || { echo 'Noninteractive launch requires --yes.' >&2; exit 2; }
    read -r -p 'Start this batch? Type yes: ' ANSWER
    [[ "$ANSWER" == yes ]] || { echo 'Cancelled; no runs started.'; exit 0; }
fi
[[ -n "${TMUX:-}" ]] || { echo 'Enter tmux before launching.' >&2; exit 2; }
: "${ISAAC:?Set ISAAC to the existing Isaac wrapper directory}"
: "${COSMOS:?Set COSMOS to the existing Cosmos Python}"
HOST_PYTHON=${HOST_PYTHON:-$HERE/.venv/bin/python}
for executable in "$ISAAC/python.sh" "$COSMOS" "$HOST_PYTHON"; do
    [[ -x "$executable" ]] || { echo "Missing executable: $executable" >&2; exit 2; }
done
for executable in flock timeout setsid nvidia-smi docker; do
    command -v "$executable" >/dev/null || { echo "Missing command: $executable" >&2; exit 2; }
done
cd "$HERE"
exec 9>"$HERE/.stress_batch.lock"
flock -n 9 || { echo 'Another batch launcher holds the workspace lock.' >&2; exit 2; }
if ps -eo args= | grep -e '[r]un_sort_pipeline.sh' -e '[c]apture_scene.py' -e '[p]erceive.py' -e '[s]ort_scene.py'; then
    echo 'Existing pipeline process found. No batch started.' >&2; exit 2
fi
nvidia-smi
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
echo 'If another teammate has started heavy work, cancel at the next prompt.'
if ((!YES)); then
    read -r -p 'Resources available for this batch? Type yes: ' ANSWER
    [[ "$ANSWER" == yes ]] || exit 0
fi
env -u CUBE_YAW_DEG -u CUBE_YAW_MAX_DEG -u CUBE_YAW_SEED CUBE_YAW_MODE=original CUBE_LAYOUT_SEED=20261009 "$HOST_PYTHON" -B -m unittest test_sort_planning test_cube_perception test_cube_pose test_runtime_config test_batch_progress test_yaw_config
export HOST_PYTHON CUBE_LAYOUT_SEED="$SEED"
unset CUBE_YAW_DEG CUBE_YAW_MAX_DEG CUBE_YAW_SEED
export CUBE_YAW_MODE="$YAW_MODE"
if [[ "$YAW_MODE" == fixed ]]; then export CUBE_YAW_DEG="$YAW_DEG"; fi
if [[ "$YAW_MODE" == random ]]; then export CUBE_YAW_MAX_DEG="$YAW_MAX" CUBE_YAW_SEED="$YAW_SEED"; fi
mkdir -p "$HERE/runs"
BATCH=$(date -u +%Y%m%d_%H%M%S)_$$
ACTIVE=
STOP=0
trap 'STOP=1; if [[ -n "$ACTIVE" ]]; then kill -TERM -- "-$ACTIVE" 2>/dev/null || true; fi' INT TERM
for ((INDEX=1; INDEX<=COUNT; INDEX++)); do
    ((STOP == 0)) || break
    RUN=$(mktemp -d "$HERE/runs/lieland_seed_${SEED}_dev_${BATCH}_${INDEX}_XXXXXX")
    printf '\n\n\n========== RUN %s OF %s: %s ==========\n' "$INDEX" "$COUNT" "$RUN"
    START=$(date +%s)
    START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf 'Start UTC: %s\n' "$START_UTC" > "$RUN.timing.txt"
    # A separate process group lets an interruption stop the active pipeline.
    PYTHONUNBUFFERED=1 setsid timeout --kill-after=30s 90m bash "$HERE/run_sort_pipeline.sh" "$RUN" > "$RUN.log" 2>&1 &
    ACTIVE=$!
    "$HOST_PYTHON" -u -B "$HERE/batch_progress.py" --log "$RUN.log" --run "$RUN" --pid "$ACTIVE" --index "$INDEX" --count "$COUNT" --seed "$SEED" --started "$START" --interval "${PROGRESS_INTERVAL_SECONDS:-5}" &
    MONITOR=$!
    printf 'Live log (from another terminal): tail -f %s.log\n' "$RUN"
    STATUS=0
    wait "$ACTIVE" || STATUS=$?
    if ((STOP)); then wait "$ACTIVE" 2>/dev/null || true; STATUS=130; fi
    ACTIVE=
    ELAPSED=$(($(date +%s)-START))
    FINISH_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    wait "$MONITOR" || echo 'Progress display stopped; pipeline status and saved reports remain authoritative.'
    printf 'Finish UTC: %s\nElapsed seconds: %s\nPipeline exit status: %s\n' "$FINISH_UTC" "$ELAPSED" "$STATUS" >> "$RUN.timing.txt"
    "$HOST_PYTHON" -B - "$RUN" "$STATUS" "$ELAPSED" "$BATCH" "$INDEX" "$COUNT" "$START_UTC" "$FINISH_UTC" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
status, elapsed = map(int, sys.argv[2:4])
def load(name):
    path = run / name
    return json.loads(path.read_text()) if path.is_file() else None
report = load('execution_report.json')
plan = load('sort_plan.json')
config = load('run_config.json')
if status in (124, 137):
    outcome, stage = 'timed_out', 'inspect_log'
elif status in (130, 143):
    outcome, stage = 'interrupted', 'inspect_log'
elif report is not None:
    complete = (report.get('task_complete') is True
                and report.get('verified_pickups') == 6
                and report.get('verified_placements') == 6
                and report.get('abort_reason') is None)
    outcome = 'physics_success' if complete else 'execution_incomplete'
    stage = ('complete' if status == 0 else 'post_execution_pipeline') if complete else 'execution'
elif plan is not None and not plan.get('picks'):
    reasons = [item.get('skip_reason') for item in plan.get('skipped', [])]
    outcome = 'failed_before_execution'
    stage = 'perception_pose_validation' if 'cube_pose_missing_or_invalid' in reasons else 'planning_no_eligible_picks'
else:
    outcome, stage = 'no_execution_report', 'undetermined_inspect_log'
record = {
    'schema_version': 1, 'experiment_class': 'development',
    'run_id': run.name, 'batch_id': sys.argv[4],
    'batch_run_number': int(sys.argv[5]), 'batch_requested_runs': int(sys.argv[6]),
    'started_at_utc': sys.argv[7], 'finished_at_utc': sys.argv[8],
    'outcome': outcome, 'classification': outcome, 'failure_stage': stage,
    'pipeline_exit_status': status, 'elapsed_seconds': elapsed,
    'cube_layout_seed': config['effective']['cube_layout_seed'] if config else None,
    'cube_yaw_control': config['effective'].get('cube_yaw_control') if config else None,
    'cube_yaws_deg': config['effective'].get('cube_yaws_deg') if config else None,
    'execution_report_present': report is not None,
    'task_complete': report.get('task_complete') if report else None,
    'verified_pickups': report.get('verified_pickups') if report else None,
    'verified_placements': report.get('verified_placements') if report else None,
    'abort_reason': report.get('abort_reason') if report else None,
    'planned_picks': len(plan.get('picks', [])) if plan else None,
    'notes': ['Missing measurements remain null.', 'Inspect saved reports and log for detailed diagnosis.',
              'Ground truth is evaluation only. All validation gates unchanged.'],
}
with (run / 'development_outcome.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(record, indent=2))
PY
    printf '\n========== RUN %s OF %s ENDED - OUTCOME SAVED ==========\n' "$INDEX" "$COUNT"
    # Timeout/interruption may leave container children: never start a new run.
    if ((STATUS == 124 || STATUS == 137 || STATUS == 130 || STATUS == 143 || STOP)); then
        echo 'Batch stopped. Inspect processes and containers before any further run.'
        break
    fi
done
printf '\n========== BATCH ENDED - READY TO REVIEW AND UPLOAD ==========\n'
echo 'Run the Windows Sync-Overclocked.ps1 -Publish helper after checking these results.'
# Two terminal bells mean batch ended, not necessarily physics success.
if [[ -t 1 ]]; then
    printf '\a'
    sleep 0.25
    printf '\a'
fi
