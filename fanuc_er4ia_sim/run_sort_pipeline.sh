#!/usr/bin/env bash
# Capture -> Cosmos -> plan -> physical execution -> review videos.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaacsim}"
: "${COSMOS:?Set COSMOS to the Python interpreter in your Cosmos environment}"
HOST_PYTHON="${HOST_PYTHON:-$HERE/.venv/bin/python}"
for executable in "$ISAAC/python.sh" "$COSMOS" "$HOST_PYTHON"; do
    if [ ! -x "$executable" ]; then
        echo "Python launcher missing or not executable: $executable" >&2
        exit 1
    fi
done
RUN="${1:-$HERE/runs/cube_stack_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"
if [ -n "$(ls -A "$RUN")" ]; then
    echo "Use an empty run directory: $RUN" >&2
    exit 1
fi
echo '[stage] CONFIGURATION'
"$HOST_PYTHON" "$HERE/record_run_config.py" --run "$RUN"
CUBE_LAYOUT_SEED="$("$HOST_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["effective"]["cube_layout_seed"])' "$RUN/run_config.json")"
export CUBE_LAYOUT_SEED
echo '[stage] CAPTURE'
"$ISAAC/python.sh" "$HERE/isaac_run_entrypoint.py" "$RUN/run_config.json" "$HERE/capture_scene.py" --out "$RUN"
echo '[stage] PERCEPTION'
"$COSMOS" "$HERE/perceive.py" --run "$RUN" --stack-cubes
echo '[stage] PLANNING'
"$HOST_PYTHON" "$HERE/plan_sort.py" --run "$RUN"
echo '[stage] EXECUTION'
"$ISAAC/python.sh" "$HERE/isaac_run_entrypoint.py" "$RUN/run_config.json" "$HERE/sort_scene.py" --run "$RUN"
echo '[stage] VERIFY_REPORT'
# Isaac shutdown can return zero even after an exception; check the result.
"$HOST_PYTHON" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
report = Path(sys.argv[1]) / 'execution_report.json'
if not report.is_file():
    raise SystemExit('Execution report missing; inspect the simulator output.')
result = json.loads(report.read_text())
if not result.get('task_complete'):
    raise SystemExit(f"Stack incomplete: {result.get('abort_reason')}; inspect {report}")
PY
echo '[stage] VIDEO_TABLE'
"$HOST_PYTHON" "$HERE/encode_video.py" --frames "$RUN/frames" --out "$RUN/sort_table.mp4" --fps 30
echo '[stage] VIDEO_SIDE'
"$HOST_PYTHON" "$HERE/encode_video.py" --frames "$RUN/side_frames" --out "$RUN/sort_side.mp4" --fps 30
echo "Review $RUN: videos, detections_overlay.png, sort_plan.json, perception_evaluation.json, execution_report.json"
echo '[stage] COMPLETE'
