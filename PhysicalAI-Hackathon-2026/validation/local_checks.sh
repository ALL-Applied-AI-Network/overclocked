#!/bin/sh
# Local-only checks. Never starts the real client or reads request credentials.
set -eu
cd "$(dirname "$0")/.."
command -v uv >/dev/null 2>&1 || { echo "BLOCKED: uv is not installed/on PATH." >&2; exit 127; }
unset TRIAL_MODES TARGET_OBJECT ADAPTIVE_OFFSET GRASP_FRAME SKIP_WRIST_XY GRASP_OFFSETS CUBE_DETECTOR CAPTURE_ONLY CAPTURE_DEPTH
private_dir="logs/validation-local-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$private_dir"
run_check() {
    name="$1"
    shift
    if "$@" >"$private_dir/$name.txt" 2>&1; then
        cat "$private_dir/$name.txt"
        printf '%s,0\n' "$name" >>"$private_dir/exit-codes.csv"
    else
        code=$?
        cat "$private_dir/$name.txt"
        printf '%s,%s\n' "$name" "$code" >>"$private_dir/exit-codes.csv"
        exit "$code"
    fi
}
printf 'check,exit_code\n' >"$private_dir/exit-codes.csv"
run_check uv-sync uv sync
run_check vision uv run test_vision.py
run_check mock uv run run.py --mock --trials 1 --trial-s 40
if [ "${1:-}" = "--battery-mock" ]; then
    export TRIAL_MODES=B TARGET_OBJECT=battery ADAPTIVE_OFFSET=1 GRASP_FRAME=gripper
    run_check battery-mock uv run run.py --mock --trials 1 --trial-s 40
fi
printf 'LOCAL CHECKS FINISHED. No real client started. Private logs: %s\n' "$private_dir"
