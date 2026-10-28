#!/bin/sh
# keep_running_cfg.sh A|B : the client in a loop (restarts every 15 s when the cell says "all trials finished"), with the settings of mode_A.env / mode_B.env.
cd "$(dirname "$0")"
MODE="$1"; [ -f "mode_$MODE.env" ] || { echo "usage: $0 A|B"; exit 1; }
while true; do
  set -a; . "./mode_$MODE.env"; set +a
  uv run run.py >> logs/slot_table_loop.log 2>&1
  sleep 15
done
