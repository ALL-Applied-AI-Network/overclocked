#!/bin/sh
# Keeps the client alive between runs: when the cell says "all trials finished" the client exits; this starts it again every 15 s until
# the operator reopens the run (or approves another request). Stop it with:  pkill -f keep_running.sh; pkill -f "python3 run.py"
cd "$(dirname "$0")"
while true; do
  SKIP_WRIST_XY=1 ADAPTIVE_OFFSET=1 GRASP_FRAME=gripper uv run run.py >> logs/slot_table_loop.log 2>&1
  sleep 15
done
