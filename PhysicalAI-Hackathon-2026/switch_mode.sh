#!/bin/sh
# switch_mode.sh A|B : change the client's mode. Refuses while a trial is running (killing the client mid-trial stops the arm and loses the trial).
cd "$(dirname "$0")"
[ -f "mode_$1.env" ] || { echo "usage: $0 A|B"; exit 1; }
L=logs/slot_table_loop.log
last=$(grep -E "trial .* (running|ended)" "$L" | tail -n 1)
newer=$(grep -n "trial .* ended" "$L" | tail -n 1 | cut -d: -f1)
lastpol=$(grep -n "\[policy\]" "$L" | tail -n 1 | cut -d: -f1)
if [ -n "$last" ] && ! { echo "$last" | grep -q "ended" && [ "${lastpol:-0}" -lt "${newer:-0}" ]; }; then
  echo "NOT switched: a trial is in progress. Wait for it to end, then run this again."; exit 2
fi
pkill -f keep_running; pkill -f "python3 run.py"; sleep 2
echo "starting mode $1:"; cat "mode_$1.env" | grep -v "^#"
(nohup ./keep_running_cfg.sh "$1" >/dev/null 2>&1 &)
sleep 6; pgrep -f "python3 run.py" >/dev/null && echo "client running in mode $1" || echo "client between runs (the loop will start it)"
