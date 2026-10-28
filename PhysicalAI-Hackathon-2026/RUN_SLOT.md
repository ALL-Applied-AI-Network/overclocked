# Running the Overclocked slot (request q9dceeb)

Only ONE laptop runs the client during the slot. Keep `.request.json` private: it is the key to this request.

## Before the slot (once, ~5 min)

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (Mac/Linux), then open a new terminal.
2. Unzip this folder and go into it:
   `cd PhysicalAI-Hackathon-2026`
3. Install and self-check (no robot, ~1 min, must end with "✓ Ready for the real robot"):
   `uv sync`
   `uv run run.py --mock --trials 1 --trial-s 40`
4. Check the request: `uv run run.py --status`
   `pending` = waiting for the organizers; `approved` = good to go.

## During the slot

1. Be online ~5 min before the slot starts. The robot is reached over the internet (hackathon.sidclab.org); no VPN needed.
2. Run, and leave the terminal open until it says "all trials finished":
   `uv run run.py`
   It waits for approval and for the operator by itself. You do nothing between rollouts.
3. Watch the `[policy]` lines. Each rollout should show:
   `saw N cube(s)` -> `pick ... -> R ...` -> (the arm moves) -> `nothing left to do ... holding`
4. If something goes wrong (error line, arm faults), take a screenshot of the terminal and tell the operator.
   Ctrl-C stops the client; running `uv run run.py` again rejoins.

## After the slot

1. Results link (scores and videos, private to the team): `uv run run.py --status`
2. Zip the `debug/` folder (what the camera saw on each look) and send it to Saai, together with the terminal output.
   These are what we use to calibrate the next run.
3. Optional dataset download: `uv run run.py --download`

## What this run is for

It is a calibration run: one 10 mm cube at (420, -100) mm, yawed 30 deg, and the yellow bin at (510, 110).
Successes are a bonus; the debug frames and logs tell us the real grasp height, jaw angle and colour thresholds.
