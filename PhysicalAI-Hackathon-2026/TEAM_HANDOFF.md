# Team Overclocked: hand-off (28 Oct 2026)

Policy for the SIDCLab FANUC ER-4iA cell: the cameras find objects on the table (any position and angle), the arm
picks one up and puts it in the yellow bin, nothing hard-coded from `placement.yaml`. Cubes work; an AA battery
worked once end to end on 27 Oct (mode B). Deadline: submission 1 Nov, last real trials Saturday 31 Oct.

## 1. Set up (5 min)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # then open a new terminal
unzip Overclocked_code_2026-10-28.zip && cd PhysicalAI-Hackathon-2026
uv sync
uv run run.py --mock --trials 1 --trial-s 40          # must end with "Ready for the real robot"
```

Offline checks on saved real frames (no robot): `uv run test_vision.py`, and `uv run python compare_detectors.py`
(needs the detector tunnel, see section 5). Frames of the 27 Oct battery trial are in `debug_samples/`.

## 2. Secrets: do not put these in the zip, a chat or GitHub

- `.request.<id>.json` (the key to a request) and the results link `https://hackathon.sidclab.org/#key=...`.
  The key is NOT in this zip. Ask Saai for it over a private channel. Only ONE laptop runs the client at a time.
- The Hugging Face token on the Spark. Never print or paste it.

## 3. How the policy works (`policy.py`, `cube_vision.py`)

1. Arm goes to a look pose (220, -330, -150); the table camera (1920x1080, fixed) takes a picture.
2. Detect objects: open-vocabulary OWLv2 on the Spark (`spark/detect_server.py`) if the tunnel is up; otherwise the
   colour pipeline (`detect_cubes`) plus class-agnostic blob segmentation (`table_blobs`), which finds any object
   (the battery was found this way). Cosmos3-Nano (`spark/cosmos_server.py`) is only logged, not trusted.
3. Detect the yellow bin from the camera (`detect_bin`). **No bin in view = the policy refuses to pick** (it will not
   pick what it cannot place). Keep the bin visible, away from tape and yellow things.
4. Go over the object, look with the wrist camera, turn the jaws to the object (long axis for elongated objects),
   descend to z = -330 mm (UF0), close, lift, carry over the bin, release, look again, repeat until nothing is left.

Frames: UF0 mm, surface z about -335, z_UF0 = z_base - 330, `w,p,r = -180,0,R`, R = J1 + J6, jaws close along robot y
at R = 0.

## 4. Calibration findings (the part that cost us the most time)

- The wrist camera's calibration moved between days (gap wrist-vs-table across the jaws: 22 Oct -0.8 mm,
  26 Oct -3.7 mm, 27 Oct +12.5 mm). It turns with R. If another team swaps the gripper, expect it to change again.
- Two trial modes, in `mode_*.env` and `TRIAL_MODES`:
  - **A**: stay at the table camera's position; the wrist view only turns the jaws. Grasp offset -5.0 mm (gripper frame).
  - **B**: the wrist camera repositions the arm, with its measured shift (2.3, 13.4) mm taken out (`WRIST_SHIFT_B`).
  Mode B looked good on 27 Oct. The offset is set in `manual_offset.txt` (negative = further right, robot -y at R 0);
  edit it by hand without restarting. Adaptive search lives in `grasp_offset*.json`.
- The jaws are about 12 mm open on straight faces, 16 mm at the hook tips. Cube 10 mm: little margin. AA (14.5 mm): very
  little. Objects with a short side over about 26 mm are skipped.
- Grasp height is fixed at -330 mm: fine for cubes and lying batteries, not for tall or thin objects. The wrist depth
  sensor (`CAPTURE_DEPTH=1 CAPTURE_ONLY=1`) is meant to fix this, but the cell operator must switch it on first.

## 5. Running on the real cell

```bash
./keep_running_cfg.sh B        # the client in a loop with mode_B.env (mode_Bbat.env = battery, B)
./switch_mode.sh A|B|Bbat      # change mode; refuses while a trial is running
tail -f logs/slot_table_loop.log
```

**Never restart the client while a trial is running.** The arm stops dead and the trial is lost. `switch_mode.sh`
checks for this; if you kill it by hand, first check that the last log event is `trial ... ended` with no newer
`[policy]` line. Read the log with `grep -E "trial .* (running|ended)|trial mode|grasp probe|arrived|table blobs|pick any|no bin|holding"`.

Spark (needs the CSI VPN, which drops every 20-30 min): `ssh yt-overclocked728b@172.30.102.93`, tmux windows `cosmos`
and `detect`, then on the Mac
`ssh -f -N -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 yt-overclocked728b@172.30.102.93`. With the tunnel down the
policy falls back to the colour pipeline and still works.

Request/run status: request q013484 (approved, run r20261027-4a6ed5): 3 of 10 trials used on 27 Oct, `uv run run.py
--status`. Task text is overridden by `TARGET_OBJECT` in the mode file (`Bbat` -> battery).

## 6. Files

| File | What |
|---|---|
| `policy.py` | the policy (modes, offsets, battery/any-object logic) |
| `cube_vision.py` | colour cubes, wrist locate, bin, blobs, open-vocabulary glue |
| `detect_client.py`, `cosmos_client.py`, `spark/*.py` | clients and servers for OWLv2 and Cosmos on the Spark |
| `mode_*.env`, `keep_running*.sh`, `switch_mode.sh` | modes and the keep-alive loop |
| `manual_offset.txt`, `grasp_offset*.json` | grasp offset by hand / learned |
| `depth_sim.py` | ray-traced simulator of the real cameras (wrist depth, bias checks) |
| `sim/isaac_cell.py`, `sim/deploy.sh` | an Isaac Sim stand-in for the cell with several objects. Written but NEVER RUN; expect fixes. Not needed for the real trials |
| `policy_battery_experiments.py` | older experiments, reference only |
| `submission/` | `submission.yaml` (battery task) and `placement.yaml`; `*.cube.yaml` are the cube versions |

## 7. What to do next

1. Real trials with the battery and then 2-3 objects at once (the pick-place-look loop has only run with one object
   on the real cell). Vary positions and angles. Keep the bin in view.
2. Decide A or B as the default once there are 4-5 repeats, and make the offset permanent.
3. Wrist depth for objects of other heights, once the operator enables it.
4. Put the code on GitHub (below) before 1 Nov, and re-check `uv run run.py --mock` still passes.

## 8. GitHub

This folder is a clone of the organizers' public repo (`origin` = SIDCLab/PhysicalAI-Hackathon-2026), and the zip does
not include `.git`. Do not push to `origin` (it is theirs, and it is public). Either fork it on GitHub and push there, or
create a private team repo: `git init && git add -A && git commit`. The `.gitignore` here already excludes `.venv`,
`debug/`, `logs/` and `.request*.json`. Ask the organizers on Slack whether they want a link to a repo or a zip for
the final submission.
