"""Your policy. Edit this file.

act(obs) returns the arm's next target, or a list of targets to play in order. When they have been played, act is
called again with a new observation. act has no time limit: the arm holds still while it runs.

This policy: vision-guided pick and place of 10 mm cubes, in "tcp" mode.

  1. Park the arm out of the table camera's view.
  2. Look: full-resolution table frame -> cube_vision.detect_cubes -> x, y, yaw of each cube (UF0 mm).
     If nothing is found, fall back to the cube's position in submission/placement.yaml.
  3. Pick the next cube: above it, wrist turned to its yaw, open; descend; close; lift.
  4. Place it: TASK "bin" drops it in the bin, TASK "stack" sets it on the stack at its level.
  5. Back to 1 until there is nothing left to do, then hold still.

Every target is clipped to the cell's limits before it is sent. Constants marked CALIBRATE are what the first real
slot should check (see the log lines and the debug frames written to debug/).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from calibration.geometry import SURFACE_Z, flange_pose, load, project
from cosmos_client import understand as cosmos_understand
from detect_client import find as find_objects
import cube_vision
import verifier_client
from cube_vision import MAX_GRASP_SHORT_MM, blob_colour, table_blobs, detect_bin, detect_cubes, objects_from_detections, draw, square_yaw, wrist_locate, wrist_locate_bin, wrist_locate_depth

HERE = Path(__file__).resolve().parent

# --- task -------------------------------------------------------------------------------------------------------
TASK = "bin"                                  # "bin": every cube into the bin; "stack": build STACK_ORDER
COLOURS = ("yellow", "blue", "red")           # which cube colours to look for
STACK_ORDER = ["yellow", "yellow", "blue", "blue", "red", "red"]  # bottom to top
STACK_XY = (450.0, -150.0)                    # mm, UF0; where the stack is built
BIN_XY = (510.0, 110.0)                       # mm; the default scene's bin. placement.yaml's "bin" overrides it
# Where the fingers go relative to the wrist-corrected cube position (mm, robot frame). The 26 Oct probes: -3 mm in y
# grasped (trials 5, 6; x of +/-3 made no difference), +3 mm in y landed the finger on the cube's top (trials 4, 7),
# and so did 0 (trials 1-3): the wrist camera's centre sits ~3 mm to +y of the fingers. GRASP_PROBES cycles it per rollout.
GRASP_PROBES = [(0.0, float(v)) for v in os.environ["GRASP_OFFSETS"].split(",")] if os.environ.get("GRASP_OFFSETS") else [(0, -3)]
# GRASP_OFFSETS="-3,-4.5,-1.5" uv run run.py: one across-jaw offset (mm, robot y) per rollout, by rollout number; default -3 for all.
CAPTURE_ONLY = os.environ.get("CAPTURE_ONLY") == "1"  # with CAPTURE_DEPTH=1: hover over the cube, save colour + depth, do not descend
BIN_CLEAR_MM = 40.0                           # a cube this close to the bin's centre is already in it (or on its rim)
# Which detector finds the cubes: "open_vocab" (OWLv2 on the Spark, text queries) or "colour" (colour gates + silhouette
# fit, the version that grasped on 26 Oct). Switch without editing: CUBE_DETECTOR=colour uv run run.py
DETECTOR = os.environ.get("CUBE_DETECTOR", "open_vocab")
# TARGET_OBJECT=battery: treat every instruction as "Put the battery into the yellow bin." (the request's own text says "cube"); the objects
# the detector cannot name are found as "anything on the table that is not the bin" (BLOB_FALLBACK), ranked by how well each looks like the name.
OBJECT_OVERRIDE = os.environ.get("TARGET_OBJECT", "").strip()
BLOB_FALLBACK = True
DETECTOR_FALLBACK_IF_UNREACHABLE = True       # open_vocab only: a dead tunnel falls back to the colour pipeline, nothing else does
OPEN_VOCAB = True                             # colour mode only: the open-vocabulary detector as a second chance on the first look
OPEN_VOCAB_MIN_SCORE = 0.18                   # its confidence to count as a cube (real cubes scored 0.23-0.64 for "a small blue cube"; nothing else >0.1 in reach)
COSMOS_TIMEOUT = 15.0                         # s to wait for Cosmos on the Spark before carrying on without it
FALLBACK_TO_PLACEMENT = False                 # no cube seen: use placement.yaml's cube position (once per rollout)

# --- geometry (mm, UF0: z = base z - 330) -------------------------------------------------------------------------
SURFACE = SURFACE_Z * 1000 - 330.0            # the work surface, about -335.1
CUBE = 10.0                                   # cube edge
TIP_CLEARANCE = 5.0                           # CALIBRATE: fingertips above the mat when gripping (twin: 2.5 mm;
                                              # slot 1: rollouts ended as the tips reached 2.5 mm)
R_OFFSET = 0.0                                # CALIBRATE: wrist R that puts the jaws across a yaw-0 cube's faces
APPROACH = 60.0                               # hover this far above the cube before descending
LIFT = 80.0                                   # lift this far after closing
BIN_TOP = 55.0                                # bin wall height above the surface
DROP_ABOVE_BIN = 8.0                          # release this far above the bin wall (35 mm bounced the cube out of the bin)
PARK = (220.0, -330.0, -150.0)                # out of the table camera's view of the work area
DOWN = (-180.0, 0.0)                          # w, p: fingers straight down

# --- motion -------------------------------------------------------------------------------------------------------
DT = 0.1                                      # seconds per action row (the cell's step)
SPEED = 60.0                                  # mm/s in free space (the real arm trailed far behind at 120)
SLOW = 10.0                                   # mm/s for the last approach, the grasp and the release
TURN = 60.0                                   # deg/s for the wrist
GRIP_WAIT = 1.2                               # s to hold still after closing / opening (the arm lags ~0.8 s)
LOOK_SETTLE = 3.0                             # s at PARK before looking; at 0.5 s the gripper was still in view
BIN_SERVO = os.environ.get("BIN_SERVO", "0") == "1"     # before the release, the wrist camera re-centres the jaws over the bin
BIN_VIEW_Z = SURFACE + 160.0                  # hover height for that view: the camera looks ahead of the fingertips, and lower the near half of the bin is out of frame
LONG_PEEK_EXTRA = 0.0                                   # mm higher for the wrist view of a long object (a battery is 50 mm long)
# Grasp check after the lift: what the wrist depth sees between the fingers. VERIFY saves the frames and a verdict for every attempt (the training
# data for a learned verifier, labelled at the next table look); VERIFY_GATE makes a verdict of "empty" skip the carry and look again instead.
VERIFY = os.environ.get("VERIFY", "1") != "0"
VERIFY_GATE = os.environ.get("VERIFY_GATE", "0") == "1"
VERIFY_DIR = HERE / "verifier_data"
# A sweep: each trial uses the next across-jaw offset of this list (same cube, same spot, no randomness), to find the window by experiment.
SWEEP = [float(v) for v in os.environ.get("EXPLORE_SWEEP", "").replace(",", " ").split()]
SWEEP_FROM = int(os.environ.get("SWEEP_FROM", "0"))  # (no longer used: the sweep position is counted in sweep_state.txt)
SWEEP_STATE = HERE / "sweep_state.txt"  # the index of the next sweep value; delete it to start the sweep over
EXPLORE_BAD = float(os.environ.get("EXPLORE_BAD", "0"))  # share of attempts given a deliberately bad across-jaw offset (+17..20 mm, always to the + side: the gap centre sits ~9 mm from the tool point, so on the - side the near finger lands on the object, as it did on 30 Oct at -24.5 and -24.7): empty lifts for the verifier
EXPLORE_MM = float(os.environ.get("EXPLORE_MM", "0"))  # data collection: add a random +-EXPLORE_MM to the across-jaw offset of each grasp, so the
                                                      # recorded attempts cover a range of offsets (a model of success against offset needs that)
VERIFY_HELD_PX = 120      # pixels this much nearer than the fingertip's own distance = something held
VERIFY_UNSURE_PX = 40
SIDE_CROP_DV = int(os.environ.get("SIDE_CROP_DV", "0"))  # px (640-wide frame) the verifier's crop is moved, negative = up: the side camera's projection drifted by day (1 Nov: the fingertip sat ~14 px above the crop's centre, held 4/9 -> 9/9 at -14)
# The lifted wrist camera cannot see between the fingers, but it looks at the table: a cube of the target's colour still near the grasp spot means the
# grasp missed. 0 = off. 1 Nov: 9 held lifts showed none within 60 mm, the one empty grasp showed it 49 mm away (radius picked on those frames: validate on new lifts).
WRIST_STILL_MM = float(os.environ.get("WRIST_STILL_MM", "0"))
SIDE_HELD_P = 0.5         # the side-camera verifier's cut-off on P(held) (held-out: 50/56 held kept, 1/10 real empties passed at 0.5)
# Release bias on the bin point: 0. A (-24, +8) correction tried on 30 Oct put the cube at x = 260, 86 mm outside the bin: the cubes had come to rest against the
# near wall, which is not the pocket's centre. The table camera's bin point itself was close.
_bb = [float(v) for v in os.environ.get("BIN_DROP_BIAS", "0,0").replace(",", " ").split()]
BIN_DROP_BIAS = (_bb[0], _bb[1])  # mm added to the bin point for the release only (fitted for this bin position: check landings.jsonl if the bin is moved)
DROP_COMPENSATE = os.environ.get("DROP_COMPENSATE", "1") != "0"  # release with the held object (not the fingertip) over the bin's centre
BIN_FIX_MM = 3.0                                        # re-hover over the bin when the wrist view moves it more than this
BIN_FIXES = 3
PEEK_SETTLE = 1.5                             # s hovering over the cube before the wrist frame is saved
PEEK_SIZE = (1280, 720)                       # the wrist camera's full size
TABLE_XY_OFFSET = (0.0, 10.0)                 # mm added to table-camera positions: the wrist camera and the
                                              # operator's placements both put cubes ~10 mm further +y (slot 1)
WRIST_FIX_MM = 1.5                            # re-hover when the wrist view moves the cube more than this
WRIST_FIXES = 4                               # at most this many re-hovers per cube

# --- self-calibration (calib_learn.py): the table camera's position error against the wrist camera, learned from our own trials ---
# Every first wrist view of a cube is logged next to the table estimate (calib_pairs.jsonl). calib_learn.py fits a bounded linear correction to
# calibration_model.json; CALIB_APPLY=1 applies it to the table estimate before the first approach. Only that file changes: no motion, limit or gate does.
CALIB_APPLY = os.environ.get("CALIB_APPLY", "0") == "1"
CALIB_FILE = HERE / "calibration_model.json"
CALIB_LOG = HERE / "verifier_data" / "calib_pairs.jsonl"
CALIB_MAX_MM = 20.0                           # the correction never exceeds this, whatever the file says


def load_calib() -> dict | None:
    try:
        return json.loads(CALIB_FILE.read_text())
    except Exception:  # noqa: BLE001
        return None


def calib_correction(model: dict | None, x: float, y: float) -> tuple[float, float]:
    """(dx, dy) mm to add to a table position (offset already in): linear in (x, y) / 100, zero outside the positions it was fitted on."""
    if not model:
        return 0.0, 0.0
    x0, x1, y0, y1 = model["bounds"]
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return 0.0, 0.0
    f = np.array([1.0, x / 100.0, y / 100.0])
    d = f @ np.asarray(model["W"], float)
    d = np.clip(d, -CALIB_MAX_MM, CALIB_MAX_MM)
    return float(d[0]), float(d[1])
ARRIVE_MM = 3.0                               # the real fingertip must be this close to a waypoint before going on
ARRIVE_TIMEOUT = 15.0                         # s after the rows played; then go on anyway (logged)

# --- the cell's limits (mock_cell.TCP_BOX, the ~550 mm reach) ---------------------------------------------------
X_LIM, Y_LIM, Z_LIM = (150.0, 650.0), (-450.0, 450.0), (-333.0, 300.0)
REACH = 540.0
MAX_CHUNK = 100


def colour_from_instruction(text) -> str | None:
    """The cube colour the instruction asks for ("pick the red cube"); "the yellow bin" is not a cube."""
    m = re.search(r"\b(" + "|".join(COLOURS) + r")\s+cubes?\b", str(text or "").lower())
    return m.group(1) if m else None


def object_phrase(text) -> str:
    """What the instruction asks to move ("Put the 10 mm screw into the bin" -> "screw"); "cube" if it does not say."""
    m = re.search(r"\b(?:pick up|pick|put|place|grab|move|take)\s+(?:the|a|an)\s+(.+?)\s+(?:and|then|into|in|onto|on|to|inside)\b",
                  str(text or "").lower())
    phrase = re.sub(r"\b\d+(?:\.\d+)?\s*mm\b", "", m.group(1)).strip() if m else ""
    return phrase or "cube"


TARGET_FILE = Path(__file__).resolve().parent / "target_object.txt"
GENERIC_PHRASES = ("object", "thing", "item", "part", "anything", "everything")
# what each named object looks like to the table camera when the detector cannot name it: (long mm, short mm, long/short)
SHAPES = {"battery": ((34, 72), (9, 30), 1.9), "cube": ((6, 18), (6, 18), 0.0),
          "screw": ((50, 100), (22, 55), 1.2),
          "bolt": ((14, 48), (6, 24), 1.3),
          "key": ((40, 260), (3, 90), 1.0)}  # an L-shaped hex key: the blob is its bounding shape, long arm ~100-200 mm  # the short screw/bolt ~10 mm across and ~20 mm long (with its shadow the blob reads ~25 x 15)  # the big grey printed screw: ~65 x 46 mm with its head and shadow
THIN_MM = 20.0  # a table blob this narrow (shadow included) is too thin for the wrist depth to see
BIG_PHRASES = ("screw", "key")  # blobs this large are looked for, and not held back by the jaw-width gate (the shaft is tried anyway)
TRIED_MM, TRIED_MAX = 15.0, 2   # a spot picked this often and still occupied is left alone


def read_target_object() -> str | None:
    """target_object.txt (one word: battery, cube, any ...) chooses what to pick for the next trial, without a new request."""
    try:
        text = TARGET_FILE.read_text().strip().lower()
    except OSError:
        return None
    return {"any": "object", "all": "object", "everything": "object"}.get(text, text) or None


def shape_matches(blob: dict, phrase: str) -> bool:
    if phrase not in SHAPES:
        return True
    (l0, l1), (s0, s1), aspect = SHAPES[phrase]
    long_mm, short_mm = float(blob.get("long_mm", 0.0)), float(blob.get("short_mm", 0.0))
    return l0 <= long_mm <= l1 and s0 <= short_mm <= s1 and long_mm / max(short_mm, 1e-6) >= aspect


def names_a_colour(text) -> bool:
    """A cube colour is asked for in words the regex does not parse ("pick the red one"), not "the yellow bin"."""
    return bool(re.search(r"\b(" + "|".join(COLOURS) + r")\b(?!\s+(bin|box|container|tray))", str(text or "").lower()))


def clip_target(row):
    """Keep one [x, y, z, w, p, r, g] row inside the box and the reach circle."""
    x, y, z, w, p, r, g = row
    x, y = float(np.clip(x, *X_LIM)), float(np.clip(y, *Y_LIM))
    rad = math.hypot(x, y)
    if rad > REACH:
        x, y = x * REACH / rad, y * REACH / rad
    return [x, y, float(np.clip(z, *Z_LIM)), float(w), float(p), float(r), float(g)]


def tip_from_joints(joints) -> tuple[float, float, float]:
    """Fingertip position (UF0 mm) from J1..J6 with the cell's FK: 293 mm along the flange's tool axis."""
    T = flange_pose(joints)
    tip = T[:3, 3] + T[:3, 0] * 0.293
    return float(tip[0] * 1000), float(tip[1] * 1000), float(tip[2] * 1000 - 330.0)


def wrist_r(yaw_deg: float, current_r: float = 0.0, elongated: bool = False) -> float:
    """The R that aligns the jaws with a square of this yaw, the 90-degree twin nearest the current R. An elongated object (yaw = its long
    axis) has one right way round, 180 degrees apart: the jaws close at 90 degrees to R, so R = the long axis puts them across the narrow side."""
    if elongated:
        base = (yaw_deg + R_OFFSET + 90.0) % 180.0 - 90.0
        return min((base + k * 180.0 for k in range(-1, 2)), key=lambda r: (abs(r - current_r), abs(r)))
    base = square_yaw(yaw_deg + R_OFFSET)
    return min((base + k * 90.0 for k in range(-3, 4)), key=lambda r: (abs(r - current_r), abs(r)))


# ADAPTIVE_OFFSET=1: find the across-jaw offset by itself. The grippers on this cell are swapped in and out by other teams, so the
# fingertip's position relative to the camera is not a constant. A rollout that ends while still going down on the cube (phase "grasp")
# is a miss at that offset: the next rollout tries the next candidate. One that reaches pick and drop is a hit: that offset is kept.
# The state survives restarts and slots in grasp_offset.json (delete it to start the search again).
# SKIP_WRIST_XY=1: no repositioning from the wrist camera. The arm goes to the table camera's position and stays; the wrist view only turns the
# jaws. 27 Oct: the wrist camera put the cube 12.7 mm (+-1.5) further +y than the table camera did (26 Oct: -3.9 +-4.7), and the search for the
# working offset ended at -12, i.e. back at the table camera's first position. Run with GRASP_FRAME=gripper: the offsets are in the gripper's frame.
SKIP_WRIST_XY = os.environ.get("SKIP_WRIST_XY") == "1"
GRASP_FRAME = os.environ.get("GRASP_FRAME", "robot")  # "gripper": the grasp offset turns with the wrist angle
ADAPTIVE_OFFSET = os.environ.get("ADAPTIVE_OFFSET") == "1"
# robot y, mm, in the order they are tried. 27 Oct, one cube, the operator's reports: -3 the right finger touches the cube's END, -4.5 more of it,
# -6 the finger's hook on its edge (photo), -9 the whole right finger almost on the cube. The right finger sweeps over the cube on its way: the
# cube enters the gap only after it has passed, so the target is further the SAME way. Photo + reports estimate it near -13 (+/-3).
OFFSET_CANDIDATES = [-3.0, -4.5, -6.0, -9.0, -12.0, -13.5, -10.5, -15.0, -16.5, -18.0, -1.5, 0.0, 1.5]
OFFSET_STATE = HERE / "grasp_offset.json"
if SKIP_WRIST_XY:  # offsets are now corrections to the table position: small, both ways round zero
    # 27 Oct: offset 0 (the table position) still put the finger on the cube; the operator says the gripper needs to go further right (-y)
    OFFSET_CANDIDATES = [0.0, -1.5, -3.0, -4.5, -6.0, -7.5, 1.5, 3.0, -9.0]
    OFFSET_STATE = HERE / "grasp_offset_table.json"


MANUAL_OFFSET_FILE = HERE / "manual_offset.txt"
# When the target was located from the wrist DEPTH (a raised blob: no shadow, a shiny battery seen too) the offset is across the jaws from the
# blob's true centre: manual_offset_depth.txt, default DEPTH_ACROSS_MM. -7.0 is where the one success with depth (28 Oct, R +5) sat, 28 Oct.
MANUAL_OFFSET_DEPTH_FILE = HERE / "manual_offset_depth.txt"
DEPTH_ACROSS_MM = -7.0


def read_manual_offset(depth: bool = False) -> tuple[float, float] | None:
    """manual_offset.txt: "across" (mm) or "along, across" (mm, in the frame GRASP_FRAME says). Negative across = further right (robot -y at
    R 0). Missing, empty or unreadable = no manual offset."""
    try:
        vals = [float(v) for v in (MANUAL_OFFSET_DEPTH_FILE if depth else MANUAL_OFFSET_FILE).read_text().replace(",", " ").split()]
    except Exception:  # noqa: BLE001
        return (0.0, DEPTH_ACROSS_MM) if depth else None
    if len(vals) == 1:
        return 0.0, vals[0]
    return (vals[0], vals[1]) if len(vals) >= 2 else None


def load_offset_state() -> dict:
    try:
        return json.loads(OFFSET_STATE.read_text())
    except Exception:  # noqa: BLE001
        return {"offset": OFFSET_CANDIDATES[0], "failed": [], "hits": 0}


def save_offset_state(state: dict) -> None:
    try:
        OFFSET_STATE.write_text(json.dumps(state, indent=1))
    except Exception:  # noqa: BLE001  (never stop the arm over a state file)
        pass


def learn_offset(state: dict, descended_offset: float | None, reached_pick: bool) -> str | None:
    """Update state after a rollout; returns a log line, or None if the rollout says nothing about the offset."""
    if descended_offset is None:
        return None
    if reached_pick:
        state.update(offset=descended_offset, failed=[], hits=state.get("hits", 0) + 1)
        return f"offset {descended_offset:+.1f} mm worked (hit {state['hits']}): keeping it"
    state["failed"] = sorted(set(state.get("failed", [])) | {descended_offset})
    left = [c for c in OFFSET_CANDIDATES if c not in state["failed"]]
    if not left:  # everything missed: the gripper may have changed again, start over
        state["failed"], left = [], OFFSET_CANDIDATES
    state["offset"] = left[0]
    return f"offset {descended_offset:+.1f} mm missed (failed so far: {state['failed']}): next {state['offset']:+.1f} mm"


# TRIAL_MODES="A,B": the mode of each trial, by the cell's trial number, cycling (trial 0 -> A, 1 -> B, 2 -> A ...). A: the arm stays at the table
# camera's position and the wrist camera only turns the jaws. B: the wrist camera repositions the arm, with its 27 Oct calibration shift removed.
# Both use manual_offset.txt if it exists. Unset: every trial uses the environment's SKIP_WRIST_XY / WRIST_SHIFT as before.
TRIAL_MODES = os.environ.get("TRIAL_MODES", "").replace(",", " ").upper().split()
_wsb = [float(v) for v in os.environ.get("WRIST_SHIFT_B", "2.3,13.4").replace(",", " ").split()]
WRIST_SHIFT_B = tuple(_wsb[:2])  # the wrist camera's calibration shift taken out in mode B; "0,0" = none (it was gone on 28 Oct)


def apply_trial_mode(rollout) -> str | None:
    """Set this trial's mode; returns "A" or "B" (None if TRIAL_MODES is unset)."""
    global SKIP_WRIST_XY
    if not TRIAL_MODES:
        return None
    mode = TRIAL_MODES[(rollout or 0) % len(TRIAL_MODES)]
    SKIP_WRIST_XY = mode == "A"
    cube_vision.WRIST_SHIFT = None if mode == "A" else WRIST_SHIFT_B
    return mode


class Policy:
    CAMERAS = ["table", "wrist"] + (["side"] if os.environ.get("SIDE") == "1" else [])  # table: finding cubes; wrist: close-up refinement; side: the fingertip crop for the grasp check
    IMAGE_SIZE = (640, 360)               # (width, height) of the images you get; act switches it per phase below
    IMAGE_FORMAT = "png"                  # or "jpeg": about 7x smaller, for a slow connection
    ACTION = "tcp"                        # what act returns: "joints" or "tcp"
    PREFETCH = False                      # whole moves are planned at once; nothing to gain from prefetching
    DEPTH = ["table", "wrist"] if os.environ.get("CAPTURE_DEPTH") == "1" else []  # ["table"] and/or ["wrist"]: also those cameras' depth, in obs["depth"]
    LOOK_SIZE = (1920, 1080)              # a 10 mm cube is ~15 px wide at full size, ~5 px at 640 x 360

    def __init__(self, checkpoint: str | None = None):
        self.offset_state = load_offset_state()
        self.cams = load()
        self.placement = self._load_placement()
        self.debug_dir = HERE / "debug"
        self.reset()

    # -- bookkeeping ------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Called at the start of every rollout."""
        if ADAPTIVE_OFFSET:
            msg = learn_offset(self.offset_state, getattr(self, "descended_offset", None), getattr(self, "phase", "start") != "grasp") \
                if hasattr(self, "offset_state") else None
            if msg:
                save_offset_state(self.offset_state)
                self.log("adaptive offset: " + msg)
        self.descended_offset = None  # set when the fingers go down on a cube this rollout
        self.queue: list[list[float]] = []
        self.phase = "start"
        self.pose = None              # the last commanded [x, y, z, w, p, r, g]
        self.placed = 0               # cubes put away this rollout
        self.attempt = 0              # grasps tried this rollout (for the verifier's frames)
        self.pending = None           # the last attempt, labelled at the next table look
        self._raw_cubes = []
        self.tried: list = []         # table positions picked so far this rollout (with the table offset applied)
        self.used_fallback = False
        self.want = None
        self.bin_seen = None
        self.instruction = None
        self.cosmos_done = False
        self.rollout = None
        self.arrive = None            # (x, y, z, since): a waypoint the real arm must reach before the next phase
        self.IMAGE_SIZE = Policy.IMAGE_SIZE

    @staticmethod
    def _load_placement() -> list[dict]:
        path = HERE / "submission" / "placement.yaml"
        if not path.exists():
            return []
        return (yaml.safe_load(path.read_text()) or {}).get("rollouts") or []

    def _scene(self) -> dict:
        """This rollout's placement.yaml entry (or the only one), {} if none."""
        if not self.placement:
            return {}
        idx = self.rollout if isinstance(self.rollout, int) else 0
        return self.placement[min(idx, len(self.placement) - 1)].get("objects", {}) or {}

    def log(self, msg: str) -> None:
        print(f"[policy] {msg}", flush=True)

    # -- motion -----------------------------------------------------------------------------------------------------
    def move(self, x, y, z, r=None, g=None, speed=SPEED) -> None:
        """Append rows that go in a straight line from the last target to (x, y, z, r), at speed mm/s.
        A new gripper value goes on the last row only: the cell switches it once the arm reaches that row."""
        p0 = self.pose
        r = p0[5] if r is None else r
        g = p0[6] if g is None else g
        dist = math.dist(p0[:3], (x, y, z))
        n = max(1, math.ceil(dist / (speed * DT)), math.ceil(abs(r - p0[5]) / (TURN * DT)))
        for k in range(1, n):
            a = k / n
            self.queue.append(clip_target([p0[0] + a * (x - p0[0]), p0[1] + a * (y - p0[1]), p0[2] + a * (z - p0[2]),
                                           DOWN[0], DOWN[1], p0[5] + a * (r - p0[5]), p0[6]]))
        self.pose = clip_target([x, y, z, DOWN[0], DOWN[1], r, g])
        self.queue.append(list(self.pose))

    def wait(self, seconds: float) -> None:
        self.queue.extend(list(self.pose) for _ in range(max(1, round(seconds / DT))))

    def _r(self, yaw) -> float:
        """The wrist R for this yaw, for the current target's kind (cube-like or elongated)."""
        return wrist_r(yaw, self.pose[5], bool((self.target or {}).get("elongated")))

    def hover(self, x, y, yaw) -> None:
        """Above the cube, jaws turned to its yaw, open; then hold still for the wrist camera."""
        up = LONG_PEEK_EXTRA if (self.target or {}).get("elongated") else 0.0  # the camera looks ahead of the fingertips: a long object is cut off at its edge from low down
        self.move(x, y, SURFACE + CUBE + APPROACH + up, r=self._r(yaw), g=0.0)
        self.wait(PEEK_SETTLE)

    def grasp_xy(self) -> tuple[float, float]:
        """Where the fingers go down: the corrected cube position plus this rollout's probe offset."""
        dx, dy = GRASP_PROBES[(self.rollout or 0) % len(GRASP_PROBES)]
        if ADAPTIVE_OFFSET:
            dx, dy = 0.0, float(self.offset_state["offset"])
            self.descended_offset = dy
        manual = read_manual_offset(depth=self.target.get("located_by") == "depth")  # manual_offset.txt, read at every grasp: edit it between trials, no restart; it wins over everything above
        if manual is not None:
            dx, dy = manual
            self.descended_offset = None  # a hand-set offset is not the adaptive search's business
        sweep_dy = self.target.get("sweep_dy")
        if SWEEP and sweep_dy is None:
            # the next value of the sweep, counted per attempt in a file: a trial the cell re-runs must not repeat a value
            try:
                idx = int(SWEEP_STATE.read_text().strip())
            except Exception:  # noqa: BLE001
                idx = 0
            if idx < len(SWEEP):
                sweep_dy = self.target["sweep_dy"] = SWEEP[idx]
                SWEEP_STATE.write_text(str(idx + 1))
        if SWEEP and sweep_dy is not None:
            dx, dy = 0.0, sweep_dy
            self.target["explore_bad"] = False
        elif EXPLORE_MM > 0 or EXPLORE_BAD > 0:
            if "explore_dy" not in self.target:  # drawn once per attempt: the descent and the pick agree
                bad = random.random() < EXPLORE_BAD
                self.target["explore_dy"] = random.uniform(17.0, 20.0) if bad else random.uniform(-EXPLORE_MM, EXPLORE_MM)
                self.target["explore_bad"] = bad
            dy += self.target["explore_dy"]
        self.last_offset = (float(dx), float(dy))  # in the GRASP_FRAME, before it is turned into the robot's frame
        if GRASP_FRAME == "gripper":  # the offset is across the jaws: it turns with the wrist (at R 0 it is the robot's y)
            r = math.radians(self.pose[5])
            dx, dy = dx * math.cos(r) - dy * math.sin(r), dx * math.sin(r) + dy * math.cos(r)
        self.last_offset_xy = (float(dx), float(dy))  # the offset in the robot's frame: the held object sits this far from the fingertip
        x, y = self.target["xy_mm"]
        if dx or dy:
            self.log(f"grasp probe: {dx:+.1f}, {dy:+.1f} mm from the cube in x, y ({GRASP_FRAME} frame, R {self.pose[5]:+.0f}, rollout {self.rollout})")
        return x + dx, y + dy

    def descend(self, x, y) -> None:
        """Straight down to grasp height, jaws open; the caller waits for the real arm to arrive."""
        self.move(x, y, self.pose[2], speed=SPEED / 2)  # first sideways, at hover height, well above the object; then straight down (28-30 Oct: the diagonal descent swept a finger across it)
        self.move(x, y, SURFACE + CUBE + 10.0, speed=SPEED / 2)
        self.move(x, y, SURFACE + TIP_CLEARANCE, speed=SLOW)

    def pick(self, x, y, yaw) -> None:
        """Close, then lift (the arm is already at grasp height)."""
        top = SURFACE + CUBE
        grip = SURFACE + TIP_CLEARANCE
        self.wait(0.3)
        self.move(x, y, grip, g=1.0)
        self.wait(GRIP_WAIT)
        self.move(x, y, grip + 15.0, speed=SLOW)
        self.move(x, y, top + LIFT)

    def place(self, level: int) -> None:
        if TASK == "stack":
            x, y = STACK_XY
            r = wrist_r(0.0, self.pose[5])
            seat = SURFACE + level * CUBE + TIP_CLEARANCE  # this cube's bottom sits on the one below
            self.move(x, y, seat + CUBE + APPROACH, r=r)
            self.move(x, y, seat + 8.0, speed=SPEED / 2)
            self.move(x, y, seat + 0.5, speed=SLOW)
            self.wait(0.5)
            self.move(x, y, seat + 0.5, g=0.0)
            self.wait(GRIP_WAIT)
            self.move(x, y, seat + CUBE + APPROACH, speed=SLOW)
        else:
            bx, by = self.bin_seen  # look() only hands out a cube once the camera has seen the bin
            # the object is held where the fingers closed on it: the grasp offset from the fingertip. Put the OBJECT over the bin's centre
            ox, oy = getattr(self, "last_offset_xy", (0.0, 0.0)) if DROP_COMPENSATE else (0.0, 0.0)
            bx, by = bx + ox + BIN_DROP_BIAS[0], by + oy + BIN_DROP_BIAS[1]
            release = SURFACE + BIN_TOP + DROP_ABOVE_BIN
            r_rel = None
            if (self.target or {}).get("elongated"):
                # a battery (50 mm) is as long as the bin's pocket is wide (48): drop it along the pocket's diagonal (68 mm) so it fits in
                r_rel = wrist_r(cube_vision.LAST_BIN_YAW[0] + 45.0, self.pose[5], True)
                self.log(f"release: turning the long object to the bin's diagonal (bin {cube_vision.LAST_BIN_YAW[0]:+.0f} deg -> R {r_rel:+.0f})")
            self.move(bx, by, release + 20.0, r=r_rel)
            self.move(bx, by, release, speed=SPEED / 2)
            self.wait(0.5)  # steady over the bin before the jaws open
            self.move(bx, by, release, g=0.0)
            self.wait(GRIP_WAIT)
            self.move(bx, by, release + 40.0)

    # -- perception -------------------------------------------------------------------------------------------------
    def look(self, obs) -> dict | None:
        """The next cube to pick. DETECTOR "open_vocab": found by the open-vocabulary detector on the Spark; "colour":
        the colour pipeline (the version that grasped on 26 Oct, kept as it was). The bin always comes from the camera."""
        img = obs["images"].get("table")
        cubes = []
        if img is not None and tuple(img.shape[1::-1]) == tuple(self.LOOK_SIZE):
            self._find_bin(img)  # first: a cube at the bin is told apart from a cube to pick
            if DETECTOR == "open_vocab":
                found = self._open_vocab(img)
                if found is None and DETECTOR_FALLBACK_IF_UNREACHABLE:  # only a dead tunnel, never "it found nothing"
                    self.log("open-vocabulary detector unreachable: using the colour pipeline for this look")
                    found = self._colour_cubes(img) if object_phrase(self.instruction) in ("cube",) + GENERIC_PHRASES else []
                cubes = found or []
            else:
                cubes = self._colour_cubes(img)
            self._save_debug(img, cubes)
            self._save_table_depth(obs)
            self._ask_cosmos(img, cubes)
        if TASK == "stack":
            # skip what is already stacked, and pick the colour the next level needs
            cubes = [c for c in cubes if math.dist(c["xy_mm"], STACK_XY) > 20.0]
            if self.placed >= len(STACK_ORDER):
                return None
            cubes = [c for c in cubes if c["colour"] == STACK_ORDER[self.placed]]
        if TASK == "bin" and self.bin_seen is None:
            self.log("no bin seen by the camera: not picking a cube it cannot place")
            return None
        if TASK == "bin" and self.bin_seen is not None:
            for c in cubes:  # an object resting near the bin: where it landed relative to the point the policy aimed at (log + landings.jsonl)
                d = (c["xy_mm"][0] - self.bin_seen[0], c["xy_mm"][1] - self.bin_seen[1])
                if math.hypot(*d) <= BIN_CLEAR_MM and self.placed > 0:
                    self.log(f"landed near the bin: {d[0]:+.0f}, {d[1]:+.0f} mm from the point aimed at (x, y)")
                    try:
                        VERIFY_DIR.mkdir(exist_ok=True)
                        with open(VERIFY_DIR / "landings.jsonl", "a") as f:
                            f.write(json.dumps({"stamp": time.strftime("%Y%m%d-%H%M%S"), "rollout": self.rollout, "dx": round(d[0], 1), "dy": round(d[1], 1),
                                                "compensated": DROP_COMPENSATE}) + "\n")
                    except Exception:  # noqa: BLE001
                        pass
            cubes = [c for c in cubes if math.dist(c["xy_mm"], self.bin_seen) > BIN_CLEAR_MM]
        if TASK == "bin" and self.want:
            cubes = [c for c in cubes if c["colour"] == self.want]
        self._raw_cubes = list(cubes)  # before any 'already tried' skipping: what the label counts
        cubes = [c for c in cubes if not self._tried_out(c)]
        if (not cubes and DETECTOR == "open_vocab" and BLOB_FALLBACK and img is not None
                and object_phrase(self.instruction) != "cube" and tuple(img.shape[1::-1]) == tuple(self.LOOK_SIZE)):
            # the instruction names something the detector could not find: take what sits on the table that is not the bin, whatever it is;
            # the size gate below decides if the jaws can hold it
            big = object_phrase(self.instruction) in BIG_PHRASES
            cubes = [b for b in (table_blobs(img, self.cams, max_short=90.0, max_long=260.0) if object_phrase(self.instruction) == "key"
                                                              else table_blobs(img, self.cams, max_short=55.0, max_long=100.0) if big else table_blobs(img, self.cams)) if self.bin_seen is None
                     or math.dist(b["xy_mm"], self.bin_seen) > BIN_CLEAR_MM]
            known = object_phrase(self.instruction) in SHAPES  # a named shape is judged by its class, not by a blob that includes its shadow
            cubes = [c for c in cubes if (known or c["short_mm"] <= MAX_GRASP_SHORT_MM) and shape_matches(c, object_phrase(self.instruction))]
            self._raw_cubes = [dict(c, xy_mm=(c["xy_mm"][0] + TABLE_XY_OFFSET[0], c["xy_mm"][1] + TABLE_XY_OFFSET[1])) for c in cubes]
            cubes = [c for c in cubes if not self._tried_out(c)]
            for c in cubes:
                c["colour"] = blob_colour(img, c)
            cubes = self._rank_by_name(img, cubes, object_phrase(self.instruction))
            if big and object_phrase(self.instruction) == "screw":  # grip the shaft, not the middle of the whole screw: the head end is the wider half
                for c in cubes:
                    h = np.array(c["head_dir"]) * c["head_sign"]
                    c["xy_mm"] = (c["xy_mm"][0] - 0.2 * c["long_mm"] * h[0], c["xy_mm"][1] - 0.2 * c["long_mm"] * h[1])
            self.log(f"table blobs (no named match): {len(cubes)} candidate(s): " + ", ".join(
                f"({b['xy_mm'][0]:.0f}, {b['xy_mm'][1]:.0f}) {b['long_mm']:.0f}x{b['short_mm']:.0f} mm" for b in cubes))
        too_wide = [] if object_phrase(self.instruction) in BIG_PHRASES else [c for c in cubes if c.get("short_mm", 0.0) > MAX_GRASP_SHORT_MM]
        if too_wide:
            self.log("too wide for the jaws, not picking: " + ", ".join(
                f"({c['xy_mm'][0]:.0f}, {c['xy_mm'][1]:.0f}) short side {c['short_mm']:.0f} mm > {MAX_GRASP_SHORT_MM:.0f}" for c in too_wide))
            cubes = [c for c in cubes if c.get("short_mm", 0.0) <= MAX_GRASP_SHORT_MM]
        if (not cubes and DETECTOR == "colour" and OPEN_VOCAB and self.placed == 0 and img is not None
                and tuple(img.shape[1::-1]) == tuple(self.LOOK_SIZE)):
            cubes = self._open_vocab(img) or []  # colour mode: the detector as a second chance, first look only
            if TASK == "bin" and self.bin_seen is not None:
                cubes = [c for c in cubes if math.dist(c["xy_mm"], self.bin_seen) > BIN_CLEAR_MM]
        calib = load_calib() if CALIB_APPLY else None
        for c in cubes:
            c["xy_mm"] = (c["xy_mm"][0] + TABLE_XY_OFFSET[0], c["xy_mm"][1] + TABLE_XY_OFFSET[1])
            c["xy_pre_model"] = c["xy_mm"]  # the table estimate before any learned correction (what calib_learn.py fits, and what 'tried' compares)
            c["calib_corr"] = (0.0, 0.0)
            if calib and c.get("colour") in COLOURS and not c.get("elongated"):  # square objects only: the colour wrist view they are measured by
                dx, dy = calib_correction(calib, *c["xy_mm"])
                c["calib_corr"] = (dx, dy)
                c["xy_mm"] = (c["xy_mm"][0] + dx, c["xy_mm"][1] + dy)
                c["calib_version"] = calib.get("version")
        raw = getattr(self, "_raw_cubes", cubes)
        self._label_pending(raw)
        self.pre_objects = [tuple(c["xy_mm"]) for c in raw]  # what was on the table before the next attempt (counting the spots it has given up on)
        if cubes:
            self.tried.append(cubes[0].get("xy_pre_model", cubes[0]["xy_mm"]))
            return cubes[0]
        if FALLBACK_TO_PLACEMENT and not self.used_fallback and "cube" in self._scene():
            self.used_fallback = True
            pos = self._scene()["cube"]
            pos = pos[0] if isinstance(pos[0], (list, tuple)) else pos
            self.log(f"no cube seen: using placement.yaml {pos}")
            return {"colour": "?", "xy_mm": (float(pos[0]), float(pos[1])),
                    "yaw_deg": float(pos[2]) if len(pos) > 2 else 0.0, "iou": 0.0}
        return None

    def _tried_out(self, c: dict) -> bool:
        """True if this spot was already picked TRIED_MAX times and something is still there (the grasp keeps failing)."""
        xy = (c["xy_mm"][0] + TABLE_XY_OFFSET[0], c["xy_mm"][1] + TABLE_XY_OFFSET[1])
        return sum(math.dist(xy, t) < TRIED_MM for t in self.tried) >= TRIED_MAX

    def _rank_by_name(self, img, blobs: list[dict], phrase: str, half: int = 48) -> list[dict]:
        """Order the candidate blobs by how well their close-up matches the named object, each crop scored on its own by the open-vocabulary
        detector, so a battery beats a nut beside it. Without the detector: largest first."""
        if len(blobs) < 2:
            return blobs
        H, W = img.shape[:2]
        for b in blobs:
            u, v = int(b["px"][0]), int(b["px"][1])
            crop = img[max(0, v - half):min(H, v + half), max(0, u - half):min(W, u + half)]
            crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            dets = find_objects(crop, [f"a {phrase}"], threshold=0.01, region=(0, 0, crop.shape[1], crop.shape[0]))
            b["name_score"] = max((d["score"] for d in dets), default=0.0) if dets is not None else None
        if any(b["name_score"] is None for b in blobs):
            return sorted(blobs, key=lambda b: -b["area_px"])
        self.log("name scores for '" + phrase + "': " + ", ".join(
            f"({b['xy_mm'][0]:.0f}, {b['xy_mm'][1]:.0f}) {b['name_score']:.2f}" for b in blobs))
        return sorted(blobs, key=lambda b: -b["name_score"])

    def _colour_cubes(self, img) -> list[dict]:
        """The colour pipeline: colour gates plus a calibrated silhouette fit."""
        t = time.perf_counter()
        cubes = detect_cubes(img, COLOURS, self.cams)
        self.log(f"colour pipeline saw {len(cubes)} cube(s) in {1000 * (time.perf_counter() - t):.0f} ms: " + ", ".join(
            f"{c['colour']} ({c['xy_mm'][0]:.1f}, {c['xy_mm'][1]:.1f}) yaw {c['yaw_deg']:+.1f} iou {c['iou']:.2f}"
            for c in cubes))
        return cubes

    def _open_vocab(self, img) -> list[dict] | None:
        """Cubes found by the open-vocabulary detector (on the Spark, via the SSH tunnel), best first; None if the
        detector is unreachable. It is asked for a small cube of each wanted colour, or for what the instruction
        names ("screw"). The colour in the winning query becomes the cube's colour, so the wrist camera's colour
        refinement can run as usual. Not the bin, and nothing outside the arm's reach."""
        phrase = object_phrase(self.instruction)
        if phrase == "cube":
            queries = [f"a small {c} cube" for c in ([self.want] if self.want else COLOURS)]
        else:
            queries = [f"a small {self.want} {phrase}" if self.want and self.want not in phrase else f"a small {phrase}"]
        t = time.perf_counter()
        dets = find_objects(img, queries, threshold=OPEN_VOCAB_MIN_SCORE)
        if dets is None:
            return None
        objs = []
        for o in objects_from_detections(img, dets, self.cams):  # best score first
            if o["iou"] < OPEN_VOCAB_MIN_SCORE or any(math.dist(o["xy_mm"], k["xy_mm"]) < 8.0 for k in objs):
                continue  # below the bar, or the same object found by another query
            o["colour"] = next((c for c in COLOURS if c in o["label"]), "any")
            objs.append(o)
        self.log(f"open-vocabulary ({time.perf_counter() - t:.1f} s) saw {len(objs)} object(s): " + ", ".join(
            f"{o['label']} {o['iou']:.2f} ({o['xy_mm'][0]:.1f}, {o['xy_mm'][1]:.1f}) yaw {o['yaw_deg']:+.1f}"
            for o in objs))
        return objs

    def _find_bin(self, img) -> None:
        """Where the bin is, from the table camera. Kept from the last look that saw it."""
        xy = detect_bin(img, self.cams)
        if xy is None:
            self.log("no bin seen" + (f": keeping the last one {self.bin_seen[0]:.1f}, {self.bin_seen[1]:.1f}" if self.bin_seen else ""))
            return
        self.bin_seen = xy  # no TABLE_XY_OFFSET: that was fitted to cubes at table height, and the cubes landed on its side
        self.log(f"bin at ({self.bin_seen[0]:.1f}, {self.bin_seen[1]:.1f})")

    def _ask_cosmos(self, img, cubes) -> None:
        """Once per rollout: a second opinion from Cosmos (on the Spark, via the SSH tunnel). Never blocks the pick.

        It only fills in the colour when the instruction asks for one in words the regex missed, and only if the
        camera pipeline actually sees a cube of that colour. Cosmos also mislabels the yellow bin as a cube and
        guesses a colour when none is asked for, so its object list is logged but never acted on.
        """
        if self.cosmos_done:
            return
        self.cosmos_done = True
        t = time.perf_counter()
        out = cosmos_understand(img, str(self.instruction or ""), timeout=COSMOS_TIMEOUT)
        if out is None:
            self.log("cosmos: unreachable, using the camera pipeline alone")
            return
        seen = sorted(f"{o.get('colour')} {o.get('category')}" for o in out["objects"])
        self.log(f"cosmos ({time.perf_counter() - t:.1f} s): sees {seen}, pick_colour {out['pick_colour']}; "
                 f"camera pipeline sees {sorted(c['colour'] for c in cubes)}")
        pick = out["pick_colour"]
        if self.want is None and pick and names_a_colour(self.instruction) and any(c["colour"] == pick for c in cubes):
            self.want = pick
            self.log(f"cosmos: picking the {pick} cube")

    def _save_debug(self, img, cubes) -> None:
        try:
            self.debug_dir.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            name = f"{stamp}_r{self.rollout}_look{self.placed}"
            cv2.imwrite(str(self.debug_dir / f"{name}.png"), draw(img, cubes, self.cams))
            cv2.imwrite(str(self.debug_dir / f"{name}_raw.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # for retuning
        except Exception as e:  # noqa: BLE001  (debug output must never stop the arm)
            self.log(f"debug frame not saved: {e}")

    def _save_table_depth(self, obs) -> None:
        d = (obs.get("depth") or {}).get("table")
        if d is None:
            return
        try:
            self.debug_dir.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            np.savez_compressed(self.debug_dir / f"{stamp}_r{self.rollout}_look{self.placed}_table_depth.npz", depth=d,
                                joints=np.asarray(obs["joints"], float))
            v = d[d > 0]
            self.log(f"table depth saved: {d.shape}, {100 * float((d > 0).mean()):.0f}% valid, "
                     f"median {float(np.median(v)) if v.size else 0:.0f} mm")
        except Exception as e:  # noqa: BLE001
            self.log(f"table depth not saved: {e}")

    def _wrist_fix(self, obs) -> bool:
        """Correct the target from the wrist camera while hovering; True if the arm is sent to re-hover."""
        img = obs["images"].get("wrist")
        colour = self.target.get("colour")
        if img is None or colour not in COLOURS + ("any",):
            return False
        loc = None
        wrist_depth = (obs.get("depth") or {}).get("wrist")
        if wrist_depth is not None and colour == "any" and self.target.get("elongated"):  # a long object: its height is the clearest cue. A square one (a cube) goes the colour way below: the depth blob blurs its corners, so it cannot give its angle (30 Oct: a thrown cube hit at the table blob's angle, 20 deg off)
            try:
                d = wrist_locate_depth(wrist_depth, obs["joints"][:6], self.target["xy_mm"], self.cams)
            except Exception as e:  # noqa: BLE001
                d = None
                self.log(f"wrist depth locate failed: {e}")
            if d is not None and 4.0 <= d["short_mm"] <= 30.0 and d["long_mm"] >= 14.0:
                axis_diff = abs((d["axis_deg"] - self.target["yaw_deg"] + 90.0) % 180.0 - 90.0)
                yaw_d = d["axis_deg"] if self.target.get("elongated") and axis_diff < 25.0 and d["long_mm"] >= 40.0 else self.target["yaw_deg"]  # a cut-off blob has no reliable axis
                xy_d, base = d["xy_mm"], (self.target.get("table_xy_mm") or self.target["xy_mm"])
                if self.target.get("elongated"):
                    # the wrist view often cuts a long object off at its edge, so the blob's centre slides along the object at every re-hover (28 Oct:
                    # 24 mm, to its very tip). Only the position ACROSS the jaws matters: take that from depth, and keep the table camera's position along it.
                    n = np.array([-math.sin(math.radians(yaw_d)), math.cos(math.radians(yaw_d))])
                    across = float(np.dot(np.asarray(xy_d) - np.asarray(base), n))
                    xy_d = (float(base[0] + n[0] * across), float(base[1] + n[1] * across))
                hist = (self.target.get("depth_hist") or [])[-2:] + [tuple(xy_d)]  # the depth's across position jitters ~1 mm view to view: average the last three
                xy_d = (float(np.mean([h[0] for h in hist])), float(np.mean([h[1] for h in hist])))
                loc = {"xy_mm": xy_d, "yaw_deg": yaw_d, "located_by": "depth", "table_xy_mm": tuple(base), "depth_hist": hist}
                self.log(f"wrist depth: object {d['long_mm']:.0f} x {d['short_mm']:.0f} mm, {d['height_mm']:.0f} mm high, axis {d['axis_deg']:+.0f}")
            else:
                thin = float(self.target.get("short_mm") or 99.0) <= THIN_MM  # a thin dark object is below the depth's noise and casts no big shadow: the colour view is the better one
                self.log("wrist depth: " + ("no object-sized raised blob" if d is None else f"raised blob {d['long_mm']:.0f} x {d['short_mm']:.0f} mm is not object-sized")
                         + (" near the target; a thin object: using the colour view" if thin else " near the target; keeping the table-camera estimate (the colour view follows shadows)"))
                if not thin:
                    return False
        if loc is None:
            try:
                loc = wrist_locate(img, obs["joints"][:6], colour, self.target["xy_mm"], self.cams)
            except Exception as e:  # noqa: BLE001
                self.log(f"wrist locate failed: {e}")
                return False
        if loc is None:
            self.log("wrist view: cube not found near the target; keeping the table-camera estimate")
            return False
        if not self.target.get("calib_logged") and loc.get("located_by") != "depth" and colour in COLOURS:
            self._log_calib_pair(loc)  # the first view of this cube, taken right above the table camera's (corrected) estimate
            self.target = {**self.target, "calib_logged": True}
        shift = math.dist(loc["xy_mm"], self.target["xy_mm"])
        self.log(f"wrist view: cube at ({loc['xy_mm'][0]:.1f}, {loc['xy_mm'][1]:.1f}) yaw {loc['yaw_deg']:+.1f}, "
                 f"{shift:.1f} mm from the target")
        yaw = loc["yaw_deg"] if loc.get("located_by") == "depth" else (self.target["yaw_deg"] if self.target.get("elongated") else loc["yaw_deg"])  # the colour view's yaw is mod 90: wrong for a long object
        if SKIP_WRIST_XY:  # keep the table camera's position, take only the angle: no re-hover, no repositioning
            self.target = {**self.target, "yaw_deg": yaw}
            if abs(self._r(yaw) - self.pose[5]) > 0.5:
                self.move(*self.pose[:3], r=self._r(yaw))
            return False
        self.target = {**self.target, "xy_mm": loc["xy_mm"], "yaw_deg": yaw, "located_by": loc.get("located_by"),
                       "table_xy_mm": loc.get("table_xy_mm", self.target.get("table_xy_mm")), "depth_hist": loc.get("depth_hist", self.target.get("depth_hist"))}
        if shift > WRIST_FIX_MM and self.fixes < WRIST_FIXES:
            self.fixes += 1
            self.hover(*self.target["xy_mm"], self.target["yaw_deg"])
            return True
        if abs(self._r(yaw) - self.pose[5]) > 0.5:  # turn the jaws to the measured yaw
            self.move(*self.pose[:3], r=self._r(yaw))
        return False

    def _log_calib_pair(self, loc: dict) -> None:
        """One line per cube: table estimate (before any learned correction), the correction applied, and where the wrist camera then sees it.
        wrist - pre = what the model should predict; wrist - hover = the error the arm actually started with."""
        try:
            pre = self.target.get("xy_pre_model") or self.target["xy_mm"]
            corr = self.target.get("calib_corr") or (0.0, 0.0)
            CALIB_LOG.parent.mkdir(exist_ok=True)
            with open(CALIB_LOG, "a") as f:
                f.write(json.dumps({"t": time.time(), "stamp": time.strftime("%Y%m%d-%H%M%S"), "rollout": self.rollout, "colour": self.target.get("colour"),
                                    "pre_xy": [round(float(v), 2) for v in pre], "corr": [round(float(v), 2) for v in corr],
                                    "applied": self.target.get("calib_version") is not None,
                                    "version": self.target.get("calib_version"), "hover_xy": [round(float(v), 2) for v in self.target["xy_mm"]],
                                    "wrist_xy": [round(float(v), 2) for v in loc["xy_mm"]], "yaw_deg": round(float(loc["yaw_deg"]), 1),
                                    "R": round(float(self.pose[5]), 1)}) + "\n")
        except Exception as e:  # noqa: BLE001
            self.log(f"calibration pair not recorded: {e}")

    def hold_until_there(self) -> None:
        """The next phase starts only once the real fingertip reaches the last queued target."""
        self.arrive = (*self.pose[:3], None)

    def _arrived(self, obs) -> bool:
        x, y, z, since = self.arrive
        if since is None:  # the clock starts once the queued rows have played
            since = time.time()
            self.arrive = (x, y, z, since)
        tcp = obs.get("tcp_pose")
        if tcp is None or len(tcp) < 3:  # no feedback (the mock before any tcp command): trust the timing
            self.arrive = None
            return True
        err = math.dist((float(tcp[0]), float(tcp[1]), float(tcp[2])), (x, y, z))
        if err <= ARRIVE_MM or time.time() - since > ARRIVE_TIMEOUT:
            if err > ARRIVE_MM:
                self.log(f"waypoint ({x:.0f}, {y:.0f}, {z:.0f}) not reached after {ARRIVE_TIMEOUT:.0f} s "
                         f"(off by {err:.1f} mm): going on")
            else:
                self.log(f"arrived at ({x:.0f}, {y:.0f}, {z:.0f}) after {time.time() - since:.1f} s (err {err:.1f} mm)")
            self.arrive = None
            return True
        return False

    def _record_descent(self, x: float, y: float) -> None:
        """One line per descent, written BEFORE the fingers go down: a hit ends the trial, so it never reaches the lift check. The cell's end reason
        for the trial (manipbench.json: "controller fault" = a hit) is matched to this by time."""
        try:
            VERIFY_DIR.mkdir(exist_ok=True)
            row = {"t": time.time(), "stamp": time.strftime("%Y%m%d-%H%M%S"), "rollout": self.rollout, "phrase": object_phrase(self.instruction),
                   "located_by": self.target.get("located_by"), "offset_gripper_mm": getattr(self, "last_offset", None),
                   "deliberately_bad": bool(self.target.get("explore_bad")), "grasp_xy_mm": [round(x, 1), round(y, 1)],
                   "target_xy_mm": [round(float(v), 1) for v in self.target["xy_mm"]], "R": round(float(self.pose[5]), 1),
                   "short_mm": self.target.get("short_mm"), "long_mm": self.target.get("long_mm"), "wrist_fixes": self.fixes,
                   "calib_version": self.target.get("calib_version")}
            with open(VERIFY_DIR / "descents.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:  # noqa: BLE001
            self.log(f"descent not recorded: {e}")

    def _after_pick(self) -> None:
        """Carry the object to the bin (re-centred over it by the wrist camera if BIN_SERVO) and release."""
        if BIN_SERVO and TASK == "bin" and self.bin_seen is not None:
            self.bin_fixes = 0
            self.move(*self.bin_seen, BIN_VIEW_Z)  # carry it over the bin, high, and let the wrist camera centre the jaws on it
            self.wait(PEEK_SETTLE)
            self.hold_until_there()
            self.phase = "bin_peek"
            self.IMAGE_SIZE = PEEK_SIZE
        else:
            self.place(self.placed)
            self.placed += 1
            self.phase = "placed" if self.placed < len(STACK_ORDER) else "done"

    def _held_pixels(self, obs) -> tuple[int, float] | None:
        """(pixels clearly nearer than the fingertips' own distance, that distance in mm) in the wrist depth around where the fingertips project:
        an object held between the fingers is at the tip's distance, the table is 70+ mm beyond it. None without wrist depth."""
        dep = (obs.get("depth") or {}).get("wrist")
        tcp = obs.get("tcp_pose")
        if dep is None or tcp is None:
            return None
        wd = self.cams["wrist_depth"]
        T = flange_pose(obs["joints"][:6]) @ np.asarray(wd["T"])
        pc = np.linalg.inv(T) @ np.array([tcp[0] / 1000, tcp[1] / 1000, (tcp[2] + 330.0) / 1000, 1.0])
        if pc[2] <= 0.02:
            return None
        u, d_tip = wd["fx"] * pc[0] / pc[2] + wd["cx"], pc[2] * 1000.0
        d = np.asarray(dep, float)
        H, W = d.shape
        strip = d[int(H * 0.75):, max(0, int(u) - 110):min(W, int(u) + 110)]
        n = int(((strip > 40) & (strip < d_tip + 45.0)).sum())
        return n, d_tip

    def _side_p_held(self, obs) -> float | None:
        """P(held) from the fixed side camera: a crop around the projected fingertip, as the verifier was trained on it (cut from the 640-wide recording,
        96 px square, shown at 240). None if there is no side image or the verifier is unreachable."""
        side, tcp = obs["images"].get("side"), obs.get("tcp_pose")
        if side is None or tcp is None:
            return None
        cam = self.cams["side"]
        p = project(cam, np.array([[tcp[0] / 1000, tcp[1] / 1000, (tcp[2] + 330.0) / 1000]]))[0]
        small = cv2.resize(side, (640, 360), interpolation=cv2.INTER_AREA)
        u, v, h = int(p[0] * 640 / cam["size"][0]), int(p[1] * 640 / cam["size"][0]) + SIDE_CROP_DV, 48
        if not (h <= u < 640 - h and h <= v < 360 - h):
            return None  # the fingertip is outside the side camera's view
        crop = cv2.resize(small[v - h:v + h, u - h:u + h], (240, 240), interpolation=cv2.INTER_CUBIC)
        try:
            VERIFY_DIR.mkdir(exist_ok=True)
            cv2.imwrite(str(VERIFY_DIR / f"{(self.pending or {}).get('id', time.strftime('%Y%m%d-%H%M%S'))}_sidecrop.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        except Exception:  # noqa: BLE001
            pass
        return verifier_client.p_held(crop)

    def _still_on_table(self, obs) -> float | None:
        """Distance (mm) from the grasp spot to a cube of the target's colour that the lifted wrist camera still sees on the table within WRIST_STILL_MM
        (not one of the other cubes seen at the look), else None."""
        colour, img = (self.target or {}).get("colour"), obs["images"].get("wrist")
        if img is None or colour not in COLOURS or (self.pending or {}).get("xy") is None:
            return None
        spot = tuple(self.pending["xy"])
        try:
            loc = wrist_locate(img, obs["joints"][:6], colour, spot, self.cams, max_shift_mm=WRIST_STILL_MM)
        except Exception as e:  # noqa: BLE001
            self.log(f"still-on-table check failed: {e}")
            return None
        if loc is None:
            return None
        others = [o for o in (getattr(self, "pre_objects", None) or []) if math.dist(o, self.target["xy_mm"]) > 20.0]
        if any(math.dist(loc["xy_mm"], o) < 25.0 for o in others):
            return None  # that is another cube of the layout, not the one we went for
        return math.dist(loc["xy_mm"], spot)

    def _verify(self, obs) -> str:
        """Save the lifted wrist view and a verdict ("held", "unsure", "empty"); the label comes at the next table look."""
        held = self._held_pixels(obs)
        verdict = "unsure" if held is None else "held" if held[0] >= VERIFY_HELD_PX else "unsure" if held[0] >= VERIFY_UNSURE_PX else "empty"
        ph = self._side_p_held(obs)  # the side camera sees a held object of any size (the wrist depth only a long one)
        if ph is not None:
            verdict = "held" if ph >= SIDE_HELD_P else "empty"
        elif not (self.target or {}).get("elongated"):
            verdict = "n/a"  # a small object between the fingertips is below the wrist camera's view and there is no side verdict
        still = self._still_on_table(obs) if WRIST_STILL_MM > 0 else None
        if WRIST_STILL_MM > 0:  # this check decides; the side verifier is only logged (it was fooled by a cube lying near the fingertips)
            verdict = "empty" if still is not None else "held"
        meta = dict(self.pending or {}, verdict=verdict, still_on_table_mm=None if still is None else round(still, 1), near_px=None if held is None else held[0], side_p_held=None if ph is None else round(ph, 3), d_tip_mm=None if held is None else round(held[1], 1),
                    tcp=[round(float(v), 1) for v in (obs.get("tcp_pose") or [])], joints=[round(float(v), 1) for v in obs["joints"]])
        try:
            VERIFY_DIR.mkdir(exist_ok=True)
            stem = VERIFY_DIR / meta.get("id", time.strftime("%Y%m%d-%H%M%S"))
            for cam, img in obs["images"].items():
                cv2.imwrite(f"{stem}_{cam}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if (obs.get("depth") or {}).get("wrist") is not None:
                np.savez_compressed(f"{stem}_wrist_depth.npz", depth=obs["depth"]["wrist"])
            pk = getattr(self, "last_peek", None)
            if pk is not None and pk["wrist"] is not None:  # the pre-grasp view the offset was chosen from
                cv2.imwrite(f"{stem}_pre_wrist.png", cv2.cvtColor(pk["wrist"], cv2.COLOR_RGB2BGR))
                np.savez_compressed(f"{stem}_pre.npz", depth=pk["depth"] if pk["depth"] is not None else np.zeros(0), joints=pk["joints"], tcp=pk["tcp"])
            with open(VERIFY_DIR / "attempts.jsonl", "a") as f:
                f.write(json.dumps(meta) + "\n")
        except Exception as e:  # noqa: BLE001
            self.log(f"grasp check frames not saved: {e}")
        if self.pending is not None:
            self.pending["verdict"] = verdict
        self.log(f"grasp check: {verdict}" + ("" if still is None else f" (the cube is still on the table, {still:.0f} mm from the grasp spot)") + ("" if ph is None else f" (side camera: P(held) {ph:.2f})") + ("" if held is None else f" ({held[0]} px nearer than the fingertips at {held[1]:.0f} mm)"))
        return verdict

    def _label_pending(self, cubes: list) -> None:
        """At the first table look after an attempt, label it from how the table changed (cubes: the candidates of this look, offset applied):
        "cleared"    one fewer object, none at the pick spot: it left the table (in the bin, or lost on the way: the bin is told apart by the bin look)
        "at_spot"    an object is still where the fingers went: the grasp missed
        "moved"      the same number of objects but none at the spot: the grasp nudged it away, or it was dropped somewhere on the table
        Anything where the verdict and the label disagree is also listed in disagreements.txt, to check by eye."""
        if self.pending is None:
            return
        pend, self.pending = self.pending, None
        try:
            pre = getattr(self, "pre_objects", None)
            now = [tuple(c["xy_mm"]) for c in cubes]
            at_spot = any(math.dist(p, pend["xy"]) < 30.0 for p in now)
            n_pre = len(pre) if pre is not None else None
            if at_spot:
                label = "at_spot"
            elif n_pre is not None and len(now) < n_pre:
                label = "cleared"
            else:
                label = "moved"
            row = {"id": pend["id"], "verdict": pend.get("verdict"), "label": label, "n_before": n_pre, "n_after": len(now), "bin": self.bin_seen}
            VERIFY_DIR.mkdir(exist_ok=True)
            with open(VERIFY_DIR / "labels.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            disagree = pend.get("verdict") != "n/a" and ((pend.get("verdict") == "held" and label != "cleared") or (pend.get("verdict") == "empty" and label == "cleared"))
            if disagree:
                with open(VERIFY_DIR / "disagreements.txt", "a") as f:
                    f.write(f"{pend['id']}: verdict {pend.get('verdict')} but label {label}: look at {pend['id']}_wrist.png\n")
            self.log(f"grasp label: attempt {pend['id']}: verdict {pend.get('verdict')}, outcome {label}" + ("   <-- DISAGREE, check by eye" if disagree else ""))
        except Exception as e:  # noqa: BLE001
            self.log(f"grasp label not written: {e}")

    def _bin_fix(self, obs) -> bool:
        """Centre the jaws over the bin from the wrist camera while carrying: True if the arm is sent to re-hover."""
        img = obs["images"].get("wrist")
        if img is None:
            return False
        try:
            xy = wrist_locate_bin(img, obs["joints"][:6], self.bin_seen, self.cams)
        except Exception as e:  # noqa: BLE001
            self.log(f"wrist bin locate failed: {e}")
            return False
        if xy is None:
            self.log("wrist view: bin not found near the table-camera estimate; keeping it")
            return False
        shift = math.dist(xy, self.bin_seen)
        self.log(f"wrist view: bin at ({xy[0]:.1f}, {xy[1]:.1f}), {shift:.1f} mm from where the table camera put it")
        self.bin_seen = xy
        if shift > BIN_FIX_MM and self.bin_fixes < BIN_FIXES:
            self.bin_fixes += 1
            self.move(xy[0], xy[1], BIN_VIEW_Z)
            self.wait(PEEK_SETTLE)
            return True
        return False

    def _save_peek(self, obs, tag: str = "peek") -> None:
        if tag == "peek":  # the last hover before the descent: what the grasp was aimed from (saved with the attempt's verdict)
            self.last_peek = {"wrist": obs["images"].get("wrist"), "depth": (obs.get("depth") or {}).get("wrist"),
                              "joints": np.asarray(obs["joints"], float), "tcp": np.asarray(obs.get("tcp_pose") or [], float)}
        """The wrist (and table) view while hovering, open jaws turned to the cube: shows any x, y or yaw offset."""
        try:
            self.debug_dir.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            for cam, img in obs["images"].items():
                cv2.imwrite(str(self.debug_dir / f"{stamp}_r{self.rollout}_{tag}{self.placed}_{cam}.png"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            tcp = obs.get("tcp_pose")
            if obs.get("depth") and "wrist" in obs["depth"]:
                np.savez_compressed(self.debug_dir / f"{stamp}_r{self.rollout}_{tag}{self.placed}_wrist_depth.npz",
                                    depth=obs["depth"]["wrist"], colour=obs["images"].get("wrist"),
                                    joints=np.asarray(obs["joints"], float), tcp=np.asarray(tcp if tcp is not None else [], float))
                d = obs["depth"]["wrist"]
                self.log(f"wrist depth saved: {d.shape}, {100 * float((d > 0).mean()):.0f}% of pixels valid")
            self.log(f"peek: tcp {None if tcp is None else [round(float(v), 1) for v in tcp]} "
                     f"joints {[round(float(v), 1) for v in obs['joints']]}")
        except Exception as e:  # noqa: BLE001
            self.log(f"peek frame not saved: {e}")

    # -- the policy -------------------------------------------------------------------------------------------------
    def act(self, obs: dict):
        """obs["images"]   {camera: (H, W, 3) uint8 RGB}
        obs["joints"]      [J1..J6 deg (the v0.5 twin's convention), gripper 0 open / 1 closed]
        obs["tcp_pose"]    the fingertip [x, y, z mm, w, p, r deg]
        obs["trial"]["instruction"]  the task in words (you may ignore it)

        Returns up to 100 rows of [x, y, z, w, p, r, gripper] (ACTION = "tcp"), absolute targets.
        """
        if self.pose is None:
            trial = obs.get("trial") or {}
            self.rollout = trial.get("index")
            mode = apply_trial_mode(self.rollout)
            self.trial_mode = mode
            if mode:
                self.log(f"trial mode {mode}: " + ("the arm stays at the table camera's position, the wrist view only turns the jaws" if mode == "A"
                                                    else f"the wrist camera repositions the arm, its shift {WRIST_SHIFT_B} mm taken out"))
            target = read_target_object() or OBJECT_OVERRIDE
            self.instruction = f"Put the {target} into the yellow bin." if target else trial.get("instruction")
            self.want = colour_from_instruction(self.instruction)
            self.cosmos_done = False
            self.log(f"instruction: {trial.get('instruction')!r} -> colour {self.want or 'any'}")
            tcp = obs.get("tcp_pose")
            if tcp is not None and len(tcp) >= 6:
                x, y, z, r = float(tcp[0]), float(tcp[1]), float(tcp[2]), float(tcp[5])
            else:  # the mock reports no tcp_pose until the first tcp command: use FK on the joints
                (x, y, z), r = tip_from_joints(obs["joints"][:6]), 0.0
            g = float(obs["joints"][6]) if len(obs["joints"]) > 6 else 0.0
            self.pose = clip_target([x, y, z, DOWN[0], DOWN[1], r, g])
            self.log(f"rollout {self.rollout}: start at ({x:.0f}, {y:.0f}, {z:.0f}) R {r:.0f}; task {TASK}")

        if not self.queue and self.arrive is not None and not self._arrived(obs):
            self.wait(0.5)  # hold the waypoint until the real arm is there
            chunk, self.queue = self.queue[:MAX_CHUNK], self.queue[MAX_CHUNK:]
            return chunk

        if not self.queue:
            if self.phase in ("start", "placed"):
                self.move(*PARK, r=0.0, g=0.0)  # out of the camera's way, gripper open
                self.hold_until_there()
                self.phase = "settle"
            elif self.phase == "settle":
                self.wait(LOOK_SETTLE)  # at PARK: let the arm stop swinging before the look
                self.phase = "look"
                self.IMAGE_SIZE = self.LOOK_SIZE  # the next observation is the full-resolution look
            elif self.phase == "look":
                target = self.look(obs)
                self.IMAGE_SIZE = Policy.IMAGE_SIZE
                if target is None:
                    self.log(f"nothing left to do after {self.placed} cube(s): holding")
                    self.phase = "done"
                else:
                    self.log(f"pick {target['colour']} at ({target['xy_mm'][0]:.1f}, {target['xy_mm'][1]:.1f}) "
                             f"yaw {target['yaw_deg']:+.1f} -> R {wrist_r(target['yaw_deg'], self.pose[5], target.get('elongated', False)):+.1f}"
                             + (f" ({target['long_mm']:.0f} x {target['short_mm']:.0f} mm, elongated)" if target.get("elongated") else ""))
                    self.target = target
                    self.fixes = 0
                    self.hover(*target["xy_mm"], target["yaw_deg"])
                    self.hold_until_there()
                    self.phase = "peek"
                    self.IMAGE_SIZE = PEEK_SIZE  # the next observation: the wrist camera right above the cube
            elif self.phase == "peek":
                self._save_peek(obs)
                if self._wrist_fix(obs):  # re-hovering over the corrected spot: peek again once there
                    self.hold_until_there()
                    chunk, self.queue = self.queue[:MAX_CHUNK], self.queue[MAX_CHUNK:]
                    return chunk
                self.IMAGE_SIZE = Policy.IMAGE_SIZE
                if CAPTURE_ONLY:  # hover-and-record: the depth is saved above; never go down
                    self.log("capture only: not descending")
                    self.phase = "done"  # act() then holds still at the end of this call
                else:
                    gx, gy = self.grasp_xy()
                    self._record_descent(gx, gy)
                    self.descend(gx, gy)
                    self.hold_until_there()
                    self.phase = "grasp"
            elif self.phase == "grasp":
                self.pick(*self.grasp_xy(), self.target["yaw_deg"])
                self.attempt += 1
                self.pending = {"id": f"{time.strftime('%Y%m%d-%H%M%S')}_r{self.rollout}_a{self.attempt}", "xy": tuple(self.pose[:2]),
                                "phrase": object_phrase(self.instruction), "located_by": self.target.get("located_by"),
                                "yaw_deg": float(self.target["yaw_deg"]), "offset_mm": self.descended_offset,
                                "elongated": bool(self.target.get("elongated")), "mode": getattr(self, "trial_mode", None),
                                "offset_gripper_mm": getattr(self, "last_offset", None), "deliberately_bad": bool(self.target.get("explore_bad")), "grasp_xy_mm": [round(float(v), 1) for v in self.pose[:2]],
                                "target_xy_mm": [round(float(v), 1) for v in self.target["xy_mm"]], "short_mm": self.target.get("short_mm"),
                                "long_mm": self.target.get("long_mm")}
                if VERIFY:  # lifted: hold still, and let the wrist camera look between the fingers
                    self.wait(PEEK_SETTLE)
                    self.hold_until_there()
                    self.phase = "verify"
                    self.IMAGE_SIZE = PEEK_SIZE
                else:
                    self._after_pick()
            elif self.phase == "verify":
                verdict = self._verify(obs)
                self.IMAGE_SIZE = Policy.IMAGE_SIZE
                if VERIFY_GATE and verdict == "empty":
                    self.log("grasp check: nothing between the fingers; opening them and looking again")
                    self.move(*self.pose[:3], g=0.0)
                    self.wait(0.5)
                    self.phase = "placed"
                else:
                    self._after_pick()
            elif self.phase == "bin_peek":
                self._save_peek(obs, tag="bin")
                if self._bin_fix(obs):
                    self.hold_until_there()
                    chunk, self.queue = self.queue[:MAX_CHUNK], self.queue[MAX_CHUNK:]
                    return chunk
                self.IMAGE_SIZE = Policy.IMAGE_SIZE
                self.place(self.placed)
                self.placed += 1
                self.phase = "placed" if self.placed < len(STACK_ORDER) else "done"
            if self.phase == "done" and not self.queue:
                self.wait(2.0)  # hold still; act is called again after these play

        chunk, self.queue = self.queue[:MAX_CHUNK], self.queue[MAX_CHUNK:]
        return chunk
