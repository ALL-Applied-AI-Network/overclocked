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

import math
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from calibration.geometry import SURFACE_Z, flange_pose, load
from cosmos_client import understand as cosmos_understand
from detect_client import find as find_objects
from cube_vision import MAX_GRASP_SHORT_MM, blob_colour, detect_bin, detect_cubes, objects_from_detections, table_blobs, draw, square_yaw, wrist_locate

HERE = Path(__file__).resolve().parent

# --- task -------------------------------------------------------------------------------------------------------
TASK = "bin"                                  # "bin": every cube into the bin; "stack": build STACK_ORDER
COLOURS = ("yellow", "blue", "red")           # which cube colours to look for
STACK_ORDER = ["yellow", "yellow", "blue", "blue", "red", "red"]  # bottom to top
STACK_XY = (450.0, -150.0)                    # mm, UF0; where the stack is built
BIN_XY = (510.0, 110.0)                       # mm; the default scene's bin. placement.yaml's "bin" overrides it
# Where the fingers go relative to the wrist-corrected position, in the GRIPPER's frame (mm): (along the jaws, across them).
# The 26 Oct cube probes: -3 mm across the jaws grasped, +3 and 0 landed the finger on the cube's top; along the jaws made no
# difference. The error is the wrist camera's, so it turns with the wrist: at R 0 across = the robot's y. A battery lying
# along y (R 74 deg) faulted on 27 Oct with the offset held in the robot's y, where it moved the jaws along the battery.
# 27 Oct: (0, -3) put a finger on the cube twice (13:09, 13:13) although the measurement matched yesterday's successes to the pixel, and
# the fingers had taken ~8 hits since: re-probe. One value per rollout, indexed by the rollout number (rollouts 0, 1 keep -3).
GRASP_PROBES = [(0, -3), (0, -3), (0, -4.5), (0, -1.5), (0, -6), (0, -7.5), (0, -0.8), (0, -9), (0, -4.5), (0, -6)]
# Which frame the offset above lives in: "gripper" (turns with the wrist) or "robot" (fixed x, y). A frame test with a cube turned to
# 90 degrees tells them apart: GRASP_FRAME=robot / gripper, FORCE_CUBE_R=90 uv run run.py. Both are off by default (robot frame, free R).
GRASP_FRAME = os.environ.get("GRASP_FRAME", "robot")  # "robot": what grasped cubes on 26 Oct (trial 0 of 27 Oct, rotating, hit one)
FORCE_CUBE_R = float(os.environ["FORCE_CUBE_R"]) if os.environ.get("FORCE_CUBE_R") else None
BIN_CLEAR_MM = 40.0                           # a cube this close to the bin's centre is already in it (or on its rim)
# Which detector finds the cubes: "open_vocab" (OWLv2 on the Spark, text queries) or "colour" (colour gates + silhouette
# fit, the version that grasped on 26 Oct). Switch without editing: CUBE_DETECTOR=colour uv run run.py
DETECTOR = os.environ.get("CUBE_DETECTOR", "open_vocab")
DETECTOR_FALLBACK_IF_UNREACHABLE = True       # open_vocab only: a dead tunnel falls back to the colour pipeline, nothing else does
BLOB_FALLBACK = True                          # open_vocab only: a named non-cube object the detector cannot find -> anything on the table
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
PEEK_SETTLE = 1.5                             # s hovering over the cube before the wrist frame is saved
PEEK_SIZE = (1280, 720)                       # the wrist camera's full size
TABLE_XY_OFFSET = (0.0, 10.0)                 # mm added to table-camera positions: the wrist camera and the
                                              # operator's placements both put cubes ~10 mm further +y (slot 1)
# A second wrist view from low down (LOW_PEEK_ABOVE mm over the cube's top). Trial 4 of 27 Oct put a finger on a cube at y -346, where
# the fixed offset that worked at y -130..+300 did not. If the wrist camera is tilted against the tool axis, the cube appears shifted by an
# amount that grows with the camera's height, so the shift between the two heights measures it. LOW_PEEK logs it; LOW_PEEK_USE=1 also
# re-centres on the low view. Off unless LOW_PEEK=1.
# DRY_DESCENT=1 uv run run.py: no cube, no bin. Park, go over (DRY_XY, -265), then come straight down at SLOW until the controller faults
# on the table. The z where it stops is the fingertip's real contact height, to compare with the table the code assumes (SURFACE -335.1).
DRY_DESCENT = os.environ.get("DRY_DESCENT") == "1"
DRY_XY = (400.0, -100.0)
LOW_PEEK = os.environ.get("LOW_PEEK") == "1"
LOW_PEEK_USE = os.environ.get("LOW_PEEK_USE") == "1"
LOW_PEEK_ABOVE = 30.0
WRIST_FIX_MM = 2.0                            # re-hover when the wrist view moves the cube more than this
WRIST_FIXES = 2                               # at most this many re-hovers per cube
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
    """The R that aligns the jaws with a square of this yaw, the 90-degree twin nearest the current R.
    An elongated object (yaw = its long axis) has one right way round, so its twin is 180 degrees away: with the jaws
    closing at 90 degrees to R (the cube trials showed y, not x, is the closing axis at R 0) R = the long axis puts
    them across the narrow side. EXPERIMENTAL: not yet seen working on the robot."""
    if elongated:
        base = (yaw_deg + R_OFFSET + 90.0) % 180.0 - 90.0
        return min((base + k * 180.0 for k in range(-1, 2)), key=lambda r: (abs(r - current_r), abs(r)))
    base = square_yaw(yaw_deg + R_OFFSET)
    target = current_r if FORCE_CUBE_R is None else FORCE_CUBE_R  # the frame test turns the cube's jaws to 90 degrees
    return min((base + k * 90.0 for k in range(-3, 4)), key=lambda r: (abs(r - target), abs(r)))


class Policy:
    CAMERAS = ["table", "wrist"]          # table: finding cubes; wrist: reserved for close-up yaw refinement
    IMAGE_SIZE = (640, 360)               # (width, height) of the images you get; act switches it per phase below
    IMAGE_FORMAT = "png"                  # or "jpeg": about 7x smaller, for a slow connection
    ACTION = "tcp"                        # what act returns: "joints" or "tcp"
    PREFETCH = False                      # whole moves are planned at once; nothing to gain from prefetching
    DEPTH = []                            # ["table"] and/or ["wrist"]: also those cameras' depth, in obs["depth"]
    LOOK_SIZE = (1920, 1080)              # a 10 mm cube is ~15 px wide at full size, ~5 px at 640 x 360

    def __init__(self, checkpoint: str | None = None):
        self.cams = load()
        self.placement = self._load_placement()
        self.debug_dir = HERE / "debug"
        self.reset()

    # -- bookkeeping ------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Called at the start of every rollout."""
        self.queue: list[list[float]] = []
        self.phase = "start"
        self.pose = None              # the last commanded [x, y, z, w, p, r, g]
        self.placed = 0               # cubes put away this rollout
        self.used_fallback = False
        self.want = None
        self.bin_seen = None
        self.instruction = None
        self.cosmos_done = False
        self.rollout = None
        self.target = None
        self.high_xy = None
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
        self.move(x, y, SURFACE + CUBE + APPROACH, r=self._r(yaw), g=0.0)
        self.wait(PEEK_SETTLE)

    def grasp_xy(self) -> tuple[float, float]:
        """Where the fingers go down: the corrected cube position plus this rollout's probe offset."""
        gx, gy = GRASP_PROBES[(self.rollout or 0) % len(GRASP_PROBES)]
        r = math.radians(self.pose[5]) if GRASP_FRAME == "gripper" else 0.0  # the wrist angle the fingers go down at
        dx, dy = gx * math.cos(r) - gy * math.sin(r), gx * math.sin(r) + gy * math.cos(r)
        x, y = self.target["xy_mm"]
        if gx or gy:
            self.log(f"grasp offset ({GRASP_FRAME} frame) {gx:+.1f} along, {gy:+.1f} across at R {self.pose[5]:+.0f}: {dx:+.1f}, {dy:+.1f} mm in x, y")
        return x + dx, y + dy

    def descend(self, x, y) -> None:
        """Straight down to grasp height, jaws open; the caller waits for the real arm to arrive."""
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
            release = SURFACE + BIN_TOP + DROP_ABOVE_BIN
            self.move(bx, by, release + 20.0)
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
                    found = self._colour_cubes(img)
                cubes = found or []
            else:
                cubes = self._colour_cubes(img)
            self._save_debug(img, cubes)
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
            cubes = [c for c in cubes if math.dist(c["xy_mm"], self.bin_seen) > BIN_CLEAR_MM]
        if TASK == "bin" and self.want:
            cubes = [c for c in cubes if c["colour"] == self.want]
        if (not cubes and DETECTOR == "open_vocab" and BLOB_FALLBACK and self.placed == 0 and img is not None
                and object_phrase(self.instruction) != "cube" and tuple(img.shape[1::-1]) == tuple(self.LOOK_SIZE)):
            # the instruction names something the detector could not find (it never recognised a real bolt): take what
            # sits on the table that is not the bin, whatever it is; the size gate below decides if it can be held
            cubes = [b for b in table_blobs(img, self.cams) if self.bin_seen is None
                     or math.dist(b["xy_mm"], self.bin_seen) > BIN_CLEAR_MM]
            cubes = [c for c in cubes if c["short_mm"] <= MAX_GRASP_SHORT_MM]
            for c in cubes:
                c["colour"] = blob_colour(img, c)
            cubes = self._rank_by_name(img, cubes, object_phrase(self.instruction))
            self.log(f"table blobs (no named match): {len(cubes)} candidate(s): " + ", ".join(
                f"({b['xy_mm'][0]:.0f}, {b['xy_mm'][1]:.0f}) {b['long_mm']:.0f}x{b['short_mm']:.0f} mm" for b in cubes))
        too_wide = [c for c in cubes if c.get("short_mm", 0.0) > MAX_GRASP_SHORT_MM]
        if too_wide:
            self.log("too wide for the jaws, not picking: " + ", ".join(
                f"({c['xy_mm'][0]:.0f}, {c['xy_mm'][1]:.0f}) short side {c['short_mm']:.0f} mm > {MAX_GRASP_SHORT_MM:.0f}" for c in too_wide))
            cubes = [c for c in cubes if c.get("short_mm", 0.0) <= MAX_GRASP_SHORT_MM]
        if (not cubes and DETECTOR == "colour" and OPEN_VOCAB and self.placed == 0 and img is not None
                and tuple(img.shape[1::-1]) == tuple(self.LOOK_SIZE)):
            cubes = self._open_vocab(img) or []  # colour mode: the detector as a second chance, first look only
            if TASK == "bin" and self.bin_seen is not None:
                cubes = [c for c in cubes if math.dist(c["xy_mm"], self.bin_seen) > BIN_CLEAR_MM]
        for c in cubes:
            c["xy_mm"] = (c["xy_mm"][0] + TABLE_XY_OFFSET[0], c["xy_mm"][1] + TABLE_XY_OFFSET[1])
        if cubes:
            return cubes[0]
        if FALLBACK_TO_PLACEMENT and not self.used_fallback and "cube" in self._scene():
            self.used_fallback = True
            pos = self._scene()["cube"]
            pos = pos[0] if isinstance(pos[0], (list, tuple)) else pos
            self.log(f"no cube seen: using placement.yaml {pos}")
            return {"colour": "?", "xy_mm": (float(pos[0]), float(pos[1])),
                    "yaw_deg": float(pos[2]) if len(pos) > 2 else 0.0, "iou": 0.0}
        return None

    def _rank_by_name(self, img, blobs: list[dict], phrase: str, half: int = 48) -> list[dict]:
        """Order the candidate blobs by how well their close-up matches the named object. Each crop is scored on its own by
        the open-vocabulary detector, so a battery beats a nut sitting beside it. Without the detector: largest first."""
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

    def _wrist_fix(self, obs) -> bool:
        """Correct the target from the wrist camera while hovering; True if the arm is sent to re-hover."""
        img = obs["images"].get("wrist")
        colour = self.target.get("colour")
        if img is None or colour not in COLOURS + ("any",):
            return False
        try:
            loc = wrist_locate(img, obs["joints"][:6], colour, self.target["xy_mm"], self.cams)
        except Exception as e:  # noqa: BLE001
            self.log(f"wrist locate failed: {e}")
            return False
        if loc is None:
            self.log("wrist view: cube not found near the target; keeping the table-camera estimate")
            return False
        shift = math.dist(loc["xy_mm"], self.target["xy_mm"])
        self.log(f"wrist view: cube at ({loc['xy_mm'][0]:.1f}, {loc['xy_mm'][1]:.1f}) yaw {loc['yaw_deg']:+.1f}, "
                 f"{shift:.1f} mm from the target")
        yaw = self.target["yaw_deg"] if self.target.get("elongated") else loc["yaw_deg"]
        self.target = {**self.target, "xy_mm": loc["xy_mm"], "yaw_deg": yaw}
        if shift > WRIST_FIX_MM and self.fixes < WRIST_FIXES:
            self.fixes += 1
            self.hover(*self.target["xy_mm"], self.target["yaw_deg"])
            return True
        if abs(self._r(yaw) - self.pose[5]) > 0.5:  # turn the jaws to the measured yaw
            self.move(*self.pose[:3], r=self._r(yaw))
        return False

    def _next_chunk(self):
        chunk, self.queue = self.queue[:MAX_CHUNK], self.queue[MAX_CHUNK:]
        return chunk

    def _low_view(self, obs) -> None:
        """The cube seen from LOW_PEEK_ABOVE mm over it: log how far its apparent position moved against the high view."""
        img = obs["images"].get("wrist")
        if img is None:
            return
        try:
            loc = wrist_locate(img, obs["joints"][:6], self.target["colour"], self.target["xy_mm"], self.cams)
        except Exception as e:  # noqa: BLE001
            self.log(f"low wrist locate failed: {e}")
            return
        if loc is None:
            self.log("low wrist view: cube not found near the target")
            return
        tx, ty = self.target["xy_mm"]
        self.log(f"low wrist view: cube at ({loc['xy_mm'][0]:.1f}, {loc['xy_mm'][1]:.1f}), {loc['xy_mm'][0] - tx:+.1f}, "
                 f"{loc['xy_mm'][1] - ty:+.1f} mm from the tool axis (parallax between the heights; R {self.pose[5]:+.0f})")
        if LOW_PEEK_USE:
            yaw = self.target["yaw_deg"] if self.target.get("elongated") else loc["yaw_deg"]
            self.target = {**self.target, "xy_mm": loc["xy_mm"], "yaw_deg": yaw}
            self.log("re-centred on the low view")

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

    def _save_peek(self, obs) -> None:
        """The wrist (and table) view while hovering, open jaws turned to the cube: shows any x, y or yaw offset."""
        try:
            self.debug_dir.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            for cam, img in obs["images"].items():
                cv2.imwrite(str(self.debug_dir / f"{stamp}_r{self.rollout}_peek{self.placed}_{cam}.png"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            tcp = obs.get("tcp_pose")
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
            self.instruction = trial.get("instruction")
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
            elif self.phase == "look" and DRY_DESCENT:
                self.log(f"dry descent over {DRY_XY}: hover, then down at {SLOW:.0f} mm/s until the controller stops it")
                self.move(DRY_XY[0], DRY_XY[1], SURFACE + CUBE + APPROACH, r=0.0, g=0.0)
                self.hold_until_there()
                self.wait(1.0)
                z = self.pose[2]
                while z > SURFACE - 8.0:  # a few mm past the table: the controller will stop it first
                    z -= 1.0
                    self.move(DRY_XY[0], DRY_XY[1], z, r=0.0, g=0.0, speed=SLOW)
                self.phase = "done"
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
                if LOW_PEEK and self.target.get("colour") in COLOURS + ("any",):
                    self.high_xy = self.target["xy_mm"]
                    self.move(*self.target["xy_mm"], SURFACE + CUBE + LOW_PEEK_ABOVE, speed=SPEED / 2)  # straight down, no offset
                    self.wait(PEEK_SETTLE)
                    self.hold_until_there()
                    self.phase = "peek2"
                    return self._next_chunk()
                self.IMAGE_SIZE = Policy.IMAGE_SIZE
                self.descend(*self.grasp_xy())
                self.hold_until_there()
                self.phase = "grasp"
            elif self.phase == "peek2":
                self._save_peek(obs)
                self._low_view(obs)
                self.IMAGE_SIZE = Policy.IMAGE_SIZE
                self.descend(*self.grasp_xy())
                self.hold_until_there()
                self.phase = "grasp"
            elif self.phase == "grasp":
                self.pick(*self.grasp_xy(), self.target["yaw_deg"])
                self.place(self.placed)
                self.placed += 1
                self.phase = "placed" if self.placed < len(STACK_ORDER) else "done"
            if self.phase == "done" and not self.queue:
                self.wait(2.0)  # hold still; act is called again after these play

        chunk, self.queue = self.queue[:MAX_CHUNK], self.queue[MAX_CHUNK:]
        return chunk
