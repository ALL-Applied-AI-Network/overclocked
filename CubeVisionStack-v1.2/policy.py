"""CubeVisionStack-v1.2 expanded-region three-cube policy for the FANUC ER-4iA cell.

Task: detect three blue 10 mm cubes, then pick and place them into the yellow
bin one at a time during the same rollout.  Cube positions may vary anywhere
inside the approved expanded pick region.

This file is designed to replace ``policy.py`` in the official
PhysicalAI-Hackathon-2026 client after its constants have been reviewed with
the CSI operator.  It deliberately does not use simulator ground truth or a
blind placement fallback.

Safety model
------------
* One phase is planned at a time; a fresh observation is used between phases.
* Horizontal travel happens only at a conservative raised TCP height.
* Every Cartesian target is checked before it is returned.
* Each failed pickup may be retried once, at the same supervised grasp height.
* Perception gets a short observation window but motion begins only after three
  stable detections of the selected cube.  The expected number of remaining
  cubes must be visible and safely separated before every pick.
* After every drop, two observations must confirm the expected loose-cube count.
  Ambiguity, unsafe targets, and failed verification cause a hold.

The CSI operator confirmed that the standard fingers put UTOOL1 at the
fingertips, table contact is about -335 mm, the server clamps targets below
-333 mm to -333 mm and rejects targets below -334.1 mm.  Physical v1 trials
successfully picked 10-12 mm blue cubes at a supervised grasp target of
-333 mm, which is retained unchanged here.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import flange_pose, load
from cube_vision import detect_cubes, draw


HERE = Path(__file__).resolve().parent

# ----- v1.2 expanded-region task -----------------------------------------------------------------
TARGET_COLOUR = "blue"
TARGET_CUBE_COUNT = 3
BIN_XY = (430.0, 80.0)              # mm, UF0; keep the bin comfortably inside the reach limit
FIXED_WRIST_R = 0.0                  # every submitted cube is placed at yaw 0

# ----- Physical TCP heights (mm, UF0) -------------------------------------------------------------
# Confirmed with the CSI operator for the standard fingers on 2026-10-26.
TABLE_TOUCH_TCP_Z = -335.0           # standard fingertips touch the table here, approximately
GRASP_TCP_Z = -333.0                 # physically successful v1 value; operator-approved lower limit
PREGRASP_TCP_Z = -315.0              # pause above the cube before the slow descent
VERIFY_LIFT_TCP_Z = -285.0           # lift 48 mm before checking the source location
SAFE_TRANSIT_TCP_Z = -235.0          # all horizontal transport occurs at or above this height
BIN_RELEASE_TCP_Z = -255.0           # operator-confirmed clearance above the 55 mm bin rim

# ----- Observation and recovery -------------------------------------------------------------------
LOOK_FRAMES = 3
MAX_LOOK_ATTEMPTS = 10
MAX_LOOK_SPREAD_MM = 2.5
MIN_CUBE_SEPARATION_MM = 28.0
MIN_PICK_IOU = 0.55
SOURCE_VERIFY_RADIUS_MM = 18.0
MAX_PICK_RETRIES = 1
PLACE_VERIFY_FRAMES = 2
MAX_PLACE_VERIFY_ATTEMPTS = 6

# Expanded from v1.1's 80 x 80 mm box.  The farthest corner is about 485 mm radially.
# The left edge includes the strong (291, -55) blue-cube detection seen just outside v1.1's box.
# The y=-30 boundary deliberately excludes the bin-side blue clutter visible around y=-11 and above.
PICK_X_LIM = (280.0, 450.0)
PICK_Y_LIM = (-180.0, -30.0)

# ----- Motion -------------------------------------------------------------------------------------
DT = 0.1
TRANSIT_SPEED_MM_S = 60.0
APPROACH_SPEED_MM_S = 15.0
TURN_SPEED_DEG_S = 30.0
GRIPPER_WAIT_S = 1.5
MAX_CHUNK = 25

DOWN_W = -180.0
DOWN_P = 0.0
PARK_XYZ = (260.0, -300.0, -180.0)
INSPECTION_XYZ = (330.0, -250.0, SAFE_TRANSIT_TCP_Z)

# Conservative software envelope.  Invalid targets cause a safe hold; they are not silently clipped.
X_LIM = (200.0, 500.0)
Y_LIM = (-350.0, 250.0)
Z_LIM = (-333.0, 50.0)               # server clamps below -333 and rejects below -334.1
RADIAL_LIM_MM = 500.0


def _angle_delta(target: float, current: float) -> float:
    """Shortest signed angular change in degrees."""
    return (target - current + 180.0) % 360.0 - 180.0


def _nearest_equivalent(target: float, current: float) -> float:
    """Representation of target nearest current (for FANUC's +/-180 degree equivalence)."""
    return current + _angle_delta(target, current)


def _tip_from_joints(joints) -> tuple[float, float, float, float, float, float]:
    """Mock-only fallback when tcp_pose is unavailable; real runs provide tcp_pose."""
    T = flange_pose(joints)
    tcp = T[:3, 3] + T[:3, 0] * 0.293
    return (
        float(tcp[0] * 1000),
        float(tcp[1] * 1000),
        float(tcp[2] * 1000 - 330.0),
        DOWN_W,
        DOWN_P,
        0.0,
    )


class UnsafeTarget(ValueError):
    """Raised internally before an out-of-envelope target can be returned."""


class Policy:
    CAMERAS = ["table", "wrist"]
    IMAGE_SIZE = (1280, 720)
    IMAGE_FORMAT = "jpeg"
    ACTION = "tcp"
    PREFETCH = False
    DEPTH = []

    def __init__(self, checkpoint: str | None = None):
        del checkpoint
        self.cams = load()
        self.debug_dir = HERE / "debug_cubevisionstack_v1_2"
        self.reset()

    def reset(self) -> None:
        self.queue: list[list[float]] = []
        self.pose: list[float] | None = None
        self.phase = "start"
        self.rollout = None
        self.look_samples: list[dict] = []
        self.look_attempts = 0
        self.target: dict | None = None
        self.placed_count = 0
        self.pick_retries = 0
        self.place_verify_attempts = 0
        self.place_verify_matches = 0
        self.failure_reason: str | None = None
        self.last_wrist_blue_fraction = 0.0

    def log(self, message: str) -> None:
        print(f"[v1.2] {message}", flush=True)

    # ----- State and safety ------------------------------------------------------------------------
    def _sync_pose(self, obs: dict) -> None:
        """Resynchronise with the real TCP whenever a phase has finished."""
        tcp = obs.get("tcp_pose")
        if tcp is not None and len(tcp) >= 6:
            g = float(obs.get("joints", [0.0] * 7)[6]) if len(obs.get("joints", [])) >= 7 else 0.0
            self.pose = [*(float(v) for v in tcp[:6]), g]
            return
        joints = obs.get("joints")
        if joints is None:
            joints = [0.0] * 7
        x, y, z, w, p, r = _tip_from_joints(joints[:6])
        g = float(joints[6]) if len(joints) >= 7 else 0.0
        self.pose = [x, y, z, w, p, r, g]

    @staticmethod
    def _validate_xyz(x: float, y: float, z: float) -> None:
        if not (X_LIM[0] <= x <= X_LIM[1]):
            raise UnsafeTarget(f"x={x:.1f} outside {X_LIM}")
        if not (Y_LIM[0] <= y <= Y_LIM[1]):
            raise UnsafeTarget(f"y={y:.1f} outside {Y_LIM}")
        if not (Z_LIM[0] <= z <= Z_LIM[1]):
            raise UnsafeTarget(f"z={z:.1f} outside {Z_LIM}")
        if math.hypot(x, y) > RADIAL_LIM_MM:
            raise UnsafeTarget(f"radial reach {math.hypot(x, y):.1f} mm exceeds {RADIAL_LIM_MM:.1f} mm")

    def _hold_row(self) -> list[float]:
        assert self.pose is not None
        return list(self.pose)

    def _wait(self, seconds: float) -> None:
        assert self.pose is not None
        self.queue.extend(self._hold_row() for _ in range(max(1, round(seconds / DT))))

    def _move(
        self,
        x: float,
        y: float,
        z: float,
        *,
        w: float | None = None,
        p: float | None = None,
        r: float | None = None,
        gripper: float | None = None,
        speed: float = TRANSIT_SPEED_MM_S,
    ) -> None:
        assert self.pose is not None
        self._validate_xyz(float(x), float(y), float(z))

        start = list(self.pose)
        w = start[3] if w is None else _nearest_equivalent(float(w), start[3])
        p = start[4] if p is None else _nearest_equivalent(float(p), start[4])
        r = start[5] if r is None else _nearest_equivalent(float(r), start[5])
        g = start[6] if gripper is None else float(gripper)

        distance = math.dist(start[:3], (x, y, z))
        max_angle = max(abs(w - start[3]), abs(p - start[4]), abs(r - start[5]))
        steps = max(1, math.ceil(distance / max(speed * DT, 1e-6)),
                    math.ceil(max_angle / max(TURN_SPEED_DEG_S * DT, 1e-6)))

        # Gripper changes only on the final row, after the arm has reached the target.
        for index in range(1, steps + 1):
            alpha = index / steps
            row_gripper = g if index == steps else start[6]
            row = [
                start[0] + alpha * (x - start[0]),
                start[1] + alpha * (y - start[1]),
                start[2] + alpha * (z - start[2]),
                start[3] + alpha * (w - start[3]),
                start[4] + alpha * (p - start[4]),
                start[5] + alpha * (r - start[5]),
                row_gripper,
            ]
            self._validate_xyz(*row[:3])
            self.queue.append([float(v) for v in row])
        self.pose = list(self.queue[-1])

    def _safe_failure(self, reason: str) -> None:
        self.failure_reason = reason
        self.phase = "failed"
        self.queue.clear()
        self.log(f"SAFE HOLD: {reason}")

    # ----- Perception ------------------------------------------------------------------------------
    def _save_debug(self, label: str, image: np.ndarray, cubes: list[dict]) -> None:
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{stamp}_r{self.rollout}_{label}.jpg"
            cv2.imwrite(str(self.debug_dir / filename), draw(image, cubes, self.cams))
        except Exception as exc:  # Debug output must never move or stop the robot.
            self.log(f"debug image not saved: {exc}")

    def _table_cubes(self, obs: dict, label: str) -> list[dict]:
        image = (obs.get("images") or {}).get("table")
        if image is None:
            self.log("table image unavailable")
            return []
        cubes = detect_cubes(image, (TARGET_COLOUR,), self.cams, min_iou=MIN_PICK_IOU)
        self._save_debug(label, image, cubes)
        return cubes

    @staticmethod
    def _wrist_blue_fraction(obs: dict) -> float:
        """Advisory only: fraction of saturated blue pixels in the central wrist view."""
        image = (obs.get("images") or {}).get("wrist")
        if image is None or image.size == 0:
            return 0.0
        h, w = image.shape[:2]
        crop = image[h // 3:h, w // 4:3 * w // 4]
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([95, 55, 35], np.uint8), np.array([135, 255, 255], np.uint8))
        return float(np.count_nonzero(mask) / max(mask.size, 1))

    def _stable_target(self, obs: dict) -> tuple[str, dict | None]:
        self.look_attempts += 1
        cubes = self._table_cubes(obs, f"look_{self.look_attempts}")
        candidates = [
            cube for cube in cubes
            if PICK_X_LIM[0] <= cube["xy_mm"][0] <= PICK_X_LIM[1]
            and PICK_Y_LIM[0] <= cube["xy_mm"][1] <= PICK_Y_LIM[1]
        ]
        detected = ", ".join(
            f"({cube['xy_mm'][0]:.1f}, {cube['xy_mm'][1]:.1f}; IoU {cube['iou']:.2f})"
            for cube in cubes
        ) or "none"
        self.log(
            f"look {self.look_attempts}/{MAX_LOOK_ATTEMPTS}: detected {len(cubes)} blue cube(s) "
            f"[{detected}]; {len(candidates)} inside pick box"
        )
        expected = TARGET_CUBE_COUNT - self.placed_count
        if len(candidates) != expected:
            self.look_samples.clear()
            self.log(f"waiting for exactly {expected} remaining cube(s) inside the pick box")
            if self.look_attempts >= MAX_LOOK_ATTEMPTS:
                return "failed", None
            return "waiting", None

        separations = [
            math.dist(a["xy_mm"], b["xy_mm"])
            for index, a in enumerate(candidates)
            for b in candidates[index + 1:]
        ]
        if separations and min(separations) < MIN_CUBE_SEPARATION_MM:
            self.look_samples.clear()
            self.log(
                f"closest cubes are {min(separations):.1f} mm apart; "
                f"minimum separation is {MIN_CUBE_SEPARATION_MM:.1f} mm"
            )
            if self.look_attempts >= MAX_LOOK_ATTEMPTS:
                return "failed", None
            return "waiting", None

        if self.look_samples:
            anchor = self.look_samples[-1]["xy_mm"]
            selected = min(candidates, key=lambda cube: math.dist(cube["xy_mm"], anchor))
        else:
            # Start with the highest-confidence isolated candidate.  Coordinate tie-breakers keep equal-IoU
            # scenes deterministic.  Later frames track this same cube by proximity instead of switching targets.
            selected = max(
                candidates,
                key=lambda cube: (cube["iou"], -cube["xy_mm"][0], -cube["xy_mm"][1]),
            )

        self.look_samples.append(selected)
        if len(self.look_samples) < LOOK_FRAMES:
            if self.look_attempts >= MAX_LOOK_ATTEMPTS:
                return "failed", None
            return "waiting", None

        samples = self.look_samples[-LOOK_FRAMES:]
        xy = np.asarray([sample["xy_mm"] for sample in samples], dtype=float)
        centre = np.median(xy, axis=0)
        spread = float(np.max(np.linalg.norm(xy - centre, axis=1)))
        self.look_samples.clear()
        if spread > MAX_LOOK_SPREAD_MM:
            self.log(f"unstable cube location: {spread:.1f} mm spread")
            if self.look_attempts >= MAX_LOOK_ATTEMPTS:
                return "failed", None
            return "waiting", None

        best = max(samples, key=lambda sample: sample["iou"])
        self.log(
            f"cube {self.placed_count + 1}/{TARGET_CUBE_COUNT} stable at detected "
            f"({centre[0]:.1f}, {centre[1]:.1f})"
        )
        self.look_attempts = 0
        return "ready", {**best, "xy_mm": (float(centre[0]), float(centre[1])), "yaw_deg": 0.0}

    # ----- Verification and recovery ---------------------------------------------------------------
    def _source_cube_present(self, obs: dict) -> tuple[bool | None, float | None]:
        assert self.target is not None
        image = (obs.get("images") or {}).get("table")
        if image is None or image.size == 0:
            return None, None
        cubes = self._table_cubes(obs, "verify_pick")
        source = self.target["xy_mm"]
        distances = [math.dist(cube["xy_mm"], source) for cube in cubes]
        nearest = min(distances) if distances else None
        return nearest is not None and nearest <= SOURCE_VERIFY_RADIUS_MM, nearest

    def _placement_status(self, obs: dict) -> tuple[str, int, int]:
        """Confirm the expected number of cubes remains loose after the current drop."""
        self.place_verify_attempts += 1
        cubes = self._table_cubes(obs, f"verify_place_{self.place_verify_attempts}")
        loose = [cube for cube in cubes if math.dist(cube["xy_mm"], BIN_XY) > 35.0]
        expected = TARGET_CUBE_COUNT - self.placed_count - 1
        actual = len(loose)
        if actual == expected:
            self.place_verify_matches += 1
        else:
            self.place_verify_matches = 0
        self.log(
            f"placement verification {self.place_verify_attempts}/{MAX_PLACE_VERIFY_ATTEMPTS}: "
            f"expected {expected} loose cube(s), detected {actual}; "
            f"stable confirmations {self.place_verify_matches}/{PLACE_VERIFY_FRAMES}"
        )
        if self.place_verify_matches >= PLACE_VERIFY_FRAMES:
            return "ready", actual, expected
        if self.place_verify_attempts >= MAX_PLACE_VERIFY_ATTEMPTS:
            return "failed", actual, expected
        return "waiting", actual, expected

    def _plan_retry(self) -> None:
        assert self.pose is not None
        self.pick_retries += 1
        self.log(f"pickup failed; planning one retry ({self.pick_retries}/{MAX_PICK_RETRIES})")
        self._move(*INSPECTION_XYZ, gripper=0.0, speed=APPROACH_SPEED_MM_S)
        self._wait(GRIPPER_WAIT_S)
        self._move(*PARK_XYZ, w=DOWN_W, p=DOWN_P, r=FIXED_WRIST_R, gripper=0.0)
        self.target = None
        self.look_samples.clear()
        self.look_attempts = 0
        self.phase = "look"

    # ----- Main policy ----------------------------------------------------------------------------
    def act(self, obs: dict):
        if self.pose is None:
            self.rollout = (obs.get("trial") or {}).get("index")
            self._sync_pose(obs)
            assert self.pose is not None
            self.log(
                f"rollout {self.rollout}: start TCP "
                f"({self.pose[0]:.1f}, {self.pose[1]:.1f}, {self.pose[2]:.1f}, "
                f"{self.pose[3]:.1f}, {self.pose[4]:.1f}, {self.pose[5]:.1f})"
            )
            self.log(
                f"calibrated grasp Z {GRASP_TCP_Z:.1f}; table-touch TCP near "
                f"{TABLE_TOUCH_TCP_Z:.1f} with standard fingers"
            )
            self.log(
                f"task: place {TARGET_CUBE_COUNT} blue 10 mm cubes into the bin; "
                f"pick region x={PICK_X_LIM}, y={PICK_Y_LIM}"
            )

        if self.queue:
            chunk, self.queue = self.queue[:MAX_CHUNK], self.queue[MAX_CHUNK:]
            return chunk

        # A phase has finished.  Use the real observed pose before planning the next phase.
        self._sync_pose(obs)

        try:
            if self.phase == "start":
                self.log("phase: park for an unobstructed table image")
                self._move(*PARK_XYZ, w=DOWN_W, p=DOWN_P, r=FIXED_WRIST_R, gripper=0.0)
                self._wait(0.5)
                self.phase = "look"

            elif self.phase == "look":
                status, target = self._stable_target(obs)
                if status == "waiting":
                    self._wait(0.25)
                elif status == "failed" or target is None:
                    remaining = TARGET_CUBE_COUNT - self.placed_count
                    self._safe_failure(
                        f"expected {remaining} remaining blue cube(s) and a stable next target "
                        "inside the approved v1.2 pick box"
                    )
                else:
                    self.target = target
                    x, y = target["xy_mm"]
                    self.log(
                        f"verified cube {self.placed_count + 1}/{TARGET_CUBE_COUNT} at "
                        f"x={x:.1f}, y={y:.1f}, IoU={target['iou']:.2f}; "
                        f"commanded grasp Z will be {GRASP_TCP_Z:.1f}"
                    )
                    self.phase = "approach"
                    self._wait(0.1)

            elif self.phase == "approach":
                assert self.target is not None
                x, y = self.target["xy_mm"]
                self.log("phase: raised transit and pre-grasp approach")
                self._move(self.pose[0], self.pose[1], SAFE_TRANSIT_TCP_Z,
                           w=DOWN_W, p=DOWN_P, r=FIXED_WRIST_R, gripper=0.0)
                self._move(x, y, SAFE_TRANSIT_TCP_Z, w=DOWN_W, p=DOWN_P, r=FIXED_WRIST_R, gripper=0.0)
                self._move(x, y, PREGRASP_TCP_Z, w=DOWN_W, p=DOWN_P, r=FIXED_WRIST_R,
                           gripper=0.0, speed=APPROACH_SPEED_MM_S)
                self.phase = "grasp"

            elif self.phase == "grasp":
                assert self.target is not None
                x, y = self.target["xy_mm"]
                self.log(f"phase: supervised grasp at TCP z={GRASP_TCP_Z:.1f}")
                self._move(x, y, GRASP_TCP_Z, w=DOWN_W, p=DOWN_P, r=FIXED_WRIST_R,
                           gripper=0.0, speed=APPROACH_SPEED_MM_S)
                self._wait(0.5)
                self._move(x, y, GRASP_TCP_Z, gripper=1.0, speed=APPROACH_SPEED_MM_S)
                self._wait(GRIPPER_WAIT_S)
                self._move(x, y, VERIFY_LIFT_TCP_Z, gripper=1.0, speed=APPROACH_SPEED_MM_S)
                self.phase = "inspection_move"

            elif self.phase == "inspection_move":
                self.log("phase: move clear of source for visual pickup verification")
                self._move(self.pose[0], self.pose[1], SAFE_TRANSIT_TCP_Z, gripper=1.0,
                           speed=APPROACH_SPEED_MM_S)
                self._move(*INSPECTION_XYZ, gripper=1.0)
                self._wait(0.5)
                self.phase = "verify_pick"

            elif self.phase == "verify_pick":
                assert self.target is not None
                source_present, nearest = self._source_cube_present(obs)
                self.last_wrist_blue_fraction = self._wrist_blue_fraction(obs)
                nearest_text = "none detected" if nearest is None else f"{nearest:.1f} mm"
                self.log(
                    f"pickup verification: nearest blue cube to source={nearest_text}; "
                    f"wrist blue fraction={self.last_wrist_blue_fraction:.4f} (advisory)"
                )
                if source_present is None:
                    self._safe_failure("table image unavailable during pickup verification")
                elif source_present:
                    if self.pick_retries < MAX_PICK_RETRIES:
                        self._plan_retry()
                    else:
                        self._safe_failure("cube remained at source after the allowed retry")
                else:
                    self.log("pickup verified by source clearance; proceeding to bin")
                    self.phase = "place_approach"
                    self._wait(0.1)

            elif self.phase == "place_approach":
                self.log("phase: raised transit to bin")
                self._move(self.pose[0], self.pose[1], SAFE_TRANSIT_TCP_Z, gripper=1.0)
                self._move(BIN_XY[0], BIN_XY[1], SAFE_TRANSIT_TCP_Z,
                           w=DOWN_W, p=DOWN_P, r=FIXED_WRIST_R, gripper=1.0)
                self.phase = "place"

            elif self.phase == "place":
                self.log(f"phase: lower and release at bin TCP z={BIN_RELEASE_TCP_Z:.1f}")
                self._move(BIN_XY[0], BIN_XY[1], BIN_RELEASE_TCP_Z,
                           gripper=1.0, speed=APPROACH_SPEED_MM_S)
                self._wait(0.5)
                self._move(BIN_XY[0], BIN_XY[1], BIN_RELEASE_TCP_Z,
                           gripper=0.0, speed=APPROACH_SPEED_MM_S)
                self._wait(GRIPPER_WAIT_S)
                self._move(BIN_XY[0], BIN_XY[1], SAFE_TRANSIT_TCP_Z,
                           gripper=0.0, speed=APPROACH_SPEED_MM_S)
                self.phase = "post_place"

            elif self.phase == "post_place":
                self.log("phase: park and verify remaining loose cubes")
                self._move(*PARK_XYZ, w=DOWN_W, p=DOWN_P, r=FIXED_WRIST_R, gripper=0.0)
                self._wait(0.5)
                self.place_verify_attempts = 0
                self.place_verify_matches = 0
                self.phase = "verify_place"

            elif self.phase == "verify_place":
                status, actual, expected = self._placement_status(obs)
                if status == "waiting":
                    self._wait(0.25)
                elif status == "failed":
                    self._safe_failure(
                        f"placement verification expected {expected} loose cube(s), detected {actual}"
                    )
                else:
                    self.placed_count += 1
                    self.log(f"verified placement {self.placed_count}/{TARGET_CUBE_COUNT}")
                    self.target = None
                    self.pick_retries = 0
                    self.look_samples.clear()
                    self.look_attempts = 0
                    self.place_verify_attempts = 0
                    self.place_verify_matches = 0
                    if self.placed_count >= TARGET_CUBE_COUNT:
                        self.log("TASK COMPLETE: all three cubes verified in the bin; holding")
                        self.phase = "done"
                    else:
                        self.log("returning to perception for the next cube")
                        self.phase = "look"
                        self._wait(0.25)

            elif self.phase in ("done", "failed"):
                self._wait(1.0)

            else:
                self._safe_failure(f"unknown policy phase {self.phase!r}")

        except UnsafeTarget as exc:
            self._safe_failure(f"unsafe target rejected before transmission: {exc}")

        if not self.queue:
            self.queue.append(self._hold_row())
        chunk, self.queue = self.queue[:MAX_CHUNK], self.queue[MAX_CHUNK:]
        return chunk

