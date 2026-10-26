"""Automated safety and behavior tests for blue best-candidate selection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from mock_cell import validate_tcp
from policy import (
    LOOK_FRAMES,
    MAX_LOOK_ATTEMPTS,
    MAX_PICK_RETRIES,
    MIN_PICK_IOU,
    PICK_X_LIM,
    PICK_Y_LIM,
    Policy,
)


POSITIONS = ((390.0, -105.0), (345.0, -55.0), (435.0, -55.0),
             (345.0, -155.0), (435.0, -155.0))


def detection(x: float, y: float, iou: float = 0.80) -> dict:
    return {"colour": "blue", "xy_mm": (x, y), "yaw_deg": 0.0,
            "iou": iou, "px": (100.0, 100.0), "area_px": 50}


class BlueBestCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = Policy()
        self.policy.rollout = 0
        self.policy._save_debug = lambda *_args, **_kwargs: None
        self.obs = {"images": {"table": np.zeros((8, 8, 3), dtype=np.uint8)},
                    "tcp_pose": [300.0, 0.0, -90.0, -180.0, 0.0, 0.0],
                    "joints": [0.0] * 7}

    def stable(self, frames):
        with patch("policy.detect_cubes", side_effect=frames):
            result = None
            for _ in frames:
                result = self.policy._stable_target(self.obs)
        return result

    def test_each_approved_position_is_selected_from_live_detections(self) -> None:
        for xy in POSITIONS:
            with self.subTest(xy=xy):
                self.policy.reset()
                status, target = self.stable([[detection(*xy)] for _ in range(LOOK_FRAMES)])
                self.assertEqual(status, "ready")
                self.assertEqual(target["xy_mm"], xy)

    def test_detected_xy_not_declared_xy_drives_motion(self) -> None:
        live_xy = (402.0, -118.0)  # rollout 0 declares (390, -105)
        self.policy.target = detection(*live_xy)
        self.policy.pose = list(self.obs["tcp_pose"]) + [0.0]
        self.policy.phase = "approach"
        actions = self.policy.act(self.obs)
        all_actions = actions + self.policy.queue
        self.assertTrue(any(abs(row[0] - live_xy[0]) < 1e-9 and
                            abs(row[1] - live_xy[1]) < 1e-9 for row in all_actions))
        self.assertFalse(any(abs(row[0] - 390.0) < 1e-9 and
                             abs(row[1] + 105.0) < 1e-9 for row in all_actions))

    def test_missing_detection_has_no_placement_coordinate_fallback(self) -> None:
        status, target = self.stable([[] for _ in range(MAX_LOOK_ATTEMPTS)])
        self.assertEqual((status, target), ("failed", None))

    def test_multiple_blue_cubes_selects_most_positionally_confirmed(self) -> None:
        frames = [
            [detection(388, -105, 0.90), detection(430, -145, 0.76)],
            [detection(392, -105, 0.90), detection(430, -145, 0.77)],
            [detection(390, -105, 0.90), detection(430, -145, 0.75)],
        ]
        status, target = self.stable(frames)
        self.assertEqual(status, "ready")
        self.assertEqual(target["xy_mm"], (430.0, -145.0))

    def test_unstable_blue_cube_does_not_block_a_stable_blue_cube(self) -> None:
        frames = [
            [detection(350, -80), detection(420, -140)],
            [detection(360, -80), detection(420, -140)],
            [detection(345, -80), detection(420, -140)],
        ]
        status, target = self.stable(frames)
        self.assertEqual(status, "ready")
        self.assertEqual(target["xy_mm"], (420.0, -140.0))

    def test_outside_pick_envelope_safe_hold(self) -> None:
        self.assertEqual(self.stable([[detection(PICK_X_LIM[1] + 1, -105)]
                                      for _ in range(MAX_LOOK_ATTEMPTS)]), ("failed", None))

    def test_unstable_detection_safe_hold(self) -> None:
        frames = [[detection(380 if i % 2 else 390, -105)] for i in range(MAX_LOOK_ATTEMPTS)]
        self.assertEqual(self.stable(frames), ("failed", None))

    def test_low_confidence_detection_safe_hold(self) -> None:
        frames = [[detection(390, -105, MIN_PICK_IOU - 0.01)] for _ in range(MAX_LOOK_ATTEMPTS)]
        self.assertEqual(self.stable(frames), ("failed", None))

    def test_successful_post_place_source_removal_completes_rollout(self) -> None:
        self.policy.pose = list(self.obs["tcp_pose"]) + [0.0]
        self.policy.phase = "verify_place"
        self.policy.target = detection(390, -105)
        with patch.object(self.policy, "_placement_status", return_value=("ready", 0, 0)):
            self.policy.act(self.obs)
        self.assertEqual(self.policy.placed_count, 1)
        self.assertEqual(self.policy.phase, "done")

    def test_cube_remaining_at_source_triggers_no_more_than_one_retry(self) -> None:
        self.policy.pose = list(self.obs["tcp_pose"]) + [0.0]
        self.policy.phase = "verify_pick"
        self.policy.target = detection(390, -105)
        with patch.object(self.policy, "_source_cube_present", return_value=(True, 0.0)):
            self.policy.act(self.obs)
        self.assertEqual(self.policy.pick_retries, MAX_PICK_RETRIES)
        self.policy.queue.clear()
        self.policy.phase = "verify_pick"
        self.policy.target = detection(390, -105)
        with patch.object(self.policy, "_source_cube_present", return_value=(True, 0.0)):
            self.policy.act(self.obs)
        self.assertEqual(self.policy.pick_retries, MAX_PICK_RETRIES)
        self.assertEqual(self.policy.phase, "failed")

    def test_generated_pick_and_place_actions_pass_official_mock_validator(self) -> None:
        self.policy.pose = list(self.obs["tcp_pose"]) + [0.0]
        self.policy.target = detection(435, -155)
        for phase in ("approach", "grasp", "inspection_move", "place_approach", "place", "post_place"):
            self.policy.phase = phase
            self.policy.queue.clear()
            rows = self.policy.act(self.obs) + self.policy.queue
            for start in range(0, len(rows), 100):
                validate_tcp(rows[start:start + 100], 0.1)

    def test_bin_is_outside_source_selection_region(self) -> None:
        self.assertFalse(PICK_X_LIM[0] <= 430 <= PICK_X_LIM[1] and
                         PICK_Y_LIM[0] <= 80 <= PICK_Y_LIM[1])


if __name__ == "__main__":
    unittest.main()
