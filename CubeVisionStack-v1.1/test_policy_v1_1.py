"""Regression checks for CubeVisionStack-v1.1's three-cube safety gates.

Run this file after copying the v1.1 files into an official client checkout.
No robot or network connection is used.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from policy import BIN_XY, MAX_LOOK_ATTEMPTS, Policy


def detection(x: float, y: float, iou: float = 0.8) -> dict:
    return {
        "colour": "blue",
        "xy_mm": (x, y),
        "yaw_deg": 0.0,
        "iou": iou,
        "px": (100.0, 100.0),
        "area_px": 50,
    }


def first_layout() -> list[dict]:
    return [
        detection(370.0, -130.0),
        detection(400.0, -100.0),
        detection(430.0, -70.0),
    ]


class ThreeCubeGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = Policy()
        self.policy.rollout = 0
        self.policy._save_debug = lambda *_args, **_kwargs: None
        self.obs = {
            "images": {"table": np.zeros((8, 8, 3), dtype=np.uint8)},
            "tcp_pose": [300.0, 0.0, -90.0, -180.0, 0.0, 0.0],
            "joints": [0.0] * 7,
        }

    def test_three_stable_scene_observations_select_first_planned_cube(self) -> None:
        frames = [first_layout(), first_layout(), first_layout()]
        with patch("policy.detect_cubes", side_effect=frames):
            self.assertEqual(self.policy._stable_target(self.obs)[0], "waiting")
            self.assertEqual(self.policy._stable_target(self.obs)[0], "waiting")
            status, target = self.policy._stable_target(self.obs)
        self.assertEqual(status, "ready")
        self.assertEqual(target["xy_mm"], (370.0, -130.0))

    def test_missing_cube_never_starts_motion(self) -> None:
        two = first_layout()[1:]
        frames = [two for _ in range(MAX_LOOK_ATTEMPTS)]
        with patch("policy.detect_cubes", side_effect=frames):
            statuses = [self.policy._stable_target(self.obs)[0] for _ in frames]
        self.assertTrue(all(status == "waiting" for status in statuses[:-1]))
        self.assertEqual(statuses[-1], "failed")

    def test_after_first_placement_two_remaining_cubes_are_required(self) -> None:
        self.policy.placed_count = 1
        two = first_layout()[1:]
        frames = [two, two, two]
        with patch("policy.detect_cubes", side_effect=frames):
            statuses = [self.policy._stable_target(self.obs)[0] for _ in frames]
        self.assertEqual(statuses, ["waiting", "waiting", "ready"])

    def test_next_cube_must_be_near_its_submitted_position(self) -> None:
        displaced = [
            detection(390.0, -130.0),
            detection(400.0, -100.0),
            detection(430.0, -70.0),
        ]
        frames = [displaced for _ in range(MAX_LOOK_ATTEMPTS)]
        with patch("policy.detect_cubes", side_effect=frames):
            statuses = [self.policy._stable_target(self.obs)[0] for _ in frames]
        self.assertEqual(statuses[-1], "failed")

    def test_drop_requires_two_confirmations_of_remaining_count(self) -> None:
        loose_and_binned = [
            detection(400.0, -100.0),
            detection(430.0, -70.0),
            detection(*BIN_XY),
        ]
        frames = [loose_and_binned, loose_and_binned]
        with patch("policy.detect_cubes", side_effect=frames):
            first = self.policy._placement_status(self.obs)
            second = self.policy._placement_status(self.obs)
        self.assertEqual(first[0], "waiting")
        self.assertEqual(second, ("ready", 2, 2))

    def test_final_drop_requires_zero_loose_cubes(self) -> None:
        self.policy.placed_count = 2
        frames = [[detection(*BIN_XY)], [detection(*BIN_XY)]]
        with patch("policy.detect_cubes", side_effect=frames):
            self.policy._placement_status(self.obs)
            status = self.policy._placement_status(self.obs)
        self.assertEqual(status, ("ready", 0, 0))

    def test_verified_first_drop_loops_back_for_second_cube(self) -> None:
        self.policy.pose = list(self.obs["tcp_pose"]) + [0.0]
        self.policy.phase = "verify_place"
        self.policy.target = detection(370.0, -130.0)
        with patch.object(self.policy, "_placement_status", return_value=("ready", 2, 2)):
            self.policy.act(self.obs)
        self.assertEqual(self.policy.placed_count, 1)
        self.assertEqual(self.policy.phase, "look")

    def test_only_third_verified_drop_completes_task(self) -> None:
        self.policy.pose = list(self.obs["tcp_pose"]) + [0.0]
        self.policy.phase = "verify_place"
        self.policy.placed_count = 2
        self.policy.target = detection(430.0, -70.0)
        with patch.object(self.policy, "_placement_status", return_value=("ready", 0, 0)):
            self.policy.act(self.obs)
        self.assertEqual(self.policy.placed_count, 3)
        self.assertEqual(self.policy.phase, "done")


if __name__ == "__main__":
    unittest.main()
