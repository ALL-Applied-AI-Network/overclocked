"""Check portable model discovery without CUDA or development-machine paths."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from perceive import _find_nano


class RuntimeConfigTests(unittest.TestCase):
    def test_explicit_snapshot_works_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "downloaded_model"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(
                json.dumps({"model_type": "cosmos3"})
            )
            with patch.dict(
                os.environ, {"COSMOS3_NANO_PATH": str(checkpoint)}, clear=True
            ):
                self.assertEqual(_find_nano(), checkpoint)

    def test_shared_hub_cache_is_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "shared_cache"
            checkpoint = hub / "models--nvidia--Cosmos3-Nano/snapshots/revision"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}")
            with patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}, clear=True):
                self.assertEqual(_find_nano(), checkpoint)

    def test_incomplete_download_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"COSMOS3_NANO_PATH": directory}, clear=True):
                with self.assertRaisesRegex(SystemExit, "missing config.json"):
                    _find_nano()


class LayoutSeedTests(unittest.TestCase):
    def run_scene(self, seed=None):
        import subprocess
        import sys

        env = dict(os.environ)
        env.pop("CUBE_LAYOUT_SEED", None)
        if seed is not None:
            env["CUBE_LAYOUT_SEED"] = seed
        return subprocess.run(
            [sys.executable, "-B", "-c",
             "import json, scene_config as C; "
             "print(json.dumps([C.CUBE_LAYOUT_SEED, C.SORT_CUBES]))"],
            cwd=Path(__file__).resolve().parent,
            env=env, capture_output=True, text=True,
        )

    def layout(self, seed=None):
        result = self.run_scene(seed)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_unset_seed_preserves_default(self):
        default = self.layout()
        self.assertEqual(default[0], 20261009)
        self.assertEqual(default, self.layout("20261009"))

    def test_alternate_seed_is_repeatable_and_keeps_yellow_positions(self):
        default = self.layout()
        alternate = self.layout("20261010")
        self.assertEqual(alternate, self.layout("20261010"))
        self.assertEqual(alternate[0], 20261010)
        self.assertEqual(alternate[1][:2], default[1][:2])
        self.assertNotEqual(alternate[1][2:], default[1][2:])

    def test_invalid_seed_has_clear_error(self):
        for value in ("", "banana", "1.5"):
            with self.subTest(value=value):
                result = self.run_scene(value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "CUBE_LAYOUT_SEED must be an integer", result.stderr
                )


class IsaacSeedBridgeTests(unittest.TestCase):
    def test_bridge_sets_seed_before_script_import_and_preserves_arguments(self):
        import subprocess
        import sys

        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory) / "probe.py"
            probe.write_text(
                "import json, os, sys\n"
                "print(json.dumps([os.environ['CUBE_LAYOUT_SEED'], sys.argv[1:]]))\n"
            )
            env = dict(os.environ, CUBE_LAYOUT_SEED="999")
            result = subprocess.run(
                [sys.executable, "-B", str(root / "isaac_seed_entrypoint.py"),
                 "20261010", str(probe), "--out", "folder with spaces"],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(result.stdout),
                ["20261010", ["--out", "folder with spaces"]],
            )

    def test_bridge_rejects_invalid_seed_before_loading_script(self):
        import subprocess
        import sys

        bridge = Path(__file__).resolve().parent / "isaac_seed_entrypoint.py"
        result = subprocess.run(
            [sys.executable, "-B", str(bridge), "banana", "missing_script.py"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUBE_LAYOUT_SEED must be an integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
