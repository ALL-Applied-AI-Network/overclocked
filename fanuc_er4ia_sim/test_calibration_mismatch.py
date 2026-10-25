import copy
import json
import tempfile
import os
import subprocess
import sys
import unittest
from pathlib import Path
from calibration_mismatch import configuration, assumed_calibration, environment, write_calibrations


class CalibrationMismatchTests(unittest.TestCase):
    def setUp(self):
        self.true = dict(camera_to_world=[[1.,0.,0.,0.], [0.,1.,0.,0.],
                                         [0.,0.,1.,0.], [0.2,0.3,1.,1.]],
                         resolution=[1920,1080], table_top_z=0.8949)

    def test_default_preserves_calibration_and_input(self):
        before = copy.deepcopy(self.true)
        self.assertEqual(assumed_calibration(self.true, configuration({})), before)
        self.assertEqual(self.true, before)

    def test_translation_units_and_direction(self):
        for mm in (1,2,3):
            control = configuration({'CAMERA_TRANSLATION_MM': str(mm)})
            assumed = assumed_calibration(self.true, control)
            self.assertAlmostEqual(assumed['camera_to_world'][3][0], .2 + mm/1000)
            self.assertEqual(assumed['camera_to_world'][:3], self.true['camera_to_world'][:3])
            self.assertEqual(assumed['camera_to_world'][3][1:], [.3,1.,1.])
            self.assertEqual(configuration(environment(control)), control)

    def test_rotation_sign_pivot_and_orthonormality(self):
        control = configuration({'CAMERA_ROTATION_DEG': '.5'})
        matrix = assumed_calibration(self.true, control)['camera_to_world']
        self.assertGreater(matrix[0][1], 0)
        self.assertLess(matrix[1][0], 0)
        self.assertEqual(matrix[3], self.true['camera_to_world'][3])
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(sum(matrix[i][k]*matrix[j][k] for k in range(3)), float(i == j))

    def test_invalid_controls(self):
        for value in ('nan','inf','-1','4','abc',''):
            with self.assertRaises(ValueError):
                configuration({'CAMERA_TRANSLATION_MM': value})
        with self.assertRaises(ValueError):
            configuration({'CAMERA_TRANSLATION_MM':'1','CAMERA_ROTATION_DEG':'.5'})
        with self.assertRaises(ValueError):
            configuration({'CAMERA_ROTATION_DEG':'1'})

    def test_separate_true_and_assumed_files(self):
        with tempfile.TemporaryDirectory() as folder:
            control = configuration({'CAMERA_TRANSLATION_MM':'1'})
            write_calibrations(folder, self.true, control)
            root = Path(folder)
            self.assertEqual(json.loads((root/'camera_true.json').read_text()), self.true)
            self.assertNotEqual(json.loads((root/'camera.json').read_text()), self.true)
            self.assertEqual(json.loads((root/'camera_mismatch.json').read_text()), control)

    def test_recorded_control_overrides_stale_environment_before_target(self):
        project = Path(__file__).resolve().parent
        for controls in ({}, {'CAMERA_TRANSLATION_MM':'1'}, {'CAMERA_ROTATION_DEG':'.5'}):
            with self.subTest(controls=controls), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                env = dict(os.environ)
                for key in ('CAMERA_TRANSLATION_MM','CAMERA_ROTATION_DEG','CUBE_YAW_DEG',
                            'CUBE_YAW_MAX_DEG','CUBE_YAW_SEED'):
                    env.pop(key, None)
                env.update(CUBE_YAW_MODE='original', CUBE_LAYOUT_SEED='20261009', **controls)
                subprocess.run([sys.executable,'-B',str(project/'record_run_config.py'),
                                '--run',folder], cwd=project, env=env, check=True, capture_output=True)
                expected = json.loads((root/'run_config.json').read_text())['effective']['camera_mismatch']
                probe = root/'probe.py'
                probe.write_text('import json,sys; from calibration_mismatch import configuration; print(json.dumps([configuration(),sys.argv[1:]]))')
                env.update(CAMERA_TRANSLATION_MM='invalid', CAMERA_ROTATION_DEG='invalid')
                result = subprocess.run([sys.executable,'-B',str(project/'isaac_run_entrypoint.py'),
                                         str(root/'run_config.json'),str(probe),'argument with spaces'],
                                        cwd=project, env=env, check=True, capture_output=True, text=True)
                self.assertEqual(json.loads(result.stdout.splitlines()[-1]), [expected,['argument with spaces']])

    def test_execution_watchdog_arms_for_target_and_allows_clean_exit(self):
        project = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            env = dict(os.environ, CUBE_LAYOUT_SEED='20261009', CUBE_YAW_MODE='original',
                       CAMERA_TRANSLATION_MM='0', CAMERA_ROTATION_DEG='0')
            for key in ('CUBE_YAW_DEG','CUBE_YAW_MAX_DEG','CUBE_YAW_SEED'):
                env.pop(key, None)
            subprocess.run([sys.executable,'-B',str(project/'record_run_config.py'),
                            '--run',folder], cwd=project, env=env, check=True, capture_output=True)
            # Synthetic target with the execution filename; never imports Isaac.
            target = root/'sort_scene.py'
            target.write_text("print('SYNTHETIC TARGET FINISHED', flush=True)\n")
            result = subprocess.run([sys.executable,'-B',str(project/'isaac_run_entrypoint.py'),
                                     str(root/'run_config.json'),str(target)],
                                    cwd=project, env=env, check=True, capture_output=True, text=True,
                                    timeout=15)
            self.assertIn('Traceback watchdog armed', result.stdout)
            self.assertIn('SYNTHETIC TARGET FINISHED', result.stdout)
            self.assertEqual((root/'execution_startup_traceback.txt').read_text(), '')


if __name__ == '__main__':
    unittest.main()
