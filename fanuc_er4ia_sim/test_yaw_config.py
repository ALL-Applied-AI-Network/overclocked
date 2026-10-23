"""Host checks for yaw controls, recording, bridge transport and launcher validation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from yaw_config import ENV_NAMES, ORIGINAL_YAWS_DEG, environment, resolve

ROOT = Path(__file__).resolve().parent


def clean_env(extra=None):
    env = dict(os.environ)
    for key in (*ENV_NAMES, 'CUBE_LAYOUT_SEED'):
        env.pop(key, None)
    env.update(extra or {})
    return env


class YawConfigTests(unittest.TestCase):
    def test_original_is_exact_and_is_not_all_zero(self):
        settings, values = resolve({})
        self.assertEqual(settings['mode'], 'original')
        self.assertEqual(values, [0, 26, 32, -23, 14, -36])
        self.assertEqual(values, resolve({'CUBE_YAW_MODE': 'original'})[1])

    def test_fixed_applies_to_all_six_and_includes_boundaries(self):
        for value in ('0', '15.5', '45'):
            self.assertEqual(resolve({'CUBE_YAW_MODE': 'fixed', 'CUBE_YAW_DEG': value})[1], [float(value)] * 6)

    def test_random_is_repeatable_bounded_independent_and_does_not_touch_global_rng(self):
        import random
        controls = {'CUBE_YAW_MODE': 'random', 'CUBE_YAW_MAX_DEG': '30', 'CUBE_YAW_SEED': '-42'}
        state = random.getstate()
        settings, values = resolve(controls)
        self.assertEqual(state, random.getstate())
        self.assertEqual(values, resolve(controls)[1])
        self.assertEqual(len(values), 6)
        self.assertTrue(all(0 <= v <= 30 for v in values))
        self.assertNotEqual(values, resolve(dict(controls, CUBE_YAW_SEED='43'))[1])
        self.assertEqual((settings, values), resolve(environment(settings)))
        self.assertEqual(resolve(dict(controls, CUBE_YAW_MAX_DEG='0'))[1], [0] * 6)

    def test_invalid_controls_are_rejected(self):
        for value in ('-1', '45.01', 'nan', 'inf', '', 'banana'):
            for mode, key in (('fixed', 'CUBE_YAW_DEG'), ('random', 'CUBE_YAW_MAX_DEG')):
                with self.subTest(mode=mode, value=value), self.assertRaises(ValueError):
                    resolve({'CUBE_YAW_MODE': mode, key: value})
        for env in ({'CUBE_YAW_MODE': 'other'}, {'CUBE_YAW_MODE': 'fixed'},
                    {'CUBE_YAW_DEG': '20'}, {'CUBE_YAW_MODE': 'fixed', 'CUBE_YAW_DEG': '20', 'CUBE_YAW_SEED': '1'},
                    {'CUBE_YAW_MODE': 'random', 'CUBE_YAW_SEED': '1.5'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                resolve(env)

    def scene(self, env):
        result = subprocess.run([sys.executable, '-B', '-c',
            'import json,scene_config as C; print(json.dumps([C.SORT_CUBES,C.SORT_CUBE_YAWS_DEG]))'],
            cwd=ROOT, env=clean_env(env), capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def test_yaw_and_layout_randomness_are_independent(self):
        default = self.scene({})
        random = {'CUBE_YAW_MODE': 'random', 'CUBE_YAW_SEED': '123'}
        yaw = self.scene(random)
        alternate = self.scene(dict(random, CUBE_LAYOUT_SEED='20261010'))
        self.assertEqual(default[0], yaw[0])
        self.assertEqual(yaw[1], alternate[1])
        self.assertEqual(default[0][:2], alternate[0][:2])
        self.assertNotEqual(default[0][2:], alternate[0][2:])

    def test_recorder_and_bridge_match_before_target_import_and_preserve_arguments(self):
        for controls in ({}, {'CUBE_YAW_MODE': 'fixed', 'CUBE_YAW_DEG': '45'},
                         {'CUBE_YAW_MODE': 'random', 'CUBE_YAW_MAX_DEG': '22.5', 'CUBE_YAW_SEED': '123'}):
            with self.subTest(controls=controls), tempfile.TemporaryDirectory() as temp:
                run = Path(temp)
                subprocess.run([sys.executable, '-B', str(ROOT / 'record_run_config.py'), '--run', str(run)],
                    env=clean_env(controls), cwd=ROOT, capture_output=True, check=True)
                record = json.loads((run / 'run_config.json').read_text())
                probe = run / 'probe.py'
                probe.write_text('import json,scene_config as C,sys; print(json.dumps([C.SORT_CUBE_YAWS_DEG,sys.argv[1:]]))')
                command = [sys.executable, '-B', str(ROOT / 'isaac_run_entrypoint.py'),
                           str(run / 'run_config.json'), str(probe), '--out', 'folder with spaces']
                result = subprocess.run(command, cwd=ROOT,
                    env=clean_env({'CUBE_YAW_MODE': 'invalid', 'CUBE_LAYOUT_SEED': '999', 'CUBE_YAW_DEG': '999'}),
                    capture_output=True, text=True, check=True)
                values, args = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(values, record['effective']['cube_yaws_deg'])
                self.assertEqual(args, ['--out', 'folder with spaces'])
                record['effective']['cube_yaws_deg'][0] += 1
                (run / 'run_config.json').write_text(json.dumps(record))
                result = subprocess.run(command, cwd=ROOT, env=clean_env(), capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('yaws disagree', result.stderr)

    def test_launcher_preview_validates_without_creating_runs(self):
        for options in ([], ['--yaw-mode', 'fixed', '--yaw-deg', '45'],
                        ['--yaw-mode', 'random', '--yaw-max-deg', '20', '--yaw-seed', '-3']):
            result = subprocess.run(['bash', str(ROOT / 'run_stress_batch.sh'), '--runs', '2', '--preview', *options],
                                    cwd=ROOT, env=clean_env(), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('PREVIEW ONLY', result.stdout)
        for options in (['--yaw-mode', 'fixed', '--yaw-deg', '46'], ['--yaw-mode', 'random', '--yaw-max-deg', 'nan'],
                        ['--yaw-mode', 'original', '--yaw-deg', '10'], ['--yaw-mode', 'bad']):
            result = subprocess.run(['bash', str(ROOT / 'run_stress_batch.sh'), '--runs', '2', '--preview', *options],
                                    cwd=ROOT, env=clean_env(), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
