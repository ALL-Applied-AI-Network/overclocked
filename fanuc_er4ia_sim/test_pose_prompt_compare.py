"""Check that comparison results cannot bypass the existing pose gates."""
import math
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import compare_pose_prompts as tool
from compare_pose_prompts import validate_reply, CANDIDATE_PROMPT


class ComparisonGateTests(unittest.TestCase):
    def setUp(self):
        self.measurement = {'expected_colour': 'blue', 'yaw_deg': -36.0864}
        self.answer = {'category': 'cube', 'colour': 'blue', 'yaw_deg': -36.0864, 'pose_confidence': 0.9}

    def test_correct_reply_passes(self):
        self.assertTrue(validate_reply(self.answer, self.measurement)['accepted'])

    def test_known_example_copy_fails(self):
        self.answer['yaw_deg'] = 14.2
        result = validate_reply(self.answer, self.measurement)
        self.assertFalse(result['accepted'])
        self.assertAlmostEqual(result['square_yaw_disagreement_deg'], 39.7136)

    def test_square_symmetry_passes(self):
        self.answer['yaw_deg'] += 90
        self.assertTrue(validate_reply(self.answer, self.measurement)['accepted'])

    def test_wrong_colour_and_category_fail(self):
        for key, value in [('colour', 'red'), ('category', 'cap')]:
            with self.subTest(key=key):
                self.assertFalse(validate_reply(dict(self.answer, **{key: value}), self.measurement)['accepted'])

    def test_confidence_not_filled_in(self):
        del self.answer['pose_confidence']
        self.assertFalse(validate_reply(self.answer, self.measurement)['accepted'])

    def test_invalid_numeric_replies_fail(self):
        for key in ('yaw_deg', 'pose_confidence'):
            for value in (None, 'banana', math.nan, math.inf):
                with self.subTest(key=key, value=value):
                    self.assertFalse(validate_reply(dict(self.answer, **{key: value}), self.measurement)['accepted'])

    def test_unparsed_answer_fails(self):
        for value in (None, [], 'not JSON'):
            self.assertFalse(validate_reply(value, self.measurement)['accepted'])

    def test_exact_boundaries_unchanged(self):
        measurement = dict(self.measurement, yaw_deg=0)
        self.assertTrue(validate_reply(dict(self.answer, yaw_deg=3, pose_confidence=0.8), measurement)['accepted'])
        self.assertFalse(validate_reply(dict(self.answer, yaw_deg=3.001), measurement)['accepted'])
        self.assertFalse(validate_reply(dict(self.answer, yaw_deg=0, pose_confidence=0.799), measurement)['accepted'])

    def test_candidate_has_no_example_angle_or_confidence(self):
        self.assertNotIn('14.2', CANDIDATE_PROMPT)
        self.assertNotIn('0.9', CANDIDATE_PROMPT)


class EvidencePreservationTests(unittest.TestCase):
    def test_paired_comparison_keeps_original_inputs_and_refuses_rerun(self):
        fake = types.ModuleType('perceive')
        fake.CUBE_POSE_PROMPT = 'ORIGINAL INSTRUCTIONS'
        fake.load_model = lambda gpu: (None, None)
        fake.parse_json = json.loads
        fake.ask = lambda model, processor, image, prompt, budget: json.dumps({
            'category': 'cube', 'colour': 'blue', 'yaw_deg': -36, 'pose_confidence': 0.9})
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, perceive=fake):
            root = Path(temp)
            folder = root / 'runs' / 'old_run' / 'cube_pose_crops'
            folder.mkdir(parents=True)
            image = folder / 'cube_3.png'
            image.write_bytes(b'fixture: no real model is called')
            (folder / 'cube_3.txt').write_text('preserved historical reply')
            tool.write_json(folder / 'cube_3_measurement.json', {
                'expected_colour': 'blue', 'yaw_deg': -36, 'silhouette_iou': 0.97})
            for name in ('perceive.py', 'cube_pose.py', 'compare_pose_prompts.py'):
                (root / name).write_text('# fixture')
            before = {p: p.read_bytes() for p in folder.iterdir()}
            tool.prepare(root, ['old_run'])
            output = next((root / 'runs').glob('lieland_pose_compare_*'))
            tool.execute(output, root)
            result = json.loads((output / 'development_outcome.json').read_text())
            self.assertEqual(result['outcome'], 'saved_crop_comparison_complete')
            self.assertEqual(result['completed_calls'], 2)
            self.assertEqual({r['variant'] for r in result['results']}, {'original', 'candidate'})
            self.assertEqual(before, {p: p.read_bytes() for p in folder.iterdir()})
            saved = {p: p.read_bytes() for p in (output / 'cube_pose_crops').iterdir()}
            with self.assertRaises(FileExistsError):
                tool.execute(output, root)
            self.assertEqual(saved, {p: p.read_bytes() for p in (output / 'cube_pose_crops').iterdir()})


if __name__ == '__main__':
    unittest.main()
