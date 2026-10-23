import unittest
from compare_inventory import validate


class InventoryGateTests(unittest.TestCase):
    def test_missing_confidence_stays_rejected(self):
        self.assertFalse(validate({'category': 'cube', 'colour': 'red'}, 'red'))

    def test_orange_stays_rejected_as_red(self):
        self.assertFalse(validate({'category': 'cube', 'colour': 'orange', 'confidence': .99}, 'red'))

    def test_confident_red_is_accepted(self):
        self.assertTrue(validate({'category': 'cube', 'colour': 'red', 'confidence': .9}, 'red'))

    def test_low_or_invalid_confidence_stays_rejected(self):
        for confidence in (.79, None, 'invalid'):
            self.assertFalse(validate({'category': 'cube', 'colour': 'red', 'confidence': confidence}, 'red'))
