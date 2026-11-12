import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_guard import CalibrationGuard


class CalibrationGuardTests(unittest.TestCase):
    def test_new_scale_can_be_previewed_before_manual_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = CalibrationGuard(directory)
            path = Path(directory) / 'glove/calibrations/skeleton/record.yaml'
            path.parent.mkdir(parents=True)

            def write_profile(record_id, scale, operator='default'):
                path.write_text(yaml.safe_dump({
                    '_metadata': {'record_id': record_id, 'operator_id': operator,
                                  'canonical_sn': 'test-glove', 'target': 'hand.skeleton.left'},
                    'data': {'raw_scale': scale - .1, 'scale': scale},
                }), encoding='utf-8')

            write_profile('initial', 1.465)
            self.assertEqual(guard.validate(['initial']), 1.465)
            original = path.read_bytes()
            guard.begin()
            write_profile('new_capture', .697)
            self.assertEqual(guard.validate(['new_capture']), .697)
            guard.rollback()
            self.assertEqual(path.read_bytes(), original)
