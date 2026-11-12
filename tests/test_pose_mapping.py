"""Verify the personalized target mapping with the real model and synthetic captures."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dex_hand_retargeting import DexHandRetargeter, FINGER_LINKS, LANDMARK_NAMES, canonical_landmarks
from pose_mapping import POSES, SAMPLE_COUNT, PoseMapping, mapping_path, robot_references


class PoseMappingTests(unittest.TestCase):
    def test_new_calibration_refits_targets_and_reuses_saved_mapping(self):
        adapter = DexHandRetargeter.from_project()
        robot = adapter.retargeting.optimizer.robot
        references = robot_references(robot, FINGER_LINKS, adapter.limits, adapter.collision)
        previous_target = None
        for scale in ([.84, 1.07, .95], [1.13, .92, 1.04]):
            measured = references * scale
            captures = {
                key: [dict(zip(LANDMARK_NAMES, points)) for _ in range(SAMPLE_COUNT)]
                for (key, _), points in zip(POSES, measured)
            }
            adapter.fit_pose_mapping(captures)
            nodes = captures['pinch_index'][0]
            points = canonical_landmarks(nodes)
            corrected = adapter.pose_mapping.apply(points)
            tip_indices = [4, 8, 12, 16, 20]
            target_tips = references[3, tip_indices]
            self.assertLess(np.linalg.norm(corrected[tip_indices] - target_tips),
                            .15 * np.linalg.norm(points[tip_indices] - target_tips))
            if previous_target is not None:
                self.assertFalse(np.allclose(previous_target, adapter.pose_mapping.residual_weights))
            previous_target = adapter.pose_mapping.residual_weights.copy()

            # A separate nearby pose still flows through the live optimizer.
            nearby = measured[3].copy()
            nearby[4] += [.0001, -.0002, .0001]
            nearby_nodes = dict(zip(LANDMARK_NAMES, nearby))
            expected = adapter.pose_mapping.apply(canonical_landmarks(nearby_nodes))
            output = adapter.retarget_skeleton(nearby_nodes)
            self.assertTrue(np.isfinite(output).all())
            q = output[adapter.sim_indices]
            self.assertGreaterEqual(adapter.collision.distances(q).min(), -adapter.collision.tolerance)
            segments = np.diff(expected[1:].reshape(5, 4, 3), axis=1).reshape(15, 3)
            np.testing.assert_allclose(adapter.objective.targets,
                                       segments / np.linalg.norm(segments, axis=1, keepdims=True))

        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            model = home / 'hand.xml'
            model.write_text('test robot', encoding='utf-8')
            record = home / 'glove' / 'calibrations' / 'record-test' / 'record.yaml'
            record.parent.mkdir(parents=True)
            record.write_text('first official profile', encoding='utf-8')
            first = mapping_path(home, model, 'vector', collision=True)
            adapter.pose_mapping.save(first)
            restored = PoseMapping.load(first)
            np.testing.assert_array_equal(restored.apply(points), adapter.pose_mapping.apply(points))
            record.write_text('second official profile', encoding='utf-8')
            self.assertNotEqual(first, mapping_path(home, model, 'vector', collision=True))
            record.write_text('first official profile', encoding='utf-8')
            self.assertEqual(first, mapping_path(home, model, 'vector', collision=True))

    def test_rejects_repeated_capture_poses(self):
        rng = np.random.default_rng(13)
        points = np.repeat(rng.uniform(-.1, .1, (1, 21, 3)), len(POSES), axis=0)
        with self.assertRaises(ValueError):
            PoseMapping.fit(points, points)
