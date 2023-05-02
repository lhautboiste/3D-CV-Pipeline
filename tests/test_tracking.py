import unittest
import numpy as np
import cv2
import os
# Assuming tests are run from the project root directory
from src.tracking.feature_matching import FeatureMatcher
from src.tracking.visual_odometry import VisualOdometry

class TestFeatureMatching(unittest.TestCase):
    def setUp(self):
        self.matcher = FeatureMatcher(detector_type='ORB')
        # Create simple test images
        self.img1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.img2 = np.roll(self.img1, shift=5, axis=1) # Simple horizontal shift

    def test_detect_and_compute(self):
        kp, des = self.matcher.detect_and_compute(self.img1)
        self.assertIsNotNone(kp)
        self.assertIsNotNone(des)
        self.assertGreater(len(kp), 0)
        self.assertEqual(len(kp), des.shape[0] if des is not None else 0)

    def test_match_features(self):
        kp1, des1 = self.matcher.detect_and_compute(self.img1)
        kp2, des2 = self.matcher.detect_and_compute(self.img2)
        matches = self.matcher.match_features(des1, des2)
        # With a shifted image, we expect some matches
        # This is a basic check; real-world tests would be more robust
        self.assertIsInstance(matches, list)

    def test_get_matched_points(self):
        kp1, des1 = self.matcher.detect_and_compute(self.img1)
        kp2, des2 = self.matcher.detect_and_compute(self.img2)
        matches = self.matcher.match_features(des1, des2)
        if len(matches) >= 2: # Need at least 2 for fundamental matrix
            pts1, pts2 = self.matcher.get_matched_points(kp1, kp2, matches)
            self.assertEqual(pts1.shape[0], pts2.shape[0])
            self.assertEqual(pts1.shape[1:], (1, 2))
            self.assertEqual(pts2.shape[1:], (1, 2))

class TestVisualOdometry(unittest.TestCase):
    def setUp(self):
        # Simple camera matrix for test images
        self.camera_matrix = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        self.vo = VisualOdometry(self.camera_matrix, detector_type='ORB')
        # Create test frames
        self.frame1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.frame2 = np.roll(self.frame1, shift=5, axis=1)

    def test_process_frame_initial(self):
        R, t = self.vo.process_frame(self.frame1)
        # Initial pose should be identity rotation and zero translation
        np.testing.assert_array_almost_equal(R, np.eye(3))
        np.testing.assert_array_almost_equal(t, np.zeros((3, 1)))

    def test_process_frame_subsequent(self):
        # Process first frame
        _ = self.vo.process_frame(self.frame1)
        # Process second frame
        R, t = self.vo.process_frame(self.frame2)
        # We don't assert specific values due to randomness, but check types/shape
        self.assertEqual(R.shape, (3, 3))
        self.assertEqual(t.shape, (3, 1))

    def test_reset(self):
        _ = self.vo.process_frame(self.frame1)
        self.vo.reset()
        # After reset, processing the same first frame should give identity pose
        R, t = self.vo.process_frame(self.frame1)
        np.testing.assert_array_almost_equal(R, np.eye(3))
        np.testing.assert_array_almost_equal(t, np.zeros((3, 1)))

if __name__ == '__main__':
    unittest.main()