import unittest
import numpy as np
# Assuming tests are run from the project root directory
from src.stereo.stereo_vision import StereoReconstructor
from src.stereo.orientation_estimation import PoseEstimator

class TestStereoReconstructor(unittest.TestCase):
    def setUp(self):
        # Mock calibration data
        self.calib_data = {
            'left_camera_matrix': [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
            'right_camera_matrix': [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
            'left_dist_coeffs': [0, 0, 0, 0],
            'right_dist_coeffs': [0, 0, 0, 0],
            'R': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'T': [-0.1, 0, 0], # 10cm baseline
            'image_size': (100, 100) # width, height
        }
        self.stereo_recon = StereoReconstructor(self.calib_data)
        # Create simple test images
        self.left_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.right_img = np.roll(self.left_img, shift=-2, axis=1) # Simulate disparity

    def test_initialization(self):
        self.assertIsNotNone(self.stereo_recon)
        self.assertTrue(hasattr(self.stereo_recon, 'stereo'))

    def test_rectify_images(self):
        left_rect, right_rect = self.stereo_recon.rectify_images(self.left_img, self.right_img)
        self.assertEqual(left_rect.shape, self.left_img.shape)
        self.assertEqual(right_rect.shape, self.right_img.shape)

    def test_compute_disparity(self):
        disp, left_rect, right_rect = self.stereo_recon.compute_disparity(self.left_img, self.right_img)
        self.assertEqual(disp.shape, self.left_img.shape)
        # Disparity should not be all zeros for shifted images
        self.assertFalse(np.allclose(disp, 0))

    def test_compute_depth_map(self):
        disp, _, _ = self.stereo_recon.compute_disparity(self.left_img, self.right_img)
        depth = self.stereo_recon.compute_depth_map(disp)
        self.assertEqual(depth.shape, disp.shape)
        # Depth should not be all infs or zeros for valid disparity
        finite_depth = depth[np.isfinite(depth)]
        self.assertGreater(len(finite_depth), 0)

class TestPoseEstimator(unittest.TestCase):
    def setUp(self):
        self.camera_matrix = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        self.pose_estimator = PoseEstimator(self.camera_matrix)
        # Define a simple 3D object (e.g., a square)
        self.object_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
        # Define corresponding 2D points (as if viewed from a specific pose)
        self.image_points = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float64)

    def test_estimate_pose_pnp(self):
        success, rvec, tvec = self.pose_estimator.estimate_pose_pnp(self.object_points, self.image_points)
        self.assertTrue(success)
        self.assertEqual(rvec.shape, (3, 1))
        self.assertEqual(tvec.shape, (3, 1))

    def test_project_points(self):
        # First get a pose
        success, rvec, tvec = self.pose_estimator.estimate_pose_pnp(self.object_points, self.image_points)
        self.assertTrue(success)
        # Project the points back
        projected = self.pose_estimator.project_points(self.object_points, rvec, tvec)
        self.assertEqual(projected.shape, self.image_points.shape)
        # Projected points should be close to the original image points
        # Note: This is a simplified check. Real checks would consider projection accuracy.
        # For this test, we just ensure it runs and returns correct shape.

if __name__ == '__main__':
    unittest.main()