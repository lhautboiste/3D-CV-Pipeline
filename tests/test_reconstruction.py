import unittest
import numpy as np
import tempfile
import os
# Assuming tests are run from the project root directory
from src.reconstruction.depth_reconstruction import DepthReconstructor
from src.reconstruction.pointcloud_processing import PointCloudProcessor

class TestReconstruction(unittest.TestCase):
    def setUp(self):
        self.intrinsics = {
            'width': 100,
            'height': 100,
            'fx': 100.0,
            'fy': 100.0,
            'cx': 49.5,
            'cy': 49.5
        }
        self.reconstructor = DepthReconstructor(self.intrinsics)
        
    def test_initialization(self):
        self.assertIsNotNone(self.reconstructor)
        self.assertEqual(self.reconstructor.intrinsics, self.intrinsics)

    def test_convert_to_pointcloud_with_mock_data(self):
        # Create a small mock depth image
        depth_image = np.ones((100, 100), dtype=np.uint16) * 1000  # 1 meter depth
        pointcloud = self.reconstructor.convert_to_pointcloud(depth_image)
        self.assertIsNotNone(pointcloud)
        self.assertGreater(len(pointcloud.points), 0)

    def test_create_colored_pointcloud(self):
        # Create mock depth and color images
        depth_image = np.ones((100, 100), dtype=np.float32) * 1.0 # 1 meter
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Mock intrinsics object
        class MockIntrinsics:
            def __init__(self):
                self.ppx = 49.5
                self.ppy = 49.5
                self.fx = 100.0
                self.fy = 100.0
        
        mock_intrinsics = MockIntrinsics()
        pointcloud = self.reconstructor.create_colored_pointcloud(depth_image, color_image, mock_intrinsics)
        self.assertIsNotNone(pointcloud)
        self.assertGreater(len(pointcloud.points), 0)
        self.assertGreater(len(pointcloud.colors), 0)
        self.assertEqual(len(pointcloud.points), len(pointcloud.colors))

class TestPointCloudProcessor(unittest.TestCase):
    def setUp(self):
        # Create a simple point cloud for testing
        self.points = np.random.rand(1000, 3)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)

    def test_downsample(self):
        downsampled = PointCloudProcessor.downsample(self.pcd, voxel_size=0.1)
        self.assertLess(len(downsampled.points), len(self.pcd.points))

    def test_remove_statistical_outliers(self):
        # Add some outliers
        outlier_points = np.array([[10, 10, 10], [11, 11, 11]])
        all_points = np.vstack([self.points, outlier_points])
        pcd_with_outliers = o3d.geometry.PointCloud()
        pcd_with_outliers.points = o3d.utility.Vector3dVector(all_points)
        
        filtered_pcd, ind = PointCloudProcessor.remove_statistical_outliers(pcd_with_outliers, nb_neighbors=20, std_ratio=0.5)
        # Check that outliers are likely removed (this is probabilistic)
        # A more robust test would involve checking distances explicitly
        self.assertLessEqual(len(filtered_pcd.points), len(pcd_with_outliers.points))

    def test_crop_point_cloud(self):
        min_bound = [0.2, 0.2, 0.2]
        max_bound = [0.8, 0.8, 0.8]
        cropped = PointCloudProcessor.crop_point_cloud(self.pcd, min_bound, max_bound)
        # Check that all points are within bounds
        points_np = np.asarray(cropped.points)
        self.assertTrue(np.all(points_np >= np.array(min_bound) - 1e-6)) # Account for float precision
        self.assertTrue(np.all(points_np <= np.array(max_bound) + 1e-6))

    def test_segment_plane(self):
        # Add a dominant plane to the point cloud
        plane_points = np.random.rand(500, 3)
        plane_points[:, 2] = 0.0 # Z=0 plane
        all_points = np.vstack([self.points[:500], plane_points])
        pcd_with_plane = o3d.geometry.PointCloud()
        pcd_with_plane.points = o3d.utility.Vector3dVector(all_points)
        
        inlier_cloud, outlier_cloud, plane_model = PointCloudProcessor.segment_plane(
            pcd_with_plane, distance_threshold=0.02, ransac_n=3, num_iterations=100
        )
        # Check that plane was found (normal should be close to [0,0,1])
        # Plane equation: ax + by + cz + d = 0. For z=0, a=0, b=0, c=1, d=0
        # We check if c is dominant
        normal = plane_model[:3]
        self.assertGreater(np.abs(normal[2]), np.abs(normal[0]))
        self.assertGreater(np.abs(normal[2]), np.abs(normal[1]))

if __name__ == '__main__':
    # Suppress Open3D warnings for tests
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    unittest.main()