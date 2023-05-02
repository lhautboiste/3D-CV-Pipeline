import cv2
import numpy as np
import open3d as o3d
from ..utils.image_processing import load_depth_image

class DepthReconstructor:
    """Class for reconstructing 3D models from depth images"""
    
    def __init__(self, camera_intrinsics):
        """
        Initialize the DepthReconstructor
        
        Args:
            camera_intrinsics (dict): Camera intrinsic parameters
        """
        self.intrinsics = camera_intrinsics
        
    def convert_to_pointcloud(self, depth_image):
        """
        Convert depth image to point cloud
        
        Args:
            depth_image (numpy.ndarray): Depth image
            
        Returns:
            open3d.geometry.PointCloud: Point cloud
        """
        depth_image_np = np.asarray(depth_image, dtype=np.uint16)
        depth_o3d = o3d.geometry.Image.from_numpy(depth_image_np)
        
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.intrinsics['width'], self.intrinsics['height'],
            self.intrinsics['fx'], self.intrinsics['fy'],
            self.intrinsics['cx'], self.intrinsics['cy']
        )
        
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, pinhole_camera_intrinsic
        )
        
        return point_cloud
        
    def create_colored_pointcloud(self, depth_image, color_image, depth_intrinsics):
        """
        Create colored point cloud from depth and color images
        
        Args:
            depth_image (numpy.ndarray): Depth image
            color_image (numpy.ndarray): Color image
            depth_intrinsics: Depth camera intrinsics
            
        Returns:
            open3d.geometry.PointCloud: Colored point cloud
        """
        height, width = depth_image.shape
        v, u = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        x = (u - depth_intrinsics.ppx) / depth_intrinsics.fx * depth_image
        y = (v - depth_intrinsics.ppy) / depth_intrinsics.fy * depth_image
        z = depth_image

        points_3d = np.dstack((x, y, z)).reshape(-1, 3)
        colors = color_image.reshape(-1, 3) / 255.

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_3d)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        return point_cloud
        
    def estimate_normals(self, point_cloud, search_param=None):
        """
        Estimate normals for a point cloud
        
        Args:
            point_cloud (open3d.geometry.PointCloud): Input point cloud
            search_param: Search parameter for normal estimation
            
        Returns:
            open3d.geometry.PointCloud: Point cloud with normals
        """
        if search_param is None:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            
        point_cloud.estimate_normals(search_param)
        return point_cloud
        
    def create_mesh_ball_pivoting(self, point_cloud, radii=None):
        """
        Create mesh using ball pivoting algorithm
        
        Args:
            point_cloud (open3d.geometry.PointCloud): Input point cloud
            radii (list): Radii for ball pivoting
            
        Returns:
            open3d.geometry.TriangleMesh: Generated mesh
        """
        if radii is None:
            radii = [0.005, 0.01, 0.02, 0.04]
            
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            point_cloud, o3d.utility.DoubleVector(radii)
        )
        return mesh
        
    def create_mesh_poisson(self, point_cloud, depth=8):
        """
        Create mesh using Poisson surface reconstruction
        
        Args:
            point_cloud (open3d.geometry.PointCloud): Input point cloud
            depth (int): Depth for Poisson reconstruction
            
        Returns:
            open3d.geometry.TriangleMesh: Generated mesh
        """
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=depth
        )
        return mesh