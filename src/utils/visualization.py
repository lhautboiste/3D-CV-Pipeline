import cv2
import numpy as np
import open3d as o3d

def visualize_pointcloud(point_cloud):
    """
    Visualize a point cloud
    
    Args:
        point_cloud (open3d.geometry.PointCloud): Point cloud to visualize
    """
    o3d.visualization.draw_geometries([point_cloud])

def visualize_mesh(mesh):
    """
    Visualize a mesh
    
    Args:
        mesh (open3d.geometry.TriangleMesh): Mesh to visualize
    """
    o3d.visualization.draw_geometries([mesh])

def draw_matches(img1, kp1, img2, kp2, matches, max_matches=50):
    """
    Draw matches between two images
    
    Args:
        img1 (numpy.ndarray): First image
        kp1: Keypoints from first image
        img2 (numpy.ndarray): Second image
        kp2: Keypoints from second image
        matches: Matched features
        max_matches (int): Maximum number of matches to draw
        
    Returns:
        numpy.ndarray: Image with matches drawn
    """
    # Select top matches
    good_matches = matches[:max_matches]
    
    # Draw matches
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return img_matches