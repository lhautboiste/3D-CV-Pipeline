#!/usr/bin/env python3

import sys
import os
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.interfaces.realsense_interface import RealSenseCamera
from src.reconstruction.depth_reconstruction import DepthReconstructor
from src.utils.visualization import visualize_pointcloud
from config.settings import DEFAULT_INTRINSICS

def main():
    parser = argparse.ArgumentParser(description='Real-time 3D reconstruction with RealSense')
    parser.add_argument('--method', type=str, choices=['pointcloud', 'mesh'], 
                        default='pointcloud', help='Reconstruction method')
    args = parser.parse_args()
    
    # Initialize camera
    camera = RealSenseCamera()
    if not camera.initialize():
        print("Failed to initialize camera")
        sys.exit(1)
    
    try:
        # Capture frames
        frames = camera.capture_frames()
        
        # Get intrinsics
        intrinsics = camera.get_intrinsics()['depth']
        
        # Convert to Open3D format
        o3d_intrinsics = {
            'width': intrinsics.width,
            'height': intrinsics.height,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'cx': intrinsics.ppx,
            'cy': intrinsics.ppy
        }
        
        # Initialize reconstructor
        reconstructor = DepthReconstructor(o3d_intrinsics)
        
        # Create colored point cloud
        pointcloud = reconstructor.create_colored_pointcloud(
            frames['depth'], frames['color'], intrinsics
        )
        
        if args.method == 'pointcloud':
            visualize_pointcloud(pointcloud)
        else:
            # Estimate normals and create mesh
            pointcloud_with_normals = reconstructor.estimate_normals(pointcloud)
            mesh = reconstructor.create_mesh_ball_pivoting(pointcloud_with_normals)
            visualize_mesh(mesh)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.stop()

if __name__ == "__main__":
    main()