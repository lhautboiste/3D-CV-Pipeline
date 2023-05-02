#!/usr/bin/env python3

import sys
import os
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.reconstruction.depth_reconstruction import DepthReconstructor
from src.utils.visualization import visualize_pointcloud, visualize_mesh
from config.settings import DEFAULT_INTRINSICS, load_config

def main():
    parser = argparse.ArgumentParser(description='Reconstruct 3D model from depth image')
    parser.add_argument('depth_image_path', type=str, help='Path to the depth image')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--method', type=str, choices=['pointcloud', 'ball_pivoting', 'poisson'], 
                        default='pointcloud', help='Reconstruction method')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        intrinsics = config.get('camera_intrinsics', DEFAULT_INTRINSICS)
    else:
        intrinsics = DEFAULT_INTRINSICS
    
    # Initialize reconstructor
    reconstructor = DepthReconstructor(intrinsics)
    
    try:
        # Load depth image
        depth_image = reconstructor.load_depth_image(args.depth_image_path)
        
        # Convert to pointcloud
        pointcloud = reconstructor.convert_to_pointcloud(depth_image)
        
        if args.method == 'pointcloud':
            visualize_pointcloud(pointcloud)
        else:
            # Estimate normals
            pointcloud_with_normals = reconstructor.estimate_normals(pointcloud)
            
            # Create mesh
            if args.method == 'ball_pivoting':
                mesh = reconstructor.create_mesh_ball_pivoting(pointcloud_with_normals)
            else:  # poisson
                mesh = reconstructor.create_mesh_poisson(pointcloud_with_normals)
            
            visualize_mesh(mesh)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()