#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import open3d as o3d

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.reconstruction.depth_reconstruction import DepthReconstructor
from src.reconstruction.pointcloud_processing import PointCloudProcessor
from src.utils.visualization import visualize_pointcloud, visualize_mesh

def create_cube_point_cloud(size=1.0, num_points_per_face=100):
    """Generate a point cloud representing a cube."""
    points = []
    # Define the 8 vertices of the cube
    vertices = np.array([
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
        [0, 0, size], [size, 0, size], [size, size, size], [0, size, size]
    ])
    
    # Define the 12 edges of the cube (vertex indices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]
    
    # Sample points along each edge
    for start_idx, end_idx in edges:
        start_point = vertices[start_idx]
        end_point = vertices[end_idx]
        # Sample points along the edge
        t_vals = np.linspace(0, 1, num_points_per_face)
        edge_points = start_point + (end_point - start_point) * t_vals[:, np.newaxis]
        points.append(edge_points)
    
    all_points = np.vstack(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    return pcd

def main():
    parser = argparse.ArgumentParser(description='Reconstruct a 3D cube.')
    parser.add_argument('--size', type=float, default=1.0, help='Size of the cube (meters)')
    parser.add_argument('--points_per_edge', type=int, default=100, help='Number of points per edge')
    parser.add_argument('--noise', type=float, default=0.0, help='Standard deviation of noise to add')
    parser.add_argument('--method', type=str, choices=['pointcloud', 'ball_pivoting', 'poisson'], 
                        default='pointcloud', help='Reconstruction method')
    args = parser.parse_args()

    print("Generating cube point cloud...")
    cube_pcd = create_cube_point_cloud(size=args.size, num_points_per_face=args.points_per_edge)
    
    # Add noise if specified
    if args.noise > 0:
        points = np.asarray(cube_pcd.points)
        noise = np.random.normal(0, args.noise, points.shape)
        cube_pcd.points = o3d.utility.Vector3dVector(points + noise)
        print(f"Added noise with std dev {args.noise}")

    if args.method == 'pointcloud':
        print("Visualizing cube point cloud...")
        visualize_pointcloud(cube_pcd)
    else:
        print("Estimating normals...")
        cube_pcd_with_normals = PointCloudProcessor.downsample(cube_pcd, voxel_size=0.05)
        cube_pcd_with_normals.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        
        if args.method == 'ball_pivoting':
            print("Creating mesh with Ball Pivoting Algorithm...")
            radii = [0.05, 0.1, 0.2, 0.4]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                cube_pcd_with_normals, o3d.utility.DoubleVector(radii)
            )
        else: # poisson
            print("Creating mesh with Poisson Reconstruction...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cube_pcd_with_normals, depth=9
            )
            # Optional: Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print("Visualizing reconstructed mesh...")
        visualize_mesh(mesh)

if __name__ == "__main__":
    main()