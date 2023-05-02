#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import cv2

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.stereo.orientation_estimation import PoseEstimator
from config.settings import load_config

def main():
    parser = argparse.ArgumentParser(description='Estimate 3D pose of a known object.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--object_points', type=str, required=True, help='Path to file with 3D object points (Nx3, one point per line)')
    parser.add_argument('--image_points', type=str, required=True, help='Path to file with 2D image points (Nx2, one point per line)')
    parser.add_argument('--config', type=str, default='config/camera_calibration.yaml', 
                        help='Path to camera calibration config file')
    parser.add_argument('--visualize', action='store_true', help='Draw projected points on the image')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    camera_matrix = np.array(config['camera_matrix'], dtype=np.float64)
    dist_coeffs = np.array(config.get('distortion_coefficients', [0, 0, 0, 0, 0]), dtype=np.float64)

    # Load 3D and 2D points
    try:
        object_points = np.loadtxt(args.object_points, dtype=np.float64)
        image_points = np.loadtxt(args.image_points, dtype=np.float64)
    except Exception as e:
        print(f"Error loading points: {e}")
        sys.exit(1)

    if object_points.shape[0] != image_points.shape[0]:
        print("Error: Number of 3D points must match number of 2D points.")
        sys.exit(1)
    if object_points.shape[1] != 3:
        print("Error: Object points must be Nx3.")
        sys.exit(1)
    if image_points.shape[1] != 2:
        print("Error: Image points must be Nx2.")
        sys.exit(1)

    # Initialize pose estimator
    pose_estimator = PoseEstimator(camera_matrix, dist_coeffs)

    # Estimate pose
    success, rvec, tvec = pose_estimator.estimate_pose_pnp(object_points, image_points)
    
    if not success:
        print("Error: Pose estimation failed.")
        sys.exit(1)

    print("Pose estimated successfully!")
    print(f"Rotation Vector (rvec): {rvec.flatten()}")
    print(f"Translation Vector (tvec): {tvec.flatten()}")

    # Convert rotation vector to matrix for easier interpretation
    R, _ = cv2.Rodrigues(rvec)
    print(f"Rotation Matrix (R):\n{R}")
    
    # Optionally visualize
    if args.visualize:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Warning: Could not load image {args.image} for visualization.")
        else:
            # Project points back onto the image
            projected_points = pose_estimator.project_points(object_points, rvec, tvec)
            for i, (pt_2d, orig_pt_2d) in enumerate(zip(projected_points, image_points)):
                x_proj, y_proj = int(pt_2d[0]), int(pt_2d[1])
                x_orig, y_orig = int(orig_pt_2d[0]), int(orig_pt_2d[1])
                # Draw original point (green) and projected point (red)
                cv2.circle(image, (x_orig, y_orig), 5, (0, 255, 0), -1) # Green
                cv2.circle(image, (x_proj, y_proj), 3, (0, 0, 255), -1) # Red
                cv2.line(image, (x_orig, y_orig), (x_proj, y_proj), (255, 0, 0), 1) # Blue line
            
            cv2.imshow('Pose Estimation', image)
            print("Press any key to close the image window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()