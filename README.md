# 3D Computer Vision Toolkit

A modular, object-oriented toolkit for 3D computer vision tasks including depth reconstruction, visual odometry, and stereo vision.
## Features

- Depth image to 3D reconstruction (point clouds and meshes)
- Real-time 3D reconstruction with Intel RealSense cameras
- Feature matching and pose estimation
- Visual odometry
- Stereo vision (disparity/depth computation, pose estimation)
- Point cloud processing (filtering, segmentation, clustering)
- 3D pose estimation of known objects from 2D-3D correspondences

## Usage

### Depth Reconstruction
```bash
python scripts/reconstruct_from_depth.py path/to/depth_image.png --method pointcloud
```

### Real-time Reconstruction with RealSense
```bash
python scripts/realtime_reconstruction.py --method pointcloud
```

### Visual Odometry
```bash
# From video
python scripts/visual_odometry.py --video path/to/video.mp4

# From two images
python scripts/visual_odometry.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

### Stereo Vision
```bash
# Compute disparity map (requires stereo pair and calibration file)
# Example script would go here when implemented
```

### 3D Pose Estimation
```bash
# Estimate pose of a known cube
python scripts/estimate_pose.py --image examples/sample_data/cube_image.jpg \
                               --object_points examples/sample_data/object_points_cube.txt \
                               --image_points examples/sample_data/image_points_cube.txt \
                               --visualize
```

### Cube Reconstruction
```bash
# Generate and reconstruct a synthetic cube
python scripts/cube_reconstruction.py --size 0.1 --method ball_pivoting
```

## Project Structure

The project is organized into modular components:

- `src/reconstruction/` - Depth image processing and 3D reconstruction
- `src/tracking/` - Feature matching and visual odometry
- `src/stereo/` - Stereo vision algorithms and orientation estimation
- `src/utils/` - Utility functions for image processing and visualization
- `src/interfaces/` - Camera interface abstractions
- `scripts/` - Example scripts demonstrating usage
- `config/` - Configuration files
- `tests/` - Unit tests
- `examples/sample_data/` - Sample data files for examples

## Configuration

Camera calibration parameters can be configured in `config/camera_calibration.yaml`:

```yaml
camera_matrix:
  - [1361.66092, 0, 954.355844]
  - [0, 1353.53462, 539.492797]
  - [0, 0, 1]
distortion_coefficients: [0.1557793, -0.35602181, 0.00255032, 0.00145971, 0.07611791]
depth_scale: 0.001
baseline: 0.05
```