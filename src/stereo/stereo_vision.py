import cv2
import numpy as np

class StereoReconstructor:
    """Class for stereo vision processing to compute disparity and depth maps."""
    
    def __init__(self, calibration_data):
        """
        Initialize the StereoReconstructor with calibration data.
        
        Args:
            calibration_data (dict): Calibration parameters including left/right camera matrices,
                                     distortion coefficients, rotation, translation, and rectification maps.
        """
        self.left_camera_matrix = np.array(calibration_data['left_camera_matrix'])
        self.right_camera_matrix = np.array(calibration_data['right_camera_matrix'])
        self.left_dist_coeffs = np.array(calibration_data['left_dist_coeffs'])
        self.right_dist_coeffs = np.array(calibration_data['right_dist_coeffs'])
        self.R = np.array(calibration_data['R'])  # Rotation from left to right
        self.T = np.array(calibration_data['T'])  # Translation from left to right
        self.image_size = tuple(calibration_data['image_size']) # (width, height)
        
        # Compute rectification maps
        (self.left_rectification, self.right_rectification, 
         self.left_projection, self.right_projection, 
         self.disp_to_depth_map, self.roi_left, self.roi_right) = cv2.stereoRectify(
            self.left_camera_matrix, self.left_dist_coeffs,
            self.right_camera_matrix, self.right_dist_coeffs,
            self.image_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
        )
        
        self.left_map_x, self.left_map_y = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_dist_coeffs, self.left_rectification, 
            self.left_projection, self.image_size, cv2.CV_32FC1)
            
        self.right_map_x, self.right_map_y = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_dist_coeffs, self.right_rectification, 
            self.right_projection, self.image_size, cv2.CV_32FC1)

        # Create Stereo Matcher (SGBM)
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64, # Needs to be divisible by 16
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def rectify_images(self, left_image, right_image):
        """
        Rectify the stereo image pair.
        
        Args:
            left_image (numpy.ndarray): Left image.
            right_image (numpy.ndarray): Right image.
            
        Returns:
            tuple: Rectified left and right images.
        """
        left_rectified = cv2.remap(left_image, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
        return left_rectified, right_rectified

    def compute_disparity(self, left_image, right_image):
        """
        Compute disparity map from rectified stereo images.
        
        Args:
            left_image (numpy.ndarray): Left image (grayscale or color).
            right_image (numpy.ndarray): Right image (grayscale or color).
            
        Returns:
            numpy.ndarray: Disparity map.
        """
        left_rect, right_rect = self.rectify_images(left_image, right_image)
        
        # Convert to grayscale if needed
        if len(left_rect.shape) == 3:
            left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_rect
            right_gray = right_rect
            
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity, left_rect, right_rect # Return rectified images for visualization

    def compute_depth_map(self, disparity_map):
        """
        Convert disparity map to depth map using Q matrix from stereoRectify.
        
        Args:
            disparity_map (numpy.ndarray): Disparity map.
            
        Returns:
            numpy.ndarray: Depth map.
        """
        # Reproject disparity to 3D point cloud
        points_3D = cv2.reprojectImageTo3D(disparity_map, self.disp_to_depth_map)
        # The Z coordinate is the depth
        depth_map = points_3D[:, :, 2]
        return depth_map

# Placeholder for future development
class DisparityRefinement:
    """Class for refining disparity maps (e.g., filtering, hole filling)."""
    def __init__(self):
        raise NotImplementedError("Disparity refinement module is under development.")