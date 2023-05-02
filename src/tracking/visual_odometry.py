import numpy as np
import cv2
from .feature_matching import FeatureMatcher

class VisualOdometry:
    """Class for visual odometry using feature matching"""
    
    def __init__(self, camera_matrix, dist_coeffs=None, detector_type='ORB'):
        """
        Initialize the VisualOdometry
        
        Args:
            camera_matrix (numpy.ndarray): Camera intrinsic matrix
            dist_coeffs (numpy.ndarray): Distortion coefficients
            detector_type (str): Type of feature detector
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.feature_matcher = FeatureMatcher(detector_type)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.pose = (np.eye(3), np.zeros((3, 1)))
        
    def process_frame(self, frame):
        """
        Process a frame for visual odometry
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: Rotation matrix and translation vector
        """
        if self.prev_frame is None:
            self.prev_frame = frame
            self.prev_kp, self.prev_des = self.feature_matcher.detect_and_compute(frame)
            return self.pose
            
        # Detect features in current frame
        kp_cur, des_cur = self.feature_matcher.detect_and_compute(frame)
        
        if des_cur is not None and self.prev_des is not None:
            # Match features
            matches = self.feature_matcher.match_features(des_cur, self.prev_des)
            
            if len(matches) > 8:  # Minimum points for essential matrix
                # Get matched points
                src_pts, dst_pts = self.feature_matcher.get_matched_points(kp_cur, self.prev_kp, matches)
                
                # Filter matches with fundamental matrix
                src_pts, dst_pts, F, mask = self.feature_matcher.filter_matches_with_fundamental_matrix(
                    src_pts, dst_pts
                )
                
                if len(src_pts) >= 8:
                    # Compute essential matrix
                    E, mask = cv2.findEssentialMat(
                        src_pts, dst_pts, self.camera_matrix, 
                        method=cv2.RANSAC, prob=0.999, threshold=1.0
                    )
                    
                    # Recover pose
                    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix, mask=mask)
                    
                    # Update overall pose
                    prev_R, prev_t = self.pose
                    self.pose = (R @ prev_R, R @ prev_t + t)
        
        # Update previous frame data
        self.prev_frame = frame
        self.prev_kp = kp_cur
        self.prev_des = des_cur
        
        return self.pose
        
    def reset(self):
        """Reset the visual odometry state"""
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.pose = (np.eye(3), np.zeros((3, 1)))