import cv2
import numpy as np

class FeatureMatcher:
    """Class for feature detection and matching"""
    
    def __init__(self, detector_type='ORB', matcher_type='BF'):
        """
        Initialize the FeatureMatcher
        
        Args:
            detector_type (str): Type of feature detector ('ORB', 'SIFT', 'AKAZE')
            matcher_type (str): Type of feature matcher ('BF')
        """
        self.detector = self._create_detector(detector_type)
        self.matcher = self._create_matcher(matcher_type)
        
    def _create_detector(self, detector_type):
        """Create feature detector based on type"""
        if detector_type == 'ORB':
            return cv2.ORB_create()
        elif detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif detector_type == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
            
    def _create_matcher(self, matcher_type):
        """Create feature matcher based on type"""
        if matcher_type == 'BF':
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            raise ValueError(f"Unsupported matcher type: {matcher_type}")
            
    def detect_and_compute(self, image):
        """
        Detect keypoints and compute descriptors
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: Keypoints and descriptors
        """
        return self.detector.detectAndCompute(image, None)
        
    def match_features(self, des1, des2):
        """
        Match features between two descriptor sets
        
        Args:
            des1: Descriptors from first image
            des2: Descriptors from second image
            
        Returns:
            list: Matched features
        """
        if des1 is None or des2 is None:
            return []
            
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
        
    def get_matched_points(self, kp1, kp2, matches):
        """
        Get matched points from keypoints and matches
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Matched features
            
        Returns:
            tuple: Matched points from first and second images
        """
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return points1, points2
        
    def filter_matches_with_fundamental_matrix(self, points1, points2, method=cv2.FM_RANSAC):
        """
        Filter matches using fundamental matrix estimation
        
        Args:
            points1: Points from first image
            points2: Points from second image
            method: Method for fundamental matrix estimation
            
        Returns:
            tuple: Inlier points, fundamental matrix, and mask
        """
        if len(points1) < 8:
            return points1, points2, None, None
            
        F, mask = cv2.findFundamentalMat(points1, points2, method)
        if mask is not None:
            inlier_points1 = points1[mask.ravel() == 1]
            inlier_points2 = points2[mask.ravel() == 1]
            return inlier_points1, inlier_points2, F, mask
        else:
            return points1, points2, None, None