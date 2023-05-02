import cv2
import numpy as np

def load_depth_image(depth_image_path):
    """
    Load a depth image and convert to proper format
    
    Args:
        depth_image_path (str): Path to the depth image
        
    Returns:
        numpy.ndarray: Processed depth image
    """
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"Could not load image from {depth_image_path}")
        
    if depth_image.dtype != np.uint16:
        depth_image = cv2.convertScaleAbs(depth_image, alpha=(65535.0 / 255.0))
        depth_image = depth_image.astype(np.uint16)

    # Extract the red channel as the depth information if it's a 3-channel image
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]

    return depth_image

def load_image(image_path, as_gray=False):
    """
    Load an image from file
    
    Args:
        image_path (str): Path to the image
        as_gray (bool): Whether to load as grayscale
        
    Returns:
        numpy.ndarray: Loaded image
    """
    if as_gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(image_path, cv2.IMREAD_COLOR)

def preprocess_image(image, resize_ratio=1.0, apply_gaussian=True, normalize=True):
    """
    Preprocess an image for computer vision tasks
    
    Args:
        image (numpy.ndarray): Input image
        resize_ratio (float): Ratio to resize the image
        apply_gaussian (bool): Whether to apply Gaussian blur
        normalize (bool): Whether to normalize the image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    if resize_ratio != 1.0:
        image = cv2.resize(image, (0, 0), fx=resize_ratio, fy=resize_ratio)
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if apply_gaussian:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    
    if normalize:
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    return image

def detect_edges(image, low_threshold=100, high_threshold=200):
    """
    Detect edges in an image using Canny edge detection
    
    Args:
        image (numpy.ndarray): Input image
        low_threshold (int): Low threshold for Canny
        high_threshold (int): High threshold for Canny
        
    Returns:
        numpy.ndarray: Edge image
    """
    return cv2.Canny(image, low_threshold, high_threshold)