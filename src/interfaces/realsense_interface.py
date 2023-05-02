import pyrealsense2 as rs
import numpy as np
from .camera_interface import CameraInterface

class RealSenseCamera(CameraInterface):
    """Intel RealSense camera interface implementation"""
    
    def __init__(self, config=None):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.setup_config(config)
        self.initialized = False
        
    def setup_config(self, config):
        """Configure the RealSense streams"""
        # Enable streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        
    def initialize(self):
        """Initialize the RealSense camera"""
        try:
            self.profile = self.pipeline.start(self.config)
            self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            
            # Get intrinsics
            self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics
            self.color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize RealSense camera: {e}")
            return False
        
    def capture_frames(self):
        """Capture frames from the RealSense camera"""
        if not self.initialized:
            raise RuntimeError("Camera not initialized. Call initialize() first.")
            
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        infrared_frame = aligned_frames.get_infrared_frame(1)
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
        infrared_image = np.asanyarray(infrared_frame.get_data()) if infrared_frame else None
        
        return {
            'depth': depth_image,
            'color': color_image,
            'infrared': infrared_image,
            'depth_scale': self.depth_scale,
            'depth_intrinsics': self.depth_intrinsics,
            'color_intrinsics': self.color_intrinsics
        }
        
    def stop(self):
        """Stop the RealSense camera"""
        self.pipeline.stop()
        self.initialized = False
        
    def get_intrinsics(self):
        """Get camera intrinsics"""
        if not self.initialized:
            raise RuntimeError("Camera not initialized. Call initialize() first.")
            
        return {
            'depth': self.depth_intrinsics,
            'color': self.color_intrinsics
        }