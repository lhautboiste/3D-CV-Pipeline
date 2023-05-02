from abc import ABC, abstractmethod
import numpy as np

class CameraInterface(ABC):
    """Abstract base class for camera interfaces"""
    
    @abstractmethod
    def initialize(self):
        """Initialize the camera"""
        pass
    
    @abstractmethod
    def capture_frames(self):
        """Capture frames from the camera"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the camera"""
        pass
    
    @abstractmethod
    def get_intrinsics(self):
        """Get camera intrinsics"""
        pass