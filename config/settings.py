import yaml
import os

def load_config(config_path="config/camera_calibration.yaml"):
    """
    Load camera calibration configuration from YAML file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration parameters
    """
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default values.")
        return DEFAULT_INTRINSICS
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Default camera intrinsics
DEFAULT_INTRINSICS = {
    'width': 640,
    'height': 480,
    'fx': 525.0,
    'fy': 525.0,
    'cx': 319.5,
    'cy': 239.5
}