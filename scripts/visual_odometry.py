#!/usr/bin/env python3

import sys
import os
import argparse
import cv2

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tracking.visual_odometry import VisualOdometry
from config.settings import load_config

def main():
    parser = argparse.ArgumentParser(description='Visual odometry from video or images')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--image1', type=str, help='Path to first image')
    parser.add_argument('--image2', type=str, help='Path to second image')
    parser.add_argument('--config', type=str, default='config/camera_calibration.yaml', 
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    camera_matrix = np.array(config['camera_matrix'])
    
    # Initialize visual odometry
    vo = VisualOdometry(camera_matrix)
    
    if args.video:
        # Process video
        cap = cv2.VideoCapture(args.video)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process frame
            R, t = vo.process_frame(gray)
            print(f"Rotation: {R}")
            print(f"Translation: {t}")
            
            # Display frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    elif args.image1 and args.image2:
        # Process two images
        img1 = cv2.imread(args.image1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(args.image2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print("Error: Could not load images")
            sys.exit(1)
            
        # Process first frame
        vo.process_frame(img1)
        
        # Process second frame
        R, t = vo.process_frame(img2)
        print(f"Rotation: {R}")
        print(f"Translation: {t}")
        
    else:
        print("Error: Either --video or --image1 and --image2 must be specified")
        sys.exit(1)

if __name__ == "__main__":
    main()