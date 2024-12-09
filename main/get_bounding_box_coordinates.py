#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
from math import floor

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.model = YOLO('yolov8n.pt')  # Initialize YOLO model with a specific weights file
    
    def process_image(self):
        # Read the image from the specified path
        cv_image = cv2.imread(self.image_path)

        if cv_image is None:
            print("Error: Could not read the image from the given path.")
            return

        # Run object detection with YOLO model
        results = self.model.track(cv_image, persist=True, classes=[0])  # Tracking class 0 (typically 'person')

        bounding_boxes = []

        # Print bounding box coordinates for each detected person
        for result in results[0].boxes:
            bb_coords_list = [int(floor(i)) for i in (list(result.xyxy)[0]).tolist()]
            x1, y1, x2, y2 = bb_coords_list
            
            bounding_boxes.append([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            
            print(f"Bounding Box Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
        self.bounding_boxes = bounding_boxes
        print(bounding_boxes)

if __name__ == '__main__':
    image_path = '../bus.jpg'
    ip = ImageProcessor(image_path)
    ip.process_image()
