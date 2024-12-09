#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
from math import floor
import numpy as np
import os

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.model = YOLO('yolov8n.pt')  # Initialize YOLO model with a specific weights file
        self.segmentation_model = YOLO("yolov8n-seg.pt")
        
        # Ensure the directories exist
        self.cropped_images_dir = './cropped_images'
        self.masks_dir = './masks'
        os.makedirs(self.cropped_images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)

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

    def crop_image_with_points(self, points):
        # Read the image
        image = cv2.imread(self.image_path)

        # Ensure points are in the form [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        # Get the top-left and bottom-right coordinates
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        # Calculate the bounding box coordinates
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Crop the image using the bounding box
        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image
    
    def crop_all_images(self):
        i = 1
        for box in self.bounding_boxes:
            cropped_image = self.crop_image_with_points(box)
            cropped_image_path = f'{self.cropped_images_dir}/cropped_image_{i}_RGB.png'
            cv2.imwrite(cropped_image_path, cropped_image)
            i += 1

    def generate_masks(self):
        # Iterate over each image in the cropped_images directory
        for cropped_image_name in os.listdir(self.cropped_images_dir):
            cropped_image_path = os.path.join(self.cropped_images_dir, cropped_image_name)
            img = cv2.imread(cropped_image_path)

            if img is None:
                print(f"Error: Could not read the image {cropped_image_name}")
                continue

            H, W, _ = img.shape

            # Predict masks using the YOLO segmentation model
            results = self.segmentation_model(img, classes=[0])

            # Iterate over each result
            for result in results:
                for j, mask in enumerate(result.masks.data):
                    # Move the mask to CPU and convert to numpy array
                    mask = mask.cpu().numpy() * 255
                    
                    # Ensure the mask is of the correct type and reshape
                    mask = mask.astype(np.uint8)

                    # Resize the mask to the original image size
                    mask = cv2.resize(mask, (W, H))

                    # Construct the mask file name and save it
                    mask_filename = f'{(cropped_image_name.split(".")[0])[:-4]}_Mask.png'
                    mask_path = os.path.join(self.masks_dir, mask_filename)
                    cv2.imwrite(mask_path, mask)
                    print(f"Mask saved successfully: {mask_path}")

if __name__ == '__main__':
    image_path = '/home/anshium/pose/deep-orientation/custom_scenes/indian_bazaar/indian_market.jpg'
    ip = ImageProcessor(image_path)
    ip.process_image()
    ip.crop_all_images()
    ip.generate_masks()
