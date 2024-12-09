#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
from math import floor
import numpy as np
import torch
import time
import os

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        
        # Ensure the model uses GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(self.device)
        
        # Initialize the YOLO segmentation model (this model will also provide bounding boxes)
        self.model = YOLO("yolov8n-seg.pt").to(self.device)

        self.cropped_images_dir = './cropped_images'
        self.masks_dir = './masks'
        
        os.system(f"rm -rf {self.cropped_images_dir}/*")
        os.system(f"rm -rf {self.masks_dir}/*")

        self.bounding_boxes = []

    def process_image(self):
        # Read the image from the specified path
        cv_image = cv2.imread(self.image_path)

        if cv_image is None:
            print("Error: Could not read the image from the given path.")
            return

        # Run object detection and segmentation with YOLO segmentation model
        results = self.model(cv_image, classes=[0], device=self.device)  # Class 0 typically represents 'person'

        # Clear previous bounding boxes
        self.bounding_boxes.clear()

        # Iterate over the results to extract bounding boxes and masks
        for result in results[0].boxes:
            bb_coords_list = [int(floor(i)) for i in (list(result.xyxy)[0]).tolist()]
            x1, y1, x2, y2 = bb_coords_list
            
            self.bounding_boxes.append([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            
            print(f"Bounding Box Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
        print("All Bounding Boxes:", self.bounding_boxes)
        
        return results  # Returning results for further use in generating masks

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

    def generate_masks(self, results):
        # Read the image from the specified path
        img = cv2.imread(self.image_path)

        if img is None:
            print("Error: Could not read the image.")
            return

        H, W, _ = img.shape

        start_time = time.time()

        # if results and len(results) > 0:
        #     # Check the device of the first result's tensor
        #     device_used = results[0].boxes.data.device
        #     print(f"Tensors are on device: {device_used}")

        # Iterate over the masks results
        for i, result in enumerate(results[0].masks.data):
            # Move the mask to CPU and convert to numpy array
            mask = result.cpu().numpy() * 255
            
            # Ensure the mask is of the correct type and reshape
            mask = mask.astype(np.uint8)

            # Resize the mask to the original image size
            mask = cv2.resize(mask, (W, H))

            # Crop the mask using the bounding boxes
            # for i, box in enumerate(self.bounding_boxes):
            
            box = self.bounding_boxes[i]
            
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            cropped_mask = mask[y_min:y_max, x_min:x_max]

            mask_path = f'{self.masks_dir}/cropped_image_{i+1}_Mask.png'
            cv2.imwrite(mask_path, cropped_mask)
            print(f"Mask for Bounding Box {i+1} cropped successfully.")

        end_time = time.time()
        print(f"Mask Generation Time: {end_time - start_time:.4f} seconds")


if __name__ == '__main__':
    image_path = '/home/anshium/pose/deep-orientation/custom_scenes/indian_bazaar/indian_market.jpg'
    ip = ImageProcessor(image_path)
    results = ip.process_image()  # Capture the results from process_image
    ip.crop_all_images()  # Pass the results to generate_masks
    ip.generate_masks(results)  # Pass the results to generate_masks
