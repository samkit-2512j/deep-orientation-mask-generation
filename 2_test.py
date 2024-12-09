from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load YOLO model
model = YOLO("yolov8n-seg.pt")  # Use the correct path for your model if different

# Define a function to process the image
def process_image(image_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return
    
    # Run YOLO model on the image
    results = model(image)
    
    # Get detections and filter only human class
    for i, (result) in enumerate(results):
        # YOLOv8-seg produces `segments` array and `boxes` object, use them appropriately
        segments = result.segments
        boxes = result.boxes
        
        for j, (box, segment) in enumerate(zip(boxes, segments)):
            cls_id = box.cls.item()  # Class ID for the detected object
            cls_name = model.names[int(cls_id)]  # Class name
            
            # Check if the detected class is 'person'
            if cls_name == 'person':
                x1, y1, x2, y2 = box.xyxy.tolist()[0]  # Get bounding box coordinates
                cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
                
                # Save the cropped image
                output_file = os.path.join(output_dir, f"human_{i}_{j}.png")
                cv2.imwrite(output_file, cropped_image)
                print(f"Cropped image saved as {output_file}")

# Define input and output paths
input_image_path = "bus.jpg"  # Replace with your image file
output_directory = "cropped_images"

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process the image
process_image(input_image_path, output_directory)
