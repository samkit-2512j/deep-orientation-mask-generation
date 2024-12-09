from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("yolov8n-seg.pt")  # load an official model

# Predict with the model
image_path = 'bus.jpg'
img = cv2.imread(image_path)

H, W, _ = img.shape

results = model(image_path)

for result in results:
    for j, mask in enumerate(result.masks.data):
        # Move the mask to CPU and convert to numpy array
        mask = mask.cpu().numpy() * 255
        
        # Ensure the mask is of the correct type and reshape
        mask = mask.astype(np.uint8)

        # Resize the mask to the original image size
        mask = cv2.resize(mask, (W, H))

        # Save the mask as an image
        cv2.imwrite(f'./output_{j}.png', mask)

print("Mask images saved successfully!")
