import cv2

def crop_image_with_points(image_path, points):
    # Read the image
    image = cv2.imread(image_path)

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

# Define the image path and points
image_path = '../bus.jpg'
points = [(100, 50), (400, 50), (100, 300), (400, 300)]  # Example points

# Crop the image
cropped_img = crop_image_with_points(image_path, points)

# Save or display the cropped image
# cv2.imshow('Cropped Image', cropped_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('cropped_image.jpg', cropped_img)
