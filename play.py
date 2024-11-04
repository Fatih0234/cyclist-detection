import cv2
import tempfile
from ultralytics import YOLO

# Load YOLO model
model = YOLO('best.pt')

# Constants for distance calculation
ACTUAL_HEIGHT = 1.7  # Real-world height in meters
FOCAL_LENGTH = 800   # Estimated camera focal length in pixels

def calculate_distance(object_height_in_image, actual_height=ACTUAL_HEIGHT, focal_length=FOCAL_LENGTH, scale_factor=0.8):
    """Calculate distance and scale it for closer effect."""
    distance = (actual_height * focal_length) / object_height_in_image
    return distance * scale_factor

def process_image(image):
    """Run YOLO model on the image and add bounding boxes with distance labels."""
    results = model.predict(image)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            object_height = y2 - y1
            distance = calculate_distance(object_height)
            
            # Draw bounding box and add distance text
            color = (0, 255, 0) if distance > 2 else (0, 0, 255)  # Red if closer than 2m
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"Cyclist {distance:.2f} m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image