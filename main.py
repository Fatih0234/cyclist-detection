import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO('best10.pt')

# Constants for distance calculation
ACTUAL_HEIGHT = 1.7  # Real-world height in meters
FOCAL_LENGTH = 800   # Estimated camera focal length in pixels

def calculate_distance(object_height_in_image, actual_height=ACTUAL_HEIGHT, focal_length=FOCAL_LENGTH, scale_factor=0.8):
    """Calculate distance and scale it for closer effect."""
    distance = (actual_height * focal_length) / object_height_in_image
    return distance * scale_factor

def process_detection(image):
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

def calculate_tracking_distance(bbox_width, focal_length=700, real_object_width=0.5):
    """Calculate distance from bounding box width using a simple inverse relationship."""
    return (focal_length * real_object_width) / bbox_width

def process_tracking_results(results, output_path, conf_threshold=0.35):
    """Process tracking results and save tracked video with bounding boxes, IDs, confidence levels, and distances."""
    global video_writer  # Use global video_writer to initialize once
    video_writer = None

    for result in results:
        frame = result.orig_img  # Original frame with detections and tracking
        tracked_objects = result.boxes  # Tracked objects with bounding boxes and IDs

        # Initialize video writer if not already done (using the frame's dimensions)
        if video_writer is None:
            height, width, _ = frame.shape
            video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        # Draw bounding boxes, IDs, confidence levels, and distances on the frame
        for box in tracked_objects:
            if box.conf[0] < conf_threshold:
                continue  # Skip detections below confidence threshold

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # Bounding box coordinates
            obj_id = int(box.id.item()) if box.id is not None else "N/A"  # Extract integer ID or set to "N/A"
            confidence = box.conf[0]  # Confidence score
            bbox_width = x2 - x1  # Width of the bounding box
            distance = calculate_tracking_distance(bbox_width)  # Calculate distance

            # Set color based on distance
            color = (0, 0, 255) if distance < 2 else (0, 255, 0)  # Red if distance < 2m, otherwise green

            # Draw rectangle, ID, confidence, and distance on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Dist: {distance:.2f} m", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame to the output video
        video_writer.write(frame)

# Streamlit UI
st.title("Cyclist Detection and Distance Estimation Model Showcase")

# About the Model section
st.header("About the Model")

# Overview
st.subheader("Overview")
st.write(
    """
    This application utilizes a **YOLOv11n model** from Ultralytics, trained on custom data from 
    [Roboflow’s Bicycle Detection Dataset](https://universe.roboflow.com/bicycle-detection/bike-detect-ct/dataset/5), to detect cyclists effectively. 
    The model is fine-tuned to recognize cyclists in diverse environments, making it suitable for vehicle-mounted cameras.
    """
)

# Goals
st.subheader("Goals")
st.write(
    """
    - **Cyclist Detection**: Accurately identifies cyclists on the road for real-time alerts.
    - **Distance Estimation**: Calculates the distance to each detected cyclist using geometric parameters.
    - **Warning Indicator**: Cyclists closer than **2 meters** are highlighted with a **red bounding box** as a safety alert.
    - **Tracking**: Continuously tracks cyclists using the **ByteTrack algorithm** for stable detection across frames.
    """
)

# Distance Calculation
st.subheader("Distance Calculation")
st.write(
    """
    Accurate distance estimation is essential for road safety. We use a perspective projection formula:
    
    **Distance = (Actual Height * Focal Length) / Image Height**

    - **Actual Height** is set to 1.7 meters, representing an average cyclist’s height.
    - **Focal Length** is approximated at 800 pixels for this camera.
    - A scaling factor is applied to the calculated distance to enhance proximity detection accuracy.
    """
)

# How It Works
st.subheader("How It Works")
st.write(
    """
    - **Detection**: Each cyclist is enclosed in a bounding box with the distance displayed on it.
    - **Tracking**: The ByteTrack algorithm tracks each cyclist across frames for smooth monitoring.
    - **Warnings**: Cyclists within **2 meters** trigger a red bounding box, giving drivers a visual alert.

    This model is designed to improve **driver awareness** and **road safety**, making it a useful tool in 
    driver-assistance systems and autonomous vehicles.
    """
)


# Example media paths
example_images = ["example1.jpg", "example2.jpg"]
preprocessed_videos = ["processed_example_video1.mp4", "processed_example_video2.mp4"]

# Check and display example images
st.header("Model Performance Examples")
for img_path in example_images:
    if os.path.isfile(img_path):
        st.subheader(f"Example Image: {img_path}")
        original_image = cv2.imread(img_path)  # Read the original image
        processed_image = process_detection(original_image.copy())  # Process a copy of the original image
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image", channels="BGR")  # Display original image
        with col2:
            st.image(processed_image, caption="Processed Image", channels="BGR")  # Display processed image
    else:
        st.warning(f"Image file {img_path} not found.")


st.subheader("Example Videos")
for idx, video_path in enumerate(preprocessed_videos, start=1):
    # Print the video path for debugging
    st.write(f"Checking Video Path: {video_path}")

    # Check if the video file exists
    if os.path.isfile(video_path):
        st.write(f"Processed Example Video {idx}")
        
        # Display the video
        try:
            st.video(video_path)
        except Exception as e:
            st.error(f"Error playing video: {str(e)}")
    else:
        st.warning(f"Video file {video_path} not found.")


# User Upload Section for Image and Video Processing
st.header("Try It Yourself")
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file:
        st.session_state.last_uploaded_file = uploaded_file

        # Use a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name  # Get the temp file path

        # Debug statement to check the uploaded file path
        st.write(f"Uploaded file saved to: {tmp_file_path}")

        if uploaded_file.type in ["image/jpeg", "image/png"]:
            # Process and display image in real-time
            image = cv2.imread(tmp_file_path)
            if image is not None:
                processed_image = process_detection(image)
                st.image(processed_image, channels="BGR", caption="Processed Image")
                
                # Save processed image to session for download
                is_success, buffer = cv2.imencode(".jpg", processed_image)
                st.session_state.processed_image = buffer.tobytes()
            else:
                st.error("Error loading image.")

        elif uploaded_file.type == "video/mp4":
            st.warning("Video processing can take some time. You may download the processed video once it's ready.")
            cap = cv2.VideoCapture(tmp_file_path)

            if not cap.isOpened():
                st.error("Error opening video file.")
                st.stop()

            output_path = "processed_user_video.mp4"

            # Get video properties and set up VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            results_bytetrack = model.track(source=tmp_file_path, stream=True, tracker="bytetrack.yaml", conf=0.35)

            # Process the tracking results and save the output video
            process_tracking_results(results_bytetrack, output_path)

            cap.release()
            video_writer.release()

            st.session_state.processed_video_path = output_path

        st.success("Processing complete!")

# Download buttons
if "processed_image" in st.session_state:
    st.download_button("Download Processed Image", st.session_state.processed_image, file_name="processed_image.jpg")

if "processed_video_path" in st.session_state:
    with open(st.session_state.processed_video_path, "rb") as video_file:
        st.download_button("Download Processed Video", video_file, file_name="processed_video.mp4")
