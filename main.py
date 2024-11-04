import streamlit as st
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

st.title("Cyclist Detection and Distance Estimation Model Showcase")

# Introduction section
st.header("About the Model")
st.write(
    """
    This model detects cyclists in images and videos, estimates their distance from the camera, 
    and highlights them with bounding boxes. The distance is color-coded: **green** for more than 
    2 meters away and **red** for less than 2 meters, allowing users to identify close proximity 
    of cyclists effectively.
    """
)

# Example Media Display
st.header("Model Performance Examples")

# Example Images Displayed with Real-Time Processing
example_images = ["example1.jpg", "example2.jpg"]
for img_path in example_images:
    st.subheader(f"Example Image: {img_path}")
    image = cv2.imread(img_path)
    processed_image = process_image(image.copy())
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", channels="BGR")
    with col2:
        st.image(processed_image, caption="Processed Image", channels="BGR")

# Pre-processed Example Videos
st.subheader("Example Videos")
st.write("Below are two pre-processed videos that demonstrate the model's performance on video input.")

# Display two pre-processed video files
preprocessed_videos = ["processed_example_video1.mp4", "processed_example_video2.mp4"]
for idx, video_path in enumerate(preprocessed_videos, start=1):
    st.write(f"Processed Example Video {idx}")
    st.video(video_path)

# User Upload Section for Image and Video Processing
st.header("Try It Yourself")
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file:
        st.session_state.last_uploaded_file = uploaded_file

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if uploaded_file.type in ["image/jpeg", "image/png"]:
            # Process and display image in real-time
            image = cv2.imread(tmp_file_path)
            processed_image = process_image(image)
            
            st.image(processed_image, channels="BGR", caption="Processed Image")
            
            # Save processed image to session for download
            is_success, buffer = cv2.imencode(".jpg", processed_image)
            st.session_state.processed_image = buffer.tobytes()

        elif uploaded_file.type == "video/mp4":
            # Notify user that video processing will take time and allow them to download it once complete
            st.warning("Video processing can take some time. You may download the processed video once it's ready.")
            cap = cv2.VideoCapture(tmp_file_path)
            output_path = "processed_user_video.mp4"
            
            # Get video properties and set up VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = process_image(frame)
                out.write(processed_frame)

            cap.release()
            out.release()

            st.session_state.processed_video_path = output_path

        st.success("Processing complete!")

# Download buttons
if "processed_image" in st.session_state:
    st.download_button("Download Processed Image", st.session_state.processed_image, file_name="processed_image.jpg")

if "processed_video_path" in st.session_state:
    with open(st.session_state.processed_video_path, "rb") as video_file:
        st.download_button("Download Processed Video", video_file, file_name="processed_video.mp4")