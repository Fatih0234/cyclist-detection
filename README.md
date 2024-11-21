
Streamlit App => https://cyclist-detection-tj84rgs7cshsel8zfzpcm5.streamlit.app/

# Cyclist Detection and Distance Estimation

This application leverages a YOLO-based model to detect cyclists on the road and estimate their distance from the camera, enhancing road safety when deployed in vehicles.

# Cyclist Detection and Distance Estimation Model

## About the Model

### Overview
This application leverages a **YOLOv11n model** from Ultralytics, trained on custom data from [Roboflow’s Bicycle Detection Dataset](https://universe.roboflow.com/bicycle-detection/bike-detect-ct/dataset/5), to detect cyclists effectively. The model is fine-tuned to recognize cyclists in various environments, making it suitable for vehicle-mounted cameras.

### Goals
- **Cyclist Detection**: Accurately identifies cyclists on the road for real-time alerts.
- **Distance Estimation**: Calculates the distance to each detected cyclist using geometric parameters.
- **Warning Indicator**: Cyclists closer than **2 meters** are highlighted with a **red bounding box** as a safety alert.
- **Tracking**: Continuously tracks cyclists using the **ByteTrack algorithm** for stable detection across frames.

### Distance Calculation
Accurate distance estimation is essential for road safety. We use a perspective projection formula:

**Distance = (Actual Height * Focal Length) / Image Height**

- **Actual Height** is set to 1.7 meters, representing an average cyclist’s height.
- **Focal Length** is approximated at 800 pixels for this camera.
- A scaling factor is applied to the calculated distance to enhance proximity detection accuracy.

### How It Works
- **Detection**: Each cyclist is enclosed in a bounding box with the distance displayed on it.
- **Tracking**: The ByteTrack algorithm tracks each cyclist across frames for smooth monitoring.
- **Warnings**: Cyclists within **2 meters** trigger a red bounding box, giving drivers a visual alert.

This model is designed to improve **driver awareness** and **road safety**, making it a valuable tool in driver-assistance systems and autonomous vehicles.

## Dataset Augmentation and Refinement

To improve model performance and reduce false positives, we plan to enhance the dataset with additional images, including:
- People without bikes and bikes without people in varied settings.
- Different angles of cyclists, as well as challenging scenarios involving diverse lighting conditions and partial occlusions.

This dataset refinement will allow the model to better distinguish between cyclists and similar objects in various real-world conditions, helping it learn nuanced differences and enhancing detection accuracy.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/cyclist-detection.git
   cd cyclist-detection


## System Related

Note: Ensure that `libGL` is installed on your system. For Ubuntu/Debian, use:
   ```bash
   sudo apt install -y libgl1-mesa-glx