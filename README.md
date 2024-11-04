# Cyclist Detection and Distance Estimation

This application leverages a YOLO-based model to detect cyclists on the road and estimate their distance from the camera, enhancing road safety when deployed in vehicles.

## Project Overview

This app provides real-time cyclist detection and distance estimation, showcasing the model’s performance on sample images and videos. Users can upload their own images or videos for testing, with results processed and displayed in the app. The application calculates distance to detected cyclists based on real-world height, estimated focal length, and a scaling factor, providing a practical distance estimate.

Key functionalities include:
- Real-time detection and annotation of cyclists.
- Distance calculation and display, with adaptive coloring based on distance (e.g., green if over 2m, red if within 2m).
- Object tracking to maintain cyclist identity across frames, providing insights into cyclist proximity and movement patterns.

## Dataset Augmentation and Refinement

To improve model performance and reduce false positives, we plan to enhance the dataset with additional images, including:
- People without bikes and bikes without people in varied settings.
- Different angles of cyclists, as well as challenging scenarios involving diverse lighting conditions and partial occlusions.

This dataset refinement will allow the model to better distinguish between cyclists and similar objects in various real-world conditions, helping it learn nuanced differences and enhancing detection accuracy.

## Object Tracking Benefits

We consider adding object tracking to enhance detection accuracy and further reduce false positives. Key reasons for implementing object tracking include:
- **Enhanced Accuracy and Reduced False Positives**: Tracking objects across frames helps confirm object identities over time, reducing the likelihood of misclassifications. For example, a moving cyclist is more likely to be a true cyclist, whereas a stationary bike or pedestrian is less likely to be confused as a cyclist.
- **Improved Consistency Across Frames**: Object tracking maintains the identity of a cyclist as they move, making it possible to observe changes in proximity, velocity, and behavior, all of which can be valuable in safety applications.

## Model and Future Improvements

The current model is trained using YOLO with a default confidence threshold (`conf=0.25`). Additional improvements planned include tuning parameters like confidence and IoU thresholds, adjusting input resolutions, and experimenting with custom anchor boxes. These enhancements aim to optimize detection accuracy and performance in real-world applications.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/cyclist-detection.git
   cd cyclist-detection
