# 01. Project Overview

## What problem does this project solve?
This project automates employee attendance tracking by replacing manual logs, ID cards, or fingerprint scanners with facial recognition technology. It allows organizations to passively monitor an area (like an office entrance) and automatically record when recognized employees enter or exit. It captures their image, timestamps their arrival, and saves the data in a standardized CSV format for easy review by administrators.

## What happens when a user uses the system?
1. The system processes a continuous video feed or a recorded video.
2. A user (employee) walks into the camera's field of view.
3. The system immediately detects the user's face and assigns a tracking ID.
4. When the user crosses a virtual "attendance line" drawn in the middle of the frame:
   - The system crops their face from the frame.
   - It passes the cropped face to a machine learning pipeline (FaceNet + SVM).
   - If the system recognizes the face with a high degree of confidence (>= 87%), it registers the attendance.
   - The user's face crop is saved locally, and their ID, Name, Timestamp, and a link to the saved image are recorded in a daily CSV file.
5. The processed video frame is updated with bounding boxes and labels showing the user's identity and prediction probability.

## Technology Stack

| Layer | Technology | Purpose | Actual Files |
|-------|------------|---------|--------------|
| Application Entry | Python 3.11 | Core execution logic | `main.py`, `training.py` |
| Image Processing | OpenCV (`cv2`) | Frame reading, drawing boxes, lines, resizing | `main.py`, `training.py` |
| Detection & Tracking | YOLOv8 + ByteTrack (`supervision`) | Fast face detection and continuous tracking | `main.py`, `yolo_models/yolov8n-face.pt` |
| Feature Extraction | FaceNet (`keras-facenet`) | Converting face images into 512D embeddings | `facenet_files/facent_svm_rec_passing.py`, `training.py` |
| Classification | Scikit-Learn (SVM) | Predicting the employee identity from embeddings | `facenet_models/new_classifier_Jun27_759.pkl` |
| Face Alignment | MTCNN | Extracting aligned faces for model training | `training.py` |
| Storage / Database | CSV & Local File System | Logging attendance data and saving cropped faces | `main.py` (`csv` module) |
| Containerization | Docker | Packaging the environment | `Dockerfile` |

> Note: The project does not currently use a dedicated Backend API framework (like Flask or FastAPI) or a traditional Relational Database (like PostgreSQL), relying instead on local script execution and CSV file storage.
