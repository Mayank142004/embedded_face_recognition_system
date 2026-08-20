# 02. Complete Architecture

The Face Recognition System relies on a monolithic script-based architecture. Instead of client-server interaction, it operates as a continuous data processing pipeline on a video feed.

## Architectural Layers

```text
Video Source (File / Stream)
  ↓
Video Processing Layer (Supervision)
  ↓
Detection & Tracking Layer (YOLOv8 + ByteTrack)
  ↓
Application Logic Layer (main.py callback)
  ↓
AI / ML Recognition Layer (FaceNet + SVM)
  ↓
Persistence Layer (CSV + OS File System)
  ↓
Annotated Output Video
```

## Component Breakdown

### 1. Video Processing Layer
**Component**: Supervision Video Processor
**Purpose**: Reads the input video frame-by-frame and writes the annotated output video.
**Location**: `main.py` (via `sv.process_video`)
**Input**: Raw video file (`test_datas/testing_video.mp4`)
**Processing**: Calls a custom `callback` function for every frame.
**Output**: Annotated video file (`result_datas/testig_video_result.mp4`)
**Dependencies**: `supervision`
**Called By**: Application Execution
**Calls**: `callback()`

### 2. Detection & Tracking Layer
**Component**: YOLOv8 Face Detector & ByteTracker
**Purpose**: Detect faces in the current frame and maintain identities across consecutive frames.
**Location**: `main.py`
**Input**: Raw image frame (numpy array).
**Processing**: YOLO infers bounding boxes; ByteTrack associates boxes with previous frames to assign tracker IDs.
**Output**: `sv.Detections` object containing bounding boxes, class IDs, and tracker IDs.
**Dependencies**: `ultralytics`, `supervision`
**Called By**: `callback()`
**Calls**: YOLO model (`yolov8n-face.pt`), `tracker.update_with_detections()`

### 3. Application Logic Layer
**Component**: Frame Callback
**Purpose**: The central coordinator for each frame. It handles the "line crossing" logic, coordinates the AI model calls, and manages saving.
**Location**: `main.py` (`callback` function)
**Input**: Raw frame, frame index.
**Processing**: 
1. Runs detection.
2. Draws bounding boxes and the attendance line.
3. Crops faces crossing the line.
4. Triggers recognition.
5. Saves results if confidence is high.
**Output**: Annotated frame.
**Dependencies**: `cv2`, `os`, `csv`, `datetime`
**Called By**: `sv.process_video`
**Calls**: `predict_face()`, `cv.imwrite()`, CSV `writer.writerow()`

### 4. AI / ML Recognition Layer
**Component**: Face Recognition Service
**Purpose**: Identifies the specific person in a cropped face image.
**Location**: `facenet_files/facent_svm_rec_passing.py`
**Input**: Cropped face image (numpy array from OpenCV).
**Processing**: 
1. Resizes to 160x160.
2. Extracts 512D embedding using FaceNet.
3. Predicts identity and probability using a pre-trained SVM model.
**Output**: Label (string name) and Probability (float).
**Dependencies**: `keras_facenet`, `scikit-learn`, `pickle`, `numpy`
**Called By**: `callback()`
**Calls**: `get_embedding()`, `model[0].predict()`, `model[0].predict_proba()`

### 5. Persistence Layer
**Component**: File Storage System
**Purpose**: Saves the attendance log and proof (image).
**Location**: `main.py`
**Input**: Employee name, unique ID, timestamp, face image.
**Processing**: Checks if the folder for the current date exists. If not, creates it. Appends a new row to the CSV. Saves the image as a `.jpg`.
**Output**: Files on disk.
**Dependencies**: `os`, `csv`
**Called By**: `callback()`
