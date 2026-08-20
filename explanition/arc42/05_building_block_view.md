# 05. Building Block View

This chapter describes the static structure of the Face Recognition System.

## Level 1 — System

```text
Face Recognition System
```
The overall system that processes video and outputs CSV attendance logs.

## Level 2 — Major Building Blocks

Based on the actual monolithic implementation in `main.py`:

```text
Face Recognition System
├── Video Processing (Supervision)
├── Detection & Tracking (YOLOv8 + ByteTrack)
├── Application Logic Layer
├── Face Recognition Service (FaceNet + SVM)
└── Persistence (CSV Writer)
```

## Level 3 — Internal Building Blocks

### 3.1 Application Logic Layer (The Coordinator)
**Building Block:** Frame Callback Coordinator
**Purpose:** Orchestrates the flow for each frame. Defines the "line crossing" business logic.
**Location:** `main.py` (function `callback(frame, frame_idx)`)
**Interfaces:** `supervision.process_video`
**Dependencies:** OpenCV (`cv2`), `uuid`, `csv`, `datetime`
**Input:** Raw `np.ndarray` frame
**Output:** Annotated `np.ndarray` frame
**Calls:** YOLO model, ByteTrack, Face Recognition Service, Persistence System.

### 3.2 Detection & Tracking
**Building Block:** YOLO Face Detector
**Purpose:** Rapidly locate faces in an image.
**Location:** `main.py` (instantiation of `model = YOLO("yolo_models/yolov8n-face.pt")`)
**Dependencies:** `ultralytics`
**Input:** `np.ndarray` frame
**Output:** Ultralytics Results object.

**Building Block:** ByteTracker
**Purpose:** Maintain temporal identity of bounding boxes across frames.
**Location:** `main.py` (`tracker = sv.ByteTrack()`)
**Dependencies:** `supervision`
**Input:** Detections from YOLO
**Output:** Detections with `tracker_id`.

### 3.3 Face Recognition Service
**Building Block:** Embedding & Classification
**Purpose:** Identify a specific person from a cropped face image.
**Location:** `facenet_files/facent_svm_rec_passing.py`
**Dependencies:** `keras_facenet`, `scikit-learn`, `pickle`, `numpy`
**Input:** Cropped face image (numpy array, any size)
**Processing:**
1. Resizes to 160x160.
2. `FaceNet` generates a 512-dimensional embedding.
3. `SVC` predicts the identity string and probability.
**Output:** `(name: str, probability: float)`

### 3.4 Training Pipeline (Offline Block)
**Building Block:** MTCNN + FaceNet + SVM Trainer
**Purpose:** Train the SVM classifier on new employee datasets.
**Location:** `training.py`
**Dependencies:** `mtcnn`, `keras_facenet`, `scikit-learn`
**Processing:**
1. Reads raw images from `facenet_files/dataset2`.
2. Extracts aligned faces using MTCNN.
3. Generates embeddings using FaceNet.
4. Trains an SVM classifier.
5. Saves model to `facenet_models/new_classifier_Jun27_759.pkl`.

### 3.5 Persistence System
**Building Block:** CSV Log Writer & Image Saver
**Purpose:** Save the attendance record and proof image.
**Location:** `main.py`
**Dependencies:** `os`, `csv`, `cv2`
**Input:** Employee Name, Probability, Face Crop Array.
**Output:** Appends row to CSV in `marked_attendance/YYYY_MM_DD/`. Saves `.jpg` image.
