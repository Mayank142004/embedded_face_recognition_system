# 08. AI / ML Pipeline

The AI/ML pipeline is the core of this system. It relies on a three-stage process: Detection, Feature Extraction, and Classification.

## Complete Pipeline Flow

```text
Input Video Frame
 ↓
YOLOv8 (`yolov8n-face.pt`)  <-- Face Detection
 ↓
Cropped Face Region
 ↓
OpenCV (Resize to 160x160 RGB)  <-- Preprocessing
 ↓
FaceNet (`keras-facenet`)  <-- Feature Extraction (Embedding)
 ↓
512-Dimensional Vector
 ↓
SVM (`facenet_models/new_classifier_Jun27_759.pkl`)  <-- Classification
 ↓
Label Encoder (`inverse_transform`)  <-- Post-processing
 ↓
Final Response (Name, Probability)
```

## Stage 1: Face Detection
**Model**: YOLOv8 (nano variant, customized for faces: `yolov8n-face.pt`)
**Location**: `main.py` -> `model(frame)`
**Input**: Full video frame (NumPy array).
**Output**: `Detections` object containing bounding boxes `[x1, y1, x2, y2]`.
**Purpose**: Rapidly locate all faces in the frame. YOLO is chosen for its speed and accuracy in real-time video processing.

## Stage 2: Feature Extraction (Embedding)
**Model**: FaceNet (Inception ResNet v1 architecture, via `keras-facenet` library)
**Location**: `facenet_files/facent_svm_rec_passing.py` -> `get_embedding(face_img)`
**Input Type**: 160x160 RGB image (NumPy array, shape `(1, 160, 160, 3)`).
**Preprocessing**: Converting BGR to RGB and resizing.
**Model Call**: `embedder.embeddings(face_img)`
**Output Type**: 512-dimensional vector (`np.ndarray` of floats).
**Purpose**: FaceNet maps face images to a compact Euclidean space where distances directly correspond to a measure of face similarity.

## Stage 3: Classification
**Model**: Support Vector Machine (SVM) from `scikit-learn` (`sklearn.svm.SVC`)
**Location**: `facenet_files/facent_svm_rec_passing.py` -> `predict_face()`
**Input Type**: 512-dimensional vector.
**Model Call**: 
- `ypreds = model[0].predict(embedding)`
- `ypreds_probability_list = model[0].predict_proba(embedding)`
**Output Type**: Integer index representing the class, and a probability float (`0.0` to `1.0`).
**Purpose**: Classifies the 512D embedding into one of the known employee classes.

## Training Pipeline (`training.py`)
While `main.py` is the inference pipeline, `training.py` shows how the SVM was created:
1. `MTCNN` (Multi-task Cascaded Convolutional Networks) detects and crops faces from the training dataset.
2. `FaceNet` converts all cropped faces into embeddings.
3. A `LabelEncoder` encodes string directory names into integer labels.
4. `SVC(kernel='linear', probability=True)` is trained on the embeddings.
5. The model and classes are saved as a `.pkl` file via `pickle`.
