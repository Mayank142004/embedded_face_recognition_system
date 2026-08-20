# 03. Directory Structure

```text
FaceRecognitionSystem/
├── main.py
│   ├── Purpose: The core entry point for the face recognition attendance system.
│   ├── Important classes: N/A
│   ├── Important functions: `callback()`
│   ├── Called by: User execution (`python main.py`)
│   └── Calls: `predict_face()`, `sv.process_video()`, YOLO model
├── training.py
│   ├── Purpose: Prepares FaceNet embeddings from cropped faces and trains the SVM model.
│   ├── Important classes: `FACELOADING`
│   ├── Important functions: `get_embedding()`, `load_classes()`, `extract_face()`
│   ├── Called by: Admin/Developer during setup
│   └── Calls: MTCNN, `keras_facenet.FaceNet`, `sklearn.svm.SVC`
├── crop_face_images_using_yolo.py
│   ├── Purpose: A utility script to crop faces from a dataset using YOLO instead of MTCNN.
│   ├── Important classes: N/A
│   ├── Important functions: `process_image()`
│   ├── Called by: Admin/Developer during setup
│   └── Calls: YOLO model, `cv2.imwrite()`
├── Dockerfile
│   ├── Purpose: Containerizes the application.
│   ├── Important classes: N/A
│   ├── Important functions: N/A
│   ├── Called by: `docker build`
│   └── Calls: N/A
├── requirement.txt & requirement_clean.txt
│   ├── Purpose: Lists Python dependencies.
│   ├── Important classes: N/A
│   ├── Important functions: N/A
│   ├── Called by: `pip install -r`
│   └── Calls: N/A
├── facenet_files/
│   ├── facent_svm_rec_passing.py
│   │   ├── Purpose: Provides the inference pipeline combining FaceNet and the loaded SVM model.
│   │   ├── Important classes: N/A
│   │   ├── Important functions: `predict_face()`, `get_embedding()`, `write_labels_to_file()`
│   │   ├── Called by: `main.py`
│   │   └── Calls: `keras_facenet.FaceNet`, `model[0].predict()`
│   └── (Backup & copy files: `facenet_svm_passing_withoutNPZ.py`, etc.)
├── facenet_models/
│   └── new_classifier_Jun27_759.pkl
│       ├── Purpose: The pre-trained SVM model and its label encoder classes.
│       ├── Important classes: `sklearn.svm.SVC`
│       ├── Important functions: N/A
│       ├── Called by: `facent_svm_rec_passing.py`
│       └── Calls: N/A
├── yolo_models/
│   └── yolov8n-face.pt
│       ├── Purpose: YOLOv8 weights trained for face detection.
│       ├── Important classes: N/A
│       ├── Important functions: N/A
│       ├── Called by: `main.py`
│       └── Calls: N/A
├── supervision/
│   ├── Purpose: A local copy of the `supervision` computer vision framework for drawing boxes and tracking.
│   ├── Important classes: `sv.Detections`, `sv.ByteTrack`, `sv.BoundingBoxAnnotator`
│   ├── Important functions: `process_video()`, `update_with_detections()`
│   ├── Called by: `main.py`
│   └── Calls: N/A
├── marked_attendance/
│   ├── Purpose: The directory where output results (cropped faces and CSV files) are stored dynamically by date.
│   ├── Important classes: N/A
│   ├── Important functions: N/A
│   ├── Called by: `main.py`
│   └── Calls: N/A
├── yolo_with_facenet_svm/
│   ├── Purpose: Contains various experimental scripts (e.g., `yolo_with_facenet_webcam.py`, `rtsp.py`) for running the system on webcams, RTSP streams, and images.
│   ├── Important classes: N/A
│   ├── Important functions: `predict_face()`
│   ├── Called by: Admin/Developer optionally
│   └── Calls: `cv2.VideoCapture()`, `predict_face()`
└── test_datas/ & result_datas/
    ├── Purpose: Stores the input test videos and the annotated output videos.
    ├── Important classes: N/A
    ├── Important functions: N/A
    ├── Called by: `main.py`
    └── Calls: N/A
```
