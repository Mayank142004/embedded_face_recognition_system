# Model Reference

| Model | Location | Input | Output | Purpose | Configuration |
| ----- | -------- | ----- | ------ | ------- | ------------- |
| **YOLOv8** | `yolo_models/yolov8n-face.pt` | Video Frame / Image | Bounding Box Coordinates | Rapid face detection | Loaded via `ultralytics.YOLO` |
| **FaceNet** | Downloads at runtime (or uses local `keras_facenet`) | 160x160 RGB Face Image | 512D Embedding Vector | Feature extraction / Face Mapping | `embedder = FaceNet()` |
| **SVM Classifier** | `facenet_models/new_classifier_Jun27_759.pkl` | 512D Embedding Vector | Class Index & Probabilities | Predicting employee identity | `SVC(kernel='linear', probability=True)` |
| **MTCNN** | Loaded via `mtcnn` package | Static Image | Bounding Boxes | Face detection/alignment for training dataset | `detector = MTCNN()` |
