# Appendix D: AI/ML Reference

### Models Used

| Role | Model | Framework | File Location | Purpose |
| ---- | ----- | --------- | ------------- | ------- |
| **Object Detection** | YOLOv8 (Nano) | Ultralytics | `yolo_models/yolov8n-face.pt` | Quickly draws bounding boxes around all faces in a video frame. |
| **Face Alignment** | MTCNN | `mtcnn` | (Downloaded at runtime) | Detects facial landmarks to align faces properly before training. (Used only in `training.py`) |
| **Feature Extraction** | FaceNet | `keras-facenet` | (TensorFlow weights) | Converts a 160x160 face image into a 512D embedding. |
| **Classification** | SVM (Linear SVC) | Scikit-Learn | `facenet_models/new_classifier_Jun27_759.pkl` | Predicts the identity string (and probability) from the 512D embedding. |

### ML Pipeline Data Flow

1. **Preprocessing (YOLO crop):** Arbitrary sized `numpy` array representing a face bounding box.
2. **Preprocessing (Resize):** `cv2.resize(face, (160, 160))` — Target size required by FaceNet.
3. **Embedding:** `embedder.embeddings(face)` outputs `(1, 512)` shape array.
4. **Inference:** `model.predict(embedding)` and `model.predict_proba(embedding)`.
5. **Post-processing:** The system only accepts the classification if the `predict_proba` is `>= 0.87`.
