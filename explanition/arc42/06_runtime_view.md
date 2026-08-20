# 06. Runtime View

This chapter explains the dynamic behavior of the system, outlining how components interact during execution. 

*(Note: Corresponding Mermaid sequence diagrams are available in the `diagrams/` folder.)*

### Scenario 1 — Application Startup

**Trigger:** Administrator runs `python main.py` or starts the Docker container.
**Flow:**
1. Imports libraries (`numpy`, `supervision`, `ultralytics`, `keras_facenet`, etc.).
2. Loads the YOLOv8 face model (`yolo_models/yolov8n-face.pt`).
3. Initializes the ByteTracker (`sv.ByteTrack()`).
4. Executes `facent_svm_rec_passing.py` initialization (loads FaceNet weights and the SVM `.pkl` model).
5. Initializes Annotators (`sv.BoundingBoxAnnotator`, `sv.LabelAnnotator`).
6. Starts `sv.process_video()`, opening `test_datas/testing_video.mp4` for reading.

### Scenario 2 — Normal Video Frame (No Line Crossing)

**Trigger:** `sv.process_video` reads a new frame and calls `callback(frame, int)`.
**Flow:**
1. YOLO infers bounding boxes on the frame.
2. ByteTrack updates tracking IDs.
3. The system draws the green "Attendance Line" across the middle of the frame (`y = height // 2 - 50`).
4. Iterates over detections.
5. Since the bounding boxes do *not* intersect the line (`y1 <= line_y <= y2` is False), the Face Recognition Service is **not** called.
6. Supervision Annotators draw boxes and generic labels (e.g., `#1 face`).
7. Frame is written to `live.png` and output video.

### Scenario 3 — AI/ML Request (Attendance Trigger)

**Trigger:** An employee's face bounding box crosses the green Attendance Line.
**Flow:**
1. `callback()` detects that `(y1 <= line_y <= y2)` is True.
2. The face is cropped from the frame: `face = frame[y1:y2, x1:x2]`.
3. Calls `predict_face(face)`.
   - `predict_face` resizes to 160x160.
   - Extracts 512D embedding via FaceNet.
   - Predicts Name and Probability via SVM.
4. If Probability >= 0.87 and Name is not in `saved_names`:
   - System generates a UUID and Timestamp.
   - Saves cropped face via `cv.imwrite()`.
   - Appends Name, UUID, Timestamp, and Image Hyperlink to today's CSV file.
   - Adds Name to `saved_names` list to prevent duplicate logging for the remainder of the session.
5. Updates frame labels with the predicted Name and Probability.

### Scenario 4 — Offline Model Training

**Trigger:** Administrator runs `python training.py` to add new employees.
**Flow:**
1. `FACELOADING.load_classes()` reads images from `facenet_files/dataset2/`.
2. MTCNN detects and aligns faces.
3. FaceNet generates embeddings.
4. Scikit-Learn trains an SVC (`kernel='linear'`).
5. Model is saved to `.pkl` via `pickle`.
