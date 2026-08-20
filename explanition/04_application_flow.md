# 04. Complete Application Flow

## Startup Flow

```text
Application starts (`python main.py`)
 ↓
Entry point (`main.py`)
 ↓
Model initialization
  - Loads YOLOv8 (`yolov8n-face.pt`)
  - Loads FaceNet (implicitly via import `facent_svm_rec_passing.py`)
  - Loads SVM Model (`facenet_models/new_classifier_Jun27_759.pkl`)
 ↓
Tracking and Annotation Initialization
  - `sv.ByteTrack()` initialized
  - `sv.BoundingBoxAnnotator()` initialized
  - `sv.LabelAnnotator()` initialized
 ↓
Application ready
 ↓
Video Processing starts (`sv.process_video()`)
```

## Frame Processing Lifecycle

For every frame in `test_datas/testing_video.mp4`:

1. `sv.process_video()` reads a frame and calls `callback(frame, frame_index)`.
2. **Detection**: `model(frame)` runs YOLO to detect faces.
3. **Tracking**: `tracker.update_with_detections()` updates IDs based on previous frames.
4. **Line Drawing**: An attendance line is drawn in the middle of the frame:
   `cv.line(annotated_frame, start_point, end_point, color, thickness)`
5. **Face Extraction**: For each detected face bounding box:
   - The bounding box coordinates `x1, y1, x2, y2` are extracted.
   - The face is cropped from the frame: `face = frame[y1:y2, x1:x2]`.
6. **Recognition**: `predict_face(face)` is called (located in `facent_svm_rec_passing.py`).
   - The cropped face is converted to RGB and resized to 160x160.
   - `get_embedding(face)` creates a 512D array via FaceNet.
   - The SVM model predicts the identity and returns `name` and `result_probability`.
7. **Attendance Logic**: The code checks:
   - `if (y1 <= line_y <= y2)`: Did the face cross the middle line?
   - `and result_probability >= 0.87`: Is the system confident?
   - `and name not in saved_names`: Has the person not been logged yet?
8. **Logging (If conditions are met)**:
   - Saves the cropped face: `cv.imwrite(filepath, face)`.
   - Generates a UUID.
   - Appends a new row to `{current_date}_attendance_sheet.csv` with `Name`, `UniqueID`, `Timestamp`, and `Hyperlink`.
   - Adds the name to `saved_names` memory list to avoid duplicates.
9. **Annotation**: The `annotated_frame` is updated with bounding boxes and labels (showing tracker ID, name, probability).
10. **Live Snapshot**: The frame is saved as `live.png`.
11. **Return**: The `callback` returns the `annotated_frame`, which `supervision` writes to `result_datas/testig_video_result.mp4`.
