# 18. Execution Lifecycle

## 1. Before startup
- Ensure Python 3.11+ is installed.
- Ensure all packages in `requirement_clean.txt` are installed.
- Ensure `yolov8n-face.pt` and `new_classifier_Jun27_759.pkl` are present in their respective directories.
- Provide a target video: `test_datas/testing_video.mp4`.

## 2. During startup (`python main.py`)
- Python parses `main.py`.
- Global imports trigger the loading of the SVM model in `facent_svm_rec_passing.py`.
- `LabelEncoder` is re-fit with the unpickled labels.
- YOLO model is loaded into memory.
- `sv.ByteTrack()`, `sv.BoundingBoxAnnotator()`, and `sv.LabelAnnotator()` are instantiated.

## 3. First request (First Frame)
- `sv.process_video()` reads the first frame.
- Passes the frame to `callback()`.
- YOLO takes slightly longer (initialization overhead) to process the first frame.
- `tracker` initializes a new track for any faces found.
- The first frame is annotated and returned to `sv.process_video`, which initializes the video writer for `result_datas/testig_video_result.mp4`.

## 4. Normal request (Normal Frame)
- Frame read -> YOLO detects -> ByteTrack updates ID -> Frame annotated -> Frame written.

## 5. AI/ML request (Line Crossed)
- When a tracked face's bounding box intersects the horizontal line:
  - Face is cropped and passed to `predict_face()`.
  - FaceNet extracts the embedding.
  - SVM predicts the name.
  - If probability >= 0.87, it moves to the Database Request stage.

## 6. Database request (CSV Logging)
- Checks if the folder `marked_attendance/YYYY_MM_DD` exists.
- Writes the cropped face to `.jpg`.
- Appends the detection data to `YYYY_MM_DD_attendance_sheet.csv`.
- Adds the recognized name to the `saved_names` memory list to prevent logging them again on the next frame.

## 7. Error
- If an exception occurs during annotation, `try/except Exception as E` catches it, prints the error, and skips annotation for that frame.

## 8. Shutdown
- Once the video stream (`test_datas/testing_video.mp4`) ends, `sv.process_video()` closes the video writer and the script terminates naturally.
