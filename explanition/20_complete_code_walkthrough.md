# 20. Complete Code Walkthrough

This walkthrough traces the exact path data takes from the moment you run `python main.py`.

### 1. Initialization
**File:** `main.py`
**Lines:** 18-26
```python
model = YOLO("yolo_models/yolov8n-face.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
saved_names = []
```
- **What happens:** The YOLO model, tracking engine, and annotators are loaded into memory. A list `saved_names` is initialized to keep track of who has already been logged.

### 2. Video Processing Trigger
**File:** `main.py`
**Lines:** 146-150
```python
sv.process_video(
    source_path="test_datas/testing_video.mp4",
    target_path="result_datas/testig_video_result.mp4",
    callback=callback
)
```
- **What happens:** The `supervision` library starts reading the video file frame-by-frame and invokes `callback` for each one.

### 3. The Callback (Detection)
**File:** `main.py`
**Lines:** 28-30
```python
results = model(frame)[0]
detections = sv.Detections.from_ultralytics(results)
detections = tracker.update_with_detections(detections)
```
- **What happens:** The current frame is passed to YOLO. The raw results are converted to a standard `sv.Detections` format. ByteTrack assigns persistent IDs to the faces.

### 4. Drawing the Virtual Line
**File:** `main.py`
**Lines:** 54-60
```python
height, width, _ = frame.shape
line_y = (height // 2) - 50
start_point = (0, line_y) 
end_point = (width, line_y)
cv.line(annotated_frame, start_point, end_point, color, thickness)
```
- **What happens:** A horizontal line is drawn slightly above the middle of the frame. This acts as the trigger point for marking attendance.

### 5. Iterating Over Faces
**File:** `main.py`
**Lines:** 62-67
```python
for detection,label in zip(detections.xyxy,labels):
    x1, y1, x2, y2 = map(int, detection[:4])
    face = frame[y1:y2, x1:x2]
```
- **What happens:** The bounding box coordinates are extracted, and the actual face pixels are cropped from the frame.

### 6. Face Recognition
**File:** `main.py` -> `facenet_files/facent_svm_rec_passing.py`
**Line:** 72 (`main.py`), 45-68 (`facent_svm_rec_passing.py`)
```python
facenet_result, result_probabilty = predict_face(face)
```
- **What happens:** The cropped face is sent to the AI service. It is resized, embedded by FaceNet, and the SVM predicts who it is and with what confidence.

### 7. Directory & CSV Setup
**File:** `main.py`
**Lines:** 79-102
- **What happens:** The script checks today's date, creates `marked_attendance/YYYY_MM_DD`, and creates `YYYY_MM_DD_attendance_sheet.csv` with headers if it doesn't exist.

### 8. Logging Attendance
**File:** `main.py`
**Lines:** 107-119
```python
if (y1 <= line_y <= y2) and result_probabilty >=0.87 and name not in saved_names :
    cv.imwrite(filepath,face)
    saved_names.append(first_name[0])
    with open(csv_file_path,mode='a',newline='') as file:
        writer.writerow([first_name[0],unique_id,timestamp,hyperlink])
```
- **What happens:** If the face bounding box crosses the virtual line, the confidence is over 87%, and they haven't been logged yet:
  1. Save the face image.
  2. Add their name to `saved_names` so they aren't logged again in the next frame.
  3. Write a row to the CSV.

### 9. Return Annotated Frame
**File:** `main.py`
**Lines:** 132-142
- **What happens:** The frame is annotated with the name and probability, saved as `live.png` for a quick snapshot, and returned to the video writer.
