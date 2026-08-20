# 11. Data Flow Diagrams

## Core Face Recognition Data Flow

```text
SOURCE
 Video File / Stream (`test_datas/testing_video.mp4`)
 ↓
VALIDATION
 YOLO Detection Confidence (Implicit within YOLO model)
 ↓
TRANSFORMATION
 Frame -> Face Crop (Bounding Box Slicing)
 BGR -> RGB (OpenCV Conversion)
 Resize -> 160x160 (OpenCV Resize)
 Image -> 512D Vector (FaceNet Embedding)
 ↓
PROCESSING
 512D Vector -> SVM Classification -> Label Index
 Label Index -> String Name (`inverse_transform`)
 ↓
STORAGE
 Cropped Image -> `.jpg` file
 Attendance Data -> `.csv` row
 ↓
RETRIEVAL
 Not handled in this script (Administrators read the CSV manually)
 ↓
OUTPUT
 Annotated Video Frame written to `result_datas/testig_video_result.mp4`
```

## Data Types Documented
- **Input Format**: 1080p/720p Video Frames (H.264/MP4).
- **Transformation**: `np.ndarray(H, W, 3)` -> `np.ndarray(160, 160, 3)` -> `np.ndarray(1, 512)`.
- **Storage Format**: `.csv` (Comma Separated Values) and `.jpg` (JPEG Image compression).
- **Retrieval Format**: CSV readable by Excel or Pandas.
- **Output Format**: Annotated MP4 Video and PNG `live.png` snapshot.
