# 03. System Scope and Context

This chapter describes the boundaries of the Face Recognition System, differentiating between what is part of the system and what is external.

## 3.1 Business Context

The system automates the logging of employee attendance by analyzing a video stream or file. 

**External Actors & Systems:**
- **Employee (Actor):** Walks into the camera frame.
- **Administrator (Actor):** Reads the output CSV files to verify attendance.
- **Video Source (System):** Provides the input `testing_video.mp4` or a live camera feed.

```text
Employee
   ↓ (Visual Appearance)
Video Source
   ↓ (Raw Frames)
Face Recognition System
   ↓ (Attendance Log & Face Images)
Administrator
```

## 3.2 Technical Context

The system is entirely self-contained. It takes video as input and produces files (CSV and JPG/MP4) as output. There are no external REST APIs or remote databases accessed during runtime.

### External Interfaces

| System / Boundary | Direction | Protocol / Method | Data | Purpose | Failure Behavior |
| ----------------- | --------- | ----------------- | ---- | ------- | ---------------- |
| **Video Source** | Input | File I/O (`cv2.VideoCapture` via `supervision`) | Video stream (`test_datas/testing_video.mp4`) | Provides the raw frames for processing. | Application fails to start or exits gracefully if the video ends. |
| **File System (Storage)** | Output | OS File I/O (`csv.writer`, `cv2.imwrite`) | `.csv` logs, `.jpg` cropped faces, `.png` live frames, `.mp4` result video | Persists attendance data and diagnostic video output. | Will crash if disk is full or permissions are denied. |
| **FaceNet Model** | Internal | Local File I/O (`pickle.load`, `keras_facenet`) | SVM model `.pkl` and FaceNet weights | Identifies the face. | ⚠️ NOT CONFIRMED FROM CODE if weights are downloaded dynamically or purely local, but assumed local based on `facenet_models/` directory. |

### Note on Container Context
If deployed via Docker, the "System" is the Docker container, and the File System interactions occur via mounted volumes for `marked_attendance/` and `result_datas/`.
