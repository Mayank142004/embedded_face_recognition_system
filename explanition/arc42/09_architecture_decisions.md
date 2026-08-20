# 09. Architecture Decisions

This chapter documents major architectural decisions made during the system's development.

## ADR-001: Hybrid Detection and Recognition Pipeline

### Status
Accepted

### Context
Real-time face recognition requires fast face detection to locate faces in a video stream. Traditional models like MTCNN provide excellent face alignment but are often too slow to run on every frame of a 30 FPS video feed. 

### Decision
Use YOLOv8 and ByteTrack for real-time bounding box detection and tracking, but use MTCNN for offline training face extraction, and FaceNet for embeddings.

### Alternatives
- **End-to-End model (e.g., RetinaFace + ArcFace):** Highly accurate but harder to implement quickly.
- **MTCNN for everything:** Too slow for real-time video processing on standard CPUs.

### Reason
YOLOv8 is state-of-the-art for fast object detection. By tracking IDs with ByteTrack, we only need to run the heavy FaceNet/SVM recognition model *once* when the person crosses the attendance line, rather than on every frame.

### Consequences
- **Positive:** Massive performance improvement for video processing.
- **Negative:** Requires managing multiple models (YOLO, MTCNN, FaceNet, SVM).
- **Negative:** Cropped faces from YOLO may not be perfectly aligned compared to MTCNN's landmark-based alignment, potentially reducing FaceNet accuracy slightly.

### Implementation
- `main.py` (YOLO + ByteTrack)
- `training.py` (MTCNN for training data)

---

## ADR-002: CSV for Persistence

### Status
Accepted

### Context
The system needs to log attendance records mapping Names to Timestamps.

### Decision
Use local CSV files organized by date (`marked_attendance/YYYY_MM_DD/`) rather than a Relational Database (like PostgreSQL or SQLite).

### Alternatives
- SQLite
- PostgreSQL / MySQL

### Reason
Simplicity. The project operates as a monolithic script without a web backend. CSV files are immediately accessible to administrators without requiring database querying tools. 

### Consequences
- **Positive:** Zero setup required. Very easy to copy/paste data into Excel.
- **Negative:** Not scalable. If multiple instances of the script run simultaneously, writing to the same CSV will cause data corruption. Extremely difficult to query historical data across multiple dates.

### Implementation
- `main.py` (`csv` module usage).

---

## ADR-003: High Confidence Threshold

### Status
Accepted

### Context
The SVM classifier will always predict *some* class, even if a stranger walks in. We must prevent false positives.

### Decision
Only register attendance if the SVM `predict_proba` is `>= 0.87` (87%).

### Alternatives
- Lower threshold (e.g., 50%).
- Distance-based threshold on the embeddings directly instead of using SVM probability.

### Reason
An 87% threshold ensures high precision. It is better to fail to recognize an employee (they can manually log in later) than to incorrectly log a stranger as an employee.

### Consequences
- **Positive:** Reduces false positives.
- **Negative:** May increase false negatives (employees not recognized if lighting is poor).

### Implementation
- `main.py` (`if (y1 <= line_y <= y2) and result_probabilty >= 0.87`)
