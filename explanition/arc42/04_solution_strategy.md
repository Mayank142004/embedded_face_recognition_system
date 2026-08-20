# 04. Solution Strategy

This chapter outlines the major architectural decisions and strategies that shape the Face Recognition System.

## 1. Monolithic Script Architecture
**Strategy:** Implement the entire pipeline (video ingestion, detection, recognition, and saving) in a single, synchronous Python script (`main.py`).
**Why used:** Simplicity and ease of development. It avoids the overhead of managing microservices, message queues, or network APIs.
**Where implemented:** `main.py`
**Benefits:** Very easy to debug locally; no complex infrastructure required.
**Trade-offs:** Poor scalability. Processing one frame blocks the next. Cannot easily distribute workloads across multiple machines.
**Alternatives:** Microservices architecture using a message broker (e.g., Kafka or RabbitMQ) to pass frames to separate detection and recognition workers.

## 2. Decoupled Face Detection and Recognition
**Strategy:** Use YOLOv8 for face detection and tracking, but use FaceNet for feature extraction (embeddings).
**Why used:** YOLOv8 is extremely fast for real-time bounding box detection but doesn't identify people. FaceNet produces highly accurate embeddings for identification but its native detectors (like MTCNN used in `training.py`) are often too slow for real-time video tracking.
**Where implemented:** `main.py` (YOLO for detection), `facenet_files/facent_svm_rec_passing.py` (FaceNet for recognition).
**Benefits:** Achieves real-time tracking performance without sacrificing recognition accuracy.
**Trade-offs:** Requires loading two distinct, heavy ML models into memory simultaneously.
**Alternatives:** Using a single end-to-end model, or using MTCNN for both training and runtime (slower).

## 3. ByteTrack for Temporal Consistency
**Strategy:** Employ ByteTrack to maintain consistent IDs across video frames.
**Why used:** Without a tracker, the system would treat a person standing in front of the camera as a "new" detection in every single frame, potentially triggering the heavy FaceNet recognition pipeline 30 times a second.
**Where implemented:** `main.py` (`sv.ByteTrack()`).
**Benefits:** Allows the system to know when it has already seen a person, optimizing logic (like only saving attendance when crossing a line).
**Trade-offs:** Can lose tracking IDs if the person turns their head sharply or leaves the frame temporarily.

## 4. Local File System Persistence (CSV)
**Strategy:** Use CSV files organized by date (`marked_attendance/YYYY_MM_DD/YYYY_MM_DD_attendance_sheet.csv`) instead of a traditional relational database (like PostgreSQL).
**Why used:** Requires zero setup. Easy for administrators to copy and open in Excel.
**Where implemented:** `main.py` (CSV writing logic).
**Benefits:** Lightweight, no external database dependencies.
**Trade-offs:** Prone to data corruption if multiple processes attempt to write simultaneously. Harder to query historically.
**Alternatives:** SQLite or PostgreSQL.

## 5. Virtual "Attendance Line" Trigger
**Strategy:** Only trigger the attendance registration (and FaceNet inference) when the bounding box crosses a specific horizontal line in the middle of the frame.
**Why used:** To prevent continuous logging and inference while an employee is merely standing in the room. It acts as a clear "entry" event.
**Where implemented:** `main.py` (`if (y1 <= line_y <= y2)`).
**Benefits:** Reduces redundant processing and duplicate logs.
**Trade-offs:** Assumes a specific physical camera angle and walking path.
