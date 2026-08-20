# 08. Crosscutting Concepts

This chapter covers concepts that apply across multiple components of the system.

## AI / ML Architecture

The most critical crosscutting concept is the AI pipeline, which relies on a multi-model approach to achieve both speed and accuracy.

- **Object Detection (YOLOv8):** Used purely for speed to find "where" the faces are in a video frame. YOLO does not know "who" the person is.
- **Tracking (ByteTrack):** Adds temporal memory to YOLO. It tracks the bounding boxes across frames so the system knows a person is the same person.
- **Feature Extraction (FaceNet):** Converts a 160x160 pixel crop of a face into a 512-dimensional array of numbers (an embedding). This embedding represents the facial structure.
- **Classification (SVM):** A Support Vector Machine trained on known embeddings. It takes a new 512D embedding and predicts which employee class it belongs to, outputting a probability.

## Persistence (CSV)

Data logging is entirely file-based to avoid the overhead of a database.

- **Directory Structure:** `marked_attendance/YYYY_MM_DD/`
- **File Naming:** `YYYY_MM_DD_attendance_sheet.csv`
- **File Format:** A single CSV row contains `Name, UniqueID, Timestamp, Hyperlink`.
- **Proof:** Cropped face images are saved as `.jpg` alongside the CSV in the daily folder.

## State Management

The script maintains in-memory state to prevent duplicate logging.

- **`saved_names` List:** When an employee crosses the attendance line and is recognized, their name is added to this Python list. Subsequent frames will not log their attendance again, even if they cross the line again.
- *Limitation:* If the script is stopped and restarted, the `saved_names` memory is cleared.

## Error Handling

- **Try/Except Blocks:** Used around the annotation logic (`annotated_frame = box_annotator.annotate...`) to prevent the entire video processing loop from crashing due to malformed bounding boxes.
- **Silent Failures in Training:** In `training.py`, `try/except` is used to skip images that do not contain detectable faces, ensuring the training data loader doesn't crash on bad data.
