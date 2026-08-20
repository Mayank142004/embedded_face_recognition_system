# 10. Quality Requirements

This chapter details the measurable quality requirements of the Face Recognition System.

## Measurable Targets

| Quality Attribute | Requirement | Evidence | Current Implementation |
| ----------------- | ----------- | -------- | ---------------------- |
| **Accuracy** | High precision for identity matching. | `main.py` | The system enforces a strict probability threshold: `result_probabilty >= 0.87`. Faces below this confidence are not logged. |
| **Performance** | Must process video streams effectively. | `main.py` | Uses YOLOv8 (designed for real-time detection) and ByteTrack. Face recognition (FaceNet) is only triggered on specific line-crossing events, not every frame. |
| **Maintainability** | Must be easy to add new employees. | `training.py` | Automated `FACELOADING` class implemented in `training.py` to extract faces and retrain the SVM. |
| **Usability** | Must not require user interaction. | `main.py` | Uses a virtual attendance line (`line_y`). Users just walk past the camera. |

## Quality Scenarios

### Accuracy Scenario
```text
Stimulus: A stranger walks past the camera.
 ↓
System: YOLO detects the face, FaceNet extracts embedding, SVM predicts a class but with low confidence (< 0.87).
 ↓
Expected Response: Attendance is NOT logged. No CSV entry is created.
 ↓
Measure: Probability threshold enforced.
```

### Performance Scenario
```text
Stimulus: Multiple people are standing in the camera frame talking.
 ↓
System: YOLO detects all faces. ByteTrack tracks them.
 ↓
Expected Response: The system does NOT run the heavy FaceNet model on them continuously.
 ↓
Measure: FaceNet inference is bypassed unless the bounding box crosses the `line_y`.
```

### Data Integrity Scenario
```text
Stimulus: An employee stands directly on the attendance line for 5 seconds.
 ↓
System: The bounding box satisfies `y1 <= line_y <= y2` for 150 consecutive frames.
 ↓
Expected Response: Attendance is only logged ONCE.
 ↓
Measure: `name not in saved_names` condition prevents duplicate writes.
```
