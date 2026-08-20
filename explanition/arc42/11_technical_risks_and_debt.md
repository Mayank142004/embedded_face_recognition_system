# 11. Technical Risks and Debt

This section outlines the technical risks and accumulated technical debt identified from analyzing the codebase.

| Risk | Probability | Impact | Evidence | Mitigation |
| ---- | ----------- | ------ | -------- | ---------- |
| **Data Corruption (CSV Concurrency)** | Low (Single Instance) / High (Multi-Instance) | High | `main.py` uses standard file `open(..., mode='a')` without file locking mechanisms. | If scaling to multiple cameras (running multiple script instances), migrate to SQLite or PostgreSQL immediately. |
| **Duplicate Logs on Script Restart** | High | Low | The `saved_names` list is held in RAM. Restarting `main.py` clears the list. | Query today's CSV file on script startup to populate `saved_names` initially. |
| **Monolithic Blocking** | High | Medium | `facenet_result, result_probabilty = predict_face(face)` runs synchronously inside the frame processing loop. | If multiple people cross the line simultaneously, the video feed will stutter while inference runs. Move inference to an asynchronous thread or queue. |
| **Hardcoded Paths and Models** | Medium | Low | `main.py` hardcodes paths like `yolo_models/yolov8n-face.pt` and `test_datas/testing_video.mp4`. | Refactor to use environment variables or CLI arguments via `argparse`. |
| **Uncaught Exceptions in Loop** | Low | High | The `try/except` block in `main.py` only wraps the annotation logic, not the FaceNet inference or CSV writing. | Wrap the core logic in exception handlers to prevent the entire pipeline from crashing on a bad image or disk error. |
| **No Face Alignment in Runtime** | High | Medium | `training.py` uses MTCNN for face alignment. `main.py` uses raw YOLO crops without alignment. | FaceNet is sensitive to alignment. This mismatch between training data and runtime data reduces SVM accuracy. Add an alignment step to `main.py` before passing to `predict_face`. |
