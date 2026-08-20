# Appendix A: Function Reference

| Function | File | Input | Output | Called By | Calls |
| -------- | ---- | ----- | ------ | --------- | ----- |
| `callback(frame, frame_idx)` | `main.py` | `np.ndarray` (frame), `int` (frame index) | `np.ndarray` (annotated frame) | `sv.process_video()` | `model()`, `tracker.update_with_detections()`, `predict_face()`, `cv.line()`, `cv.imwrite()`, `csv.writer.writerow()` |
| `predict_face(face_img)` | `facenet_files/facent_svm_rec_passing.py` | `np.ndarray` (cropped face image) | `tuple(str, float)` (Name, Probability) | `callback()` (in `main.py`) | `cv2.resize()`, `get_embedding()`, `model[0].predict()`, `model[0].predict_proba()` |
| `get_embedding(face_img)` | `facenet_files/facent_svm_rec_passing.py` / `training.py` | `np.ndarray` (160x160 face image) | `np.ndarray` (1x512 embedding) | `predict_face()` / Training Loop | `embedder.embeddings()` |
| `extract_face(self, filename)` | `training.py` (Class `FACELOADING`) | `str` (image path) | `np.ndarray` (160x160 aligned face) | `load_faces()` | `cv2.imread()`, `detector.detect_faces()`, `cv2.resize()` |
| `load_classes(self)` | `training.py` (Class `FACELOADING`) | `None` | `tuple(np.ndarray, np.ndarray)` (X, Y) | Training Script | `load_faces()` |
