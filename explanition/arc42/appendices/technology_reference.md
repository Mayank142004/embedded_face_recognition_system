# Appendix E: Technology Reference

| Technology | Version | Purpose | Location |
| ---------- | ------- | ------- | -------- |
| Python | 3.11 | Core Runtime Environment | `Dockerfile`, `.python-version` |
| OpenCV (`cv2`) | *See `requirement.txt`* | Image arrays, resizing, line drawing, file saving | `main.py`, `training.py` |
| Ultralytics (YOLO) | *See `requirement.txt`* | Real-time object detection | `main.py` |
| Supervision | *See `requirement.txt`* | Video processing loop, ByteTrack, Annotators | `main.py` |
| Keras-FaceNet | *See `requirement.txt`* | Feature extraction | `facenet_files/...`, `training.py` |
| Scikit-Learn | *See `requirement.txt`* | SVM Classifier and Label Encoder | `training.py`, `main.py` (via unpickling) |
| MTCNN | *See `requirement.txt`* | Offline face alignment | `training.py` |
| Docker | - | Containerization and environment isolation | `Dockerfile` |
| Pickle | Built-in | Saving/Loading the trained SVM model | `training.py`, `facent_svm_rec_passing.py` |
| CSV | Built-in | Writing attendance logs | `main.py` |
