# Class Reference

| Class | File | Responsibility | Important Methods | Dependencies |
| ----- | ---- | -------------- | ----------------- | ------------ |
| `FACELOADING` | `training.py` | Automates the process of finding faces in images, cropping them, and preparing the dataset (features `X`, labels `Y`) for SVM training. | `extract_face()`, `load_classes()` | `mtcnn.MTCNN`, `cv2`, `os` |
