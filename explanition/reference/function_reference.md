# Function Reference

| Function | File | Class | Input | Output | Called By | Calls | Purpose |
| -------- | ---- | ----- | ----- | ------ | --------- | ----- | ------- |
| `callback` | `main.py` | N/A | `frame`, `_` | `annotated_frame` | `sv.process_video` | `model`, `predict_face`, OpenCV drawing functions | Processes each video frame, draws boxes/lines, crops faces, and handles attendance logging logic. |
| `predict_face` | `facenet_files/facent_svm_rec_passing.py` | N/A | `face_image` | `result`, `result_probability` | `callback` | `get_embedding`, SVM `.predict()` | Recognizes a person from a cropped face image. |
| `get_embedding` | `facenet_files/facent_svm_rec_passing.py` (and `training.py`) | N/A | `face_img` | 512D array | `predict_face` (or loop) | `embedder.embeddings()` | Uses FaceNet to extract features from a 160x160 face. |
| `write_labels_to_file` | `facenet_files/facent_svm_rec_passing.py` | N/A | `labels`, `filename` | None | Module load | `open()` | Saves the list of known names to a text file on startup. |
| `extract_face` | `training.py` | `FACELOADING` | `filename` | 160x160 array | `load_faces` | MTCNN, `cv.imread` | Finds and crops a face from an image file using MTCNN for training. |
| `load_faces` | `training.py` | `FACELOADING` | `dir` (path) | List of arrays | `load_classes` | `extract_face` | Iterates through a directory to extract all faces. |
| `load_classes` | `training.py` | `FACELOADING` | None | `X`, `Y` arrays | Module load | `load_faces` | Iterates through all class folders to build training dataset. |
| `plot_images` | `training.py` | `FACELOADING` | None | None | Developer | `plt.imshow` | Visualizes the loaded dataset. |
| `process_image` | `crop_face_images_using_yolo.py` | N/A | `image_path`, `output_path` | None | Main loop | YOLO model, `cv2.imwrite` | Utility to crop faces from a static image using YOLO. |
