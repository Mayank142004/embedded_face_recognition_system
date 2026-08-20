# 12. Glossary

This section explains important technical terminology specific to the Face Recognition System.

| Term | Meaning |
| ---- | ------- |
| **Embedding** | A mathematical representation of an image. Specifically, FaceNet converts a face image into a 512-dimensional vector (array of 512 numbers) that captures the unique structural features of the face. |
| **FaceNet** | A neural network architecture used to extract embeddings from face images. It learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. |
| **YOLOv8** | "You Only Look Once". A state-of-the-art, real-time object detection model. In this project, a version trained specifically on faces (`yolov8n-face.pt`) is used to draw bounding boxes around heads. |
| **ByteTrack** | A multi-object tracking algorithm. It associates bounding boxes from frame to frame, allowing the system to assign a unique ID (`tracker_id`) to a person as long as they remain in the camera view. |
| **MTCNN** | Multi-task Cascaded Convolutional Networks. Used in `training.py` to detect faces and align them based on facial landmarks (eyes, nose, mouth) before creating training embeddings. |
| **SVM** | Support Vector Machine. A machine learning classification algorithm. It takes the 512-dimensional embedding and predicts which employee class (Name) the embedding belongs to. |
| **Supervision** | A Python library (`roboflow/supervision`) used to simplify writing video processing loops, handling trackers, and drawing bounding boxes/labels on frames. |
| **Attendance Line** | A virtual horizontal line (`line_y`) drawn across the middle of the video frame. The system only triggers the FaceNet recognition process when an employee's bounding box crosses this line. |
