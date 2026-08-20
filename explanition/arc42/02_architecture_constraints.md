# 02. Architecture Constraints

This section outlines the constraints that restrict the design and implementation of the Face Recognition System. These have been identified directly from the source code, dependencies, and configuration.

### Technical Constraints

| Constraint | Source | Impact | Reason / Architectural Consequence |
| ---------- | ------ | ------ | ---------------------------------- |
| **Python 3.11** | `.python-version`, `Dockerfile` | All development and runtime environments must support Python 3.11. | Required by specific dependency versions (e.g., `keras-facenet`, `ultralytics`). Limits deployment to compatible environments. |
| **No Backend API Framework** | `main.py` | The system runs as a continuous local script, not a web server. | Interaction with the system is done by providing video files or feeds directly to the script. It cannot be queried via REST or GraphQL. |
| **No Relational Database** | `main.py` (CSV module) | Data persistence is entirely file-based (CSV files and `.jpg` images). | Lack of ACID transactions. Concurrent access by multiple script instances could corrupt the CSV. Queries (e.g., "Get all attendance for user X") require parsing CSVs manually. |
| **OpenCV GUI Limitations** | `main.py` comments | `cv2.imshow` is disabled/problematic in the current environment. | Real-time debugging relies on saving the output to `live.png` or writing the entire annotated video to `result_datas/`. |

### AI/ML Constraints

| Constraint | Source | Impact | Reason / Architectural Consequence |
| ---------- | ------ | ------ | ---------------------------------- |
| **YOLOv8 + ByteTrack** | `main.py` | The system depends heavily on Ultralytics and Supervision for detection/tracking. | Tight coupling to these libraries for bounding box generation. |
| **160x160 Face Crop** | `facenet_files/facent_svm_rec_passing.py`, `training.py` | Detected faces must be resized to 160x160 before embedding. | Required by the FaceNet model input layer. |
| **Confidence Threshold >= 0.87** | `main.py` (Line crossing logic) | The SVM must output a prediction probability of at least 87% to log attendance. | Ensures high precision. Faces below this threshold are ignored or treated as 'Unknown'. |

### Infrastructure Constraints

| Constraint | Source | Impact | Reason / Architectural Consequence |
| ---------- | ------ | ------ | ---------------------------------- |
| **Local File System Volumes** | `main.py` (`marked_attendance/`) | The system writes directly to the local directory. | When running in Docker, volume mounts must be correctly configured to persist attendance logs and cropped images beyond container lifecycle. |
| **Containerization** | `Dockerfile` | The application is packaged using Docker. | The Docker container must install necessary OS-level libraries (e.g., `libgl1-mesa-glx`) for OpenCV to function. |

### Organizational Constraints

| Constraint | Source | Impact | Reason / Architectural Consequence |
| ---------- | ------ | ------ | ---------------------------------- |
| **Monolithic Script** | Codebase structure | Both business logic (attendance) and video processing happen in `main.py` synchronously. | Scaling requires running separate independent processes rather than scaling individual microservices. |
