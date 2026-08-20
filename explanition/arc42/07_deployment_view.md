# 07. Deployment View

This chapter describes the technical infrastructure used to execute the Face Recognition System.

The system is designed to run either directly on a host machine with Python or inside a Docker container.

## 7.1 Docker Deployment (Primary)

The application is containerized to manage the complex system dependencies required by OpenCV and AI libraries.

### Infrastructure Nodes

**Node 1: Docker Container (`face_recognition_app`)**
- **Technology:** Docker (Linux Base)
- **Purpose:** Executes `main.py` and isolates the Python environment.
- **Dependencies (OS Level):** `libgl1-mesa-glx` (Required by OpenCV for video processing).
- **Dependencies (Python):** Sourced from `requirement.txt` (TensorFlow, PyTorch, Ultralytics, OpenCV, Supervision).
- **Execution:** Runs `CMD ["python", "main.py"]`.

### Volume Mounts (Critical)

To ensure that attendance logs and video results persist after the container shuts down, volumes must be mapped.

1. **Input Video:**
   - Mount: `./test_datas:/app/test_datas`
   - Purpose: Provides the source video.
2. **Output Data:**
   - Mount: `./result_datas:/app/result_datas`
   - Purpose: Persists the annotated output video.
3. **Attendance Logs & Images:**
   - Mount: `./marked_attendance:/app/marked_attendance`
   - Purpose: Persists the CSV logs and cropped `.jpg` faces. If not mounted, attendance records will be lost when the container stops.

## 7.2 Hardware Requirements

Due to the heavy machine learning workloads (YOLOv8, FaceNet), the underlying host requires:
- **CPU:** Multi-core modern processor.
- **RAM:** Minimum 8GB (TensorFlow and YOLO concurrently consume significant memory).
- **GPU (Optional but Recommended):** While not explicitly configured for GPU in the provided codebase snippets, running Ultralytics YOLOv8 and TensorFlow/FaceNet on a CPU is slow. Deploying on a CUDA-enabled machine with Nvidia drivers will drastically improve frames-per-second (FPS).
