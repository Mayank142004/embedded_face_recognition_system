# 13. Configuration and Environment

The project does **not** use a `.env` file or environment variables for configuration. All settings, paths, and hyperparameters are hardcoded into the Python scripts.

## Hardcoded Configurations

| Setting | File | Purpose | Example |
|---------|------|---------|---------|
| **YOLO Weights Path** | `main.py` | Location of the YOLOv8 face model. | `"yolo_models/yolov8n-face.pt"` |
| **SVM Model Path** | `facenet_files/facent_svm_rec_passing.py` | Location of the pickled SVM classifier. | `'facenet_models/new_classifier_Jun27_759.pkl'` |
| **Input Video Path** | `main.py` | The video to be processed. | `"test_datas/testing_video.mp4"` |
| **Output Video Path** | `main.py` | Where to save the annotated video. | `"result_datas/testig_video_result.mp4"` |
| **Attendance Threshold**| `main.py` | The minimum probability required to log attendance. | `0.87` |
| **Line Position** | `main.py` | Y-coordinate offset for the virtual attendance line. | `(height // 2) - 50` |

> ⚠️ **Warning**: Because these paths are hardcoded, moving the scripts or running them from a different working directory will result in `FileNotFoundError` exceptions.

## Environment Variables

Only one environment variable is explicitly set within the scripts to control external library behavior:

| Variable | File | Purpose | Required | Example |
|----------|------|---------|----------|---------|
| `TF_CPP_MIN_LOG_LEVEL` | `training.py`, `facent_svm_rec_passing.py` | Suppresses TensorFlow C++ warnings (like AVX instructions missing). | No | `'2'` |
