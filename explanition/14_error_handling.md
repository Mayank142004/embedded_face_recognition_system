# 14. Error Handling

The application has very minimal error handling, relying mostly on "happy path" execution.

## Trace of Error Handling Paths

### 1. Drawing Annotations Error
**Location**: `main.py` -> `callback()`
```text
Error occurs drawing boxes/labels
 ↓
Exception raised by `supervision`
 ↓
Caught by general `try/except Exception as E` block in `callback`
 ↓
Logged to console: `print(E)`
 ↓
Execution continues (but frame might be unannotated)
```

### 2. File Loading Errors (Training)
**Location**: `training.py` -> `load_faces()`
```text
Error occurs extracting a face (e.g., image corrupt, no face found by MTCNN)
 ↓
Exception raised by `MTCNN` or OpenCV
 ↓
Caught by `try/except Exception as e`
 ↓
Ignored (`pass`)
 ↓
Script skips the bad image and continues
```

### 3. Missing Output Directory
**Location**: `main.py` -> CSV creation
If the directory does not exist, the script uses `os.makedirs(output_dir, exist_ok=True)`. This is a built-in safety mechanism that prevents `FileNotFoundError` when creating the CSV file on a new day.

## Unhandled Failure Scenarios
The following scenarios will cause the script to crash (`Stack Trace` printed and process terminates):
- **Missing Models**: If `yolov8n-face.pt` or the SVM `.pkl` file are missing, the script will crash on import/initialization.
- **Missing Input Video**: If `testing_video.mp4` is not found, `supervision` will throw an error.
- **Corrupt Image on Inference**: If YOLO detects a bounding box but it's invalid (e.g., negative coordinates), OpenCV array slicing `frame[y1:y2, x1:x2]` might fail.
