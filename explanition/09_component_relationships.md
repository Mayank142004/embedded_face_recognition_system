# 09. Component Relationships

The system is tightly coupled within the `main.py` execution script.

```text
Supervision Video Engine
   │
   ↓ Frame, Index
Main Application (callback)
   │
   ├────────→ YOLO Model (Detection)
   │             ↳ Returns Bounding Boxes
   │
   ├────────→ ByteTracker (Tracking)
   │             ↳ Returns Track IDs
   │
   ├────────→ FaceNet + SVM Service (facenet_files)
   │             ↳ Returns Identity & Probability
   │
   └────────→ File System (CSV & JPG Storage)
                 ↳ Saves Attendance
```

### 1. Supervision <-> Main Application
**Communication**: Function Call (`sv.process_video` calling `callback`)
**What is sent**: `frame` (NumPy array) and frame index.
**Why**: To process the video frame-by-frame.
**Initiator**: Supervision engine.
**What comes back**: `annotated_frame` (NumPy array with boxes/labels).

### 2. Main Application <-> YOLO Model
**Communication**: Class Call (`model(frame)`)
**What is sent**: Raw video `frame`.
**Why**: To find faces.
**Initiator**: Main application (`callback`).
**What comes back**: YOLO results list containing coordinates.

### 3. Main Application <-> FaceNet/SVM Service
**Communication**: Function Call (`predict_face(face)`)
**What is sent**: Cropped `face` (NumPy array).
**Why**: To identify the person in the crop.
**Initiator**: Main application (`callback`).
**What comes back**: A tuple `(name, result_probability)`.

### 4. Main Application <-> File System
**Communication**: OS system calls (`os.makedirs`, `open`, `cv.imwrite`)
**What is sent**: File paths, image arrays, CSV rows.
**Why**: To persist attendance records permanently.
**Initiator**: Main application (`callback`).
**What comes back**: OS confirmation (implicitly, unless an `Exception` is thrown).
