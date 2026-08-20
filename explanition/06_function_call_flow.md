# 06. Function Call Flow

## Main Execution Trace

```text
User execution
    ↓
sv.process_video()  [supervision framework]
    ↓
callback()  [main.py]
    ↓
model(frame)  [ultralytics YOLO]
    ↓
tracker.update_with_detections()  [supervision]
    ↓
predict_face()  [facenet_files/facent_svm_rec_passing.py]
    ↓
get_embedding()  [facenet_files/facent_svm_rec_passing.py]
    ↓
cv.imwrite() & csv.writer.writerow()  [main.py]
```

## Detailed Function Documentation

### Function: `callback`
**File:** `main.py`
**Class:** None
**Called By:** `sv.process_video`
**Calls:** `model`, `tracker.update_with_detections`, `predict_face`, OpenCV drawing functions.

**Input:**
- `frame`: `np.ndarray` - The current video frame.
- `_`: `int` - The frame index (unused).

**Processing:**
- Step 1: Detect faces using YOLO.
- Step 2: Track faces using ByteTrack.
- Step 3: Draw a horizontal line in the middle of the frame.
- Step 4: Iterate over detected faces, crop them, and call `predict_face`.
- Step 5: Check if the face crosses the line and confidence is high. If so, log attendance to CSV and save the face crop.
- Step 6: Annotate the frame with names and probabilities.

**Output:**
- `annotated_frame`: `np.ndarray` - The frame with bounding boxes and labels drawn.

**Exceptions:**
- Catches general `Exception` when annotating the frame and prints it.

---

### Function: `predict_face`
**File:** `facenet_files/facent_svm_rec_passing.py`
**Class:** None
**Called By:** `callback` in `main.py`
**Calls:** `get_embedding`, SVM `model.predict()`, `model.predict_proba()`

**Input:**
- `face_image`: `np.ndarray` - Cropped face image in BGR format.

**Processing:**
- Step 1: Convert image to RGB.
- Step 2: Resize image to 160x160.
- Step 3: Call `get_embedding()` to get FaceNet features.
- Step 4: Expand dimensions and pass to SVM model to get predictions and probabilities.
- Step 5: Map prediction index back to actual string label using `LabelEncoder`.

**Output:**
- `result`: `str` - The predicted name of the person.
- `result_probability`: `float` - The confidence score of the prediction.

---

### Function: `get_embedding`
**File:** `facenet_files/facent_svm_rec_passing.py`
**Class:** None
**Called By:** `predict_face`
**Calls:** `embedder.embeddings` (FaceNet)

**Input:**
- `face_img`: `np.ndarray` - 160x160 RGB face image.

**Processing:**
- Step 1: Convert to float32.
- Step 2: Expand dimensions to match expected input `(1, 160, 160, 3)`.
- Step 3: Pass to FaceNet model.

**Output:**
- `yhat[0]`: `np.ndarray` - 512-dimensional embedding vector.
