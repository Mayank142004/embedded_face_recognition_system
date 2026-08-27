# Face Recognition Workflow

## 1. Enrollment
When an HR admin adds an employee via the Dashboard:
1.  The UI captures a minimum of 10 photos of the employee's face.
2.  Each photo is passed through FaceNet to generate a 128-dimensional embedding.
3.  The 10 embeddings are mathematically averaged to create a single, highly stable **Master Embedding**.
4.  The Master Embedding is saved to the MongoDB `employees` collection.

## 2. Augmentation & Training
1.  When "Train Model" is clicked, `training.py` retrieves all master embeddings.
2.  *Wait, actually:* The training pipeline uses `albumentations` to apply brightness, contrast, and noise variations to the raw images saved on disk, expanding a 10-image dataset into 40+ variants to create robust boundaries.
3.  An SVM (`sklearn.svm.SVC`) with a `linear` kernel is trained on these embeddings.
4.  The model calculates an optimal prediction probability threshold. If the prediction probability is below this threshold, the face is marked as `"unknown"`.
5.  The model is serialized to `facenet_models/svm_classifier.pkl`.

## 3. Matching & Debounce Logic
*   **Matching:** On the Pi, the live face embedding is fed to the SVM. If `probability > threshold`, it outputs the `emp_id`.
*   **Debounce:** If `emp_id` crosses the vertical Y-line, the `AttendanceDetector` checks its internal dictionary `_last_event`. If the current time minus `_last_event[emp_id]` is less than `DEBOUNCE_SECONDS` (e.g., 3.0s), the event is suppressed. This prevents duplicate check-ins when an employee loiters in the camera frame.
