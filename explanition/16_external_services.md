# 16. External Services

This application operates completely locally and does **not** rely on external network services, cloud APIs (like AWS/GCP), or external databases. 

All machine learning models are downloaded locally and run on the host machine.

### Local Libraries/Frameworks Acting as Services:
1. **YOLOv8**
   - **Purpose:** Face detection.
   - **Used By:** `main.py`
   - **File:** `yolo_models/yolov8n-face.pt`
   - **Offline/Online:** Offline
2. **FaceNet**
   - **Purpose:** Face feature extraction.
   - **Used By:** `facent_svm_rec_passing.py`
   - **Offline/Online:** Offline (Loads keras-facenet weights locally).
3. **Supervision**
   - **Purpose:** Video frame parsing, tracking (ByteTrack), and annotation.
   - **Used By:** `main.py`
   - **Offline/Online:** Offline
