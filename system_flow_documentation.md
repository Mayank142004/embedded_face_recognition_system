# Face Attendance System: Complete Architecture & Data Flow

This document details the exact function-by-function execution flow, architectural design, and data transmission between the Raspberry Pi (Edge) and the Laptop (Server).

---

## 1. System Architecture Overview

The system operates on a publisher-subscriber model combined with real-time WebSocket streaming.
*   **Raspberry Pi (Edge):** Dedicated entirely to camera capture, AI inference, and event publishing. It holds no database and runs no UI.
*   **Laptop (Server):** Dedicated to database management, machine learning training, UI dashboard, and message routing.

---

## 2. Execution Flow & Function Calls (Raspberry Pi)

The Pi's execution starts in `pi_runner.py`.

### A. The Main Camera Loop
1.  **`pi_runner.py -> main()`**: Initializes the camera `cv2.VideoCapture(0)`.
2.  **`While True:`**: Captures a frame and immediately passes it to the AI pipeline by calling `callback(frame)`.

### B. The AI Pipeline (`main.py -> callback()`)
To prevent the Pi from overheating, `callback()` implements a 5-frame skip logic.

**For every 5th frame:**
1.  **`yolo_tflite.py -> model.predict(frame)`**: 
    *   Resizes the frame to 320x320 and invokes `yolov8n-face.tflite`.
    *   Mathematical coordinates are scaled back to the original image dimensions.
    *   *Returns:* A list of bounding boxes and confidences.
2.  **`supervision -> tracker.update_with_detections()`**: 
    *   Correlates the YOLO boxes with previous frames using Kalman filtering to assign a persistent Tracker ID (e.g., `ID: 5`).
3.  **`facenet_svm_rec_passing.py -> predict_face(cropped_face)`**:
    *   Iterates through each tracked face, resizes it to 160x160.
    *   Invokes `facenet.tflite` to generate a 128-dimensional embedding.
    *   Passes the embedding to `svm_classifier.pkl.predict_proba()`.
    *   *Returns:* The predicted `emp_id` and confidence score.
4.  **`line_crossing.py -> attendance_detector.update()`**:
    *   Checks the vertical Y-coordinate of the face. If the face crosses the predefined horizontal line, it checks the debounce dictionary.
    *   *Returns:* A valid attendance event if the employee hasn't been logged in the last 3 seconds.
5.  **`mqtt_publisher.py -> publish_event(emp_id, "in")`**:
    *   If an event is returned, this function packages the `emp_id` and timestamp into a JSON payload and transmits it over WiFi to the Laptop's Mosquitto broker via port `1883`.

**For the 4 skipped frames:**
*   The function bypasses the AI pipeline entirely and simply uses the last known bounding boxes to draw rectangles on the frame (saving 80% CPU).

### C. The WebSocket Broadcaster
At the end of *every* frame (skipped or analyzed), `callback()` pushes the frame to two background threads.
1.  **`main.py -> WSStreamer.send_frame()`**: Pushes the frame to a `queue.Queue(maxsize=1)`.
2.  **`WSStreamer._run()`**: A background thread pops the frame, compresses it using `cv2.imencode('.jpg')`, and sends the binary bytes to the Laptop via `websocket.send_binary()`.

---

## 3. Execution Flow & Function Calls (Laptop / Server)

The Laptop runs three separate processes simultaneously.

### Process 1: The WebSocket Router (`server.py`)
1.  **`@app.websocket("/ws/stream/pi/raw")`**: Listens for the incoming JPEG bytes from the Pi.
2.  **`websocket_pi_stream()`**: When bytes arrive, it iterates through `ui_clients` (a list of connected Dashboards) and calls `await client.send_bytes(data)`. This perfectly routes the video without touching the hard drive.

### Process 2: The MQTT Subscriber (`mqtt_subscriber.py`)
This is a background daemon script running permanently.
1.  **`on_message()`**: Triggered the exact millisecond the Mosquitto broker receives the JSON payload from the Pi.
2.  **`db.py -> get_employee_dict()`**: Looks up the employee's real name based on their `emp_id`.
3.  **`db.py -> record_attendance_in(emp_id, emp_name, timestamp)`**: 
    *   Queries MongoDB to see if the employee already clocked in today.
    *   If not, it executes `get_attendance_col().insert_one()` to permanently save the attendance record.

### Process 3: The User Dashboard (`dashboard.py`)
The Streamlit UI is where the user interacts with the system.
1.  **`dashboard.py -> consume_both_streams()`**: Creates two asynchronous tasks that connect to `ws://192.168.1.29:8000/ws/stream/ui/raw` and `.../analyzed`.
2.  **`consume_single_stream()`**: Awaits incoming bytes from the FastAPI router, decodes them via `cv2.imdecode`, and renders them to the screen using `st.image()`.
3.  **`db.py -> get_today_attendance()`**: Every 30 frames, the dashboard triggers this function to query MongoDB and update the Pandas dataframe on the screen, showing the live attendance logs that were just inserted by the MQTT subscriber.

---

## 4. Summary of Data Flow
1. **Video Data:** Pi Camera → `cv2` → AI Pipeline → `cv2.imencode` → `websocket` → WiFi → FastAPI → `websocket` → Streamlit → `st.image`.
2. **Attendance Data:** Pi Line Cross → JSON Payload → `paho-mqtt` → WiFi → Mosquitto Broker → `mqtt_subscriber.py` → `pymongo` → MongoDB.

---

## 5. How to Run the Project

To successfully launch the system, services must be started in the correct order across both devices.

### A. Start Services on the Laptop (Server)
Your laptop acts as the central hub. It requires three separate terminal windows to run its microservices.

**Terminal 1: Start the FastAPI Server (WebSocket Router)**
```bash
cd ~/Desktop/FaceRecognitionSystem
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000
```
*(This must be running first, otherwise the Pi has nowhere to send its video).*

**Terminal 2: Start the MQTT Subscriber (Database Writer)**
```bash
cd ~/Desktop/FaceRecognitionSystem
source .venv/bin/activate
python mqtt_subscriber.py
```
*(This must be running, otherwise attendance events will be ignored).*

**Terminal 3: Start the Streamlit Dashboard (UI)**
```bash
cd ~/Desktop/FaceRecognitionSystem
source .venv/bin/activate
streamlit run dashboard.py
```
*(This will open the dashboard in your web browser).*

*Note: Ensure your MongoDB and Mosquitto background services are running (`sudo systemctl start mongod mosquitto`).*

### B. Start the Camera on the Raspberry Pi (Edge)
Once the laptop services are running, you can boot up the edge node.

**Terminal: SSH into the Pi**
```bash
cd ~/FaceAttend
source .venv/bin/activate
python pi_runner.py
```
*(The Pi will automatically download the latest AI model from the laptop, connect to the camera, and begin streaming video and attendance events).*
