# System Architecture

## High-Level Architecture
The system is divided into two physical nodes:
1.  **Edge Node (Raspberry Pi):** Responsible for video capture, heavy AI inference (YOLO + FaceNet), and broadcasting events via MQTT.
2.  **Server Node (Laptop/PC):** Hosts the MongoDB database, Mosquitto MQTT Broker, FastAPI WebSocket Router, and the Streamlit Dashboard.

```mermaid
graph TD
    subgraph Edge Node [Raspberry Pi 3/4]
        Cam[USB/Pi Camera] --> PiRunner[pi_runner.py]
        PiRunner --> YOLO[YOLOv8n TFLite]
        YOLO --> Tracker[ByteTrack]
        Tracker --> FaceNet[FaceNet TFLite]
        FaceNet --> SVM[SVM Classifier]
        SVM --> MQTT_Pub[MQTT Publisher]
        PiRunner -.-> WS_Pub[WebSocket Emitter]
    end

    subgraph Server Node [Laptop/PC]
        MQTT_Broker[Mosquitto Broker]
        MQTT_Sub[mqtt_subscriber.py]
        FastAPI[server.py]
        MongoDB[(MongoDB)]
        Dashboard[dashboard.py Streamlit]
        
        MQTT_Pub == Topic: attendance/events ==> MQTT_Broker
        MQTT_Broker --> MQTT_Sub
        MQTT_Sub --> MongoDB
        
        WS_Pub == Raw & Analyzed Bytes ==> FastAPI
        FastAPI == WS Stream ==> Dashboard
        Dashboard <--> MongoDB
    end
```

## Data Flow Diagram
```mermaid
sequenceDiagram
    participant Camera
    participant MainLoop as main.py (Edge)
    participant Models as YOLO/FaceNet
    participant MQTT as Mosquitto (Server)
    participant Sub as mqtt_subscriber
    participant DB as MongoDB

    Camera->>MainLoop: Capture Frame (N)
    alt is N % 5 == 0 (Every 5th Frame)
        MainLoop->>Models: Predict (Frame)
        Models-->>MainLoop: Bounding Boxes + Embeddings
        MainLoop->>Models: Predict ID (SVM)
        Models-->>MainLoop: Employee ID
        alt Crosses Attendance Line
            MainLoop->>MQTT: Publish 'IN' Event
        end
    else
        MainLoop->>MainLoop: Reuse frozen bounding boxes
    end
    MQTT->>Sub: Receive Event
    Sub->>DB: Upsert Attendance Record
```

## Repository Structure

*   `/facenet_files/`: Core recognition scripts (embedding extraction, SVM).
*   `/facenet_models/`: Pre-trained weights (`svm_classifier.pkl`).
*   `/yolo_models/`: YOLO TFLite models for face detection.
*   `config.py`: Centralized environment configurations.
*   `dashboard.py`: Streamlit frontend.
*   `db.py`: MongoDB CRUD wrappers.
*   `main.py`: Edge pipeline orchestrator.
*   `mqtt_publisher.py` / `mqtt_subscriber.py`: Pub/Sub logic.
*   `pi_runner.py`: Edge node entry point and camera loop.
*   `server.py`: FastAPI server for WS streams.
*   `training.py`: SVM training and Albumentations pipeline.
*   `yolo_tflite.py`: Custom zero-dependency YOLO TFLite wrapper.
