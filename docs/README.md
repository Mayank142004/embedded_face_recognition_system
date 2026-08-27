# Face Recognition Attendance System

## Executive Summary
**Project Purpose:** An automated, frictionless attendance tracking system leveraging edge AI.
**Problem Statement:** Traditional attendance systems (RFID, biometric scanners) require physical interaction, causing bottlenecks and hygiene concerns. Existing cloud-AI solutions suffer from high latency and bandwidth costs.
**Solution Overview:** A distributed edge-server architecture. A Raspberry Pi edge node performs real-time face detection, tracking, and embedding extraction using optimized TFLite models. A centralized server (laptop/cloud) handles dashboarding, database management, SVM training, and MQTT event ingestion.
**Key Features:**
*   **Edge Inference:** YOLOv8 and FaceNet run purely on TFLite on the Raspberry Pi without heavy PyTorch dependencies.
*   **Real-time Tracking:** ByteTrack prevents duplicate counts and tracks individuals across frames.
*   **Zero-Latency Dashboard:** WebSockets provide dual live streams (Raw & Analyzed) to a Streamlit dashboard.
*   **Fault-Tolerant Pub/Sub:** MQTT handles attendance event transmission, ensuring decoupling between edge and server.
*   **Rapid Retraining:** SVM classification allows instant retraining for new employees using only 10-15 augmented images.

## Documentation Index
*   [System Architecture](Architecture.md)
*   [Raspberry Pi Pipeline](RaspberryPi.md)
*   [Backend Services](Backend.md)
*   [API Reference](API.md)
*   [Face Recognition Workflow](Recognition.md)
*   [Configuration Guide](Configuration.md)
*   [Deployment Guide](Deployment.md)
*   [Performance Analysis](Performance.md)
*   [Troubleshooting](Troubleshooting.md)
*   [Future Roadmap](FutureRoadmap.md)
