# Face Recognition System - Technical Documentation

## Overview
This system automates attendance tracking using advanced computer vision and machine learning. It detects, tracks, and recognizes faces from video inputs (e.g., CCTV or recorded video), marking attendance with a timestamp when a recognized individual crosses a virtual line in the frame. Unrecognized faces are ignored.

## Technology Stack
- **Detection**: YOLOv8 (`yolov8n-face.pt`)
- **Recognition**: FaceNet (via `keras-facenet`)
- **Tracking**: ByteTrack (via `supervision`)
- **Classification**: Support Vector Machine (SVM) from `scikit-learn`
- **Computer Vision Utilities**: OpenCV, `supervision`
- **Data Storage**: CSV (local files) and saved images

## Major Components
1. **Face Detection & Tracking**: Powered by YOLOv8 and ByteTrack to identify faces across multiple frames.
2. **Face Feature Extraction**: FaceNet extracts a 512-dimensional embedding from the detected face.
3. **Face Classification**: An SVM classifier trained on the FaceNet embeddings predicts the identity.
4. **Attendance Logging**: Successful recognitions with high probability (>87%) are saved to daily CSV files, along with face crops.

## Component Flow
```text
User (Face in Video)
 ↓
YOLOv8 + ByteTrack (Detection & Tracking)
 ↓
Face Cropping & Resizing
 ↓
FaceNet (Feature Extraction)
 ↓
SVM Classifier (Identity Prediction)
 ↓
CSV Log & Image Save (Attendance Marking)
 ↓
Annotated Video Output (Response/Result)
```

## Execution Starts Here
The primary entry point is `main.py`.

## Running the Project
```bash
# Set up the environment
pip install -r requirement_clean.txt

# Run the main script
python main.py
```
> **Note**: The current implementation of `main.py` processes a hardcoded video file (`test_datas/testing_video.mp4`) and saves the result to `result_datas/testig_video_result.mp4`. To run it on a webcam, modifications to the `sv.process_video` call or using scripts in `yolo_with_facenet_svm/` are necessary.

## Important Environment Variables
The application primarily relies on hardcoded paths in the scripts rather than environment variables. See [13_configuration_and_environment.md](13_configuration_and_environment.md) for details.

## Main API Endpoints
*UNKNOWN / NOT CONFIRMED FROM CODE*
The current implementation runs as an offline video processing script and does not expose a web API.

## Documentation Index
- [01. Project Overview](01_project_overview.md)
- [02. Complete Architecture](02_architecture.md)
- [03. Directory Structure](03_directory_structure.md)
- [04. Application Flow](04_application_flow.md)
- [05. API Flow](05_api_flow.md)
- [06. Function Call Flow](06_function_call_flow.md)
- [07. Database Architecture](07_database_architecture.md)
- [08. AI / ML Pipeline](08_ai_ml_pipeline.md)
- [09. Component Relationships](09_component_relationships.md)
- [10. Sequence Diagrams](10_sequence_diagrams.md)
- [11. Data Flow Diagrams](11_data_flow_diagrams.md)
- [12. Class and Module Relationships](12_class_and_module_relationships.md)
- [13. Configuration and Environment](13_configuration_and_environment.md)
- [14. Error Handling](14_error_handling.md)
- [15. Authentication & Authorization](15_authentication_and_authorization.md)
- [16. External Services](16_external_services.md)
- [17. Deployment Architecture](17_deployment_architecture.md)
- [18. Execution Lifecycle](18_execution_lifecycle.md)
- [19. Testing Architecture](19_testing_architecture.md)
- [20. Complete Code Walkthrough](20_complete_code_walkthrough.md)

### Reference
- [Function Reference](reference/function_reference.md)
- [API Reference](reference/api_reference.md)
- [Class Reference](reference/class_reference.md)
- [Model Reference](reference/model_reference.md)
- [Environment Reference](reference/environment_reference.md)

### Diagrams
- [System Architecture](diagrams/system_architecture.mmd)
- [Application Flow](diagrams/application_flow.mmd)
- [API Flow](diagrams/api_flow.mmd)
- [Database Relationship](diagrams/database_relationship.mmd)
- [Component Relationship](diagrams/component_relationship.mmd)
- [Function Call Flow](diagrams/function_call_flow.mmd)
- [Sequence Diagram](diagrams/sequence_diagram.mmd)
- [AI/ML Pipeline](diagrams/ai_ml_pipeline.mmd)
- [Deployment Architecture](diagrams/deployment_architecture.mmd)
