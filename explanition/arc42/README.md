# Face Recognition System - Architecture Documentation

This directory contains the arc42 architecture documentation for the Face Recognition System. The system automates employee attendance tracking using facial recognition technology, identifying employees as they cross a virtual attendance line and recording their entry in a standardized CSV format.

## Table of Contents

| Chapter | Description |
| ------- | ----------- |
| [01 Introduction & Goals](01_introduction_and_goals.md) | Requirements, Quality Goals, and Stakeholders |
| [02 Architecture Constraints](02_architecture_constraints.md) | Technical, Organizational, and Infrastructure constraints |
| [03 System Scope & Context](03_system_scope_and_context.md) | Business and Technical Context Boundaries |
| [04 Solution Strategy](04_solution_strategy.md) | Key architectural approaches and decisions |
| [05 Building Block View](05_building_block_view.md) | Hierarchical component decomposition |
| [06 Runtime View](06_runtime_view.md) | Important runtime scenarios and flows |
| [07 Deployment View](07_deployment_view.md) | Docker and local deployment topology |
| [08 Crosscutting Concepts](08_crosscutting_concepts.md) | Persistence, AI/ML, Error Handling, etc. |
| [09 Architecture Decisions](09_architecture_decisions.md) | Major architectural decision records (ADRs) |
| [10 Quality Requirements](10_quality_requirements.md) | Measurable system quality targets |
| [11 Risks & Technical Debt](11_technical_risks_and_debt.md) | Identified risks and mitigations |
| [12 Glossary](12_glossary.md) | Domain and technical terms used in the project |

## Diagrams

* [System Context](diagrams/system_context.mmd)
* [Container Context](diagrams/container_context.mmd)
* [Building Blocks](diagrams/building_blocks.mmd)
* [Runtime Request Flow](diagrams/runtime_request_flow.mmd)
* [Authentication Flow](diagrams/authentication_flow.mmd)
* [Database Flow](diagrams/database_flow.mmd)
* [AI/ML Flow](diagrams/ai_ml_flow.mmd)
* [Error Flow](diagrams/error_flow.mmd)
* [Deployment](diagrams/deployment.mmd)
* [Component Relationships](diagrams/component_relationships.mmd)

## Appendices

* [Function Reference](appendices/function_reference.md)
* [API Reference](appendices/api_reference.md)
* [Database Reference](appendices/database_reference.md)
* [AI/ML Reference](appendices/ai_ml_reference.md)
* [Technology Reference](appendices/technology_reference.md)

---

# Architecture Summary

**System Type**: Automated Face Recognition & Attendance Tracking Pipeline
**Primary Architecture Style**: Monolithic Script-based Data Processing Pipeline
**Frontend**: None (Raw Video Feed processing and outputting annotated frames)
**Backend**: Python Core Logic (`main.py`)
**Database**: Local File System (CSV for data, `.jpg` for cropped faces)
**AI/ML**: YOLOv8 (Detection), ByteTrack (Tracking), FaceNet (Embeddings), SVM (Classification)
**External Services**: None (Fully local AI execution)
**Deployment**: Local Execution or Docker Container

**Main Entry Point**: `main.py`
**Main Request Flow**: Video Frame -> YOLOv8 Detection -> Line Crossing Logic -> Face Cropping -> FaceNet Embedding -> SVM Prediction -> CSV Logging

**Major Building Blocks**: 
- Video Processing (`supervision`)
- Object Detection (`ultralytics`)
- Feature Extraction (`keras-facenet`)
- Classifier (`scikit-learn`)
- Persistence System (`csv`, `os`)

**Key Architecture Decisions**:
- Use YOLOv8 over standard MTCNN for real-time video face detection due to speed and integration with ByteTrack.
- Use CSV for local logging instead of a heavy Relational DB for simplicity.
- Require high classification probability (`>= 0.87`) to log attendance.

**Major Quality Attributes**:
- **Accuracy**: Highly dependent on the FaceNet+SVM threshold.
- **Performance**: Real-time or near-real-time processing of video streams.

**Major Technical Risks**:
- Concurrency issues when scaling (CSV locking).
- Scalability limits due to monolithic synchronous processing of frames.
