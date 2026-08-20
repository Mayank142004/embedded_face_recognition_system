# 01. Introduction and Goals

## 1.1 Requirements Overview

The Face Recognition System automates employee attendance tracking by replacing traditional methods (manual logs, ID cards, fingerprint scanners) with non-intrusive facial recognition technology. 

**Core Problem Solved:** It eliminates manual interaction for attendance logging.
**Users:** Administrators reviewing attendance logs; Employees (passively interacting with the system).
**Major Use Case:** The system continuously monitors an area (e.g., an office entrance). When an employee crosses a virtual "attendance line" in the camera's view, their face is detected, cropped, identified using AI, and their attendance is logged along with a saved face image for proof.

## 1.2 Quality Goals

| Quality Goal | Description | Motivation |
| ------------ | ----------- | ---------- |
| **Accuracy** | The system must correctly identify employees with high confidence to prevent false attendance records. | Avoid incorrectly logging attendance for the wrong employee or unknown individuals. |
| **Performance** | The system must process video frames rapidly to track individuals moving in real-time. | A slow pipeline will drop frames, missing employees crossing the attendance line. |
| **Maintainability** | The structure should allow easy retraining or replacing of the ML model. | New employees must be added to the SVM classifier regularly. |
| **Usability** | The system must operate passively without requiring employee interaction. | Reduce friction for employees entering the office. |

## 1.3 Stakeholders

| Stakeholder | Interest | Architectural Concern |
| ----------- | -------- | --------------------- |
| **Administrators** | Accurate attendance records for payroll and monitoring. | The output CSV files must be reliable, timestamped correctly, and easy to parse. Image proofs must be properly linked. |
| **Employees** | Frictionless entry into the premises. | The system must process faces fast enough so they don't have to stop and wait. |
| **System Operators** | Easy deployment and operation of the tracker. | The system must be self-contained and easily deployable via Docker or simple Python scripts. |
| **Developers** | Easy debugging and ML model updates. | Clear separation between face detection (YOLO), embedding (FaceNet), and classification (SVM). |
