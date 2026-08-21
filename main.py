
import time
import numpy as np
import supervision as sv
from ultralytics import YOLO
from facenet_files.facent_svm_rec_passing import predict_face
from supervision.annotators import core

import cv2 as cv
from datetime import datetime

import uuid
import csv
import os


# ─────────────────────────────────────────────────────────────────────────────
# Model / tracker singletons  (loaded once at import time)
# ─────────────────────────────────────────────────────────────────────────────
model = YOLO("yolo_models/yolov8n-face.pt")
tracker = sv.ByteTrack()          # single instance — was accidentally duplicated before
box_annotator   = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

saved_names: list = []


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame callback  (used by dashboard.py and the standalone loop below)
# ─────────────────────────────────────────────────────────────────────────────
def callback(frame: np.ndarray, _: int) -> np.ndarray:
    """
    Process one frame: detect faces → track → recognise → annotate.

    Returns:
        annotated_frame: BGR numpy array with bounding boxes + labels drawn.
    """
    results    = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # Draw bounding boxes first (labels added after recognition loop below)
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)

    # ── Drawing the attendance line (horizontal, 50 px above centre) ─────────
    height, width, _ = frame.shape
    line_y      = (height // 2) - 50
    start_point = (0, line_y)
    end_point   = (width, line_y)
    cv.line(annotated_frame, start_point, end_point, (0, 255, 0), 1)

    # ── Per-detection face recognition ───────────────────────────────────────
    facenet_results    = []
    result_probabiltys = []

    for detection in detections.xyxy:
        x1, y1, x2, y2 = map(int, detection[:4])

        # Extract the face crop
        face = frame[y1:y2, x1:x2]

        print("Passing extracted face to recognition model …")
        facenet_result, result_probabilty = predict_face(face)
        print("Recognition complete")

        name      = facenet_result
        timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        filename  = f"{name}_{timestamp}.jpg"

        current_date = datetime.now().strftime('%Y_%m_%d')
        output_dir   = os.path.join('marked_attendance', current_date)
        os.makedirs(output_dir, exist_ok=True)
        filepath  = os.path.join(output_dir, filename)
        hyperlink = os.path.abspath(filepath)

        # ── CSV attendance sheet ──────────────────────────────────────────────
        """ Each day's folder gets a single CSV:
            Name | UniqueID | Timestamp | Hyperlink (path to face-crop image)
        """
        csv_file_path = os.path.join(output_dir, f"{current_date}_attendance_sheet.csv")
        csv_header    = ['Name', 'UniqueID', 'Timestamp', 'Hyperlink']
        unique_id     = str(uuid.uuid4())

        if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(csv_header)

        # Save face + log attendance when the person crosses the line
        if (y1 <= line_y <= y2) and result_probabilty >= 0.87 and name not in saved_names:
            cv.imwrite(filepath, face)
            # Fixed: `filename.split('_'[0])` was a latent bug — index goes outside string
            first_name = filename.split('_')[0]
            saved_names.append(first_name)

            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([first_name, unique_id, timestamp, hyperlink])

        facenet_results.append(facenet_result)
        result_probabiltys.append(result_probabilty)

    # ── Build labels AFTER recognition; annotate once with correct API ────────
    if detections.tracker_id is not None and len(facenet_results) == len(detections.tracker_id):
        labels = [
            f"#{tracker_id} {name} ({prob:.2f})"
            for tracker_id, name, prob
            in zip(detections.tracker_id, facenet_results, result_probabiltys)
        ]
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels
        )

    cv.imwrite('live.png', annotated_frame)
    return annotated_frame


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point — manual capture/write loop with 30 fps pacing.
# Replaces sv.process_video() which has no fps-override argument; without
# pacing the saved video plays back fast-forwarded when inference < 30 fps.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SOURCE_PATH = "test_datas/Deepak.mp4"
    TARGET_PATH = "result_datas/testig_video_result.mp4"
    TARGET_FPS  = 30

    cap = cv.VideoCapture(SOURCE_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {SOURCE_PATH}")

    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(TARGET_PATH) or ".", exist_ok=True)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(TARGET_PATH, fourcc, TARGET_FPS, (w, h))

    frame_interval  = 1.0 / TARGET_FPS
    next_write_time = time.time()
    last_annotated  = None
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_annotated = callback(frame, idx)
        idx += 1

        # Frame-pacing: duplicate latest annotated frame to fill real-time gaps
        # so the output video plays at correct speed even when inference is slow.
        now = time.time()
        while next_write_time <= now:
            if last_annotated is not None:
                writer.write(last_annotated)
            next_write_time += frame_interval

    # Flush any remaining wall-clock slots up to the last source frame
    now = time.time()
    while next_write_time <= now:
        if last_annotated is not None:
            writer.write(last_annotated)
        next_write_time += frame_interval

    cap.release()
    writer.release()
    print(f"Done. {idx} source frames processed → {TARGET_PATH}")