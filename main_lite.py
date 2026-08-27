"""
main_lite.py — Ultra-lightweight per-frame face detection pipeline.
Uses MediaPipe for detection (No YOLO/PyTorch), TFLite for FaceNet.
"""
import logging
import os
import time

import cv2 as cv
import numpy as np
import supervision as sv
import mediapipe as mp
import websocket
import threading

from config import WS_BASE_URL, MODEL_SYNC_INTERVAL
from facenet_files.facent_svm_rec_passing_lite import predict_face
from line_crossing import AttendanceDetector
from db import get_employee_dict
from model_sync import start_background_sync

logger = logging.getLogger(__name__)

# Start model synchronizer
start_background_sync(interval_seconds=MODEL_SYNC_INTERVAL)

# ── WebSocket Streamer ──────────────────────────────────────
_ws_app = None
def _get_ws():
    global _ws_app
    if _ws_app is None:
        try:
            _ws_app = websocket.WebSocket()
            _ws_app.connect(f"{WS_BASE_URL}/ws/stream/pi")
        except Exception:
            _ws_app = None
    return _ws_app

# ── MQTT publisher ──────────────────────────────────────────
_mqtt_available = False
try:
    from mqtt_publisher import publish_event
    _mqtt_available = True
except Exception:
    def publish_event(emp_id, event_type, confidence=0.0):
        return False

# ── Trackers & Detectors ────────────────────────────────────
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

attendance_detector = AttendanceDetector(line_y=0)

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# ── Employee dict cache ─────────────────────────────────────
_emp_dict_cache: dict = {}
_emp_dict_ts: float = 0.0
_EMP_DICT_TTL = 30.0

def _get_emp_dict() -> dict:
    global _emp_dict_cache, _emp_dict_ts
    now = time.time()
    if now - _emp_dict_ts > _EMP_DICT_TTL:
        try:
            _emp_dict_cache = get_employee_dict()
            _emp_dict_ts = now
        except Exception:
            pass
    return _emp_dict_cache

_last_confidence: dict = {}

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    ih, iw, _ = frame.shape
    
    # 1. MediaPipe Face Detection
    results = face_detector.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    xyxy = []
    confs = []
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            x2 = min(iw, max(0, x + w))
            y2 = min(ih, max(0, y + h))
            x1 = max(0, x)
            y1 = max(0, y)
            if x2 > x1 and y2 > y1:
                xyxy.append([x1, y1, x2, y2])
                confs.append(detection.score[0])
                
    if xyxy:
        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confs)
        )
    else:
        detections = sv.Detections.empty()

    # 2. ByteTrack tracking
    detections = tracker.update_with_detections(detections)

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)

    line_y = (ih // 2) - 50
    attendance_detector.set_line_y(line_y)
    cv.line(annotated_frame, (0, line_y), (iw, line_y), (0, 255, 0), 1)

    emp_dict = _get_emp_dict()
    emp_ids, confidences, face_cys, tid_list = [], [], [], []

    # 3. Recognition
    if detections.tracker_id is not None:
        for i, det in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, det[:4])
            face = frame[y1:y2, x1:x2]
            
            if face.size == 0:
                emp_ids.append("unknown")
                confidences.append(0.0)
                face_cys.append((y1 + y2) // 2)
                tid_list.append(int(detections.tracker_id[i]))
                continue

            try:
                emp_id, conf = predict_face(face)
            except Exception:
                emp_id, conf = "unknown", 0.0

            emp_ids.append(emp_id)
            confidences.append(conf)
            face_cys.append((y1 + y2) // 2)
            tid_list.append(int(detections.tracker_id[i]))
            _last_confidence[emp_id] = conf

    # 4. Attendance
    if tid_list:
        events = attendance_detector.update(
            tracker_ids=tid_list,
            face_centers_y=face_cys,
            emp_ids=emp_ids,
        )
        for ev in events:
            eid = ev["emp_id"]
            conf = _last_confidence.get(eid, 0.0)
            ename = emp_dict.get(eid, eid)
            logger.info("Attendance: %s (%s) → IN  (conf=%.2f)", eid, ename, conf)
            publish_event(eid, "in", conf)

    # 5. Annotation
    if detections.tracker_id is not None and len(emp_ids) == len(detections.tracker_id):
        labels = []
        for tid, eid, conf in zip(detections.tracker_id, emp_ids, confidences):
            ename = emp_dict.get(eid, eid) if eid != "unknown" else "Unknown"
            labels.append(f"#{tid} {ename} ({conf:.2f})")
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

    cv.imwrite("live.png", annotated_frame)
    
    ws = _get_ws()
    if ws is not None:
        try:
            _, buffer = cv.imencode('.jpg', annotated_frame, [cv.IMWRITE_JPEG_QUALITY, 80])
            ws.send_binary(buffer.tobytes())
        except Exception:
            global _ws_app
            _ws_app = None

    return annotated_frame
