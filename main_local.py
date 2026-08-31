"""
main_local.py — Laptop Local Camera Inference Pipeline.

Completely isolated from main.py (Raspberry Pi mode).
- Auto-detects USB camera; falls back to built-in camera.
- Same YOLO -> ByteTrack -> FaceNet -> SVM pipeline as main.py.
- Publishes attendance events to MQTT topic: attendance/local_events
  (separate from the Pi's attendance/events topic).
- Streams raw and analyzed frames over WebSocket to server.py,
  using stream types 'raw_local' and 'analyzed_local'.

Frame Skipping:
    Set FRAME_SKIP = 1  -> AI runs on EVERY frame (no skipping, high CPU).
    Set FRAME_SKIP = 5  -> AI runs every 5th frame (saves CPU, default).
"""


import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone

import cv2 as cv
import numpy as np
import supervision as sv
import websocket
import paho.mqtt.client as mqtt_lib

from config import (
    YOLO_MODEL_PATH,
    WS_BASE_URL,
    MQTT_LOCAL_TOPIC,
    MQTT_CLIENT_ID_LOCAL_PUB,
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
)
from facenet_files.facent_svm_rec_passing import predict_face
from line_crossing import AttendanceDetector
from db import get_employee_dict
from yolo_tflite import YOLOTFLite

logger = logging.getLogger(__name__)

# ── Frame Skip Interval ────────────────────────────────────
# Set to 1 to run AI on EVERY frame (laptop has more CPU).
# Set to 5 to run AI every 5th frame (saves CPU, same as Pi).
FRAME_SKIP = 5


# ── Camera Auto-Detection ──────────────────────────────────
def detect_camera() -> int:
    """
    Scan camera indices 0-3.
    Prefer the first non-zero index (external USB camera).
    Fall back to index 0 (built-in) if no USB camera is found.
    Returns the camera index to use.
    """
    usb_index = None
    builtin_index = None

    for idx in range(4):
        cap = cv.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                if idx == 0:
                    builtin_index = 0
                else:
                    if usb_index is None:
                        usb_index = idx

    if usb_index is not None:
        logger.info("Local mode: USB camera detected at index %d", usb_index)
        return usb_index

    if builtin_index is not None:
        logger.info("Local mode: No USB camera found, using built-in camera (index 0)")
        return builtin_index

    logger.error("Local mode: No camera found at indices 0-3")
    return 0


# ── Thread-Safe WebSocket Streamer (Local) ─────────────────
class WSStreamerLocal:
    """
    Sends JPEG frames to server.py over WebSocket in a background thread.
    Uses maxsize=1 queue to always send the freshest frame.
    Dropping frames is intentional — prevents network lag buildup.
    """
    def __init__(self, stream_type: str):
        self.stream_type = stream_type
        self.ws = None
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _connect(self):
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(
                f"{WS_BASE_URL}/ws/stream/pi/{self.stream_type}",
                timeout=2.0,
            )
            logger.info("Local WS streamer connected: %s", self.stream_type)
        except Exception as e:
            logger.warning("Local WS connect failed (%s): %s", self.stream_type, e)
            self.ws = None

    def _run(self):
        while self.running:
            try:
                frame = self.q.get(timeout=1.0)
                if self.ws is None:
                    self._connect()
                if self.ws:
                    # Resize before encoding to save CPU and bandwidth
                    h, w = frame.shape[:2]
                    new_w = 480
                    new_h = int(new_w * h / w)
                    small = cv.resize(frame, (new_w, new_h))
                    _, buf = cv.imencode(".jpg", small, [cv.IMWRITE_JPEG_QUALITY, 60])
                    self.ws.send_binary(buf.tobytes())
            except queue.Empty:
                pass
            except Exception:
                # C5 FIX: Close old socket before dropping reference
                if self.ws:
                    try:
                        self.ws.close()
                    except Exception:
                        pass
                self.ws = None  # reconnect on next frame

    def send_frame(self, frame: np.ndarray):
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            pass  # drop stale frame

    def stop(self):
        self.running = False


# ── MQTT Publisher (Local Topic) ───────────────────────────
_mqtt_local_client = None


def _get_local_mqtt_client():
    global _mqtt_local_client
    if _mqtt_local_client is None or not _mqtt_local_client.is_connected():
        _mqtt_local_client = mqtt_lib.Client(client_id=MQTT_CLIENT_ID_LOCAL_PUB)
        try:
            _mqtt_local_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
            _mqtt_local_client.loop_start()
            logger.info("Local MQTT publisher connected to %s:%s", MQTT_BROKER_HOST, MQTT_BROKER_PORT)
        except Exception as e:
            logger.error("Local MQTT connect failed: %s", e)
            _mqtt_local_client = None
    return _mqtt_local_client


def publish_local_event(emp_id: str, confidence: float = 0.0):
    """Publish an attendance event to the local camera MQTT topic."""
    payload = {
        "emp_id": emp_id,
        "status": "in",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "confidence": round(confidence, 4),
    }
    try:
        client = _get_local_mqtt_client()
        if client:
            client.publish(MQTT_LOCAL_TOPIC, json.dumps(payload), qos=1)
            logger.info("Local MQTT published: %s", payload)
    except Exception as e:
        logger.error("Local MQTT publish error: %s", e)


# ── Pipeline Singletons (isolated from main.py) ────────────
_model = None
_tracker = None
_box_annotator = None
_label_annotator = None
_attendance_detector = None
_stream_raw = None
_stream_analyzed = None

# ── Per-frame state (isolated from main.py globals) ────────
_frame_counter = 0
_last_detections = None
_last_labels = []
_last_confidence = {}

# P1 FIX: Cache recognition results per track ID to skip FaceNet
_face_cache = {}        # tracker_id -> (emp_id, confidence, monotonic_time)
RECONFIRM_SEC = 10.0    # Re-run FaceNet after this many seconds

# ── Employee dict cache ────────────────────────────────────
_emp_dict_cache = {}
_emp_dict_ts = 0.0


def _get_emp_dict() -> dict:
    global _emp_dict_cache, _emp_dict_ts
    now = time.time()
    if now - _emp_dict_ts > 30.0:
        try:
            _emp_dict_cache = get_employee_dict()
            _emp_dict_ts = now
        except Exception:
            pass
    return _emp_dict_cache


def load_local_models():
    """
    Load all models into module-level singletons.
    Called once when the dashboard starts Local Analysis.
    Safe to call multiple times — skips loading if already done.
    """
    global _model, _tracker, _box_annotator, _label_annotator
    global _attendance_detector, _stream_raw, _stream_analyzed

    if _model is not None:
        return  # already loaded

    logger.info("Local mode: Loading YOLO model from %s", YOLO_MODEL_PATH)
    _model = YOLOTFLite(YOLO_MODEL_PATH)
    _tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_consecutive_frames=1,
    )
    _box_annotator = sv.BoundingBoxAnnotator()
    _label_annotator = sv.LabelAnnotator()
    _attendance_detector = AttendanceDetector(line_y=0)
    _stream_raw = WSStreamerLocal("raw_local")
    _stream_analyzed = WSStreamerLocal("analyzed_local")
    logger.info("Local mode: All models loaded.")


def reset_local_state():
    """
    Reset per-session state. Call before starting a new local session
    so ByteTracker IDs and frame counters don't carry over between runs.
    """
    global _frame_counter, _last_detections, _last_labels, _last_confidence, _face_cache
    global _tracker
    _frame_counter = 0
    _last_detections = None
    _last_labels = []
    _last_confidence = {}
    _face_cache = {}
    if _tracker is not None:
        _tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_consecutive_frames=1,
        )


def callback_local(frame: np.ndarray, _: int) -> np.ndarray:
    """
    Process one frame from the local camera.

    Pipeline (every FRAME_SKIP-th frame):
      1. YOLO face detection
      2. ByteTrack tracking
      3. FaceNet + SVM recognition -> emp_id
      4. Attendance line cross check with debounce
      5. Publish to MQTT local topic
      6. Annotate frame

    On skipped frames: reuses frozen bounding boxes from last detection.
    Returns the annotated frame (BGR).
    """
    global _frame_counter, _last_detections, _last_labels

    # Stream raw frame immediately
    if _stream_raw:
        _stream_raw.send_frame(frame)

    ih, iw, _ = frame.shape
    line_y = (ih // 2) - 50
    _attendance_detector.set_line_y(line_y)

    if _frame_counter % FRAME_SKIP == 0:
        results = _model.predict(frame)
        xyxy, confidences = [], []
        for res in results:
            xyxy.append(res[:4])
            confidences.append(res[4])

        if xyxy:
            detections = sv.Detections(
                xyxy=np.array(xyxy),
                confidence=np.array(confidences),
                class_id=np.zeros(len(xyxy), dtype=int),
            )
        else:
            detections = sv.Detections.empty()

        detections = _tracker.update_with_detections(detections)
        _last_detections = detections

        emp_dict = _get_emp_dict()
        emp_ids, face_cys, tid_list, final_confs = [], [], [], []

        if detections.tracker_id is not None:
            now_mono = time.monotonic()
            active_tids = set()
            for i, det in enumerate(detections.xyxy):
                tid = int(detections.tracker_id[i])
                active_tids.add(tid)
                x1, y1, x2, y2 = map(int, det[:4])
                face = frame[y1:y2, x1:x2]

                # P1 FIX: Check cache before running expensive FaceNet
                cached = _face_cache.get(tid)
                if cached and (now_mono - cached[2]) < RECONFIRM_SEC:
                    eid, conf = cached[0], cached[1]
                elif face.size == 0:
                    eid, conf = "unknown", 0.0
                else:
                    try:
                        eid, conf = predict_face(face)
                        _face_cache[tid] = (eid, conf, now_mono)
                    except Exception:
                        eid, conf = "unknown", 0.0

                emp_ids.append(eid)
                final_confs.append(conf)
                if eid != "unknown":
                    _last_confidence[eid] = conf

                face_cys.append((y1 + y2) // 2)
                tid_list.append(tid)

            # Prune stale entries from cache
            for stale_tid in list(_face_cache.keys()):
                if stale_tid not in active_tids:
                    del _face_cache[stale_tid]

        if tid_list:
            events = _attendance_detector.update(tid_list, face_cys, emp_ids)
            for ev in events:
                eid = ev["emp_id"]
                conf = _last_confidence.get(eid, 0.0)
                logger.info("Local Attendance: %s -> IN (conf=%.2f)", eid, conf)
                publish_local_event(eid, conf)

        _last_labels = []
        if detections.tracker_id is not None:
            for tid, eid, conf in zip(detections.tracker_id, emp_ids, final_confs):
                ename = emp_dict.get(eid, eid) if eid != "unknown" else "Unknown"
                _last_labels.append(f"#{tid} {ename} ({conf:.2f})")

    _frame_counter += 1

    # Draw frozen/latest boxes on the frame
    annotated = frame.copy()
    cv.line(annotated, (0, line_y), (iw, line_y), (0, 255, 0), 1)

    if _last_detections is not None:
        annotated = _box_annotator.annotate(annotated, detections=_last_detections)
        if _last_labels:
            annotated = _label_annotator.annotate(
                annotated, detections=_last_detections, labels=_last_labels
            )

    if _stream_analyzed:
        _stream_analyzed.send_frame(annotated)

    return annotated
