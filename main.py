import logging
import os
import time
import queue

import cv2 as cv
import numpy as np
import supervision as sv
from yolo_tflite import YOLOTFLite
import websocket
import threading

from config import YOLO_MODEL_PATH, WS_BASE_URL, MODEL_SYNC_INTERVAL
from facenet_files.facent_svm_rec_passing import predict_face
from line_crossing import AttendanceDetector
from db import get_employee_dict
from model_sync import start_background_sync

logger = logging.getLogger(__name__)

start_background_sync(interval_seconds=MODEL_SYNC_INTERVAL)

# ── Dual Thread-Safe WebSocket Streamers ────────────────────
class WSStreamer:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.ws = None
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _connect(self):
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(f"{WS_BASE_URL}/ws/stream/pi/{self.endpoint}", timeout=2.0)
        except Exception:
            self.ws = None

    def _run(self):
        while self.running:
            try:
                frame = self.q.get(timeout=1.0)
                if self.ws is None:
                    self._connect()
                if self.ws:
                    _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 50])
                    self.ws.send_binary(buffer.tobytes())
            except queue.Empty:
                pass
            except Exception:
                self.ws = None

    def send_frame(self, frame):
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            # Drop frame if network is lagging behind
            pass

stream_raw = WSStreamer("raw")
stream_analyzed = WSStreamer("analyzed")

# ── MQTT ───────────────────────────────────────────────────
try:
    from mqtt_publisher import publish_event
except Exception:
    def publish_event(emp_id, event_type, confidence=0.0):
        pass

# ── Globals ────────────────────────────────────────────────
model = YOLOTFLite(YOLO_MODEL_PATH)
tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, minimum_consecutive_frames=1)
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
attendance_detector = AttendanceDetector(line_y=0)

_emp_dict_cache = {}
_emp_dict_ts = 0.0

def _get_emp_dict():
    global _emp_dict_cache, _emp_dict_ts
    now = time.time()
    if now - _emp_dict_ts > 30.0:
        try:
            _emp_dict_cache = get_employee_dict()
            _emp_dict_ts = now
        except Exception:
            pass
    return _emp_dict_cache

_last_confidence = {}

# ── Frame Skip State ───────────────────────────────────────
_frame_counter = 0
_last_detections = None
_last_labels = []

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    global _frame_counter, _last_detections, _last_labels
    
    # 1. Send Raw Feed instantly
    stream_raw.send_frame(frame)
    
    ih, iw, _ = frame.shape
    line_y = (ih // 2) - 50
    attendance_detector.set_line_y(line_y)
    
    # 2. Only run Heavy AI Pipeline every 5th frame
    if _frame_counter % 5 == 0:
        results = model.predict(frame)
        xyxy, confidences = [], []
        for res in results:
            xyxy.append(res[:4])
            confidences.append(res[4])
            
        if xyxy:
            detections = sv.Detections(
                xyxy=np.array(xyxy),
                confidence=np.array(confidences),
                class_id=np.zeros(len(xyxy), dtype=int)
            )
        else:
            detections = sv.Detections.empty()
            
        detections = tracker.update_with_detections(detections)
        _last_detections = detections
        
        emp_dict = _get_emp_dict()
        emp_ids, face_cys, tid_list, final_confs = [], [], [], []
        
        if detections.tracker_id is not None:
            for i, det in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, det[:4])
                face = frame[y1:y2, x1:x2]
                
                if face.size == 0:
                    emp_ids.append("unknown")
                    final_confs.append(0.0)
                else:
                    try:
                        eid, conf = predict_face(face)
                        emp_ids.append(eid)
                        final_confs.append(conf)
                        _last_confidence[eid] = conf
                    except Exception:
                        emp_ids.append("unknown")
                        final_confs.append(0.0)
                
                face_cys.append((y1 + y2) // 2)
                tid_list.append(int(detections.tracker_id[i]))
        
        # Check attendance
        if tid_list:
            events = attendance_detector.update(tid_list, face_cys, emp_ids)
            for ev in events:
                eid = ev["emp_id"]
                conf = _last_confidence.get(eid, 0.0)
                logger.info(f"Attendance: {eid} → IN")
                publish_event(eid, "in", conf)
                
        # Build labels
        _last_labels = []
        if detections.tracker_id is not None:
            for tid, eid, conf in zip(detections.tracker_id, emp_ids, final_confs):
                ename = emp_dict.get(eid, eid) if eid != "unknown" else "Unknown"
                _last_labels.append(f"#{tid} {ename} ({conf:.2f})")

    _frame_counter += 1
    
    # 3. Draw the frozen/tracked boxes on the current frame
    annotated = frame.copy()
    cv.line(annotated, (0, line_y), (iw, line_y), (0, 255, 0), 1)
    
    if _last_detections is not None:
        annotated = box_annotator.annotate(annotated, detections=_last_detections)
        if _last_labels:
            annotated = label_annotator.annotate(annotated, detections=_last_detections, labels=_last_labels)
            
    # 4. Send Analyzed Feed
    stream_analyzed.send_frame(annotated)
    return annotated

