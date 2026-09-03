"""
main.py — Raspberry Pi inference pipeline.

Camera frame -> motion gate -> YOLO -> ByteTrack -> FaceNet -> SVM
             -> line crossing -> MQTT, with the annotated frame streamed
             to the FastAPI server over WebSocket.

The pipeline is deliberately asymmetric: video streams at the capture rate so
the dashboard looks smooth, while the heavy AI runs on its own schedule
(AI_INTERVAL_SEC) and only when the motion gate says something changed.
"""
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

from config import (
    YOLO_MODEL_PATH,
    WS_BASE_URL,
    MODEL_SYNC_INTERVAL,
    AI_INTERVAL_SEC,
    FACE_RECONFIRM_SEC,
    STREAM_WIDTH,
    STREAM_JPEG_QUALITY,
    MOTION_GATE_ENABLED,
    PERF_LOG_INTERVAL,
)
from facenet_files.facent_svm_rec_passing import predict_face
from line_crossing import AttendanceDetector
from db import get_employee_name_map
from model_sync import start_background_sync
from motion_gate import MotionGate

logger = logging.getLogger(__name__)

start_background_sync(interval_seconds=MODEL_SYNC_INTERVAL)


# ── Thread-Safe WebSocket Streamer ──────────────────────────
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
                    # Resize before encoding to save CPU and bandwidth
                    h, w = frame.shape[:2]
                    new_w = STREAM_WIDTH
                    new_h = int(new_w * h / w)
                    small = cv.resize(frame, (new_w, new_h))
                    _, buffer = cv.imencode(
                        '.jpg', small, [cv.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
                    )
                    self.ws.send_binary(buffer.tobytes())
            except queue.Empty:
                pass
            except Exception:
                # C5 FIX: Close the old socket before dropping the reference
                if self.ws:
                    try:
                        self.ws.close()
                    except Exception:
                        pass
                self.ws = None

    def send_frame(self, frame):
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            # Drop frame if network is lagging behind
            pass


# Only the analyzed feed is streamed. The raw feed was a second full
# resize + JPEG encode of essentially the same image, every frame.
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
motion_gate = MotionGate()

_emp_dict_cache = {}
_emp_dict_lock = threading.Lock()


def _bg_emp_dict_refresh():
    """Background thread that refreshes the employee dict every 30 seconds.
    Never blocks the camera loop, even if MongoDB is unreachable."""
    global _emp_dict_cache
    while True:
        try:
            fresh = get_employee_name_map()
            with _emp_dict_lock:
                _emp_dict_cache = fresh
        except Exception:
            pass
        time.sleep(30.0)


_emp_thread = threading.Thread(target=_bg_emp_dict_refresh, daemon=True)
_emp_thread.start()


def _get_emp_dict():
    with _emp_dict_lock:
        return _emp_dict_cache


_last_confidence = {}

# ── AI scheduling state ────────────────────────────────────
_last_ai_time = 0.0
_last_detections = None
_last_labels = []

# Cache recognition results per track ID to skip FaceNet
_face_cache = {}        # tracker_id -> (emp_id, confidence, monotonic_time)

# ── Perf counters (step 9 instrumentation) ─────────────────
_perf = {
    "frames": 0,
    "ai_passes": 0,
    "yolo_ms": 0.0,
    "facenet_ms": 0.0,
    "facenet_calls": 0,
    "ai_ms": 0.0,
    "last_log": time.monotonic(),
}


def _read_soc_temp():
    """Return SoC temperature in C, or None if unavailable."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def _maybe_log_perf():
    now = time.monotonic()
    elapsed = now - _perf["last_log"]
    if elapsed < PERF_LOG_INTERVAL:
        return

    frames = _perf["frames"]
    passes = _perf["ai_passes"]
    gs = motion_gate.stats()
    temp = _read_soc_temp()

    logger.info(
        "perf | %.1f fps capture | AI %.2f/s (%d passes) | "
        "YOLO %.0f ms | FaceNet %.0f ms x%d | AI total %.0f ms | "
        "gate open %.0f%% | temp %s",
        frames / elapsed if elapsed else 0.0,
        passes / elapsed if elapsed else 0.0,
        passes,
        _perf["yolo_ms"] / passes if passes else 0.0,
        _perf["facenet_ms"] / _perf["facenet_calls"] if _perf["facenet_calls"] else 0.0,
        _perf["facenet_calls"],
        _perf["ai_ms"] / passes if passes else 0.0,
        gs["open_pct"],
        f"{temp:.1f}C" if temp is not None else "n/a",
    )

    _perf.update(frames=0, ai_passes=0, yolo_ms=0.0, facenet_ms=0.0,
                 facenet_calls=0, ai_ms=0.0, last_log=now)
    motion_gate.reset_stats()


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    global _last_ai_time, _last_detections, _last_labels

    ih, iw = frame.shape[:2]
    line_y = (ih // 2) - 50
    attendance_detector.set_line_y(line_y)

    now_mono = time.monotonic()
    _perf["frames"] += 1

    # 1. Motion gate — an entrance camera watches an empty corridor most of
    #    the day, and running YOLO on that is where the idle heat comes from.
    moving = motion_gate.update(frame) if MOTION_GATE_ENABLED else True

    # Once the keep-alive has expired there is genuinely nobody there, so
    # drop the frozen boxes rather than leaving them painted on the stream.
    if not moving and motion_gate.is_idle():
        _last_detections = None
        _last_labels = []

    # 2. Heavy AI pipeline — time-scheduled, not every-Nth-frame, so its rate
    #    no longer silently changes when the capture rate does.
    if moving and (now_mono - _last_ai_time) >= AI_INTERVAL_SEC:
        _last_ai_time = now_mono
        ai_t0 = time.perf_counter()

        results = model.predict(frame)
        _perf["yolo_ms"] += model.last_ms

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
            active_tids = set()
            for i, det in enumerate(detections.xyxy):
                tid = int(detections.tracker_id[i])
                active_tids.add(tid)
                x1, y1, x2, y2 = map(int, det[:4])
                face = frame[y1:y2, x1:x2]

                # Check cache before running expensive FaceNet
                cached = _face_cache.get(tid)
                if cached and (now_mono - cached[2]) < FACE_RECONFIRM_SEC:
                    eid, conf = cached[0], cached[1]
                elif face.size == 0:
                    eid, conf = "unknown", 0.0
                else:
                    try:
                        fn_t0 = time.perf_counter()
                        eid, conf = predict_face(face)
                        _perf["facenet_ms"] += (time.perf_counter() - fn_t0) * 1000.0
                        _perf["facenet_calls"] += 1
                        _face_cache[tid] = (eid, conf, now_mono)
                    except Exception:
                        eid, conf = "unknown", 0.0

                emp_ids.append(eid)
                final_confs.append(conf)
                if eid != "unknown":
                    _last_confidence[eid] = conf

                face_cys.append((y1 + y2) // 2)
                tid_list.append(tid)

            # Prune stale entries from cache (tracks that disappeared)
            for stale_tid in list(_face_cache.keys()):
                if stale_tid not in active_tids:
                    del _face_cache[stale_tid]

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

        _perf["ai_passes"] += 1
        _perf["ai_ms"] += (time.perf_counter() - ai_t0) * 1000.0

    # 3. Draw the frozen/tracked boxes on the current frame
    annotated = frame.copy()
    cv.line(annotated, (0, line_y), (iw, line_y), (0, 255, 0), 1)

    if _last_detections is not None:
        annotated = box_annotator.annotate(annotated, detections=_last_detections)
        if _last_labels:
            annotated = label_annotator.annotate(
                annotated, detections=_last_detections, labels=_last_labels
            )

    # 4. Send Analyzed Feed
    stream_analyzed.send_frame(annotated)

    _maybe_log_perf()
    return annotated
