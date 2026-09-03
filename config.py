"""
config.py — Central configuration for the Face Attendance System.
Reads from environment variables / .env file.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars

# ── Project root ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── MongoDB ────────────────────────────────────────────────
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://192.168.1.29:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "face_attendance")

# ── API / WebSockets (FastAPI on Laptop) ───────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://192.168.1.29:8000")
WS_BASE_URL = os.getenv("WS_BASE_URL", "ws://192.168.1.29:8000")

# ── Cloud / Google Cloud Storage ───────────────────────────
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "face-attendance-bucket")
# Set GOOGLE_APPLICATION_CREDENTIALS env var to your service-account JSON path

# ── MQTT ───────────────────────────────────────────────────
MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "192.168.1.29")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "attendance/events")              # Pi camera topic
MQTT_LOCAL_TOPIC = os.getenv("MQTT_LOCAL_TOPIC", "attendance/local_events")  # Laptop camera topic
MQTT_CLIENT_ID_PUB = os.getenv("MQTT_CLIENT_ID_PUB", "pi-publisher")
MQTT_CLIENT_ID_LOCAL_PUB = os.getenv("MQTT_CLIENT_ID_LOCAL_PUB", "local-publisher")
MQTT_CLIENT_ID_SUB = os.getenv("MQTT_CLIENT_ID_SUB", "attendance-subscriber")

# ── Timezone (date boundaries use this for "today") ────────
TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")

# ── Local paths ────────────────────────────────────────────
DATASET_DIR = os.getenv("DATASET_DIR", str(PROJECT_ROOT / "data" / "dataset"))
PHOTOS_DIR = os.getenv("PHOTOS_DIR", str(PROJECT_ROOT / "data" / "photos"))
MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "facenet_models"))
MODEL_FILENAME = "svm_classifier.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    str(PROJECT_ROOT / "yolo_models" / "yolov8n-face_saved_model" / "yolov8n-face_float16.tflite"),
)
RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", str(PROJECT_ROOT / "recordings"))

# ── GCS path prefixes ─────────────────────────────────────
GCS_DATASET_PREFIX = "dataset/"          # gs://<bucket>/dataset/{emp_id}/
GCS_MODEL_PREFIX = "models/"             # gs://<bucket>/models/svm_classifier.pkl
GCS_EMBEDDINGS_PREFIX = "embeddings/"    # gs://<bucket>/embeddings/{emp_id}.npy

# ── Recognition tunables ──────────────────────────────────
DEBOUNCE_SECONDS = float(os.getenv("DEBOUNCE_SECONDS", "3.0"))
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.85"))

# ── Model sync (Pi background thread) ────────────────────
MODEL_SYNC_INTERVAL = int(os.getenv("MODEL_SYNC_INTERVAL", "300"))  # seconds (default 5 min)


# ── Pi runtime tunables ───────────────────────────────────
# Capture/stream rate. The AI runs on its own schedule (AI_INTERVAL_SEC),
# so this only controls how smooth the video looks, not how often we detect.
PI_TARGET_FPS = int(os.getenv("PI_TARGET_FPS", "15"))

# Minimum seconds between heavy AI passes (YOLO -> FaceNet -> line check).
# Floor is set by walking speed: a person must be sampled 2-3 times while
# below the attendance line, so do not raise this much above ~0.5s.
AI_INTERVAL_SEC = float(os.getenv("AI_INTERVAL_SEC", "0.35"))

# Threads per TFLite interpreter. The Pi 3 has 4 cores; leaving one free for
# capture, JPEG encode, MQTT and networking beats saturating all four.
TFLITE_NUM_THREADS = int(os.getenv("TFLITE_NUM_THREADS", "3"))

# Re-run FaceNet on an already-identified track after this many seconds.
FACE_RECONFIRM_SEC = float(os.getenv("FACE_RECONFIRM_SEC", "10.0"))

# ── Outgoing stream (Pi -> server) ────────────────────────
STREAM_WIDTH = int(os.getenv("STREAM_WIDTH", "480"))
STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "50"))

# ── Motion gate ───────────────────────────────────────────
# An entrance camera watches an empty corridor most of the day. Skipping YOLO
# while nothing moves is the single largest thermal saving available, and it
# costs no accuracy: the gate opens before the pipeline would have run anyway.
MOTION_GATE_ENABLED = os.getenv("MOTION_GATE_ENABLED", "1") not in ("0", "false", "False")
# Per-pixel intensity delta counted as "changed" (0-255).
MOTION_PIXEL_DELTA = int(os.getenv("MOTION_PIXEL_DELTA", "12"))
# Fraction of changed pixels needed to call it motion.
MOTION_MIN_FRACTION = float(os.getenv("MOTION_MIN_FRACTION", "0.004"))
# Keep the AI running this long after motion stops, so someone standing
# still at the line is not dropped mid-approach.
MOTION_KEEPALIVE_SEC = float(os.getenv("MOTION_KEEPALIVE_SEC", "2.0"))

# ── Performance logging ───────────────────────────────────
# Seconds between perf summary lines (YOLO/FaceNet latency, gate rate, temp).
PERF_LOG_INTERVAL = float(os.getenv("PERF_LOG_INTERVAL", "20.0"))
