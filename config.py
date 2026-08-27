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
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "attendance/events")
MQTT_CLIENT_ID_PUB = os.getenv("MQTT_CLIENT_ID_PUB", "pi-publisher")
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
    str(PROJECT_ROOT / "yolo_models" / "yolov8n-face_saved_model" / "yolov8n-face_float32.tflite"),
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
