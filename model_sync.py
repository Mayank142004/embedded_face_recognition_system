"""
model_sync.py — Background task for the Raspberry Pi to sync the latest SVM model.

Strategy (GCS-first, laptop-fallback):
  1. Try downloading the model from Google Cloud Storage (GCS).
     Check if the GCS file's last-modified timestamp is newer than the local file.
  2. If GCS fails → try downloading from the laptop's FastAPI server over WiFi.
  3. If both fail → keep running with the existing local .pkl and log a warning.
"""
import os
import json
import time
import logging
import threading
import requests

from config import API_BASE_URL, MODEL_DIR, MODEL_PATH, MODEL_FILENAME

logger = logging.getLogger(__name__)

CONFIG_JSON_PATH = os.path.join(MODEL_DIR, "config.json")

# ── GCS availability check (optional dependency) ─────────
_gcs_available = False
try:
    from gcs_storage import download_model as gcs_download_model, _get_bucket
    from config import GCS_MODEL_PREFIX
    _gcs_available = True
except Exception:
    logger.info("GCS library not available — GCS sync disabled, will use laptop fallback only.")


# ═══════════════════════════════════════════════════════════
# Local version tracking
# ═══════════════════════════════════════════════════════════
def _get_local_info() -> dict:
    """Read the local config.json with version + timestamps."""
    if not os.path.exists(CONFIG_JSON_PATH):
        return {"version": "default_v1", "updated_at": 0, "source": "bundled"}
    try:
        with open(CONFIG_JSON_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"version": "default_v1", "updated_at": 0, "source": "bundled"}


def _get_local_model_mtime() -> float:
    """Get the modification timestamp of the local .pkl file."""
    if os.path.exists(MODEL_PATH):
        return os.path.getmtime(MODEL_PATH)
    return 0


def _update_local_info(version: str, updated_at: float, source: str):
    """Write version info to config.json."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CONFIG_JSON_PATH, "w") as f:
        json.dump({
            "version": version,
            "updated_at": updated_at,
            "source": source,
            "synced_at": time.time(),
        }, f, indent=2)


def _hot_reload():
    """Reload the SVM model in memory without restarting."""
    from facenet_files.facent_svm_rec_passing import load_model
    load_model()


# ═══════════════════════════════════════════════════════════
# Source 1: Google Cloud Storage (Primary)
# ═══════════════════════════════════════════════════════════
def _try_gcs_download() -> bool:
    """
    Check if the GCS model is newer than the local file using
    the blob's last-modified timestamp. Download if newer.
    Returns True if a new model was downloaded.
    """
    if not _gcs_available:
        return False

    try:
        bucket = _get_bucket()
        gcs_path = f"{GCS_MODEL_PREFIX}{MODEL_FILENAME}"
        blob = bucket.blob(gcs_path)
        blob.reload()  # fetch metadata from GCS

        if not blob.exists():
            logger.debug("GCS: Model blob does not exist at %s", gcs_path)
            return False

        # Compare GCS last-modified vs local file mtime
        gcs_updated = blob.updated  # datetime object (UTC)
        gcs_timestamp = gcs_updated.timestamp()
        local_timestamp = _get_local_model_mtime()

        if gcs_timestamp <= local_timestamp:
            logger.debug("GCS: Model is up to date (GCS: %.0f, Local: %.0f)", gcs_timestamp, local_timestamp)
            return False

        # Newer model found on GCS — download it
        logger.info("GCS: Newer model found! GCS modified: %s, Local mtime: %.0f", gcs_updated, local_timestamp)
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        gcs_download_model(MODEL_PATH)

        version = f"gcs_{int(gcs_timestamp)}"
        _update_local_info(version, gcs_timestamp, source="gcs")
        _hot_reload()
        logger.info("✅ GCS: Model synced and hot-reloaded (version: %s)", version)
        return True

    except Exception as e:
        logger.warning("⚠️  GCS download failed: %s — falling back to laptop.", e)
        return False


# ═══════════════════════════════════════════════════════════
# Source 2: Laptop FastAPI Server (Fallback)
# ═══════════════════════════════════════════════════════════
def _try_fastapi_download() -> bool:
    """
    Ask the laptop's FastAPI server for the latest model version.
    Compare with the local version. Download if newer.
    Returns True if a new model was downloaded.
    """
    try:
        url = f"{API_BASE_URL}/api/model/latest"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        remote_version = data.get("version")
        download_url = data.get("url")
        remote_updated_at = data.get("updated_at", 0)

        if not remote_version or not download_url:
            logger.debug("Laptop: No model info available from FastAPI.")
            return False

        local_info = _get_local_info()
        local_version = local_info.get("version", "default_v1")

        if remote_version == local_version:
            logger.debug("Laptop: Model is up to date (version: %s)", local_version)
            return False

        # Also check timestamps — only download if remote is actually newer
        local_updated_at = local_info.get("updated_at", 0)
        if remote_updated_at and remote_updated_at <= local_updated_at:
            logger.debug("Laptop: Remote timestamp not newer, skipping.")
            return False

        logger.info("Laptop: Newer model found! Local: %s, Remote: %s", local_version, remote_version)

        # Download file directly via HTTP
        full_url = f"{API_BASE_URL}{download_url}" if download_url.startswith("/") else download_url
        model_resp = requests.get(full_url, stream=True, timeout=30)
        model_resp.raise_for_status()

        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in model_resp.iter_content(chunk_size=8192):
                f.write(chunk)

        _update_local_info(remote_version, remote_updated_at or time.time(), source="laptop")
        _hot_reload()
        logger.info("✅ Laptop: Model synced and hot-reloaded (version: %s)", remote_version)
        return True

    except Exception as e:
        logger.warning("⚠️  Laptop FastAPI download failed: %s", e)
        return False


# ═══════════════════════════════════════════════════════════
# Main sync function
# ═══════════════════════════════════════════════════════════
def sync_model_once() -> bool:
    """
    Try to sync the model:
      1. GCS (primary)
      2. Laptop FastAPI (fallback)
      3. Keep existing local model (last resort)
    Returns True if a new model was downloaded from either source.
    """
    # Try GCS first
    if _try_gcs_download():
        return True

    # Fallback to laptop
    if _try_fastapi_download():
        return True

    # Both failed — keep existing model
    local_info = _get_local_info()
    logger.info(
        "Model sync: No update available or both sources failed. "
        "Continuing with local model (version: %s, source: %s).",
        local_info.get("version"), local_info.get("source", "unknown"),
    )
    return False


# ═══════════════════════════════════════════════════════════
# Background thread
# ═══════════════════════════════════════════════════════════
def _sync_loop(interval_seconds: int):
    logger.info("Model sync background thread started (interval: %s s)", interval_seconds)
    local_info = _get_local_info()
    logger.info(
        "Current model on startup — version: %s, source: %s",
        local_info.get("version"), local_info.get("source", "unknown"),
    )

    while True:
        sync_model_once()
        time.sleep(interval_seconds)


def start_background_sync(interval_seconds: int = 300):
    """Start the model sync in a background daemon thread (default every 5 min)."""
    t = threading.Thread(target=_sync_loop, args=(interval_seconds,), daemon=True)
    t.start()
