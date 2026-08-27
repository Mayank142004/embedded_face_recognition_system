"""
gcs_storage.py — Google Cloud Storage operations for photos, embeddings, and models.
"""
import io
import os
from typing import Optional

import numpy as np
from google.cloud import storage

from config import (
    GCS_BUCKET_NAME,
    GCS_DATASET_PREFIX,
    GCS_MODEL_PREFIX,
    GCS_EMBEDDINGS_PREFIX,
    MODEL_FILENAME,
)

# ── Singleton bucket ──────────────────────────────────────
_client: Optional[storage.Client] = None
_bucket = None


def _get_bucket():
    global _client, _bucket
    if _bucket is None:
        _client = storage.Client()
        _bucket = _client.bucket(GCS_BUCKET_NAME)
    return _bucket


# ═══════════════════════════════════════════════════════════
# Generic upload / download
# ═══════════════════════════════════════════════════════════
def upload_file(local_path: str, gcs_path: str) -> str:
    """Upload a local file to GCS. Returns the gs:// URL."""
    blob = _get_bucket().blob(gcs_path)
    blob.upload_from_filename(local_path)
    return f"gs://{GCS_BUCKET_NAME}/{gcs_path}"


def download_file(gcs_path: str, local_path: str) -> str:
    """Download a GCS object to a local file. Returns the local path."""
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    blob = _get_bucket().blob(gcs_path)
    blob.download_to_filename(local_path)
    return local_path


def file_exists(gcs_path: str) -> bool:
    return _get_bucket().blob(gcs_path).exists()


def list_blobs(prefix: str) -> list[str]:
    return [b.name for b in _get_bucket().list_blobs(prefix=prefix)]


def delete_blob(gcs_path: str):
    blob = _get_bucket().blob(gcs_path)
    if blob.exists():
        blob.delete()


# ═══════════════════════════════════════════════════════════
# Photo operations  — gs://<bucket>/dataset/{emp_id}/
# ═══════════════════════════════════════════════════════════
def upload_employee_photo(
    emp_id: str,
    local_photo_path: str,
    filename: str = None,
) -> str:
    """Upload a single photo to GCS under dataset/{emp_id}/."""
    if filename is None:
        filename = os.path.basename(local_photo_path)
    gcs_path = f"{GCS_DATASET_PREFIX}{emp_id}/{filename}"
    return upload_file(local_photo_path, gcs_path)


def download_employee_photos(emp_id: str, local_dir: str) -> list[str]:
    """Download all photos for one employee from GCS."""
    prefix = f"{GCS_DATASET_PREFIX}{emp_id}/"
    blobs = list_blobs(prefix)
    os.makedirs(local_dir, exist_ok=True)
    downloaded = []
    for blob_name in blobs:
        fname = os.path.basename(blob_name)
        if not fname:
            continue
        lp = os.path.join(local_dir, fname)
        download_file(blob_name, lp)
        downloaded.append(lp)
    return downloaded


def download_all_dataset(local_dataset_dir: str) -> dict:
    """
    Download the entire dataset from GCS to local.
    Returns {emp_id: [local_path, ...]}.
    """
    prefix = GCS_DATASET_PREFIX
    blobs = list_blobs(prefix)
    result: dict[str, list[str]] = {}
    for blob_name in blobs:
        parts = blob_name[len(prefix):].split("/")
        if len(parts) < 2 or not parts[1]:
            continue
        emp_id, fname = parts[0], parts[1]
        local_path = os.path.join(local_dataset_dir, emp_id, fname)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        download_file(blob_name, local_path)
        result.setdefault(emp_id, []).append(local_path)
    return result


def upload_dataset_folder(emp_id: str, local_dir: str) -> list[str]:
    """Upload all images in a local employee folder to GCS."""
    uploaded = []
    if not os.path.isdir(local_dir):
        return uploaded
    for fname in os.listdir(local_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            lp = os.path.join(local_dir, fname)
            url = upload_employee_photo(emp_id, lp, fname)
            uploaded.append(url)
    return uploaded


# ═══════════════════════════════════════════════════════════
# Model operations  — gs://<bucket>/models/
# ═══════════════════════════════════════════════════════════
def upload_model(local_model_path: str) -> str:
    gcs_path = f"{GCS_MODEL_PREFIX}{MODEL_FILENAME}"
    return upload_file(local_model_path, gcs_path)


def download_model(local_model_path: str) -> str:
    gcs_path = f"{GCS_MODEL_PREFIX}{MODEL_FILENAME}"
    return download_file(gcs_path, local_model_path)


# ═══════════════════════════════════════════════════════════
# Embedding operations  — gs://<bucket>/embeddings/
# ═══════════════════════════════════════════════════════════
def upload_embedding(emp_id: str, embedding: np.ndarray) -> str:
    """Upload a numpy embedding array to GCS as .npy."""
    gcs_path = f"{GCS_EMBEDDINGS_PREFIX}{emp_id}.npy"
    blob = _get_bucket().blob(gcs_path)
    buf = io.BytesIO()
    np.save(buf, embedding)
    buf.seek(0)
    blob.upload_from_file(buf, content_type="application/octet-stream")
    return f"gs://{GCS_BUCKET_NAME}/{gcs_path}"


def download_embedding(emp_id: str) -> Optional[np.ndarray]:
    """Download one employee's embedding from GCS."""
    gcs_path = f"{GCS_EMBEDDINGS_PREFIX}{emp_id}.npy"
    blob = _get_bucket().blob(gcs_path)
    if not blob.exists():
        return None
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return np.load(buf)


def download_all_embeddings() -> dict:
    """Download all embeddings.  Returns {emp_id: np.ndarray}."""
    prefix = GCS_EMBEDDINGS_PREFIX
    blobs = list_blobs(prefix)
    result = {}
    for name in blobs:
        if name.endswith(".npy"):
            eid = os.path.splitext(os.path.basename(name))[0]
            emb = download_embedding(eid)
            if emb is not None:
                result[eid] = emb
    return result


def upload_npz(local_path: str, gcs_filename: str = "embeddings_cache.npz") -> str:
    """Upload the compressed embeddings cache to GCS."""
    gcs_path = f"{GCS_EMBEDDINGS_PREFIX}{gcs_filename}"
    return upload_file(local_path, gcs_path)
