"""
db.py — MongoDB client and CRUD operations for employees and attendance.

Collections:
  employees:  { emp_id, emp_name, photo_local_path, photo_gcs_url,
                face_embedding, registered_at }
  attendance: { emp_id, emp_name, timestamp, date, status }
"""
import os
from datetime import datetime, timezone
from typing import Optional

import pytz
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from config import MONGODB_URI, MONGODB_DB_NAME, TIMEZONE

# ── Singleton client ──────────────────────────────────────
_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_db() -> Database:
    """Return (and cache) the MongoDB database handle."""
    global _client, _db
    if _db is None:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        _db = _client[MONGODB_DB_NAME]
    return _db


def get_employees_col() -> Collection:
    return get_db()["employees"]


def get_attendance_col() -> Collection:
    return get_db()["attendance"]


def get_local_attendance_col() -> Collection:
    """Separate collection for laptop/local camera attendance."""
    return get_db()["attendance_local"]


# ── Helpers ────────────────────────────────────────────────
def _local_today() -> str:
    """Return today's date as YYYY-MM-DD in the configured timezone."""
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz).strftime("%Y-%m-%d")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ── Indexes ────────────────────────────────────────────────
def ensure_indexes():
    """Create or verify all required indexes."""
    employees = get_employees_col()
    employees.create_index("emp_id", unique=True)

    attendance = get_attendance_col()
    attendance.create_index([("emp_id", ASCENDING), ("timestamp", ASCENDING)])
    attendance.create_index(
        [("emp_id", ASCENDING), ("date", ASCENDING), ("status", ASCENDING)]
    )
    attendance.create_index("date")

    # Same indexes for local attendance collection
    local_att = get_local_attendance_col()
    local_att.create_index([("emp_id", ASCENDING), ("timestamp", ASCENDING)])
    local_att.create_index(
        [("emp_id", ASCENDING), ("date", ASCENDING), ("status", ASCENDING)]
    )
    local_att.create_index("date")


# ═══════════════════════════════════════════════════════════
# Employee CRUD
# ═══════════════════════════════════════════════════════════
def register_employee(
    emp_id: str,
    emp_name: str,
    photo_local_path: str = "",
    photo_gcs_url: str = "",
    face_embedding: list = None,
) -> dict:
    """Insert or update an employee record (upsert on emp_id)."""
    doc = {
        "emp_id": emp_id,
        "emp_name": emp_name,
        "photo_local_path": photo_local_path,
        "photo_gcs_url": photo_gcs_url,
        "face_embedding": face_embedding or [],
        "registered_at": _utc_now(),
    }
    get_employees_col().update_one(
        {"emp_id": emp_id},
        {"$set": doc},
        upsert=True,
    )
    return doc


def get_employee(emp_id: str) -> Optional[dict]:
    return get_employees_col().find_one({"emp_id": emp_id}, {"_id": 0})


def get_all_employees() -> list[dict]:
    return list(get_employees_col().find({}, {"_id": 0}).sort("emp_id", ASCENDING))


def get_employee_dict() -> dict:
    """Return {emp_id: emp_name} mapping for all registered employees."""
    employees = get_all_employees()
    return {e["emp_id"]: e["emp_name"] for e in employees}


def update_employee_embedding(emp_id: str, embedding: list):
    """Replace the stored face embedding for an employee."""
    get_employees_col().update_one(
        {"emp_id": emp_id},
        {"$set": {"face_embedding": embedding}},
    )


def update_employee_photo(emp_id: str, photo_local_path: str, photo_gcs_url: str):
    """Update the photo paths for an employee."""
    get_employees_col().update_one(
        {"emp_id": emp_id},
        {"$set": {
            "photo_local_path": photo_local_path,
            "photo_gcs_url": photo_gcs_url,
        }},
    )


def delete_employee(emp_id: str):
    get_employees_col().delete_one({"emp_id": emp_id})


# ═══════════════════════════════════════════════════════════
# Attendance CRUD
# ═══════════════════════════════════════════════════════════
def record_attendance_in(
    emp_id: str,
    emp_name: str,
    timestamp: datetime = None,
) -> bool:
    """
    Record an "in" event.  Written **once per day per employee**.
    Returns True if a new record was created, False if already exists.
    """
    if timestamp is None:
        timestamp = _utc_now()

    tz = pytz.timezone(TIMEZONE)
    local_dt = (
        timestamp.astimezone(tz)
        if timestamp.tzinfo
        else pytz.utc.localize(timestamp).astimezone(tz)
    )
    date_str = local_dt.strftime("%Y-%m-%d")

    # Already clocked-in today?
    if get_attendance_col().find_one(
        {"emp_id": emp_id, "date": date_str, "status": "in"}
    ):
        return False

    get_attendance_col().insert_one({
        "emp_id": emp_id,
        "emp_name": emp_name,
        "timestamp": timestamp,
        "date": date_str,
        "status": "in",
    })
    return True


def record_attendance_out(
    emp_id: str,
    emp_name: str,
    timestamp: datetime = None,
) -> bool:
    """
    Record / update an "out" event.
    Only writes if an "in" exists for the same calendar day (IST).
    Upserts on {emp_id, date, status: "out"} so the timestamp always
    reflects the last seen going-out time.

    Returns True if written/updated, False if no "in" exists today.
    """
    if timestamp is None:
        timestamp = _utc_now()

    tz = pytz.timezone(TIMEZONE)
    local_dt = (
        timestamp.astimezone(tz)
        if timestamp.tzinfo
        else pytz.utc.localize(timestamp).astimezone(tz)
    )
    date_str = local_dt.strftime("%Y-%m-%d")

    # Must have an "in" first
    if not get_attendance_col().find_one(
        {"emp_id": emp_id, "date": date_str, "status": "in"}
    ):
        return False

    get_attendance_col().update_one(
        {"emp_id": emp_id, "date": date_str, "status": "out"},
        {"$set": {"emp_name": emp_name, "timestamp": timestamp}},
        upsert=True,
    )
    return True


def get_today_attendance() -> list[dict]:
    """Return all attendance records for today (IST), sorted by timestamp."""
    date_str = _local_today()
    return list(
        get_attendance_col()
        .find({"date": date_str}, {"_id": 0})
        .sort("timestamp", ASCENDING)
    )


def get_today_attendance_count() -> int:
    """Return the number of unique "in" events for today."""
    date_str = _local_today()
    return get_attendance_col().count_documents(
        {"date": date_str, "status": "in"}
    )


# ═══════════════════════════════════════════════════════════
# Local Camera Attendance CRUD (attendance_local collection)
# ═══════════════════════════════════════════════════════════
def record_local_attendance_in(
    emp_id: str,
    emp_name: str,
    timestamp: datetime = None,
) -> bool:
    """
    Record an "in" event into the local camera attendance collection.
    Written once per day per employee.
    Returns True if a new record was created, False if already exists.
    """
    if timestamp is None:
        timestamp = _utc_now()

    tz = pytz.timezone(TIMEZONE)
    local_dt = (
        timestamp.astimezone(tz)
        if timestamp.tzinfo
        else pytz.utc.localize(timestamp).astimezone(tz)
    )
    date_str = local_dt.strftime("%Y-%m-%d")

    if get_local_attendance_col().find_one(
        {"emp_id": emp_id, "date": date_str, "status": "in"}
    ):
        return False

    get_local_attendance_col().insert_one({
        "emp_id": emp_id,
        "emp_name": emp_name,
        "timestamp": timestamp,
        "date": date_str,
        "status": "in",
        "source": "local_camera",
    })
    return True


def get_today_local_attendance() -> list[dict]:
    """Return all local camera attendance records for today, sorted by timestamp."""
    date_str = _local_today()
    return list(
        get_local_attendance_col()
        .find({"date": date_str}, {"_id": 0})
        .sort("timestamp", ASCENDING)
    )


def get_today_local_attendance_count() -> int:
    """Return count of unique local camera 'in' events for today."""
    date_str = _local_today()
    return get_local_attendance_col().count_documents(
        {"date": date_str, "status": "in"}
    )
