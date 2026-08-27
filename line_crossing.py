"""
line_crossing.py — Below-the-line attendance detector with per-employee debounce.

If a recognised face's vertical centre is at or below the horizontal
reference line, a candidate "in" event is emitted (subject to debounce).

No direction tracking is needed — the original system's logic is preserved:
below the line = mark attendance.
"""
import time

from config import DEBOUNCE_SECONDS


class AttendanceDetector:
    """
    Usage::

        detector = AttendanceDetector(line_y=300)
        events = detector.update(tracker_ids, face_cy_list, emp_ids)
        # events = [{"tracker_id": int, "emp_id": str, "event": "in"}, ...]
    """

    def __init__(self, line_y: int, debounce_sec: float = DEBOUNCE_SECONDS):
        self.line_y = line_y
        self.debounce_sec = debounce_sec
        # emp_id → monotonic timestamp of last emitted event
        self._last_event: dict[str, float] = {}

    def set_line_y(self, line_y: int):
        self.line_y = line_y

    def update(
        self,
        tracker_ids: list[int],
        face_centers_y: list[int],
        emp_ids: list[str],
    ) -> list[dict]:
        """
        Process one frame's detections.

        Returns a list of ``{"tracker_id", "emp_id", "event": "in"}`` dicts
        for faces whose centre is below the line and that haven't been
        debounced.
        """
        events: list[dict] = []
        now = time.monotonic()

        for tid, cy, emp_id in zip(tracker_ids, face_centers_y, emp_ids):
            if emp_id == "unknown":
                continue

            # Face centre must be at or below the attendance line
            if cy < self.line_y:
                continue

            # Debounce — suppress if we emitted for this emp_id recently
            last = self._last_event.get(emp_id, 0.0)
            if (now - last) < self.debounce_sec:
                continue

            self._last_event[emp_id] = now
            events.append({
                "tracker_id": tid,
                "emp_id": emp_id,
                "event": "in",
            })

        return events

    def reset(self):
        """Clear all debounce state."""
        self._last_event.clear()
