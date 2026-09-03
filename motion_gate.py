"""
motion_gate.py — Cheap frame-difference gate in front of the AI pipeline.

An entrance camera looks at an empty corridor for most of the day. Running
YOLO on those frames is where nearly all of the Pi's idle heat comes from,
so this gate answers one question per frame — "did anything move?" — for
about a millisecond, and lets the caller skip the ~300ms AI pass entirely.

It costs no accuracy: the gate opens as soon as motion appears, which is
before the pipeline would have had anything to detect anyway.
"""
import time

import cv2 as cv

from config import (
    MOTION_PIXEL_DELTA,
    MOTION_MIN_FRACTION,
    MOTION_KEEPALIVE_SEC,
)

# Downscaled size used for differencing. Small enough to be nearly free,
# large enough that a person in a doorway covers plenty of pixels.
_GATE_SIZE = (80, 60)


class MotionGate:
    """
    Usage::

        gate = MotionGate()
        if gate.update(frame):
            run_expensive_ai(frame)
    """

    def __init__(
        self,
        pixel_delta: int = MOTION_PIXEL_DELTA,
        min_fraction: float = MOTION_MIN_FRACTION,
        keepalive_sec: float = MOTION_KEEPALIVE_SEC,
        size: tuple = _GATE_SIZE,
    ):
        self.pixel_delta = pixel_delta
        self.min_fraction = min_fraction
        self.keepalive_sec = keepalive_sec
        self.size = size

        self._prev = None
        self._last_motion = 0.0

        # Stats for the perf log
        self.opens = 0
        self.closes = 0
        self.last_fraction = 0.0

    def update(self, frame) -> bool:
        """Return True if the AI pipeline should run for this frame."""
        # INTER_AREA averages instead of point-sampling, which suppresses
        # sensor noise that would otherwise trip the gate on an empty frame.
        small = cv.resize(frame, self.size, interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)

        now = time.monotonic()

        if self._prev is None:
            self._prev = gray
            self._last_motion = now
            self.opens += 1
            return True

        diff = cv.absdiff(gray, self._prev)
        self._prev = gray

        changed = int((diff > self.pixel_delta).sum())
        fraction = changed / diff.size
        self.last_fraction = fraction

        if fraction >= self.min_fraction:
            self._last_motion = now
            self.opens += 1
            return True

        # Keep running briefly after motion stops, so someone who pauses at
        # the line mid-approach is not dropped.
        if (now - self._last_motion) < self.keepalive_sec:
            self.opens += 1
            return True

        self.closes += 1
        return False

    def is_idle(self) -> bool:
        """True once the keep-alive window has fully expired."""
        return (time.monotonic() - self._last_motion) >= self.keepalive_sec

    def stats(self) -> dict:
        total = self.opens + self.closes
        return {
            "opens": self.opens,
            "closes": self.closes,
            "open_pct": (100.0 * self.opens / total) if total else 0.0,
            "last_fraction": self.last_fraction,
        }

    def reset_stats(self):
        self.opens = 0
        self.closes = 0
