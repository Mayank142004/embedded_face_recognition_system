"""
pi_runner.py — Standalone entry point for the Raspberry Pi.

Reads live video from a USB camera, runs the full inference pipeline
(YOLO → FaceNet → SVM → line-crossing → MQTT), and streams annotated
frames to the FastAPI server via WebSocket.

Usage:
    python pi_runner.py                     # uses /dev/video0
    python pi_runner.py --camera 1          # uses /dev/video1
    python pi_runner.py --camera 0 --show   # show local OpenCV window (for debug)
"""
import argparse
import logging
import time
import sys
import os
import json
import subprocess

import cv2 as cv

from config import PI_TARGET_FPS, AI_INTERVAL_SEC, TFLITE_NUM_THREADS, MOTION_GATE_ENABLED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pi_runner")


def _log_health():
    """Log SoC temperature and throttle status.

    Undervoltage is common on a Pi 3 with a USB camera sharing the supply,
    and it halves the clock silently — worth seeing in the log before
    drawing any conclusions from the perf numbers.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            logger.info("SoC temperature: %.1f C", int(f.read().strip()) / 1000.0)
    except Exception:
        pass

    try:
        out = subprocess.run(["vcgencmd", "get_throttled"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        # Bit 0 = under-voltage now, bit 16 = under-voltage has occurred
        raw = out.split("=")[-1] if "=" in out else "0x0"
        bits = int(raw, 16)
        notes = []
        if bits & 0x1:      notes.append("UNDER-VOLTAGE NOW")
        if bits & 0x4:      notes.append("ARM FREQUENCY CAPPED NOW")
        if bits & 0x8:      notes.append("THROTTLED NOW")
        if bits & 0x10000:  notes.append("under-voltage has occurred")
        if bits & 0x40000:  notes.append("frequency capping has occurred")
        if bits & 0x80000:  notes.append("throttling has occurred")
        logger.info("Throttle status: %s %s", out or "n/a",
                    ("<-- " + "; ".join(notes)) if notes else "(clean)")
    except FileNotFoundError:
        logger.info("Throttle status: vcgencmd not available")
    except Exception as e:
        logger.info("Throttle status: unavailable (%s)", e)


def main():
    parser = argparse.ArgumentParser(description="Face Attendance — Pi Runner")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--show", action="store_true", help="Show local OpenCV preview window")
    parser.add_argument("--fps", type=int, default=PI_TARGET_FPS,
                        help=f"Target capture/stream FPS (default {PI_TARGET_FPS}, from PI_TARGET_FPS)")
    args = parser.parse_args()

    # ── Print startup banner ───────────────────────────────
    logger.info("=" * 60)
    logger.info("Face Attendance System — Raspberry Pi Edge Node")
    logger.info("=" * 60)
    _log_health()

    # ── Print model version ────────────────────────────────
    from config import MODEL_DIR, MODEL_PATH
    config_json = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_json):
        with open(config_json) as f:
            info = json.load(f)
        logger.info("Model version: %s (synced at %s)", info.get("version"), info.get("updated_at"))
    else:
        logger.info("Model version: default (no config.json — using bundled .pkl)")

    if not os.path.exists(MODEL_PATH):
        logger.warning("Model file NOT FOUND at %s", MODEL_PATH)
        logger.info("Waiting for background sync to download a model from Laptop/GCP...")
        logger.info("Make sure your FastAPI server is running on the laptop and the Pi's .env points to the correct IP.")
        
        # Start background sync explicitly here to force an immediate check
        from model_sync import sync_model_once
        
        while not os.path.exists(MODEL_PATH):
            logger.info("Attempting to download model...")
            success = sync_model_once()
            if success and os.path.exists(MODEL_PATH):
                logger.info("✅ Model downloaded successfully!")
                break
            
            logger.info("Still waiting... (will retry in 10 seconds). Press Ctrl+C to cancel.")
            time.sleep(10)
    
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    logger.info("Model file: %s (%.1f KB)", MODEL_PATH, size_kb)

    # ── Import main callback (triggers YOLO + FaceNet + model_sync) ──
    logger.info("Loading models (YOLO + FaceNet + SVM)... this may take 30-60 seconds on Pi 3")
    from main import callback
    logger.info("Models loaded successfully.")

    # ── Open camera ────────────────────────────────────────
    logger.info("Opening camera /dev/video%d ...", args.camera)
    cap = cv.VideoCapture(args.camera, cv.CAP_V4L2)
    if not cap.isOpened():
        logger.error("Cannot open camera %d. Check USB connection.", args.camera)
        sys.exit(1)

    # L2 FIX: Minimize camera buffer and force MJPG to remove standing delay
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera opened: %dx%d", w, h)
    logger.info("Target capture FPS: %d", args.fps)
    logger.info("AI interval: %.2fs (%.1f passes/sec max)", AI_INTERVAL_SEC, 1.0 / AI_INTERVAL_SEC)
    logger.info("TFLite threads per interpreter: %d", TFLITE_NUM_THREADS)
    logger.info("Motion gate: %s", "ON" if MOTION_GATE_ENABLED else "OFF")
    logger.info("Local preview: %s", "ON" if args.show else "OFF")
    logger.info("-" * 60)
    logger.info("System running. Press Ctrl+C to stop.")
    logger.info("-" * 60)

    # ── Main loop ──────────────────────────────────────────
    frame_interval = 1.0 / args.fps
    idx = 0
    fps_counter = 0
    fps_timer = time.time()

    try:
        while True:
            loop_start = time.time()

            # grab() pulls the frame off the device without JPEG-decoding it;
            # only the frame we actually keep pays for retrieve().
            if not cap.grab():
                logger.warning("Failed to grab frame. Retrying...")
                time.sleep(0.5)
                continue
            ret, frame = cap.retrieve()
            if not ret:
                logger.warning("Failed to decode frame. Retrying...")
                time.sleep(0.5)
                continue

            # Run inference + MQTT + WebSocket streaming
            annotated = callback(frame, idx)
            idx += 1
            fps_counter += 1

            # Print FPS every 5 seconds
            elapsed = time.time() - fps_timer
            if elapsed >= 5.0:
                actual_fps = fps_counter / elapsed
                logger.info("Frames: %d | FPS: %.1f", idx, actual_fps)
                fps_counter = 0
                fps_timer = time.time()

            # Optional local preview
            if args.show:
                preview = cv.resize(annotated, (640, int(640 * h / w)))
                cv.imshow("Pi Preview", preview)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            # Throttle to target FPS
            processing_time = time.time() - loop_start
            sleep_time = frame_interval - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        cap.release()
        if args.show:
            cv.destroyAllWindows()
        logger.info("Total frames processed: %d", idx)
        logger.info("Goodbye.")


if __name__ == "__main__":
    main()
