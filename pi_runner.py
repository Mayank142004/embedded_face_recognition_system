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

import cv2 as cv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pi_runner")


def main():
    parser = argparse.ArgumentParser(description="Face Attendance — Pi Runner")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--show", action="store_true", help="Show local OpenCV preview window")
    parser.add_argument("--fps", type=int, default=15, help="Target processing FPS (default 15)")
    args = parser.parse_args()

    # ── Print startup banner ───────────────────────────────
    logger.info("=" * 60)
    logger.info("Face Attendance System — Raspberry Pi Edge Node")
    logger.info("=" * 60)

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
    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Cannot open camera %d. Check USB connection.", args.camera)
        sys.exit(1)

    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera opened: %dx%d", w, h)
    logger.info("Target FPS: %d", args.fps)
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

            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame. Retrying...")
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
