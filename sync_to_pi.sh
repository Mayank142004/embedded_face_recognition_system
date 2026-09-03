#!/usr/bin/env bash
#
# sync_to_pi.sh — push the Pi-side code (and optionally the models) to the
# Raspberry Pi over SSH.
#
# Only the files pi_runner.py actually needs are sent. The repo's .git is
# ~247MB and most of it (recordings, datasets, backups, the dashboard, the
# training pipeline) never runs on the Pi.
#
# Usage:
#   ./sync_to_pi.sh                 # code only — fast, a few KB
#   ./sync_to_pi.sh --models        # code + model weights (~53MB, first run)
#   ./sync_to_pi.sh --dry-run       # show what would transfer, change nothing
#
# Configure once (or export in your shell):
#   PI_HOST=pi@192.168.1.42 ./sync_to_pi.sh
#
set -euo pipefail

PI_HOST="${PI_HOST:-pi@raspberrypi.local}"
PI_DIR="${PI_DIR:-FaceRecognitionSystem}"

SEND_MODELS=0
DRY=""
for arg in "$@"; do
  case "$arg" in
    --models)  SEND_MODELS=1 ;;
    --dry-run) DRY="--dry-run" ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")"

# Files the Pi actually imports. Keep in sync with Dockerfile.pi.
CODE=(
  config.py
  db.py
  gcs_storage.py
  model_sync.py
  line_crossing.py
  mqtt_publisher.py
  motion_gate.py
  yolo_tflite.py
  main.py
  pi_runner.py
  requirements_pi.txt
)

echo "==> target: $PI_HOST:$PI_DIR"
ssh "$PI_HOST" "mkdir -p '$PI_DIR/facenet_files' '$PI_DIR/facenet_models' '$PI_DIR/yolo_models/yolov8n-face_saved_model'"

echo "==> code"
rsync -az --info=name1 $DRY "${CODE[@]}" "$PI_HOST:$PI_DIR/"

echo "==> facenet_files/"
rsync -az --info=name1 $DRY \
  facenet_files/__init__.py \
  facenet_files/facent_svm_rec_passing.py \
  "$PI_HOST:$PI_DIR/facenet_files/"

if [[ $SEND_MODELS -eq 1 ]]; then
  # The vendored supervision package shadows any pip-installed copy, so the
  # Pi must run the same one this was tested against. It rarely changes, so
  # it rides along with the models rather than on every sync.
  echo "==> supervision/ (vendored)"
  rsync -az --info=name1 $DRY \
    --exclude '__pycache__' --exclude '*.pyc' \
    supervision/ "$PI_HOST:$PI_DIR/supervision/"

  # config.py points at the FP16 YOLO. Without this file the Pi will not start.
  echo "==> models (~53MB)"
  rsync -az --info=progress2 $DRY \
    facenet_models/facenet.tflite \
    facenet_models/svm_classifier.pkl \
    "$PI_HOST:$PI_DIR/facenet_models/"
  rsync -az --info=progress2 $DRY \
    yolo_models/yolov8n-face_saved_model/yolov8n-face_float16.tflite \
    yolo_models/yolov8n-face_saved_model/yolov8n-face_float32.tflite \
    "$PI_HOST:$PI_DIR/yolo_models/yolov8n-face_saved_model/"
else
  echo "==> models + supervision skipped (pass --models on first sync)"
fi

# .env is deliberately NOT synced: the Pi's copy points at different hosts
# than the laptop's. Edit it on the Pi directly.
echo
echo "Done. Note: .env was not synced (Pi keeps its own)."
echo "New tunables now available — see .env.example:"
echo "  AI_INTERVAL_SEC, MOTION_GATE_ENABLED, TFLITE_NUM_THREADS, PI_TARGET_FPS"
echo
echo "Run on the Pi:"
echo "  ssh $PI_HOST 'cd $PI_DIR && python pi_runner.py --camera 0'"
