# Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Pi Camera FPS drops to 0.2** | Pi is timing out trying to connect to a dead DB/MQTT IP. | Ensure `192.168.1.x` IPs in `config.py` point to the correct laptop IP. Check laptop firewall (`sudo ufw status`). |
| **`Connection refused` on MQTT** | Mosquitto is only listening on `localhost`. | Add `listener 1883 0.0.0.0` to `/etc/mosquitto/conf.d/remote.conf` and restart Mosquitto. |
| **`BytesRequired overflow` on Pi** | The TFLite model has a dynamic batch size on a 32-bit OS. | Use the fixed-batch `facenet.tflite` model provided in the repo. |
| **Boxes disappear on Analyzed Feed** | ByteTrack dropping tracks due to frame skipping. | Ensure `main.py` initializes ByteTrack with `minimum_consecutive_frames=1`. |
| **Boxes are drawn at `[0, 0, 0, 0]`** | TFLite output coordinates are 0.0-1.0 normalized but math assumes they are pixel-scale. | Ensure `yolo_tflite.py` multiplies `cx` by `orig_w`, not `(orig_w / 320)`. |
| **No attendance in Dashboard** | `mqtt_subscriber.py` is not running. | Run `python mqtt_subscriber.py` in a background terminal on the laptop. |

