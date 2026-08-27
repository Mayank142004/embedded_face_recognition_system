#!/usr/bin/env python3
"""
mqtt_subscriber.py — Subscribe to MQTT attendance events and write to MongoDB.

Implements the business rules:
  • "in"  → once per day per employee (first arrival preserved)
  • "out" → only if an "in" exists today; upsert (always latest timestamp)

Run standalone:  python mqtt_subscriber.py
"""
import json
import logging
import signal
import sys
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from config import (
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_TOPIC,
    MQTT_CLIENT_ID_SUB,
)
from db import (
    ensure_indexes,
    record_attendance_in,
    record_attendance_out,
    get_employee_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mqtt_subscriber")


# ── MQTT callbacks ────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected — subscribing to '%s'", MQTT_TOPIC)
        client.subscribe(MQTT_TOPIC, qos=1)
    else:
        logger.error("Connection failed (rc=%s)", rc)


def on_message(client, userdata, msg):
    """Process one MQTT message and apply business rules before writing."""
    try:
        payload = json.loads(msg.payload.decode())
        emp_id = payload.get("emp_id", "")
        status = payload.get("status", "")
        ts_str = payload.get("timestamp", "")
        confidence = payload.get("confidence", 0.0)

        if not emp_id or status not in ("in", "out"):
            logger.warning("Invalid payload — skipping: %s", payload)
            return

        timestamp = (
            datetime.fromisoformat(ts_str)
            if ts_str
            else datetime.now(timezone.utc)
        )

        # Resolve emp_name from MongoDB
        emp_dict = get_employee_dict()
        emp_name = emp_dict.get(emp_id, emp_id)

        if status == "in":
            written = record_attendance_in(emp_id, emp_name, timestamp)
            if written:
                logger.info(
                    "✅ IN  recorded: %s (%s) at %s  conf=%.2f",
                    emp_id, emp_name, timestamp.isoformat(), confidence,
                )
            else:
                logger.debug("⏭  IN  skipped (already exists today): %s", emp_id)

        elif status == "out":
            written = record_attendance_out(emp_id, emp_name, timestamp)
            if written:
                logger.info(
                    "✅ OUT recorded/updated: %s (%s) at %s  conf=%.2f",
                    emp_id, emp_name, timestamp.isoformat(), confidence,
                )
            else:
                logger.debug(
                    "⏭  OUT discarded (no IN today): %s", emp_id
                )

    except Exception as e:
        logger.error(
            "Error processing message: %s — payload: %s", e, msg.payload
        )


# ── Main loop ─────────────────────────────────────────────
def main():
    ensure_indexes()
    logger.info("MongoDB indexes verified.")

    client = mqtt.Client(client_id=MQTT_CLIENT_ID_SUB)
    client.on_connect = on_connect
    client.on_message = on_message

    logger.info("Connecting to MQTT at %s:%s …", MQTT_BROKER_HOST, MQTT_BROKER_PORT)
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)

    def _shutdown(sig, frame):
        logger.info("Shutting down …")
        client.loop_stop()
        client.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Subscriber running. Waiting for messages …")
    client.loop_forever()


if __name__ == "__main__":
    main()
