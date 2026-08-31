"""
mqtt_publisher.py — Publish attendance crossing events to the MQTT broker.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import paho.mqtt.client as mqtt

from config import MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_TOPIC, MQTT_CLIENT_ID_PUB

logger = logging.getLogger(__name__)

_client: Optional[mqtt.Client] = None


def _get_client() -> mqtt.Client:
    global _client
    if _client is None or not _client.is_connected():
        # C5 FIX: Close old client before creating new one to prevent thread leaks
        if _client is not None:
            try:
                _client.loop_stop()
                _client.disconnect()
            except Exception:
                pass
        _client = mqtt.Client(client_id=MQTT_CLIENT_ID_PUB)
        _client.reconnect_delay_set(min_delay=1, max_delay=30)
        try:
            _client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
            _client.loop_start()
            logger.info(
                "MQTT publisher connected to %s:%s",
                MQTT_BROKER_HOST, MQTT_BROKER_PORT,
            )
        except Exception as e:
            logger.error("Failed to connect to MQTT broker: %s", e)
            raise
    return _client


def publish_event(
    emp_id: str,
    event_type: str,
    confidence: float = 0.0,
) -> bool:
    """
    Publish an attendance event.

    Payload::

        {
            "emp_id":      "E001",
            "status":      "in" | "out",
            "timestamp":   "2026-08-21T12:30:00+00:00",
            "confidence":  0.9512
        }

    Returns True on success, False otherwise.
    """
    payload = {
        "emp_id": emp_id,
        "status": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "confidence": round(confidence, 4),
    }
    try:
        client = _get_client()
        result = client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info("Published: %s", payload)
            return True
        logger.warning("Publish failed (rc=%s): %s", result.rc, payload)
        return False
    except Exception as e:
        logger.error("MQTT publish error: %s", e)
        return False


def disconnect():
    """Cleanly disconnect the MQTT client."""
    global _client
    if _client is not None:
        _client.loop_stop()
        _client.disconnect()
        _client = None
