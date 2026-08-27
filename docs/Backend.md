# Backend Architecture

The server architecture is decoupled into distinct micro-services to prevent blocking operations.

## Components
1.  **FastAPI (`server.py`)**: Handles WebSocket routing. Uses `collections.defaultdict(set)` to broadcast incoming binary frames from the Pi to any number of connected Streamlit UI clients.
2.  **MQTT Subscriber (`mqtt_subscriber.py`)**: A standalone daemon that listens to `attendance/events`. Upon receiving a payload, it executes MongoDB insertions. This ensures that even if the FastAPI server or Dashboard crashes, attendance is still recorded.
3.  **Database (`db.py`)**: A wrapper around `pymongo`. Connects with `serverSelectionTimeoutMS=3000` to fail fast and prevent thread lockups.

## Logging & Error Handling
*   Logging is handled via Python's built-in `logging` module. 
*   WebSocket disconnects are caught cleanly via `WebSocketDisconnect`.
*   MQTT reconnections are handled automatically by `paho-mqtt`'s `loop_forever()`.
