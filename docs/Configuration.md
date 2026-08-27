# Configuration Guide

All configuration is centralized in `config.py`. 

*Note: Environment variables (`os.getenv`) were stripped in favor of strict hardcoding to prevent rogue `.env` files on the Raspberry Pi from overriding critical network paths (a common cause of `Connection refused` errors).*

| Variable | Description | Example Value |
| :--- | :--- | :--- |
| `MONGODB_URI` | Connection string for MongoDB | `mongodb://192.168.1.29:27017` |
| `API_BASE_URL` | FastAPI base address | `http://192.168.1.29:8000` |
| `WS_BASE_URL` | WebSocket base address | `ws://192.168.1.29:8000` |
| `MQTT_BROKER_HOST` | IP address of Mosquitto | `192.168.1.29` |
| `DEBOUNCE_SECONDS` | Time to wait before logging duplicate attendance | `3.0` |
| `MODEL_SYNC_INTERVAL` | Seconds between checking for new SVM models | `300` |

