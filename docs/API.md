# API Reference

## WebSocket Endpoints

| Endpoint | Direction | Description |
| :--- | :--- | :--- |
| `/ws/stream/pi/{stream_type}` | Ingress | The Pi connects here to stream JPEG bytes. `stream_type` can be `raw` or `analyzed`. |
| `/ws/stream/ui/{stream_type}` | Egress | The Dashboard connects here to receive the JPEG bytes broadcasted from the Pi. |

## HTTP Endpoints (FastAPI)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/model/latest` | Returns the current timestamp version of `svm_classifier.pkl` from `config.json`. |
| `GET` | `/api/model/download` | Serves the `svm_classifier.pkl` file for the Pi to download. |
