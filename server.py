"""
server.py — FastAPI Backend for Face Attendance System

Handles:
1. WebSocket live video streaming from Pi -> UI.
2. Model versioning and upload endpoints for Pi auto-updates.
"""
import asyncio
import os
import time
from typing import Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse

from db import get_db
from gcs_storage import upload_model

app = FastAPI(title="Face Attendance API")

# ── WebSocket Streaming ────────────────────────────────────
from collections import defaultdict
ui_clients = defaultdict(set)

@app.websocket("/ws/stream/pi/{stream_type}")
async def websocket_pi_stream(websocket: WebSocket, stream_type: str):
    """Endpoint for Raspberry Pi to stream JPEG frames (raw or analyzed)."""
    await websocket.accept()
    print(f"Pi connected to {stream_type} stream.")
    try:
        while True:
            data = await websocket.receive_bytes()
            # L3 FIX: Fan-out concurrently so one slow viewer doesn't stall others
            clients = list(ui_clients[stream_type])
            if clients:
                results = await asyncio.gather(
                    *(c.send_bytes(data) for c in clients),
                    return_exceptions=True
                )
                # Remove clients that errored
                for client, result in zip(clients, results):
                    if isinstance(result, Exception):
                        ui_clients[stream_type].discard(client)
    except WebSocketDisconnect:
        print(f"Pi disconnected from {stream_type} stream.")

@app.websocket("/ws/stream/ui/{stream_type}")
async def websocket_ui_stream(websocket: WebSocket, stream_type: str):
    """Endpoint for Streamlit dashboard to receive the live stream."""
    await websocket.accept()
    ui_clients[stream_type].add(websocket)
    print(f"UI client connected to {stream_type} stream.")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ui_clients[stream_type].remove(websocket)
        print(f"UI client disconnected from {stream_type}.")


# ── Model Versioning ───────────────────────────────────────
@app.get("/api/model/latest")
def get_latest_model():
    """Returns the latest model version for the Pi to check."""
    db = get_db()
    doc = db["system_config"].find_one({"_id": "model_info"})
    if not doc:
        return {"version": "default_v1", "updated_at": None, "url": None}
    return {
        "version": doc.get("version"),
        "updated_at": doc.get("updated_at"),
        "url": doc.get("url")
    }


@app.post("/api/model/upload")
async def upload_model_file(file: UploadFile = File(...)):
    """Upload a new .pkl model from the Streamlit UI."""
    os.makedirs("models", exist_ok=True)
    local_model_path = os.path.join("models", file.filename)
    
    with open(local_model_path, "wb") as f:
        f.write(await file.read())
    
    # Generate version string
    version = f"v_{int(time.time())}"
    
    # URL is now a direct API download link
    url = f"/api/model/download/{file.filename}"
    
    # Update latest version in MongoDB
    db = get_db()
    db["system_config"].update_one(
        {"_id": "model_info"},
        {
            "$set": {
                "version": version,
                "updated_at": time.time(),
                "url": url
            }
        },
        upsert=True
    )
    
    return {"status": "success", "version": version, "url": url}

from fastapi.responses import FileResponse

@app.get("/api/model/download/{filename}")
def download_model(filename: str):
    """Serve the model file directly."""
    file_path = os.path.join("models", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    return JSONResponse({"error": "File not found"}, status_code=404)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
