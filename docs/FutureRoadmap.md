# Future Roadmap

## Short-Term Improvements
1.  **Containerization:** Dockerize the FastAPI, Streamlit, and MQTT Subscriber services using `docker-compose` to eliminate Python environment setup on the laptop.
2.  **Authentication:** Add JWT authentication to the FastAPI endpoints to secure the WebSocket streams.

## Long-Term Improvements
1.  **Cloud Migration:** Move MongoDB and the Mosquitto broker to AWS/GCP to allow multiple Raspberry Pis (doors) to sync to a single global database.
2.  **Optical Flow Tracking:** Replace the "frozen bounding box" logic during skipped frames with a lightweight OpenCV tracker (e.g., KCF or MOSSE) to interpolate box movement smoothly between the 5th-frame AI updates.
3.  **Anti-Spoofing:** Integrate a lightweight Liveness Detection model (e.g., MobileNetV2 checking for screen reflections or blinking) to prevent attendance fraud via printed photos or phone screens.
