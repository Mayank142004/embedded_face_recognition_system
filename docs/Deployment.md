# Deployment Guide

## 1. Laptop/Server Setup
1.  **System Dependencies:** Install MongoDB and Mosquitto.
    ```bash
    sudo apt install mongodb mosquitto
    sudo systemctl enable --now mongod
    sudo systemctl enable --now mosquitto
    ```
2.  **Mosquitto Config:** Edit `/etc/mosquitto/conf.d/remote.conf` to allow external connections:
    ```
    listener 1883 0.0.0.0
    allow_anonymous true
    ```
    *Restart Mosquitto: `sudo systemctl restart mosquitto`*
3.  **Python Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
4.  **Run Services (Separate Terminals):**
    *   `uvicorn server:app --host 0.0.0.0 --port 8000`
    *   `python mqtt_subscriber.py`
    *   `streamlit run dashboard.py`

## 2. Raspberry Pi Setup
1.  **Sync Codebase:** Do not sync heavy models or virtual environments.
    ```bash
    rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='recordings' ~/Desktop/FaceRecognitionSystem/ pi@<PI_IP>:~/FaceAttend/
    ```
2.  **Python Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements_pi.txt
    ```
3.  **Run Pipeline:**
    ```bash
    python pi_runner.py
    ```
