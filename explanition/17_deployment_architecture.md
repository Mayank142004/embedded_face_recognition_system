# 17. Deployment Architecture

The system is designed to run locally or inside a Docker container.

## Architecture Flow
```text
Developer / Host OS
 ↓
`docker build`
 ↓
Docker Container (`python:3.11-bookworm`)
 ↓
System Dependencies Installation (libgl1, etc.)
 ↓
Python pip Dependencies Installation
 ↓
Application Files Copied
 ↓
Execution (`CMD ["python", "main.py"]`)
```

## Dockerfile Analysis

**File**: `Dockerfile`
```dockerfile
FROM python:3.11-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
```
- **Base Image**: Uses Python 3.11 on Debian Bookworm.
- **Environment**: Disables pycache writing and buffers to ensure logs print immediately.

```dockerfile
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```
- **System Dependencies**: Installs `libgl1` and `libglib2.0-0`, which are strictly required for OpenCV (`cv2`) to run headless inside a Linux container.

```dockerfile
COPY requirement_clean.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirement_clean.txt
COPY . .
CMD ["python", "main.py"]
```
- **Dependencies**: Installs `requirement_clean.txt`.
- **Execution**: Runs `main.py` by default.

## Deployment Method
To deploy:
```bash
docker build -t face-recognition .
docker run -v $(pwd)/marked_attendance:/app/marked_attendance face-recognition
```
*(Volumes are needed to persist the attendance CSVs outside the container).*
