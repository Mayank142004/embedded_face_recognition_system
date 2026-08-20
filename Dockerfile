FROM python:3.11-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirement_clean.txt .

RUN python -m pip install --upgrade pip
RUN pip install -r requirement_clean.txt

COPY . .

CMD ["python", "main.py"]
