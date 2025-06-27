FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_production.txt .
RUN pip3 install --no-cache-dir -r requirements_production.txt

COPY config/ ./config/
COPY core/ ./core/
COPY api/ ./api/
COPY models/ ./models/
COPY hand_detector_v2.py .
COPY gesture_classifier_v2.py .
COPY translation_engine.py .
COPY utils_v2.py .

RUN mkdir -p /app/logs /app/temp /tmp/prometheus

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080"]