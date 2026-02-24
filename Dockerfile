FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference_pytorch/ ./inference_pytorch/
COPY app/ ./app/
COPY main.py .
COPY weights/ ./weights/

ENV RABBITMQ_URL=amqp://guest:guest@localhost:5672/
ENV QUEUE_NAME=transnet_tasks
ENV S3_ENDPOINT_URL=
ENV S3_BUCKET=
ENV S3_REGION=us-east-1
ENV USE_GPU=false
ENV WEIGHTS_PATH=/app/weights/transnetv2-pytorch-weights.pth
ENV RESULT_PREFIX=results/
ENV FRAME_IMAGE_PREFIX=frames/

ENTRYPOINT ["python", "main.py"]
