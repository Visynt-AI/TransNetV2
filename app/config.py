import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    RABBITMQ_URL: str
    QUEUE_NAME: str
    S3_ENDPOINT_URL: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str
    S3_BUCKET: str
    S3_REGION: str
    USE_GPU: bool
    CUDA_VISIBLE_DEVICES: Optional[str]
    WEIGHTS_PATH: str
    RESULT_PREFIX: str
    FRAME_IMAGE_PREFIX: str

    @classmethod
    def from_env(cls) -> "Config":
        use_gpu = os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes")
        return cls(
            RABBITMQ_URL=os.getenv(
                "RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"
            ),
            QUEUE_NAME=os.getenv("QUEUE_NAME", "transnet_tasks"),
            S3_ENDPOINT_URL=os.getenv("S3_ENDPOINT_URL", ""),
            S3_ACCESS_KEY=os.getenv("S3_ACCESS_KEY", ""),
            S3_SECRET_KEY=os.getenv("S3_SECRET_KEY", ""),
            S3_BUCKET=os.getenv("S3_BUCKET", ""),
            S3_REGION=os.getenv("S3_REGION", "us-east-1"),
            USE_GPU=use_gpu,
            CUDA_VISIBLE_DEVICES=os.getenv("CUDA_VISIBLE_DEVICES"),
            WEIGHTS_PATH=os.getenv(
                "WEIGHTS_PATH", "/app/weights/transnetv2-pytorch-weights.pth"
            ),
            RESULT_PREFIX=os.getenv("RESULT_PREFIX", "results/"),
            FRAME_IMAGE_PREFIX=os.getenv("FRAME_IMAGE_PREFIX", "frames/"),
        )

    def get_device(self) -> str:
        import torch

        if self.USE_GPU:
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        return "cpu"
