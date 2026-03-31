import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


def _normalize_s3_prefix(value: str, default: str) -> str:
    normalized = (value or default).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    if normalized and not normalized.endswith("/"):
        normalized += "/"
    return normalized or default


@dataclass
class Config:
    RABBITMQ_URL: str
    QUEUE_NAME: str
    DONE_QUEUE_NAME: str
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
    AUDIO_PREFIX: str
    SUBTITLE_PREFIX: str
    TEMP_DIR: str

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        use_gpu = os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes")
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            cuda_visible_devices = cuda_visible_devices.strip() or None
            if cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return cls(
            RABBITMQ_URL=os.getenv(
                "RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"
            ),
            QUEUE_NAME=os.getenv("QUEUE_NAME", "transnet_tasks"),
            DONE_QUEUE_NAME=os.getenv("DONE_QUEUE_NAME", "transnet_tasks_done"),
            S3_ENDPOINT_URL=os.getenv("S3_ENDPOINT_URL", ""),
            S3_ACCESS_KEY=os.getenv("S3_ACCESS_KEY", ""),
            S3_SECRET_KEY=os.getenv("S3_SECRET_KEY", ""),
            S3_BUCKET=os.getenv("S3_BUCKET", ""),
            S3_REGION=os.getenv("S3_REGION", "us-east-1"),
            USE_GPU=use_gpu,
            CUDA_VISIBLE_DEVICES=cuda_visible_devices,
            WEIGHTS_PATH=os.getenv(
                "WEIGHTS_PATH", "/app/weights/transnetv2-pytorch-weights.pth"
            ),
            RESULT_PREFIX=_normalize_s3_prefix(
                os.getenv("RESULT_PREFIX", "results/"), "results/"
            ),
            FRAME_IMAGE_PREFIX=_normalize_s3_prefix(
                os.getenv("FRAME_IMAGE_PREFIX", "frames/"), "frames/"
            ),
            AUDIO_PREFIX=_normalize_s3_prefix(
                os.getenv("AUDIO_PREFIX", "audio/"), "audio/"
            ),
            SUBTITLE_PREFIX=_normalize_s3_prefix(
                os.getenv("SUBTITLE_PREFIX", "subtitles/"), "subtitles/"
            ),
            TEMP_DIR=os.getenv("TEMP_DIR", "./.tmp"),
        )

    def get_device(self) -> str:
        import torch

        if self.USE_GPU:
            cuda_available = torch.cuda.is_available()
            cuda_device_count = torch.cuda.device_count() if cuda_available else 0
            if cuda_available and cuda_device_count > 0:
                return "cuda"

            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend and mps_backend.is_available():
                return "mps"

            cuda_version = torch.version.cuda or "none"
            cuda_visible_devices = self.CUDA_VISIBLE_DEVICES
            raise RuntimeError(
                "USE_GPU=true but no GPU backend is available. "
                f"torch={torch.__version__}, torch_cuda={cuda_version}, "
                f"cuda_available={cuda_available}, "
                f"cuda_device_count={cuda_device_count}, "
                f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}. "
                "Install a CUDA-enabled PyTorch build or disable USE_GPU."
            )
        return "cpu"
