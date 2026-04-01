import logging
import os
import tempfile
from typing import Optional

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import Config

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(self, config: Config):
        self.config = config
        self.bucket = config.S3_BUCKET

        client_kwargs = {
            "aws_access_key_id": config.S3_ACCESS_KEY,
            "aws_secret_access_key": config.S3_SECRET_KEY,
            "region_name": config.S3_REGION,
            "config": BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        }

        if config.S3_ENDPOINT_URL:
            client_kwargs["endpoint_url"] = config.S3_ENDPOINT_URL

        self.client = boto3.client("s3", **client_kwargs)

    def download_video(self, s3_key: str) -> str:
        _, ext = os.path.splitext(s3_key)
        if not ext:
            ext = ".mp4"

        os.makedirs(self.config.TEMP_DIR, exist_ok=True)
        fd, local_path = tempfile.mkstemp(suffix=ext, dir=self.config.TEMP_DIR)
        os.close(fd)

        try:
            logger.info(f"Downloading video from s3://{self.bucket}/{s3_key}")
            self.client.download_file(self.bucket, s3_key, local_path)
            logger.info(f"Video downloaded to {local_path}")
            return local_path
        except ClientError as e:
            os.unlink(local_path)
            raise RuntimeError(f"Failed to download video from S3: {e}")

    def upload_file(self, local_path: str, s3_key: str) -> str:
        try:
            logger.info(f"Uploading {local_path} to s3://{self.bucket}/{s3_key}")
            self.client.upload_file(local_path, self.bucket, s3_key)
            return f"s3://{self.bucket}/{s3_key}"
        except ClientError as e:
            raise RuntimeError(f"Failed to upload file to S3: {e}")

    def upload_bytes(
        self, data: bytes, s3_key: str, content_type: Optional[str] = None
    ) -> str:
        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            logger.info(f"Uploading bytes to s3://{self.bucket}/{s3_key}")
            self.client.put_object(
                Bucket=self.bucket, Key=s3_key, Body=data, **extra_args
            )
            return f"s3://{self.bucket}/{s3_key}"
        except ClientError as e:
            raise RuntimeError(f"Failed to upload bytes to S3: {e}")

