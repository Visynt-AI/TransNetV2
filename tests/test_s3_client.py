import os

import pytest
from unittest.mock import MagicMock, patch, call
from botocore.exceptions import ClientError

from app.config import Config
from app.s3_client import S3Client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> Config:
    return Config(
        RABBITMQ_URL="amqp://localhost/",
        QUEUE_NAME="q",
        DONE_QUEUE_NAME="dq",
        S3_ENDPOINT_URL="http://minio:9000",
        S3_ACCESS_KEY="key",
        S3_SECRET_KEY="secret",
        S3_BUCKET="mybucket",
        S3_REGION="us-east-1",
        USE_GPU=False,
        CUDA_VISIBLE_DEVICES=None,
        WEIGHTS_PATH="w",
        RESULT_PREFIX="results/",
        FRAME_IMAGE_PREFIX="frames/",
        AUDIO_PREFIX="audio/",
        SUBTITLE_PREFIX="subtitles/",
        TEMP_DIR="/tmp",
    )


@pytest.fixture
def s3(config) -> S3Client:
    with patch("app.s3_client.boto3.client") as mock_client:
        client = S3Client(config)
        client.client = MagicMock()
    return client


# ---------------------------------------------------------------------------
# download_video
# ---------------------------------------------------------------------------

class TestDownloadVideo:
    def test_returns_local_path_with_correct_extension(self, tmp_path, config):
        config = Config(**{**config.__dict__, "TEMP_DIR": str(tmp_path)})
        with patch("app.s3_client.boto3.client"):
            s3 = S3Client(config)
            s3.client = MagicMock()

        path = s3.download_video("videos/movie.mp4")

        assert path.endswith(".mp4")
        assert os.path.dirname(path) == str(tmp_path)

    def test_defaults_to_mp4_when_no_extension(self, tmp_path, config):
        config = Config(**{**config.__dict__, "TEMP_DIR": str(tmp_path)})
        with patch("app.s3_client.boto3.client"):
            s3 = S3Client(config)
            s3.client = MagicMock()

        path = s3.download_video("videos/movie_no_ext")
        assert path.endswith(".mp4")

    def test_s3_error_raises_runtime_error(self, s3):
        error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
        s3.client.download_file.side_effect = ClientError(error_response, "GetObject")

        with pytest.raises(RuntimeError, match="Failed to download"):
            s3.download_video("missing/key.mp4")

    def test_temp_file_deleted_on_download_failure(self, tmp_path, config):
        config = Config(**{**config.__dict__, "TEMP_DIR": str(tmp_path)})
        with patch("app.s3_client.boto3.client"):
            s3 = S3Client(config)
            s3.client = MagicMock()

        error = ClientError({"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject")
        s3.client.download_file.side_effect = error

        with pytest.raises(RuntimeError):
            s3.download_video("missing.mp4")

        # Temp file must not linger
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# upload_file
# ---------------------------------------------------------------------------

class TestUploadFile:
    def test_returns_s3_uri(self, s3):
        result = s3.upload_file("/tmp/local.mp4", "videos/remote.mp4")
        assert result == "s3://mybucket/videos/remote.mp4"

    def test_calls_boto3_upload(self, s3):
        s3.upload_file("/tmp/local.mp4", "k/remote.mp4")
        s3.client.upload_file.assert_called_once_with(
            "/tmp/local.mp4", "mybucket", "k/remote.mp4"
        )

    def test_client_error_raises_runtime_error(self, s3):
        error = ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "PutObject")
        s3.client.upload_file.side_effect = error

        with pytest.raises(RuntimeError, match="Failed to upload file"):
            s3.upload_file("/tmp/f.mp4", "key")


# ---------------------------------------------------------------------------
# upload_bytes
# ---------------------------------------------------------------------------

class TestUploadBytes:
    def test_returns_s3_uri(self, s3):
        result = s3.upload_bytes(b"data", "results/r.json")
        assert result == "s3://mybucket/results/r.json"

    def test_content_type_forwarded(self, s3):
        s3.upload_bytes(b"{}", "r.json", content_type="application/json")
        _, kwargs = s3.client.put_object.call_args
        assert kwargs.get("ContentType") == "application/json"

    def test_no_content_type_omits_field(self, s3):
        s3.upload_bytes(b"data", "r.bin")
        _, kwargs = s3.client.put_object.call_args
        assert "ContentType" not in kwargs

    def test_client_error_raises_runtime_error(self, s3):
        error = ClientError({"Error": {"Code": "500", "Message": "err"}}, "PutObject")
        s3.client.put_object.side_effect = error

        with pytest.raises(RuntimeError, match="Failed to upload bytes"):
            s3.upload_bytes(b"x", "key")
