import os

import pytest

from app.config import Config, _normalize_s3_prefix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> Config:
    defaults = dict(
        RABBITMQ_URL="amqp://localhost/",
        QUEUE_NAME="q",
        DONE_QUEUE_NAME="dq",
        S3_ENDPOINT_URL="",
        S3_ACCESS_KEY="key",
        S3_SECRET_KEY="secret",
        S3_BUCKET="bucket",
        S3_REGION="us-east-1",
        USE_GPU=False,
        CUDA_VISIBLE_DEVICES=None,
        WEIGHTS_PATH="/tmp/weights.pth",
        RESULT_PREFIX="results/",
        FRAME_IMAGE_PREFIX="frames/",
        AUDIO_PREFIX="audio/",
        SUBTITLE_PREFIX="subtitles/",
        TEMP_DIR="./.tmp",
    )
    defaults.update(overrides)
    return Config(**defaults)


# ---------------------------------------------------------------------------
# _normalize_s3_prefix
# ---------------------------------------------------------------------------

class TestNormalizeS3Prefix:
    def test_adds_trailing_slash(self):
        assert _normalize_s3_prefix("results", "default/") == "results/"

    def test_already_has_trailing_slash(self):
        assert _normalize_s3_prefix("results/", "default/") == "results/"

    def test_strips_leading_slash(self):
        assert _normalize_s3_prefix("/results/", "default/") == "results/"

    def test_removes_dot_slash_prefix(self):
        assert _normalize_s3_prefix("./results/", "default/") == "results/"

    def test_removes_multiple_dot_slash_prefixes(self):
        assert _normalize_s3_prefix("././results/", "default/") == "results/"

    def test_backslash_converted_to_slash(self):
        assert _normalize_s3_prefix("results\\frames", "default/") == "results/frames/"

    def test_empty_string_uses_default(self):
        assert _normalize_s3_prefix("", "default/") == "default/"

    def test_nested_path(self):
        assert _normalize_s3_prefix("a/b/c", "default/") == "a/b/c/"


# ---------------------------------------------------------------------------
# Config.validate
# ---------------------------------------------------------------------------

class TestConfigValidate:
    def test_missing_access_key_raises(self):
        config = _make_config(S3_ACCESS_KEY="")
        with pytest.raises(ValueError, match="S3_ACCESS_KEY"):
            config.validate()

    def test_missing_secret_key_raises(self):
        config = _make_config(S3_SECRET_KEY="")
        with pytest.raises(ValueError, match="S3_SECRET_KEY"):
            config.validate()

    def test_missing_bucket_raises(self):
        config = _make_config(S3_BUCKET="")
        with pytest.raises(ValueError, match="S3_BUCKET"):
            config.validate()

    def test_all_missing_s3_fields_reported_together(self):
        config = _make_config(S3_ACCESS_KEY="", S3_SECRET_KEY="", S3_BUCKET="")
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        msg = str(exc_info.value)
        assert "S3_ACCESS_KEY" in msg
        assert "S3_SECRET_KEY" in msg
        assert "S3_BUCKET" in msg

    def test_weights_file_not_found_raises(self, tmp_path):
        config = _make_config(WEIGHTS_PATH=str(tmp_path / "missing.pth"))
        with pytest.raises(FileNotFoundError, match="missing.pth"):
            config.validate()

    def test_valid_config_does_not_raise(self, tmp_path):
        weights = tmp_path / "weights.pth"
        weights.touch()
        config = _make_config(WEIGHTS_PATH=str(weights))
        config.validate()  # must not raise


# ---------------------------------------------------------------------------
# Config.get_device
# ---------------------------------------------------------------------------

class TestConfigGetDevice:
    def test_returns_cpu_when_use_gpu_false(self):
        config = _make_config(USE_GPU=False)
        assert config.get_device() == "cpu"

    def test_raises_when_use_gpu_true_but_no_gpu(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # Patch MPS so it is also unavailable
        mps_mock = type("mps", (), {"is_available": staticmethod(lambda: False)})()
        monkeypatch.setattr(torch.backends, "mps", mps_mock, raising=False)

        config = _make_config(USE_GPU=True)
        with pytest.raises(RuntimeError, match="USE_GPU=true"):
            config.get_device()
