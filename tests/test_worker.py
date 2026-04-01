import json
import os

import pytest
from unittest.mock import MagicMock, patch

from app.config import Config
from app.worker import TransNetWorker, _TaskParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> Config:
    return Config(
        RABBITMQ_URL="amqp://localhost/",
        QUEUE_NAME="tasks",
        DONE_QUEUE_NAME="done",
        S3_ENDPOINT_URL="",
        S3_ACCESS_KEY="key",
        S3_SECRET_KEY="secret",
        S3_BUCKET="bucket",
        S3_REGION="us-east-1",
        USE_GPU=False,
        CUDA_VISIBLE_DEVICES=None,
        WEIGHTS_PATH="weights.pth",
        RESULT_PREFIX="results/",
        FRAME_IMAGE_PREFIX="frames/",
        AUDIO_PREFIX="audio/",
        SUBTITLE_PREFIX="subtitles/",
        TEMP_DIR="/tmp",
    )


@pytest.fixture
def worker(config) -> TransNetWorker:
    with (
        patch("app.worker.S3Client"),
        patch("app.worker.TransNetPredictor"),
    ):
        w = TransNetWorker(config)
    return w


def _body(**kwargs) -> bytes:
    return json.dumps({"s3_key": "videos/v.mp4", **kwargs}).encode()


def _method(delivery_tag: int = 1):
    m = MagicMock()
    m.delivery_tag = delivery_tag
    return m


# ---------------------------------------------------------------------------
# _parse_bool
# ---------------------------------------------------------------------------

class TestParseBool:
    def test_none_returns_true_default(self):
        assert TransNetWorker._parse_bool(None, True) is True

    def test_none_returns_false_default(self):
        assert TransNetWorker._parse_bool(None, False) is False

    def test_bool_true_passthrough(self):
        assert TransNetWorker._parse_bool(True, False) is True

    def test_bool_false_passthrough(self):
        assert TransNetWorker._parse_bool(False, True) is False

    @pytest.mark.parametrize("s", ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"])
    def test_truthy_strings(self, s):
        assert TransNetWorker._parse_bool(s, False) is True

    @pytest.mark.parametrize("s", ["0", "false", "False", "FALSE", "no", "NO", "off", "OFF"])
    def test_falsy_strings(self, s):
        assert TransNetWorker._parse_bool(s, True) is False

    def test_int_nonzero_is_true(self):
        assert TransNetWorker._parse_bool(1, False) is True

    def test_int_zero_is_false(self):
        assert TransNetWorker._parse_bool(0, True) is False

    def test_float_nonzero_is_true(self):
        assert TransNetWorker._parse_bool(1.5, False) is True

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid boolean"):
            TransNetWorker._parse_bool("maybe", False)

    def test_whitespace_stripped_before_parse(self):
        assert TransNetWorker._parse_bool("  true  ", False) is True


# ---------------------------------------------------------------------------
# _parse_task_params
# ---------------------------------------------------------------------------

class TestParseTaskParams:
    def test_full_valid_message(self, worker):
        msg = {
            "task_id": "abc-123",
            "s3_key": "videos/v.mp4",
            "scene_threshold": 0.4,
            "max_scene_sample_interval_seconds": 3.0,
            "extract_audio": True,
            "extract_subtitles": False,
        }
        params = worker._parse_task_params(msg, "fallback")
        assert params.task_id == "abc-123"
        assert params.s3_key == "videos/v.mp4"
        assert params.scene_threshold == pytest.approx(0.4)
        assert params.max_scene_sample_interval_seconds == pytest.approx(3.0)
        assert params.extract_audio is True
        assert params.extract_subtitles is False

    def test_uses_fallback_task_id_when_absent(self, worker):
        params = worker._parse_task_params({"s3_key": "v.mp4"}, "fallback-id")
        assert params.task_id == "fallback-id"

    def test_uses_fallback_task_id_when_empty_string(self, worker):
        params = worker._parse_task_params({"s3_key": "v.mp4", "task_id": ""}, "fallback-id")
        assert params.task_id == "fallback-id"

    def test_missing_s3_key_raises(self, worker):
        with pytest.raises(ValueError, match="s3_key"):
            worker._parse_task_params({}, "id")

    def test_null_s3_key_raises(self, worker):
        with pytest.raises(ValueError, match="s3_key"):
            worker._parse_task_params({"s3_key": None}, "id")

    def test_threshold_zero_raises(self, worker):
        with pytest.raises(ValueError, match="scene_threshold"):
            worker._parse_task_params({"s3_key": "v.mp4", "scene_threshold": 0.0}, "id")

    def test_threshold_one_raises(self, worker):
        with pytest.raises(ValueError, match="scene_threshold"):
            worker._parse_task_params({"s3_key": "v.mp4", "scene_threshold": 1.0}, "id")

    def test_negative_interval_raises(self, worker):
        with pytest.raises(ValueError, match="max_scene_sample_interval_seconds"):
            worker._parse_task_params(
                {"s3_key": "v.mp4", "max_scene_sample_interval_seconds": -1.0}, "id"
            )

    def test_zero_interval_raises(self, worker):
        with pytest.raises(ValueError, match="max_scene_sample_interval_seconds"):
            worker._parse_task_params(
                {"s3_key": "v.mp4", "max_scene_sample_interval_seconds": 0.0}, "id"
            )

    def test_defaults_applied(self, worker):
        params = worker._parse_task_params({"s3_key": "v.mp4"}, "id")
        assert params.scene_threshold == pytest.approx(0.5)
        assert params.max_scene_sample_interval_seconds == pytest.approx(5.0)
        assert params.extract_audio is True
        assert params.extract_subtitles is True

    def test_extract_audio_string_false(self, worker):
        params = worker._parse_task_params(
            {"s3_key": "v.mp4", "extract_audio": "false"}, "id"
        )
        assert params.extract_audio is False


# ---------------------------------------------------------------------------
# _upload_and_cleanup
# ---------------------------------------------------------------------------

class TestUploadAndCleanup:
    def test_uploads_file_and_deletes_it(self, tmp_path, worker):
        local = tmp_path / "file.bin"
        local.write_bytes(b"content")
        worker.s3_client.upload_file.return_value = "s3://bucket/key"

        result = worker._upload_and_cleanup(str(local), "key")

        worker.s3_client.upload_file.assert_called_once_with(str(local), "key")
        assert result == "s3://bucket/key"
        assert not local.exists()

    def test_deletes_file_even_when_upload_fails(self, tmp_path, worker):
        local = tmp_path / "file.bin"
        local.write_bytes(b"content")
        worker.s3_client.upload_file.side_effect = RuntimeError("network error")

        with pytest.raises(RuntimeError, match="network error"):
            worker._upload_and_cleanup(str(local), "key")

        assert not local.exists()

    def test_does_not_fail_when_file_already_gone(self, worker):
        worker.s3_client.upload_file.return_value = "s3://bucket/key"
        # File does not exist — should not raise
        worker._upload_and_cleanup("/tmp/nonexistent_xyz.bin", "key")


# ---------------------------------------------------------------------------
# process_message
# ---------------------------------------------------------------------------

class TestProcessMessage:
    def test_valid_message_acks(self, tmp_path, worker):
        local = tmp_path / "v.mp4"
        local.write_bytes(b"x")
        ch = MagicMock()
        worker.s3_client.download_video.return_value = str(local)
        worker._process_video = MagicMock(return_value={})

        worker.process_message(ch, _method(), None, _body())

        ch.basic_ack.assert_called_once_with(delivery_tag=1)
        ch.basic_nack.assert_not_called()

    def test_invalid_json_nacks_no_requeue(self, worker):
        ch = MagicMock()
        worker.process_message(ch, _method(), None, b"not-json{{{")
        ch.basic_nack.assert_called_once_with(delivery_tag=1, requeue=False)
        ch.basic_ack.assert_not_called()

    def test_missing_s3_key_nacks_no_requeue(self, worker):
        ch = MagicMock()
        body = json.dumps({"scene_threshold": 0.5}).encode()
        worker.process_message(ch, _method(), None, body)
        ch.basic_nack.assert_called_once_with(delivery_tag=1, requeue=False)

    def test_invalid_threshold_nacks_no_requeue(self, worker):
        ch = MagicMock()
        body = json.dumps({"s3_key": "v.mp4", "scene_threshold": 2.0}).encode()
        worker.process_message(ch, _method(), None, body)
        ch.basic_nack.assert_called_once_with(delivery_tag=1, requeue=False)

    def test_processing_error_nacks_with_requeue(self, tmp_path, worker):
        local = tmp_path / "v.mp4"
        local.write_bytes(b"x")
        ch = MagicMock()
        worker.s3_client.download_video.return_value = str(local)
        worker._process_video = MagicMock(side_effect=RuntimeError("gpu oom"))

        worker.process_message(ch, _method(), None, _body())

        ch.basic_nack.assert_called_once_with(delivery_tag=1, requeue=True)

    def test_video_file_deleted_on_success(self, tmp_path, worker):
        local = tmp_path / "v.mp4"
        local.write_bytes(b"x")
        ch = MagicMock()
        worker.s3_client.download_video.return_value = str(local)
        worker._process_video = MagicMock(return_value={})

        worker.process_message(ch, _method(), None, _body())

        assert not local.exists()

    def test_video_file_deleted_on_processing_error(self, tmp_path, worker):
        local = tmp_path / "v.mp4"
        local.write_bytes(b"x")
        ch = MagicMock()
        worker.s3_client.download_video.return_value = str(local)
        worker._process_video = MagicMock(side_effect=RuntimeError("fail"))

        worker.process_message(ch, _method(), None, _body())

        assert not local.exists()

    def test_task_id_from_message_used(self, tmp_path, worker):
        local = tmp_path / "v.mp4"
        local.write_bytes(b"x")
        ch = MagicMock()
        worker.s3_client.download_video.return_value = str(local)
        worker._process_video = MagicMock(return_value={})

        body = _body(task_id="my-task-99")
        worker.process_message(ch, _method(), None, body)

        call_kwargs = worker._process_video.call_args
        assert call_kwargs[0][0] == "my-task-99"

    def test_download_failure_nacks_with_requeue(self, worker):
        ch = MagicMock()
        worker.s3_client.download_video.side_effect = RuntimeError("s3 unreachable")

        worker.process_message(ch, _method(), None, _body())

        ch.basic_nack.assert_called_once_with(delivery_tag=1, requeue=True)


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------

class TestRunOnce:
    def test_returns_process_video_result(self, tmp_path, worker):
        local = tmp_path / "v.mp4"
        local.write_bytes(b"x")
        worker.s3_client.download_video.return_value = str(local)
        worker._process_video = MagicMock(return_value={"task_id": "t1"})

        result = worker.run_once({"s3_key": "v.mp4", "task_id": "t1"})

        assert result == {"task_id": "t1"}

    def test_video_file_deleted_on_success(self, tmp_path, worker):
        local = tmp_path / "v.mp4"
        local.write_bytes(b"x")
        worker.s3_client.download_video.return_value = str(local)
        worker._process_video = MagicMock(return_value={})

        worker.run_once({"s3_key": "v.mp4"})

        assert not local.exists()

    def test_video_file_deleted_on_error(self, tmp_path, worker):
        local = tmp_path / "v.mp4"
        local.write_bytes(b"x")
        worker.s3_client.download_video.return_value = str(local)
        worker._process_video = MagicMock(side_effect=RuntimeError("fail"))

        with pytest.raises(RuntimeError):
            worker.run_once({"s3_key": "v.mp4"})

        assert not local.exists()

    def test_missing_s3_key_raises(self, worker):
        with pytest.raises(ValueError, match="s3_key"):
            worker.run_once({})
