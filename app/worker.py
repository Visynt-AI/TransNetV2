import json
import logging
import math
import os
import time
import uuid
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Dict, Optional

import pika
from pika.adapters.blocking_connection import BlockingChannel

from .config import Config
from .predictor import TransNetPredictor
from .s3_client import S3Client

logger = logging.getLogger(__name__)

_TEXT_SUBTITLE_CODECS = {
    "ass",
    "mov_text",
    "srt",
    "ssa",
    "subrip",
    "text",
    "webvtt",
}

_AUDIO_STREAM_TARGETS = {
    "aac": {"ext": ".m4a", "format": "ipod"},
    "ac3": {"ext": ".ac3", "format": "ac3"},
    "alac": {"ext": ".m4a", "format": "ipod"},
    "eac3": {"ext": ".eac3", "format": "eac3"},
    "flac": {"ext": ".flac", "format": "flac"},
    "mp3": {"ext": ".mp3", "format": "mp3"},
    "opus": {"ext": ".ogg", "format": "ogg"},
    "pcm_s16le": {"ext": ".wav", "format": "wav"},
    "vorbis": {"ext": ".ogg", "format": "ogg"},
}

_SUBTITLE_STREAM_TARGETS = {
    "ass": {"ext": ".ass", "codec": "ass", "format": "ass"},
    "mov_text": {"ext": ".srt", "codec": "srt", "format": "srt"},
    "srt": {"ext": ".srt", "codec": "srt", "format": "srt"},
    "ssa": {"ext": ".ass", "codec": "ass", "format": "ass"},
    "subrip": {"ext": ".srt", "codec": "srt", "format": "srt"},
    "text": {"ext": ".srt", "codec": "srt", "format": "srt"},
    "webvtt": {"ext": ".vtt", "codec": "webvtt", "format": "webvtt"},
}


def _upload_preview_frame(
    config_data: dict[str, Any], local_path: str, s3_key: str
) -> str:
    config = Config(**config_data)
    s3_client = S3Client(config)
    return s3_client.upload_file(local_path, s3_key)


class TransNetWorker:
    def __init__(self, config: Config):
        self.config = config
        self.s3_client = S3Client(config)
        self.predictor = TransNetPredictor(config.WEIGHTS_PATH, config.get_device())
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[BlockingChannel] = None

    def connect(self):
        parameters = pika.URLParameters(self.config.RABBITMQ_URL)
        parameters.heartbeat = 600
        parameters.blocked_connection_timeout = 300

        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue=self.config.QUEUE_NAME, durable=True)
        self.channel.queue_declare(queue=self.config.DONE_QUEUE_NAME, durable=True)
        self.channel.basic_qos(prefetch_count=1)

        logger.info("Connected to RabbitMQ")

    def _publish_result_message(self, task_id: str, result_payload: bytes):
        if self.channel is None or self.channel.is_closed:
            raise RuntimeError("RabbitMQ channel is not available for result publish")

        self.channel.basic_publish(
            exchange="",
            routing_key=self.config.DONE_QUEUE_NAME,
            body=result_payload,
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
            ),
        )
        logger.info(
            f"[{task_id}] Published result to queue '{self.config.DONE_QUEUE_NAME}'"
        )

    def disconnect(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("Disconnected from RabbitMQ")

    @staticmethod
    def _parse_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        raise ValueError(f"Invalid boolean value: {value!r}")

    @staticmethod
    def _probe_video_metadata(
        probe_data: dict[str, Any], frame_count: int
    ) -> tuple[float, float]:
        video_stream = next(
            stream
            for stream in probe_data["streams"]
            if stream["codec_type"] == "video"
        )
        duration = video_stream.get("duration") or probe_data["format"].get("duration")
        if duration is None:
            raise RuntimeError("Unable to determine video duration")
        duration_seconds = float(duration)

        fps = 0.0
        for rate_key in ("avg_frame_rate", "r_frame_rate"):
            raw_rate = video_stream.get(rate_key)
            if not raw_rate:
                continue
            try:
                fps = float(Fraction(raw_rate))
            except (ValueError, ZeroDivisionError):
                continue
            if fps > 0:
                break

        if fps <= 0 and frame_count > 0 and duration_seconds > 0:
            fps = frame_count / duration_seconds

        return duration_seconds, fps

    @staticmethod
    def _probe_media_streams(video_path: str) -> dict[str, Any]:
        import ffmpeg

        return ffmpeg.probe(video_path)

    @staticmethod
    def _extract_audio_stream(
        video_path: str, output_dir: str, task_id: str, probe_data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        import ffmpeg

        audio_streams = [
            stream
            for stream in probe_data.get("streams", [])
            if stream.get("codec_type") == "audio"
        ]
        if not audio_streams:
            return None

        stream = audio_streams[0]
        codec_name = (stream.get("codec_name") or "").lower()
        target = _AUDIO_STREAM_TARGETS.get(
            codec_name, {"ext": ".mka", "format": "matroska"}
        )
        output_path = os.path.join(output_dir, f"{task_id}{target['ext']}")

        try:
            (
                ffmpeg.input(video_path)
                .output(
                    output_path,
                    map="0:a:0",
                    acodec="copy",
                    format=target["format"],
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
        except ffmpeg.Error as exc:
            stderr = (
                exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            )
            raise RuntimeError(
                f"Failed to extract audio stream from {video_path}: {stderr}"
            ) from exc

        return {
            "stream_index": stream.get("index"),
            "codec_name": stream.get("codec_name"),
            "channels": stream.get("channels"),
            "sample_rate": int(stream["sample_rate"])
            if stream.get("sample_rate")
            else None,
            "bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else None,
            "language": (stream.get("tags") or {}).get("language"),
            "local_path": output_path,
            "file_ext": target["ext"],
        }

    @staticmethod
    def _extract_subtitle_streams(
        video_path: str, output_dir: str, task_id: str, probe_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        import ffmpeg

        subtitle_streams = [
            stream
            for stream in probe_data.get("streams", [])
            if stream.get("codec_type") == "subtitle"
        ]

        extracted_subtitles = []
        for subtitle_order, stream in enumerate(subtitle_streams):
            codec_name = (stream.get("codec_name") or "").lower()
            disposition = stream.get("disposition") or {}
            stream_info = {
                "stream_index": stream.get("index"),
                "codec_name": stream.get("codec_name"),
                "language": (stream.get("tags") or {}).get("language"),
                "title": (stream.get("tags") or {}).get("title"),
                "default": bool(disposition.get("default")),
                "forced": bool(disposition.get("forced")),
                "extractable": codec_name in _TEXT_SUBTITLE_CODECS,
            }

            target = _SUBTITLE_STREAM_TARGETS.get(codec_name)
            if target is None:
                extracted_subtitles.append(stream_info)
                continue

            output_path = os.path.join(
                output_dir,
                f"{task_id}-subtitle-{subtitle_order}{target['ext']}",
            )

            try:
                (
                    ffmpeg.input(video_path)
                    .output(
                        output_path,
                        map=f"0:s:{subtitle_order}",
                        format=target["format"],
                        **{"c:s": target["codec"]},
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )
            except ffmpeg.Error as exc:
                stderr = (
                    exc.stderr.decode("utf-8", errors="replace")
                    if exc.stderr
                    else ""
                )
                raise RuntimeError(
                    "Failed to extract subtitle stream "
                    f"{stream.get('index')} from {video_path}: {stderr}"
                ) from exc

            stream_info["local_path"] = output_path
            stream_info["file_ext"] = target["ext"]
            extracted_subtitles.append(stream_info)

        return extracted_subtitles

    @staticmethod
    def _build_scene_sampling_plan(
        scenes: list[list[int]],
        frame_count: int,
        duration_seconds: float,
        max_interval_seconds: float,
    ) -> list[dict[str, Any]]:
        if frame_count <= 0 or duration_seconds <= 0:
            return []

        average_fps = frame_count / duration_seconds
        max_frames_per_sample = max(1, math.ceil(average_fps * max_interval_seconds))
        scene_previews = []

        for scene_index, (start_frame, end_frame) in enumerate(scenes):
            scene_frame_count = end_frame - start_frame + 1
            sample_count = max(1, math.ceil(scene_frame_count / max_frames_per_sample))
            sampled_frames = []

            for sample_index in range(sample_count):
                segment_start = sample_index * scene_frame_count / sample_count
                segment_end = (sample_index + 1) * scene_frame_count / sample_count
                middle_offset = int((segment_start + segment_end - 1) / 2)
                frame_id = start_frame + middle_offset
                frame_id = min(max(frame_id, start_frame), end_frame)
                sampled_frames.append(
                    {
                        "sample_index": sample_index,
                        "frame_id": frame_id,
                    }
                )

            scene_previews.append(
                {
                    "scene_index": scene_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "sample_count": sample_count,
                    "sampled_frames": sampled_frames,
                }
            )

        return scene_previews

    def _upload_scene_preview_frames(
        self, video_path: str, task_id: str, scene_previews: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        uploaded_scenes = []
        config_data = asdict(self.config)
        os.makedirs(self.config.TEMP_DIR, exist_ok=True)

        requested_frame_ids = [
            sampled_frame["frame_id"]
            for scene_preview in scene_previews
            for sampled_frame in scene_preview["sampled_frames"]
        ]
        extracted_frame_paths = self.predictor.extract_frame_images(
            video_path, requested_frame_ids, self.config.TEMP_DIR
        )

        try:
            upload_jobs = []
            for scene_preview in scene_previews:
                for sampled_frame in scene_preview["sampled_frames"]:
                    frame_id = sampled_frame["frame_id"]
                    local_frame_path = extracted_frame_paths[frame_id]
                    frame_key = (
                        f"{self.config.FRAME_IMAGE_PREFIX}{task_id}/{frame_id}.png"
                    )
                    upload_jobs.append((frame_id, local_frame_path, frame_key))

            max_workers = 10
            if max_workers > 0:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {
                        executor.submit(
                            _upload_preview_frame,
                            config_data,
                            local_frame_path,
                            frame_key,
                        ): (frame_id, frame_key)
                        for frame_id, local_frame_path, frame_key in upload_jobs
                    }
                    uploaded_frame_keys = {
                        future_map[future][0]: future_map[future][1]
                        for future in future_map
                    }
                    for future in future_map:
                        future.result()
            else:
                uploaded_frame_keys = {}

            for scene_preview in scene_previews:
                uploaded_frames = []
                for sampled_frame in scene_preview["sampled_frames"]:
                    frame_id = sampled_frame["frame_id"]
                    uploaded_frames.append(
                        {
                            "sample_index": sampled_frame["sample_index"],
                            "frame_id": frame_id,
                            "image_key": uploaded_frame_keys[frame_id],
                        }
                    )

                uploaded_scenes.append(
                    {
                        "scene_index": scene_preview["scene_index"],
                        "start_frame": scene_preview["start_frame"],
                        "end_frame": scene_preview["end_frame"],
                        "sample_count": scene_preview["sample_count"],
                        "sampled_frames": uploaded_frames,
                    }
                )
        finally:
            for local_frame_path in extracted_frame_paths.values():
                if os.path.exists(local_frame_path):
                    os.unlink(local_frame_path)

        return uploaded_scenes

    def _process_video(
        self,
        task_id: str,
        s3_key: str,
        local_video_path: str,
        scene_threshold: float,
        max_scene_sample_interval_seconds: float,
        extract_audio: bool = True,
        extract_subtitles: bool = True,
        publish_done_queue: bool = False,
    ) -> Dict[str, Any]:
        logger.info(f"[{task_id}] Processing video...")
        result = self.predictor.predict_video(
            local_video_path,
            threshold=scene_threshold,
        )
        probe_data = self._probe_media_streams(local_video_path)

        video_duration_seconds, fps = self._probe_video_metadata(
            probe_data, result.frame_count
        )
        scene_previews = self._build_scene_sampling_plan(
            result.scenes,
            result.frame_count,
            video_duration_seconds,
            max_scene_sample_interval_seconds,
        )
        uploaded_scene_previews = self._upload_scene_preview_frames(
            local_video_path,
            task_id,
            scene_previews,
        )

        audio_artifact = None
        subtitle_artifacts = []
        os.makedirs(self.config.TEMP_DIR, exist_ok=True)
        if extract_audio:
            try:
                extracted_audio = self._extract_audio_stream(
                    local_video_path, self.config.TEMP_DIR, task_id, probe_data
                )
                if extracted_audio is not None:
                    local_audio_path = extracted_audio.pop("local_path")
                    audio_key = (
                        f"{self.config.AUDIO_PREFIX}{task_id}/"
                        f"audio{extracted_audio.pop('file_ext')}"
                    )
                    try:
                        self.s3_client.upload_file(local_audio_path, audio_key)
                    finally:
                        if os.path.exists(local_audio_path):
                            os.unlink(local_audio_path)
                    audio_artifact = {
                        **extracted_audio,
                        "audio_key": audio_key,
                    }
            except Exception as exc:
                logger.warning(
                    f"[{task_id}] Audio extraction skipped due to error: {exc}"
                )
                audio_artifact = {"error": str(exc)}

        if extract_subtitles:
            try:
                extracted_subtitles = self._extract_subtitle_streams(
                    local_video_path, self.config.TEMP_DIR, task_id, probe_data
                )
                for subtitle in extracted_subtitles:
                    local_subtitle_path = subtitle.pop("local_path", None)
                    file_ext = subtitle.pop("file_ext", None)
                    if local_subtitle_path is None or file_ext is None:
                        subtitle_artifacts.append(subtitle)
                        continue

                    subtitle_key = (
                        f"{self.config.SUBTITLE_PREFIX}{task_id}/"
                        f"subtitle-{subtitle['stream_index']}{file_ext}"
                    )
                    try:
                        self.s3_client.upload_file(local_subtitle_path, subtitle_key)
                    finally:
                        if os.path.exists(local_subtitle_path):
                            os.unlink(local_subtitle_path)
                    subtitle_artifacts.append(
                        {
                            **subtitle,
                            "subtitle_key": subtitle_key,
                        }
                    )
            except Exception as exc:
                logger.warning(
                    f"[{task_id}] Subtitle extraction skipped due to error: {exc}"
                )
                subtitle_artifacts.append({"error": str(exc)})

        result_data = {
            "task_id": task_id,
            "s3_key": s3_key,
            "frame_count": result.frame_count,
            "fps": fps,
            "source_container": (probe_data.get("format") or {}).get("format_name"),
            "source_extension": os.path.splitext(s3_key)[1].lower() or None,
            "scene_threshold": scene_threshold,
            "max_scene_sample_interval_seconds": max_scene_sample_interval_seconds,
            "audio": audio_artifact,
            "subtitles": subtitle_artifacts,
            "scene_preview_frames": uploaded_scene_previews,
        }

        result_key = f"{self.config.RESULT_PREFIX}{task_id}/result.json"

        result_data["result_key"] = result_key

        result_payload = json.dumps(result_data, ensure_ascii=False, indent=2).encode(
            "utf-8"
        )
        self.s3_client.upload_bytes(
            result_payload,
            result_key,
            content_type="application/json",
        )
        logger.info(f"[{task_id}] Uploaded result to {result_key}")

        if publish_done_queue:
            self._publish_result_message(task_id, result_payload)

        return result_data

    def process_message(self, ch: BlockingChannel, method, properties, body: bytes):
        task_id = f"vs-v1-{uuid.uuid4()}"
        local_video_path = None

        try:
            message = json.loads(body.decode("utf-8"))
            logger.info(f"[{task_id}] Received message: {message}")

            s3_key = message.get("s3_key")
            if not s3_key:
                raise ValueError("Missing 's3_key' in message")

            task_id = message.get("task_id") or task_id
            scene_threshold = float(message.get("scene_threshold", 0.5))
            max_scene_sample_interval_seconds = float(
                message.get("max_scene_sample_interval_seconds", 5.0)
            )
            extract_audio = self._parse_bool(message.get("extract_audio"), True)
            extract_subtitles = self._parse_bool(
                message.get("extract_subtitles"), True
            )
            if not 0 < scene_threshold < 1:
                raise ValueError("'scene_threshold' must be between 0 and 1")
            if max_scene_sample_interval_seconds <= 0:
                raise ValueError(
                    "'max_scene_sample_interval_seconds' must be greater than 0"
                )

            local_video_path = self.s3_client.download_video(s3_key)
            self._process_video(
                task_id,
                s3_key,
                local_video_path,
                scene_threshold,
                max_scene_sample_interval_seconds,
                extract_audio=extract_audio,
                extract_subtitles=extract_subtitles,
                publish_done_queue=True,
            )

            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"[{task_id}] Task completed successfully")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        except ValueError as e:
            logger.error(f"Invalid message format: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        except Exception as e:
            logger.exception(f"[{task_id}] Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        finally:
            if local_video_path and os.path.exists(local_video_path):
                os.unlink(local_video_path)

    def start(self):
        retry_delay = 5
        max_retry_delay = 60

        while True:
            try:
                self.connect()

                if self.channel is None:
                    raise OSError("Channel is None")

                self.channel.basic_consume(
                    queue=self.config.QUEUE_NAME,
                    on_message_callback=self.process_message,
                )

                logger.info(
                    f"Waiting for messages on queue '{self.config.QUEUE_NAME}'..."
                )
                logger.info("Press CTRL+C to exit")

                retry_delay = 5  # 连接成功后重置退避时间
                self.channel.start_consuming()

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                self.disconnect()
                break

            except (
                pika.exceptions.AMQPConnectionError,
                pika.exceptions.AMQPChannelError,
                pika.exceptions.StreamLostError,
                pika.exceptions.ConnectionClosedByBroker,
                pika.exceptions.ConnectionWrongStateError,
                OSError,
            ) as e:
                logger.warning(
                    f"RabbitMQ connection lost: {e}. "
                    f"Reconnecting in {retry_delay}s..."
                )
                self.disconnect()
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

            except Exception as e:
                logger.exception(
                    f"Unexpected error: {e}. Reconnecting in {retry_delay}s..."
                )
                self.disconnect()
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

    def run_once(self, message: Dict[str, Any]) -> Dict[str, Any]:
        task_id = message.get("task_id", str(uuid.uuid4()))
        s3_key = message.get("s3_key")

        if not s3_key:
            raise ValueError("Missing 's3_key' in message")
        scene_threshold = float(message.get("scene_threshold", 0.5))
        max_scene_sample_interval_seconds = float(
            message.get("max_scene_sample_interval_seconds", 5.0)
        )
        extract_audio = self._parse_bool(message.get("extract_audio"), True)
        extract_subtitles = self._parse_bool(message.get("extract_subtitles"), True)
        if not 0 < scene_threshold < 1:
            raise ValueError("'scene_threshold' must be between 0 and 1")
        if max_scene_sample_interval_seconds <= 0:
            raise ValueError(
                "'max_scene_sample_interval_seconds' must be greater than 0"
            )

        local_video_path = self.s3_client.download_video(s3_key)

        try:
            return self._process_video(
                task_id,
                s3_key,
                local_video_path,
                scene_threshold,
                max_scene_sample_interval_seconds,
                extract_audio=extract_audio,
                extract_subtitles=extract_subtitles,
            )

        finally:
            if os.path.exists(local_video_path):
                os.unlink(local_video_path)
