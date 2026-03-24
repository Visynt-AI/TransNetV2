import json
import logging
import math
import os
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
    def _probe_video_metadata(video_path: str, frame_count: int) -> tuple[float, float]:
        import ffmpeg

        probe = ffmpeg.probe(video_path)
        video_stream = next(
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        )
        duration = video_stream.get("duration") or probe["format"].get("duration")
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
        publish_done_queue: bool = False,
    ) -> Dict[str, Any]:
        logger.info(f"[{task_id}] Processing video...")
        result = self.predictor.predict_video(
            local_video_path,
            threshold=scene_threshold,
        )

        video_duration_seconds, fps = self._probe_video_metadata(
            local_video_path, result.frame_count
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

        result_data = {
            "task_id": task_id,
            "s3_key": s3_key,
            "frame_count": result.frame_count,
            "fps": fps,
            "scene_threshold": scene_threshold,
            "max_scene_sample_interval_seconds": max_scene_sample_interval_seconds,
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
        task_id = str(uuid.uuid4())
        local_video_path = None

        try:
            message = json.loads(body.decode("utf-8"))
            logger.info(f"[{task_id}] Received message: {message}")

            s3_key = message.get("s3_key")
            if not s3_key:
                raise ValueError("Missing 's3_key' in message")

            task_id = message.get("task_id", task_id)
            scene_threshold = float(message.get("scene_threshold", 0.5))
            max_scene_sample_interval_seconds = float(
                message.get("max_scene_sample_interval_seconds", 5.0)
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
        self.connect()

        if self.channel is None:
            raise OSError("Channel is None")

        self.channel.basic_consume(
            queue=self.config.QUEUE_NAME, on_message_callback=self.process_message
        )

        logger.info(f"Waiting for messages on queue '{self.config.QUEUE_NAME}'...")
        logger.info("Press CTRL+C to exit")

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            self.disconnect()

    def run_once(self, message: Dict[str, Any]) -> Dict[str, Any]:
        task_id = message.get("task_id", str(uuid.uuid4()))
        s3_key = message.get("s3_key")

        if not s3_key:
            raise ValueError("Missing 's3_key' in message")
        scene_threshold = float(message.get("scene_threshold", 0.5))
        max_scene_sample_interval_seconds = float(
            message.get("max_scene_sample_interval_seconds", 5.0)
        )
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
            )

        finally:
            if os.path.exists(local_video_path):
                os.unlink(local_video_path)
