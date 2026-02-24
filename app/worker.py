import json
import logging
import os
import tempfile
import uuid
from typing import Dict, Any, Optional

import pika
from pika.adapters.blocking_connection import BlockingChannel

from .config import Config
from .predictor import TransNetPredictor
from .s3_client import S3Client

logger = logging.getLogger(__name__)


class TransNetWorker:
    def __init__(self, config: Config):
        self.config = config
        self.s3_client = S3Client(config)
        self.predictor = TransNetPredictor(config.WEIGHTS_PATH, config.get_device())
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[BlockingChannel] = None

    def connect(self):
        logger.info(f"Connecting to RabbitMQ: {self.config.RABBITMQ_URL}")
        parameters = pika.URLParameters(self.config.RABBITMQ_URL)
        parameters.heartbeat = 600
        parameters.blocked_connection_timeout = 300

        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue=self.config.QUEUE_NAME, durable=True)
        self.channel.basic_qos(prefetch_count=1)

        logger.info("Connected to RabbitMQ")

    def disconnect(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("Disconnected from RabbitMQ")

    def process_message(self, ch: BlockingChannel, method, properties, body: bytes):
        task_id = str(uuid.uuid4())
        local_video_path = None

        try:
            message = json.loads(body.decode("utf-8"))
            logger.info(f"[{task_id}] Received message: {message}")

            s3_key = message.get("s3_key")
            if not s3_key:
                raise ValueError("Missing 's3_key' in message")

            request_id = message.get("task_id", task_id)
            task_id = request_id

            local_video_path = self.s3_client.download_video(s3_key)

            logger.info(f"[{task_id}] Processing video...")
            result, visualization = self.predictor.predict_video_with_visualization(
                local_video_path
            )

            result_data = {
                "task_id": task_id,
                "s3_key": s3_key,
                "frame_count": result.frame_count,
                "scenes": result.scenes,
                "single_frame_predictions": result.single_frame_predictions,
                "all_frame_predictions": result.all_frame_predictions,
            }

            result_key = f"{self.config.RESULT_PREFIX}{task_id}/result.json"
            self.s3_client.upload_json(result_data, result_key)
            logger.info(f"[{task_id}] Uploaded result to {result_key}")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                visualization.save(tmp.name, "PNG")
                frame_image_key = (
                    f"{self.config.FRAME_IMAGE_PREFIX}{task_id}/visualization.png"
                )
                self.s3_client.upload_file(tmp.name, frame_image_key)
                os.unlink(tmp.name)
                logger.info(f"[{task_id}] Uploaded visualization to {frame_image_key}")

            result_data["result_key"] = result_key
            result_data["visualization_key"] = frame_image_key

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

        local_video_path = self.s3_client.download_video(s3_key)

        try:
            logger.info(f"[{task_id}] Processing video...")
            result, visualization = self.predictor.predict_video_with_visualization(
                local_video_path
            )

            result_data = {
                "task_id": task_id,
                "s3_key": s3_key,
                "frame_count": result.frame_count,
                "scenes": result.scenes,
                "single_frame_predictions": result.single_frame_predictions,
                "all_frame_predictions": result.all_frame_predictions,
            }

            result_key = f"{self.config.RESULT_PREFIX}{task_id}/result.json"
            self.s3_client.upload_json(result_data, result_key)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                visualization.save(tmp.name, "PNG")
                frame_image_key = (
                    f"{self.config.FRAME_IMAGE_PREFIX}{task_id}/visualization.png"
                )
                self.s3_client.upload_file(tmp.name, frame_image_key)
                os.unlink(tmp.name)

            result_data["result_key"] = result_key
            result_data["visualization_key"] = frame_image_key

            return result_data

        finally:
            if os.path.exists(local_video_path):
                os.unlink(local_video_path)
