import json
import os

import pika
from dotenv import load_dotenv

load_dotenv()

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_NAME = os.getenv("QUEUE_NAME", "transnet_tasks")

MESSAGE = {
    "task_id": "f62e65f4-7f8d-4d54-bef2-6d3e4b2c4f912",
    "s3_key": "videos/6032d16f-c98d-4c58-b18d-b3d532b00b8f.mp4",
    "scene_threshold": 0.5,
    "max_scene_sample_interval_seconds": 5.0,
}


def main() -> None:
    parameters = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(parameters)

    try:
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)

        body = json.dumps(MESSAGE, ensure_ascii=False)
        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,
            body=body,
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
            ),
        )

        print(f"Message sent to queue: {QUEUE_NAME}")
        print(body)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
