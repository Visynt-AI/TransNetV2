#!/usr/bin/env python3
import logging
import os
import sys

from dotenv import load_dotenv

from app.config import Config
from app.worker import TransNetWorker

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    config = Config.from_env()

    missing_vars = []
    if not config.S3_ACCESS_KEY:
        missing_vars.append("S3_ACCESS_KEY")
    if not config.S3_SECRET_KEY:
        missing_vars.append("S3_SECRET_KEY")
    if not config.S3_BUCKET:
        missing_vars.append("S3_BUCKET")

    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        sys.exit(1)

    if not os.path.exists(config.WEIGHTS_PATH):
        logger.error(f"Weights file not found: {config.WEIGHTS_PATH}")
        logger.error(
            "Please download the weights file and place it in the weights/ directory"
        )
        sys.exit(1)

    logger.info("Starting TransNetV2 Worker")
    logger.info(f"Queue: {config.QUEUE_NAME}")
    logger.info(f"Device: {config.get_device()}")

    worker = TransNetWorker(config)

    try:
        worker.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
