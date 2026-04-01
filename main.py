#!/usr/bin/env python3
import logging
import sys

from app.config import Config
from app.worker import TransNetWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    config = Config.from_env()

    try:
        config.validate()
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("Starting TransNetV2 Worker")
    logger.info(f"Queue: {config.QUEUE_NAME}")
    logger.info(f"Done queue: {config.DONE_QUEUE_NAME}")
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
