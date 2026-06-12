

import logging
import sys
from app.core.config import settings


def setup_logging():
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Silence noisy third-party loggers
    for noisy in ["httpx", "httpcore", "qdrant_client", "aiormq", "aio_pika"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("phase3")


logger = setup_logging()
