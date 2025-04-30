import logging
import logging.config
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

logger.remove()  # remove default handler
logger.add(lambda *_: None, level="WARNING")  # keep ≥ WARNING only


def setup_logging(
    verbose: bool = False, log_dir="logs", app_name="misinformation_detection"
):
    """
    Configure logging for the application with timestamped log files.

    Args:
        verbose: If True, set log level to DEBUG; otherwise, INFO
        log_dir: Directory to store log files
        app_name: Name of the application (used for log files)
    """
    # Configure third-party library logging first
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langsmith").setLevel(logging.WARNING)
    logging.getLogger("duckduckgo_search").setLevel(logging.WARNING)
    # 2) Silence any underlying HTTP logs you don’t care about
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # This handles pesky print statements
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{app_name}_{timestamp}.log"

    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "simple": {"format": "%(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG" if verbose else "INFO",
                "formatter": "simple",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": os.path.join(log_dir, log_filename),
                "mode": "a",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": True,
            }
        },
    }

    # Apply the configuration
    logging.config.dictConfig(config)

    logger = logging.getLogger(__name__)
    logger.info(f"Started new logging session in: {log_filename}")

    return logger
