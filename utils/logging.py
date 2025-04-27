import logging
import logging.config
import os
from pathlib import Path
from datetime import datetime


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
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("langchain").setLevel(logging.ERROR)
    logging.getLogger("langsmith").setLevel(logging.ERROR)
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
