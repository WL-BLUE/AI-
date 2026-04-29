import sys
from pathlib import Path

from loguru import logger

from aida.core.config import settings


def _setup_logger() -> None:
    log_dir = Path(settings.get("logging.log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = settings.get("logging.level", "INFO")
    log_format = settings.get(
        "logging.format",
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    )
    rotation = settings.get("logging.rotation", "50 MB")
    retention = settings.get("logging.retention", "30 days")

    logger.remove()
    logger.add(sys.stderr, level=log_level, format=log_format)
    logger.add(
        log_dir / "aida_{time:YYYY-MM-DD}.log",
        level=log_level,
        format=log_format,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )


def get_logger(name: str = "aida"):
    return logger.bind(name=name)


_setup_logger()
