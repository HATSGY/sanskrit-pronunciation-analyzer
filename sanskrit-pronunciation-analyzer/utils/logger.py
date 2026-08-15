"""
logger.py
---------
Centralized logging configuration for the entire application.
All modules should obtain their logger via `get_logger(__name__)`
so that log output is consistent and easy to trace back to its source.
"""

import logging
import sys
from pathlib import Path

# Directory where log files are stored (created next to the project root)
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"

_configured = False  # Guards against re-adding handlers on repeated imports


def _configure_root_logger() -> None:
    """Configure the root logger exactly once for the whole application."""
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler -> shows INFO and above in the terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler -> persists DEBUG and above for later inspection
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger configured with console + file handlers.

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
    """
    _configure_root_logger()
    return logging.getLogger(name)
