"""Logging utilities for NusaBench."""
from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with RichHandler (configured once)."""
    logger = logging.getLogger(name)
    if name not in _CONFIGURED:
        if not logger.handlers:
            handler = RichHandler(rich_tracebacks=True, show_path=False)
            handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
            logger.addHandler(handler)
        logger.setLevel(logging.WARNING)  # default level
        _CONFIGURED.add(name)
    return logger
