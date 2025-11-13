"""Structured logging configuration."""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str | int = "INFO") -> None:
    """Configure standard library logging and structlog for the application."""

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given name."""

    return structlog.get_logger(name)


def bind_context(logger: structlog.stdlib.BoundLogger, **kwargs: Any) -> structlog.stdlib.BoundLogger:
    """Return a logger with additional context bound."""

    return logger.bind(**kwargs)


__all__ = ["configure_logging", "get_logger", "bind_context"]
