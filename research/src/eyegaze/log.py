"""Structured logging: JSON lines for machines (CI, Colab logs), console for humans."""

from __future__ import annotations

import logging
import sys

import structlog


class _StderrLogger:
    """Resolves ``sys.stderr`` on every call (it may be swapped by test runners or notebooks)."""

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    log = debug = info = warn = warning = error = critical = exception = fatal = msg


def _stderr_factory(*_args: object) -> _StderrLogger:
    return _StderrLogger()


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown log level {level!r}")
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric)
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=_stderr_factory,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
