#!/usr/bin/env python3
"""
L104 Structured Logging — Production-grade observability via structlog.

Usage:
    from l104_logging import get_logger
    logger = get_logger("MY_MODULE")
    logger.info("event_name", key="value", metric=42)

Output (JSON in production, pretty in dev):
    {"event": "event_name", "key": "value", "metric": 42, "module": "MY_MODULE", "timestamp": "2026-..."}
"""

import os
import sys
import logging
import structlog

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # "json" or "console"
_CONFIGURED = False


def _configure_once():
    """Configure structlog + stdlib logging exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if LOG_FORMAT == "console":
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


def get_logger(module_name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound to a module name.

    Args:
        module_name: Identifier for the subsystem (e.g. "SAGE_CORE", "NEURAL_MESH").

    Returns:
        A structlog BoundLogger with the module name pre-bound.
    """
    _configure_once()
    return structlog.get_logger(module=module_name)
