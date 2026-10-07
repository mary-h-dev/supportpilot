import logging
import sys

import structlog


def setup_logging(stream=None, level: int = logging.INFO) -> None:
    """JSON logs. MCP stdio servers MUST pass stream=sys.stderr: stdout is the protocol channel."""
    stream = stream or sys.stdout
    logging.basicConfig(format="%(message)s", level=level, stream=sys.stderr)
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.WriteLoggerFactory(file=stream),
    )


log = structlog.get_logger()
