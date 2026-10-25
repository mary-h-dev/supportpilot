import io

import structlog

from supportpilot.logging import log, setup_logging


def test_log_exception_includes_the_traceback():
    """Regression: without format_exc_info, log.exception() printed only `"exc_info": true`
    and the real cause of a failure (e.g. Ollama refusing connections) was invisible."""
    buf = io.StringIO()
    try:
        setup_logging(stream=buf)
        try:
            raise RuntimeError("ollama refused connection")
        except RuntimeError:
            log.exception("retrieval_failed")
    finally:
        structlog.reset_defaults()
    out = buf.getvalue()
    assert "Traceback" in out and "ollama refused connection" in out
