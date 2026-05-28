import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger. Configures the root logger exactly once,
    so any module can call get_logger(__name__) without worrying
    about who called basicConfig first.

    Log level is read from the LOG_LEVEL env var (default: INFO).
    """
    global _CONFIGURED
    if not _CONFIGURED:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
