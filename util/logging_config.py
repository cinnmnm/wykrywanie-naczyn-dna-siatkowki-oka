import os
import sys
import inspect
import logging
from logging.handlers import RotatingFileHandler
try:
    from pythonjsonlogger import jsonlogger
    _HAS_JSONLOGGER = True
except ImportError:
    _HAS_JSONLOGGER = False
from pathlib import Path
import uuid
from datetime import datetime

__all__ = ["setup_logging", "get_logger", "DEFAULT_LOG_DIR"]

DEFAULT_LOG_DIR = Path("logs")


def _default_formatter(run_id: str):
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    return logging.Formatter(fmt)


def _make_log_dir(base_dir: Path):
    base_dir = Path(base_dir)
    run_dir = base_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(
    level: int = logging.INFO,
    log_dir: str | Path | None = None,
    run_id: str | None = None,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    file_name: str = "project.log",
    log_format: str = "text",
):
    """
    Centralized logging configuration.

    - Creates a per-run directory under `log_dir` (if provided) or `logs/DATE_TIME`.
    - Installs a rotating file handler and a console handler.
    - Use `get_logger(__name__)` in modules to retrieve namespaced loggers.
    """
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    log_dir = Path(log_dir)
    run_dir = _make_log_dir(log_dir)

    if run_id is None:
        run_id = uuid.uuid4().hex

    root = logging.getLogger()
    # Avoid multiple handlers in case setup_logging is called multiple times
    if root.handlers:
        # Keep existing handlers but ensure level
        root.setLevel(level)
        return run_dir

    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if log_format == "json" and _HAS_JSONLOGGER:
        json_formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(levelname)s %(name)s %(message)s'
        )
        ch.setFormatter(json_formatter)
    else:
        ch.setFormatter(_default_formatter(run_id))
    root.addHandler(ch)

    # File handler with rotation
    log_path = run_dir / file_name
    fh = RotatingFileHandler(filename=str(log_path), maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(level)
    if log_format == "json" and _HAS_JSONLOGGER:
        fh.setFormatter(json_formatter)
    else:
        fh.setFormatter(_default_formatter(run_id))
    root.addHandler(fh)

    # Optionally add a small banner
    root.info(f"Logging initialized — run_id={run_id}, logs={run_dir}")

    return run_dir


def _resolve_logger_name(requested_name: str | None = None) -> str:
    """Resolve a stable logger name even when modules run as __main__."""
    if requested_name and requested_name != "__main__":
        return requested_name

    stack = inspect.stack()
    try:
        # Skip current frame (this helper) and search for the first real module caller
        for frame_info in stack[1:]:
            module = inspect.getmodule(frame_info.frame)
            if module is None:
                continue

            if getattr(module, "__name__", None) == __name__:
                continue

            spec = getattr(module, "__spec__", None)
            if spec and getattr(spec, "name", None):
                return spec.name

            module_name = getattr(module, "__name__", None)
            if module_name and module_name != "__main__":
                return module_name
    finally:
        for frame_info in stack:
            del frame_info

    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if main_spec and getattr(main_spec, "name", None):
        return main_spec.name

    return requested_name or "__main__"


def get_logger(name: str | None = None):
    """Return a module logger with normalized names."""
    resolved_name = _resolve_logger_name(name)
    return logging.getLogger(resolved_name)


def get_adapter(name: str | None = None, extra: dict | None = None):
    """Return a LoggerAdapter that injects the `extra` context into logs.

    Example:
        adapter = get_adapter(__name__, {"run_id": run_id, "component": "train"})
        adapter.info("Starting phase")
    """
    logger = get_logger(name or __name__)
    return logging.LoggerAdapter(logger, extra or {})
