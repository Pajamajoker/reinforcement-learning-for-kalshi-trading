"""
Logging setup using loguru.

Configures loguru for console and file logging based on configuration.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.common.config import get_config
from src.common.paths import ensure_dir, get_logs_dir


def setup_logger(config_path: Optional[Path] = None) -> None:
    """
    Set up loguru logger based on configuration.

    Configures console and file logging according to the config.yaml settings.
    If config is not loaded yet, uses sensible defaults.

    Args:
        config_path: Optional path to config file. If None, uses default.
    """
    # Remove default handler
    logger.remove()

    # Get config (will load if not already loaded)
    from src.common.config import load_config

    config = load_config(config_path)
    log_config = config.logging

    # Determine log level
    log_level = log_config.level.upper()

    # Add console handler if enabled
    if log_config.console:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True,
        )

    # Add file handler if enabled
    if log_config.file:
        logs_dir = ensure_dir(get_logs_dir())
        log_file_path = logs_dir / Path(log_config.file_path).name

        logger.add(
            log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=log_config.rotation,
            retention=log_config.retention,
            compression="zip",
        )


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance.

    Args:
        name: Optional logger name. If None, returns the default logger.

    Returns:
        Logger: Loguru logger instance.
    """
    if name:
        return logger.bind(name=name)
    return logger


