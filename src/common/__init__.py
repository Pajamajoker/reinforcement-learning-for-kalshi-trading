"""
Common utilities and shared modules.

This package contains:
- config: Configuration loading from YAML and environment variables
- logger: Logging setup using loguru
- paths: Path resolution utilities
"""

from src.common.config import load_config
from src.common.logger import setup_logger, get_logger
from src.common.paths import get_repo_root, get_data_dir, get_logs_dir

__all__ = [
    "load_config",
    "setup_logger",
    "get_logger",
    "get_repo_root",
    "get_data_dir",
    "get_logs_dir",
]


