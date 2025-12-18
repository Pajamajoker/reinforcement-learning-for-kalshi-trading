"""
Path resolution utilities.

Provides robust path resolution for the repository structure,
ensuring paths work correctly regardless of where the code is executed from.
"""

import os
from pathlib import Path
from typing import Optional


def get_repo_root() -> Path:
    """
    Get the repository root directory.

    Assumes this file is in src/common/, so we go up two levels.
    Returns the absolute path to the repository root.

    Returns:
        Path: Absolute path to the repository root directory.
    """
    # Get the directory containing this file (src/common/)
    current_file = Path(__file__).resolve()
    # Go up two levels to reach repo root
    repo_root = current_file.parent.parent.parent
    return repo_root


def get_data_dir() -> Path:
    """
    Get the data directory path.

    Returns:
        Path: Absolute path to the data directory.
    """
    return get_repo_root() / "data"


def get_logs_dir() -> Path:
    """
    Get the logs directory path.

    Returns:
        Path: Absolute path to the logs directory.
    """
    return get_repo_root() / "logs"


def get_config_dir() -> Path:
    """
    Get the configs directory path.

    Returns:
        Path: Absolute path to the configs directory.
    """
    return get_repo_root() / "configs"


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        Path: The same path, now guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


