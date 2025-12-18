"""
Data loader for historical BTC candle data.

Loads CSV files containing OHLCV candle data and provides utilities
to slice data by day for backtesting.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.common.logger import get_logger


logger = get_logger(__name__)


def load_candles(path: str) -> pd.DataFrame:
    """
    Load BTC candles from CSV file.

    Expected columns: timestamp, open, high, low, close (volume optional)
    Timestamp can be ISO string or epoch (seconds or milliseconds).
    Converts to timezone-aware UTC datetime.

    Args:
        path: Path to CSV file containing candle data.

    Returns:
        DataFrame with columns: timestamp (datetime), open, high, low, close, volume (if present).
        Sorted ascending by timestamp, de-duplicated.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Candle data file not found: {path}")

    logger.info(f"Loading candles from {path}")

    # Load CSV
    df = pd.read_csv(file_path)

    # Check required columns
    required_cols = ["timestamp", "open", "high", "low", "close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Parse timestamp
    if df["timestamp"].dtype == "object" or df["timestamp"].dtype.name.startswith("int"):
        # Try parsing as ISO string first
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        except (ValueError, TypeError):
            # Try as epoch (seconds)
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            except (ValueError, TypeError):
                # Try as epoch (milliseconds)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Ensure timezone-aware (UTC)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    # Sort by timestamp ascending
    df = df.sort_values("timestamp").reset_index(drop=True)

    # De-duplicate (keep first occurrence)
    initial_len = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    if len(df) < initial_len:
        logger.warning(f"Removed {initial_len - len(df)} duplicate timestamps")

    # Ensure numeric columns are numeric
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Log summary
    logger.info(
        f"Loaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}"
    )

    return df


def get_day_slice(df: pd.DataFrame, day_utc: str) -> pd.DataFrame:
    """
    Extract candles for a specific day (UTC).

    Args:
        df: DataFrame with timestamp column (timezone-aware).
        day_utc: Day in format "YYYY-MM-DD" (UTC).

    Returns:
        DataFrame containing only candles from the specified day (00:00:00 to 23:59:59 UTC).

    Raises:
        ValueError: If day_utc format is invalid.
    """
    try:
        # Parse day and create date range
        day_start = pd.Timestamp(day_utc, tz="UTC")
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        # Filter data
        mask = (df["timestamp"] >= day_start) & (df["timestamp"] <= day_end)
        day_df = df[mask].copy().reset_index(drop=True)

        logger.info(f"Extracted {len(day_df)} candles for {day_utc}")

        return day_df

    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid day format '{day_utc}'. Expected YYYY-MM-DD: {e}")


