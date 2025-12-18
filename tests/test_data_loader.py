"""
Tests for data_loader module.
"""

import pandas as pd
import pytest
import tempfile
from pathlib import Path

from backtest.data_loader import load_candles, get_day_slice


def test_load_candles_iso_timestamp():
    """Test loading candles with ISO timestamp format."""
    # Create synthetic CSV with ISO timestamps
    csv_content = """timestamp,open,high,low,close,volume
2025-12-10T09:00:00Z,64321.50,64350.25,64300.00,64325.75,1250.5
2025-12-10T09:01:00Z,64325.75,64340.00,64320.00,64335.50,980.2
2025-12-10T09:02:00Z,64335.50,64360.00,64330.00,64355.25,1100.8"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = f.name

    try:
        df = load_candles(temp_path)

        assert len(df) == 3
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns

        # Check timestamp is timezone-aware
        assert df["timestamp"].dt.tz is not None

        # Check sorted
        assert df["timestamp"].is_monotonic_increasing

    finally:
        Path(temp_path).unlink()


def test_load_candles_epoch_timestamp():
    """Test loading candles with epoch timestamp format."""
    # Create synthetic CSV with epoch timestamps (seconds)
    csv_content = """timestamp,open,high,low,close
1733821200,64321.50,64350.25,64300.00,64325.75
1733821260,64325.75,64340.00,64320.00,64335.50
1733821320,64335.50,64360.00,64330.00,64355.25"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = f.name

    try:
        df = load_candles(temp_path)

        assert len(df) == 3
        assert df["timestamp"].dt.tz is not None

    finally:
        Path(temp_path).unlink()


def test_load_candles_deduplication():
    """Test that duplicate timestamps are removed."""
    csv_content = """timestamp,open,high,low,close
2025-12-10T09:00:00Z,64321.50,64350.25,64300.00,64325.75
2025-12-10T09:01:00Z,64325.75,64340.00,64320.00,64335.50
2025-12-10T09:01:00Z,64330.00,64345.00,64325.00,64340.00
2025-12-10T09:02:00Z,64335.50,64360.00,64330.00,64355.25"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_path = f.name

    try:
        df = load_candles(temp_path)

        # Should have 3 rows (duplicate removed)
        assert len(df) == 3

    finally:
        Path(temp_path).unlink()


def test_get_day_slice():
    """Test slicing data by day."""
    # Create synthetic data spanning multiple days
    dates = pd.date_range("2025-12-10 00:00:00", "2025-12-12 23:59:00", freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 64000.0,
            "high": 64500.0,
            "low": 63500.0,
            "close": 64200.0,
        }
    )

    # Get slice for 2025-12-11
    day_slice = get_day_slice(df, "2025-12-11")

    # Should have 24 hours of data
    assert len(day_slice) == 24

    # All timestamps should be on 2025-12-11
    assert all(day_slice["timestamp"].dt.date == pd.Timestamp("2025-12-11").date())


def test_get_day_slice_empty():
    """Test slicing with no data for that day."""
    dates = pd.date_range("2025-12-10 00:00:00", "2025-12-10 23:59:00", freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 64000.0,
            "high": 64500.0,
            "low": 63500.0,
            "close": 64200.0,
        }
    )

    # Get slice for a day with no data
    day_slice = get_day_slice(df, "2025-12-15")

    # Should be empty
    assert len(day_slice) == 0

