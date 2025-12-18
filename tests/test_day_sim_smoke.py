"""
Smoke tests for day simulation core functions.

Tests that the core simulation components work together without crashing.
"""

import pandas as pd
import numpy as np
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from backtest.data_loader import load_candles, get_day_slice
from env.market_simulator import generate_hourly_events, resolve_event
from env.pricing import implied_prob
from env.execution import quote_from_mid, execute_buy_yes
from env.portfolio import Portfolio


def create_synthetic_day_data(day_utc: str) -> pd.DataFrame:
    """Create synthetic minute-level BTC data for a single day."""
    # Parse day
    day_start = pd.Timestamp(day_utc, tz="UTC")

    # Create minute-level data from 00:00 to 23:59
    timestamps = pd.date_range(day_start, day_start + pd.Timedelta(days=1), freq="1min", tz="UTC")
    timestamps = timestamps[:-1]  # Remove the last timestamp (next day 00:00)

    # Generate synthetic price data (random walk)
    n = len(timestamps)
    base_price = 64000.0
    returns = pd.Series([0.0] + [np.random.normal(0, 0.001) for _ in range(n - 1)])
    prices = base_price * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": 1000.0,
        }
    )

    return df


def test_market_simulator_smoke():
    """Test that market simulator generates events without crashing."""
    day_utc = "2025-12-10"

    # Create synthetic data
    day_df = create_synthetic_day_data(day_utc)

    # Generate events
    events = generate_hourly_events(day_df, day_utc)

    # Should generate events for hours 9-24 (16 events)
    assert len(events) > 0
    assert all(hasattr(e, "event_id") for e in events)
    assert all(hasattr(e, "threshold") for e in events)
    assert all(hasattr(e, "expiry_time") for e in events)
    assert all(hasattr(e, "decision_time") for e in events)


def test_pricing_smoke():
    """Test that pricing function works without crashing."""
    day_utc = "2025-12-10"
    day_df = create_synthetic_day_data(day_utc)

    # Create a simple event
    day_start = pd.Timestamp(day_utc, tz="UTC")
    decision_time = day_start + pd.Timedelta(hours=9) - pd.Timedelta(minutes=5)
    expiry_time = day_start + pd.Timedelta(hours=9)
    threshold = 64000.0

    # Compute probability
    prob = implied_prob(day_df, decision_time, expiry_time, threshold, lookback_minutes=60)

    # Should return a valid probability
    assert 0 <= prob <= 1


def test_execution_smoke():
    """Test that execution functions work without crashing."""
    mid = 0.6
    spread = 0.02

    # Generate bid/ask
    bid, ask = quote_from_mid(mid, spread)

    assert 0 <= bid <= 1
    assert 0 <= ask <= 1
    assert bid <= ask

    # Execute buy
    cost, fees = execute_buy_yes(ask, 1, 0.01)

    assert cost > 0
    assert fees >= 0


def test_portfolio_smoke():
    """Test that portfolio tracking works without crashing."""
    portfolio = Portfolio(cash=10000.0)

    # Add a position
    portfolio.add_position("event-1", 1, 0.6, 0.61)  # Buy 1 contract at 0.6, cost 0.61

    assert portfolio.cash < 10000.0
    assert "event-1" in portfolio.positions

    # Mark to market
    unrealized_pnl = portfolio.mark_to_market("event-1", 0.7)

    # Should have positive unrealized P&L (price went up)
    assert unrealized_pnl > 0

    # Resolve position
    realized_pnl = portfolio.resolve_position("event-1", 1)  # Outcome = 1

    assert realized_pnl != 0
    assert portfolio.positions["event-1"].quantity == 0


def test_end_to_end_smoke():
    """Test that core simulation flow works end-to-end."""
    day_utc = "2025-12-10"

    # Create synthetic data
    day_df = create_synthetic_day_data(day_utc)

    # Generate events
    events = generate_hourly_events(day_df, day_utc)
    assert len(events) > 0

    # Process first event
    event = events[0]

    # Compute probability
    prob = implied_prob(day_df, event.decision_time, event.expiry_time, event.threshold)

    # Generate bid/ask
    bid, ask = quote_from_mid(prob, 0.02)

    # Update event
    event.mid_price = prob
    event.bid = bid
    event.ask = ask

    # Create portfolio
    portfolio = Portfolio(cash=10000.0)

    # Execute trade if prob is high enough
    if prob > 0.55:
        cost, fees = execute_buy_yes(ask, 1, 0.01)
        portfolio.add_position(event.event_id, 1, ask, cost)

    # Resolve event
    resolve_event(event, day_df)

    # Resolve position
    if event.event_id in portfolio.positions:
        position = portfolio.positions[event.event_id]
        if position.quantity != 0 and event.resolved:
            portfolio.resolve_position(event.event_id, event.outcome)

    # Check final state
    summary = portfolio.get_summary()
    assert "cash" in summary
    assert "realized_pnl" in summary


def test_data_loader_with_synthetic():
    """Test data loader with synthetic CSV file."""
    day_utc = "2025-12-10"
    day_df = create_synthetic_day_data(day_utc)

    # Save to temporary CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        day_df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        # Load it back
        df = load_candles(temp_path)

        assert len(df) > 0
        assert "timestamp" in df.columns
        assert "close" in df.columns

        # Get day slice
        day_slice = get_day_slice(df, day_utc)

        assert len(day_slice) > 0

    finally:
        Path(temp_path).unlink()

