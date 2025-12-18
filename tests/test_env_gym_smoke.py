"""
Smoke tests for Gym environment.

Tests that the environment can be created, reset, and run through a full day
without crashing.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from backtest.data_loader import load_candles, get_day_slice
from env.kalshi_btc_env import KalshiBTCEnv


def create_synthetic_day_data(day_utc: str) -> pd.DataFrame:
    """Create synthetic minute-level BTC data for a single day."""
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


def test_env_creation():
    """Test that environment can be created."""
    day_utc = "2025-12-10"
    day_df = create_synthetic_day_data(day_utc)

    env = KalshiBTCEnv(
        day_df=day_df,
        day_utc=day_utc,
        observation_window=60,
        turnover_penalty=0.001,
        seed=42,
    )

    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None
    assert len(env.events) > 0


def test_env_reset():
    """Test that environment can be reset."""
    day_utc = "2025-12-10"
    day_df = create_synthetic_day_data(day_utc)

    env = KalshiBTCEnv(
        day_df=day_df,
        day_utc=day_utc,
        observation_window=60,
        turnover_penalty=0.001,
        seed=42,
    )

    obs, info = env.reset(seed=42)

    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert "day" in info
    assert "step" in info
    assert "cash" in info
    assert "equity" in info


def test_env_step():
    """Test that environment can step."""
    day_utc = "2025-12-10"
    day_df = create_synthetic_day_data(day_utc)

    env = KalshiBTCEnv(
        day_df=day_df,
        day_utc=day_utc,
        observation_window=60,
        turnover_penalty=0.001,
        seed=42,
    )

    obs, info = env.reset(seed=42)

    # Take a step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "action_taken" in info


def test_env_full_episode():
    """Test that environment can run through a full day."""
    day_utc = "2025-12-10"
    day_df = create_synthetic_day_data(day_utc)

    env = KalshiBTCEnv(
        day_df=day_df,
        day_utc=day_utc,
        observation_window=60,
        turnover_penalty=0.001,
        seed=42,
    )

    obs, info = env.reset(seed=42)

    step_count = 0
    total_reward = 0.0
    done = False

    while not done:
        # Choose random action
        action = env.action_space.sample()

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        total_reward += reward
        done = terminated or truncated

        # Safety check to prevent infinite loops
        assert step_count <= len(env.events) + 5, "Episode did not terminate"

    # Episode should complete
    assert done
    assert step_count > 0
    assert isinstance(total_reward, (int, float))
    assert np.isfinite(total_reward)


def test_env_deterministic():
    """Test that environment is deterministic with same seed."""
    day_utc = "2025-12-10"
    day_df = create_synthetic_day_data(day_utc)

    # Run episode 1
    env1 = KalshiBTCEnv(
        day_df=day_df,
        day_utc=day_utc,
        observation_window=60,
        turnover_penalty=0.001,
        seed=42,
    )
    obs1, info1 = env1.reset(seed=42)
    action = 1  # Buy
    obs1_step, reward1, term1, trunc1, info1_step = env1.step(action)

    # Run episode 2 with same seed
    env2 = KalshiBTCEnv(
        day_df=day_df,
        day_utc=day_utc,
        observation_window=60,
        turnover_penalty=0.001,
        seed=42,
    )
    obs2, info2 = env2.reset(seed=42)
    obs2_step, reward2, term2, trunc2, info2_step = env2.step(action)

    # Observations should be the same (or very close due to floating point)
    assert np.allclose(obs1, obs2, rtol=1e-5)
    assert np.allclose(obs1_step, obs2_step, rtol=1e-5)
    assert abs(reward1 - reward2) < 1e-5


def test_env_with_csv_data():
    """Test environment with CSV-loaded data."""
    day_utc = "2025-12-10"
    day_df = create_synthetic_day_data(day_utc)

    # Save to temporary CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        day_df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        # Load it back
        df = load_candles(temp_path)
        day_slice = get_day_slice(df, day_utc)

        # Create environment
        env = KalshiBTCEnv(
            day_df=day_slice,
            day_utc=day_utc,
            observation_window=60,
            turnover_penalty=0.001,
            seed=42,
        )

        # Run a few steps
        obs, info = env.reset(seed=42)
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert True  # If we get here, no crashes

    finally:
        Path(temp_path).unlink()


