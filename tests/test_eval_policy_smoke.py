"""
Smoke tests for policy evaluation.

Tests that evaluation completes without crashing and produces valid outputs.
"""

import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.eval_policy import evaluate_policy, save_evaluation_results, update_summary_metrics


def test_eval_random_policy():
    """Test evaluating random policy on synthetic data."""
    # Check if synthetic data exists
    data_path = Path("data/btc_synthetic.csv")
    if not data_path.exists():
        pytest.skip("Synthetic data not found, skipping evaluation smoke test")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # Evaluate random policy
        results = evaluate_policy(
            policy_name="random",
            data_path=str(data_path),
            start_date="2025-12-10",
            end_date="2025-12-10",
            seed=42,
            out_dir=str(out_dir),
        )

        # Check results
        assert results is not None
        assert "policy" in results
        assert results["policy"] == "random"
        assert "equity_curve" in results
        assert "daily_metrics" in results
        assert len(results["equity_curve"]) > 0
        assert not results["daily_metrics"].empty

        # Check that values are finite
        assert np.isfinite(results["equity_curve"]).all(), "Equity values should be finite"
        assert np.isfinite(results["daily_metrics"]["return"]).all(), "Returns should be finite"

        # Save results (may fail on plot, that's OK)
        try:
            save_evaluation_results(results, out_dir, "random")
        except Exception:
            pass  # Plot may fail, but CSV should still be created

        # Check that files were created
        equity_csv = out_dir / "eval_random_equity_curve.csv"
        daily_csv = out_dir / "eval_random_daily_metrics.csv"

        assert equity_csv.exists(), "Equity curve CSV should be created"
        assert daily_csv.exists(), "Daily metrics CSV should be created"

        # Check CSV contents
        equity_df = pd.read_csv(equity_csv)
        assert len(equity_df) > 0
        assert "equity" in equity_df.columns

        daily_df = pd.read_csv(daily_csv)
        assert len(daily_df) > 0
        assert "day" in daily_df.columns
        assert "return" in daily_df.columns


def test_eval_baseline_policy():
    """Test evaluating baseline policy on synthetic data."""
    # Check if synthetic data exists
    data_path = Path("data/btc_synthetic.csv")
    if not data_path.exists():
        pytest.skip("Synthetic data not found, skipping evaluation smoke test")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # Evaluate baseline policy
        results = evaluate_policy(
            policy_name="baseline",
            data_path=str(data_path),
            start_date="2025-12-10",
            end_date="2025-12-10",
            seed=42,
            out_dir=str(out_dir),
        )

        # Check results
        assert results is not None
        assert results["policy"] == "baseline"
        assert len(results["equity_curve"]) > 0

        # Check that values are finite
        assert np.isfinite(results["equity_curve"]).all(), "Equity values should be finite"

        # Save results (may fail on plot, that's OK)
        try:
            save_evaluation_results(results, out_dir, "baseline")
        except Exception:
            pass  # Plot may fail, but CSV should still be created
        
        update_summary_metrics(results, out_dir)

        # Check that summary metrics file was created
        summary_csv = out_dir / "eval_metrics.csv"
        assert summary_csv.exists(), "Summary metrics CSV should be created"

        # Check summary contents
        summary_df = pd.read_csv(summary_csv)
        assert len(summary_df) > 0
        assert "policy" in summary_df.columns
        assert "total_return" in summary_df.columns
        assert "baseline" in summary_df["policy"].values


def test_eval_dqn_policy():
    """Test evaluating DQN policy (requires checkpoint)."""
    # Check if synthetic data exists
    data_path = Path("data/btc_synthetic.csv")
    checkpoint_path = Path("models/dqn_last.pt")

    if not data_path.exists():
        pytest.skip("Synthetic data not found, skipping DQN evaluation test")
    if not checkpoint_path.exists():
        pytest.skip("DQN checkpoint not found, skipping DQN evaluation test")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # Evaluate DQN policy
        results = evaluate_policy(
            policy_name="dqn",
            data_path=str(data_path),
            start_date="2025-12-10",
            end_date="2025-12-10",
            checkpoint_path=str(checkpoint_path),
            seed=42,
            out_dir=str(out_dir),
        )

        # Check results
        assert results is not None
        assert results["policy"] == "dqn"
        assert len(results["equity_curve"]) > 0

        # Check that values are finite
        assert np.isfinite(results["equity_curve"]).all(), "Equity values should be finite"

        # Save results (may fail on plot, that's OK)
        try:
            save_evaluation_results(results, out_dir, "dqn")
        except Exception:
            pass  # Plot may fail, but CSV should still be created

        # Check that files were created
        equity_csv = out_dir / "eval_dqn_equity_curve.csv"
        assert equity_csv.exists(), "Equity curve CSV should be created"


def test_eval_policy_invalid():
    """Test that invalid policy raises error."""
    data_path = Path("data/btc_synthetic.csv")
    if not data_path.exists():
        pytest.skip("Synthetic data not found")

    with pytest.raises(ValueError, match="Unknown policy"):
        evaluate_policy(
            policy_name="invalid",
            data_path=str(data_path),
            start_date="2025-12-10",
            end_date="2025-12-10",
        )


def test_eval_dqn_missing_checkpoint():
    """Test that DQN policy requires checkpoint."""
    data_path = Path("data/btc_synthetic.csv")
    if not data_path.exists():
        pytest.skip("Synthetic data not found")

    with pytest.raises(ValueError, match="checkpoint_path is required"):
        evaluate_policy(
            policy_name="dqn",
            data_path=str(data_path),
            start_date="2025-12-10",
            end_date="2025-12-10",
            checkpoint_path=None,
        )

