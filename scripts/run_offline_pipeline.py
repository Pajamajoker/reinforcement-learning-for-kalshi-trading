"""
Offline pipeline script for end-to-end workflow.

Runs the complete offline workflow:
1. Validates data and date range
2. Runs baseline multi-day backtest
3. Trains DQN agent
4. Evaluates both baseline and DQN policies
5. Prints summary of results

Usage:
    python scripts/run_offline_pipeline.py --data_path data/btc_1m_last7d.csv --start 2025-12-14 --end 2025-12-20
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.data_loader import load_candles, get_day_slice
from backtest.run_backtest import run_backtest, save_results
from scripts.train_dqn import train_dqn
from scripts.eval_policy import evaluate_policy, save_evaluation_results, update_summary_metrics
from src.common.config import get_config
from src.common.logger import setup_logger, get_logger


def validate_data_and_dates(data_path: str, start_date: str, end_date: str) -> None:
    """
    Validate that data exists and date range is present in the dataset.

    Args:
        data_path: Path to CSV file with candle data.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Raises:
        FileNotFoundError: If data file doesn't exist.
        ValueError: If date range is not available in the data.
    """
    logger = get_logger(__name__)

    # Check file exists
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please ensure the file exists or run: python data/fetch_yfinance_btc.py"
        )

    # Load and check date range
    logger.info(f"Validating data and date range...")
    df = load_candles(data_path)

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    # Check if dates are in range
    data_start = df["timestamp"].min()
    data_end = df["timestamp"].max()

    # Allow dates to be at the boundaries (start can equal data_start, end can equal data_end)
    if start.date() < data_start.date():
        raise ValueError(
            f"Start date {start_date} is before data start {data_start.date()}\n"
            f"Available data range: {data_start.date()} to {data_end.date()}"
        )

    if end.date() > data_end.date():
        raise ValueError(
            f"End date {end_date} is after data end {data_end.date()}\n"
            f"Available data range: {data_start.date()} to {data_end.date()}"
        )

    # Check if we have data for the requested days
    days = pd.date_range(start, end, freq="D")
    days_str = [d.strftime("%Y-%m-%d") for d in days]
    missing_days = []

    for day_str in days_str:
        day_df = get_day_slice(df, day_str)
        if day_df.empty:
            missing_days.append(day_str)

    if missing_days:
        logger.warning(f"Missing data for {len(missing_days)} days: {missing_days[:5]}...")
        logger.warning("Pipeline will continue but may skip these days")

    logger.info(f"Validation passed: {len(days_str)} days requested, data available from {data_start.date()} to {data_end.date()}")


def run_baseline_backtest(
    data_path: str,
    start_date: str,
    end_date: str,
    seed: Optional[int],
    results_dir: Path,
) -> Dict[str, Any]:
    """
    Run baseline multi-day backtest.

    Args:
        data_path: Path to CSV file with candle data.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        seed: Random seed.
        results_dir: Directory to save results.

    Returns:
        Dictionary with backtest results and metrics.
    """
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("STEP 1: Running baseline backtest")
    logger.info("=" * 60)

    results = run_backtest(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        policy_name="baseline",
        seed=seed,
    )

    # Save results (save_results expects the full results dict)
    try:
        save_results(results, results_dir)
    except Exception as e:
        logger.warning(f"Failed to save baseline backtest results (plotting may have failed): {e}")
        # Try to save at least the CSV files manually
        try:
            metrics_df = pd.DataFrame([{
                "policy": results["policy"],
                "start_date": results["start_date"],
                "end_date": results["end_date"],
                "days_tested": results["days_tested"],
                "total_trades": results["total_trades"],
                "total_wins": results.get("total_wins", 0),
                "total_losses": results.get("total_losses", 0),
                "win_rate": results["win_rate"],
                "total_return": results["total_return"],
                "max_drawdown": results["max_drawdown"],
                "sharpe_ratio": results["sharpe_ratio"],
            }])
            (results_dir / "metrics.csv").parent.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(results_dir / "metrics.csv", index=False)
            logger.info("Saved baseline metrics CSV (plotting skipped)")
        except Exception as e2:
            logger.error(f"Failed to save baseline metrics CSV: {e2}")

    logger.info("Baseline backtest complete")
    return results


def run_dqn_training(
    data_path: str,
    start_date: str,
    end_date: str,
    total_steps: int,
    seed: Optional[int],
    models_dir: Path,
    results_dir: Path,
    friction_mode: str = "low_friction",
) -> Path:
    """
    Train DQN agent.

    Args:
        data_path: Path to CSV file with candle data.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        total_steps: Total number of training steps.
        seed: Random seed.
        models_dir: Directory to save models.

    Returns:
        Path to final checkpoint.
    """
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("STEP 2: Training DQN agent")
    logger.info("=" * 60)

    try:
        train_dqn(
            data_path=data_path,
            start_date=start_date,
            end_date=end_date,
            total_steps=total_steps,
            seed=seed,
            save_dir=str(models_dir),
            log_csv=str(results_dir / "dqn_train_metrics.csv"),
            checkpoint_every=1000,
            friction_mode=friction_mode,
        )
    except Exception as e:
        logger.error(f"DQN training failed: {e}", exc_info=True)
        raise

    checkpoint_path = models_dir / "dqn_last.pt"
    logger.info(f"DQN training complete, checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def run_policy_evaluations(
    data_path: str,
    start_date: str,
    end_date: str,
    checkpoint_path: Path,
    seed: Optional[int],
    results_dir: Path,
    friction_mode: str = "realistic",
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate both baseline and DQN policies.

    Args:
        data_path: Path to CSV file with candle data.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        checkpoint_path: Path to DQN checkpoint.
        seed: Random seed.
        results_dir: Directory to save results.

    Returns:
        Dictionary with evaluation results for both policies.
    """
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("STEP 3: Evaluating policies")
    logger.info("=" * 60)

    all_results = {}

    # Evaluate baseline
    logger.info("Evaluating baseline policy...")
    baseline_results = evaluate_policy(
        policy_name="baseline",
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        seed=seed,
        out_dir=str(results_dir),
        friction_mode=friction_mode,
    )
    save_evaluation_results(baseline_results, results_dir, "baseline")
    update_summary_metrics(baseline_results, results_dir)
    all_results["baseline"] = baseline_results

    # Evaluate DQN
    logger.info("Evaluating DQN policy...")
    dqn_results = evaluate_policy(
        policy_name="dqn",
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        checkpoint_path=str(checkpoint_path),
        seed=seed,
        out_dir=str(results_dir),
        friction_mode=friction_mode,
    )
    save_evaluation_results(dqn_results, results_dir, "dqn")
    update_summary_metrics(dqn_results, results_dir)
    all_results["dqn"] = dqn_results

    logger.info("Policy evaluations complete")
    return all_results


def print_summary(
    baseline_results: Dict[str, Any],
    dqn_results: Dict[str, Any],
    results_dir: Path,
) -> None:
    """
    Print summary of results.

    Args:
        baseline_results: Baseline evaluation results.
        dqn_results: DQN evaluation results.
        results_dir: Directory where results are saved.
    """
    logger = get_logger(__name__)

    print("\n" + "=" * 80)
    print("OFFLINE PIPELINE SUMMARY")
    print("=" * 80)

    print("\nBASELINE POLICY:")
    print(f"  Total Return:    {baseline_results['total_return']:>8.2%}")
    print(f"  Max Drawdown:    {baseline_results['max_drawdown']:>8.2%}")
    print(f"  Sharpe Ratio:    {baseline_results['sharpe_ratio']:>8.2f}")
    print(f"  Avg Trades/Day:  {baseline_results['avg_trades_per_day']:>8.2f}")
    print(f"  Win Rate:        {baseline_results['win_rate']:>8.2%}")

    print("\nDQN POLICY:")
    print(f"  Total Return:    {dqn_results['total_return']:>8.2%}")
    print(f"  Max Drawdown:    {dqn_results['max_drawdown']:>8.2%}")
    print(f"  Sharpe Ratio:    {dqn_results['sharpe_ratio']:>8.2f}")
    print(f"  Avg Trades/Day:  {dqn_results['avg_trades_per_day']:>8.2f}")
    print(f"  Win Rate:        {dqn_results['win_rate']:>8.2%}")

    print("\nOUTPUT FILES:")
    print(f"  Results directory: {results_dir.absolute()}")
    print("\n  Baseline files:")
    print(f"    - equity_curve.csv")
    print(f"    - daily_metrics.csv")
    print(f"    - equity_curve.png")
    print(f"    - eval_baseline_equity_curve.csv")
    print(f"    - eval_baseline_daily_metrics.csv")
    print(f"    - eval_baseline_equity_curve.png")
    print("\n  DQN files:")
    print(f"    - eval_dqn_equity_curve.csv")
    print(f"    - eval_dqn_daily_metrics.csv")
    print(f"    - eval_dqn_equity_curve.png")
    print("\n  Summary files:")
    print(f"    - metrics.csv")
    print(f"    - eval_metrics.csv")
    print(f"    - dqn_train_metrics.csv")

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run full offline pipeline: backtest, train, evaluate"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/btc_1m_last7d.csv",
        help="Path to CSV file with candle data (default: data/btc_1m_last7d.csv)",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=20000,
        help="Total training steps for DQN (default: 20000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config)",
    )
    parser.add_argument(
        "--train_friction_mode",
        type=str,
        choices=["realistic", "low_friction"],
        default="low_friction",
        help="Friction mode for training: realistic or low_friction (default: low_friction)",
    )
    parser.add_argument(
        "--eval_friction_mode",
        type=str,
        choices=["realistic", "low_friction"],
        default="realistic",
        help="Friction mode for evaluation: realistic or low_friction (default: realistic)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger()
    logger = get_logger(__name__)

    # Get config
    config = get_config()
    if args.seed is None:
        args.seed = config.backtest.random_seed

    # Set up directories
    results_dir = Path("backtest/results")
    models_dir = Path("models")
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 0: Validate data and dates
        logger.info("=" * 60)
        logger.info("STEP 0: Validating data and date range")
        logger.info("=" * 60)
        validate_data_and_dates(args.data_path, args.start, args.end)

        # Step 1: Run baseline backtest
        baseline_backtest_results = run_baseline_backtest(
            data_path=args.data_path,
            start_date=args.start,
            end_date=args.end,
            seed=args.seed,
            results_dir=results_dir,
        )

        # Step 2: Train DQN
        checkpoint_path = run_dqn_training(
            data_path=args.data_path,
            start_date=args.start,
            end_date=args.end,
            total_steps=args.total_steps,
            seed=args.seed,
            models_dir=models_dir,
            results_dir=results_dir,
            friction_mode=args.train_friction_mode,
        )

        # Step 3: Evaluate both policies
        eval_results = run_policy_evaluations(
            data_path=args.data_path,
            start_date=args.start,
            end_date=args.end,
            checkpoint_path=checkpoint_path,
            seed=args.seed,
            results_dir=results_dir,
            friction_mode=args.eval_friction_mode,
        )

        # Step 4: Print summary
        print_summary(
            baseline_results=eval_results["baseline"],
            dqn_results=eval_results["dqn"],
            results_dir=results_dir,
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

