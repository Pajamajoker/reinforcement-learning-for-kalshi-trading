"""
Multi-day backtest runner.

Runs the Gym environment across multiple days and evaluates different policies.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.data_loader import load_candles, get_day_slice
from env.kalshi_btc_env import KalshiBTCEnv
from src.common.config import get_config
from src.common.logger import setup_logger, get_logger


def random_policy(obs: np.ndarray, action_space) -> int:
    """Random policy: choose random action."""
    return action_space.sample()


def baseline_policy(obs: np.ndarray, env: KalshiBTCEnv) -> int:
    """
    Baseline policy: buy YES if implied_prob > 0.55 else do nothing.

    Args:
        obs: Observation array.
        env: Environment instance (to access current event).

    Returns:
        Action (0=do nothing, 1=buy YES, 2=sell YES).
    """
    # Extract mid price from observation (it's at index observation_window + 1)
    mid_price = obs[env.observation_window + 1]

    if mid_price > 0.55:
        return 1  # Buy YES
    else:
        return 0  # Do nothing


def calculate_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate performance metrics from equity curve.

    Args:
        equity_curve: Series of equity values over time.

    Returns:
        Dictionary of metrics.
    """
    if len(equity_curve) == 0:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    total_return = (final / initial) - 1.0

    # Calculate drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    # Calculate Sharpe ratio (simple version using returns)
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0.0

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }


def run_backtest(
    data_path: str,
    start_date: str,
    end_date: str,
    policy_name: str = "baseline",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run backtest across multiple days.

    Args:
        data_path: Path to CSV file with candle data.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        policy_name: Name of policy to evaluate ("random" or "baseline").
        seed: Random seed.

    Returns:
        Dictionary with results and metrics.
    """
    logger = get_logger(__name__)
    config = get_config()

    # Load candles
    logger.info(f"Loading candles from {data_path}")
    df = load_candles(data_path)

    # Parse dates
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    # Get list of days to backtest
    days = pd.date_range(start, end, freq="D")
    days_str = [d.strftime("%Y-%m-%d") for d in days]

    logger.info(f"Running backtest from {start_date} to {end_date} ({len(days_str)} days)")

    # Track results
    all_equity = []
    all_trades = []
    all_wins = []
    all_losses = []
    daily_returns = []
    daily_metrics = []

    # Run each day
    for day_str in days_str:
        try:
            # Get day slice
            day_df = get_day_slice(df, day_str)
            if day_df.empty:
                logger.warning(f"No data for {day_str}, skipping")
                continue

            # Create environment
            env = KalshiBTCEnv(
                day_df=day_df,
                day_utc=day_str,
                observation_window=config.backtest.observation_window,
                turnover_penalty=config.backtest.turnover_penalty,
                seed=seed,
            )

            # Reset environment
            obs, info = env.reset(seed=seed)

            # Run episode
            episode_equity = [info["equity"]]
            done = False

            while not done:
                # Choose action based on policy
                if policy_name == "random":
                    action = random_policy(obs, env.action_space)
                elif policy_name == "baseline":
                    action = baseline_policy(obs, env)
                else:
                    raise ValueError(f"Unknown policy: {policy_name}")

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Track equity
                episode_equity.append(info["equity"])

            # Record daily results
            final_info = env._get_info()
            all_equity.extend(episode_equity)
            all_trades.append(final_info["trades"])
            all_wins.append(final_info["wins"])
            all_losses.append(final_info["losses"])

            # Calculate daily return
            daily_return = (episode_equity[-1] / episode_equity[0]) - 1.0
            daily_returns.append(daily_return)

            # Daily metrics
            daily_metrics.append(
                {
                    "day": day_str,
                    "trades": final_info["trades"],
                    "wins": final_info["wins"],
                    "losses": final_info["losses"],
                    "win_rate": (
                        final_info["wins"] / (final_info["wins"] + final_info["losses"])
                        if (final_info["wins"] + final_info["losses"]) > 0
                        else 0.0
                    ),
                    "return": daily_return,
                    "final_equity": episode_equity[-1],
                }
            )

            env.close()

        except Exception as e:
            logger.error(f"Error processing day {day_str}: {e}", exc_info=True)
            continue

    # Calculate overall metrics
    if len(all_equity) == 0:
        logger.error("No data collected, cannot calculate metrics")
        return {}

    equity_series = pd.Series(all_equity)
    overall_metrics = calculate_metrics(equity_series)

    # Overall statistics
    total_trades = sum(all_trades)
    total_wins = sum(all_wins)
    total_losses = sum(all_losses)
    win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0.0

    results = {
        "policy": policy_name,
        "start_date": start_date,
        "end_date": end_date,
        "days_tested": len(days_str),
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "win_rate": win_rate,
        "total_return": overall_metrics["total_return"],
        "max_drawdown": overall_metrics["max_drawdown"],
        "sharpe_ratio": overall_metrics["sharpe_ratio"],
        "equity_curve": equity_series,
        "daily_metrics": pd.DataFrame(daily_metrics),
    }

    return results


def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Save backtest results to files.

    Args:
        results: Results dictionary from run_backtest.
        output_dir: Directory to save results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics CSV
    metrics_df = pd.DataFrame(
        [
            {
                "policy": results["policy"],
                "start_date": results["start_date"],
                "end_date": results["end_date"],
                "days_tested": results["days_tested"],
                "total_trades": results["total_trades"],
                "total_wins": results["total_wins"],
                "total_losses": results["total_losses"],
                "win_rate": results["win_rate"],
                "total_return": results["total_return"],
                "max_drawdown": results["max_drawdown"],
                "sharpe_ratio": results["sharpe_ratio"],
            }
        ]
    )
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Save equity curve CSV
    equity_df = pd.DataFrame(
        {
            "step": range(len(results["equity_curve"])),
            "equity": results["equity_curve"].values,
        }
    )
    equity_path = output_dir / "equity_curve.csv"
    equity_df.to_csv(equity_path, index=False)

    # Save daily metrics
    if not results["daily_metrics"].empty:
        daily_path = output_dir / "daily_metrics.csv"
        results["daily_metrics"].to_csv(daily_path, index=False)

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(results["equity_curve"].values)
    plt.title(f"Equity Curve - {results['policy']} Policy")
    plt.xlabel("Step")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.tight_layout()

    plot_path = output_dir / "equity_curve.png"
    plt.savefig(plot_path)
    plt.close()

    logger = get_logger(__name__)
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run multi-day backtest")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to CSV file with candle data",
        default=None,
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
        "--policy",
        type=str,
        choices=["random", "baseline"],
        default="baseline",
        help="Policy to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="backtest/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger()
    logger = get_logger(__name__)

    # Get config
    config = get_config()
    data_path = args.data_path or config.backtest.data_path
    seed = args.seed or config.backtest.random_seed

    if not data_path:
        logger.error("data_path must be provided via --data_path or config")
        sys.exit(1)

    # Run backtest
    logger.info(f"Running {args.policy} policy backtest")
    results = run_backtest(
        data_path=data_path,
        start_date=args.start,
        end_date=args.end,
        policy_name=args.policy,
        seed=seed,
    )

    if not results:
        logger.error("Backtest failed, no results")
        sys.exit(1)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Policy: {results['policy']}")
    logger.info(f"Period: {results['start_date']} to {results['end_date']}")
    logger.info(f"Days tested: {results['days_tested']}")
    logger.info(f"Total trades: {results['total_trades']}")
    logger.info(f"Win rate: {results['win_rate']:.2%}")
    logger.info(f"Total return: {results['total_return']:.2%}")
    logger.info(f"Max drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
    logger.info("=" * 60)

    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()


