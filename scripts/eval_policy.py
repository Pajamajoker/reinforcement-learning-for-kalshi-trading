"""
Policy evaluation script for Kalshi BTC trading agent.

Evaluates different policies (random, baseline, dqn) on historical data
and generates metrics and plots.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.data_loader import load_candles, get_day_slice
from backtest.metrics import calculate_metrics
from env.kalshi_btc_env import KalshiBTCEnv
from agent.dqn import DQNAgent
from src.common.config import get_config
from src.common.logger import setup_logger, get_logger


def random_policy(obs: np.ndarray, env: KalshiBTCEnv) -> int:
    """Random policy: sample action from action space."""
    return env.action_space.sample()


def baseline_policy(obs: np.ndarray, env: KalshiBTCEnv) -> int:
    """
    Baseline policy: buy YES if implied_prob > 0.55 else do nothing.

    Args:
        obs: Observation array.
        env: Environment instance.

    Returns:
        Action (0=do nothing, 1=buy YES, 2=sell YES).
    """
    # Extract mid price from observation (it's at index observation_window + 1)
    mid_price = obs[env.observation_window + 1]

    if mid_price > 0.55:
        return 1  # Buy YES
    else:
        return 0  # Do nothing


def dqn_policy(obs: np.ndarray, agent: DQNAgent) -> int:
    """
    DQN policy: act greedily using loaded agent.

    Args:
        obs: Observation array.
        agent: Loaded DQN agent.

    Returns:
        Action (0=do nothing, 1=buy YES, 2=sell YES).
    """
    return agent.select_action(obs, explore=False)


def evaluate_policy(
    policy_name: str,
    data_path: str,
    start_date: str,
    end_date: str,
    checkpoint_path: Optional[str] = None,
    seed: Optional[int] = None,
    out_dir: str = "backtest/results",
) -> Dict[str, Any]:
    """
    Evaluate a policy on historical data.

    Args:
        policy_name: Name of policy ("random", "baseline", or "dqn").
        data_path: Path to CSV file with candle data.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        checkpoint_path: Path to DQN checkpoint (required for policy="dqn").
        seed: Random seed.
        out_dir: Output directory for results.

    Returns:
        Dictionary with evaluation results and metrics.
    """
    logger = get_logger(__name__)

    # Validate policy
    if policy_name not in ["random", "baseline", "dqn"]:
        raise ValueError(f"Unknown policy: {policy_name}")

    if policy_name == "dqn" and checkpoint_path is None:
        raise ValueError("checkpoint_path is required for policy='dqn'")

    # Load config
    config = get_config()
    if seed is None:
        seed = config.backtest.random_seed

    # Set seeds
    import torch
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load candles
    logger.info(f"Loading candles from {data_path}")
    df = load_candles(data_path)

    # Parse dates
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    # Get list of days to evaluate
    days = pd.date_range(start, end, freq="D")
    days_str = [d.strftime("%Y-%m-%d") for d in days]

    logger.info(f"Evaluating {policy_name} policy on {len(days_str)} days from {start_date} to {end_date}")

    # Load DQN agent if needed
    agent = None
    if policy_name == "dqn":
        logger.info(f"Loading DQN checkpoint from {checkpoint_path}")
        # Need to create agent with correct obs_dim first
        # We'll initialize it after first env creation
        temp_env = None

    # Track results
    all_equity = []
    daily_metrics = []
    all_trades = []
    all_wins = []
    all_losses = []

    # Evaluate each day
    for day_idx, day_str in enumerate(days_str):
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

            # Initialize DQN agent on first day if needed
            if policy_name == "dqn" and agent is None:
                obs_dim = env.observation_space.shape[0]
                agent = DQNAgent(
                    obs_dim=obs_dim,
                    learning_rate=config.rl.learning_rate,
                    gamma=config.rl.gamma,
                    replay_buffer_size=config.rl.replay_buffer_size,
                    batch_size=config.rl.batch_size,
                    target_update_freq=config.rl.target_update_freq,
                    epsilon_start=config.rl.epsilon_start,
                    epsilon_end=config.rl.epsilon_end,
                    epsilon_decay_steps=config.rl.epsilon_decay_steps,
                    hidden_sizes=config.rl.hidden_sizes,
                    seed=seed,
                )
                agent.load(checkpoint_path)
                logger.info(f"Loaded DQN agent with obs_dim={obs_dim}")

            # Reset environment
            obs, info = env.reset(seed=seed)

            # Episode metrics
            episode_equity = [info["equity"]]
            episode_trades = 0
            episode_wins = 0
            episode_losses = 0
            initial_equity = info["equity"]

            # Run episode
            done = False
            step_count = 0

            while not done:
                # Select action based on policy
                if policy_name == "random":
                    action = random_policy(obs, env)
                elif policy_name == "baseline":
                    action = baseline_policy(obs, env)
                elif policy_name == "dqn":
                    action = dqn_policy(obs, agent)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Track equity
                episode_equity.append(info["equity"])

                # Track trades
                if info.get("action_taken") in ["BUY", "SELL"]:
                    episode_trades += 1

                # Track wins/losses from resolved events
                if info.get("event_resolved") and "event_outcome" in info:
                    # This is a bit indirect - we track based on realized P&L changes
                    # For simplicity, we'll count based on final episode info
                    pass

                step_count += 1

            # Get final episode info
            final_info = env._get_info()

            # Calculate daily return
            final_equity = final_info["equity"]
            daily_return = (final_equity / initial_equity) - 1.0

            # Track wins/losses from episode info
            episode_wins = final_info.get("wins", 0)
            episode_losses = final_info.get("losses", 0)

            # Record daily metrics
            daily_metrics.append(
                {
                    "day": day_str,
                    "initial_equity": initial_equity,
                    "final_equity": final_equity,
                    "return": daily_return,
                    "trades": episode_trades,
                    "wins": episode_wins,
                    "losses": episode_losses,
                    "win_rate": (
                        episode_wins / (episode_wins + episode_losses)
                        if (episode_wins + episode_losses) > 0
                        else 0.0
                    ),
                }
            )

            # Track overall metrics
            all_equity.extend(episode_equity)
            all_trades.append(episode_trades)
            all_wins.append(episode_wins)
            all_losses.append(episode_losses)

            logger.info(
                f"Day {day_str}: equity=${final_equity:.2f}, return={daily_return:.2%}, "
                f"trades={episode_trades}, wins={episode_wins}, losses={episode_losses}"
            )

            env.close()

        except Exception as e:
            logger.error(f"Error evaluating day {day_str}: {e}", exc_info=True)
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
    avg_trades_per_day = np.mean(all_trades) if all_trades else 0.0

    results = {
        "policy": policy_name,
        "start_date": start_date,
        "end_date": end_date,
        "days_tested": len(days_str),
        "total_return": overall_metrics["total_return"],
        "max_drawdown": overall_metrics["max_drawdown"],
        "sharpe_ratio": overall_metrics["sharpe_ratio"],
        "total_trades": total_trades,
        "avg_trades_per_day": avg_trades_per_day,
        "win_rate": win_rate,
        "equity_curve": equity_series,
        "daily_metrics": pd.DataFrame(daily_metrics),
    }

    return results


def save_evaluation_results(results: Dict[str, Any], out_dir: Path, policy_name: str) -> None:
    """
    Save evaluation results to files.

    Args:
        results: Results dictionary from evaluate_policy.
        out_dir: Output directory.
        policy_name: Name of policy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(__name__)

    # Save equity curve CSV
    equity_df = pd.DataFrame(
        {
            "step": range(len(results["equity_curve"])),
            "equity": results["equity_curve"].values,
        }
    )
    equity_path = out_dir / f"eval_{policy_name}_equity_curve.csv"
    equity_df.to_csv(equity_path, index=False)
    logger.info(f"Saved equity curve to {equity_path}")

    # Save daily metrics CSV
    daily_path = out_dir / f"eval_{policy_name}_daily_metrics.csv"
    results["daily_metrics"].to_csv(daily_path, index=False)
    logger.info(f"Saved daily metrics to {daily_path}")

    # Plot equity curve
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results["equity_curve"].values)
        ax.set_title(f"Equity Curve - {policy_name.capitalize()} Policy")
        ax.set_xlabel("Step")
        ax.set_ylabel("Equity ($)")
        ax.grid(True)

        plot_path = out_dir / f"eval_{policy_name}_equity_curve.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved equity curve plot to {plot_path}")
    except Exception as e:
        logger.warning(f"Failed to create plot: {e}")
        # Ensure figure is closed even on error
        try:
            plt.close("all")
        except Exception:
            pass


def update_summary_metrics(results: Dict[str, Any], out_dir: Path) -> None:
    """
    Update or create summary metrics CSV.

    Args:
        results: Results dictionary from evaluate_policy.
        out_dir: Output directory.
    """
    summary_path = out_dir / "eval_metrics.csv"

    # Create summary row
    summary_row = {
        "policy": results["policy"],
        "start_date": results["start_date"],
        "end_date": results["end_date"],
        "days_tested": results["days_tested"],
        "total_return": results["total_return"],
        "max_drawdown": results["max_drawdown"],
        "sharpe_ratio": results["sharpe_ratio"],
        "avg_trades_per_day": results["avg_trades_per_day"],
        "win_rate": results["win_rate"],
    }

    # Load existing or create new
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        # Remove existing row for this policy if it exists
        summary_df = summary_df[summary_df["policy"] != results["policy"]]
        # Append new row
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])

    # Save
    summary_df.to_csv(summary_path, index=False)
    logger = get_logger(__name__)
    logger.info(f"Updated summary metrics at {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trading policy")
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "baseline", "dqn"],
        required=True,
        help="Policy to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to DQN checkpoint (required for policy=dqn)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to CSV file with candle data",
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
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--out_dir",
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

    if not data_path:
        logger.error("data_path must be provided via --data_path or config")
        sys.exit(1)

    # Evaluate policy
    logger.info(f"Evaluating {args.policy} policy")
    results = evaluate_policy(
        policy_name=args.policy,
        data_path=data_path,
        start_date=args.start,
        end_date=args.end,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    if not results:
        logger.error("Evaluation failed, no results")
        sys.exit(1)

    # Save results
    out_dir_path = Path(args.out_dir)
    save_evaluation_results(results, out_dir_path, args.policy)
    update_summary_metrics(results, out_dir_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Policy: {results['policy']}")
    logger.info(f"Period: {results['start_date']} to {results['end_date']}")
    logger.info(f"Days tested: {results['days_tested']}")
    logger.info(f"Total return: {results['total_return']:.2%}")
    logger.info(f"Max drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Avg trades/day: {results['avg_trades_per_day']:.2f}")
    logger.info(f"Win rate: {results['win_rate']:.2%}")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to {out_dir_path}")


if __name__ == "__main__":
    main()

