"""
DQN training script for Kalshi BTC trading agent.

Trains a DQN agent on the Gym environment using historical data.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.data_loader import load_candles, get_day_slice
from env.kalshi_btc_env import KalshiBTCEnv
from agent.dqn import DQNAgent
from src.common.config import get_config
from src.common.logger import setup_logger, get_logger


def train_dqn(
    data_path: str,
    start_date: str,
    end_date: str,
    total_steps: int = 2000,
    seed: Optional[int] = None,
    save_dir: str = "models",
    log_csv: str = "logs/dqn_train_metrics.csv",
    checkpoint_every: int = 1000,
) -> None:
    """
    Train DQN agent.

    Args:
        data_path: Path to CSV file with candle data.
        start_date: Start date for training (YYYY-MM-DD).
        end_date: End date for training (YYYY-MM-DD).
        total_steps: Total number of training steps.
        seed: Random seed.
        save_dir: Directory to save checkpoints.
        log_csv: Path to CSV file for logging metrics.
        checkpoint_every: Frequency of checkpoint saves (in steps).
    """
    # Setup logging
    setup_logger()
    logger = get_logger(__name__)

    # Load config
    config = get_config()
    rl_config = config.rl

    # Use config defaults if not provided
    if seed is None:
        seed = config.backtest.random_seed

    # Set seeds
    import torch
    import numpy as np
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

    # Get list of days to train on
    days = pd.date_range(start, end, freq="D")
    days_str = [d.strftime("%Y-%m-%d") for d in days]

    logger.info(f"Training on {len(days_str)} days from {start_date} to {end_date}")

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize agent (will get obs_dim from first environment)
    agent = None
    obs_dim = None

    # Training metrics
    episode_metrics = []
    current_episode = 0
    step_count = 0
    episode_losses = []

    # Training loop
    while step_count < total_steps:
        # Cycle through days
        for day_str in days_str:
            if step_count >= total_steps:
                break

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

                # Initialize agent on first episode
                if agent is None:
                    obs_dim = env.observation_space.shape[0]
                    agent = DQNAgent(
                        obs_dim=obs_dim,
                        learning_rate=rl_config.learning_rate,
                        gamma=rl_config.gamma,
                        replay_buffer_size=rl_config.replay_buffer_size,
                        batch_size=rl_config.batch_size,
                        target_update_freq=rl_config.target_update_freq,
                        epsilon_start=rl_config.epsilon_start,
                        epsilon_end=rl_config.epsilon_end,
                        epsilon_decay_steps=rl_config.epsilon_decay_steps,
                        hidden_sizes=rl_config.hidden_sizes,
                        seed=seed,
                    )
                    logger.info(f"Initialized DQN agent with obs_dim={obs_dim}")

                # Reset environment
                obs, info = env.reset(seed=seed)

                # Episode metrics
                episode_return = 0.0
                episode_steps = 0
                episode_losses_ep = []

                # Run episode
                done = False
                while not done and step_count < total_steps:
                    # Select action
                    action = agent.select_action(obs, explore=True)

                    # Step environment
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # Store transition
                    agent.store(obs, action, reward, next_obs, done)

                    # Train after warmup
                    if step_count >= rl_config.warmup_steps:
                        loss = agent.train_step()
                        if loss is not None:
                            episode_losses_ep.append(loss)

                    # Update
                    obs = next_obs
                    episode_return += reward
                    episode_steps += 1
                    step_count += 1

                    # Checkpoint
                    if step_count % checkpoint_every == 0:
                        checkpoint_path = save_path / f"dqn_step_{step_count}.pt"
                        agent.save(str(checkpoint_path))
                        logger.info(f"Checkpoint saved at step {step_count}")

                # Episode complete
                final_info = env._get_info()
                avg_loss = (
                    sum(episode_losses_ep) / len(episode_losses_ep)
                    if episode_losses_ep
                    else 0.0
                )

                episode_metrics.append(
                    {
                        "episode": current_episode,
                        "day": day_str,
                        "steps": episode_steps,
                        "return": episode_return,
                        "final_equity": final_info["equity"],
                        "epsilon": agent.get_epsilon(),
                        "avg_loss": avg_loss,
                        "total_steps": step_count,
                    }
                )

                logger.info(
                    f"Episode {current_episode} ({day_str}): steps={episode_steps}, "
                    f"return={episode_return:.2f}, equity=${final_info['equity']:.2f}, "
                    f"epsilon={agent.get_epsilon():.3f}, avg_loss={avg_loss:.4f}"
                )

                current_episode += 1
                env.close()

            except Exception as e:
                logger.error(f"Error in episode for {day_str}: {e}", exc_info=True)
                continue

    # Save final model
    final_path = save_path / "dqn_last.pt"
    agent.save(str(final_path))
    logger.info(f"Final model saved to {final_path}")

    # Save training metrics
    metrics_df = pd.DataFrame(episode_metrics)
    log_path = Path(log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(log_path, index=False)
    logger.info(f"Training metrics saved to {log_csv}")

    logger.info(f"Training complete: {step_count} steps, {current_episode} episodes")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train DQN agent")
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
        "--total_steps",
        type=int,
        default=None,
        help="Total training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_csv",
        type=str,
        default="logs/dqn_train_metrics.csv",
        help="Path to CSV file for logging metrics",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1000,
        help="Frequency of checkpoint saves (in steps)",
    )

    args = parser.parse_args()

    # Get config
    config = get_config()
    data_path = args.data_path or config.backtest.data_path
    total_steps = args.total_steps or config.rl.total_steps

    if not data_path:
        print("Error: data_path must be provided via --data_path or config")
        sys.exit(1)

    train_dqn(
        data_path=data_path,
        start_date=args.start,
        end_date=args.end,
        total_steps=total_steps,
        seed=args.seed,
        save_dir=args.save_dir,
        log_csv=args.log_csv,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()

