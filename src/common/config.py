"""
Configuration loading from YAML and environment variables.

Safely loads configuration from config.yaml and .env files,
with clear warnings if files are missing (does not crash).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.common.paths import get_config_dir, get_repo_root


class TradingConfig(BaseModel):
    """Trading window configuration."""

    window_start: int = Field(default=9, description="Trading window start hour (24h format)")
    window_end: int = Field(default=16, description="Trading window end hour (24h format)")
    timezone: str = Field(default="America/New_York", description="Timezone for trading window")


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_size: int = Field(default=1000, description="Maximum position size in contracts")
    max_daily_loss: float = Field(default=500.0, description="Maximum daily loss in dollars")
    max_drawdown: float = Field(default=0.10, description="Maximum drawdown as fraction")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    console: bool = Field(default=True, description="Enable console logging")
    file: bool = Field(default=True, description="Enable file logging")
    file_path: str = Field(default="logs/kalshi_agent.log", description="Log file path")
    rotation: str = Field(default="10 MB", description="Log rotation size")
    retention: str = Field(default="7 days", description="Log retention period")


class PathsConfig(BaseModel):
    """File paths configuration."""

    data_dir: str = Field(default="data", description="Data directory")
    logs_dir: str = Field(default="logs", description="Logs directory")
    models_dir: str = Field(default="models", description="Models directory")


class AgentConfig(BaseModel):
    """Agent configuration."""

    model_type: str = Field(default="ppo", description="Model type")
    learning_rate: float = Field(default=0.0003, description="Learning rate")
    batch_size: int = Field(default=64, description="Batch size")


class StrategyConfig(BaseModel):
    """Trading strategy configuration."""

    buy_threshold: float = Field(default=0.55, description="Buy YES if prob > this")
    sell_threshold: float = Field(default=0.45, description="Sell YES if prob < this")


class BacktestConfig(BaseModel):
    """Backtest configuration."""

    initial_capital: float = Field(default=10000.0, description="Initial capital")
    spread: float = Field(default=0.02, description="Half-spread (2% total spread)")
    fee_per_contract: float = Field(default=0.01, description="Fee per contract in dollars")
    lookback_minutes: int = Field(default=60, description="Minutes to look back for volatility")
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    data_path: str = Field(default="data/btc.csv", description="Default path to BTC data")
    observation_window: int = Field(default=60, description="Number of minutes in observation")
    turnover_penalty: float = Field(default=0.001, description="Penalty per contract traded")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    start_date: Optional[str] = Field(default=None, description="Start date for backtest (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date for backtest (YYYY-MM-DD)")


class RLConfig(BaseModel):
    """Reinforcement learning training configuration."""

    learning_rate: float = Field(default=0.001, description="Learning rate for DQN")
    gamma: float = Field(default=0.99, description="Discount factor")
    replay_buffer_size: int = Field(default=50000, description="Size of replay buffer")
    batch_size: int = Field(default=64, description="Batch size for training")
    target_update_freq: int = Field(default=500, description="Frequency of target network updates (steps)")
    epsilon_start: float = Field(default=1.0, description="Initial epsilon for exploration")
    epsilon_end: float = Field(default=0.05, description="Final epsilon after decay")
    epsilon_decay_steps: int = Field(default=5000, description="Steps to decay epsilon")
    hidden_sizes: List[int] = Field(default_factory=lambda: [128, 128], description="Hidden layer sizes for Q-network")
    warmup_steps: int = Field(default=500, description="Steps before training starts")
    total_steps: int = Field(default=2000, description="Default total training steps")


class AppConfig(BaseModel):
    """Main application configuration."""

    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    rl: RLConfig = Field(default_factory=RLConfig)

    # Environment variables (loaded from .env)
    kalshi_api_key_id: Optional[str] = None
    kalshi_private_key_path: Optional[str] = None
    kalshi_base_url: str = "https://demo-api.kalshi.co/trade-api/v2"


# Global config instance
_config: Optional[AppConfig] = None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from YAML file and environment variables.

    If config file is missing, uses defaults and warns clearly.
    If .env is missing, uses defaults and warns clearly.

    Args:
        config_path: Optional path to config.yaml. If None, uses configs/config.yaml.

    Returns:
        AppConfig: Loaded configuration object.
    """
    global _config

    if _config is not None:
        return _config

    # Load environment variables from .env file
    repo_root = get_repo_root()
    env_path = repo_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(
            f"⚠️  Warning: .env file not found at {env_path}. "
            "Using default environment variables. Create .env from .env.example"
        )

    # Load YAML config
    if config_path is None:
        config_path = get_config_dir() / "config.yaml"

    config_dict: Dict[str, Any] = {}

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"⚠️  Warning: Failed to load config from {config_path}: {e}. Using defaults.")
    else:
        print(
            f"⚠️  Warning: Config file not found at {config_path}. "
            "Using default configuration values."
        )

    # Override logging level from environment if present
    log_level = os.getenv("LOG_LEVEL")
    if log_level and "logging" in config_dict:
        config_dict["logging"]["level"] = log_level

    # Load Kalshi API credentials from environment
    kalshi_api_key_id = os.getenv("KALSHI_API_KEY_ID")
    kalshi_private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    kalshi_base_url = os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co/trade-api/v2")

    # Build config object
    _config = AppConfig(
        **config_dict,
        kalshi_api_key_id=kalshi_api_key_id,
        kalshi_private_key_path=kalshi_private_key_path,
        kalshi_base_url=kalshi_base_url,
    )

    return _config


def get_config() -> AppConfig:
    """
    Get the loaded configuration, loading it if necessary.

    Returns:
        AppConfig: The application configuration.
    """
    if _config is None:
        return load_config()
    return _config

