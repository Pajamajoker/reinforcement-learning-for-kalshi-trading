"""
Smoke test script to verify configuration and logging setup.

This script imports config and logger, prints the loaded configuration,
and exits cleanly. Use this to verify the project setup is working.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.config import load_config
from src.common.logger import setup_logger, get_logger


def main():
    """Run smoke test."""
    # Setup logger first
    setup_logger()
    logger = get_logger(__name__)

    logger.info("Starting smoke test...")

    # Load configuration
    config = load_config()

    # Print configuration summary
    logger.info("Configuration loaded successfully")
    logger.info(f"Trading window: {config.trading.window_start}:00 - {config.trading.window_end}:00")
    logger.info(f"Risk limits: max_position={config.risk.max_position_size}, max_daily_loss=${config.risk.max_daily_loss}")
    logger.info(f"Log level: {config.logging.level}")
    logger.info(f"Kalshi API URL: {config.kalshi_base_url}")

    # Check API credentials (warn if missing, but don't fail)
    if not config.kalshi_api_key_id:
        logger.warning("KALSHI_API_KEY_ID not set in .env file")
    else:
        logger.info("KALSHI_API_KEY_ID is set (value hidden)")

    if not config.kalshi_private_key_path:
        logger.warning("KALSHI_PRIVATE_KEY_PATH not set in .env file")
    else:
        logger.info(f"KALSHI_PRIVATE_KEY_PATH: {config.kalshi_private_key_path}")

    logger.info("Smoke test completed successfully! ✓")
    print("\n✅ All systems operational. Project scaffold is ready.")


if __name__ == "__main__":
    main()

