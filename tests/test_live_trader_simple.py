"""
Simple test for live trader core functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from live_trading.live_trader import LiveTrader


def test_live_trader_logging():
    """Test that trader logs trades correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logs_dir = Path(tmpdir) / "logs"
        
        with patch("live_trading.live_trader.KalshiClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            trader = LiveTrader(
                mode="paper",
                qty=1,
                logs_dir=logs_dir,
            )
            
            # Test logging a trade
            trader._log_trade(
                timestamp_utc="2025-12-21T10:00:00Z",
                market_ticker="BTC-TEST",
                event_time_utc="2025-12-21T11:00:00Z",
                action="PAPER_BUY",
                side="yes",
                price=55,
                qty=1,
                order_id=None,
                status="paper_mode",
                error=None,
            )
            
            # Check logs were created
            csv_log = logs_dir / "kalshi_live_trades.csv"
            jsonl_log = logs_dir / "kalshi_live_trades.jsonl"
            
            assert csv_log.exists(), "CSV log should be created"
            assert jsonl_log.exists(), "JSONL log should be created"
            
            # Check CSV content
            import pandas as pd
            df = pd.read_csv(csv_log)
            assert len(df) == 1
            assert df.iloc[0]["market_ticker"] == "BTC-TEST"
            assert df.iloc[0]["action"] == "PAPER_BUY"
            
            # Check JSONL content
            import json
            with open(jsonl_log, "r") as f:
                line = f.readline()
                entry = json.loads(line)
                assert entry["market_ticker"] == "BTC-TEST"
                assert entry["action"] == "PAPER_BUY"


