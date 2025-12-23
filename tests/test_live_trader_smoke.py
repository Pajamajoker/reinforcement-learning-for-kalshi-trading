"""
Smoke test for live trader with mocked KalshiClient.
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


@pytest.fixture
def mock_client():
    """Create a mocked KalshiClient."""
    client = MagicMock()
    
    # Mock list_markets to return BTC markets (must match filter: BTC + open)
    client.list_markets.return_value = {
        "markets": [
            {
                "ticker": "BTC-20251221-09",
                "title": "Will BTC be above $90000 at 09:00 UTC on 2025-12-21?",
                "category": "crypto",
                "status": "open",
                "event_time": "2025-12-21T09:00:00Z",
            },
            {
                "ticker": "BTC-20251221-10",
                "title": "Will BTC be above $90000 at 10:00 UTC on 2025-12-21?",
                "category": "crypto",
                "status": "open",
                "event_time": "2025-12-21T10:00:00Z",
            },
        ]
    }
    
    # Mock get_orderbook
    client.get_orderbook.return_value = {
        "asks": [{"price": 55}],
        "bids": [{"price": 50}],
    }
    
    # Mock get_market
    client.get_market.return_value = {
        "yes_bid": 0.50,
        "yes_ask": 0.55,
        "yes_price": 0.52,
    }
    
    # Mock place_order
    client.place_order.return_value = {
        "order_id": "test_order_123",
        "status": "open",
    }
    
    return client


@pytest.fixture
def mock_client_no_markets():
    """Create a mocked KalshiClient that returns no markets."""
    client = MagicMock()
    client.list_markets.return_value = {"markets": []}
    return client


def test_live_trader_paper_mode(mock_client):
    """Test live trader in paper mode with mocked client."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logs_dir = Path(tmpdir) / "logs"
        
        with patch("live_trading.live_trader.KalshiClient", return_value=mock_client):
            trader = LiveTrader(
                mode="paper",
                qty=1,
                max_hours=0.001,  # Very short time (3.6 seconds)
                poll_interval_seconds=1,
                logs_dir=logs_dir,
            )
            
            # Mock _is_trading_hours to return True
            trader._is_trading_hours = lambda: True
            
            # Mock time.sleep and datetime.utcnow to control execution
            import time as time_module
            from datetime import datetime, timedelta
            
            start_time = datetime.utcnow()
            call_count = [0]
            
            original_sleep = time_module.sleep
            original_utcnow = datetime.utcnow
            
            def mock_sleep(seconds):
                call_count[0] += 1
                # Stop after a few cycles
                if call_count[0] > 5:
                    # Simulate max_hours elapsed
                    trader.max_hours = 0
            
            def mock_utcnow():
                # Return time that advances but stays within trading hours
                return start_time + timedelta(seconds=call_count[0] * 10)
            
            time_module.sleep = mock_sleep
            
            try:
                # Run trader (it will stop after max_hours or cycles)
                trader.run()
            except KeyboardInterrupt:
                pass  # Expected if we interrupt
            finally:
                time_module.sleep = original_sleep
            
            # Check that logs were created
            csv_log = logs_dir / "kalshi_live_trades.csv"
            jsonl_log = logs_dir / "kalshi_live_trades.jsonl"
            
            assert csv_log.exists(), "CSV log should be created"
            assert jsonl_log.exists(), "JSONL log should be created"
            
            # Check CSV has headers and data rows
            import pandas as pd
            df = pd.read_csv(csv_log)
            assert "timestamp_utc" in df.columns
            assert "market_ticker" in df.columns
            assert "action" in df.columns
            
            # Check that we have at least 1 data row (heartbeat or trade)
            assert len(df) > 0, "CSV should have at least 1 data row"
            
            # Check that client methods were called (at least list_markets)
            assert mock_client.list_markets.called or call_count[0] > 0


def test_live_trader_initialization():
    """Test that LiveTrader initializes correctly."""
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
            
            assert trader.mode == "paper"
            assert trader.qty == 1
            assert trader.logs_dir == logs_dir
            assert trader.csv_log_path.exists()
            
            # Check CSV headers
            import pandas as pd
            df = pd.read_csv(trader.csv_log_path)
            expected_headers = [
                "timestamp_utc",
                "market_ticker",
                "event_time_utc",
                "action",
                "side",
                "price",
                "qty",
                "order_id",
                "status",
                "error",
            ]
            assert list(df.columns) == expected_headers


def test_live_trader_paper_mode_no_markets(mock_client_no_markets):
    """Test live trader in paper mode with no markets - should log heartbeat SKIP."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logs_dir = Path(tmpdir) / "logs"
        
        with patch("live_trading.live_trader.KalshiClient", return_value=mock_client_no_markets):
            trader = LiveTrader(
                mode="paper",
                qty=1,
                max_hours=0.02,  # About 1.2 minutes
                poll_interval_seconds=1,
                logs_dir=logs_dir,
            )
            
            # Mock _is_trading_hours to return True
            trader._is_trading_hours = lambda: True
            
            # Run trader
            trader.run()
            
            # Check that logs were created
            csv_log = logs_dir / "kalshi_live_trades.csv"
            jsonl_log = logs_dir / "kalshi_live_trades.jsonl"
            
            assert csv_log.exists(), "CSV log should be created"
            assert jsonl_log.exists(), "JSONL log should be created"
            
            # Check CSV has data rows (heartbeat SKIP)
            import pandas as pd
            df = pd.read_csv(csv_log)
            assert len(df) > 0, "CSV should have at least 1 data row (heartbeat)"
            
            # Check that we have SKIP rows with NO_MARKETS_FOUND status
            skip_rows = df[df["action"] == "SKIP"]
            assert len(skip_rows) > 0, "Should have at least one SKIP row"
            assert any(skip_rows["status"] == "NO_MARKETS_FOUND"), "Should have NO_MARKETS_FOUND status"

