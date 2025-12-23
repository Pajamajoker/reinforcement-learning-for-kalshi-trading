"""
Live Kalshi demo trading module.

Polls for BTC hourly threshold markets and places small orders for evidence.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from live_trading.kalshi_client import KalshiClient
from src.common.logger import setup_logger, get_logger


class LiveTrader:
    """Live trader that places orders on Kalshi demo markets."""

    def __init__(
        self,
        mode: str = "paper",
        qty: int = 1,
        max_hours: Optional[int] = None,
        poll_interval_seconds: int = 30,
        logs_dir: Optional[Path] = None,
    ):
        """
        Initialize the live trader.

        Args:
            mode: Trading mode ("live" or "paper"). Paper mode doesn't place orders.
            qty: Quantity of contracts per order.
            max_hours: Maximum number of trading hours to run (None = unlimited).
            poll_interval_seconds: Seconds between polling cycles.
            logs_dir: Directory for log files (default: live_trading/logs/).
        """
        self.mode = mode.lower()
        self.qty = qty
        self.max_hours = max_hours
        self.poll_interval_seconds = poll_interval_seconds
        self.logger = get_logger(__name__)

        # Setup logs directory (Windows-safe paths)
        if logs_dir is None:
            logs_dir = os.path.join(str(project_root), "live_trading", "logs")
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Log file paths (Windows-safe)
        self.csv_log_path = Path(os.path.join(str(self.logs_dir), "kalshi_live_trades.csv"))
        self.jsonl_log_path = Path(os.path.join(str(self.logs_dir), "kalshi_live_trades.jsonl"))

        # Initialize CSV log file with headers if it doesn't exist
        self._init_csv_log()

        # Initialize client
        try:
            self.client = KalshiClient()
        except Exception as e:
            self.logger.error(f"Failed to initialize KalshiClient: {e}")
            raise

        # Track processed markets to avoid duplicates
        self.processed_markets: set = set()
        
        # Track last hour when an order was placed (to ensure one order per hour)
        self.last_order_hour: Optional[int] = None
        
        # Track last heartbeat minute for throttling (log at most once per minute)
        self.last_heartbeat_minute: Optional[int] = None

        # Trading window: 09:00-24:00 UTC
        self.trading_start_hour = 9
        self.trading_end_hour = 24

    def _init_csv_log(self) -> None:
        """Initialize CSV log file with headers if it doesn't exist."""
        if not self.csv_log_path.exists():
            with open(self.csv_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
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
                ])

    def _log_trade(
        self,
        timestamp_utc: str,
        market_ticker: str,
        event_time_utc: Optional[str],
        action: str,
        side: str,
        price: Optional[int],
        qty: int,
        order_id: Optional[str],
        status: Optional[str],
        error: Optional[str],
        raw_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a trade attempt to CSV and JSONL files.

        Args:
            timestamp_utc: Current UTC timestamp.
            market_ticker: Market ticker symbol.
            event_time_utc: Event expiry time UTC.
            action: Action taken ("BUY", "SKIP", etc.).
            side: Order side ("yes", "no", or empty).
            price: Order price in cents (0-100).
            qty: Order quantity.
            order_id: Order ID if placed.
            status: Order status.
            error: Error message if any.
            raw_response: Raw API response for JSONL log.
        """
        # Write to CSV
        with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_utc,
                market_ticker,
                event_time_utc or "",
                action,
                side,
                price or "",
                qty,
                order_id or "",
                status or "",
                error or "",
            ])

        # Write to JSONL
        log_entry = {
            "timestamp_utc": timestamp_utc,
            "market_ticker": market_ticker,
            "event_time_utc": event_time_utc,
            "action": action,
            "side": side,
            "price": price,
            "qty": qty,
            "order_id": order_id,
            "status": status,
            "error": error,
            "raw_response": raw_response,
        }
        with open(self.jsonl_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _find_btc_markets(self) -> List[Dict[str, Any]]:
        """
        Find BTC markets for the current UTC day.

        Returns:
            List of market dictionaries matching BTC criteria.
        """
        try:
            # Fetch markets (increase limit to get more results)
            response = self.client.list_markets(limit=100)
            markets = response.get("markets", [])

            # Filter for BTC markets
            btc_markets = []
            current_utc = datetime.utcnow()
            today_str = current_utc.strftime("%Y-%m-%d")

            for market in markets:
                ticker = market.get("ticker", "")
                title = market.get("title", "").lower()
                status = market.get("status", "").lower()

                # Filter criteria:
                # - Ticker or title contains "BTC" or "bitcoin"
                # - Status is "open" or "active"
                is_btc = "btc" in ticker.lower() or "bitcoin" in title or "btc" in title
                is_open = status in ["open", "active", "trading"]

                if is_btc and is_open:
                    # If event_time is present, keep only markets for today
                    event_time_str = market.get("event_time", "")
                    if event_time_str:
                        # Parse event_time and check if it's today
                        try:
                            event_time = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
                            if event_time.strftime("%Y-%m-%d") == today_str:
                                btc_markets.append(market)
                        except Exception:
                            # If parsing fails, include it anyway (will be deprioritized)
                            btc_markets.append(market)
                    else:
                        # No event_time, include it (will be deprioritized)
                        btc_markets.append(market)

            return btc_markets

        except Exception as e:
            self.logger.error(f"Error fetching BTC markets: {e}", exc_info=True)
            return []

    def _get_next_expiring_market(self, markets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find the next expiring BTC market.

        Args:
            markets: List of market dictionaries.

        Returns:
            Market dictionary for the next expiring market, or None.
        """
        if not markets:
            return None

        current_utc = datetime.utcnow()
        today_end = current_utc.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Filter unprocessed markets with valid event_time
        valid_markets = []
        for market in markets:
            ticker = market.get("ticker", "")
            if ticker in self.processed_markets:
                continue

            event_time_str = market.get("event_time", "")
            if event_time_str:
                try:
                    event_time = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
                    # Keep markets with event_time >= now and <= end of today
                    if current_utc <= event_time <= today_end:
                        valid_markets.append((market, event_time))
                except Exception:
                    # If parsing fails, skip this market (will use fallback)
                    pass

        # If we have markets with valid event_time, return the one with smallest event_time (soonest expiry)
        if valid_markets:
            valid_markets.sort(key=lambda x: x[1])  # Sort by event_time
            return valid_markets[0][0]

        # Fallback: return first unprocessed market (no event_time or parsing failed)
        for market in markets:
            ticker = market.get("ticker", "")
            if ticker not in self.processed_markets:
                return market

        return None

    def _get_market_price(self, ticker: str) -> Optional[int]:
        """
        Get the best ask price for a market.

        Args:
            ticker: Market ticker symbol.

        Returns:
            Price in cents (0-100), or None if unavailable.
        """
        try:
            # Try to get orderbook first
            try:
                orderbook = self.client.get_orderbook(ticker)
                asks = orderbook.get("asks", [])
                if asks:
                    # Get best ask (lowest price)
                    best_ask = min(asks, key=lambda x: x.get("price", 100))
                    return best_ask.get("price")
            except Exception:
                pass

            # Fallback: get market details
            market = self.client.get_market(ticker)
            # Try to extract price from market data
            yes_bid = market.get("yes_bid", None)
            yes_ask = market.get("yes_ask", None)
            if yes_ask is not None:
                return int(yes_ask * 100)  # Convert from 0-1 to 0-100 cents
            elif yes_bid is not None:
                # Use bid + small buffer
                return min(100, int((yes_bid + 0.02) * 100))

            # Try mid price
            yes_price = market.get("yes_price", None)
            if yes_price is not None:
                return int(yes_price * 100)

            return None

        except Exception as e:
            self.logger.warning(f"Error getting price for {ticker}: {e}")
            return None

    def _place_order_attempt(self, market: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to place an order for a market.

        Args:
            market: Market dictionary.

        Returns:
            Dictionary with order result including order_id, status, error.
        """
        ticker = market.get("ticker", "UNKNOWN")
        event_time = market.get("event_time", "")

        timestamp_utc = datetime.utcnow().isoformat() + "Z"

        # Get market price
        price = self._get_market_price(ticker)

        if price is None:
            error_msg = "Price unavailable"
            self.logger.warning(f"Skipping {ticker}: {error_msg}")
            self._log_trade(
                timestamp_utc=timestamp_utc,
                market_ticker=ticker,
                event_time_utc=event_time,
                action="SKIP",
                side="",
                price=None,
                qty=self.qty,
                order_id=None,
                status=None,
                error=error_msg,
            )
            return {"action": "SKIP", "error": error_msg}

        # Paper mode: don't place orders
        if self.mode == "paper":
            self.logger.info(f"[PAPER] Would place BUY YES order: {ticker}, qty={self.qty}, price={price}")
            self._log_trade(
                timestamp_utc=timestamp_utc,
                market_ticker=ticker,
                event_time_utc=event_time,
                action="PAPER_BUY",
                side="yes",
                price=price,
                qty=self.qty,
                order_id=None,
                status="paper_mode",
                error=None,
            )
            return {"action": "PAPER_BUY", "status": "paper_mode"}

        # Live mode: place order
        try:
            self.logger.info(f"Placing BUY YES order: {ticker}, qty={self.qty}, price={price}")
            response = self.client.place_order(
                ticker=ticker,
                side="yes",
                action="buy",
                count=self.qty,
                price=price,
                order_type="limit",
            )

            order_id = response.get("order_id") or response.get("id")
            status = response.get("status") or response.get("order_status", "unknown")

            self.logger.info(f"Order placed: order_id={order_id}, status={status}")

            self._log_trade(
                timestamp_utc=timestamp_utc,
                market_ticker=ticker,
                event_time_utc=event_time,
                action="BUY",
                side="yes",
                price=price,
                qty=self.qty,
                order_id=str(order_id) if order_id else None,
                status=status,
                error=None,
                raw_response=response,
            )

            return {"action": "BUY", "order_id": order_id, "status": status, "response": response}

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error placing order for {ticker}: {error_msg}", exc_info=True)

            self._log_trade(
                timestamp_utc=timestamp_utc,
                market_ticker=ticker,
                event_time_utc=event_time,
                action="ERROR",
                side="yes",
                price=price,
                qty=self.qty,
                order_id=None,
                status=None,
                error=error_msg,
            )

            return {"action": "ERROR", "error": error_msg}

    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading window (09:00-24:00 UTC)."""
        current_utc = datetime.utcnow()
        hour = current_utc.hour
        return self.trading_start_hour <= hour < self.trading_end_hour

    def _log_heartbeat(
        self,
        action: str,
        status: str,
        error: str,
        market_ticker: str = "",
    ) -> None:
        """
        Log a heartbeat/wait message with throttling (at most once per minute).

        Args:
            action: Action type ("WAIT" or "SKIP").
            status: Status code.
            error: Error/reason message.
            market_ticker: Optional market ticker (empty for wait messages).
        """
        current_utc = datetime.utcnow()
        current_minute = current_utc.minute

        # Throttle: only log if minute changed
        if self.last_heartbeat_minute == current_minute:
            return

        self.last_heartbeat_minute = current_minute
        timestamp_utc = current_utc.isoformat() + "Z"

        self._log_trade(
            timestamp_utc=timestamp_utc,
            market_ticker=market_ticker,
            event_time_utc=None,
            action=action,
            side="",
            price=None,
            qty=self.qty,
            order_id=None,
            status=status,
            error=error,
        )

    def run(self) -> None:
        """Run the live trader main loop."""
        self.logger.info("=" * 60)
        self.logger.info(f"Starting Live Trader (mode={self.mode}, qty={self.qty})")
        self.logger.info(f"Trading window: {self.trading_start_hour}:00-{self.trading_end_hour}:00 UTC")
        self.logger.info(f"Poll interval: {self.poll_interval_seconds}s")
        self.logger.info(f"Logs: {self.logs_dir}")
        self.logger.info("=" * 60)

        cycles_completed = 0
        start_time = datetime.utcnow()

        try:
            while True:
                # Check max_hours limit
                if self.max_hours is not None:
                    elapsed_hours = (datetime.utcnow() - start_time).total_seconds() / 3600
                    if elapsed_hours >= self.max_hours:
                        self.logger.info(f"Reached max_hours limit ({self.max_hours}), stopping.")
                        break

                # Check if within trading hours
                if not self._is_trading_hours():
                    self.logger.debug("Outside trading hours, waiting...")
                    self._log_heartbeat(
                        action="WAIT",
                        status="OUTSIDE_TRADING_HOURS",
                        error=f"Outside trading window ({self.trading_start_hour}:00-{self.trading_end_hour}:00 UTC)",
                    )
                    time.sleep(self.poll_interval_seconds)
                    continue

                try:
                    # Check if we should place an order this hour (once per hour)
                    current_utc = datetime.utcnow()
                    current_hour = current_utc.hour
                    
                    # Only place orders once per hour
                    if self.last_order_hour == current_hour:
                        self.logger.debug(f"Already placed order this hour ({current_hour}:00 UTC), waiting...")
                        self._log_heartbeat(
                            action="WAIT",
                            status="ALREADY_TRADED_THIS_HOUR",
                            error=f"Already placed order at {current_hour}:00 UTC",
                        )
                        time.sleep(self.poll_interval_seconds)
                        continue
                    
                    # Find BTC markets
                    markets = self._find_btc_markets()
                    self.logger.debug(f"Found {len(markets)} BTC markets")

                    if not markets:
                        self.logger.debug("No BTC markets found, waiting...")
                        self._log_heartbeat(
                            action="SKIP",
                            status="NO_MARKETS_FOUND",
                            error="No BTC markets found matching criteria",
                        )
                        time.sleep(self.poll_interval_seconds)
                        continue

                    # Find next expiring market
                    market = self._get_next_expiring_market(markets)
                    if market is None:
                        self.logger.debug("No unprocessed markets found, waiting...")
                        self._log_heartbeat(
                            action="SKIP",
                            status="NO_UNPROCESSED_MARKET",
                            error="All markets already processed",
                        )
                        time.sleep(self.poll_interval_seconds)
                        continue

                    ticker = market.get("ticker", "UNKNOWN")
                    self.logger.info(f"Processing market: {ticker} at {current_hour}:00 UTC")

                    # Place order attempt
                    result = self._place_order_attempt(market)
                    self.processed_markets.add(ticker)
                    
                    # Update last order hour (only if order was attempted, not skipped)
                    if result.get("action") in ["BUY", "PAPER_BUY"]:
                        self.last_order_hour = current_hour

                    cycles_completed += 1
                    self.logger.info(f"Cycle {cycles_completed} completed. Result: {result.get('action', 'UNKNOWN')}")

                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {e}", exc_info=True)
                    # Continue running despite errors

                # Wait before next cycle
                time.sleep(self.poll_interval_seconds)

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down gracefully...")
        except Exception as e:
            self.logger.error(f"Fatal error in trader: {e}", exc_info=True)
            raise
        finally:
            self.logger.info(f"Trader stopped. Completed {cycles_completed} cycles.")
            self.logger.info(f"Logs saved to: {self.logs_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live Kalshi demo trading")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "paper"],
        default="paper",
        help="Trading mode: 'live' places real orders, 'paper' only logs decisions",
    )
    parser.add_argument(
        "--qty",
        type=int,
        default=1,
        help="Quantity of contracts per order (default: 1)",
    )
    parser.add_argument(
        "--max_hours",
        type=float,
        default=None,
        help="Maximum number of trading hours to run (for testing, can be fractional)",
    )
    parser.add_argument(
        "--poll_interval_seconds",
        type=int,
        default=30,
        help="Seconds between polling cycles (default: 30)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run mode (same as --mode paper)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger()
    logger = get_logger(__name__)

    # Handle dry_run flag
    mode = "paper" if args.dry_run else args.mode

    # Warn if live mode
    if mode == "live":
        logger.warning("=" * 60)
        logger.warning("LIVE MODE: Real orders will be placed!")
        logger.warning("=" * 60)
        response = input("Type 'yes' to confirm: ")
        if response.lower() != "yes":
            logger.info("Aborted by user.")
            sys.exit(0)

    # Create and run trader
    trader = LiveTrader(
        mode=mode,
        qty=args.qty,
        max_hours=args.max_hours,
        poll_interval_seconds=args.poll_interval_seconds,
    )

    trader.run()


if __name__ == "__main__":
    main()

