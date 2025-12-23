"""
Export summary of live trading logs.

Reads kalshi_live_trades.csv and generates a summary report.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logger import setup_logger, get_logger


def export_summary(csv_path: Path, output_path: Path) -> None:
    """
    Read live trading CSV and generate summary.

    Args:
        csv_path: Path to kalshi_live_trades.csv
        output_path: Path to write live_summary.csv
    """
    logger = get_logger(__name__)

    if not csv_path.exists():
        logger.error(f"CSV log file not found: {csv_path}")
        sys.exit(1)

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        sys.exit(1)

    if df.empty:
        logger.warning("CSV file is empty")
        print("No trading records found.")
        return

    # Calculate summary statistics
    total_records = len(df)
    orders_attempted = len(df[df["action"].isin(["BUY", "PAPER_BUY"])])
    orders_placed = len(df[df["action"] == "BUY"])
    orders_skipped = len(df[df["action"] == "SKIP"])
    orders_error = len(df[df["action"] == "ERROR"])

    # Status counts
    status_counts = df["status"].value_counts().to_dict() if "status" in df.columns else {}
    filled_count = status_counts.get("filled", 0) + status_counts.get("Filled", 0)
    open_count = status_counts.get("open", 0) + status_counts.get("Open", 0)
    rejected_count = status_counts.get("rejected", 0) + status_counts.get("Rejected", 0)

    # Unique tickers
    unique_tickers = df["market_ticker"].nunique() if "market_ticker" in df.columns else 0

    # Date range
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
        date_range_start = df["timestamp_utc"].min()
        date_range_end = df["timestamp_utc"].max()
    else:
        date_range_start = None
        date_range_end = None

    # Print summary
    print("=" * 60)
    print("LIVE TRADING SUMMARY")
    print("=" * 60)
    print(f"Total records: {total_records}")
    print(f"Orders attempted: {orders_attempted}")
    print(f"Orders placed (live): {orders_placed}")
    print(f"Orders skipped: {orders_skipped}")
    print(f"Orders error: {orders_error}")
    print()
    print("Order Status:")
    print(f"  Filled: {filled_count}")
    print(f"  Open: {open_count}")
    print(f"  Rejected: {rejected_count}")
    print()
    print(f"Unique tickers traded: {unique_tickers}")
    if date_range_start and date_range_end:
        print(f"Date range: {date_range_start} to {date_range_end}")
    print("=" * 60)

    # Create summary DataFrame
    summary_data = {
        "metric": [
            "total_records",
            "orders_attempted",
            "orders_placed",
            "orders_skipped",
            "orders_error",
            "filled_count",
            "open_count",
            "rejected_count",
            "unique_tickers",
            "date_range_start",
            "date_range_end",
        ],
        "value": [
            total_records,
            orders_attempted,
            orders_placed,
            orders_skipped,
            orders_error,
            filled_count,
            open_count,
            rejected_count,
            unique_tickers,
            str(date_range_start) if date_range_start else "",
            str(date_range_end) if date_range_end else "",
        ],
    }

    summary_df = pd.DataFrame(summary_data)

    # Write summary CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Summary saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export live trading summary")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to kalshi_live_trades.csv (default: live_trading/logs/kalshi_live_trades.csv)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to write live_summary.csv (default: live_trading/logs/live_summary.csv)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger()
    logger = get_logger(__name__)

    # Resolve paths (Windows-safe)
    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        csv_path = Path(os.path.join(str(project_root), "live_trading", "logs", "kalshi_live_trades.csv"))

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(os.path.join(str(project_root), "live_trading", "logs", "live_summary.csv"))

    export_summary(csv_path, output_path)


if __name__ == "__main__":
    main()


