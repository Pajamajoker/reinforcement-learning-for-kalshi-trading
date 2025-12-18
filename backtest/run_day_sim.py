"""
Run a single day backtest simulation.

Simulates trading on hourly BTC threshold markets for a given day,
applying a simple baseline strategy and tracking P&L.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
from rich.console import Console
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.data_loader import load_candles, get_day_slice
from env.execution import execute_buy_yes, quote_from_mid
from env.market_simulator import generate_hourly_events, resolve_event, HourlyEvent
from env.pricing import implied_prob
from env.portfolio import Portfolio
from src.common.config import get_config
from src.common.logger import setup_logger, get_logger


def run_day_simulation(data_path: str, day_utc: str) -> None:
    """
    Run a complete day simulation.

    Args:
        data_path: Path to CSV file with candle data.
        day_utc: Day in format "YYYY-MM-DD" (UTC).
    """
    # Setup
    setup_logger()
    logger = get_logger(__name__)
    console = Console()

    # Load config
    config = get_config()
    backtest_config = config.backtest

    console.print(f"\n[bold]Running day simulation for {day_utc}[/bold]\n")

    # Load candles
    logger.info(f"Loading candles from {data_path}")
    df = load_candles(data_path)

    # Get day slice
    day_df = get_day_slice(df, day_utc)
    if day_df.empty:
        console.print(f"[red]No data available for {day_utc}[/red]")
        sys.exit(1)

    # Generate events
    logger.info("Generating hourly events")
    events = generate_hourly_events(day_df, day_utc)
    if not events:
        console.print(f"[red]No events generated for {day_utc}[/red]")
        sys.exit(1)

    # Initialize portfolio
    portfolio = Portfolio(cash=backtest_config.initial_capital)

    # Track trades
    trades = []

    # Process each event at decision time
    logger.info("Processing events at decision time")
    for event in events:
        try:
            # Compute implied probability
            prob = implied_prob(
                day_df,
                event.decision_time,
                event.expiry_time,
                event.threshold,
                lookback_minutes=backtest_config.lookback_minutes,
            )

            # Update event with probability
            event.mid_price = prob

            # Generate bid/ask from mid
            bid, ask = quote_from_mid(prob, backtest_config.spread)
            event.bid = bid
            event.ask = ask

            # Apply strategy
            action = None
            quantity = 0
            fill_price = 0.0
            cost = 0.0
            fees = 0.0

            if prob > backtest_config.strategy.buy_threshold:
                # Buy YES
                quantity = 1
                fill_price = ask
                cost, fees = execute_buy_yes(ask, quantity, backtest_config.fee_per_contract)
                action = "BUY"
                portfolio.add_position(event.event_id, quantity, fill_price, cost)

            elif prob < backtest_config.strategy.sell_threshold:
                # Sell YES (currently disabled, but structure is here)
                # For now, do nothing
                action = "HOLD"

            else:
                action = "HOLD"

            # Record trade
            if action != "HOLD":
                trades.append(
                    {
                        "time": event.decision_time,
                        "event_id": event.event_id,
                        "threshold": event.threshold,
                        "prob": prob,
                        "action": action,
                        "quantity": quantity,
                        "fill_price": fill_price,
                        "cost": cost,
                        "fees": fees,
                    }
                )

                logger.info(
                    f"Trade: {action} {quantity} @ {fill_price:.4f} for {event.event_id}, "
                    f"prob={prob:.4f}, cost=${cost:.2f}"
                )

        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}", exc_info=True)
            continue

    # Resolve all events at expiry
    logger.info("Resolving events at expiry")
    for event in events:
        try:
            resolve_event(event, day_df)

            # Resolve position if we have one
            if event.event_id in portfolio.positions:
                position = portfolio.positions[event.event_id]
                if position.quantity != 0 and event.resolved:
                    realized_pnl = portfolio.resolve_position(event.event_id, event.outcome)
                    logger.info(
                        f"Resolved {event.event_id}: outcome={event.outcome}, "
                        f"realized_pnl=${realized_pnl:.2f}"
                    )

        except Exception as e:
            logger.error(f"Error resolving event {event.event_id}: {e}", exc_info=True)
            continue

    # Calculate final equity
    # For mark-to-market, use final mid prices (or 0/1 if resolved)
    event_mid_prices = {}
    for event in events:
        if event.resolved:
            # Resolved events have outcome as mid price
            event_mid_prices[event.event_id] = float(event.outcome)
        else:
            event_mid_prices[event.event_id] = event.mid_price

    total_equity = portfolio.get_total_equity(event_mid_prices)
    realized_pnl = portfolio.get_total_realized_pnl()

    # Print results
    console.print("\n[bold]Trade Log[/bold]\n")
    if trades:
        trade_table = Table(show_header=True, header_style="bold cyan")
        trade_table.add_column("Time", style="cyan")
        trade_table.add_column("Event ID", style="white")
        trade_table.add_column("Threshold", style="yellow", justify="right")
        trade_table.add_column("Prob", style="green", justify="right")
        trade_table.add_column("Action", style="magenta")
        trade_table.add_column("Fill Price", style="blue", justify="right")
        trade_table.add_column("Cost", style="red", justify="right")
        trade_table.add_column("Fees", style="red", justify="right")

        for trade in trades:
            trade_table.add_row(
                str(trade["time"]),
                trade["event_id"],
                f"${trade['threshold']:.2f}",
                f"{trade['prob']:.4f}",
                trade["action"],
                f"{trade['fill_price']:.4f}",
                f"${trade['cost']:.2f}",
                f"${trade['fees']:.2f}",
            )

        console.print(trade_table)
    else:
        console.print("[yellow]No trades executed[/yellow]\n")

    # Print final summary
    console.print("\n[bold]Final Summary[/bold]\n")
    summary_table = Table(show_header=True, header_style="bold green")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Initial Cash", f"${portfolio.initial_cash:.2f}")
    summary_table.add_row("Final Cash", f"${portfolio.cash:.2f}")
    summary_table.add_row("Realized P&L", f"${realized_pnl:.2f}")
    summary_table.add_row("Total Equity", f"${total_equity:.2f}")
    summary_table.add_row("Total Return", f"${(total_equity - portfolio.initial_cash):.2f}")
    summary_table.add_row(
        "Return %", f"{((total_equity / portfolio.initial_cash) - 1) * 100:.2f}%"
    )

    console.print(summary_table)
    console.print()

    logger.info(
        f"Simulation complete: initial=${portfolio.initial_cash:.2f}, "
        f"final=${total_equity:.2f}, return={((total_equity / portfolio.initial_cash) - 1) * 100:.2f}%"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run a single day backtest simulation")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to CSV file with candle data",
        default=None,
    )
    parser.add_argument(
        "--day",
        type=str,
        required=True,
        help="Day in format YYYY-MM-DD (UTC)",
    )

    args = parser.parse_args()

    # Get data path from config if not provided
    config = get_config()
    data_path = args.data_path or config.backtest.data_path

    if not data_path:
        print("Error: data_path must be provided via --data_path or config")
        sys.exit(1)

    run_day_simulation(data_path, args.day)


if __name__ == "__main__":
    main()


