"""
Test script to verify Kalshi API connection and authentication.

This script:
- Instantiates a KalshiClient
- Calls get_account() to fetch account info
- Calls list_markets(limit=3) to fetch market data
- Prints results nicely
- Exits cleanly on success or fails loudly with helpful errors
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from live_trading.kalshi_client import KalshiClient
from src.common.logger import setup_logger, get_logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def print_account_info(account_data: dict, console: Console) -> None:
    """Print account information in a formatted table."""
    table = Table(title="Account Information", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    # Extract common account fields
    if "balance" in account_data:
        table.add_row("Balance", f"${account_data['balance']:,.2f}")
    if "equity" in account_data:
        table.add_row("Equity", f"${account_data['equity']:,.2f}")
    if "buying_power" in account_data:
        table.add_row("Buying Power", f"${account_data['buying_power']:,.2f}")
    if "portfolio_value" in account_data:
        table.add_row("Portfolio Value", f"${account_data['portfolio_value']:,.2f}")

    # Add any other fields
    for key, value in account_data.items():
        if key not in ["balance", "equity", "buying_power", "portfolio_value"]:
            table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


def print_markets(markets_data: dict, console: Console) -> None:
    """Print market information in a formatted table."""
    # Extract markets from response
    markets = markets_data.get("markets", [])
    if not markets:
        console.print("[yellow]No markets found in response[/yellow]")
        console.print(f"Full response: {markets_data}")
        return

    table = Table(title=f"Active Markets (showing {len(markets)} markets)", show_header=True, header_style="bold blue")
    table.add_column("Ticker", style="cyan")
    table.add_column("Title", style="white", max_width=50)
    table.add_column("Status", style="yellow")
    table.add_column("Category", style="magenta")

    for market in markets[:10]:  # Show up to 10 markets
        ticker = market.get("ticker", "N/A")
        title = market.get("title", "N/A")
        status = market.get("status", "N/A")
        category = market.get("category", "N/A")

        table.add_row(ticker, title, status, category)

    console.print(table)


def main():
    """Run the Kalshi connection test."""
    # Setup logging
    setup_logger()
    logger = get_logger(__name__)

    # Setup console for pretty printing
    console = Console()

    console.print(Panel.fit("[bold green]Kalshi API Connection Test[/bold green]", border_style="green"))

    try:
        # Initialize client
        logger.info("Initializing Kalshi client...")
        console.print("\n[cyan]Initializing Kalshi client...[/cyan]")
        client = KalshiClient()
        console.print("[green]Client initialized successfully[/green]\n")

        # Test 1: Get account info
        console.print(Panel("[bold]Test 1: Fetching Account Information[/bold]", border_style="blue"))
        try:
            account_data = client.get_account()
            console.print("[green]Account information retrieved successfully[/green]\n")
            print_account_info(account_data, console)
            console.print()
        except Exception as e:
            console.print(f"[red]Failed to fetch account info: {e}[/red]")
            logger.error(f"Failed to fetch account info: {e}", exc_info=True)
            raise

        # Test 2: List markets
        console.print(Panel("[bold]Test 2: Fetching Market List[/bold]", border_style="blue"))
        try:
            markets_data = client.list_markets(limit=3)
            console.print("[green]Market list retrieved successfully[/green]\n")
            print_markets(markets_data, console)
            console.print()
        except Exception as e:
            console.print(f"[red]Failed to fetch markets: {e}[/red]")
            logger.error(f"Failed to fetch markets: {e}", exc_info=True)
            raise

        # Verify we're on demo API
        if "demo-api" in client.base_url:
            console.print(Panel.fit("[bold green]Connected to Kalshi DEMO API[/bold green]", border_style="green"))
        else:
            console.print(
                Panel.fit(
                    "[bold yellow]WARNING: Connected to Kalshi API (not demo)[/bold yellow]",
                    border_style="yellow",
                )
            )

        console.print("\n[bold green]All tests passed! Connection verified.[/bold green]\n")
        logger.info("All connection tests passed successfully")

    except ValueError as e:
        # Configuration errors
        console.print(f"\n[bold red]Configuration Error:[/bold red] {e}")
        console.print("\n[yellow]Please check your .env file and ensure:")
        console.print("  - KALSHI_API_KEY_ID is set")
        console.print("  - KALSHI_PRIVATE_KEY_PATH points to a valid private key file")
        console.print("  - KALSHI_BASE_URL is set (defaults to demo API)\n")
        logger.error(f"Configuration error: {e}", exc_info=True)
        sys.exit(1)

    except Exception as e:
        # Other errors
        console.print(f"\n[bold red]Connection Failed:[/bold red] {e}")
        console.print("\n[yellow]Possible causes:")
        console.print("  - Invalid API credentials")
        console.print("  - Network connectivity issues")
        console.print("  - Kalshi API is down")
        console.print("  - Private key file is corrupted or incorrect\n")
        logger.error(f"Connection test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

