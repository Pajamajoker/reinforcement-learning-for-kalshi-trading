# Kalshi RL Trading Agent

A reinforcement learning trading agent for Kalshi prediction markets.

## Project Structure

```
.
├── env/              # RL environment + market simulator
├── agent/            # Policy models and training code
├── backtest/         # Backtest runner + metrics
├── live_trading/     # Kalshi demo trading client + scheduler
├── gui/              # Frontend application
├── scripts/          # Entrypoint scripts
├── configs/          # Configuration files
├── data/             # Local historical data (git-ignored)
├── logs/             # Log files (git-ignored)
├── tests/            # Test suite
└── src/              # Source code package
    └── common/       # Shared utilities (config, logger, paths)
```

## Installation

1. **Clone the repository** (if applicable)

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and fill in your Kalshi API credentials:
   ```
   KALSHI_API_KEY_ID=your_api_key_id
   KALSHI_PRIVATE_KEY_PATH=path/to/your/private_key.pem
   KALSHI_BASE_URL=https://demo-api.kalshi.co/trade-api/v2
   ```

## Configuration

Edit `configs/config.yaml` to customize:
- Trading window hours
- Risk management limits
- Logging settings
- Agent hyperparameters
- Backtest parameters

## Running

### Smoke Test

Verify the project setup is working:
```bash
python scripts/smoke_test.py
```

This will:
- Load configuration from `configs/config.yaml` and `.env`
- Set up logging
- Print a configuration summary
- Verify API credentials are set (warns if missing)

### Future Entrypoints

Once implemented, you'll be able to run:
- **Training:** `python scripts/train.py`
- **Backtesting:** `python scripts/backtest.py`
- **Live Trading:** `python scripts/live_trade.py`
- **GUI:** `streamlit run gui/app.py`

## API Keys

**Important:** Never commit your `.env` file to version control.

1. Copy `.env.example` to `.env`
2. Fill in your Kalshi API credentials:
   - `KALSHI_API_KEY_ID`: Your Kalshi API key ID
   - `KALSHI_PRIVATE_KEY_PATH`: Path to your private key file (PEM format)
   - `KALSHI_BASE_URL`: API endpoint (default: demo API)

For production trading, use the production API URL instead of the demo URL.

## Development

### Code Formatting

This project uses `black` and `ruff` for code formatting and linting. Configuration is in `pyproject.toml`.

Format code:
```bash
black src/ scripts/
```

Lint code:
```bash
ruff check src/ scripts/
```

## License

[Add your license here]


