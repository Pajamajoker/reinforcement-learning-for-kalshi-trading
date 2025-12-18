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

### Part C2 – Evaluation

Evaluate different policies on historical data:

**Random policy:**
```bash
python scripts/eval_policy.py --policy random --data_path data/btc_synthetic.csv --start 2025-12-10 --end 2025-12-10
```

**Baseline policy:**
```bash
python scripts/eval_policy.py --policy baseline --data_path data/btc_synthetic.csv --start 2025-12-10 --end 2025-12-10
```

**DQN policy (requires trained checkpoint):**
```bash
python scripts/eval_policy.py --policy dqn --checkpoint models/dqn_last.pt --data_path data/btc_synthetic.csv --start 2025-12-10 --end 2025-12-10
```

Evaluation outputs:
- `backtest/results/eval_<policy>_equity_curve.csv` - Equity over time
- `backtest/results/eval_<policy>_daily_metrics.csv` - Per-day metrics
- `backtest/results/eval_<policy>_equity_curve.png` - Equity curve plot
- `backtest/results/eval_metrics.csv` - Summary metrics for all policies

### Training

Train a DQN agent:
```bash
python scripts/train_dqn.py --data_path data/btc_synthetic.csv --start 2025-12-10 --end 2025-12-10 --total_steps 2000
```

### Backtesting

Run multi-day backtest:
```bash
python backtest/run_backtest.py --data_path data/btc_synthetic.csv --start 2025-12-01 --end 2025-12-10 --policy baseline
```

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


