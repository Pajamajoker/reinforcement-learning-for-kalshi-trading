# Kalshi RL Trading Agent

A reinforcement learning trading agent for Kalshi prediction markets, focusing on hourly BTC threshold markets. The agent uses Deep Q-Networks (DQN) to learn trading strategies through backtesting on historical data.

## Project Overview

This project simulates Kalshi-style hourly BTC threshold markets (e.g., "Will BTC > $90,000 at 14:00 UTC?") and trains an RL agent to trade these markets. The system includes:

- **Market Simulation**: Hourly events (09:00-24:00 UTC) with decision times, threshold calculation, and contract resolution
- **RL Environment**: Gymnasium-compatible environment for training DQN agents
- **Backtesting Framework**: Multi-day backtesting with baseline and RL policies
- **Policy Evaluation**: Compare random, baseline, and DQN policies with detailed metrics
- **GUI Dashboard**: Streamlit interface for visualizing results

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Keys (Optional)

For Kalshi API access (optional, not required for backtesting):

```bash
cp .env.example .env
# Edit .env and add your Kalshi API credentials
```

### 3. Fetch Data

Download recent BTC data for backtesting:

```bash
python data/fetch_yfinance_btc.py
# Output: data/btc_1m_last7d.csv
```

### 4. Run End-to-End Pipeline

Train and evaluate the agent:

```bash
python scripts/run_offline_pipeline.py \
  --data_path data/btc_1m_last7d.csv \
  --start 2025-12-14 \
  --end 2025-12-20 \
  --total_steps 20000
```

This will:
- Run baseline backtest
- Train DQN agent (with low friction by default)
- Evaluate both policies (with realistic friction)
- Save all results to `backtest/results/`

### 5. View Results

Launch the GUI dashboard:

```bash
streamlit run gui/app.py
```

## Project Structure

```
.
├── env/              # Market simulator, pricing, execution, portfolio
├── agent/            # DQN agent implementation
├── backtest/         # Backtest runner, data loader, metrics
├── live_trading/     # Kalshi API client (demo)
├── gui/              # Streamlit dashboard
├── scripts/          # Main entrypoints
├── configs/          # Configuration (YAML)
├── data/             # Historical data (git-ignored)
├── models/           # Trained model checkpoints
├── logs/             # Log files
└── tests/            # Test suite
```

## Key Scripts

### Data Fetching

```bash
# Download BTC data from Yahoo Finance
python data/fetch_yfinance_btc.py [--days 7] [--out data/btc_1m_last7d.csv]
```

### Training

```bash
# Train DQN agent
python scripts/train_dqn.py \
  --data_path data/btc_1m_last7d.csv \
  --start 2025-12-14 \
  --end 2025-12-20 \
  --total_steps 20000 \
  --friction_mode low_friction  # or realistic
```

### Evaluation

```bash
# Evaluate a policy
python scripts/eval_policy.py \
  --policy dqn \
  --checkpoint models/dqn_last.pt \
  --data_path data/btc_1m_last7d.csv \
  --start 2025-12-14 \
  --end 2025-12-20 \
  --friction_mode realistic
```

Available policies: `random`, `baseline`, `dqn`

### Backtesting

```bash
# Run multi-day backtest
python backtest/run_backtest.py \
  --data_path data/btc_1m_last7d.csv \
  --start 2025-12-14 \
  --end 2025-12-20 \
  --policy baseline
```

## Configuration

Edit `configs/config.yaml` to customize:
- Trading parameters (spread, fees, turnover penalty)
- Observation window size
- Starting capital
- Strategy thresholds
- Logging settings

## Friction Modes

The system supports two friction modes for training and evaluation:

- **`realistic`**: Uses config values (spread=0.02, fee=0.01, turnover_penalty=0.001)
- **`low_friction`**: Zero friction (spread=0, fee=0, turnover_penalty=0)

Default: Training uses `low_friction`, evaluation uses `realistic`.

## Output Files

All results are saved to `backtest/results/`:

- `eval_metrics.csv` - Summary metrics per policy
- `eval_<policy>_equity_curve.csv` - Equity over time
- `eval_<policy>_daily_metrics.csv` - Per-day metrics (trades, wins, losses, action distribution)
- `eval_<policy>_equity_curve.png` - Equity curve plots
- `logs/dqn_train_metrics.csv` - Training metrics

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Key tests:
- `test_data_loader.py` - Data loading and day slicing
- `test_day_sim_smoke.py` - Single-day simulation
- `test_env_gym_smoke.py` - Gym environment
- `test_eval_policy_smoke.py` - Policy evaluation
- `test_dqn_training_smoke.py` - DQN training

## GUI Dashboard

The Streamlit dashboard (`gui/app.py`) displays:

- **Summary Metrics**: Net P&L, total trades, trades/day, excess return vs baseline, win rate, Sharpe ratio
- **Equity Curves**: Side-by-side comparison of baseline and DQN
- **Daily Metrics**: Per-day performance with date filtering
- **Training Curves**: Episode return, epsilon, loss over training

## Development

### Code Quality

```bash
# Format code
black src/ scripts/ env/ agent/ backtest/

# Lint code
ruff check src/ scripts/ env/ agent/ backtest/
```

### Smoke Test

Verify setup:

```bash
python scripts/smoke_test.py
```

### Kalshi API Test (Optional)

Test Kalshi API connection:

```bash
python scripts/test_kalshi_connection.py
```

## Notes

- Historical data is stored in `data/` (git-ignored)
- Model checkpoints are saved to `models/`
- All backtest results go to `backtest/results/`
- The system uses timezone-aware UTC timestamps throughout
- Synthetic data is available in `data/btc_synthetic.csv` for testing

## License

[Add your license here]
