# Kalshi RL Trading Agent

A reinforcement learning trading agent for Kalshi prediction markets, focusing on hourly Bitcoin price threshold events. The agent uses Deep Q-Networks (DQN) to learn trading strategies through backtesting on historical data and deploys to Kalshi's demo environment for live trading.

## Project Overview

This project implements a complete RL trading system for Kalshi-style hourly BTC threshold markets (e.g., "Will BTC > $90,000 at 14:00 UTC?"). The system includes:

- **Market Simulation**: Hourly events (09:00-24:00 UTC) with decision times, threshold calculation, and contract resolution
- **RL Environment**: Gymnasium-compatible environment for training DQN agents
- **Backtesting Framework**: Multi-day backtesting with baseline and RL policies
- **Policy Evaluation**: Compare random, baseline, and DQN policies with detailed metrics
- **Live Trading**: Integration with Kalshi demo API for real-time trading
- **GUI Dashboard**: Streamlit interface for visualizing results and live trading activity

## Installation

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager

### 2. Setup

```bash
# Clone or extract the project
cd "Kalshi Trading Agent"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

#### Kalshi API Setup (for live trading)

Create a `.env` file in the project root:

```bash
# Copy example (if available)
cp .env.example .env

# Edit .env and add your Kalshi demo API credentials:
KALSHI_API_KEY_ID=your_api_key_id
KALSHI_PRIVATE_KEY_PATH=path/to/private_key.pem
KALSHI_BASE_URL=https://demo-api.kalshi.co/trade-api/v2
```

**Note**: Only use Kalshi's demo environment. Do not use real money trading.

#### Data Setup

Download historical Bitcoin data for backtesting:

```bash
python data/fetch_yfinance_btc.py
# Output: data/btc_1m_last7d.csv
```

## Usage

### Backtesting & Training

Run the complete offline pipeline to train and evaluate the agent:

```bash
python scripts/run_offline_pipeline.py \
  --data_path data/btc_1m_last7d.csv \
  --start 2025-12-14 \
  --end 2025-12-20 \
  --total_steps 20000
```

This pipeline:
1. Runs baseline backtest (simple threshold strategy)
2. Trains DQN agent on historical data
3. Evaluates both policies with realistic market friction
4. Saves all results to `backtest/results/`

### Individual Components

#### Train DQN Agent

```bash
python scripts/train_dqn.py \
  --data_path data/btc_1m_last7d.csv \
  --start 2025-12-14 \
  --end 2025-12-20 \
  --total_steps 20000 \
  --friction_mode low_friction
```

#### Evaluate Policy

```bash
# Evaluate baseline policy
python scripts/eval_policy.py \
  --policy baseline \
  --data_path data/btc_1m_last7d.csv \
  --start 2025-12-14 \
  --end 2025-12-20 \
  --friction_mode realistic

# Evaluate DQN policy
python scripts/eval_policy.py \
  --policy dqn \
  --checkpoint models/dqn_last.pt \
  --data_path data/btc_1m_last7d.csv \
  --start 2025-12-14 \
  --end 2025-12-20 \
  --friction_mode realistic
```

Available policies: `random`, `baseline`, `dqn`

### Live Trading

Connect your trained agent to Kalshi demo API:

```bash
# Test connection first
python scripts/test_kalshi_connection.py

# Run in paper mode (logs decisions without placing orders)
python -m live_trading.live_trader --mode paper --qty 1

# Run in live mode (places real orders on demo account)
python -m live_trading.live_trader --mode live --qty 1
```

**Options:**
- `--mode live|paper`: Paper mode logs decisions without placing orders
- `--qty N`: Quantity of contracts per order (default: 1)
- `--max_hours N`: Run for N hours then stop (for testing)
- `--poll_interval_seconds N`: Seconds between polling cycles (default: 30)

**Trading Window**: The agent trades hourly BTC threshold markets from 09:00 to 24:00 UTC.

**Logs:**
- `live_trading/logs/kalshi_live_trades.csv` - Trade attempts log
- `live_trading/logs/kalshi_live_trades.jsonl` - Full API responses

### GUI Dashboard

Launch the Streamlit dashboard to visualize results:

```bash
streamlit run gui/app.py
```

Or use the helper script:

```bash
python scripts/start_dashboard.py
```

The dashboard displays:
- **Backtest Results**: Summary metrics, equity curves, daily metrics, training curves
- **Live Trading**: Real-time P&L, trade history, activity timeline, status breakdown
- **Training Metrics**: Episode returns, epsilon decay, loss curves

Access at: `http://localhost:8501`

## Project Structure

```
.
├── env/              # Market simulator, pricing, execution, portfolio
├── agent/            # DQN agent implementation
├── backtest/         # Backtest runner, data loader, metrics
├── live_trading/     # Kalshi API client and live trader
├── gui/              # Streamlit dashboard
├── scripts/          # Main entrypoints and utilities
├── configs/          # Configuration (YAML)
├── src/              # Common utilities (config, logger, paths)
├── data/             # Historical data (git-ignored)
├── models/           # Trained model checkpoints
├── logs/             # Log files
└── tests/            # Test suite
```

## Configuration

Edit `configs/config.yaml` to customize:

- **Trading parameters**: Spread, fees, turnover penalty
- **Observation window**: Number of minutes in state representation
- **Starting capital**: Initial portfolio value (default: $10,000)
- **RL hyperparameters**: Learning rate, gamma, epsilon decay, etc.
- **Logging settings**: Log level, file paths, rotation

## RL Formulation

### State Space
- Recent Bitcoin price history (log returns over observation window)
- Time-to-expiry in minutes (normalized)
- Current mid price (implied probability)
- Current position size for the event
- Cash fraction (cash / starting capital)

### Action Space
- **0**: Do nothing (HOLD)
- **1**: Buy YES
- **2**: Sell YES

### Reward
- Base reward: Change in portfolio equity (P&L)
- Regularization: Turnover penalty per contract traded

### Algorithm
- **DQN** (Deep Q-Network) with:
  - Experience replay buffer
  - Target network for stable learning
  - Epsilon-greedy exploration
  - Adam optimizer

## Friction Modes

The system supports two friction modes:

- **`realistic`**: Uses config values (spread=0.02, fee=0.01, turnover_penalty=0.001)
- **`low_friction`**: Zero friction (spread=0, fee=0, turnover_penalty=0)

**Default**: Training uses `low_friction`, evaluation uses `realistic`.

## Output Files

### Backtest Results (`backtest/results/`)
- `eval_metrics.csv` - Summary metrics per policy
- `eval_<policy>_equity_curve.csv` - Equity over time
- `eval_<policy>_daily_metrics.csv` - Per-day metrics (trades, wins, losses)
- `eval_<policy>_equity_curve.png` - Equity curve plots

### Training Metrics (`logs/`)
- `dqn_train_metrics.csv` - Training metrics (episode return, epsilon, loss)

### Live Trading Logs (`live_trading/logs/`)
- `kalshi_live_trades.csv` - Trade attempts log
- `kalshi_live_trades.jsonl` - Full API responses

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
- `test_live_trader_smoke.py` - Live trading integration

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

## Notes

- Historical data is stored in `data/` (git-ignored)
- Model checkpoints are saved to `models/`
- All backtest results go to `backtest/results/`
- The system uses timezone-aware UTC timestamps throughout
- **Demo only**: Only use Kalshi's demo environment. No real money trading.

## Troubleshooting

### Kalshi API Connection Issues

1. Verify API credentials in `.env` file
2. Check private key file path is correct
3. Test connection: `python scripts/test_kalshi_connection.py`
4. Ensure you're using demo API URL

### Data Loading Issues

1. Verify data file exists and has correct format
2. Check date range matches available data
3. Ensure timestamps are in UTC

### Training Issues

1. Check GPU availability (optional, CPU works)
2. Verify data path and date range
3. Check config.yaml for hyperparameters

## License

This project is for educational purposes only.
