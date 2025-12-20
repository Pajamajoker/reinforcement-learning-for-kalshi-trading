"""
Streamlit GUI for visualizing Kalshi BTC Hourly RL Trader offline pipeline results.

Displays metrics, equity curves, daily metrics, and training curves from backtest results.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page config
st.set_page_config(
    page_title="Kalshi BTC Hourly RL Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("📈 Kalshi BTC Hourly RL Trader")
st.subheader("Offline Backtest + RL Training Results")


@st.cache_data(ttl=5)
def load_csv_safe(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Safely load a CSV file with error handling.

    Args:
        file_path: Path to CSV file.

    Returns:
        DataFrame if successful, None otherwise.
    """
    try:
        if not file_path.exists():
            return None
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {str(e)}")
        return None


def check_files_status(results_dir: Path) -> Dict[str, bool]:
    """
    Check which result files exist.

    Args:
        results_dir: Directory containing results.

    Returns:
        Dictionary mapping file names to existence status.
    """
    files_to_check = {
        "eval_metrics.csv": results_dir / "eval_metrics.csv",
        "eval_baseline_equity_curve.csv": results_dir / "eval_baseline_equity_curve.csv",
        "eval_dqn_equity_curve.csv": results_dir / "eval_dqn_equity_curve.csv",
        "eval_baseline_daily_metrics.csv": results_dir / "eval_baseline_daily_metrics.csv",
        "eval_dqn_daily_metrics.csv": results_dir / "eval_dqn_daily_metrics.csv",
        "metrics.csv": results_dir / "metrics.csv",
        "daily_metrics.csv": results_dir / "daily_metrics.csv",
    }
    
    # Check training metrics
    train_metrics_path = project_root / "logs" / "dqn_train_metrics.csv"
    files_to_check["dqn_train_metrics.csv"] = train_metrics_path if train_metrics_path.exists() else None

    return {name: path.exists() if path else False for name, path in files_to_check.items()}


def load_summary_metrics(results_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load summary metrics from eval_metrics.csv (preferred) or metrics.csv (fallback).
    
    Returns:
        DataFrame with metrics, or None if neither file exists.
    """
    # Try eval_metrics.csv first (preferred)
    eval_metrics_path = results_dir / "eval_metrics.csv"
    df = load_csv_safe(eval_metrics_path)
    if df is not None and not df.empty:
        return df
    
    # Fallback to metrics.csv
    metrics_path = results_dir / "metrics.csv"
    df = load_csv_safe(metrics_path)
    if df is not None and not df.empty:
        return df
    
    return None


def load_equity_curve(results_dir: Path, policy: str) -> Optional[pd.DataFrame]:
    """
    Load equity curve for a policy.

    Args:
        results_dir: Results directory.
        policy: Policy name ('baseline' or 'dqn').

    Returns:
        DataFrame with equity curve data.
    """
    if policy == "baseline":
        path = results_dir / "eval_baseline_equity_curve.csv"
    elif policy == "dqn":
        path = results_dir / "eval_dqn_equity_curve.csv"
    else:
        return None
    
    df = load_csv_safe(path)
    if df is not None and "equity" in df.columns:
        return df
    return None


def load_daily_metrics(results_dir: Path, policy: str) -> Optional[pd.DataFrame]:
    """
    Load daily metrics for a policy.

    Args:
        results_dir: Results directory.
        policy: Policy name ('baseline' or 'dqn').

    Returns:
        DataFrame with daily metrics.
    """
    if policy == "baseline":
        path = results_dir / "eval_baseline_daily_metrics.csv"
    elif policy == "dqn":
        path = results_dir / "eval_dqn_daily_metrics.csv"
    else:
        return None
    
    return load_csv_safe(path)


def load_training_metrics() -> Optional[pd.DataFrame]:
    """Load DQN training metrics."""
    train_path = project_root / "logs" / "dqn_train_metrics.csv"
    return load_csv_safe(train_path)


def display_summary_metrics(metrics_df: pd.DataFrame, show_baseline: bool, show_dqn: bool):
    """Display summary metrics as cards."""
    st.header("📊 Summary Metrics")
    
    # Filter by policies to show
    policies_to_show = []
    if show_baseline:
        policies_to_show.append("baseline")
    if show_dqn:
        policies_to_show.append("dqn")
    
    if not policies_to_show:
        st.info("Enable at least one policy in the sidebar to view metrics.")
        return
    
    filtered_df = metrics_df[metrics_df["policy"].isin(policies_to_show)]
    
    if filtered_df.empty:
        st.warning("No metrics found for selected policies.")
        return
    
    # Get baseline row for excess return calculation
    baseline_row = metrics_df[metrics_df["policy"] == "baseline"].iloc[0] if len(metrics_df[metrics_df["policy"] == "baseline"]) > 0 else None
    baseline_total_return = baseline_row.get("total_return", 0) if baseline_row is not None else None
    
    # Starting equity (default 10000)
    starting_equity = 10000.0
    
    # Display metrics in columns
    for _, row in filtered_df.iterrows():
        policy_name = row["policy"].upper()
        st.subheader(f"{policy_name} Policy")
        
        # Calculate Net P&L from total_return
        total_return = row.get("total_return", 0)
        net_pnl = total_return * starting_equity
        
        # Get total trades (prefer executed_trades if available and > 0, else total_trades)
        executed_trades = row.get("executed_trades", 0)
        if pd.notna(executed_trades) and executed_trades > 0:
            total_trades = int(executed_trades)
        else:
            total_trades = int(row.get("total_trades", 0)) if pd.notna(row.get("total_trades", 0)) else 0
        
        # Calculate trades per day
        days_tested = row.get("days_tested", 1)
        if pd.isna(days_tested) or days_tested == 0:
            days_tested = 1
        trades_per_day = total_trades / days_tested if days_tested > 0 else 0.0
        
        # Calculate excess return vs baseline (in bps)
        excess_return_bps = None
        if baseline_total_return is not None and policy_name != "BASELINE":
            excess_return_bps = (total_return - baseline_total_return) * 10000
        
        # Win rate
        win_rate = row.get("win_rate", 0)
        if abs(win_rate) < 1:
            win_rate_pct = win_rate * 100
        else:
            win_rate_pct = win_rate
        
        # Sharpe ratio
        sharpe_ratio = row.get("sharpe_ratio", 0)
        if pd.isna(sharpe_ratio):
            sharpe_ratio = 0.0
        
        # Display metrics in 6 columns
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                "Net P&L ($)",
                f"${net_pnl:.2f}",
            )
        
        with col2:
            st.metric(
                "Total Trades",
                f"{total_trades}",
            )
        
        with col3:
            st.metric(
                "Trades/Day",
                f"{trades_per_day:.1f}",
            )
        
        with col4:
            if excess_return_bps is not None:
                sign = "+" if excess_return_bps >= 0 else ""
                st.metric(
                    "Excess Return vs Baseline",
                    f"{sign}{int(excess_return_bps)} bps",
                )
            else:
                st.metric(
                    "Excess Return vs Baseline",
                    "N/A",
                )
        
        with col5:
            st.metric(
                "Win Rate (%)",
                f"{win_rate_pct:.2f}%",
            )
        
        with col6:
            st.metric(
                "Sharpe Ratio",
                f"{sharpe_ratio:.2f}",
            )
        
        st.divider()


def display_equity_curves(results_dir: Path, show_baseline: bool, show_dqn: bool):
    """Display equity curve comparison."""
    st.header("📈 Equity Curve Comparison")
    
    equity_data = {}
    
    if show_baseline:
        baseline_df = load_equity_curve(results_dir, "baseline")
        if baseline_df is not None:
            equity_data["Baseline"] = baseline_df["equity"]
    
    if show_dqn:
        dqn_df = load_equity_curve(results_dir, "dqn")
        if dqn_df is not None:
            equity_data["DQN"] = dqn_df["equity"]
    
    if not equity_data:
        st.info(
            "No equity curve data available. Run the offline pipeline to generate results:\n\n"
            "```bash\n"
            "python scripts/run_offline_pipeline.py --data_path data/btc_1m_last7d.csv --start YYYY-MM-DD --end YYYY-MM-DD\n"
            "```"
        )
        return
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(equity_data)
    
    # Use Streamlit line chart
    st.line_chart(comparison_df)
    
    # Show statistics
    with st.expander("Equity Curve Statistics"):
        stats_df = pd.DataFrame({
            "Policy": list(equity_data.keys()),
            "Initial Equity": [df.iloc[0] if len(df) > 0 else 0 for df in equity_data.values()],
            "Final Equity": [df.iloc[-1] if len(df) > 0 else 0 for df in equity_data.values()],
            "Max Equity": [df.max() for df in equity_data.values()],
            "Min Equity": [df.min() for df in equity_data.values()],
        })
        st.dataframe(stats_df, use_container_width=True)


def display_daily_metrics(results_dir: Path, show_baseline: bool, show_dqn: bool):
    """Display daily metrics tables."""
    st.header("📅 Daily Metrics")
    
    policies_to_show = []
    if show_baseline:
        policies_to_show.append(("baseline", "Baseline"))
    if show_dqn:
        policies_to_show.append(("dqn", "DQN"))
    
    if not policies_to_show:
        st.info("Enable at least one policy in the sidebar to view daily metrics.")
        return
    
    for policy_key, policy_name in policies_to_show:
        daily_df = load_daily_metrics(results_dir, policy_key)
        
        if daily_df is None or daily_df.empty:
            st.info(f"No daily metrics available for {policy_name} policy.")
            continue
        
        st.subheader(f"{policy_name} Policy - Daily Metrics")
        
        # Date filter if date column exists
        if "day" in daily_df.columns:
            # Convert to datetime if needed
            try:
                daily_df["day"] = pd.to_datetime(daily_df["day"])
                min_date = daily_df["day"].min().date()
                max_date = daily_df["day"].max().date()
                
                # Use date_input with single date or range
                date_input = st.date_input(
                    f"Filter {policy_name} by date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"date_filter_{policy_key}",
                )
                
                # Handle both single date and tuple
                if isinstance(date_input, tuple) and len(date_input) == 2:
                    start_date, end_date = date_input
                    filtered_df = daily_df[
                        (daily_df["day"].dt.date >= start_date) &
                        (daily_df["day"].dt.date <= end_date)
                    ]
                elif isinstance(date_input, tuple) and len(date_input) == 1:
                    # Single date selected
                    filtered_df = daily_df[daily_df["day"].dt.date == date_input[0]]
                else:
                    # No filter or single date
                    filtered_df = daily_df
            except Exception as e:
                # If date filtering fails, show all data
                st.warning(f"Date filtering unavailable: {e}")
                filtered_df = daily_df
        else:
            filtered_df = daily_df
        
        st.dataframe(filtered_df, use_container_width=True)
        st.divider()


def display_training_curve():
    """Display DQN training curve."""
    st.header("🎓 DQN Training Curve")
    
    train_df = load_training_metrics()
    
    if train_df is None or train_df.empty:
        st.info("No training metrics available. Training metrics are saved to `logs/dqn_train_metrics.csv` during DQN training.")
        return
    
    # Episode return curve
    if "return" in train_df.columns:
        st.subheader("Episode Return")
        st.line_chart(train_df.set_index("episode")["return"] if "episode" in train_df.columns else train_df["return"])
    
    # Epsilon curve
    if "epsilon" in train_df.columns:
        st.subheader("Epsilon (Exploration Rate)")
        st.line_chart(train_df.set_index("episode")["epsilon"] if "episode" in train_df.columns else train_df["epsilon"])
    
    # Average loss curve
    if "avg_loss" in train_df.columns:
        st.subheader("Average Loss")
        st.line_chart(train_df.set_index("episode")["avg_loss"] if "episode" in train_df.columns else train_df["avg_loss"])
    
    # Show full table in expander
    with st.expander("Full Training Metrics Table"):
        st.dataframe(train_df, use_container_width=True)


def main():
    """Main application."""
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Results directory input
        default_results_dir = "backtest/results"
        results_dir_input = st.text_input(
            "Results Directory",
            value=default_results_dir,
            help="Path to directory containing backtest results",
        )
        
        results_dir = project_root / results_dir_input
        if not results_dir.exists():
            st.error(f"Directory not found: {results_dir}")
            results_dir = project_root / default_results_dir
        
        # Policy toggles
        st.subheader("Policies")
        show_baseline = st.checkbox("Show Baseline", value=True)
        show_dqn = st.checkbox("Show DQN", value=True)
        
        # Reload button
        if st.button("🔄 Reload Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Status panel
        st.divider()
        st.subheader("📋 File Status")
        file_status = check_files_status(results_dir)
        
        for filename, exists in file_status.items():
            if exists:
                st.success(f"✓ {filename}")
            else:
                st.info(f"○ {filename} (not found)")
    
    # Main content
    # Check if summary metrics exist
    metrics_df = load_summary_metrics(results_dir)
    
    if metrics_df is None or metrics_df.empty:
        st.warning("⚠️ Summary metrics not found")
        st.info(
            "To generate results, run the offline pipeline:\n\n"
            "```bash\n"
            "python scripts/run_offline_pipeline.py --data_path data/btc_1m_last7d.csv --start YYYY-MM-DD --end YYYY-MM-DD --total_steps 20000\n"
            "```\n\n"
            "**Note:** Run the offline pipeline first to generate all result files."
        )
    else:
        # Display summary metrics
        display_summary_metrics(metrics_df, show_baseline, show_dqn)
    
    # Equity curves
    display_equity_curves(results_dir, show_baseline, show_dqn)
    
    # Daily metrics
    display_daily_metrics(results_dir, show_baseline, show_dqn)
    
    # Training curve
    display_training_curve()


if __name__ == "__main__":
    main()

